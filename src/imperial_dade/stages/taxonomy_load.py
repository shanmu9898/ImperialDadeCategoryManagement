import pandas as pd
from typing import List, Tuple, Optional, Dict
from imperial_dade.llm.client import OpenAIAgent

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Import column configurations from config
from imperial_dade.config import PipelineConfig
TRANSACTION_COLUMNS = PipelineConfig.TRANSACTION_COLUMNS

# Required columns for data processing
REQUIRED_COLUMNS = ['UNSPSC', 'UL ECOLOGO Certification']

from imperial_dade.utils.validation import (
    ValidationError,
    validate_columns_exist,
    validate_dataframe,
    validate_list_input,
    validate_string_input,
)

# Re-export the canonical key helpers so callers can keep importing them from
# this module (where they lived in the legacy `Load_Isolate_functions.py`).
from imperial_dade.utils.keys import map_erp_system_to_number  # noqa: F401

# =============================================================================
# GROUPING FUNCTIONS
# =============================================================================

def group_data(
    im_grp: pd.DataFrame,
    qty_col: str = PipelineConfig.TRANSACTION_COLUMNS['qty'],
    gross_cost_col: str = PipelineConfig.TRANSACTION_COLUMNS['gross_cost'],
    net_cost_col: str = PipelineConfig.TRANSACTION_COLUMNS['net_cost'],
    entity_item_col: str = 'Entity--Item',
    item_desc1_col: str = PipelineConfig.TRANSACTION_COLUMNS['item_desc_1'],
    item_desc2_col: str = PipelineConfig.TRANSACTION_COLUMNS['item_desc_2'],
) -> pd.DataFrame:
    """
    Group and aggregate item data, preserving key attributes for each Entity--Item.

    - Aggregates qty_col, gross_cost_col, net_cost_col by [entity_item_col, item_desc1_col, item_desc2_col].
    - Preserves the first occurrence of 'Case Pack', 'VB Flag', 'VGN', 'VPN' (if present).
    - Handles missing columns gracefully.
    - All column names can be customized via parameters.
    """
    case_pack_col = PipelineConfig.TRANSACTION_COLUMNS['case_pack']
    vb_flag_col = PipelineConfig.TRANSACTION_COLUMNS['vb_flag']
    vgn_col = PipelineConfig.TRANSACTION_COLUMNS['vgn']
    vpn_col = PipelineConfig.TRANSACTION_COLUMNS['vpn']

    agg_cols = [col for col in [qty_col, gross_cost_col, net_cost_col] if col in im_grp.columns]
    group_cols = [entity_item_col, item_desc1_col, item_desc2_col]

    # Collect extra columns to preserve — only those actually present on
    # the input frame. Sales data doesn't carry Case Pack / VB Flag / VGN /
    # VPN (those live on item_master); historically they were copied onto
    # transactions by add_po_cost, but Stage 1 no longer runs that step.
    extra_cols = [
        col for col in [case_pack_col, vb_flag_col, vgn_col, vpn_col]
        if col and col in im_grp.columns
    ]

    # Prepare DataFrame with first occurrence of extra columns per Entity--Item
    if extra_cols:
        first_extra = (
            im_grp.drop_duplicates(subset=[entity_item_col], keep='first')
            [[entity_item_col] + extra_cols]
        )
    else:
        first_extra = None

    # Fill group-by columns with empty string to avoid NaN issues
    im_grp[group_cols] = im_grp[group_cols].fillna('')

    if agg_cols:
        im_grp_agg = im_grp.groupby(group_cols, as_index=False)[agg_cols].sum()
        # Ensure all group-by columns are present (in case of missing values)
        group_meta = im_grp[group_cols].drop_duplicates()
        im_grp_out = pd.merge(im_grp_agg, group_meta, on=group_cols, how='right')
    else:
        # No aggregation, just unique rows by group_cols
        im_grp_out = im_grp.drop_duplicates(subset=[entity_item_col], keep='first')[group_cols]

    # Merge in extra columns, if any
    if extra_cols and first_extra is not None:
        im_grp_out = im_grp_out.merge(first_extra, on=entity_item_col, how='left')
        # Standardize column names for output
        rename_map = {}
        if case_pack_col:
            rename_map[case_pack_col] = PipelineConfig.TRANSACTION_COLUMNS['case_pack']
        if vb_flag_col:
            rename_map[vb_flag_col] = PipelineConfig.TRANSACTION_COLUMNS['vb_flag']
        if vgn_col:
            rename_map[vgn_col] = PipelineConfig.TRANSACTION_COLUMNS['vgn']
        if vpn_col:
            rename_map[vpn_col] = PipelineConfig.TRANSACTION_COLUMNS['vpn']
        if rename_map:
            im_grp_out = im_grp_out.rename(columns=rename_map)

    return im_grp_out

# =============================================================================
# DATA FILTERING FUNCTIONS
# =============================================================================


def _filter_sfy_by_items(
    sfy: pd.DataFrame,
    im_s2k: pd.DataFrame
) -> pd.DataFrame:
    """Filter sfy DataFrame based on items from im_s2k."""
    item_id_col = 'Entity--Item'
    items_to_keep = im_s2k[item_id_col].dropna().unique()
    
    if items_to_keep.size == 0:
        return pd.DataFrame()
    
    return sfy[sfy[item_id_col].isin(items_to_keep)].copy()

# =============================================================================
# COVERAGE ANALYSIS FUNCTIONS
# =============================================================================

def _get_columns_with_coverage(
    filtered_sfy: pd.DataFrame,
    coverage_threshold: float
) -> List[str]:
    """
    Find columns that meet the coverage threshold between UNSPSC and UL ECOLOGO Certification.
    
    Returns:
        List of column names meeting coverage threshold
    """
    if filtered_sfy.empty:
        return []
    
    all_columns = filtered_sfy.columns.tolist()
    
    try:
        unspsc_idx = all_columns.index(REQUIRED_COLUMNS[0])
        ecologo_idx = all_columns.index(REQUIRED_COLUMNS[1])
        
        if unspsc_idx >= ecologo_idx:
            logger.warning(f"Warning: Column order incorrect ('{REQUIRED_COLUMNS[1]}' before '{REQUIRED_COLUMNS[0]}')")
            return []
        
        columns_to_check = all_columns[unspsc_idx + 1:ecologo_idx]
        total_rows = len(filtered_sfy)
        
        columns_with_coverage = []
        for col in columns_to_check:
            non_null_count = filtered_sfy[col].notna().sum()
            coverage = (non_null_count / total_rows) * 100
            if coverage >= coverage_threshold:
                columns_with_coverage.append(col)
        
        return columns_with_coverage
        
    except ValueError as e:
        logger.error(f"Error: Required column not found. {e}")
        return []


def _extract_sample_data(
    filtered_sfy: pd.DataFrame,
    columns_with_coverage: List[str],
    sample_size: int = 100
) -> Tuple[Dict[str, List], List[str]]:
    """
    Extract sample data from columns with coverage.
    
    Returns:
        Tuple of (data_dict, populated_columns)
    """
    final_data_dict = {}
    columns_actually_populated = []
    
    for col in columns_with_coverage:
        values = filtered_sfy[col].dropna().tolist()
        if not values:
            continue
        
        if len(values) >= sample_size:
            sample = values[:sample_size]
        else:
            multiplier = (sample_size // len(values)) + 1
            sample = (values * multiplier)[:sample_size]
        
        final_data_dict[col] = sample
        columns_actually_populated.append(col)
    
    return final_data_dict, columns_actually_populated

# =============================================================================
# MAIN S2K PROCESSING FUNCTION
# =============================================================================

def get_columns_with_coverage(
    im_s2k: pd.DataFrame,
    sfy: pd.DataFrame,
    coverage_threshold: float,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Filter DataFrame based on category and select columns meeting coverage threshold.
    
    Args:
        s2k_div: Category value to filter by
        im_grp: DataFrame with category and Entity columns
        sfy: DataFrame to filter and extract columns from
        coverage_threshold: Minimum percentage of non-null values required
        level: Category level (2 or 3)
    
    Returns:
        Tuple of (filtered_im_grp, columns_with_coverage, extracted_values_df)
    """
    
    # Filter sfy by items
    # filtered_sfy_rows = _filter_sfy_by_items(sfy, im_s2k)
    filtered_sfy_rows = sfy.copy()
    if filtered_sfy_rows.empty:
        logger.warning("Warning: No matching items found in sfy")
        return im_s2k, [], pd.DataFrame()
    
    logger.info(f"Filtered sfy to {len(filtered_sfy_rows)} rows")
    
    # Get columns with coverage
    columns_with_coverage = _get_columns_with_coverage(filtered_sfy_rows, coverage_threshold)
    logger.info(f"Found {len(columns_with_coverage)} columns meeting {coverage_threshold}% coverage")
    
    # Extract sample data
    final_data_dict, columns_actually_populated = _extract_sample_data(
        filtered_sfy_rows, columns_with_coverage
    )
    
    result_df = pd.DataFrame(final_data_dict) if final_data_dict else pd.DataFrame()
    logger.info(f"Extracted sample data for {len(final_data_dict)} columns")
    
    return im_s2k, columns_actually_populated, result_df


def merge_with_salsify(
    grouped: pd.DataFrame,
    sfy: pd.DataFrame,
    columns_for_description: List[str],
    entity_item_col: str = "Entity--Item",
) -> pd.DataFrame:
    """Merge Salsify attribute columns onto the grouped transaction frame.

    Replicates the legacy 01_Taxonomy.ipynb cells 59–60:
      - left-join on Entity--Item
      - drop duplicate Entity--Item rows (keep first)
      - fillna('') for the merged attribute columns
    """
    validate_dataframe(grouped, "grouped")
    validate_dataframe(sfy, "sfy")
    validate_list_input(columns_for_description, "columns_for_description")
    validate_columns_exist(grouped, [entity_item_col], "grouped")
    validate_columns_exist(sfy, [entity_item_col], "sfy")

    present_cols = [c for c in columns_for_description if c in sfy.columns]
    missing_cols = [c for c in columns_for_description if c not in sfy.columns]
    if missing_cols:
        logger.warning(
            "merge_with_salsify: %d requested columns missing from sfy and skipped: %s",
            len(missing_cols), missing_cols,
        )

    merged = grouped.merge(
        sfy[[entity_item_col] + present_cols],
        on=entity_item_col,
        how="left",
        suffixes=("", "_dup"),
    )
    merged = merged.drop_duplicates(subset=[entity_item_col], keep="first")
    for col in present_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna("")

    logger.info(
        "merge_with_salsify: %d rows after merge (%d attribute columns attached)",
        len(merged), len(present_cols),
    )
    return merged
