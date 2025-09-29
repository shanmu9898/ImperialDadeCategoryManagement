import pandas as pd
from typing import List, Tuple, Optional, Dict
from openai_api import OpenAIAgent

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Import column configurations from config
from config import PipelineConfig
TRANSACTION_COLUMNS = PipelineConfig.TRANSACTION_COLUMNS

# Required columns for data processing
REQUIRED_COLUMNS = ['UNSPSC', 'UL ECOLOGO Certification']

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_dataframe(df: pd.DataFrame, name: str) -> None:
    """Validate DataFrame input and raise ValidationError if invalid."""
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"{name} must be a pandas DataFrame")
    
    if df.empty:
        raise ValidationError(f"{name} cannot be empty")

def validate_list_input(value: List[str], name: str) -> None:
    """Validate list input."""
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{name} must be a non-empty list")
    
    if not all(isinstance(item, str) for item in value):
        raise ValidationError(f"All items in {name} must be strings")

def validate_string_input(value: str, name: str) -> None:
    """Validate string input."""
    if not value or not isinstance(value, str):
        raise ValidationError(f"{name} must be a non-empty string")

def validate_columns_exist(df: pd.DataFrame, required_columns: List[str], df_name: str) -> None:
    """Validate that required columns exist in DataFrame."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValidationError(f"Missing required columns in {df_name}: {', '.join(missing_cols)}")


def map_erp_system_to_number(series):
    """
    Maps each unique ERP System to a unique integer, with 'S2K' always mapped to 1.
    All other ERP Systems are mapped to unique integers starting from 2.
    Returns a dictionary mapping and a Series of mapped values.
    """
    unique_systems = pd.Series(series.unique())
    # Ensure 'S2K' is always present and first
    unique_systems = unique_systems[unique_systems.str.upper() != 'S2K']
    mapping = {'S2K': 1}
    for idx, val in enumerate(unique_systems, start=2):
        mapping[val] = idx
    mapped_series = series.apply(lambda x: mapping.get(str(x).upper(), None) if str(x).upper() == 'S2K' else mapping.get(x, None))
    return mapping, mapped_series

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
    po_cost_amt_col: str = 'po_cost_amt',  # Optionally allow custom name
) -> pd.DataFrame:
    """
    Group and aggregate item data, preserving key attributes for each Entity--Item.

    - Aggregates qty_col, gross_cost_col, net_cost_col by [entity_item_col, item_desc1_col, item_desc2_col].
    - If po_cost_amt_col is present, computes its weighted average (weighted by qty_col) per group.
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

    # Collect extra columns to preserve
    extra_cols = [col for col in [case_pack_col, vb_flag_col, vgn_col, vpn_col] if col]

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

    # Check if po_cost_amt_col is present
    has_po_cost_amt = po_cost_amt_col in im_grp.columns

    if agg_cols or has_po_cost_amt:
        # Prepare aggregation dictionary
        agg_dict = {col: 'sum' for col in agg_cols}
        
        # Aggregate numeric columns first
        im_grp_agg = im_grp.groupby(group_cols, as_index=False)[agg_cols].sum()
        
        # Handle po_cost_amt weighted average separately if present
        if has_po_cost_amt:
            # Calculate weighted average manually after grouping
            po_cost_weighted = im_grp.groupby(group_cols).apply(
                lambda x: (x[po_cost_amt_col] * x[qty_col]).sum() / x[qty_col].sum() 
                if x[qty_col].sum() > 0 else float('nan')
            ).reset_index()
            
            # Rename the calculated column to match the original name
            po_cost_weighted = po_cost_weighted.rename(columns={0: po_cost_amt_col})
            
            # Merge the weighted average back to the aggregated data
            im_grp_agg = im_grp_agg.merge(po_cost_weighted, on=group_cols, how='left')
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
            print(f"Warning: Column order incorrect ('{REQUIRED_COLUMNS[1]}' before '{REQUIRED_COLUMNS[0]}')")
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
        print(f"Error: Required column not found. {e}")
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

def add_po_cost(transactions: pd.DataFrame, item_master: pd.DataFrame) -> pd.DataFrame:
    """
    Add PO cost amount from item master to transactions by matching on Entity--Item and location.
    
    Args:
        transactions: DataFrame with transaction data containing 'Entity--Item' and 'Whs Code' columns
        item_master: DataFrame with item master data containing 'Entity--Item', 'location_code', and 'po_cost_amt' columns
        
    Returns:
        DataFrame with transactions plus the 'po_cost_amt' column from item master
        
    Raises:
        ValidationError: If input validation fails or required columns are missing
    """
    # Validate inputs
    validate_dataframe(transactions, "transactions")
    validate_dataframe(item_master, "item_master")
    
    # Check required columns in transactions
    required_transaction_cols = ['Entity--Item', PipelineConfig.TRANSACTION_COLUMNS['whs_code']]
    validate_columns_exist(transactions, required_transaction_cols, "transactions")
    
    # Check required columns in item_master
    required_item_master_cols = ['Entity--Item', 'location_code', 'po_cost_amt']
    validate_columns_exist(item_master, required_item_master_cols, "item_master")
    
    # Create a copy to avoid modifying the original
    transactions_with_po = transactions.copy()
    
    # Create a mapping key for item_master (Entity--Item + location_code)
    item_master['match_key'] = item_master['Entity--Item'].astype(str) + '|' + item_master['location_code'].astype(str)
    
    # Create a mapping dictionary for faster lookup
    po_cost_mapping = item_master.set_index('match_key')['po_cost_amt'].to_dict()
    
    # Create match keys for transactions
    transactions_with_po['match_key'] = transactions_with_po['Entity--Item'].astype(str) + '|' + transactions_with_po['Whs Code'].astype(str)
    
    # Map PO cost amounts
    transactions_with_po['po_cost_amt'] = transactions_with_po['match_key'].map(po_cost_mapping)
    
    # Clean up temporary columns
    transactions_with_po = transactions_with_po.drop('match_key', axis=1)
    
    # Initialize counters to avoid referencing before assignment
    qty_updates = 0
    gross_cost_updates = 0
    net_cost_updates = 0
    po_cost_updates = 0

    # Handle Qty calculation when Qty = 0, Net Cost > 0, and PO cost is available
    # Calculate Qty = Net Cost / PO Cost for these cases
    qty_update_mask = (
        (transactions_with_po[PipelineConfig.TRANSACTION_COLUMNS['qty']] <= 0) & 
        (transactions_with_po[PipelineConfig.TRANSACTION_COLUMNS['net_cost']] > 0) & 
        (transactions_with_po['po_cost_amt'].notna()) &
        (transactions_with_po['po_cost_amt'] > 0)  # Avoid division by zero
    )
    
    if qty_update_mask.any():
        # Calculate new quantities and ensure proper data type
        new_qty_values = (
            transactions_with_po.loc[qty_update_mask, PipelineConfig.TRANSACTION_COLUMNS['net_cost']] / 
            transactions_with_po.loc[qty_update_mask, 'po_cost_amt']
        )
        
        # Convert to the same dtype as the original Qty column
        original_qty_dtype = transactions_with_po[PipelineConfig.TRANSACTION_COLUMNS['qty']].dtype
        new_qty_values = new_qty_values.astype(original_qty_dtype)
        
        transactions_with_po.loc[qty_update_mask, PipelineConfig.TRANSACTION_COLUMNS['qty']] = new_qty_values
        qty_updates = qty_update_mask.sum()


    print(f"  Updated {qty_updates} transactions with Qty = Net Cost / PO Cost")

    
    # Handle Gross Cost calculation when Qty > 0, Gross Cost <= 0, and PO cost is available
    # Calculate Gross Cost = PO Cost × Qty for these cases
    gross_cost_update_mask = (
        (transactions_with_po[PipelineConfig.TRANSACTION_COLUMNS['qty']] > 0) & 
        (transactions_with_po[PipelineConfig.TRANSACTION_COLUMNS['gross_cost']] <= 0) & 
        (transactions_with_po['po_cost_amt'].notna()) &
        (transactions_with_po['po_cost_amt'] > 0)  # Ensure PO cost is positive
    )
    
    if gross_cost_update_mask.any():
        # Calculate new gross costs and ensure proper data type
        new_gross_cost_values = (
            transactions_with_po.loc[gross_cost_update_mask, 'po_cost_amt'] * 
            transactions_with_po.loc[gross_cost_update_mask, PipelineConfig.TRANSACTION_COLUMNS['qty']]
        )
        
        # Convert to the same dtype as the original Gross Cost column
        original_gross_cost_dtype = transactions_with_po[PipelineConfig.TRANSACTION_COLUMNS['gross_cost']].dtype
        new_gross_cost_values = new_gross_cost_values.astype(original_gross_cost_dtype)
        
        transactions_with_po.loc[gross_cost_update_mask, PipelineConfig.TRANSACTION_COLUMNS['gross_cost']] = new_gross_cost_values
        
        gross_cost_updates = gross_cost_update_mask.sum()
        print(f"  Updated {gross_cost_updates} transactions with Gross Cost = PO Cost × Qty")
    
    # Handle Net Cost calculation when POD = "N" (no discount) for transactions with updated gross costs
    # Set Net Cost = Gross Cost for these cases
    net_cost_update_mask = (
        gross_cost_update_mask & 
        (transactions_with_po[PipelineConfig.TRANSACTION_COLUMNS['pod']] == 'N')  # POD indicates no discount
    )
    
    if net_cost_update_mask.any():
        # Set net cost equal to gross cost for no-discount transactions and ensure proper data type
        new_net_cost_values = transactions_with_po.loc[net_cost_update_mask, PipelineConfig.TRANSACTION_COLUMNS['gross_cost']].copy()
        
        # Convert to the same dtype as the original Net Cost column
        original_net_cost_dtype = transactions_with_po[PipelineConfig.TRANSACTION_COLUMNS['net_cost']].dtype
        new_net_cost_values = new_net_cost_values.astype(original_net_cost_dtype)
        
        transactions_with_po.loc[net_cost_update_mask, PipelineConfig.TRANSACTION_COLUMNS['net_cost']] = new_net_cost_values
        
        net_cost_updates = net_cost_update_mask.sum()
        print(f"  Updated {net_cost_updates} transactions with Net Cost = Gross Cost (POD = 'N')")
    
    # Handle PO Cost calculation when po_cost_amt is NaN but Qty and Gross Cost are available
    # Calculate po_cost_amt = Gross Cost / Qty for these cases
    po_cost_update_mask = (
        (transactions_with_po['po_cost_amt'].isna()) & 
        (transactions_with_po[PipelineConfig.TRANSACTION_COLUMNS['qty']] > 0) & 
        (transactions_with_po[PipelineConfig.TRANSACTION_COLUMNS['gross_cost']] > 0)
    )
    
    if po_cost_update_mask.any():
        # Calculate new PO costs and ensure proper data type
        new_po_cost_values = (
            transactions_with_po.loc[po_cost_update_mask, PipelineConfig.TRANSACTION_COLUMNS['gross_cost']] / 
            transactions_with_po.loc[po_cost_update_mask, PipelineConfig.TRANSACTION_COLUMNS['qty']]
        )
        
        # Convert to the same dtype as the original po_cost_amt column (if it exists)
        if 'po_cost_amt' in transactions_with_po.columns:
            # If po_cost_amt column exists, try to match its dtype
            try:
                original_po_cost_dtype = transactions_with_po['po_cost_amt'].dtype
                new_po_cost_values = new_po_cost_values.astype(original_po_cost_dtype)
            except:
                # If conversion fails, keep as float
                pass
        
        transactions_with_po.loc[po_cost_update_mask, 'po_cost_amt'] = new_po_cost_values
        
        po_cost_updates = po_cost_update_mask.sum()
        print(f"  Updated {po_cost_updates} transactions with po_cost_amt = Gross Cost / Qty")
    else:
        po_cost_updates = 0
    
    # Calculate total fixes across all four types
    total_fixes = qty_updates + gross_cost_updates + net_cost_updates + po_cost_updates
    
    # Report matching statistics
    total_transactions = len(transactions_with_po)
    matched_transactions = transactions_with_po['po_cost_amt'].notna().sum()
    match_rate = (matched_transactions / total_transactions) * 100 if total_transactions > 0 else 0
    
    print(f"PO Cost matching complete:")
    print(f"  Total transactions: {total_transactions}")
    print(f"  Matched transactions: {matched_transactions}")
    print(f"  Match rate: {match_rate:.1f}%")
    print(f"  Total data fixes applied: {total_fixes} (Qty: {qty_updates}, Gross Cost: {gross_cost_updates}, Net Cost: {net_cost_updates}, PO Cost: {po_cost_updates})")
    
    # Remove rows with negative values in key columns
    initial_row_count = len(transactions_with_po)
    
    # Create mask for rows to remove (negative Qty, Gross Cost, or Net Cost)
    rows_to_remove = (
        (transactions_with_po[PipelineConfig.TRANSACTION_COLUMNS['qty']] <= 0) | 
        (transactions_with_po[PipelineConfig.TRANSACTION_COLUMNS['gross_cost']] <= 0) | 
        (transactions_with_po[PipelineConfig.TRANSACTION_COLUMNS['net_cost']] <= 0)
    )
    
    if rows_to_remove.any():
        # Count rows being removed
        removed_row_count = rows_to_remove.sum()
        
        # Remove the negative rows
        transactions_with_po = transactions_with_po[~rows_to_remove].copy()
        
        # Report removal statistics
        final_row_count = len(transactions_with_po)
        print(f"  Data cleanup complete:")
        print(f"    Removed {removed_row_count} rows with negative values")
        print(f"    Final row count: {final_row_count} (from {initial_row_count})")
    else:
        print(f"  Data cleanup complete: No rows with negative values found")
    
    return transactions_with_po

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
    filtered_sfy_rows = _filter_sfy_by_items(sfy, im_s2k)
    if filtered_sfy_rows.empty:
        print("Warning: No matching items found in sfy")
        return im_s2k, [], pd.DataFrame()
    
    print(f"Filtered sfy to {len(filtered_sfy_rows)} rows")
    
    # Get columns with coverage
    columns_with_coverage = _get_columns_with_coverage(filtered_sfy_rows, coverage_threshold)
    print(f"Found {len(columns_with_coverage)} columns meeting {coverage_threshold}% coverage")
    
    # Extract sample data
    final_data_dict, columns_actually_populated = _extract_sample_data(
        filtered_sfy_rows, columns_with_coverage
    )
    
    result_df = pd.DataFrame(final_data_dict) if final_data_dict else pd.DataFrame()
    print(f"Extracted sample data for {len(final_data_dict)} columns")
    
    return im_s2k, columns_actually_populated, result_df