import pandas as pd
import os
import ast
import json
import numpy as np
import faiss
import Levenshtein
import re
from typing import List, Tuple, Optional, Dict, Any
from pyvent.tools.llm.openai_api import OpenAIAgent
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from jinja2 import Template
from xlsxwriter.utility import xl_col_to_name


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Column name mappings
DEFAULT_COLUMNS = {
    'item_id': 'Entity--Item',
    'vendor': 'vgn_name',
    'description': 'All Descriptions',
    'attributes': 'attributes',
    'matches': 'Matches',
    'sales': 'L3M_Sales',
    'cogs': 'L3M_Cogs',
    'adj_vol': 'L3M_adj_vol',
    'private_label': 'private_label_flag',
    'sales_pct': 'Sales %'
}

# Excel formatting constants
EXCEL_FORMATS = {
    'thin_border': 1,
    'thick_border_style': 5,
    'target_hl_bg': '#FFFFE0',
    'header_bg_color': '#D3D3D3',
    'summary_header_bg': '#C0C0C0',
    'diff_bg_color': '#F0F0F0'
}

# Data validation options
DATA_VALIDATION_OPTIONS_ACCEPT_REJECT = ["Accept", "Consider", "Reject"]

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

def validate_string_input(value: str, name: str) -> None:
    """Validate string input."""
    if not value or not isinstance(value, str):
        raise ValidationError(f"{name} must be a non-empty string")

def validate_list_input(value: List[str], name: str) -> None:
    """Validate list input."""
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{name} must be a non-empty list")
    
    if not all(isinstance(item, str) for item in value):
        raise ValidationError(f"All items in {name} must be strings")

def validate_columns_exist(df: pd.DataFrame, required_columns: List[str], df_name: str) -> None:
    """Validate that required columns exist in DataFrame."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValidationError(f"Missing required columns in {df_name}: {', '.join(missing_cols)}")

# =============================================================================
# DATA VALIDATION AND CLEANING FUNCTIONS
# =============================================================================

def _validate_write_matches_inputs(
    im_final_full: pd.DataFrame,
    base_filename: str,
    essential_cols: List[str]
) -> bool:
    """
    Validate inputs for write_top_matches function.
    
    Returns:
        True if valid, False otherwise
    """
    try:
        validate_dataframe(im_final_full, "im_final_full")
        validate_string_input(base_filename, "base_filename")
    except ValidationError as e:
        print(f"Error: {e}")
        return False
    
    # Check for essential columns
    missing_cols = [col for col in essential_cols if col not in im_final_full.columns]
    if missing_cols:
        print(f"Warning: Missing essential columns: {missing_cols}")
    
    return True


def _clean_dataframe_columns(
    im_final_full: pd.DataFrame,
    item_id_col: str,
    vpn_code_col: str,
    item_code_col: str,
    use_vpn: bool
) -> pd.DataFrame:
    """
    Clean DataFrame columns for processing.
    
    Returns:
        Cleaned DataFrame
    """
    df = im_final_full.copy()
    
    # Convert item_id_col to string, strip whitespace, and convert to lowercase for robust matching
    if item_id_col in df.columns:
        df[item_id_col] = df[item_id_col].astype(str).str.strip()
    else:
        print(f"Warning: '{item_id_col}' not found, cannot perform cleaning on it.")
    
    # Apply similar cleaning to vpn_code_col and item_code_col if using VPN
    if use_vpn:
        for col in [vpn_code_col, item_code_col]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()  # No .lower() as requested
    
    return df


def _ensure_essential_columns(
    im_final_full: pd.DataFrame,
    essential_cols: List[str],
    model_matches_col: str,
    openai_col: str,
    reasoning_col: str,
    vpn_code_col: str,
    item_code_col: str,
    use_vpn: bool
) -> pd.DataFrame:
    """
    Ensure all essential columns exist in DataFrame.
    
    Returns:
        DataFrame with all essential columns
    """
    df = im_final_full.copy()
    
    for col_name in essential_cols:
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found in input DataFrame. It will be treated as empty/default.")
            if col_name == model_matches_col:
                df[col_name] = [[] for _ in range(len(df))]
            elif col_name in [openai_col, reasoning_col]:
                df[col_name] = ''
            elif use_vpn and col_name in [vpn_code_col, item_code_col]:
                df[col_name] = ''
            else:
                df[col_name] = pd.NA
    
    return df


def _get_description_column(
    im_final_full: pd.DataFrame,
    main_desc_col: str,
    fallback_desc_col: str
) -> Tuple[pd.DataFrame, str]:
    """
    Determine which description column to use.
    
    Returns:
        Tuple of (DataFrame with description_display_col, actual_desc_col_to_use)
    """
    df = im_final_full.copy()
    
    if main_desc_col not in df.columns or df[main_desc_col].isnull().all():
        print(f"Warning: Specified main description column '{main_desc_col}' not found or is all null. Falling back to '{fallback_desc_col}'.")
        if fallback_desc_col not in df.columns or df[fallback_desc_col].isnull().all():
            print(f"Warning: Fallback description column '{fallback_desc_col}' also not found or is all null. Descriptions will be 'N/A'.")
            df['description_display_col'] = 'N/A'
            return df, 'description_display_col'
        else:
            df['description_display_col'] = df[fallback_desc_col]
            return df, 'description_display_col'
    else:
        df['description_display_col'] = df[main_desc_col]
        return df, 'description_display_col'

# =============================================================================
# DATA PARSING FUNCTIONS
# =============================================================================

def _parse_openai_response(response_str: str) -> Dict[str, str]:
    """
    Parse OpenAI pipe-delimited response string into dictionary.
    
    Returns:
        Dictionary of attribute names and values
    """
    attributes = {}
    if not isinstance(response_str, str) or not response_str.strip():
        return attributes
    
    pairs = response_str.split(' | ')
    for pair in pairs:
        if ': ' in pair:
            key, value = pair.split(': ', 1)
            key = key.strip()
            value = value.strip()
            attributes[key] = '' if value.lower() == 'null' else value
    
    return attributes


def _parse_model_matches(raw_model_match_data: Any) -> List[str]:
    """
    Parse model match data into a list of entity IDs.
    
    Returns:
        List of entity IDs
    """
    if isinstance(raw_model_match_data, str):
        try:
            model_match_ids = ast.literal_eval(raw_model_match_data)
            if not isinstance(model_match_ids, list):
                model_match_ids = []
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse '{raw_model_match_data}' as a list. Setting to empty list.")
            model_match_ids = []
    elif isinstance(raw_model_match_data, list):
        model_match_ids = raw_model_match_data
    else:
        model_match_ids = []
    
    # Flatten the list if it's a list of lists, and clean individual IDs
    flat_model_match_ids = []
    for x in model_match_ids:
        if isinstance(x, list):
            for y in x:
                if pd.notna(y):
                    flat_model_match_ids.append(str(y).strip())
        elif pd.notna(x):
            flat_model_match_ids.append(str(x).strip())
    
    return list(set(flat_model_match_ids))

# =============================================================================
# VPN LOGIC FUNCTIONS
# =============================================================================

def _should_exclude_vpn_match(
    match_vpn_code: str,
    target_vpn_code: str,
    target_item_code: str,
    use_vpn: bool
) -> bool:
    """
    Determine if a match should be excluded based on VPN code logic.
    
    Returns:
        True if match should be excluded, False otherwise
    """
    if not use_vpn:
        return False
    
    # Exclude if:
    # 1. Match's vpn_code is the same as target's vpn_code
    # OR
    # 2. Match's vpn_code is the same as target's item_code
    return (pd.notna(match_vpn_code) and match_vpn_code == target_vpn_code) or \
           (pd.notna(match_vpn_code) and match_vpn_code == target_item_code)

# =============================================================================
# DATA CREATION FUNCTIONS
# =============================================================================

def _create_target_data(
    item_row: pd.Series,
    item_id_col: str,
    actual_desc_col_to_use: str,
    sales_col: str,
    cogs_col: str,
    adj_vol_col: str,
    vendor_col: str,
    pl_flag_col: str,
    sales_pct_col: str,
    openai_col: str,
    vpn_code_col: str,
    item_code_col: str,
    use_vpn: bool
) -> Dict[str, Any]:
    """
    Create target data dictionary.
    
    Returns:
        Target data dictionary
    """
    target_entity_item = item_row.get(item_id_col, 'UnknownTarget')
    target_description_val = item_row.get(actual_desc_col_to_use, 'N/A')
    
    target_l3m_sales = item_row.get(sales_col, 0.0)
    target_l3m_cogs = item_row.get(cogs_col, 0.0)
    target_l3m_adj_vol = item_row.get(adj_vol_col, 0.0)
    
    target_data = {
        item_id_col: target_entity_item,
        'Description': target_description_val,
        'is_target': True,
        sales_col: target_l3m_sales,
        sales_pct_col: item_row.get(sales_pct_col),
        cogs_col: target_l3m_cogs,
        adj_vol_col: target_l3m_adj_vol,
        pl_flag_col: "Yes" if item_row.get(pl_flag_col) else "No",
        vendor_col: item_row.get(vendor_col, 'N/A')
    }
    
    # Add VPN and Item Code if using VPN
    if use_vpn:
        target_data['target_vpn_code'] = str(item_row.get(vpn_code_col, '')).strip()
        target_data['target_item_code'] = str(item_row.get(item_code_col, '')).strip()
    
    # Parse OpenAI response
    target_openai_response_str = item_row.get(openai_col, '')
    if pd.isna(target_openai_response_str):
        target_openai_response_str = ''
    target_data['attributes'] = _parse_openai_response(target_openai_response_str)
    
    # Calculate GM%
    if isinstance(target_l3m_sales, (int, float)) and isinstance(target_l3m_cogs, (int, float)) and target_l3m_sales != 0:
        target_data['GM%'] = (target_l3m_sales - target_l3m_cogs) / target_l3m_sales
    else:
        target_data['GM%'] = None
    
    return target_data


def _create_match_data(
    match_entity_item_id: str,
    im_final_full: pd.DataFrame,
    item_id_col: str,
    actual_desc_col_to_use: str,
    sales_col: str,
    cogs_col: str,
    adj_vol_col: str,
    vendor_col: str,
    pl_flag_col: str,
    sales_pct_col: str,
    openai_col: str,
    vpn_code_col: str,
    item_code_col: str,
    use_vpn: bool,
    target_vpn_code: str = '',
    target_item_code: str = ''
) -> Optional[Dict[str, Any]]:
    """
    Create match data dictionary.
    
    Returns:
        Match data dictionary or None if should be excluded
    """
    match_info = {
        item_id_col: match_entity_item_id,
        'Description': "ID not found",
        'attributes': {},
        'is_target': False,
        sales_col: None,
        sales_pct_col: None,
        cogs_col: None,
        adj_vol_col: None,
        pl_flag_col: "No",
        vendor_col: 'N/A',
        'GM%': None
    }
    
    if im_final_full.empty or item_id_col not in im_final_full.columns:
        return match_info
    
    match_df_filtered = im_final_full[im_final_full[item_id_col] == match_entity_item_id]
    if match_df_filtered.empty:
        return match_info
    
    match_series = match_df_filtered.iloc[0]
    
    # Check VPN exclusion if using VPN
    if use_vpn:
        match_vpn_code = match_series.get(vpn_code_col, '')
        if _should_exclude_vpn_match(match_vpn_code, target_vpn_code, target_item_code, use_vpn):
            print(f"Skipping match {match_entity_item_id} due to VPN matching conditions")
            return None
    
    # Populate match data
    match_info[item_id_col] = match_series.get(item_id_col, match_entity_item_id)
    match_info['Description'] = match_series.get(actual_desc_col_to_use, "Desc. N/A")
    
    match_l3m_sales = match_series.get(sales_col, 0.0)
    match_l3m_cogs = match_series.get(cogs_col, 0.0)
    match_l3m_adj_vol = match_series.get(adj_vol_col, 0.0)
    
    match_info[sales_col] = match_l3m_sales
    match_info[cogs_col] = match_l3m_cogs
    match_info[adj_vol_col] = match_l3m_adj_vol
    match_info[vendor_col] = match_series.get(vendor_col, 'N/A')
    match_info[sales_pct_col] = match_series.get(sales_pct_col)
    match_info[pl_flag_col] = "Yes" if match_series.get(pl_flag_col) else "No"
    
    # Calculate GM%
    if isinstance(match_l3m_sales, (int, float)) and isinstance(match_l3m_cogs, (int, float)) and match_l3m_sales != 0:
        match_info['GM%'] = (match_l3m_sales - match_l3m_cogs) / match_l3m_sales
    
    # Parse OpenAI response
    openai_response_str = match_series.get(openai_col, '')
    if pd.isna(openai_response_str):
        openai_response_str = ''
    match_info['attributes'] = _parse_openai_response(openai_response_str)
    
    return match_info

# =============================================================================
# MAIN MATCHING FUNCTION
# =============================================================================

def write_top_matches_consolidated(
    im_final_full2: pd.DataFrame,
    base_filename: str,
    n: int = 10,
    sales_col: str = DEFAULT_COLUMNS['sales'],
    cogs_col: str = DEFAULT_COLUMNS['cogs'],
    adj_vol_col: str = DEFAULT_COLUMNS['adj_vol'],
    vendor_col: str = DEFAULT_COLUMNS['vendor'],
    item_id_col: str = DEFAULT_COLUMNS['item_id'],
    main_desc_col: str = DEFAULT_COLUMNS['description'],
    fallback_desc_col: str = DEFAULT_COLUMNS['description'],
    reasoning_col: str = 'reasoning',
    model_matches_col: str = DEFAULT_COLUMNS['matches'],
    openai_col: str = DEFAULT_COLUMNS['attributes'],
    pl_flag_col: str = DEFAULT_COLUMNS['private_label'],
    sales_pct_col: str = DEFAULT_COLUMNS['sales_pct'],
    item_code_col: str = 'item_code',
    vpn_code_col: str = 'vpn_code',
    pl: bool = False,
    use_vpn: bool = False
) -> None:
    """
    Generate an Excel report summarizing top items and their model-recommended substitutes.
    
    This is a consolidated version of write_top_matches and write_top_matches_VPN.
    
    Args:
        im_final_full2: Main dataset containing items, matches, and associated metrics/attributes
        base_filename: Output path for the Excel report
        n: Number of top SKUs to include based on sales
        sales_col, cogs_col, adj_vol_col: Column names for sales, cost, and adjusted volume metrics
        vendor_col, item_id_col, item_code_col: Identifiers for vendors and items
        main_desc_col, fallback_desc_col: Primary and fallback columns for item descriptions
        reasoning_col: Column with reasoning text per top item
        model_matches_col: Column containing model-suggested substitutes
        openai_col: Column with OpenAI response strings to be parsed into attributes
        pl_flag_col: Column indicating whether an item is private label
        sales_pct_col: Sales percent column used in summary and reporting
        vpn_code_col: Column for VPN codes (used only if use_vpn=True)
        pl: If True, filter top-N selection to private label items only
        use_vpn: If True, enable VPN code exclusion logic and include VPN-related columns
    """
    print(f"Starting write_top_matches_consolidated with use_vpn={use_vpn}")
    
    # Define essential columns based on VPN usage
    essential_cols = [
        item_id_col, main_desc_col, sales_col, cogs_col, adj_vol_col,
        vendor_col, reasoning_col, model_matches_col,
        openai_col, pl_flag_col, sales_pct_col, item_code_col
    ]
    if use_vpn:
        essential_cols.append(vpn_code_col)
    
    # Validate inputs
    if not _validate_write_matches_inputs(im_final_full2, base_filename, essential_cols):
        return
    
    # Clean and prepare DataFrame
    im_final_full = _clean_dataframe_columns(im_final_full2, item_id_col, vpn_code_col, item_code_col, use_vpn)
    im_final_full = _ensure_essential_columns(
        im_final_full, essential_cols, model_matches_col, openai_col, reasoning_col, 
        vpn_code_col, item_code_col, use_vpn
    )
    
    # Handle vendor column
    if vendor_col in im_final_full.columns:
        im_final_full[vendor_col] = im_final_full[vendor_col].fillna('Unknown')
    
    # Get description column
    im_final_full, actual_desc_col_to_use = _get_description_column(im_final_full, main_desc_col, fallback_desc_col)
    
    # Calculate totals
    total_category_sales = im_final_full[sales_col].sum() if sales_col in im_final_full.columns and pd.api.types.is_numeric_dtype(im_final_full[sales_col]) else 0.0
    total_category_cogs = im_final_full[cogs_col].sum() if cogs_col in im_final_full.columns and pd.api.types.is_numeric_dtype(im_final_full[cogs_col]) else 0.0
    total_category_adj_vol = im_final_full[adj_vol_col].sum() if adj_vol_col in im_final_full.columns and pd.api.types.is_numeric_dtype(im_final_full[adj_vol_col]) else 0.0
    
    # Get top N items
    top_n_items = pd.DataFrame()
    if not im_final_full.empty and sales_col in im_final_full.columns and pd.api.types.is_numeric_dtype(im_final_full[sales_col]):
        if pl:
            top_n_items = im_final_full[im_final_full[pl_flag_col] == True].sort_values(by=sales_col, ascending=False).head(n)
        else:
            top_n_items = im_final_full.sort_values(by=sales_col, ascending=False).head(n)
    
    if top_n_items.empty:
        print(f"No items for 'Top {n}' section. Individual sheets will not be created.")
        return
    
    # Initialize data structures
    summary_data_accumulator = {
        'Targets': {sales_col: 0.0, cogs_col: 0.0, adj_vol_col: 0.0},
        'Substitutes': {sales_col: 0.0, cogs_col: 0.0, adj_vol_col: 0.0}
    }
    processed_sheet_info_for_formulas = []
    
    # Define headers configuration
    fixed_headers_config_list = [
        (item_id_col, 40, False),
        ("Description", 60, True),
        (sales_col, 15, False),
        (sales_pct_col, 12, False),
        (cogs_col, 15, False),
        (adj_vol_col, 15, False),
        (pl_flag_col, 8, False),
        (vendor_col, 20, False),
        ("GM%", 10, True)
    ]
    
    num_fixed_cols = len(fixed_headers_config_list)
    
    # Create mapping of metric columns to their display indices
    fixed_display_indices_for_metrics = {}
    current_idx = 0
    for header_text, _, _ in fixed_headers_config_list:
        if header_text == sales_col:
            fixed_display_indices_for_metrics[sales_col] = current_idx
        elif header_text == cogs_col:
            fixed_display_indices_for_metrics[cogs_col] = current_idx
        elif header_text == adj_vol_col:
            fixed_display_indices_for_metrics[adj_vol_col] = current_idx
        current_idx += 1
    
    # Create Excel writer
    try:
        writer = pd.ExcelWriter(base_filename, engine='xlsxwriter')
    except ImportError:
        print("xlsxwriter not found, falling back to openpyxl. Formatting will be basic, summary will be static.")
        writer = pd.ExcelWriter(base_filename, engine='openpyxl')
    except Exception as e:
        print(f"Error creating ExcelWriter: {e}")
        return
    
    # Initialize workbook and summary sheet
    summary_sheet_name = "Summary"
    summary_ws = None
    workbook = None
    if writer.engine == 'xlsxwriter':
        workbook = writer.book
        summary_ws = workbook.add_worksheet(summary_sheet_name)
    
    # Process each top item
    for index, item_row in top_n_items.iterrows():
        # Create target data
        target_data = _create_target_data(
            item_row, item_id_col, actual_desc_col_to_use, sales_col, cogs_col, adj_vol_col,
            vendor_col, pl_flag_col, sales_pct_col, openai_col, vpn_code_col, item_code_col, use_vpn
        )
        
        # Accumulate target metrics
        if pd.notna(target_data[sales_col]):
            summary_data_accumulator['Targets'][sales_col] += target_data[sales_col]
        if pd.notna(target_data[cogs_col]):
            summary_data_accumulator['Targets'][cogs_col] += target_data[cogs_col]
        if pd.notna(target_data[adj_vol_col]):
            summary_data_accumulator['Targets'][adj_vol_col] += target_data[adj_vol_col]
        
        # Get reasoning
        item_reasoning = item_row.get(reasoning_col, '')
        
        # Parse model matches
        model_match_ids = _parse_model_matches(item_row.get(model_matches_col, []))
        
        # Process matches
        all_items_for_table_data = [target_data]
        all_attribute_keys_for_table = set(target_data['attributes'].keys())
        substitute_items_data_for_current_target = []
        
        for match_entity_item_id in model_match_ids:
            match_info = _create_match_data(
                match_entity_item_id, im_final_full, item_id_col, actual_desc_col_to_use,
                sales_col, cogs_col, adj_vol_col, vendor_col, pl_flag_col, sales_pct_col,
                openai_col, vpn_code_col, item_code_col, use_vpn,
                target_data.get('target_vpn_code', ''), target_data.get('target_item_code', '')
            )
            
            if match_info is None:  # Excluded due to VPN logic
                continue
            
            all_items_for_table_data.append(match_info)
            all_attribute_keys_for_table.update(match_info['attributes'].keys())
            
            if match_info.get(item_id_col) != "ID not found":
                substitute_items_data_for_current_target.append(match_info)
                
                # Accumulate substitute metrics
                if pd.notna(match_info[sales_col]):
                    summary_data_accumulator['Substitutes'][sales_col] += match_info[sales_col]
                if pd.notna(match_info[cogs_col]):
                    summary_data_accumulator['Substitutes'][cogs_col] += match_info[cogs_col]
                if pd.notna(match_info[adj_vol_col]):
                    summary_data_accumulator['Substitutes'][adj_vol_col] += match_info[adj_vol_col]
        
        # Create individual worksheet for this target item
        _create_individual_worksheet(
            workbook, target_data, substitute_items_data_for_current_target,
            all_items_for_table_data, all_attribute_keys_for_table,
            fixed_headers_config_list, fixed_display_indices_for_metrics,
            sales_col, cogs_col, adj_vol_col, item_id_col,
            processed_sheet_info_for_formulas, item_reasoning
        )
        
        print(f"Processed target item: {target_data[item_id_col]} with {len(substitute_items_data_for_current_target)} substitutes")
    
    # Create summary sheet
    if writer.engine == 'xlsxwriter' and workbook and summary_ws:
        _create_summary_sheet(
            summary_ws, workbook, processed_sheet_info_for_formulas,
            summary_data_accumulator, total_category_sales, total_category_cogs, 
            total_category_adj_vol, sales_col, cogs_col, adj_vol_col, n
        )
    
    # Close writer
    if 'writer' in locals() and writer is not None:
        try:
            if hasattr(writer, 'close'):
                writer.close()
            elif hasattr(writer, 'save'):
                writer.save()
            
            final_check_condition = os.path.exists(base_filename) and os.path.getsize(base_filename) > 0
            if not top_n_items.empty or total_category_sales > 0:
                print(f"Excel report {'successfully created' if final_check_condition else 'creation attempted (check file)'}: {base_filename}")
            else:
                print(f"Excel report creation failed or resulted in an empty file: {base_filename}")
        except Exception as e:
            print(f"An error occurred while saving/closing the Excel file: {e}")
    else:
        print("Excel writer was not initialized. Report not created.")

def make_embeddings(
    im_final: pd.DataFrame, 
    sales_col: str = DEFAULT_COLUMNS['sales']
) -> pd.DataFrame:
    """
    Extract text from openai_response and generate embeddings using OpenAI.
    
    Args:
        im_final: DataFrame with openai_response column in "key: value | ..." format
        sales_col: Column name for computing sales-based percentages
    
    Returns:
        Filtered DataFrame with for_embedding, embeddings, and Sales % columns
    """
    print("Starting make_embeddings...")

    # Process attributes into for_embedding
    def format_response(resp: str) -> str:
        if pd.isna(resp): 
            return None
        parts = str(resp).strip().split(" | ")
        values = []
        for part in parts:
            if ':' in part:
                _, val = part.split(":", 1)
                val = val.strip()
                if val.lower() != "null":
                    values.append(val)
        return " ".join(values).strip() if values else None

    im_final['for_embedding'] = im_final[DEFAULT_COLUMNS['attributes']].apply(format_response)

    # Filter rows with valid embedding text
    im_final_full = im_final[im_final['for_embedding'].str.strip().str.len() > 0].copy()
    print(f"Rows prepared for embedding: {len(im_final_full)}")

    if im_final_full.empty:
        print("No valid rows found. Exiting.")
        return im_final_full

    # Generate embeddings
    agent = OpenAIAgent(model='gpt-4o-mini', chunk_size=1)
    batch_size = 100
    embedded_text_list = []

    descriptions = im_final_full['for_embedding'].tolist()
    for i in tqdm(range(0, len(descriptions), batch_size), desc="Embedding Batches"):
        batch = descriptions[i:i + batch_size]
        try:
            embeddings_result = agent.embeddings(batch)
            embedded_text_list.extend(embeddings_result)
        except Exception as e:
            print(f"Batch {i // batch_size} failed: {e}")
            embedded_text_list.extend([None] * len(batch))

    if len(embedded_text_list) == len(im_final_full):
        im_final_full['embeddings'] = embedded_text_list
    else:
        print("Embedding count mismatch. Skipping embeddings column assignment.")
        return im_final_full

    # Add Sales %
    total_sales = im_final_full[sales_col].sum()
    im_final_full[DEFAULT_COLUMNS['sales_pct']] = im_final_full[sales_col] / total_sales if total_sales else 0

    print("Finished embedding generation.")
    return im_final_full

def get_user_prompts(
    im_final_full: pd.DataFrame, 
    n: int = 10, 
    vendor_exclusion: bool = False, 
    use_faiss: bool = True, 
    use_levenshtein: bool = True
) -> Tuple[List[str], Dict[int, str]]:
    # Check if DataFrame is empty at the start
    if im_final_full.empty:
        print("Critical: Input DataFrame is empty!")
        print("Check your data loading and preprocessing steps.")
        return [], {}
    """
    Generate user prompts with optional FAISS and Levenshtein similarity.
    
    Args:
        im_final_full: DataFrame with embeddings, descriptions, and attributes
        n: Number of top similar items per prompt
        vendor_exclusion: Whether to exclude same-vendor matches
        use_faiss: Whether to use FAISS for cosine similarity
        use_levenshtein: Whether to use Levenshtein similarity filtering
    
    Returns:
        Tuple of (user_prompts, numerical_id_to_entity_id_map)
    """
    df = im_final_full.copy()

    # Filter rows with valid embeddings
    df = df[df['embeddings'].notna()]
    df = df[df['embeddings'].apply(lambda x: isinstance(x, (list, np.ndarray)))]

    df = df.reset_index(drop=True)
    df['Numerical_ID'] = np.arange(1, len(df) + 1)
    entity_to_num = dict(zip(df[DEFAULT_COLUMNS['item_id']], df['Numerical_ID']))
    num_to_entity = dict(zip(df['Numerical_ID'], df[DEFAULT_COLUMNS['item_id']]))

    # Levenshtein source column fallback
    lev_col = 'for_embedding' if 'for_embedding' in df.columns else DEFAULT_COLUMNS['description']
    df[lev_col] = df[lev_col].fillna('').astype(str)
    lev_texts = df[lev_col].tolist()

    # Prepare embeddings numpy array and normalize for cosine similarity
    emb_list = []
    for emb in df['embeddings']:
        if isinstance(emb, list):
            emb_list.append(np.array(emb, dtype=np.float32))
        elif isinstance(emb, np.ndarray):
            emb_list.append(emb.astype(np.float32))
        else:
            raise TypeError(f"Unsupported embedding type: {type(emb)}")
    embeddings = np.vstack(emb_list)
    faiss.normalize_L2(embeddings)

    # Build FAISS index if enabled
    if use_faiss:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

    df['description_display_str'] = df[DEFAULT_COLUMNS['description']].fillna('').astype(str)

    template_str = """Original Item:
- Entity ID: {{ eid_numerical }}
{% if description_display.strip() %}
- Description: {{ description_display.strip() }}
- {{ attributes.strip().replace('| ', '\\n- ').rstrip() }}
{% endif %}
Top Similar Items from Descriptions:
{% for item in top_items %}
- Entity ID: {{ item['Numerical_ID'] }}
- Description: {{ item['description_display'].strip() }}
- {{ item['attributes'].replace('| ', '\\n- ').rstrip() }}
{% endfor %}

Return a list of Entity IDs that are swappable with the given Entity ID from the list of similar items.
If an item doesn't seem like it is swappable you can return None.
"""
    template = Template(template_str)

    def normalized_lev_sim(s1: str, s2: str) -> float:
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        dist = Levenshtein.distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1.0 - dist / max_len if max_len else 1.0

    user_prompts = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Prompts"):
        current_num_id = row['Numerical_ID']
        current_emb = embeddings[idx]
        current_lev_text = lev_texts[idx]
        current_desc = row['description_display_str']
        current_attrs = row.get('attributes', '')
        current_vendor = row.get('vgn_name', '')

        if use_faiss:
            # Embeddings are already normalized
            _, sim_indices = index.search(np.expand_dims(current_emb, axis=0), n + 10)
            sim_indices = sim_indices[0]
        else:
            sims = cosine_similarity(current_emb.reshape(1, -1), embeddings)[0]
            sim_indices = sims.argsort()[::-1][:n + 10]

        candidates = []
        for sim_idx in sim_indices:
            if sim_idx == idx:
                continue

            candidate = df.iloc[sim_idx]
            if vendor_exclusion and candidate.get('vgn_name', '') == current_vendor:
                continue

            lev_score = None
            if use_levenshtein:
                lev_score = normalized_lev_sim(current_lev_text, lev_texts[sim_idx])
                if lev_score < 0.5:
                    continue

            candidates.append({
                'Entity--Item': candidate['Entity--Item'],
                'Numerical_ID': candidate['Numerical_ID'],
                'description_display': candidate['description_display_str'],
                'attributes': candidate.get('attributes', ''),
                'vgn_name': candidate.get('vgn_name', ''),
                'levenshtein_sim': lev_score
            })
            if len(candidates) >= n:
                break

        prompt = template.render(
            eid_numerical=current_num_id,
            description_display=current_desc,
            attributes=current_attrs,
            top_items=candidates
        )
        user_prompts.append(prompt)

    return user_prompts, num_to_entity

def generate_matches(
    im_final_full: pd.DataFrame,
    user_prompts: List[str],
    numerical_id_to_entity_id_map: Dict[int, str],
    hard_rules: Optional[str] = None,
    batch_model: bool = True,
    chunk_size: int = 32,
    incorrect_value: str = 'Not Applicable'
) -> pd.DataFrame:
    """
    Generate item-to-item matches by querying an AI model with structured prompts.
    
    This function sends formatted prompts to an AI model in batches, enforces strict JSON
    response format, parses responses, maps numerical IDs back to original identifiers,
    and populates the 'Matches' and 'reasoning' columns in the input DataFrame.
    
    Args:
        im_final_full: DataFrame of items to be updated with AI-generated matches
        user_prompts: List of formatted text prompts for AI evaluation
        numerical_id_to_entity_id_map: Mapping from numerical IDs to original Entity--Item identifiers
        hard_rules: Optional additional instructions for AI system prompt
        batch_model: Whether to use batch processing model
        chunk_size: Number of prompts to send in single batch
        incorrect_value: String value to clean from final matches list
        
    Returns:
        DataFrame updated with 'Matches' and 'reasoning' columns from AI responses
        
    Raises:
        ValidationError: If input validation fails
    """
    # Validate inputs
    validate_dataframe(im_final_full, "im_final_full")
    validate_list_input(user_prompts, "user_prompts")
    validate_string_input(incorrect_value, "incorrect_value")
    
    if not numerical_id_to_entity_id_map:
        raise ValidationError("numerical_id_to_entity_id_map cannot be empty")
    
    # Create JSON schema for response validation
    response_schema = _create_response_schema()
    
    # Configure model parameters
    model_kwargs = {
        'temperature': 0.0,
        'response_format': {
            "type": "json_schema",
            "json_schema": {
                "name": "item_swap_evaluation",
                "description": "Schema for evaluating item swaps with numerical IDs",
                "schema": response_schema
            }
        }
    }
    
    # Initialize AI agent
    model = 'gpt-4o-batch' if batch_model else 'gpt-4o'
    agent = OpenAIAgent(model=model, chunk_size=chunk_size)
    
    # Generate system prompt
    system_prompt = _create_system_prompt(hard_rules)
    
    # Get AI responses
    prompts = agent.generate_prompts(system_prompt, user_prompts)
    responses = agent.get_responses(prompts, **model_kwargs, batch=batch_model)
    
    # Initialize result columns
    _initialize_result_columns(im_final_full)
    
    # Process responses
    _process_ai_responses(responses, im_final_full, numerical_id_to_entity_id_map, incorrect_value)
    
    # Clean final matches
    im_final_full['Matches'] = im_final_full['Matches'].apply(
        lambda x: _clean_matches(x, incorrect_value)
    )
    
    print("Finished processing matches.")
    return im_final_full


def _create_response_schema() -> Dict:
    """Create JSON schema for AI response validation."""
    return {
        "type": "object",
        "properties": {
            "Entity ID": {
                "type": "integer",
                "description": "The NUMERICAL Entity ID of the primary item that was evaluated"
            },
            "Matches": {
                "type": "array",
                "description": "List of NUMERICAL Entity IDs considered suitable swaps",
                "items": {
                    "type": "integer",
                    "description": "NUMERICAL Entity ID of a swappable item from the prompt options"
                }
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why these items were chosen as swappable"
            }
        },
        "required": ["Entity ID", "Matches", "reasoning"],
        "additionalProperties": False
    }


def _create_system_prompt(hard_rules: Optional[str]) -> str:
    """Create system prompt for AI model."""
    hard_rules_str = f"Also: {hard_rules}" if hard_rules else ""
    
    return f"""
You work at a company that buys products. The company that vends you the products has a list of items that they sell that they are trying to get you to buy instead of the item you currently buy.

You are to judge whether the swaps are valid and something you would be willing to do or not based on the similarities of the items.

You will be given an item and attributes of the item. You will also be given a list of items that are similar to the item you currently buy.

Please disregard the Entity-ID of the similar items, you are to focus on the description and attributes of the items.

Please pay close attention to the fields below 'Description'. The more of these fields that have matching or similar values between your original item and potential substitute items, the better the chance that they are a true match.

If all of these fields match, the products are definitely swappable. If some of these fields match, the products are probably swappable. If none of these fields match, the products are probably not swappable.

Please ignore fields where the result is "null". Two values of "null" are not a match.

You are to return a list of the Entity IDs that you would be willing to swap with the item you currently buy separated by columns.

If you do not think any of the items are swappable, return a blank list.

{hard_rules_str}

Respond in JSON format with the following fields:
- Entity ID: The Entity ID of the item you currently buy.
- Matches: The list of Entity IDs that you would be willing to swap with the item you currently buy. This list should only contain Entity IDs that were provided in the user prompt.
- reasoning: A brief explanation of why you chose those items.
"""


def _initialize_result_columns(df: pd.DataFrame) -> None:
    """Initialize 'Matches' and 'reasoning' columns if they don't exist."""
    if 'Matches' not in df.columns:
        df['Matches'] = pd.Series([None] * len(df), dtype='object')
    if 'reasoning' not in df.columns:
        df['reasoning'] = pd.Series([None] * len(df), dtype='str')


def _clean_matches(match_list: List[str], incorrect_value: str) -> List[str]:
    """Clean and validate match list."""
    if not isinstance(match_list, list):
        return []
    return [str(m).strip() for m in match_list if m and str(m).strip() != incorrect_value]


def _process_ai_responses(
    responses: List[str],
    df: pd.DataFrame,
    id_mapping: Dict[int, str],
    incorrect_value: str
) -> None:
    """Process AI responses and update DataFrame with matches and reasoning."""
    for i, json_str in enumerate(responses):
        if json_str is None:
            print(f"Warning: No AI response received for prompt index {i}. Skipping.")
            continue
            
        try:
            data = json.loads(json_str)
            _process_single_response(data, df, id_mapping, i)
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response {i}: {e}")
        except Exception as e:
            print(f"Unexpected error processing response {i}: {e}")


def _process_single_response(
    data: Dict,
    df: pd.DataFrame,
    id_mapping: Dict[int, str],
    response_index: int
) -> None:
    """Process a single AI response and update DataFrame."""
    responded_numerical_eid = data.get("Entity ID")
    numerical_matches_list = data.get("Matches", [])
    reasoning_text = data.get("reasoning")
    
    # Validate Entity ID
    if responded_numerical_eid is None or not isinstance(responded_numerical_eid, int):
        print(f"Warning: Invalid or missing 'Entity ID' in AI response index {response_index}. Skipping.")
        return
    
    # Map numerical ID back to original Entity ID
    original_entity_id = id_mapping.get(responded_numerical_eid)
    if original_entity_id is None:
        print(f"Warning: Responded numerical EID {responded_numerical_eid} not found in mapping. Skipping.")
        return
    
    # Process matches list
    string_matches_list = _process_matches_list(numerical_matches_list, id_mapping, response_index)
    
    # Update DataFrame
    _update_dataframe_with_matches(df, original_entity_id, string_matches_list, reasoning_text)


def _process_matches_list(
    numerical_matches_list: List,
    id_mapping: Dict[int, str],
    response_index: int
) -> List[str]:
    """Process numerical matches list and convert to string Entity IDs."""
    string_matches_list = []
    
    if not isinstance(numerical_matches_list, list):
        print(f"Warning: 'Matches' field in response {response_index} is not a list. Defaulting to empty.")
        return string_matches_list
    
    for num_id in numerical_matches_list:
        if not isinstance(num_id, int):
            print(f"Warning: Non-integer match ID '{num_id}' in response {response_index}. Skipping match.")
            continue
            
        mapped_str = id_mapping.get(num_id)
        if mapped_str:
            string_matches_list.append(mapped_str)
        else:
            print(f"Warning: Numerical match ID {num_id} not found in mapping. Skipping match.")
    
    return string_matches_list


def _update_dataframe_with_matches(
    df: pd.DataFrame,
    entity_id: str,
    matches_list: List[str],
    reasoning: str
) -> None:
    """Update DataFrame with matches and reasoning for given Entity ID."""
    matched_indices = df[df['Entity--Item'] == entity_id].index
    
    if not matched_indices.empty:
        for idx in matched_indices:
            df.at[idx, 'Matches'] = matches_list
            df.at[idx, 'reasoning'] = reasoning
    else:
        print(f"Warning: Original Entity ID '{entity_id}' not found in DataFrame.")


# =============================================================================
# EXCEL WRITING HELPER FUNCTIONS
# =============================================================================

def _create_individual_worksheet(
    workbook,
    target_data: Dict,
    substitute_items_data_for_current_target: List[Dict],
    all_items_for_table_data: List[Dict],
    all_attribute_keys_for_table: set,
    fixed_headers_config_list: List[Tuple],
    fixed_display_indices_for_metrics: Dict,
    sales_col: str,
    cogs_col: str,
    adj_vol_col: str,
    item_id_col: str,
    processed_sheet_info_for_formulas: List[Dict],
    item_reasoning: str
) -> None:
    """Create individual worksheet for a target item with its substitutes."""
    sorted_attribute_keys_for_table = sorted(list(all_attribute_keys_for_table))
    sheet_name_cleaned = re.sub(r'[\[\]*?:/\\ \n\r\t]', '_', str(target_data[item_id_col]))
    sheet_name = sheet_name_cleaned[:31]

    accept_reject_col_idx_current_sheet = len(fixed_headers_config_list) + len(sorted_attribute_keys_for_table)
    feedback_col_idx_current_sheet = accept_reject_col_idx_current_sheet + 1
    accept_reject_col_letter_current_sheet = xl_col_to_name(accept_reject_col_idx_current_sheet)
    
    metric_col_letters = {}
    if sales_col in fixed_display_indices_for_metrics:
         metric_col_letters[sales_col] = xl_col_to_name(fixed_display_indices_for_metrics[sales_col])
    if cogs_col in fixed_display_indices_for_metrics:
         metric_col_letters[cogs_col] = xl_col_to_name(fixed_display_indices_for_metrics[cogs_col])
    if adj_vol_col in fixed_display_indices_for_metrics:
         metric_col_letters[adj_vol_col] = xl_col_to_name(fixed_display_indices_for_metrics[adj_vol_col])

    sub_start_row_excel = 3
    sub_end_row_excel = sub_start_row_excel + len(substitute_items_data_for_current_target) - 1

    if substitute_items_data_for_current_target:
        processed_sheet_info_for_formulas.append({
            'name': sheet_name,
            'sub_rows': (sub_start_row_excel, sub_end_row_excel),
            'cols': metric_col_letters,
            'correct_col': accept_reject_col_letter_current_sheet
        })

    current_row_excel_on_sheet = 0
    worksheet = workbook.add_worksheet(sheet_name)
    worksheet.freeze_panes(1, 1)

    # Create formats
    thin_border = EXCEL_FORMATS['thin_border']
    thick_border_style = EXCEL_FORMATS['thick_border_style']
    bold_format = workbook.add_format({'bold': True})
    
    header_base = {
        'bold': True, 'bg_color': EXCEL_FORMATS['header_bg_color'], 
        'border': thin_border, 'text_wrap': True, 'valign': 'top', 'align': 'center'
    }
    header_fmt = workbook.add_format(header_base)
    header_fmt_thick_right = workbook.add_format({**header_base, 'right': thick_border_style})
    
    cell_base = {'border': thin_border, 'text_wrap': True, 'valign': 'top'}
    cell_fmt = workbook.add_format(cell_base)
    cell_fmt_thick_right = workbook.add_format({**cell_base, 'right': thick_border_style})
    
    curr_fmt = workbook.add_format({**cell_base, 'num_format': '$#,##0.00'})
    number_fmt = workbook.add_format({**cell_base, 'num_format': '#,##0.00'})
    pct_fmt = workbook.add_format({**cell_base, 'num_format': '0.00%'})
    pct_fmt_thick_right = workbook.add_format({**cell_base, 'num_format': '0.00%', 'right': thick_border_style})

    target_hl_bg = EXCEL_FORMATS['target_hl_bg']
    target_base = {'bg_color': target_hl_bg, **cell_base}
    target_fmt = workbook.add_format(target_base)
    target_fmt_thick_right = workbook.add_format({**target_base, 'right': thick_border_style})
    
    target_curr_fmt = workbook.add_format({**target_base, 'num_format': '$#,##0.00'})
    target_number_fmt = workbook.add_format({**target_base, 'num_format': '#,##0.00'})
    target_pct_fmt = workbook.add_format({**target_base, 'num_format': '0.00%'})
    target_pct_fmt_thick_right = workbook.add_format({**target_base, 'num_format': '0.00%', 'right': thick_border_style})

    accept_reject_cell_fmt_props = {**cell_base, 'align': 'center'}
    accept_reject_cell_fmt = workbook.add_format(accept_reject_cell_fmt_props)
    target_accept_reject_cell_fmt = workbook.add_format({**target_base, 'align': 'center'})

    # Write headers
    col_idx_write = 0
    for header_text, width, sep_after in fixed_headers_config_list:
        current_header_fmt = header_fmt_thick_right if sep_after else header_fmt
        worksheet.write(current_row_excel_on_sheet, col_idx_write, header_text, current_header_fmt)
        worksheet.set_column(col_idx_write, col_idx_write, width)
        col_idx_write += 1

    for attr_key in sorted_attribute_keys_for_table:
        worksheet.write(current_row_excel_on_sheet, col_idx_write, attr_key, header_fmt)
        worksheet.set_column(col_idx_write, col_idx_write, 20)
        col_idx_write += 1

    worksheet.write(current_row_excel_on_sheet, col_idx_write, "Accept/Reject", header_fmt)
    worksheet.set_column(col_idx_write, col_idx_write, 12)
    col_idx_write += 1
    worksheet.write(current_row_excel_on_sheet, col_idx_write, "Feedback", header_fmt)
    worksheet.set_column(col_idx_write, col_idx_write, 30)
    current_row_excel_on_sheet += 1

    # Write data rows
    for df_row_idx_loop, item_d in enumerate(all_items_for_table_data):
        excel_data_row = df_row_idx_loop + 1
        is_tgt = item_d['is_target']
        
        # Determine base formats
        item_id_f = target_fmt if is_tgt else cell_fmt
        desc_f = target_fmt_thick_right if is_tgt else cell_fmt_thick_right
        sales_f = target_curr_fmt if is_tgt else curr_fmt
        sales_pct_f = target_pct_fmt if is_tgt else pct_fmt
        cogs_f = target_curr_fmt if is_tgt else curr_fmt
        adj_vol_f = target_number_fmt if is_tgt else number_fmt
        pl_flag_f = target_fmt if is_tgt else cell_fmt
        vendor_f = target_fmt if is_tgt else cell_fmt
        gm_pct_f = target_pct_fmt_thick_right if is_tgt else pct_fmt_thick_right
        attr_f = target_fmt if is_tgt else cell_fmt
        ar_f = target_accept_reject_cell_fmt if is_tgt else accept_reject_cell_fmt
        feedback_f = target_fmt if is_tgt else cell_fmt

        c_idx_write = 0
        worksheet.write(excel_data_row, c_idx_write, item_d.get(item_id_col), item_id_f)
        c_idx_write += 1
        worksheet.write(excel_data_row, c_idx_write, item_d.get('Description'), desc_f)
        c_idx_write += 1
        worksheet.write(excel_data_row, c_idx_write, item_d.get(sales_col), sales_f)
        c_idx_write += 1
        worksheet.write(excel_data_row, c_idx_write, item_d.get('sales_pct_col'), sales_pct_f)
        c_idx_write += 1
        worksheet.write(excel_data_row, c_idx_write, item_d.get(cogs_col), cogs_f)
        c_idx_write += 1
        worksheet.write(excel_data_row, c_idx_write, item_d.get(adj_vol_col), adj_vol_f)
        c_idx_write += 1
        worksheet.write(excel_data_row, c_idx_write, item_d.get('pl_flag_col'), pl_flag_f)
        c_idx_write += 1
        worksheet.write(excel_data_row, c_idx_write, item_d.get('vendor_col'), vendor_f)
        c_idx_write += 1
        worksheet.write(excel_data_row, c_idx_write, item_d.get('GM%'), gm_pct_f)
        c_idx_write += 1

        for attr_key in sorted_attribute_keys_for_table:
            worksheet.write(excel_data_row, c_idx_write, item_d['attributes'].get(attr_key, ''), attr_f)
            c_idx_write += 1

        worksheet.write(excel_data_row, c_idx_write, "Accept", ar_f)
        worksheet.data_validation(excel_data_row, c_idx_write, excel_data_row, c_idx_write,
                                      {'validate': 'list', 'source': DATA_VALIDATION_OPTIONS_ACCEPT_REJECT})
        c_idx_write += 1
        worksheet.write(excel_data_row, c_idx_write, "", feedback_f)

    current_row_excel_on_sheet = len(all_items_for_table_data) + 1
    current_row_excel_on_sheet += 1
    worksheet.write(current_row_excel_on_sheet, 0, "Reasoning:", bold_format)
    current_row_excel_on_sheet += 1
    last_col_for_reasoning_merge = feedback_col_idx_current_sheet
    worksheet.merge_range(current_row_excel_on_sheet, 0, current_row_excel_on_sheet, max(1, last_col_for_reasoning_merge),
                          item_reasoning if pd.notna(item_reasoning) else "", cell_fmt)


def _create_summary_sheet(
    summary_ws,
    workbook,
    processed_sheet_info_for_formulas: List[Dict],
    summary_data_accumulator: Dict,
    total_category_sales: float,
    total_category_cogs: float,
    total_category_adj_vol: float,
    sales_col: str,
    cogs_col: str,
    adj_vol_col: str,
    n: int
) -> None:
    """Create summary sheet with aggregated metrics and formulas."""
    s_hdr_f = workbook.add_format({'bold': True, 'bg_color': EXCEL_FORMATS['summary_header_bg'], 'border': 1, 'align': 'left'})
    s_met_f = workbook.add_format({'border': 1, 'align': 'left', 'bold': True})
    s_val_f = workbook.add_format({'border': 1, 'num_format': '$#,##0.00', 'align': 'right'})
    s_num_f = workbook.add_format({'border': 1, 'num_format': '#,##0.00', 'align': 'right'})
    s_diff_val_f = workbook.add_format({'border': 1, 'num_format': '$#,##0.00', 'align': 'right', 'bg_color': EXCEL_FORMATS['diff_bg_color']})
    s_diff_num_f = workbook.add_format({'border': 1, 'num_format': '#,##0.00', 'align': 'right', 'bg_color': EXCEL_FORMATS['diff_bg_color']})

    summary_hdrs = ['Category', f'Total {sales_col}', f'Total {cogs_col}', f'Total {adj_vol_col}']
    summary_ws.set_column(0, 0, 45)
    summary_ws.set_column(1, 3, 20)
    for c, h_title in enumerate(summary_hdrs):
        summary_ws.write(0, c, h_title, s_hdr_f)
    
    current_summary_row = 1
    
    summary_ws.write_row(current_summary_row, 0, ["Total Category", total_category_sales, total_category_cogs, total_category_adj_vol], None)
    summary_ws.conditional_format(current_summary_row, 0, current_summary_row, 0, {'type': 'no_blanks', 'format': s_met_f})
    summary_ws.conditional_format(current_summary_row, 1, current_summary_row, 1, {'type': 'no_blanks', 'format': s_val_f})
    summary_ws.conditional_format(current_summary_row, 2, current_summary_row, 2, {'type': 'no_blanks', 'format': s_val_f})
    summary_ws.conditional_format(current_summary_row, 3, current_summary_row, 3, {'type': 'no_blanks', 'format': s_num_f})
    current_summary_row += 1
    
    target_summary_label = f"Top {n} SKUs by Sales (Targets)"
    summary_ws.write(current_summary_row, 0, target_summary_label, s_met_f)
    summary_ws.write(current_summary_row, 1, summary_data_accumulator['Targets'][sales_col], s_val_f)
    summary_ws.write(current_summary_row, 2, summary_data_accumulator['Targets'][cogs_col], s_val_f)
    summary_ws.write(current_summary_row, 3, summary_data_accumulator['Targets'][adj_vol_col], s_num_f)
    excel_row_for_top_n_targets_data = current_summary_row + 1
    current_summary_row += 1
    
    subs_summary_label = f"Potential Subs (SKUs matched to a top {n} SKU)"
    summary_ws.write(current_summary_row, 0, subs_summary_label, s_met_f)

    potential_formula_sales = "0"
    potential_formula_cogs = "0"
    potential_formula_adj_vol = "0"
    approved_formula_sales_terms_str = '0'
    approved_formula_cogs_terms_str = '0'
    approved_formula_adj_vol_terms_str = '0'

    if processed_sheet_info_for_formulas:
        potential_sales_terms, potential_cogs_terms, potential_adj_vol_terms = [], [], []
        approved_sales_terms, approved_cogs_terms, approved_adj_vol_terms = [], [], []
        
        for sheet_info in processed_sheet_info_for_formulas:
            s_name, s_rows, s_cols_map, s_correct_col_letter = sheet_info['name'], sheet_info['sub_rows'], sheet_info['cols'], sheet_info['correct_col']
            escaped_s_name_for_excel = s_name.replace("'", "''")
            quoted_sheet_name = f"'{escaped_s_name_for_excel}'"
            correct_r = f"{quoted_sheet_name}!{s_correct_col_letter}{s_rows[0]}:{s_correct_col_letter}{s_rows[1]}"
            
            sales_r_col_letter = s_cols_map.get(sales_col)
            cogs_r_col_letter = s_cols_map.get(cogs_col)
            adj_vol_r_col_letter = s_cols_map.get(adj_vol_col)

            if sales_r_col_letter:
                sales_r = f"{quoted_sheet_name}!{sales_r_col_letter}{s_rows[0]}:{sales_r_col_letter}{s_rows[1]}"
                potential_sales_terms.append(f"SUM({sales_r})")
                approved_sales_terms.append(f'SUMIFS({sales_r},{correct_r},"Accept")')
            if cogs_r_col_letter:
                cogs_r = f"{quoted_sheet_name}!{cogs_r_col_letter}{s_rows[0]}:{cogs_r_col_letter}{s_rows[1]}"
                potential_cogs_terms.append(f"SUM({cogs_r})")
                approved_cogs_terms.append(f'SUMIFS({cogs_r},{correct_r},"Accept")')
            if adj_vol_r_col_letter:
                adj_vol_r = f"{quoted_sheet_name}!{adj_vol_r_col_letter}{s_rows[0]}:{adj_vol_r_col_letter}{s_rows[1]}"
                potential_adj_vol_terms.append(f"SUM({adj_vol_r})")
                approved_adj_vol_terms.append(f'SUMIFS({adj_vol_r},{correct_r},"Accept")')
        
        if potential_sales_terms:
            potential_formula_sales = f"={'+'.join(potential_sales_terms)}"
        if potential_cogs_terms:
            potential_formula_cogs = f"={'+'.join(potential_cogs_terms)}"
        if potential_adj_vol_terms:
            potential_formula_adj_vol = f"={'+'.join(potential_adj_vol_terms)}"
        if approved_sales_terms:
            approved_formula_sales_terms_str = "+".join(approved_sales_terms) if approved_sales_terms else "0"
        if approved_cogs_terms:
            approved_formula_cogs_terms_str = "+".join(approved_cogs_terms) if approved_cogs_terms else "0"
        if approved_adj_vol_terms:
            approved_formula_adj_vol_terms_str = "+".join(approved_adj_vol_terms) if approved_adj_vol_terms else "0"

    summary_ws.write_formula(current_summary_row, 1, potential_formula_sales, s_val_f)
    summary_ws.write_formula(current_summary_row, 2, potential_formula_cogs, s_val_f)
    summary_ws.write_formula(current_summary_row, 3, potential_formula_adj_vol, s_num_f)
    excel_row_for_top_n_subs_data = current_summary_row + 1
    current_summary_row += 1
    
    summary_ws.write(current_summary_row, 0, "Impact of Non-Accepted Subs (Potential - Accepted)", s_met_f)
    summary_ws.write_formula(current_summary_row, 1, f"=B{excel_row_for_top_n_subs_data}-({approved_formula_sales_terms_str})", s_diff_val_f)
    summary_ws.write_formula(current_summary_row, 2, f"=C{excel_row_for_top_n_subs_data}-({approved_formula_cogs_terms_str})", s_diff_val_f)
    summary_ws.write_formula(current_summary_row, 3, f"=D{excel_row_for_top_n_subs_data}-({approved_formula_adj_vol_terms_str})", s_diff_num_f)

# =============================================================================
# LEGACY FUNCTION WRAPPERS
# =============================================================================

def write_top_matches(
    im_final_full2: pd.DataFrame,
    base_filename: str,
    n: int = 10,
    sales_col: str = DEFAULT_COLUMNS['sales'],
    cogs_col: str = DEFAULT_COLUMNS['cogs'],
    adj_vol_col: str = DEFAULT_COLUMNS['adj_vol'],
    vendor_col: str = DEFAULT_COLUMNS['vendor'],
    item_id_col: str = DEFAULT_COLUMNS['item_id'],
    main_desc_col: str = DEFAULT_COLUMNS['description'],
    fallback_desc_col: str = DEFAULT_COLUMNS['description'],
    reasoning_col: str = 'reasoning',
    model_matches_col: str = DEFAULT_COLUMNS['matches'],
    openai_col: str = DEFAULT_COLUMNS['attributes'],
    pl_flag_col: str = DEFAULT_COLUMNS['private_label'],
    sales_pct_col: str = DEFAULT_COLUMNS['sales_pct'],
    item_code_col: str = 'item_code',
    pl: bool = False
) -> None:
    """
    Legacy wrapper for write_top_matches_consolidated with use_vpn=False.
    """
    write_top_matches_consolidated(
        im_final_full2, base_filename, n, sales_col, cogs_col, adj_vol_col,
        vendor_col, item_id_col, main_desc_col, fallback_desc_col, reasoning_col,
        model_matches_col, openai_col, pl_flag_col, sales_pct_col, item_code_col,
        pl=pl, use_vpn=False
    )


def write_top_matches_VPN(
    im_final_full2: pd.DataFrame,
    base_filename: str,
    n: int = 10,
    sales_col: str = DEFAULT_COLUMNS['sales'],
    cogs_col: str = DEFAULT_COLUMNS['cogs'],
    adj_vol_col: str = DEFAULT_COLUMNS['adj_vol'],
    vendor_col: str = DEFAULT_COLUMNS['vendor'],
    item_id_col: str = DEFAULT_COLUMNS['item_id'],
    main_desc_col: str = DEFAULT_COLUMNS['description'],
    fallback_desc_col: str = DEFAULT_COLUMNS['description'],
    reasoning_col: str = 'reasoning',
    model_matches_col: str = DEFAULT_COLUMNS['matches'],
    openai_col: str = DEFAULT_COLUMNS['attributes'],
    pl_flag_col: str = DEFAULT_COLUMNS['private_label'],
    sales_pct_col: str = DEFAULT_COLUMNS['sales_pct'],
    item_code_col: str = 'item_code',
    vpn_code_col: str = 'vpn_code',
    pl: bool = False
) -> None:
    """
    Legacy wrapper for write_top_matches_consolidated with use_vpn=True.
    """
    write_top_matches_consolidated(
        im_final_full2, base_filename, n, sales_col, cogs_col, adj_vol_col,
        vendor_col, item_id_col, main_desc_col, fallback_desc_col, reasoning_col,
        model_matches_col, openai_col, pl_flag_col, sales_pct_col, item_code_col,
        vpn_code_col, pl=pl, use_vpn=True
    )