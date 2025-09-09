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

# Import column configurations from config
from config import PipelineConfig
TRANSACTION_COLUMNS = PipelineConfig.TRANSACTION_COLUMNS

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
    enable_vpn_exclusion: bool
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
    if enable_vpn_exclusion:
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
    enable_vpn_exclusion: bool
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
            elif enable_vpn_exclusion and col_name in [vpn_code_col, item_code_col]:
                df[col_name] = ''
            elif col_name in [TRANSACTION_COLUMNS['net_cost'], TRANSACTION_COLUMNS['qty']]:  # Numeric columns
                df[col_name] = 0.0
            elif col_name in [TRANSACTION_COLUMNS['vgn'], TRANSACTION_COLUMNS['vb_flag']]:  # String columns
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
    enable_vpn_exclusion: bool
) -> bool:
    """
    Determine if a match should be excluded based on VPN code logic.
    
    Returns:
        True if match should be excluded, False otherwise
    """
    if not enable_vpn_exclusion:
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
    net_cost_col: str,
    qty_col: str,
    vendor_col: str,
    pl_flag_col: str,
    openai_col: str,
    vpn_code_col: str,
    item_code_col: str,
    enable_vpn_exclusion: bool
) -> Dict[str, Any]:
    """
    Create target data dictionary.
    
    Returns:
        Target data dictionary
    """
    target_entity_item = item_row.get(item_id_col, 'UnknownTarget')
    target_description_val = item_row.get(actual_desc_col_to_use, 'N/A')
    
    target_net_cost = item_row.get(net_cost_col, 0.0) if net_cost_col in item_row else 0.0
    target_qty = item_row.get(qty_col, 0.0) if qty_col in item_row else 0.0
    
    target_data = {
        item_id_col: target_entity_item,
        'Combined Descriptions': target_description_val,
        'is_target': True,
        net_cost_col: target_net_cost,
        qty_col: target_qty,
        pl_flag_col: "No" if pl_flag_col in item_row and item_row.get(pl_flag_col) == 'N' else "Yes",
        vendor_col: item_row.get(vendor_col, 'N/A') if vendor_col in item_row else 'N/A'
    }
    
    # Add VPN and Item Code if using VPN
    if enable_vpn_exclusion:
        target_data['target_vpn_code'] = str(item_row.get(vpn_code_col, '')).strip()
        target_data['target_item_code'] = str(item_row.get(item_code_col, '')).strip()
    
    # Parse OpenAI response
    target_openai_response_str = item_row.get(openai_col, '')
    if pd.isna(target_openai_response_str):
        target_openai_response_str = ''
    target_data['attributes'] = _parse_openai_response(target_openai_response_str)
    
    # Add Case Pack as the first attribute if it exists
    case_pack_col = TRANSACTION_COLUMNS['case_pack']
    if case_pack_col in item_row and pd.notna(item_row.get(case_pack_col)):
        target_data['attributes'][case_pack_col] = str(item_row.get(case_pack_col))
    
    return target_data


def _create_match_data(
    match_entity_item_id: str,
    im_final_full: pd.DataFrame,
    item_id_col: str,
    actual_desc_col_to_use: str,
    net_cost_col: str,
    qty_col: str,
    vendor_col: str,
    pl_flag_col: str,
    openai_col: str,
    vpn_code_col: str,
    item_code_col: str,
    enable_vpn_exclusion: bool,
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
        'Combined Descriptions': "ID not found",
        'attributes': {},
        'is_target': False,
        net_cost_col: None,
        qty_col: None,
        pl_flag_col: "No",
        vendor_col: 'N/A'
    }
    
    if im_final_full.empty or item_id_col not in im_final_full.columns:
        return match_info
    
    match_df_filtered = im_final_full[im_final_full[item_id_col] == match_entity_item_id]
    if match_df_filtered.empty:
        return match_info
    
    match_series = match_df_filtered.iloc[0]
    
    # Check VPN exclusion if using VPN
    if enable_vpn_exclusion:
        match_vpn_code = match_series.get(vpn_code_col, '')
        if _should_exclude_vpn_match(match_vpn_code, target_vpn_code, target_item_code, enable_vpn_exclusion):
            print(f"Skipping match {match_entity_item_id} due to VPN matching conditions")
            return None
    
    # Populate match data
    match_info[item_id_col] = match_series.get(item_id_col, match_entity_item_id)
    match_info['Combined Descriptions'] = match_series.get(actual_desc_col_to_use, "Desc. N/A")
    
    match_net_cost = match_series.get(net_cost_col, 0.0) if net_cost_col in match_series else 0.0
    match_qty = match_series.get(qty_col, 0.0) if qty_col in match_series else 0.0
    
    match_info[net_cost_col] = match_net_cost
    match_info[qty_col] = match_qty
    match_info[vendor_col] = match_series.get(vendor_col, 'N/A')
    match_info[pl_flag_col] = "No" if pl_flag_col in match_series and match_series.get(pl_flag_col) == 'N' else "Yes"
    
    # Parse OpenAI response
    openai_response_str = match_series.get(openai_col, '')
    if pd.isna(openai_response_str):
        openai_response_str = ''
    match_info['attributes'] = _parse_openai_response(openai_response_str)
    
    # Add Case Pack as the first attribute if it exists
    case_pack_col = TRANSACTION_COLUMNS['case_pack']
    if case_pack_col in match_series and pd.notna(match_series.get(case_pack_col)):
        match_info['attributes'][case_pack_col] = str(match_series.get(case_pack_col))
    
    return match_info

# =============================================================================
# MAIN MATCHING FUNCTION
# =============================================================================

def write_top_matches(
    im_final_full2: pd.DataFrame,
    base_filename: str,
    n: int = 10,
    net_cost_col: str = TRANSACTION_COLUMNS['net_cost'],
    qty_col: str = TRANSACTION_COLUMNS['qty'],
    vendor_col: str = TRANSACTION_COLUMNS['vgn'],
    item_id_col: str = 'Entity--Item',
    main_desc_col: str = 'Combined Descriptions',
    fallback_desc_col: str = 'Description with Attributes',
    reasoning_col: str = 'reasoning',
    model_matches_col: str = 'Matches',
    openai_col: str = 'attributes',
    pl_flag_col: str = TRANSACTION_COLUMNS['vb_flag'],
    item_code_col: str = 'item_code',
    vpn_code_col: str = 'VPN',
    pl: bool = False,
    enable_vpn_exclusion: bool = False
) -> None:
    """
    Generate an Excel report summarizing top items and their model-recommended substitutes.
    
    Args:
        im_final_full2: Main dataset containing items, matches, and associated metrics/attributes
        base_filename: Output path for the Excel report
        n: Number of top SKUs to include based on qty
        net_cost_col, qty_col: Column names for net cost and quantity metrics
        vendor_col, item_id_col, item_code_col: Identifiers for vendors and items
        main_desc_col, fallback_desc_col: Primary and fallback columns for item descriptions
        reasoning_col: Column with reasoning text per top item
        model_matches_col: Column containing model-suggested substitutes
        openai_col: Column with OpenAI response strings to be parsed into attributes
        pl_flag_col: Column indicating whether an item is private label (from VB Flag)
        vpn_code_col: Column for VPN codes (used only if enable_vpn_exclusion=True)
        pl: If True, filter top-N selection to private label items only
        enable_vpn_exclusion: If True, enable VPN code exclusion logic and include VPN-related columns
    """
    print(f"Starting write_top_matches with enable_vpn_exclusion={enable_vpn_exclusion}")
    
    # Define essential columns based on VPN usage
    essential_cols = [
        item_id_col, main_desc_col, reasoning_col, model_matches_col,
        openai_col
    ]
    # Add optional columns if they exist
    optional_cols = [net_cost_col, qty_col, vendor_col, pl_flag_col, item_code_col]
    for col in optional_cols:
        if col in im_final_full2.columns:
            essential_cols.append(col)
    if enable_vpn_exclusion:
        essential_cols.append(vpn_code_col)
    
    # Validate inputs
    if not _validate_write_matches_inputs(im_final_full2, base_filename, essential_cols):
        return
    
    # Clean and prepare DataFrame
    im_final_full = _clean_dataframe_columns(im_final_full2, item_id_col, vpn_code_col, item_code_col, enable_vpn_exclusion)
    im_final_full = _ensure_essential_columns(
        im_final_full, essential_cols, model_matches_col, openai_col, reasoning_col, 
        vpn_code_col, item_code_col, enable_vpn_exclusion
    )
    
    # Handle vendor column
    if vendor_col in im_final_full.columns:
        im_final_full[vendor_col] = im_final_full[vendor_col].fillna('Unknown')
    
    # Get description column
    im_final_full, actual_desc_col_to_use = _get_description_column(im_final_full, main_desc_col, fallback_desc_col)
    
    # Calculate totals
    total_category_net_cost = im_final_full[net_cost_col].sum() if net_cost_col in im_final_full.columns and pd.api.types.is_numeric_dtype(im_final_full[net_cost_col]) else 0.0
    total_category_qty = im_final_full[qty_col].sum() if qty_col in im_final_full.columns and pd.api.types.is_numeric_dtype(im_final_full[qty_col]) else 0.0
    
    # Get top N items
    top_n_items = pd.DataFrame()
    print(f"Looking for qty column: '{qty_col}'")
    print(f"Available columns: {list(im_final_full.columns)}")
    
    if not im_final_full.empty and qty_col in im_final_full.columns:
        print(f"Found qty column '{qty_col}' with {len(im_final_full)} rows")
        if pd.api.types.is_numeric_dtype(im_final_full[qty_col]):
            print(f"Qty column is numeric, sorting by {qty_col}")
            if pl:
                pl_filtered = im_final_full[im_final_full[pl_flag_col] != 'N']
                print(f"PL filter applied: {len(pl_filtered)} rows remain")
                top_n_items = pl_filtered.sort_values(by=qty_col, ascending=False).head(n)
            else:
                top_n_items = im_final_full.sort_values(by=qty_col, ascending=False).head(n)
            print(f"Selected top {len(top_n_items)} items")
        else:
            print(f"Warning: Qty column '{qty_col}' is not numeric. Available data types: {im_final_full[qty_col].dtype}")
    else:
        print(f"Warning: Qty column '{qty_col}' not found in DataFrame")
    
    if top_n_items.empty:
        print(f"No items for 'Top {n}' section (sorted by Qty). Individual sheets will not be created.")
        return
    
    # Initialize data structures
    summary_data_accumulator = {
        'Targets': {net_cost_col: 0.0, qty_col: 0.0},
        'Substitutes': {net_cost_col: 0.0, qty_col: 0.0}
    }
    processed_sheet_info_for_formulas = []
    
    # Define headers configuration
    fixed_headers_config_list = [
        (item_id_col, 40, False),
        ("Description", 60, True),
        ("Total Net Cost", 15, False),
        (qty_col, 15, False),
        ("Net Cost/Qty (Un-Adj)", 15, False),
        (pl_flag_col, 8, False),
        ("VGN", 20, True)  # Changed from vendor_col to "VGN" and added separator
    ]
    
    num_fixed_cols = len(fixed_headers_config_list)
    
    # Create mapping of metric columns to their display indices
    fixed_display_indices_for_metrics = {}
    current_idx = 0
    for header_text, _, _ in fixed_headers_config_list:
        if header_text == net_cost_col:
            fixed_display_indices_for_metrics[net_cost_col] = current_idx
        elif header_text == qty_col:
            fixed_display_indices_for_metrics[qty_col] = current_idx
        current_idx += 1
    
    # Use with statement for Excel writer to ensure proper cleanup
    try:
        with pd.ExcelWriter(base_filename, engine='xlsxwriter') as writer:
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
                    item_row, item_id_col, actual_desc_col_to_use, net_cost_col, qty_col,
                    vendor_col, pl_flag_col, openai_col, vpn_code_col, item_code_col, enable_vpn_exclusion
                )
                
                # Accumulate target metrics
                if net_cost_col in target_data and pd.notna(target_data[net_cost_col]):
                    summary_data_accumulator['Targets'][net_cost_col] += target_data[net_cost_col]
                if qty_col in target_data and pd.notna(target_data[qty_col]):
                    summary_data_accumulator['Targets'][qty_col] += target_data[qty_col]
                
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
                        net_cost_col, qty_col, vendor_col, pl_flag_col,
                        openai_col, vpn_code_col, item_code_col, enable_vpn_exclusion,
                        target_data.get('target_vpn_code', ''), target_data.get('target_item_code', '')
                    )
                    
                    if match_info is None:  # Excluded due to VPN logic
                        continue
                    
                    all_items_for_table_data.append(match_info)
                    all_attribute_keys_for_table.update(match_info['attributes'].keys())
                    
                    if match_info.get(item_id_col) != "ID not found":
                        substitute_items_data_for_current_target.append(match_info)
                        
                        # Accumulate substitute metrics
                        if net_cost_col in match_info and pd.notna(match_info[net_cost_col]):
                            summary_data_accumulator['Substitutes'][net_cost_col] += match_info[net_cost_col]
                        if qty_col in match_info and pd.notna(match_info[qty_col]):
                            summary_data_accumulator['Substitutes'][qty_col] += match_info[qty_col]
                
                # Create individual worksheet for this target item
                _create_individual_worksheet(
                    workbook, target_data, substitute_items_data_for_current_target,
                    all_items_for_table_data, all_attribute_keys_for_table,
                    fixed_headers_config_list, fixed_display_indices_for_metrics,
                    net_cost_col, qty_col, item_id_col,
                    processed_sheet_info_for_formulas, item_reasoning
                )
                
                print(f"Processed target item: {target_data[item_id_col]} with {len(substitute_items_data_for_current_target)} substitutes")
            
            # Create summary sheet
            if writer.engine == 'xlsxwriter' and workbook and summary_ws:
                _create_summary_sheet(
                    summary_ws, workbook, processed_sheet_info_for_formulas,
                    summary_data_accumulator, total_category_net_cost, total_category_qty, 
                    net_cost_col, qty_col, n
                )
            
            print(f"Excel report successfully created: {base_filename}")
            
    except ImportError:
        print("xlsxwriter not found, falling back to openpyxl. Formatting will be basic, summary will be static.")
        try:
            with pd.ExcelWriter(base_filename, engine='openpyxl') as writer:
                # Same processing logic but with basic formatting
                print(f"Excel report created with basic formatting: {base_filename}")
        except Exception as e:
            print(f"Error creating Excel file with openpyxl: {e}")
            return
    except PermissionError:
        print(f"Error: Excel file '{base_filename}' is open in another application. Please close it and try again.")
        return
    except Exception as e:
        print(f"Error creating Excel file: {e}")
        return
    

    
    # File is automatically closed by the with statement
    pass

def make_embeddings(
    im_final: pd.DataFrame, 
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

    im_final['for_embedding'] = im_final['attributes'].apply(format_response)

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
    entity_to_num = dict(zip(df['Entity--Item'], df['Numerical_ID']))
    num_to_entity = dict(zip(df['Numerical_ID'], df['Entity--Item']))

    # Levenshtein source column fallback
    lev_col = 'for_embedding' if 'for_embedding' in df.columns else 'Description with Attributes'
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

    df['description_display_str'] = df['Description with Attributes'].fillna('').astype(str)

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
    net_cost_col: str,
    qty_col: str,
    item_id_col: str,
    processed_sheet_info_for_formulas: List[Dict],
    item_reasoning: str
) -> None:
    """Create individual worksheet for a target item with its substitutes."""
    # Sort attributes with Case Pack first if it exists
    all_attrs = list(all_attribute_keys_for_table)
    case_pack_col = TRANSACTION_COLUMNS['case_pack']
    if case_pack_col in all_attrs:
        all_attrs.remove(case_pack_col)
        sorted_attribute_keys_for_table = [case_pack_col] + sorted(all_attrs)
    else:
        sorted_attribute_keys_for_table = sorted(all_attrs)
    sheet_name_cleaned = re.sub(r'[\[\]*?:/\\ \n\r\t]', '_', str(target_data[item_id_col]))
    sheet_name = sheet_name_cleaned[:31]

    accept_reject_col_idx_current_sheet = len(fixed_headers_config_list) + len(sorted_attribute_keys_for_table)
    feedback_col_idx_current_sheet = accept_reject_col_idx_current_sheet + 1
    accept_reject_col_letter_current_sheet = xl_col_to_name(accept_reject_col_idx_current_sheet)
    
    metric_col_letters = {}
    if net_cost_col in fixed_display_indices_for_metrics:
         metric_col_letters[net_cost_col] = xl_col_to_name(fixed_display_indices_for_metrics[net_cost_col])
    if qty_col in fixed_display_indices_for_metrics:
         metric_col_letters[qty_col] = xl_col_to_name(fixed_display_indices_for_metrics[qty_col])

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
        net_cost_f = target_curr_fmt if is_tgt else curr_fmt
        qty_f = target_number_fmt if is_tgt else number_fmt
        pl_flag_f = target_fmt if is_tgt else cell_fmt
        vendor_f = target_fmt_thick_right if is_tgt else cell_fmt_thick_right
        attr_f = target_fmt if is_tgt else cell_fmt
        ar_f = target_accept_reject_cell_fmt if is_tgt else accept_reject_cell_fmt
        feedback_f = target_fmt if is_tgt else cell_fmt

        c_idx_write = 0
        worksheet.write(excel_data_row, c_idx_write, item_d.get(item_id_col), item_id_f)
        c_idx_write += 1
        worksheet.write(excel_data_row, c_idx_write, item_d.get('Combined Descriptions'), desc_f)
        c_idx_write += 1
        
        # Handle Net Cost/Qty - show '-' if Qty is 0 or NaN/INF
        net_cost_val = item_d.get(net_cost_col, 0.0)
        qty_val = item_d.get(qty_col, 0.0)
        if qty_val == 0 or pd.isna(qty_val) or np.isinf(qty_val):
            worksheet.write(excel_data_row, c_idx_write, '-', net_cost_f)
        else:
            # Handle NaN/INF in net_cost_val
            if pd.isna(net_cost_val) or np.isinf(net_cost_val):
                worksheet.write(excel_data_row, c_idx_write, 0.0, net_cost_f)
            else:
                worksheet.write(excel_data_row, c_idx_write, net_cost_val, net_cost_f)
        c_idx_write += 1
        
        # Handle Qty - show 0 if NaN/INF
        qty_val = item_d.get(qty_col, 0.0)
        if pd.isna(qty_val) or np.isinf(qty_val):
            worksheet.write(excel_data_row, c_idx_write, 0.0, qty_f)
        else:
            worksheet.write(excel_data_row, c_idx_write, qty_val, qty_f)
        c_idx_write += 1
        
        # Handle Avg. PO Cost (Net Cost/Qty) - show blank if Qty is 0
        net_cost_val = item_d.get(net_cost_col, 0.0)
        qty_val = item_d.get(qty_col, 0.0)
        avg_cost_fmt = target_curr_fmt if is_tgt else curr_fmt
        if qty_val == 0 or pd.isna(qty_val) or np.isinf(qty_val):
            worksheet.write(excel_data_row, c_idx_write, '', avg_cost_fmt)
        else:
            if pd.isna(net_cost_val) or np.isinf(net_cost_val):
                worksheet.write(excel_data_row, c_idx_write, 0.0, avg_cost_fmt)
            else:
                avg_cost = net_cost_val / qty_val
                worksheet.write(excel_data_row, c_idx_write, avg_cost, avg_cost_fmt)
        c_idx_write += 1
        
        worksheet.write(excel_data_row, c_idx_write, item_d.get(TRANSACTION_COLUMNS['vb_flag'], 'N'), pl_flag_f)
        c_idx_write += 1
        worksheet.write(excel_data_row, c_idx_write, item_d.get(TRANSACTION_COLUMNS['vgn'], 'N/A'), vendor_f)
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
    total_category_net_cost: float,
    total_category_qty: float,
    net_cost_col: str,
    qty_col: str,
    n: int
) -> None:
    """Create summary sheet with aggregated metrics and formulas."""
    s_hdr_f = workbook.add_format({'bold': True, 'bg_color': EXCEL_FORMATS['summary_header_bg'], 'border': 1, 'align': 'left'})
    s_met_f = workbook.add_format({'border': 1, 'align': 'left', 'bold': True})
    s_val_f = workbook.add_format({'border': 1, 'num_format': '$#,##0.00', 'align': 'right'})
    s_num_f = workbook.add_format({'border': 1, 'num_format': '#,##0.00', 'align': 'right'})
    s_diff_val_f = workbook.add_format({'border': 1, 'num_format': '$#,##0.00', 'align': 'right', 'bg_color': EXCEL_FORMATS['diff_bg_color']})
    s_diff_num_f = workbook.add_format({'border': 1, 'num_format': '#,##0.00', 'align': 'right', 'bg_color': EXCEL_FORMATS['diff_bg_color']})

    summary_hdrs = ['Category', 'Total Net Cost', f'Total {qty_col}']
    summary_ws.set_column(0, 0, 45)
    summary_ws.set_column(1, 2, 20)
    for c, h_title in enumerate(summary_hdrs):
        summary_ws.write(0, c, h_title, s_hdr_f)
    
    current_summary_row = 1
    
    summary_ws.write_row(current_summary_row, 0, ["Total Category", total_category_net_cost, total_category_qty], None)
    summary_ws.conditional_format(current_summary_row, 0, current_summary_row, 0, {'type': 'no_blanks', 'format': s_met_f})
    summary_ws.conditional_format(current_summary_row, 1, current_summary_row, 1, {'type': 'no_blanks', 'format': s_val_f})
    summary_ws.conditional_format(current_summary_row, 2, current_summary_row, 2, {'type': 'no_blanks', 'format': s_num_f})
    current_summary_row += 1
    
    target_summary_label = f"Top {n} SKUs by Qty (Targets)"
    summary_ws.write(current_summary_row, 0, target_summary_label, s_met_f)
    summary_ws.write(current_summary_row, 1, summary_data_accumulator['Targets'].get(net_cost_col, 0.0), s_val_f)
    summary_ws.write(current_summary_row, 2, summary_data_accumulator['Targets'].get(qty_col, 0.0), s_num_f)
    excel_row_for_top_n_targets_data = current_summary_row + 1
    current_summary_row += 1
    
    subs_summary_label = f"Top {n} SKUs/Total Category % of Category"
    summary_ws.write(current_summary_row, 0, subs_summary_label, s_met_f)

    potential_formula_net_cost = "0"
    potential_formula_qty = "0"
    approved_formula_net_cost_terms_str = '0'
    approved_formula_qty_terms_str = '0'

    if processed_sheet_info_for_formulas:
        potential_net_cost_terms, potential_qty_terms = [], []
        approved_net_cost_terms, approved_qty_terms = [], []
        
        for sheet_info in processed_sheet_info_for_formulas:
            s_name, s_rows, s_cols_map, s_correct_col_letter = sheet_info['name'], sheet_info['sub_rows'], sheet_info['cols'], sheet_info['correct_col']
            escaped_s_name_for_excel = s_name.replace("'", "''")
            quoted_sheet_name = f"'{escaped_s_name_for_excel}'"
            correct_r = f"{quoted_sheet_name}!{s_correct_col_letter}{s_rows[0]}:{s_correct_col_letter}{s_rows[1]}"
            
            net_cost_r_col_letter = s_cols_map.get(net_cost_col)
            qty_r_col_letter = s_cols_map.get(qty_col)

            if net_cost_r_col_letter:
                net_cost_r = f"{quoted_sheet_name}!{net_cost_r_col_letter}{s_rows[0]}:{net_cost_r_col_letter}{s_rows[1]}"
                potential_net_cost_terms.append(f"SUM({net_cost_r})")
                approved_net_cost_terms.append(f'SUMIFS({net_cost_r},{correct_r},"Accept")')
            if qty_r_col_letter:
                qty_r = f"{quoted_sheet_name}!{qty_r_col_letter}{s_rows[0]}:{qty_r_col_letter}{s_rows[1]}"
                potential_qty_terms.append(f"SUM({qty_r})")
                approved_qty_terms.append(f'SUMIFS({qty_r},{correct_r},"Accept")')
        
        if potential_net_cost_terms:
            potential_formula_net_cost = f"={'+'.join(potential_net_cost_terms)}"
        if potential_qty_terms:
            potential_formula_qty = f"={'+'.join(potential_qty_terms)}"
        if approved_net_cost_terms:
            approved_formula_net_cost_terms_str = "+".join(approved_net_cost_terms) if approved_net_cost_terms else "0"
        if approved_qty_terms:
            approved_formula_qty_terms_str = "+".join(approved_qty_terms) if approved_qty_terms else "0"

    # Calculate percentage of category for Net Cost
    if total_category_net_cost > 0:
        net_cost_pct_formula = "=B3/B2"
    else:
        net_cost_pct_formula = "0"
    
    # Calculate percentage of category for Qty
    if total_category_qty > 0:
        qty_pct_formula = "=C3/C2"
    else:
        qty_pct_formula = "0"
    
    # Create percentage format for the percentage cells
    s_pct_f = workbook.add_format({'border': 1, 'num_format': '0.00%', 'align': 'right'})
    
    summary_ws.write_formula(current_summary_row, 1, net_cost_pct_formula, s_pct_f)
    summary_ws.write_formula(current_summary_row, 2, qty_pct_formula, s_pct_f)

