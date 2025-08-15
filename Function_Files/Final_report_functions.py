"""
Final Report Functions for Category Management

This module provides functions for creating comprehensive Excel reports with optimization results,
including suggested swaps, drop lists, vendor performance summaries, and customer performance summaries.
It includes utilities for data processing, Excel formatting, and report generation.
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Optional, Any
import ast
from datetime import datetime
from collections import defaultdict

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Common column names
DEFAULT_COLUMNS = {
    'item_id': 'Entity--Item',
    'vendor': 'vgn_name',
    'description': 'All Descriptions',
    'fallback_description': 'description',
    'attributes': 'attributes',
    'matches': 'Matches',
    'sales': 'L3M_Sales',
    'cogs': 'L3M_Cogs',
    'adj_vol': 'L3M_adj_vol',
    'private_label': 'private_label_flag',
    'sales_pct': 'Sales %',
    'item_code': 'item_code',
    'vpn_code': 'vpn_code',
    'reasoning': 'reasoning'
}
# Excel formatting configuration
EXCEL_CONFIG = {
    'engine': 'xlsxwriter',
    'float_format': '%.2f'
}

# Report sheet names
REPORT_SHEETS = {
    'suggested_swaps': 'Suggested Swaps',
    'drop_list': 'Drop List',
    'vendor_summary': 'Vendor Performance Summary',
    'customer_summary': 'Customer Performance Summary'
}

# Excel formatting constants
EXCEL_FORMATS = {
    'thin_border': 1,
    'thick_border_style': 5,
    'header_bg_color': '#D3D3D3',
    'orig_header_bg': '#DAEEF3',
    'recv_header_bg': '#E2EFDA',
    'reviewed_header_bg': '#F2F2F2',
    'vendor_header_bg': '#E0EBF5',
    'vendor_volume_bg': '#CCE5FF',
    'vendor_sales_bg': '#FFE5CC',
    'customer_header_bg': '#E0EBF5',
    'customer_volume_bg': '#CCE5FF',
    'customer_sales_bg': '#FFE5CC',
    'dark_blue_bg': '#1f497d'
}

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Validate that DataFrame is not empty and is a pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"{name} must be a pandas DataFrame")
    if df.empty:
        raise ValidationError(f"{name} cannot be empty")

def validate_columns_exist(df: pd.DataFrame, required_columns: List[str], name: str = "DataFrame") -> None:
    """Validate that DataFrame contains all required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValidationError(f"{name} missing required columns: {missing_columns}")

def validate_string_input(value: str, name: str) -> None:
    """Validate that input is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")

def validate_list_input(value: List, name: str) -> None:
    """Validate that input is a non-empty list."""
    if not isinstance(value, list) or len(value) == 0:
        raise ValidationError(f"{name} must be a non-empty list")

# =============================================================================
# EXCEL FORMATTING HELPER FUNCTIONS
# =============================================================================

def _create_excel_formats(workbook) -> Dict:
    """Create all Excel formatting objects for the workbook."""
    formats = {}
    
    # General Properties
    formats['general_header'] = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 
        'fg_color': EXCEL_FORMATS['header_bg_color'], 'border': 1, 'align': 'center'
    })
    formats['general_cell'] = workbook.add_format({
        'text_wrap': True, 'valign': 'top', 'border': 1
    })
    formats['accept_reject_cell'] = workbook.add_format({
        'text_wrap': True, 'valign': 'top', 'border': 1, 'align': 'center'
    })
    formats['general_number'] = workbook.add_format({
        'num_format': '#,##0.00', 'text_wrap': True, 'valign': 'top', 'border': 1
    })
    formats['percentage'] = workbook.add_format({
        'num_format': '0.00%', 'text_wrap': True, 'valign': 'top', 'border': 1
    })
    formats['dollar_number'] = workbook.add_format({
        'num_format': '$#,##0.00', 'text_wrap': True, 'valign': 'top', 'border': 1
    })
    formats['gm_unit'] = workbook.add_format({
        'num_format': '#,##0.00', 'text_wrap': True, 'valign': 'top', 'border': 1
    })

    # Suggested Swaps sheet specific properties
    formats['orig_header'] = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 
        'fg_color': EXCEL_FORMATS['orig_header_bg'], 'border': 1, 'align': 'center'
    })
    formats['recv_header'] = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 
        'fg_color': EXCEL_FORMATS['recv_header_bg'], 'border': 1, 'align': 'center'
    })
    formats['reviewed_header'] = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 
        'fg_color': EXCEL_FORMATS['reviewed_header_bg'], 'border': 1, 'align': 'center'
    })

    # Vendor Performance Summary sheet specific properties
    formats['vendor_header_general'] = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 
        'fg_color': EXCEL_FORMATS['vendor_header_bg'], 'border': 1, 'align': 'center'
    })
    formats['vendor_header_volume'] = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 
        'fg_color': EXCEL_FORMATS['vendor_volume_bg'], 'border': 1, 'align': 'center'
    })
    formats['vendor_header_sales'] = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 
        'fg_color': EXCEL_FORMATS['vendor_sales_bg'], 'border': 1, 'align': 'center'
    })

    # Customer Performance Summary sheet specific properties
    formats['customer_header_general'] = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 
        'fg_color': EXCEL_FORMATS['customer_header_bg'], 'border': 1, 'align': 'center'
    })
    formats['customer_header_volume'] = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 
        'fg_color': EXCEL_FORMATS['customer_volume_bg'], 'border': 1, 'align': 'center'
    })
    formats['customer_header_sales'] = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 
        'fg_color': EXCEL_FORMATS['customer_sales_bg'], 'border': 1, 'align': 'center'
    })

    # Dark blue format for headers
    formats['dark_blue'] = workbook.add_format({
        'bold': False, 'fg_color': EXCEL_FORMATS['dark_blue_bg'], 
        'font_color': 'white', 'align': 'left'
    })
    
    return formats


def _get_cell_format(formats: Dict, column_name: str, is_header: bool = False) -> Any:
    """Get appropriate cell format based on column name and type."""
    if is_header:
        if 'Volume' in column_name:
            return formats['vendor_header_volume']
        elif 'Sales' in column_name:
            return formats['vendor_header_sales']
        else:
            return formats['vendor_header_general']
    else:
        if 'Sales' in column_name and '$' in column_name:
            return formats['dollar_number']
        elif 'Sales' in column_name and '%' in column_name:
            return formats['percentage']
        elif '%' in column_name:
            return formats['percentage']
        elif 'Volume' in column_name:
            return formats['general_number']
        else:
            return formats['general_cell']


def _write_worksheet_data(worksheet, df: pd.DataFrame, formats: Dict, 
                         start_row: int = 0, start_col: int = 0) -> None:
    """Write DataFrame to worksheet with appropriate formatting."""
    if df.empty:
        return
    
    # Write headers
    for col_idx, col_name in enumerate(df.columns):
        header_format = _get_cell_format(formats, col_name, is_header=True)
        worksheet.write(start_row, start_col + col_idx, col_name, header_format)
    
    # Write data
    for df_row_idx, data_tuple in enumerate(df.itertuples(index=False)):
        excel_row_num = start_row + 1 + df_row_idx
        for col_idx, col_name in enumerate(df.columns):
            cell_value = data_tuple[col_idx]
            cell_format = _get_cell_format(formats, col_name, is_header=False)
            
            if pd.notna(cell_value):
                worksheet.write(excel_row_num, start_col + col_idx, cell_value, cell_format)
            else:
                worksheet.write_blank(excel_row_num, start_col + col_idx, None, cell_format)


# =============================================================================
# DATA PROCESSING HELPER FUNCTIONS
# =============================================================================

def _is_empty_match_list(match_val) -> bool:
    """Check if a match value represents an empty list."""
    if isinstance(match_val, list):
        return not bool(match_val)
    return True

def _calculate_gm_percentage(sales: float, cogs: float) -> float:
    """Calculate gross margin percentage."""
    if sales > 0:
        return (sales - cogs) / sales
    return -np.inf

def _calculate_gm_per_unit(sales: float, cogs: float, volume: float) -> float:
    """Calculate gross margin per unit."""
    if volume > 0:
        return (sales - cogs) / volume
    return 0

def _identify_protected_skus(true_l3m_grouped: pd.DataFrame, item_id_col: str,
                           redi_customers: Optional[List], customer_exclusions: Optional[List]) -> set:
    """Identify SKUs that serve protected customers."""
    protected_skus = set()
    
    if redi_customers:
        redi_skus = true_l3m_grouped[true_l3m_grouped['customer'].isin(redi_customers)][item_id_col].unique()
        protected_skus.update(redi_skus)
    
    if customer_exclusions:
        excluded_skus = true_l3m_grouped[true_l3m_grouped['customer'].isin(customer_exclusions)][item_id_col].unique()
        protected_skus.update(excluded_skus)
    
    return protected_skus

def _prepare_data_for_processing(df: pd.DataFrame, suggest_volume_col: str, volume_change_col: str,
                               sales_col: str, cogs_col: str, l3m_adj_vol_col: str) -> pd.DataFrame:
    """Prepare DataFrame by converting columns to numeric and adding calculated fields."""
    df_prepared = df.copy()
    
    # Convert numeric columns
    numeric_cols = [suggest_volume_col, volume_change_col, sales_col, cogs_col, l3m_adj_vol_col]
    for col in numeric_cols:
        df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce').fillna(0)
    
    # Calculate GM percentage
    df_prepared['GM_Percentage'] = df_prepared.apply(
        lambda row: _calculate_gm_percentage(row[sales_col], row[cogs_col]), axis=1
    )
    
    return df_prepared

def _determine_excel_engine(output_excel_filepath: str) -> str:
    """Determine which Excel engine to use based on availability."""
    try:
        # Test if xlsxwriter is available and writable
        with pd.ExcelWriter(output_excel_filepath, engine='xlsxwriter') as test_writer:
            pass
        return 'xlsxwriter'
    except ImportError:
        print("xlsxwriter not found, falling back to openpyxl. Formatting will be basic.")
        return 'openpyxl'
    except Exception as e_init:
        print(f"Initial xlsxwriter test failed: {e_init}. Falling back to openpyxl.")
        return 'openpyxl'

def _validate_report_inputs(updated_df: pd.DataFrame, volume_redistribution_dict: Dict,
                           item_to_idx_map: Dict, feedback_df: pd.DataFrame,
                           true_l3m_grouped: pd.DataFrame, swaps_df: pd.DataFrame,
                           suggest_volume_col: str, volume_change_col: str,
                           item_id_col: str, vendor_col: str, all_descriptions_col: str,
                           attribute_col: str, matches_col: str, sales_col: str,
                           cogs_col: str, private_label_col: str, l3m_adj_vol_col: str) -> None:
    """Validate all inputs for the report creation function."""
    validate_dataframe(updated_df, "updated_df")
    validate_dataframe(feedback_df, "feedback_df")
    validate_dataframe(true_l3m_grouped, "true_l3m_grouped")
    validate_dataframe(swaps_df, "swaps_df")
    
    if not isinstance(volume_redistribution_dict, dict):
        raise ValidationError("volume_redistribution_dict must be a dictionary")
    if not isinstance(item_to_idx_map, dict):
        raise ValidationError("item_to_idx_map must be a dictionary")
    
    # Validate required columns for updated_df
    required_cols = [suggest_volume_col, volume_change_col, item_id_col, vendor_col, 
                    all_descriptions_col, attribute_col, matches_col, sales_col, 
                    cogs_col, private_label_col, l3m_adj_vol_col]
    validate_columns_exist(updated_df, required_cols, "updated_df")
    
    # Validate required columns for feedback_df
    required_feedback_cols = ['Targets', 'Subs', 'Correct']
    validate_columns_exist(feedback_df, required_feedback_cols, "feedback_df")
    
    # Validate required columns for true_l3m_grouped
    required_l3m_cols = ['customer', item_id_col, l3m_adj_vol_col, sales_col]
    validate_columns_exist(true_l3m_grouped, required_l3m_cols, "true_l3m_grouped")


def _prepare_dataframe_for_report(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """Prepare DataFrame by adding missing columns with NaN values."""
    df_prepared = df.copy()
    missing_cols = [col for col in required_cols if col not in df_prepared.columns]
    
    if missing_cols:
        print(f"Warning: DataFrame is missing expected columns: {', '.join(missing_cols)}. Adding with NaN values.")
        for col in missing_cols:
            df_prepared[col] = np.nan
    
    return df_prepared


def _create_suggested_swaps_data(df: pd.DataFrame, feedback_df: pd.DataFrame,
                               suggest_volume_col: str, volume_change_col: str,
                               item_id_col: str, vendor_col: str, all_descriptions_col: str,
                               attribute_col: str, matches_col: str, sales_col: str,
                               cogs_col: str, private_label_col: str, l3m_adj_vol_col: str) -> pd.DataFrame:
    """Create the suggested swaps data for the report."""
    # Filter for items with suggested volume > 0
    suggested_swaps = df[df[suggest_volume_col] > 0].copy()
    
    if suggested_swaps.empty:
        return pd.DataFrame()
    
    # Create originating SKU data
    orig_data = suggested_swaps[[item_id_col, vendor_col, all_descriptions_col, 
                               attribute_col, sales_col, cogs_col, private_label_col, 
                               l3m_adj_vol_col, suggest_volume_col, volume_change_col]].copy()
    
    # Add 'Reviewed' column based on feedback
    orig_data['Reviewed'] = 'No'
    for _, feedback_row in feedback_df.iterrows():
        target = feedback_row['Targets']
        if target in orig_data[item_id_col].values:
            orig_data.loc[orig_data[item_id_col] == target, 'Reviewed'] = 'Yes'
    
    return orig_data


def _create_drop_list_data(df: pd.DataFrame, item_id_col: str, vendor_col: str,
                          all_descriptions_col: str, attribute_col: str,
                          sales_col: str, cogs_col: str, private_label_col: str,
                          l3m_adj_vol_col: str) -> pd.DataFrame:
    """Create the drop list data for the report."""
    # Filter for items with suggested volume = 0
    drop_list = df[df['suggest_volume'] == 0].copy()
    
    if drop_list.empty:
        return pd.DataFrame()
    
    # Select relevant columns
    drop_list_data = drop_list[[item_id_col, vendor_col, all_descriptions_col, 
                               attribute_col, sales_col, cogs_col, private_label_col, 
                               l3m_adj_vol_col]].copy()
    
    return drop_list_data


# =============================================================================
# SHEET CREATION HELPER FUNCTIONS
# =============================================================================

def _create_suggested_swaps_sheet(writer, workbook, df: pd.DataFrame, volume_redistribution_dict: Dict,
                                item_to_idx_map: Dict, feedback_df: pd.DataFrame, swaps_df: pd.DataFrame,
                                formats: Dict, engine_to_use: str, **kwargs) -> None:
    """Create the Suggested Swaps sheet."""
    sheet_name = REPORT_SHEETS['suggested_swaps']
    print(f"\n--- Processing Sheet: {sheet_name} ---")
    
    # Extract column names from kwargs
    suggest_volume_col = kwargs.get('suggest_volume_col', 'suggest_volume')
    item_id_col = kwargs.get('item_id_col', DEFAULT_COLUMNS['item_id'])
    vendor_col = kwargs.get('vendor_col', DEFAULT_COLUMNS['vendor'])
    all_descriptions_col = kwargs.get('all_descriptions_col', DEFAULT_COLUMNS['description'])
    attribute_col = kwargs.get('attribute_col', DEFAULT_COLUMNS['attributes'])
    sales_col = kwargs.get('sales_col', DEFAULT_COLUMNS['sales'])
    cogs_col = kwargs.get('cogs_col', DEFAULT_COLUMNS['cogs'])
    private_label_col = kwargs.get('private_label_col', DEFAULT_COLUMNS['private_label'])
    l3m_adj_vol_col = kwargs.get('l3m_adj_vol_col', DEFAULT_COLUMNS['adj_vol'])
    
    # Define output columns
    swap_output_columns = [
        'Originating SKU ID', 'Originating SKU Vendor', 'Originating SKU Description', 'Originating SKU Attributes',
        'Originating PL Flag', 'Originating L3M Volume', 'Originating SKU L3M Sales', 'Originating SKU L3M COGS', 'Originating SKU GM/Unit',
        'Receiving SKU ID', 'Receiving SKU Vendor', 'Receiving SKU Description', 'Receiving SKU Attributes',
        'Receiving PL Flag', 'Receiving L3M Volume', 'Receiving SKU L3M Sales', 'Receiving SKU L3M COGS', 'Receiving SKU GM/Unit',
        'Reviewed', 'Accept/Reject', 'Feedback'
    ]
    
    if not volume_redistribution_dict:
        print("volume_redistribution_dict is empty. No swaps to list on this sheet.")
        suggest_swaps_df_final = pd.DataFrame(columns=swap_output_columns)
    else:
        suggest_swaps_df_final = _create_swap_details_dataframe(
            df, volume_redistribution_dict, item_to_idx_map, feedback_df, swaps_df,
            swap_output_columns, **kwargs
        )
    
    # Create worksheet and write data
    worksheet = _create_worksheet(writer, workbook, sheet_name, engine_to_use)
    _write_suggested_swaps_data(worksheet, workbook, suggest_swaps_df_final, formats, engine_to_use)

def _create_drop_list_sheet(writer, workbook, df: pd.DataFrame, formats: Dict, 
                          engine_to_use: str, **kwargs) -> None:
    """Create the Drop List sheet."""
    sheet_name = REPORT_SHEETS['drop_list']
    print(f"\n--- Processing Sheet: {sheet_name} ---")
    
    # Extract column names from kwargs
    suggest_volume_col = kwargs.get('suggest_volume_col', 'suggest_volume')
    item_id_col = kwargs.get('item_id_col', DEFAULT_COLUMNS['item_id'])
    vendor_col = kwargs.get('vendor_col', DEFAULT_COLUMNS['vendor'])
    all_descriptions_col = kwargs.get('all_descriptions_col', DEFAULT_COLUMNS['description'])
    attribute_col = kwargs.get('attribute_col', DEFAULT_COLUMNS['attributes'])
    sales_col = kwargs.get('sales_col', DEFAULT_COLUMNS['sales'])
    cogs_col = kwargs.get('cogs_col', DEFAULT_COLUMNS['cogs'])
    private_label_col = kwargs.get('private_label_col', DEFAULT_COLUMNS['private_label'])
    l3m_adj_vol_col = kwargs.get('l3m_adj_vol_col', DEFAULT_COLUMNS['adj_vol'])
    
    # Create drop list data
    drop_list_data = _create_drop_list_data(df, item_id_col, vendor_col, all_descriptions_col,
                                          attribute_col, sales_col, cogs_col, private_label_col, l3m_adj_vol_col)
    
    # Create worksheet and write data
    worksheet = _create_worksheet(writer, workbook, sheet_name, engine_to_use)
    _write_worksheet_data(worksheet, drop_list_data, formats)

def _create_vendor_summary_sheet(writer, workbook, df: pd.DataFrame, formats: Dict,
                               engine_to_use: str, **kwargs) -> None:
    """Create the Vendor Performance Summary sheet."""
    sheet_name = REPORT_SHEETS['vendor_summary']
    print(f"\n--- Processing Sheet: {sheet_name} ---")
    
    # Extract column names from kwargs
    suggest_volume_col = kwargs.get('suggest_volume_col', 'suggest_volume')
    item_id_col = kwargs.get('item_id_col', DEFAULT_COLUMNS['item_id'])
    vendor_col = kwargs.get('vendor_col', DEFAULT_COLUMNS['vendor'])
    sales_col = kwargs.get('sales_col', DEFAULT_COLUMNS['sales'])
    l3m_adj_vol_col = kwargs.get('l3m_adj_vol_col', DEFAULT_COLUMNS['adj_vol'])
    
    # Create vendor summary data
    vendor_summary_data = _create_vendor_summary_data(df, suggest_volume_col, item_id_col,
                                                    vendor_col, sales_col, l3m_adj_vol_col)
    
    # Create worksheet and write data
    worksheet = _create_worksheet(writer, workbook, sheet_name, engine_to_use)
    _write_vendor_summary_data(worksheet, workbook, vendor_summary_data, formats, engine_to_use)

def _create_customer_summary_sheet(writer, workbook, df: pd.DataFrame, true_l3m_grouped: pd.DataFrame,
                                 formats: Dict, engine_to_use: str, redi_customers: Optional[List] = None,
                                 customer_exclusions: Optional[List] = None, **kwargs) -> None:
    """Create the Customer Performance Summary sheet."""
    sheet_name = REPORT_SHEETS['customer_summary']
    print(f"\n--- Processing Sheet: {sheet_name} ---")
    
    # Extract column names from kwargs
    suggest_volume_col = kwargs.get('suggest_volume_col', 'suggest_volume')
    item_id_col = kwargs.get('item_id_col', DEFAULT_COLUMNS['item_id'])
    l3m_adj_vol_col = kwargs.get('l3m_adj_vol_col', DEFAULT_COLUMNS['adj_vol'])
    sales_col = kwargs.get('sales_col', DEFAULT_COLUMNS['sales'])
    
    # Create customer summary data
    customer_summary_data = _create_customer_summary_data(df, true_l3m_grouped, suggest_volume_col,
                                                        item_id_col, l3m_adj_vol_col, sales_col,
                                                        redi_customers, customer_exclusions)
    
    # Create worksheet and write data
    worksheet = _create_worksheet(writer, workbook, sheet_name, engine_to_use)
    _write_customer_summary_data(worksheet, workbook, customer_summary_data, formats, engine_to_use)

def _create_worksheet(writer, workbook, sheet_name: str, engine_to_use: str):
    """Create a worksheet with proper formatting."""
    worksheet = writer.sheets.get(sheet_name)
    if worksheet is None and engine_to_use == 'xlsxwriter':
        worksheet = workbook.add_worksheet(sheet_name)
        worksheet.hide_gridlines(2)
    return worksheet

# =============================================================================
# DATA CREATION HELPER FUNCTIONS
# =============================================================================

def _create_swap_details_dataframe(df: pd.DataFrame, volume_redistribution_dict: Dict,
                                 item_to_idx_map: Dict, feedback_df: pd.DataFrame,
                                 swaps_df: pd.DataFrame, swap_output_columns: List[str], **kwargs) -> pd.DataFrame:
    """Create the swap details DataFrame for the Suggested Swaps sheet."""
    idx_to_item_map = {v: k for k, v in item_to_idx_map.items()}
    swap_details_list = []
    
    # Prepare lookup DataFrame
    if not df[kwargs.get('item_id_col', DEFAULT_COLUMNS['item_id'])].is_unique:
        print(f"Warning: '{kwargs.get('item_id_col', DEFAULT_COLUMNS['item_id'])}' is not unique in the input DataFrame. Using first occurrence for lookups in 'Suggested Swaps'.")
        df_for_swap_lookup = df.drop_duplicates(subset=[kwargs.get('item_id_col', DEFAULT_COLUMNS['item_id'])]).set_index(kwargs.get('item_id_col', DEFAULT_COLUMNS['item_id']))
    else:
        df_for_swap_lookup = df.set_index(kwargs.get('item_id_col', DEFAULT_COLUMNS['item_id']))
    
    # Process feedback data
    feedback_df_processed = feedback_df.copy()
    feedback_df_processed['Targets'] = feedback_df_processed['Targets'].map(str)
    feedback_df_processed['Subs'] = feedback_df_processed['Subs'].map(str)
    
    target_sku_ids_in_feedback = set(feedback_df_processed['Targets'].dropna().unique())
    
    accepted_or_considered_feedback = feedback_df_processed[
        (feedback_df_processed['Correct'] == 'Accept') | (feedback_df_processed['Correct'] == 'Consider')
    ].copy()
    
    accepted_or_considered_feedback_set = set(
        accepted_or_considered_feedback[['Targets', 'Subs']].apply(
            lambda x: (str(x['Targets']), str(x['Subs'])), axis=1
        )
    )
    
    target_to_accepted_subs = defaultdict(set)
    for _, row in accepted_or_considered_feedback.iterrows():
        target_to_accepted_subs[str(row['Targets'])].add(str(row['Subs']))
    
    # Process existing swaps
    swaps_df_processed = swaps_df.copy()
    swaps_df_processed['Originating SKU ID'] = swaps_df_processed['Originating SKU ID'].astype(str)
    swaps_df_processed['Receiving SKU ID'] = swaps_df_processed['Receiving SKU ID'].astype(str)
    existing_swaps_set = set(
        swaps_df_processed[['Originating SKU ID', 'Receiving SKU ID']].apply(
            lambda x: (x['Originating SKU ID'], x['Receiving SKU ID']), axis=1
        )
    )
    
    # Extract column names
    item_id_col = kwargs.get('item_id_col', DEFAULT_COLUMNS['item_id'])
    vendor_col = kwargs.get('vendor_col', DEFAULT_COLUMNS['vendor'])
    all_descriptions_col = kwargs.get('all_descriptions_col', DEFAULT_COLUMNS['description'])
    attribute_col = kwargs.get('attribute_col', DEFAULT_COLUMNS['attributes'])
    sales_col = kwargs.get('sales_col', DEFAULT_COLUMNS['sales'])
    cogs_col = kwargs.get('cogs_col', DEFAULT_COLUMNS['cogs'])
    private_label_col = kwargs.get('private_label_col', DEFAULT_COLUMNS['private_label'])
    l3m_adj_vol_col = kwargs.get('l3m_adj_vol_col', DEFAULT_COLUMNS['adj_vol'])
    
    # Process each swap
    for (from_idx_str, to_idx_str), volume in volume_redistribution_dict.items():
        if pd.isna(volume) or volume <= 1e-5:
            continue
        
        try:
            from_idx_int = int(from_idx_str)
            to_idx_int = int(to_idx_str)
        except ValueError:
            continue
        
        from_entity_id = idx_to_item_map.get(from_idx_int)
        to_entity_id = idx_to_item_map.get(to_idx_int)
        if not from_entity_id or not to_entity_id:
            continue
        
        # Get SKU details
        from_sku_details = df_for_swap_lookup.loc[from_entity_id] if from_entity_id in df_for_swap_lookup.index else pd.Series(dtype='object')
        to_sku_details = df_for_swap_lookup.loc[to_entity_id] if to_entity_id in df_for_swap_lookup.index else pd.Series(dtype='object')
        
        if isinstance(from_sku_details, pd.DataFrame):
            from_sku_details = from_sku_details.iloc[0] if not from_sku_details.empty else pd.Series(dtype='object')
        if isinstance(to_sku_details, pd.DataFrame):
            to_sku_details = to_sku_details.iloc[0] if not to_sku_details.empty else pd.Series(dtype='object')
        
        # Determine if reviewed
        is_reviewed = _determine_review_status(from_entity_id, to_entity_id, target_sku_ids_in_feedback,
                                             target_to_accepted_subs, existing_swaps_set)
        
        # Calculate GM/Unit
        orig_sales = from_sku_details.get(sales_col, 0)
        orig_cogs = from_sku_details.get(cogs_col, 0)
        orig_volume = from_sku_details.get(l3m_adj_vol_col, 0)
        
        recv_sales = to_sku_details.get(sales_col, 0)
        recv_cogs = to_sku_details.get(cogs_col, 0)
        recv_volume = to_sku_details.get(l3m_adj_vol_col, 0)
        
        orig_gm_per_unit = _calculate_gm_per_unit(orig_sales, orig_cogs, orig_volume)
        recv_gm_per_unit = _calculate_gm_per_unit(recv_sales, recv_cogs, recv_volume)
        
        # Create swap details
        swap_details_list.append({
            'Originating SKU ID': from_entity_id,
            'Originating SKU Vendor': from_sku_details.get(vendor_col, 'N/A'),
            'Originating SKU Description': from_sku_details.get(all_descriptions_col, 'N/A'),
            'Originating SKU Attributes': from_sku_details.get(attribute_col, ''),
            'Originating PL Flag': from_sku_details.get(private_label_col, 'N/A'),
            'Originating L3M Volume': orig_volume,
            'Originating SKU L3M Sales': orig_sales,
            'Originating SKU L3M COGS': orig_cogs,
            'Originating SKU GM/Unit': orig_gm_per_unit,
            'Receiving SKU ID': to_entity_id,
            'Receiving SKU Vendor': to_sku_details.get(vendor_col, 'N/A'),
            'Receiving SKU Description': to_sku_details.get(all_descriptions_col, 'N/A'),
            'Receiving SKU Attributes': to_sku_details.get(attribute_col, ''),
            'Receiving PL Flag': to_sku_details.get(private_label_col, 'N/A'),
            'Receiving L3M Volume': recv_volume,
            'Receiving SKU L3M Sales': recv_sales,
            'Receiving SKU L3M COGS': recv_cogs,
            'Receiving SKU GM/Unit': recv_gm_per_unit,
            'Reviewed': is_reviewed,
            'Accept/Reject': "Accept",
            'Feedback': ""
        })
    
    if swap_details_list:
        suggest_swaps_df_final = pd.DataFrame(swap_details_list)
        suggest_swaps_df_final = suggest_swaps_df_final[swap_output_columns]
    else:
        print("No valid swaps to list after processing volume_redistribution_dict.")
        suggest_swaps_df_final = pd.DataFrame(columns=swap_output_columns)
    
    return suggest_swaps_df_final

def _determine_review_status(from_entity_id: str, to_entity_id: str, target_sku_ids_in_feedback: set,
                           target_to_accepted_subs: Dict, existing_swaps_set: set) -> bool:
    """Determine if a swap has been reviewed based on feedback and existing swaps."""
    # Condition 1: If Originating or Receiving ID is in feedback_df['Targets']
    if from_entity_id in target_sku_ids_in_feedback or to_entity_id in target_sku_ids_in_feedback:
        return True
    
    # Condition 2: If mutual 'Accept' or 'Consider' for the same Target
    for target_id, subs_set in target_to_accepted_subs.items():
        if to_entity_id in subs_set and from_entity_id in subs_set:
            return True
    
    # Condition 3: If the (Originating, Receiving) pair exists in swaps_df
    if (from_entity_id, to_entity_id) in existing_swaps_set:
        return True
    
    return False

def _create_vendor_summary_data(df: pd.DataFrame, suggest_volume_col: str, item_id_col: str,
                              vendor_col: str, sales_col: str, l3m_adj_vol_col: str) -> pd.DataFrame:
    """Create vendor summary data for the Vendor Performance Summary sheet."""
    # Group by vendor and calculate metrics
    vendor_summary = df.groupby(vendor_col).agg({
        l3m_adj_vol_col: ['sum', 'count'],
        sales_col: 'sum'
    }).reset_index()
    
    # Flatten column names
    vendor_summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in vendor_summary.columns]
    
    # Calculate additional metrics
    vendor_summary['GM_Percentage'] = vendor_summary.apply(
        lambda row: _calculate_gm_percentage(row[f'{sales_col}_sum'], row.get(f'{sales_col}_sum', 0)), axis=1
    )
    
    # Rename columns for output
    vendor_summary = vendor_summary.rename(columns={
        f'{l3m_adj_vol_col}_sum': 'Total L3M Volume (Units)',
        f'{l3m_adj_vol_col}_count': 'SKU Count',
        f'{sales_col}_sum': 'Total L3M Sales ($)',
        'GM_Percentage': 'GM %'
    })
    
    return vendor_summary.sort_values(by='Total L3M Volume (Units)', ascending=False)

def _create_customer_summary_data(df: pd.DataFrame, true_l3m_grouped: pd.DataFrame, suggest_volume_col: str,
                                item_id_col: str, l3m_adj_vol_col: str, sales_col: str,
                                redi_customers: Optional[List] = None, customer_exclusions: Optional[List] = None) -> pd.DataFrame:
    """Create customer summary data for the Customer Performance Summary sheet."""
    dropped_skus_set = set(df[df[suggest_volume_col] == 0][item_id_col].unique())
    true_l3m_grouped['is_effected_sku'] = true_l3m_grouped[item_id_col].isin(dropped_skus_set)
    
    # Calculate total and affected performance
    customer_total_performance = true_l3m_grouped.groupby(['customer']).agg(
        Total_L3M_Volume=(l3m_adj_vol_col, 'sum'),
        Total_L3M_Sales=(sales_col, 'sum')
    ).reset_index()
    
    effected_customer_performance = true_l3m_grouped[true_l3m_grouped['is_effected_sku']].groupby(['customer']).agg(
        Effected_L3M_Volume=(l3m_adj_vol_col, 'sum'),
        Effected_L3M_Sales=(sales_col, 'sum')
    ).reset_index()
    
    # Merge and calculate percentages
    customer_summary_df = pd.merge(customer_total_performance, effected_customer_performance, on=['customer'], how='left').fillna(0)
    customer_summary_df['% Volume Effected'] = np.where(
        customer_summary_df['Total_L3M_Volume'] > 0,
        customer_summary_df['Effected_L3M_Volume'] / customer_summary_df['Total_L3M_Volume'],
        0
    )
    customer_summary_df['% Sales Effected'] = np.where(
        customer_summary_df['Total_L3M_Sales'] > 0,
        customer_summary_df['Effected_L3M_Sales'] / customer_summary_df['Total_L3M_Sales'],
        0
    )
    
    # Add Redi and Excluded columns
    customer_summary_df['Redi'] = 'No'
    customer_summary_df['Excluded'] = 'No'
    
    if redi_customers:
        customer_summary_df.loc[customer_summary_df['customer'].isin(redi_customers), 'Redi'] = 'Yes'
    
    if customer_exclusions:
        customer_summary_df.loc[customer_summary_df['customer'].isin(customer_exclusions), 'Excluded'] = 'Yes'
    
    # Rename columns for output
    customer_summary_final = customer_summary_df[[
        'customer', 'Redi', 'Excluded', 'Total_L3M_Volume', 'Effected_L3M_Volume',
        '% Volume Effected', 'Total_L3M_Sales', 'Effected_L3M_Sales', '% Sales Effected'
    ]].rename(columns={
        'customer': 'Customer Name',
        'Total_L3M_Volume': 'Total L3M Volume (Units)',
        'Effected_L3M_Volume': 'Effected L3M Volume (Units)',
        'Total_L3M_Sales': 'Total L3M Sales ($)',
        'Effected_L3M_Sales': 'Effected L3M Sales ($)'
    })
    
    return customer_summary_final.sort_values(by='Effected L3M Volume (Units)', ascending=False)

# =============================================================================
# EXCEL WRITING HELPER FUNCTIONS
# =============================================================================

def _write_suggested_swaps_data(worksheet, workbook, df: pd.DataFrame, formats: Dict, engine_to_use: str) -> None:
    """Write suggested swaps data to worksheet with proper formatting."""
    if engine_to_use == 'xlsxwriter':
        # Add dark blue placeholder row
        dark_blue_format = workbook.add_format({'bold': True, 'fg_color': '#1f497d', 'font_color': 'white', 'align': 'center'})
        worksheet.write(0, 0, "Hi", dark_blue_format)
        worksheet.merge_range(0, 0, 0, len(df.columns), "Hi", dark_blue_format)
        
        # Set column A to be narrow
        worksheet.set_column(0, 0, 5)
        
        # Write headers starting at B3
        for col_num, value in enumerate(df.columns.values):
            header_format = _get_suggested_swaps_header_format(formats, col_num, len(df.columns))
            worksheet.write(2, col_num + 1, value, header_format)
        
        # Write data starting at B4
        if not df.empty:
            for df_row_idx, data_tuple in enumerate(df.itertuples(index=False)):
                excel_row_num = df_row_idx + 3
                for col_idx, col_name in enumerate(df.columns):
                    cell_value = data_tuple[col_idx]
                    cell_format = _get_suggested_swaps_cell_format(formats, col_name, col_idx, len(df.columns))
                    
                    if pd.notna(cell_value):
                        worksheet.write(excel_row_num, col_idx + 1, cell_value, cell_format)
                    else:
                        worksheet.write_blank(excel_row_num, col_idx + 1, None, cell_format)

def _write_vendor_summary_data(worksheet, workbook, df: pd.DataFrame, formats: Dict, engine_to_use: str) -> None:
    """Write vendor summary data to worksheet with proper formatting."""
    if engine_to_use == 'xlsxwriter':
        # Add dark blue placeholder row
        dark_blue_format = workbook.add_format({'bold': False, 'fg_color': '#1f497d', 'font_color': 'white', 'align': 'left'})
        worksheet.write(0, 0, "Vendor Summary - Note any Redistributor or Non-Addressable Customer (like WFM), are uneffected", dark_blue_format)
        worksheet.merge_range(0, 0, 0, len(df.columns), "Vendor Summary - Note any Redistributor or Non-Addressable Customer (like WFM), are uneffected", dark_blue_format)
        
        # Set column A to be narrow
        worksheet.set_column(0, 0, 5)
        
        # Write headers starting at B3
        for col_num, value in enumerate(df.columns.values):
            header_format = _get_vendor_header_format(formats, value)
            worksheet.write(2, col_num + 1, value, header_format)
        
        # Write data starting at B4
        if not df.empty:
            for df_row_idx, data_tuple in enumerate(df.itertuples(index=False)):
                excel_row_num = df_row_idx + 3
                for col_idx, col_name in enumerate(df.columns):
                    cell_value = data_tuple[col_idx]
                    cell_format = _get_vendor_cell_format(formats, col_name)
                    
                    if pd.notna(cell_value):
                        worksheet.write(excel_row_num, col_idx + 1, cell_value, cell_format)
                    else:
                        worksheet.write_blank(excel_row_num, col_idx + 1, None, cell_format)
        
        # Set column widths and freeze panes
        for i, col_name in enumerate(df.columns):
            worksheet.set_column(i + 1, i + 1, 15)
        worksheet.freeze_panes(3, 0)

def _write_customer_summary_data(worksheet, workbook, df: pd.DataFrame, formats: Dict, engine_to_use: str) -> None:
    """Write customer summary data to worksheet with proper formatting."""
    if engine_to_use == 'xlsxwriter':
        # Add dark blue placeholder row
        dark_blue_format = workbook.add_format({'bold': False, 'fg_color': '#1f497d', 'font_color': 'white', 'align': 'left'})
        worksheet.write(0, 0, "Customer Summary - Note any Redistributor or Non-Addressable Customer (like WFM), are uneffected", dark_blue_format)
        worksheet.merge_range(0, 0, 0, len(df.columns), "Customer Summary - Note any Redistributor or Non-Addressable Customer (like WFM), are uneffected", dark_blue_format)
        
        # Set column A to be narrow
        worksheet.set_column(0, 0, 5)
        
        # Write headers starting at B3
        for col_num, value in enumerate(df.columns.values):
            header_format = _get_customer_header_format(formats, value)
            worksheet.write(2, col_num + 1, value, header_format)
        
        # Write data starting at B4
        if not df.empty:
            for df_row_idx, data_tuple in enumerate(df.itertuples(index=False)):
                excel_row_num = df_row_idx + 3
                for col_idx, col_name in enumerate(df.columns):
                    cell_value = data_tuple[col_idx]
                    cell_format = _get_customer_cell_format(formats, col_name)
                    
                    if pd.notna(cell_value):
                        worksheet.write(excel_row_num, col_idx + 1, cell_value, cell_format)
                    else:
                        worksheet.write_blank(excel_row_num, col_idx + 1, None, cell_format)
        
        # Set column widths and freeze panes
        for i, col_name in enumerate(df.columns):
            worksheet.set_column(i + 1, i + 1, 15)
        worksheet.freeze_panes(3, 0)

def _get_suggested_swaps_header_format(formats: Dict, col_idx: int, total_cols: int):
    """Get appropriate header format for suggested swaps sheet."""
    orig_cols_end_idx = 9  # Index of 'Originating SKU GM/Unit'
    recv_cols_start_idx = 9  # Index of 'Receiving SKU ID'
    recv_cols_end_idx = 18  # Index of 'Receiving SKU GM/Unit'
    reviewed_col_idx = 18  # Index of 'Reviewed'
    
    if col_idx < orig_cols_end_idx:
        return formats['orig_header']
    elif recv_cols_start_idx <= col_idx < recv_cols_end_idx:
        return formats['recv_header']
    elif col_idx >= reviewed_col_idx:
        return formats['reviewed_header']
    else:
        return formats['general_header']

def _get_suggested_swaps_cell_format(formats: Dict, col_name: str, col_idx: int, total_cols: int):
    """Get appropriate cell format for suggested swaps sheet."""
    orig_cols_end_idx = 9
    recv_cols_start_idx = 9
    recv_cols_end_idx = 18
    
    if col_idx < orig_cols_end_idx:
        base_format = formats['general_cell']
    elif recv_cols_start_idx <= col_idx < recv_cols_end_idx:
        base_format = formats['general_cell']
    else:
        base_format = formats['general_cell']
    
    # Apply number formatting
    if 'Volume' in col_name:
        return formats['general_number']
    elif 'Sales' in col_name and '$' in col_name:
        return formats['dollar_number']
    elif 'GM/Unit' in col_name:
        return formats['gm_unit']
    elif col_name == 'Accept/Reject':
        return formats['accept_reject_cell']
    else:
        return base_format

def _get_vendor_header_format(formats: Dict, col_name: str):
    """Get appropriate header format for vendor summary sheet."""
    if 'Volume' in col_name:
        return formats['vendor_header_volume']
    elif 'Sales' in col_name:
        return formats['vendor_header_sales']
    else:
        return formats['vendor_header_general']

def _get_vendor_cell_format(formats: Dict, col_name: str):
    """Get appropriate cell format for vendor summary sheet."""
    if 'Sales' in col_name and '$' in col_name:
        return formats['dollar_number']
    elif 'Sales' in col_name and '%' in col_name:
        return formats['percentage']
    elif '%' in col_name:
        return formats['percentage']
    elif 'Volume' in col_name:
        return formats['general_number']
    else:
        return formats['general_cell']

def _get_customer_header_format(formats: Dict, col_name: str):
    """Get appropriate header format for customer summary sheet."""
    if 'Volume' in col_name:
        return formats['customer_header_volume']
    elif 'Sales' in col_name:
        return formats['customer_header_sales']
    else:
        return formats['customer_header_general']

def _get_customer_cell_format(formats: Dict, col_name: str):
    """Get appropriate cell format for customer summary sheet."""
    if 'Sales' in col_name and '$' in col_name:
        return formats['dollar_number']
    elif 'Sales' in col_name and '%' in col_name:
        return formats['percentage']
    elif '%' in col_name:
        return formats['percentage']
    elif 'Volume' in col_name:
        return formats['general_number']
    else:
        return formats['general_cell']

# =============================================================================
# REPORTING AND FEEDBACK FUNCTIONS
# =============================================================================

def create_enhanced_sku_report_post_feedback(
    updated_df: pd.DataFrame,
    output_excel_filepath: str,
    volume_redistribution_dict: Dict,
    item_to_idx_map: Dict[str, int],
    feedback_df: pd.DataFrame,
    true_l3m_grouped: pd.DataFrame,
    swaps_df: pd.DataFrame,
    suggest_volume_col: str = 'suggest_volume',
    volume_change_col: str = 'Volume_Change',
    item_id_col: str = DEFAULT_COLUMNS['item_id'],
    vendor_col: str = DEFAULT_COLUMNS['vendor'],
    all_descriptions_col: str = DEFAULT_COLUMNS['description'],
    attribute_col: str = DEFAULT_COLUMNS['attributes'],
    matches_col: str = DEFAULT_COLUMNS['matches'],
    sales_col: str = DEFAULT_COLUMNS['sales'],
    cogs_col: str = DEFAULT_COLUMNS['cogs'],
    private_label_col: str = DEFAULT_COLUMNS['private_label'],
    l3m_adj_vol_col: str = DEFAULT_COLUMNS['adj_vol'],
    redi_customers: Optional[List] = None,
    customer_exclusions: Optional[List] = None
) -> bool:
    """
    Creates an Excel file with "Suggested Swaps", "Drop List", "Vendor Performance Summary",
    and "Customer Performance Summary" sheets.
    
    This function generates a comprehensive report with optimization results, including:
    - Suggested swaps with originating and receiving SKU details
    - Drop list of items with zero suggested volume
    - Vendor performance summary with volume and sales metrics
    - Customer performance summary showing impact of changes
    
    Args:
        updated_df (pd.DataFrame): Input DataFrame with item details and optimization results.
        output_excel_filepath (str): Full path for the output Excel file.
        volume_redistribution_dict (dict): Dictionary from optimization.
        item_to_idx_map (dict): Mapping from string 'Entity--Item' ID to its numerical index.
        feedback_df (pd.DataFrame): DataFrame with columns 'Targets', 'Subs', 'Correct' for review feedback.
        true_l3m_grouped (pd.DataFrame): DataFrame with customer, sku, volume, and sales for L3M.
        swaps_df (pd.DataFrame): DataFrame of existing swaps.
        suggest_volume_col (str): Name of the column indicating suggested final volume.
        volume_change_col (str): Name of the column indicating volume change from original.
        item_id_col (str): Name of the Entity ID column in updated_df.
        vendor_col (str): Name of the vendor name column in updated_df.
        all_descriptions_col (str): Name of the column containing all descriptions.
        attribute_col (str): Name of the column for "Attributes".
        matches_col (str): Name of the column in updated_df with list of matches.
        sales_col (str): Name of the L3M Sales column.
        cogs_col (str): Name of the L3M COGS column.
        private_label_col (str): Name of the private label flag column.
        l3m_adj_vol_col (str): Name of the L3M Adjusted Volume column.
        redi_customers (list, optional): List of redistributor customer names.
        customer_exclusions (list, optional): List of customer names to exclude.

    Returns:
        bool: True if the Excel file was created successfully, False otherwise.
    """
    print(f"--- Starting to generate Enhanced SKU Report: {output_excel_filepath} ---")

    # Validate inputs using helper function
    try:
        _validate_report_inputs(updated_df, volume_redistribution_dict, item_to_idx_map, feedback_df,
                              true_l3m_grouped, swaps_df, suggest_volume_col, volume_change_col,
                              item_id_col, vendor_col, all_descriptions_col, attribute_col, matches_col,
                              sales_col, cogs_col, private_label_col, l3m_adj_vol_col)
    except ValidationError as e:
        print(f"Validation Error: {e}")
        return False

    # Work on a copy and prepare data
    df = updated_df.copy()
    df = _prepare_data_for_processing(df, suggest_volume_col, volume_change_col, sales_col, cogs_col, l3m_adj_vol_col)
    
    # Add calculated fields
    df['Has_No_Matches'] = df[matches_col].apply(_is_empty_match_list)
    
    # Identify protected SKUs
    protected_skus = _identify_protected_skus(true_l3m_grouped, item_id_col, redi_customers, customer_exclusions)
    df['Serves_Protected_Customers'] = df[item_id_col].isin(protected_skus)

    try:
        # Determine Excel engine
        engine_to_use = _determine_excel_engine(output_excel_filepath)
        
        with pd.ExcelWriter(output_excel_filepath, engine=engine_to_use) as writer:
            workbook = writer.book
            
            # Create Excel formats
            formats = _create_excel_formats(workbook)
            
            # Create all sheets using helper functions
            kwargs = {
                'suggest_volume_col': suggest_volume_col,
                'volume_change_col': volume_change_col,
                'item_id_col': item_id_col,
                'vendor_col': vendor_col,
                'all_descriptions_col': all_descriptions_col,
                'attribute_col': attribute_col,
                'matches_col': matches_col,
                'sales_col': sales_col,
                'cogs_col': cogs_col,
                'private_label_col': private_label_col,
                'l3m_adj_vol_col': l3m_adj_vol_col
            }
            
            # Create Suggested Swaps sheet
            _create_suggested_swaps_sheet(writer, workbook, df, volume_redistribution_dict,
                                        item_to_idx_map, feedback_df, swaps_df, formats, engine_to_use, **kwargs)
            
            # Create Drop List sheet
            _create_drop_list_sheet(writer, workbook, df, formats, engine_to_use, **kwargs)
            
            # Create Vendor Performance Summary sheet
            _create_vendor_summary_sheet(writer, workbook, df, formats, engine_to_use, **kwargs)
            
            # Create Customer Performance Summary sheet
            _create_customer_summary_sheet(writer, workbook, df, true_l3m_grouped, formats, engine_to_use,
                                         redi_customers, customer_exclusions, **kwargs)

        print(f"\nSuccessfully created Excel report: {output_excel_filepath}")
        return True

    except Exception as e:
        print(f"An error occurred while creating the Excel file: {e}")
        import traceback
        traceback.print_exc()
        return False
        
def read_and_process_feedback_excel(
    excel_filepath: str,
    drop_list_sheet_name: str = REPORT_SHEETS['drop_list'],
    suggest_swaps_sheet_name: str = REPORT_SHEETS['suggested_swaps'],
    drop_list_item_col: str = DEFAULT_COLUMNS['item_id'],
    suggest_swaps_orig_sku_col: str = 'Originating_SKU_ID',
    suggest_swaps_recv_sku_col: str = 'Receiving_SKU_ID',
    accept_reject_col: str = 'Accept/Reject',
    feedback_col: str = 'Feedback'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads the "Drop List" and "Suggested Swaps" sheets from a specified Excel file
    and returns two DataFrames with selected columns.

    Args:
        excel_filepath (str): The full path to the Excel file.
        drop_list_sheet_name (str): Name of the sheet containing the drop list.
        suggest_swaps_sheet_name (str): Name of the sheet containing suggested swaps.
        drop_list_item_col (str): Name of the SKU ID column in the "Drop List" sheet.
        suggest_swaps_orig_sku_col (str): Name of the originating SKU ID column in "Suggested Swaps".
        suggest_swaps_recv_sku_col (str): Name of the receiving SKU ID column in "Suggested Swaps".
        accept_reject_col (str): Name of the column containing "Accept/Reject" status.
        feedback_col (str): Name of the column containing feedback.

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame)
            - drop_list_df: DataFrame with [drop_list_item_col, accept_reject_col, feedback_col]
                            from the "Drop List" sheet.
            - suggested_swaps_df: DataFrame with [suggest_swaps_orig_sku_col, 
                                   suggest_swaps_recv_sku_col, accept_reject_col, feedback_col]
                                   from the "Suggested Swaps" sheet.
            Returns (None, None) if the file doesn't exist or sheets are not found.
    """
    print(f"--- Reading feedback from Excel file: {excel_filepath} ---")

    if not os.path.exists(excel_filepath):
        print(f"Error: Excel file not found at {excel_filepath}")
        return None, None

    try:
        xls = pd.ExcelFile(excel_filepath)
    except Exception as e:
        print(f"Error opening Excel file: {e}")
        return None, None

    drop_list_df = pd.DataFrame()
    suggested_swaps_df = pd.DataFrame()

    # --- Process "Drop List" sheet ---
    if drop_list_sheet_name in xls.sheet_names:
        print(f"Processing sheet: {drop_list_sheet_name}...")
        try:
            # Read Excel starting from B3 (skip first 2 rows and first column)
            df_drop = pd.read_excel(xls, sheet_name=drop_list_sheet_name, skiprows=2, usecols=lambda x: x != 'A')
            required_drop_cols = [drop_list_item_col, accept_reject_col, feedback_col]
            
            # Check if all required columns exist
            missing_drop_cols = [col for col in required_drop_cols if col not in df_drop.columns]
            if missing_drop_cols:
                print(f"Warning: Sheet '{drop_list_sheet_name}' is missing columns: {', '.join(missing_drop_cols)}. Skipping this sheet for detailed extraction.")
                drop_list_df = pd.DataFrame(columns=required_drop_cols) # Return empty df with expected columns
            else:
                drop_list_df = df_drop[required_drop_cols].copy()
                print(f"Successfully extracted data from '{drop_list_sheet_name}'. Found {len(drop_list_df)} rows.")
                
        except Exception as e:
            print(f"Error processing sheet '{drop_list_sheet_name}': {e}")
            drop_list_df = pd.DataFrame(columns=[drop_list_item_col, accept_reject_col, feedback_col]) # Ensure it's an empty DF with correct columns on error
    else:
        print(f"Warning: Sheet '{drop_list_sheet_name}' not found in the Excel file.")
        drop_list_df = pd.DataFrame(columns=[drop_list_item_col, accept_reject_col, feedback_col])


    # --- Process "Suggested Swaps" sheet ---
    if suggest_swaps_sheet_name in xls.sheet_names:
        print(f"\nProcessing sheet: {suggest_swaps_sheet_name}...")
        try:
            # Read Excel starting from B3 (skip first 2 rows and first column)
            df_swaps = pd.read_excel(xls, sheet_name=suggest_swaps_sheet_name, skiprows=2, usecols=lambda x: x != 'A')
            required_swaps_cols = [
                suggest_swaps_orig_sku_col, 
                suggest_swaps_recv_sku_col, 
                accept_reject_col, 
                feedback_col
            ]
            
            missing_swaps_cols = [col for col in required_swaps_cols if col not in df_swaps.columns]
            if missing_swaps_cols:
                print(f"Warning: Sheet '{suggest_swaps_sheet_name}' is missing columns: {', '.join(missing_swaps_cols)}. Skipping this sheet for detailed extraction.")
                suggested_swaps_df = pd.DataFrame(columns=required_swaps_cols)
            else:
                suggested_swaps_df = df_swaps[required_swaps_cols].copy()
                print(f"Successfully extracted data from '{suggest_swaps_sheet_name}'. Found {len(suggested_swaps_df)} rows.")

        except Exception as e:
            print(f"Error processing sheet '{suggest_swaps_sheet_name}': {e}")
            suggested_swaps_df = pd.DataFrame(columns=[suggest_swaps_orig_sku_col, suggest_swaps_recv_sku_col, accept_reject_col, feedback_col])
    else:
        print(f"Warning: Sheet '{suggest_swaps_sheet_name}' not found in the Excel file.")
        suggested_swaps_df = pd.DataFrame(columns=[suggest_swaps_orig_sku_col, suggest_swaps_recv_sku_col, accept_reject_col, feedback_col])

    return drop_list_df, suggested_swaps_df

def incorporate_swaps_feedback(
    im_final: pd.DataFrame,
    swap: pd.DataFrame,
    entity_col: str = DEFAULT_COLUMNS['item_id'],
    matches_col: str = DEFAULT_COLUMNS['matches'],
    origin_col: str = 'Originating SKU ID',
    receiving_col: str = 'Receiving SKU ID',
    reject_col: str = 'Accept/Reject'
) -> pd.DataFrame:
    """
    Updates a DataFrame by removing rejected swaps from the 'Matches' column based on feedback.
    This process is bidirectional, meaning if an A-to-B swap is rejected, the B-to-A swap
    is also removed from the corresponding 'Matches' list.

    Args:
        im_final (pd.DataFrame): The primary DataFrame containing the 'Matches' column to be cleaned.
        swap (pd.DataFrame): DataFrame containing the feedback with one row per swap suggestion.
        entity_col (str): The name of the main item identifier column in 'im_final'.
        matches_col (str): The name of the column in 'im_final' that holds the list of potential swap items.
        origin_col (str): The column name for the originating SKU in the 'swap' DataFrame.
        receiving_col (str): The column name for the receiving SKU in the 'swap' DataFrame.
        reject_col (str): The column name in 'swap' that indicates if a swap is accepted or rejected.

    Returns:
        pd.DataFrame: The 'im_final' DataFrame with the 'Matches' lists updated to exclude rejected pairs.
    """

    # Copy data to avoid modifying originals
    im_final = im_final.copy()
    swap = swap.copy()

    # Ensure IDs are all strings and trimmed
    for col in [entity_col, origin_col, receiving_col]:
        im_final[entity_col] = im_final[entity_col].astype(str).str.strip()
        swap[origin_col] = swap[origin_col].astype(str).str.strip()
        swap[receiving_col] = swap[receiving_col].astype(str).str.strip()

    # Fix Matches: ensure it's a list (convert from string if needed)
    def ensure_list(val: Any) -> List[str]:
        if isinstance(val, list):
            return [str(x).strip() for x in val]
        if pd.isna(val):
            return []
        try:
            # Attempt to parse from string
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except:
            pass
        return []
    
    im_final[matches_col] = im_final[matches_col].apply(ensure_list)

    # Get all rejected pairs
    rejected_swaps = swap[swap[reject_col] != "Accept"][[origin_col, receiving_col]]
    rejected_pairs = set()
    for _, row in rejected_swaps.iterrows():
        a, b = row[origin_col], row[receiving_col]
        rejected_pairs.add((a, b))
        rejected_pairs.add((b, a))

    # Function to filter matches
    def clean_matches(row: pd.Series) -> List[str]:
        sku = row[entity_col]
        matches = row[matches_col]
        return [m for m in matches if (sku, m) not in rejected_pairs]

    im_final[matches_col] = im_final.apply(clean_matches, axis=1)

    return im_final

def create_enhanced_sku_report(
    updated_df: pd.DataFrame,
    output_excel_filepath: str,
    volume_redistribution_dict: Dict,
    item_to_idx_map: Dict[str, int],
    feedback_df: pd.DataFrame,
    true_l3m_grouped: pd.DataFrame,
    suggest_volume_col: str = 'suggest_volume',
    volume_change_col: str = 'Volume_Change',
    item_id_col: str = DEFAULT_COLUMNS['item_id'],
    vendor_col: str = DEFAULT_COLUMNS['vendor'],
    all_descriptions_col: str = DEFAULT_COLUMNS['description'],
    attribute_col: str = DEFAULT_COLUMNS['attributes'],
    matches_col: str = DEFAULT_COLUMNS['matches'],
    sales_col: str = DEFAULT_COLUMNS['sales'],
    cogs_col: str = DEFAULT_COLUMNS['cogs'],
    private_label_col: str = DEFAULT_COLUMNS['private_label'],
    l3m_adj_vol_col: str = DEFAULT_COLUMNS['adj_vol'],
    redi_customers: Optional[pd.DataFrame] = None,
    customer_exclusions: Optional[pd.DataFrame] = None
) -> None:
    """
    Creates an Excel file with "Suggested Swaps", "Drop List", "Vendor Performance Summary",
    and "Customer Performance Summary" sheets. This version removes the 'Reviewed' column
    and all related logic from the "Suggested Swaps" sheet.
    
    Args:
        updated_df (pd.DataFrame): Input DataFrame with item details and optimization results.
        output_excel_filepath (str): Full path for the output Excel file.
        volume_redistribution_dict (dict): Dictionary from optimization.
        item_to_idx_map (dict): Mapping from string 'Entity--Item' ID to its numerical index.
        feedback_df (pd.DataFrame): DataFrame with columns 'Targets', 'Subs', 'Correct' for review feedback.
        true_l3m_grouped (pd.DataFrame): DataFrame with customer, sku, volume, and sales for L3M.
        suggest_volume_col (str): Name of the column indicating suggested final volume.
        volume_change_col (str): Name of the column indicating volume change from original.
        item_id_col (str): Name of the Entity ID column in updated_df.
        vendor_col (str): Name of the vendor name column in updated_df.
        all_descriptions_col (str): Name of the column containing all descriptions.
        attribute_col (str): Name of the column for "Attributes".
        matches_col (str): Name of the column in updated_df with list of matches.
        sales_col (str): Name of the L3M Sales column.
        cogs_col (str): Name of the L3M COGS column.
        private_label_col (str): Name of the private label flag column.
        l3m_adj_vol_col (str): Name of the L3M Adjusted Volume column.
        redi_customers (list, optional): List of redistributor customer names.
        customer_exclusions (list, optional): List of customer names to exclude.
    
    Returns:
        bool: True if the Excel file was created successfully, False otherwise.
    """
    print(f"--- Starting to generate Enhanced SKU Report: {output_excel_filepath} ---")

    # --- Input Validation ---
    if not isinstance(updated_df, pd.DataFrame):
        print("Error: Input 'updated_df' must be a Pandas DataFrame.")
        return False
    required_cols_input = [suggest_volume_col, volume_change_col, item_id_col,
                           vendor_col, all_descriptions_col, attribute_col,
                           matches_col, sales_col, cogs_col, private_label_col, l3m_adj_vol_col]
    missing_cols = [col for col in required_cols_input if col not in updated_df.columns]
    if missing_cols:
        print(f"Warning: Input DataFrame 'updated_df' is missing expected columns: {', '.join(missing_cols)}. Calculations involving these columns might be affected.")
        for col in missing_cols:
            updated_df[col] = np.nan

    if not isinstance(volume_redistribution_dict, dict):
        print("Error: 'volume_redistribution_dict' must be a dictionary.")
        return False
    if not isinstance(item_to_idx_map, dict):
        print("Error: 'item_to_idx_map' must be a dictionary.")
        return False
    if not isinstance(feedback_df, pd.DataFrame):
        print("Error: 'feedback_df' must be a Pandas DataFrame.")
        return False
    required_feedback_cols = ['Targets', 'Subs', 'Correct']
    if not all(col in feedback_df.columns for col in required_feedback_cols):
        print(f"Error: 'feedback_df' is missing required columns: {', '.join(required_feedback_cols)}.")
        return False
    if not isinstance(true_l3m_grouped, pd.DataFrame):
        print("Error: 'true_l3m_grouped' must be a Pandas DataFrame.")
        return False
    required_l3m_grouped_cols = ['customer', item_id_col, l3m_adj_vol_col, sales_col]
    if not all(col in true_l3m_grouped.columns for col in required_l3m_grouped_cols):
        print(f"Error: 'true_l3m_grouped' is missing required columns: {', '.join(required_l3m_grouped_cols)}.")
        return False

    df = updated_df.copy()

    try:
        engine_to_use = 'xlsxwriter'
        try:
            with pd.ExcelWriter(output_excel_filepath, engine='xlsxwriter') as test_writer:
                pass
        except ImportError:
            print("xlsxwriter not found, falling back to openpyxl. Formatting will be basic.")
            engine_to_use = 'openpyxl'
        except Exception as e_init:
            print(f"Initial xlsxwriter test failed: {e_init}. Falling back to openpyxl.")
            engine_to_use = 'openpyxl'

        with pd.ExcelWriter(output_excel_filepath, engine=engine_to_use) as writer:
            workbook = writer.book

            # --- Define Dictionaries of Format Properties ---
            general_header_props = {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D3D3D3', 'border': 1, 'align': 'center'}
            general_cell_props = {'text_wrap': True, 'valign': 'top', 'border': 1}
            accept_reject_cell_props = {'text_wrap': True, 'valign': 'top', 'border': 1, 'align': 'center'}
            general_number_props = {'num_format': '#,##0.00', 'text_wrap': True, 'valign': 'top', 'border': 1}
            percentage_props = {'num_format': '0.00%', 'text_wrap': True, 'valign': 'top', 'border': 1}
            dollar_number_props = {'num_format': '$#,##0.00', 'text_wrap': True, 'valign': 'top', 'border': 1}
            gm_unit_props = {'num_format': '#,##0.00', 'text_wrap': True, 'valign': 'top', 'border': 1}
            orig_header_props = {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#DAEEF3', 'border': 1, 'align': 'center'}
            recv_header_props = {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#E2EFDA', 'border': 1, 'align': 'center'}
            reviewed_header_props = {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#F2F2F2', 'border': 1, 'align': 'center'}
            vendor_header_general_props = {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#E0EBF5', 'border': 1, 'align': 'center'}
            vendor_header_volume_props = {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#CCE5FF', 'border': 1, 'align': 'center'}
            vendor_header_sales_props = {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#FFE5CC', 'border': 1, 'align': 'center'}
            vendor_cell_props = {'text_wrap': True, 'valign': 'top', 'border': 1}
            vendor_number_props = {'num_format': '#,##0.00', 'text_wrap': True, 'valign': 'top', 'border': 1}
            vendor_percentage_props = {'num_format': '0.00%', 'text_wrap': True, 'valign': 'top', 'border': 1}
            vendor_dollar_props = {'num_format': '$#,##0.00', 'text_wrap': True, 'valign': 'top', 'border': 1}
            customer_header_general_props = {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#E0EBF5', 'border': 1, 'align': 'center'}
            customer_header_volume_props = {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D9EAD3', 'border': 1, 'align': 'center'}
            customer_header_sales_props = {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#FCE5CD', 'border': 1, 'align': 'center'}

            # --- Create Format Objects from Dictionaries ---
            header_format = workbook.add_format(general_header_props)
            general_cell_format = workbook.add_format(general_cell_props)
            accept_reject_cell_format = workbook.add_format(accept_reject_cell_props)
            general_number_format = workbook.add_format(general_number_props)
            percentage_format = workbook.add_format(percentage_props)
            dollar_number_format = workbook.add_format(dollar_number_props)
            gm_unit_format = workbook.add_format(gm_unit_props)
            orig_header_fmt = workbook.add_format(orig_header_props)
            recv_header_fmt = workbook.add_format(recv_header_props)
            orig_cell_fmt_no_bg = workbook.add_format(general_cell_props)
            recv_cell_fmt_no_bg = workbook.add_format(general_cell_props)
            reviewed_header_fmt = workbook.add_format(reviewed_header_props)
            reviewed_cell_fmt_no_bg = workbook.add_format(accept_reject_cell_props)

            DATA_VALIDATION_OPTIONS_ACCEPT_REJECT = ["Accept", "Reject"]

            # --- Prepare Data for "Drop List" ---
            df[suggest_volume_col] = pd.to_numeric(df[suggest_volume_col], errors='coerce').fillna(0)
            df[volume_change_col] = pd.to_numeric(df[volume_change_col], errors='coerce').fillna(0)
            df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce').fillna(0)
            df[cogs_col] = pd.to_numeric(df[cogs_col], errors='coerce').fillna(0)
            df[l3m_adj_vol_col] = pd.to_numeric(df[l3m_adj_vol_col], errors='coerce').fillna(0)
            dropped_swapped_filter = (df[suggest_volume_col] == 0) & (df[volume_change_col] < 0)
            df['GM_Percentage'] = np.where(df[sales_col] > 0, (df[sales_col] - df[cogs_col]) / df[sales_col], -np.inf)
            gm_threshold = df[df[sales_col]>0]['GM_Percentage'].quantile(0.05)
            def is_empty_match_list(match_val):
                if isinstance(match_val, list): return not bool(match_val)
                return True
            df['Has_No_Matches'] = df[matches_col].apply(is_empty_match_list)
            
            # Identify SKUs that serve redistributor or excluded customers (protected from dropping)
            protected_skus = set()
            
            # Get SKUs that serve redistributor customers
            if redi_customers is not None and len(redi_customers) > 0:
                redi_skus = true_l3m_grouped[true_l3m_grouped['customer'].isin(redi_customers)][item_id_col].unique()
                protected_skus.update(redi_skus)
            
            # Get SKUs that serve excluded customers
            if customer_exclusions is not None and len(customer_exclusions) > 0:
                excluded_skus = true_l3m_grouped[true_l3m_grouped['customer'].isin(customer_exclusions)][item_id_col].unique()
                protected_skus.update(excluded_skus)
            
            # Create protection filter - SKUs that serve protected customers should not be dropped for low GM
            df['Serves_Protected_Customers'] = df[item_id_col].isin(protected_skus)
            
            # Modified low GM drop filter - exclude SKUs that serve protected customers
            dropped_low_gm_no_swaps_filter = (df['GM_Percentage'] < gm_threshold) & df['Has_No_Matches'] & ~df['Serves_Protected_Customers']
            
            final_drop_filter_for_list = dropped_swapped_filter | dropped_low_gm_no_swaps_filter
            final_dropped_skus_df_for_list = df[final_drop_filter_for_list].copy()
            final_dropped_skus_df_for_list['Drop_Reason'] = 'Unknown'
            swapped_away_indices = df[dropped_swapped_filter].index
            low_gm_indices = df[dropped_low_gm_no_swaps_filter].index
            final_dropped_skus_df_for_list.loc[final_dropped_skus_df_for_list.index.isin(swapped_away_indices), 'Drop_Reason'] = 'Volume Swapped Away'
            final_dropped_skus_df_for_list.loc[final_dropped_skus_df_for_list.index.isin(low_gm_indices), 'Drop_Reason'] = 'Bottom 5% GM & No Swaps'
            both_criteria_indices = df[dropped_swapped_filter & dropped_low_gm_no_swaps_filter].index
            final_dropped_skus_df_for_list.loc[final_dropped_skus_df_for_list.index.isin(both_criteria_indices), 'Drop_Reason'] = 'Volume Swapped & Low GM/No Swaps'

            # --- Sheet 1: Suggested Swaps ---
            sheet_name_swaps = "Suggested Swaps"
            print(f"\n--- Processing Sheet: {sheet_name_swaps} ---")
            swap_output_columns = [
                'Originating SKU ID', 'Originating SKU Vendor', 'Originating SKU Description', 'Originating SKU Attributes',
                'Originating PL Flag', 'Originating L3M Volume', 'Originating SKU L3M Sales', 'Originating SKU L3M COGS', 'Originating SKU GM/Unit',
                'Receiving SKU ID', 'Receiving SKU Vendor', 'Receiving SKU Description', 'Receiving SKU Attributes',
                'Receiving PL Flag', 'Receiving L3M Volume', 'Receiving SKU L3M Sales', 'Receiving SKU L3M COGS', 'Receiving SKU GM/Unit',
                'Same VPN Code',
                'Reviewed',
                'Accept/Reject',
                'Feedback'
            ]
            if not volume_redistribution_dict:
                print("volume_redistribution_dict is empty. No swaps to list on this sheet.")
                suggest_swaps_df_final = pd.DataFrame(columns=swap_output_columns)
            else:
                idx_to_item_map = {v: k for k, v in item_to_idx_map.items()}
                swap_details_list = []
                if not df[item_id_col].is_unique:
                    print(f"Warning: '{item_id_col}' is not unique. Using first occurrence for lookups.")
                    df_for_swap_lookup = df.drop_duplicates(subset=[item_id_col]).set_index(item_id_col)
                else:
                    df_for_swap_lookup = df.set_index(item_id_col)

                # Prepare feedback-based review signals
                feedback_df_processed = feedback_df.copy()
                feedback_df_processed['Targets'] = feedback_df_processed['Targets'].map(str)
                feedback_df_processed['Subs'] = feedback_df_processed['Subs'].map(str)

                target_sku_ids_in_feedback = set(feedback_df_processed['Targets'].dropna().unique())
                accepted_or_considered_feedback = feedback_df_processed[
                    (feedback_df_processed['Correct'] == 'Accept') | (feedback_df_processed['Correct'] == 'Consider')
                ].copy()

                target_to_accepted_subs = defaultdict(set)
                for _, row in accepted_or_considered_feedback.iterrows():
                    target_to_accepted_subs[str(row['Targets'])].add(str(row['Subs']))

                for (from_idx_str, to_idx_str), volume in volume_redistribution_dict.items():
                    if pd.isna(volume) or volume <= 1e-5: continue
                    try:
                        from_idx_int = int(from_idx_str); to_idx_int = int(to_idx_str)
                    except ValueError: continue

                    from_entity_id = idx_to_item_map.get(from_idx_int)
                    to_entity_id = idx_to_item_map.get(to_idx_int)
                    if not from_entity_id or not to_entity_id: continue

                    from_sku_details = df_for_swap_lookup.loc[from_entity_id] if from_entity_id in df_for_swap_lookup.index else pd.Series(dtype='object')
                    to_sku_details = df_for_swap_lookup.loc[to_entity_id] if to_entity_id in df_for_swap_lookup.index else pd.Series(dtype='object')
                    if isinstance(from_sku_details, pd.DataFrame): from_sku_details = from_sku_details.iloc[0] if not from_sku_details.empty else pd.Series(dtype='object')
                    if isinstance(to_sku_details, pd.DataFrame): to_sku_details = to_sku_details.iloc[0] if not to_sku_details.empty else pd.Series(dtype='object')

                    orig_sales = from_sku_details.get(sales_col, 0)
                    orig_cogs = from_sku_details.get(cogs_col, 0)
                    orig_volume = from_sku_details.get(l3m_adj_vol_col, 0)
                    recv_sales = to_sku_details.get(sales_col, 0)
                    recv_cogs = to_sku_details.get(cogs_col, 0)
                    recv_volume = to_sku_details.get(l3m_adj_vol_col, 0)
                    orig_gm_per_unit = (orig_sales - orig_cogs) / orig_volume if orig_volume > 0 else 0
                    recv_gm_per_unit = (recv_sales - recv_cogs) / recv_volume if recv_volume > 0 else 0

                    # Determine Reviewed status (feedback-based)
                    is_reviewed = False
                    if from_entity_id in target_sku_ids_in_feedback or to_entity_id in target_sku_ids_in_feedback:
                        is_reviewed = True
                    else:
                        for target_id, subs_set in target_to_accepted_subs.items():
                            if to_entity_id in subs_set and from_entity_id in subs_set:
                                is_reviewed = True
                                break

                    same_vendor_code = str(from_sku_details.get('vpn_code', '')).strip().upper() == str(to_sku_details.get('vpn_code', '')).strip().upper()
                    swap_details_list.append({
                        'Originating SKU ID': from_entity_id,
                        'Originating SKU Vendor': from_sku_details.get(vendor_col, 'N/A'),
                        'Originating SKU Description': from_sku_details.get(all_descriptions_col, 'N/A'),
                        'Originating SKU Attributes': from_sku_details.get(attribute_col, ''),
                        'Originating PL Flag': from_sku_details.get(private_label_col, 'N/A'),
                        'Originating L3M Volume': orig_volume,
                        'Originating SKU L3M Sales': orig_sales,
                        'Originating SKU L3M COGS': orig_cogs,
                        'Originating SKU GM/Unit': orig_gm_per_unit,
                        'Receiving SKU ID': to_entity_id,
                        'Receiving SKU Vendor': to_sku_details.get(vendor_col, 'N/A'),
                        'Receiving SKU Description': to_sku_details.get(all_descriptions_col, 'N/A'),
                        'Receiving SKU Attributes': to_sku_details.get(attribute_col, ''),
                        'Receiving PL Flag': to_sku_details.get(private_label_col, 'N/A'),
                        'Receiving L3M Volume': recv_volume,
                        'Receiving SKU L3M Sales': recv_sales,
                        'Receiving SKU L3M COGS': recv_cogs,
                        'Receiving SKU GM/Unit': recv_gm_per_unit,
                        'Same VPN Code': bool(same_vendor_code),
                        'Reviewed': is_reviewed,
                        'Accept/Reject': "Accept",
                        'Feedback': ""
                    })

                if swap_details_list:
                    suggest_swaps_df_final = pd.DataFrame(swap_details_list)
                    suggest_swaps_df_final = suggest_swaps_df_final[swap_output_columns]
                else:
                    print("No valid swaps to list after processing.")
                    suggest_swaps_df_final = pd.DataFrame(columns=swap_output_columns)

            worksheet_swaps = writer.sheets.get(sheet_name_swaps)
            if worksheet_swaps is None and engine_to_use == 'xlsxwriter':
                worksheet_swaps = workbook.add_worksheet(sheet_name_swaps)
                worksheet_swaps.hide_gridlines(2)

            if engine_to_use == 'xlsxwriter':
                # Add dark blue placeholder row at the top spanning all columns
                dark_blue_format = workbook.add_format({'bold': False, 'fg_color': '#1f497d', 'font_color': 'white', 'align': 'left'})
                # Write Hi in A1 and merge across all columns
                worksheet_swaps.write(0, 0, "List of Suggested Swaps between an Originating SKU and Receiving SKU. SKUs will not be swapped out if 1) They are from a preferred vendor,  2) They go to a preferred customer, or  3) They go to a redistributor", dark_blue_format)
                worksheet_swaps.merge_range(0, 0, 0, len(swap_output_columns), "List of Suggested Swaps between an Originating SKU and Receiving SKU. SKUs will not be swapped out if 1) They are from a preferred vendor,  2) They go to a preferred customer, or  3) They go to a redistributor", dark_blue_format)
                
                # Set column A to be narrow
                worksheet_swaps.set_column(0, 0, 5)
                
                orig_cols_end_idx = swap_output_columns.index('Originating SKU GM/Unit') + 1
                recv_cols_start_idx = swap_output_columns.index('Receiving SKU ID')
                recv_cols_end_idx = swap_output_columns.index('Receiving SKU GM/Unit') + 1
                reviewed_col_idx = swap_output_columns.index('Reviewed')
                feedback_col_idx = swap_output_columns.index('Feedback')
                
                # Write headers starting at B3 (row 2, column 1)
                for col_num, value in enumerate(suggest_swaps_df_final.columns.values):
                    if col_num < orig_cols_end_idx: current_h_fmt = orig_header_fmt
                    elif col_num >= recv_cols_start_idx and col_num < recv_cols_end_idx: current_h_fmt = recv_header_fmt
                    elif col_num == reviewed_col_idx: current_h_fmt = header_format  # Same grey as feedback
                    elif col_num == feedback_col_idx: current_h_fmt = header_format  # Grey header for feedback
                    else: current_h_fmt = header_format
                    worksheet_swaps.write(2, col_num + 1, value, current_h_fmt)  # +1 to start at column B

                if not suggest_swaps_df_final.empty:
                    # Create conditional formatting for Accept/Reject column
                    accept_format = workbook.add_format({'text_wrap': True, 'valign': 'top', 'border': 1, 'align': 'center', 'fg_color': '#90EE90'})  # Light green
                    reject_format = workbook.add_format({'text_wrap': True, 'valign': 'top', 'border': 1, 'align': 'center', 'fg_color': '#FFB6C1'})  # Light red
                    
                    for df_row_idx, data_tuple in enumerate(suggest_swaps_df_final.itertuples(index=False)):
                        excel_row_num = df_row_idx + 3  # Start at row 3 (index 2)
                        for col_idx, col_name in enumerate(suggest_swaps_df_final.columns):
                            cell_value = data_tuple[col_idx]
                            current_c_fmt = general_cell_format
                            if col_idx < orig_cols_end_idx: current_c_fmt = orig_cell_fmt_no_bg
                            elif col_idx >= recv_cols_start_idx and col_idx < recv_cols_end_idx: current_c_fmt = recv_cell_fmt_no_bg
                            if col_name in ['Originating L3M Volume', 'Receiving L3M Volume']: current_c_fmt = general_number_format
                            elif col_name in ['Originating SKU L3M Sales', 'Originating SKU L3M COGS', 'Receiving SKU L3M Sales', 'Receiving SKU L3M COGS']: current_c_fmt = dollar_number_format
                            elif col_name in ['Originating SKU GM/Unit', 'Receiving SKU GM/Unit']: current_c_fmt = gm_unit_format
                            elif col_name == 'Reviewed': current_c_fmt = general_cell_format  # Same grey background as feedback
                            elif col_name == 'Accept/Reject': 
                                current_c_fmt = accept_reject_cell_format
                            elif col_name == 'Feedback': current_c_fmt = general_cell_format  # Grey background for feedback
                            if pd.notna(cell_value): worksheet_swaps.write(excel_row_num, col_idx + 1, cell_value, current_c_fmt)  # +1 to start at column B
                            else: worksheet_swaps.write_blank(excel_row_num, col_idx + 1, None, current_c_fmt)  # +1 to start at column B
            else:
                # For openpyxl, we need to handle the layout differently
                # Create a DataFrame with empty first column and placeholder row
                placeholder_df = pd.DataFrame({'A': ['List of Suggested Swaps between an Originating SKU and Receiving SKU. SKUs will not be swapped out if 1) They are from a preferred vendor,  2) They go to a preferred customer, or  3) They go to a redistributor'] + [''] * len(suggest_swaps_df_final)})
                combined_df = pd.concat([placeholder_df, suggest_swaps_df_final], axis=1)
                combined_df.to_excel(writer, sheet_name=sheet_name_swaps, index=False, header=False)

            col_widths_swaps = {
                'Originating SKU ID': 20, 'Originating SKU Vendor': 20, 'Originating SKU Description': 35, 'Originating SKU Attributes': 30,
                'Originating PL Flag': 15, 'Originating L3M Volume': 15, 'Originating SKU L3M Sales': 15, 'Originating SKU L3M COGS': 15, 'Originating SKU GM/Unit': 15,
                'Receiving SKU ID': 20, 'Receiving SKU Vendor': 20, 'Receiving SKU Description': 35, 'Receiving SKU Attributes': 30,
                'Receiving PL Flag': 15, 'Receiving L3M Volume': 15, 'Receiving SKU L3M Sales': 15, 'Receiving SKU L3M COGS': 15, 'Receiving SKU GM/Unit': 15,
                'Reviewed': 10,
                'Accept/Reject': 12,
                'Feedback': 30
            }
            for i, col_name in enumerate(swap_output_columns):
                width = col_widths_swaps.get(col_name, 15)
                border_option = None
                if engine_to_use == 'xlsxwriter':
                    if col_name == 'Originating SKU GM/Unit' or col_name == 'Receiving SKU GM/Unit':
                        border_option = {'right': 2}
                    elif col_name == 'Reviewed':
                        border_option = {'right': 2}
                if worksheet_swaps: worksheet_swaps.set_column(i + 1, i + 1, width, None, border_option if engine_to_use == 'xlsxwriter' else None)  # +1 to account for column A being narrow
            if engine_to_use == 'xlsxwriter' and not suggest_swaps_df_final.empty and worksheet_swaps:
                try:
                    accept_reject_col_idx_swaps = suggest_swaps_df_final.columns.get_loc('Accept/Reject')
                    for row_num_excel in range(3, len(suggest_swaps_df_final) + 3):  # Start at row 3
                        worksheet_swaps.data_validation(row_num_excel, accept_reject_col_idx_swaps + 1, row_num_excel, accept_reject_col_idx_swaps + 1,  # +1 to account for column A
                                                        {'validate': 'list', 'source': DATA_VALIDATION_OPTIONS_ACCEPT_REJECT})
                except KeyError: pass
            if worksheet_swaps: worksheet_swaps.freeze_panes(3, 0)  # Freeze only the header row

            # --- Sheet 2: Drop List ---
            sheet_name_drop = "Drop List"
            print(f"\n--- Processing Sheet: {sheet_name_drop} ---")
            output_cols_drop_list_ordered = [item_id_col, vendor_col, all_descriptions_col, "Attributes", 'Drop_Reason', 'Accept/Reject', 'Feedback']
            if not final_dropped_skus_df_for_list.empty:
                drop_list_output_df = final_dropped_skus_df_for_list[[item_id_col, vendor_col, all_descriptions_col, attribute_col, 'Drop_Reason']].copy()
                drop_list_output_df.rename(columns={attribute_col: "Attributes"}, inplace=True)
                drop_list_output_df['Accept/Reject'] = "Accept"
                drop_list_output_df['Feedback'] = ""
                drop_list_output_df = drop_list_output_df.reindex(columns=output_cols_drop_list_ordered)
            else:
                drop_list_output_df = pd.DataFrame(columns=output_cols_drop_list_ordered)
            worksheet_drop = writer.sheets.get(sheet_name_drop)
            if worksheet_drop is None and engine_to_use == 'xlsxwriter':
                worksheet_drop = workbook.add_worksheet(sheet_name_drop)
                worksheet_drop.hide_gridlines(2)
                
                # Add dark blue placeholder row at the top spanning all columns
                dark_blue_format = workbook.add_format({'bold': False, 'fg_color': '#1f497d', 'font_color': 'white', 'align': 'left'})
                worksheet_drop.write(0, 0, "List of SKUs being dropped. These are inclusive of the SKUs in the originating columns of the 'Suggested Swaps' tab with Drop_Reason being Volume Swapped Away. There are also some of potential SKUs to drop based on if they are low GM", dark_blue_format)
                worksheet_drop.merge_range(0, 0, 0, len(drop_list_output_df.columns), "List of SKUs being dropped. These are inclusive of the SKUs in the originating columns of the 'Suggested Swaps' tab with Drop_Reason being Volume Swapped Away. There are also some of potential SKUs to drop based on if they are low GM", dark_blue_format)
                
                # Set column A to be narrow
                worksheet_drop.set_column(0, 0, 5)
                
                # Write headers starting at B3 (row 2, column 1)
                for col_num, value in enumerate(drop_list_output_df.columns.values):
                    worksheet_drop.write(2, col_num + 1, value, header_format)  # +1 to start at column B
                
                # Write data starting at B4 (row 3, column 1)
                if not drop_list_output_df.empty:
                    for df_row_idx, data_tuple in enumerate(drop_list_output_df.itertuples(index=False)):
                        excel_row_num = df_row_idx + 3  # Start at row 3 (index 2)
                        for col_idx, col_name in enumerate(drop_list_output_df.columns):
                            cell_value = data_tuple[col_idx]
                            current_c_fmt = general_cell_format
                            if col_name == 'Accept/Reject': current_c_fmt = accept_reject_cell_format
                            if pd.notna(cell_value): worksheet_drop.write(excel_row_num, col_idx + 1, cell_value, current_c_fmt)  # +1 to start at column B
                            else: worksheet_drop.write_blank(excel_row_num, col_idx + 1, None, current_c_fmt)  # +1 to start at column B
            else:
                # For openpyxl fallback
                drop_list_output_df.to_excel(writer, sheet_name=sheet_name_drop, index=False)
                col_widths_drop = {item_id_col: 25, vendor_col: 25, all_descriptions_col: 50, "Attributes": 40, 'Drop_Reason': 35, 'Accept/Reject': 15, 'Feedback': 40}
                for i, col_name in enumerate(drop_list_output_df.columns):
                    worksheet_drop.set_column(i + 1, i + 1, col_widths_drop.get(col_name, 15))  # +1 to account for column A
                if not drop_list_output_df.empty:
                    try:
                        accept_reject_col_idx_drop = drop_list_output_df.columns.get_loc('Accept/Reject')
                        for row_num_excel in range(3, len(drop_list_output_df) + 3):  # Start at row 3
                            worksheet_drop.data_validation(row_num_excel, accept_reject_col_idx_drop + 1, row_num_excel, accept_reject_col_idx_drop + 1,  # +1 to account for column A
                                                           {'validate': 'list', 'source': DATA_VALIDATION_OPTIONS_ACCEPT_REJECT})
                    except KeyError: pass
                worksheet_drop.freeze_panes(3, 0)  # Freeze only the header row


            # --- Sheet 3: Vendor Performance Summary ---
            sheet_name_vendor = "Vendor Performance Summary"
            print(f"\n--- Processing Sheet: {sheet_name_vendor} ---")
            df['sales_per_unit'] = np.where(df[l3m_adj_vol_col] > 0, df[sales_col] / df[l3m_adj_vol_col], 0)
            df['effective_suggest_sales'] = df['sales_per_unit'] * df['suggest_volume']
            vendor_summary_agg = df.groupby(vendor_col).agg(Current_Volume=(l3m_adj_vol_col, 'sum'), Current_Sales=(sales_col, 'sum'), Optimized_Volume=(suggest_volume_col, 'sum'), Optimized_Sales=('effective_suggest_sales', 'sum')).reset_index()
            total_current_volume = vendor_summary_agg['Current_Volume'].sum()
            total_current_sales = vendor_summary_agg['Current_Sales'].sum()
            total_optimized_volume = vendor_summary_agg['Optimized_Volume'].sum()
            total_optimized_sales = vendor_summary_agg['Optimized_Sales'].sum()
            vendor_summary_agg['Current Volume %'] = np.where(total_current_volume > 0, vendor_summary_agg['Current_Volume'] / total_current_volume, 0)
            vendor_summary_agg['Current Sales %'] = np.where(total_current_sales > 0, vendor_summary_agg['Current_Sales'] / total_current_sales, 0)
            vendor_summary_agg['Optimized Volume % from Swaps'] = np.where(total_optimized_volume > 0, vendor_summary_agg['Optimized_Volume'] / total_optimized_volume, 0)
            vendor_summary_agg['Optimized Sales % from Swaps'] = np.where(total_optimized_sales > 0, vendor_summary_agg['Optimized_Sales'] / total_optimized_sales, 0)
            vendor_summary_agg['Change in Volume %'] = np.where(vendor_summary_agg['Current_Volume'] > 0, (vendor_summary_agg['Optimized_Volume'] / vendor_summary_agg['Current_Volume']) - 1, np.nan)
            vendor_summary_agg['Change in Sales %'] = np.where(vendor_summary_agg['Current_Sales'] > 0, (vendor_summary_agg['Optimized_Sales'] / vendor_summary_agg['Current_Sales']) - 1, np.nan)
            vendor_summary_df_final = vendor_summary_agg[[vendor_col, 'Current_Volume', 'Current Volume %', 'Optimized_Volume', 'Optimized Volume % from Swaps', 'Change in Volume %', 'Current_Sales', 'Current Sales %', 'Optimized_Sales', 'Optimized Sales % from Swaps', 'Change in Sales %']].rename(columns={vendor_col: 'Vendor Name', 'Current_Volume': 'Current Volume (Units)', 'Optimized_Volume': 'Optimized Volume (Units)', 'Current_Sales': 'Current Sales ($)', 'Optimized_Sales': 'Optimized Sales ($)'})
            vendor_summary_df_final = vendor_summary_df_final.sort_values(by='Optimized Volume % from Swaps', ascending=False)
            worksheet_vendor_perf = writer.sheets.get(sheet_name_vendor)
            if worksheet_vendor_perf is None and engine_to_use == 'xlsxwriter':
                worksheet_vendor_perf = workbook.add_worksheet(sheet_name_vendor)
                worksheet_vendor_perf.hide_gridlines(2)
                
                # Add dark blue placeholder row at the top spanning all columns
                dark_blue_format = workbook.add_format({'bold': False, 'fg_color': '#1f497d', 'font_color': 'white', 'align': 'left'})
                worksheet_vendor_perf.write(0, 0, "Vendor Summary", dark_blue_format)
                worksheet_vendor_perf.merge_range(0, 0, 0, len(vendor_summary_df_final.columns), "Vendor Summary", dark_blue_format)
                
                # Set column A to be narrow
                worksheet_vendor_perf.set_column(0, 0, 5)
                
                # Write headers starting at B3 (row 2, column 1)
                for col_num, value in enumerate(vendor_summary_df_final.columns.values):
                    # Apply original vendor header formatting based on column type
                    if 'Volume' in value: current_h_fmt = vendor_header_volume_props
                    elif 'Sales' in value: current_h_fmt = vendor_header_sales_props
                    else: current_h_fmt = vendor_header_general_props
                    current_h_fmt = workbook.add_format(current_h_fmt)
                    worksheet_vendor_perf.write(2, col_num + 1, value, current_h_fmt)  # +1 to start at column B
                
                # Write data starting at B4 (row 3, column 1)
                if not vendor_summary_df_final.empty:
                    for df_row_idx, data_tuple in enumerate(vendor_summary_df_final.itertuples(index=False)):
                        excel_row_num = df_row_idx + 3  # Start at row 3 (index 2)
                        for col_idx, col_name in enumerate(vendor_summary_df_final.columns):
                            cell_value = data_tuple[col_idx]
                            current_c_fmt = general_cell_format
                            # Apply appropriate formatting based on column type
                            if 'Sales' in col_name and '$' in col_name: current_c_fmt = dollar_number_format
                            elif 'Sales' in col_name and '%' in col_name: current_c_fmt = percentage_format
                            elif '%' in col_name: current_c_fmt = percentage_format
                            elif 'Volume' in col_name: current_c_fmt = general_number_format
                            if pd.notna(cell_value): worksheet_vendor_perf.write(excel_row_num, col_idx + 1, cell_value, current_c_fmt)  # +1 to start at column B
                            else: worksheet_vendor_perf.write_blank(excel_row_num, col_idx + 1, None, current_c_fmt)  # +1 to start at column B
                
                # Set column widths
                for i, col_name in enumerate(vendor_summary_df_final.columns):
                    worksheet_vendor_perf.set_column(i + 1, i + 1, 15)  # +1 to account for column A
                
                worksheet_vendor_perf.freeze_panes(3, 0)  # Freeze only the header row
            else:
                # For openpyxl fallback
                vendor_summary_df_final.to_excel(writer, sheet_name=sheet_name_vendor, index=False)

            # --- Sheet 4: Customer Performance Summary ---
            sheet_name_customer = "Customer Performance Summary"
            print(f"\n--- Processing Sheet: {sheet_name_customer} ---")
            dropped_skus_set = set(df[df[suggest_volume_col] == 0][item_id_col].unique())
            true_l3m_grouped['is_effected_sku'] = true_l3m_grouped[item_id_col].isin(dropped_skus_set)
            customer_total_performance = true_l3m_grouped.groupby(['customer']).agg(Total_L3M_Volume=(l3m_adj_vol_col, 'sum'), Total_L3M_Sales=(sales_col, 'sum')).reset_index()
            effected_customer_performance = true_l3m_grouped[true_l3m_grouped['is_effected_sku']].groupby(['customer']).agg(Effected_L3M_Volume=(l3m_adj_vol_col, 'sum'), Effected_L3M_Sales=(sales_col, 'sum')).reset_index()
            customer_summary_df = pd.merge(customer_total_performance, effected_customer_performance, on=['customer'], how='left').fillna(0)
            customer_summary_df['% Volume Effected'] = np.where(customer_summary_df['Total_L3M_Volume'] > 0, customer_summary_df['Effected_L3M_Volume'] / customer_summary_df['Total_L3M_Volume'], 0)
            customer_summary_df['% Sales Effected'] = np.where(customer_summary_df['Total_L3M_Sales'] > 0, customer_summary_df['Effected_L3M_Sales'] / customer_summary_df['Total_L3M_Sales'], 0)
            
            # Add Redi and Excluded columns
            customer_summary_df['Redi'] = 'No'
            customer_summary_df['Excluded'] = 'No'
            
            # Mark redistributor customers
            if redi_customers is not None:
                customer_summary_df.loc[customer_summary_df['customer'].isin(redi_customers), 'Redi'] = 'Yes'
            
            # Mark excluded customers
            if customer_exclusions is not None:
                customer_summary_df.loc[customer_summary_df['customer'].isin(customer_exclusions), 'Excluded'] = 'Yes'
            
            customer_summary_final = customer_summary_df[['customer', 'Redi', 'Excluded', 'Total_L3M_Volume', 'Effected_L3M_Volume', '% Volume Effected', 'Total_L3M_Sales', 'Effected_L3M_Sales', '% Sales Effected']].rename(columns={'customer': 'Customer Name', 'Total_L3M_Volume': 'Total L3M Volume (Units)', 'Effected_L3M_Volume': 'Effected L3M Volume (Units)', 'Total_L3M_Sales': 'Total L3M Sales ($)', 'Effected_L3M_Sales': 'Effected L3M Sales ($)'})
            customer_summary_final = customer_summary_final.sort_values(by='Effected L3M Volume (Units)', ascending=False)
            worksheet_customer_perf = writer.sheets.get(sheet_name_customer)
            if worksheet_customer_perf is None and engine_to_use == 'xlsxwriter':
                worksheet_customer_perf = workbook.add_worksheet(sheet_name_customer)
                worksheet_customer_perf.hide_gridlines(2)
                
                # Add dark blue placeholder row at the top spanning all columns
                dark_blue_format = workbook.add_format({'bold': False, 'fg_color': '#1f497d', 'font_color': 'white', 'align': 'left'})
                worksheet_customer_perf.write(0, 0, "Customer Summary - Note any Redistributor or Non-Addressable Customer (like WFM), are uneffected", dark_blue_format)
                worksheet_customer_perf.merge_range(0, 0, 0, len(customer_summary_final.columns), "Customer Summary - Note any Redistributor or Non-Addressable Customer (like WFM), are uneffected", dark_blue_format)
                
                # Set column A to be narrow
                worksheet_customer_perf.set_column(0, 0, 5)
                
                # Write headers starting at B3 (row 2, column 1)
                for col_num, value in enumerate(customer_summary_final.columns.values):
                    # Apply original customer header formatting based on column type
                    if 'Volume' in value: current_h_fmt = customer_header_volume_props
                    elif 'Sales' in value: current_h_fmt = customer_header_sales_props
                    else: current_h_fmt = customer_header_general_props
                    current_h_fmt = workbook.add_format(current_h_fmt)
                    worksheet_customer_perf.write(2, col_num + 1, value, current_h_fmt)  # +1 to start at column B
                
                # Write data starting at B4 (row 3, column 1)
                if not customer_summary_final.empty:
                    for df_row_idx, data_tuple in enumerate(customer_summary_final.itertuples(index=False)):
                        excel_row_num = df_row_idx + 3  # Start at row 3 (index 2)
                        for col_idx, col_name in enumerate(customer_summary_final.columns):
                            cell_value = data_tuple[col_idx]
                            current_c_fmt = general_cell_format
                            # Apply appropriate formatting based on column type
                            if 'Sales' in col_name and '$' in col_name: current_c_fmt = dollar_number_format
                            elif 'Sales' in col_name and '%' in col_name: current_c_fmt = percentage_format
                            elif '%' in col_name: current_c_fmt = percentage_format
                            elif 'Volume' in col_name: current_c_fmt = general_number_format
                            if pd.notna(cell_value): worksheet_customer_perf.write(excel_row_num, col_idx + 1, cell_value, current_c_fmt)  # +1 to start at column B
                            else: worksheet_customer_perf.write_blank(excel_row_num, col_idx + 1, None, current_c_fmt)  # +1 to start at column B
                
                # Set column widths
                for i, col_name in enumerate(customer_summary_final.columns):
                    worksheet_customer_perf.set_column(i + 1, i + 1, 15)  # +1 to account for column A
                
                worksheet_customer_perf.freeze_panes(3, 0)  # Freeze only the header row
            else:
                # For openpyxl fallback
                customer_summary_final.to_excel(writer, sheet_name=sheet_name_customer, index=False)

            print(f"\nSuccessfully created Excel report: {output_excel_filepath}")
            return True

    except Exception as e:
        print(f"An error occurred while creating the Excel file: {e}")
        import traceback
        traceback.print_exc()
        return False