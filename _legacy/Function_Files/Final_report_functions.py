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

# Import column configurations from config
from config import PipelineConfig
TRANSACTION_COLUMNS = PipelineConfig.TRANSACTION_COLUMNS
# Excel formatting configuration
EXCEL_CONFIG = {
    'engine': 'xlsxwriter',
    'float_format': '%.2f'
}

# Report sheet names
REPORT_SHEETS = {
    'suggested_swaps': 'Suggested Swaps',
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
    'light_green_bg': '#E6FFE6',
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
    formats['light_green_header'] = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 
        'fg_color': EXCEL_FORMATS['light_green_bg'], 'border': 1, 'align': 'center'
    })
    formats['light_green_cell'] = workbook.add_format({
        'text_wrap': True, 'valign': 'top', 'border': 1,
        'fg_color': EXCEL_FORMATS['light_green_bg']
    })
    formats['light_green_number'] = workbook.add_format({
        'num_format': '#,##0.00', 'text_wrap': True, 'valign': 'top', 'border': 1,
        'fg_color': EXCEL_FORMATS['light_green_bg']
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
        if 'Final Qty' in column_name or 'Change in Qty' in column_name:
            return formats['light_green_header']
        elif 'Volume' in column_name:
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
        elif 'Final Qty' in column_name or 'Change in Qty' in column_name:
            return formats['light_green_number']
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

def _calculate_avg_po_amt(sku_details: pd.Series) -> float:
    """Calculate average PO amount from SKU details."""
    # Try to get po_cost_amt first, fall back to net cost
    po_amt = sku_details.get('po_cost_amt', 0)
    if po_amt and po_amt > 0:
        return float(po_amt)
    
    # Fall back to net cost if po_cost_amt is not available
    net_cost = sku_details.get(TRANSACTION_COLUMNS['net_cost'], 0)
    qty = sku_details.get(TRANSACTION_COLUMNS['qty'], 0)
    if qty > 0:
        return float(net_cost) / float(qty)
    return 0

def _identify_protected_skus(true_l3m_grouped: pd.DataFrame, item_id_col: str,
                           redi_items: Optional[List], customer_exclusion_items: Optional[List]) -> set:
    """Identify SKUs that are protected (redi or excluded items)."""
    protected_skus = set()
    
    if redi_items:
        protected_skus.update(redi_items)
    
    if customer_exclusion_items:
        protected_skus.update(customer_exclusion_items)
    
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
    item_id_col = kwargs.get('item_id_col', 'Entity--Item')
    vendor_col = kwargs.get('vendor_col', TRANSACTION_COLUMNS['vgn'])
    all_descriptions_col = kwargs.get('all_descriptions_col', 'Combined Descriptions')
    attribute_col = kwargs.get('attribute_col', 'attributes')
    sales_col = kwargs.get('sales_col', TRANSACTION_COLUMNS['gross_cost'])
    cogs_col = kwargs.get('cogs_col', TRANSACTION_COLUMNS['net_cost'])
    private_label_col = kwargs.get('private_label_col', TRANSACTION_COLUMNS['vb_flag'])
    l3m_adj_vol_col = kwargs.get('l3m_adj_vol_col', TRANSACTION_COLUMNS['qty'])
    
    # Define output columns
    swap_output_columns = [
        'Originating SKU ID', 'Originating SKU Vendor', 'Originating SKU Description', 'Originating SKU Attributes',
        'Originating PL Flag', 'Originating L3M Qty', 'Originating SKU L3M Gross Cost', 'Originating SKU L3M Net Cost', 'Originating SKU Avg. PO Amt',
        'Receiving SKU ID', 'Receiving SKU Vendor', 'Receiving SKU Description', 'Receiving SKU Attributes',
        'Receiving PL Flag', 'Receiving L3M Qty', 'Receiving SKU L3M Gross Cost', 'Receiving SKU L3M Net Cost', 'Receiving SKU Avg. PO Amt',
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

def _create_vendor_summary_sheet(writer, workbook, df: pd.DataFrame, formats: Dict,
                               engine_to_use: str, **kwargs) -> None:
    """Create the Vendor Performance Summary sheet."""
    sheet_name = REPORT_SHEETS['vendor_summary']
    print(f"\n--- Processing Sheet: {sheet_name} ---")
    
    # Extract column names from kwargs
    suggest_volume_col = kwargs.get('suggest_volume_col', 'suggest_volume')
    item_id_col = kwargs.get('item_id_col', 'Entity--Item')
    vendor_col = kwargs.get('vendor_col', TRANSACTION_COLUMNS['vgn'])
    sales_col = kwargs.get('sales_col', TRANSACTION_COLUMNS['gross_cost'])
    l3m_adj_vol_col = kwargs.get('l3m_adj_vol_col', TRANSACTION_COLUMNS['qty'])
    
    # Create vendor summary data
    vendor_summary_data = _create_vendor_summary_data(df, suggest_volume_col, item_id_col,
                                                    vendor_col, sales_col, l3m_adj_vol_col)
    
    # Create worksheet and write data
    worksheet = _create_worksheet(writer, workbook, sheet_name, engine_to_use)
    _write_vendor_summary_data(worksheet, workbook, vendor_summary_data, formats, engine_to_use)

def _create_customer_summary_sheet(writer, workbook, df: pd.DataFrame, true_l3m_grouped: pd.DataFrame,
                                 formats: Dict, engine_to_use: str, redi_items: Optional[List] = None,
                                 customer_exclusion_items: Optional[List] = None, **kwargs) -> None:
    """Create the Customer Performance Summary sheet."""
    sheet_name = REPORT_SHEETS['customer_summary']
    print(f"\n--- Processing Sheet: {sheet_name} ---")
    
    # Extract column names from kwargs
    suggest_volume_col = kwargs.get('suggest_volume_col', 'suggest_volume')
    item_id_col = kwargs.get('item_id_col', 'Entity--Item')
    l3m_adj_vol_col = kwargs.get('l3m_adj_vol_col', TRANSACTION_COLUMNS['qty'])
    sales_col = kwargs.get('sales_col', TRANSACTION_COLUMNS['gross_cost'])
    
    # Create customer summary data
    customer_summary_data = _create_customer_summary_data(df, true_l3m_grouped, suggest_volume_col,
                                                        item_id_col, l3m_adj_vol_col, sales_col,
                                                        redi_items, customer_exclusion_items)
    
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
    if not df[kwargs.get('item_id_col', 'Entity--Item')].is_unique:
        print(f"Warning: '{kwargs.get('item_id_col', 'Entity--Item')}' is not unique in the input DataFrame. Using first occurrence for lookups in 'Suggested Swaps'.")
        df_for_swap_lookup = df.drop_duplicates(subset=[kwargs.get('item_id_col', 'Entity--Item')]).set_index(kwargs.get('item_id_col', 'Entity--Item'))
    else:
        df_for_swap_lookup = df.set_index(kwargs.get('item_id_col', 'Entity--Item'))
    
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
    item_id_col = kwargs.get('item_id_col', 'Entity--Item')
    vendor_col = kwargs.get('vendor_col', TRANSACTION_COLUMNS['vgn'])
    all_descriptions_col = kwargs.get('all_descriptions_col', 'Combined Descriptions')
    attribute_col = kwargs.get('attribute_col', 'attributes')
    sales_col = kwargs.get('sales_col', TRANSACTION_COLUMNS['gross_cost'])
    cogs_col = kwargs.get('cogs_col', TRANSACTION_COLUMNS['net_cost'])
    private_label_col = kwargs.get('private_label_col', TRANSACTION_COLUMNS['vb_flag'])
    l3m_adj_vol_col = kwargs.get('l3m_adj_vol_col', TRANSACTION_COLUMNS['qty'])
    
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
        
        # Calculate values for new columns
        orig_gross_cost = from_sku_details.get(sales_col, 0)
        orig_net_cost = from_sku_details.get(cogs_col, 0)
        orig_qty = from_sku_details.get(l3m_adj_vol_col, 0)
        
        recv_gross_cost = to_sku_details.get(sales_col, 0)
        recv_net_cost = to_sku_details.get(cogs_col, 0)
        recv_qty = to_sku_details.get(l3m_adj_vol_col, 0)
        
        orig_avg_po_amt = _calculate_avg_po_amt(from_sku_details)
        recv_avg_po_amt = _calculate_avg_po_amt(to_sku_details)
        
        # Create swap details
        swap_details_list.append({
            'Originating SKU ID': from_entity_id,
            'Originating SKU Vendor': from_sku_details.get(vendor_col, 'N/A'),
            'Originating SKU Description': from_sku_details.get(all_descriptions_col, 'N/A'),
            'Originating SKU Attributes': from_sku_details.get(attribute_col, ''),
            'Originating PL Flag': from_sku_details.get(private_label_col, 'N/A'),
            'Originating L3M Qty': orig_qty,
            'Originating SKU L3M Gross Cost': orig_gross_cost,
            'Originating SKU L3M Net Cost': orig_net_cost,
            'Originating SKU Avg. PO Amt': orig_avg_po_amt,
            'Receiving SKU ID': to_entity_id,
            'Receiving SKU Vendor': to_sku_details.get(vendor_col, 'N/A'),
            'Receiving SKU Description': to_sku_details.get(all_descriptions_col, 'N/A'),
            'Receiving SKU Attributes': to_sku_details.get(attribute_col, ''),
            'Receiving PL Flag': to_sku_details.get(private_label_col, 'N/A'),
            'Receiving L3M Qty': recv_qty,
            'Receiving SKU L3M Gross Cost': recv_gross_cost,
            'Receiving SKU L3M Net Cost': recv_net_cost,
            'Receiving SKU Avg. PO Amt': recv_avg_po_amt,
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


def _determine_review_status_comprehensive(from_entity_id: str, to_entity_id: str, feedback_df: pd.DataFrame) -> str:
    """Determine if a swap has been reviewed based on feedback DataFrame."""
    if feedback_df is None or feedback_df.empty:
        return 'No'
    
    # Convert to string for comparison
    from_entity_id = str(from_entity_id)
    to_entity_id = str(to_entity_id)
    
    # Check if either ID is in the Targets column
    if 'Targets' in feedback_df.columns:
        targets_in_feedback = set(feedback_df['Targets'].astype(str).dropna().unique())
        if from_entity_id in targets_in_feedback or to_entity_id in targets_in_feedback:
            return 'Yes'
    
    # Check for mutual acceptance/consideration
    if 'Targets' in feedback_df.columns and 'Subs' in feedback_df.columns and 'Correct' in feedback_df.columns:
        accepted_or_considered = feedback_df[
            (feedback_df['Correct'] == 'Accept') | (feedback_df['Correct'] == 'Consider')
        ]
        
        for _, row in accepted_or_considered.iterrows():
            target = str(row['Targets'])
            sub = str(row['Subs'])
            
            # Check if this is a mutual relationship
            if ((from_entity_id == target and to_entity_id == sub) or 
                (to_entity_id == target and from_entity_id == sub)):
                return 'Yes'
    
    return 'No'

def _create_vendor_summary_data(df: pd.DataFrame, suggest_volume_col: str, item_id_col: str,
                              vendor_col: str, sales_col: str, l3m_adj_vol_col: str) -> pd.DataFrame:
    """Create vendor summary data for the Vendor Performance Summary sheet."""
    # Group by vendor and calculate metrics (removed sales_col aggregation)
    vendor_summary = df.groupby(vendor_col).agg({
        l3m_adj_vol_col: ['sum', 'count']
    }).reset_index()
    
    # Flatten column names
    vendor_summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in vendor_summary.columns]
    
    # Rename columns for output (changed Volume to Qty, removed Sales)
    vendor_summary = vendor_summary.rename(columns={
        f'{l3m_adj_vol_col}_sum': 'Total L3M Qty (Units)',
        f'{l3m_adj_vol_col}_count': 'SKU Count'
    })
    
    return vendor_summary.sort_values(by='Total L3M Qty (Units)', ascending=False)

def _create_customer_summary_data(df: pd.DataFrame, true_l3m_grouped: pd.DataFrame, suggest_volume_col: str,
                                item_id_col: str, l3m_adj_vol_col: str, sales_col: str,
                                redi_items: Optional[List] = None, customer_exclusion_items: Optional[List] = None) -> pd.DataFrame:
    """Create customer summary data for the Customer Performance Summary sheet."""
    dropped_skus_set = set(df[df[suggest_volume_col] == 0][item_id_col].unique())
    true_l3m_grouped['is_effected_sku'] = true_l3m_grouped[item_id_col].isin(dropped_skus_set)
    
    # Calculate total and affected performance (removed sales)
    customer_total_performance = true_l3m_grouped.groupby(['customer']).agg(
        Total_L3M_Qty=(l3m_adj_vol_col, 'sum')
    ).reset_index()
    
    effected_customer_performance = true_l3m_grouped[true_l3m_grouped['is_effected_sku']].groupby(['customer']).agg(
        Effected_L3M_Qty=(l3m_adj_vol_col, 'sum')
    ).reset_index()
    
    # Merge and calculate percentages (removed sales percentage)
    customer_summary_df = pd.merge(customer_total_performance, effected_customer_performance, on=['customer'], how='left').fillna(0)
    customer_summary_df['% Qty Effected'] = np.where(
        customer_summary_df['Total_L3M_Qty'] > 0,
        customer_summary_df['Effected_L3M_Qty'] / customer_summary_df['Total_L3M_Qty'],
        0
    )
    
    # Add Redi and Excluded columns based on Entity--Item lists
    customer_summary_df['Redi'] = 'No'
    customer_summary_df['Excluded'] = 'No'
    
    if redi_items:
        # Find customers who have redi items
        redi_customers = true_l3m_grouped[
            true_l3m_grouped[item_id_col].isin(redi_items)
        ]['customer'].unique()
        customer_summary_df.loc[customer_summary_df['customer'].isin(redi_customers), 'Redi'] = 'Yes'
    
    if customer_exclusion_items:
        # Find customers who have excluded items
        excluded_customers = true_l3m_grouped[
            true_l3m_grouped[item_id_col].isin(customer_exclusion_items)
        ]['customer'].unique()
        customer_summary_df.loc[customer_summary_df['customer'].isin(excluded_customers), 'Excluded'] = 'Yes'
    
    # Rename columns for output (removed sales columns, changed volume to qty)
    customer_summary_final = customer_summary_df[[
        'customer', 'Redi', 'Excluded', 'Total_L3M_Qty', 'Effected_L3M_Qty', '% Qty Effected'
    ]].rename(columns={
        'customer': 'Customer Name',
        'Total_L3M_Qty': 'Total L3M Qty (Units)',
        'Effected_L3M_Qty': 'Effected L3M Qty (Units)'
    })
    
    return customer_summary_final.sort_values(by='Effected L3M Qty (Units)', ascending=False)

# =============================================================================
# EXCEL WRITING HELPER FUNCTIONS
# =============================================================================

def _write_suggested_swaps_data(worksheet, workbook, df: pd.DataFrame, formats: Dict, engine_to_use: str) -> None:
    """Write suggested swaps data to worksheet with proper formatting."""
    if engine_to_use == 'xlsxwriter':
        # Add dark blue placeholder row
        dark_blue_format = workbook.add_format({'bold': False, 'fg_color': '#1f497d', 'font_color': 'white', 'align': 'left'})
        header_text = "List of Suggested Swaps between an Originating SKU and Receiving SKU. SKUs will not be swapped out if 1) They are from a preferred vendor,  2) They go to a preferred customer, or  3) They go to a redistributor"
        worksheet.write(0, 0, header_text, dark_blue_format)
        worksheet.merge_range(0, 0, 0, len(df.columns), header_text, dark_blue_format)
        
        # Set column A to be narrow
        worksheet.set_column(0, 0, 5)
        
        # Write headers starting at B3
        for col_num, value in enumerate(df.columns.values):
            header_format = _get_suggested_swaps_header_format(formats, col_num, len(df.columns), value)
            worksheet.write(2, col_num + 1, value, header_format)
        
        # Write data starting at B4
        if not df.empty:
            for df_row_idx, data_tuple in enumerate(df.itertuples(index=False)):
                excel_row_num = df_row_idx + 3
                # Set row height to allow for text wrapping (similar to old function approach)
                worksheet.set_row(excel_row_num, 30)  # Set row height to accommodate wrapped text
                for col_idx, col_name in enumerate(df.columns):
                    cell_value = data_tuple[col_idx]
                    cell_format = _get_suggested_swaps_cell_format(formats, col_name, col_idx, len(df.columns))
                    
                    if pd.notna(cell_value):
                        worksheet.write(excel_row_num, col_idx + 1, cell_value, cell_format)
                    else:
                        worksheet.write_blank(excel_row_num, col_idx + 1, None, cell_format)
        
        # Set column widths with specific widths for each column type
        col_widths_swaps = {
            'Originating SKU ID': 20, 'Originating SKU Vendor': 20, 'Originating SKU Description': 35, 'Originating SKU Attributes': 30,
            'Originating PL Flag': 15, 'Originating Qty': 15,
            'Receiving SKU ID': 20, 'Receiving SKU Vendor': 20, 'Receiving SKU Description': 35, 'Receiving SKU Attributes': 30,
            'Receiving PL Flag': 15, 'Receiving Qty': 15,
            'Savings': 15, 'Reviewed': 10, 'Accept/Reject': 12, 'Feedback': 30
        }
        for i, col_name in enumerate(df.columns):
            width = col_widths_swaps.get(col_name, 15)
            border_option = None
            if engine_to_use == 'xlsxwriter':
                if col_name == 'Originating Qty':
                    border_option = {'right': 2}
                elif col_name == 'Receiving Qty':
                    border_option = {'right': 2}
                elif col_name == 'Savings':
                    border_option = {'right': 2}
                elif col_name == 'Reviewed':
                    border_option = {'right': 2}
            worksheet.set_column(i + 1, i + 1, width, None, border_option if engine_to_use == 'xlsxwriter' else None)
        
        # Add data validation for Accept/Reject dropdown
        if not df.empty and 'Accept/Reject' in df.columns:
            try:
                accept_reject_col_idx = df.columns.get_loc('Accept/Reject')
                data_validation_options = ["Accept", "Reject"]
                for row_num_excel in range(3, len(df) + 3):  # Start at row 3 (data rows)
                    worksheet.data_validation(row_num_excel, accept_reject_col_idx + 1, row_num_excel, accept_reject_col_idx + 1,  # +1 to account for column A
                                            {'validate': 'list', 'source': data_validation_options})
            except KeyError:
                pass  # Column doesn't exist, skip validation
        
        # Freeze panes at row 3 (headers)
        worksheet.freeze_panes(3, 0)


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

def _get_suggested_swaps_header_format(formats: Dict, col_idx: int, total_cols: int, col_name: str = ''):
    """Get appropriate header format for suggested swaps sheet."""
    orig_cols_end_idx = 6  # Index after 'Originating L3M Qty'
    recv_cols_start_idx = 6  # Index of 'Receiving SKU ID'
    recv_cols_end_idx = 12  # Index after 'Receiving L3M Qty'
    savings_col_idx = 12  # Index of 'Savings'
    reviewed_col_idx = 13  # Index of 'Reviewed'
    
    if col_idx < orig_cols_end_idx:
        return formats['orig_header']
    elif recv_cols_start_idx <= col_idx < recv_cols_end_idx:
        return formats['recv_header']
    elif col_idx == savings_col_idx or col_name == 'Savings':
        return formats.get('savings_header', formats.get('red_header', formats['general_header']))
    elif col_idx >= reviewed_col_idx:
        return formats['reviewed_header']
    else:
        return formats['general_header']

def _get_suggested_swaps_cell_format(formats: Dict, col_name: str, col_idx: int, total_cols: int):
    """Get appropriate cell format for suggested swaps sheet."""
    orig_cols_end_idx = 6
    recv_cols_start_idx = 6
    recv_cols_end_idx = 12
    savings_col_idx = 12
    
    if col_idx < orig_cols_end_idx:
        base_format = formats['general_cell']
    elif recv_cols_start_idx <= col_idx < recv_cols_end_idx:
        base_format = formats['general_cell']
    elif col_idx == savings_col_idx or col_name == 'Savings':
        base_format = formats.get('savings_cell', formats.get('red_cell', formats['general_cell']))
    else:
        base_format = formats['general_cell']
    
    # Apply number formatting
    if 'Volume' in col_name or 'Qty' in col_name:
        return formats['general_number']
    elif 'Sales' in col_name and '$' in col_name:
        return formats['dollar_number']
    elif 'GM/Unit' in col_name:
        return formats['gm_unit']
    elif col_name == 'Savings':
        return formats.get('dollar_number', base_format)
    elif col_name == 'Accept/Reject':
        return formats['accept_reject_cell']
    else:
        return base_format

def _get_vendor_header_format(formats: Dict, col_name: str):
    """Get appropriate header format for vendor summary sheet."""
    if 'Qty' in col_name or 'Volume' in col_name:
        return formats['vendor_header_volume']
    elif 'Sales' in col_name or 'Savings' in col_name:
        return formats['vendor_header_sales']
    else:
        return formats['vendor_header_general']

def _get_vendor_cell_format(formats: Dict, col_name: str):
    """Get appropriate cell format for vendor summary sheet."""
    if ('Sales' in col_name or 'Savings' in col_name) and '$' in col_name:
        return formats['dollar_number']
    elif ('Sales' in col_name or 'Savings' in col_name) and '%' in col_name:
        return formats['percentage']
    elif '%' in col_name:
        return formats['percentage']
    elif 'Qty' in col_name or 'Volume' in col_name:
        return formats['general_number']
    else:
        return formats['general_cell']

def _get_customer_header_format(formats: Dict, col_name: str):
    """Get appropriate header format for customer summary sheet."""
    if 'Qty' in col_name or 'Volume' in col_name:
        return formats['customer_header_volume']
    elif 'Sales' in col_name or 'Savings' in col_name:
        return formats['customer_header_sales']
    else:
        return formats['customer_header_general']

def _get_customer_cell_format(formats: Dict, col_name: str):
    """Get appropriate cell format for customer summary sheet."""
    if ('Sales' in col_name or 'Savings' in col_name) and '$' in col_name:
        return formats['dollar_number']
    elif ('Sales' in col_name or 'Savings' in col_name) and '%' in col_name:
        return formats['percentage']
    elif '%' in col_name:
        return formats['percentage']
    elif 'Qty' in col_name or 'Volume' in col_name:
        return formats['general_number']
    else:
        return formats['general_cell']

# =============================================================================
# REPORTING AND FEEDBACK FUNCTIONS
# =============================================================================

def _create_suggested_swaps_sheet_new(writer, workbook, updated_df: pd.DataFrame, 
                                     im_final_with_feedback: pd.DataFrame, formats: Dict, 
                                     engine_to_use: str) -> None:
    """Create the Suggested Swaps sheet using new DataFrame structure."""
    sheet_name = REPORT_SHEETS['suggested_swaps']
    print(f"\n--- Processing Sheet: {sheet_name} ---")
    
    # Filter for items that have swaps (Action contains swap information)
    swaps_df = updated_df[updated_df['Action'].str.contains('Swapped', na=False)].copy()
    
    if swaps_df.empty:
        print("No swaps found in updated_df. Creating empty sheet.")
        empty_df = pd.DataFrame(columns=[
            'Originating SKU ID', 'Originating SKU Vendor', 'Originating SKU Description', 'Originating SKU Attributes',
            'Originating PL Flag', 'Originating Qty',
            'Receiving SKU ID', 'Receiving SKU Vendor', 'Receiving SKU Description', 'Receiving SKU Attributes',
            'Receiving PL Flag', 'Receiving Qty',
            'Savings', 'Reviewed', 'Accept/Reject', 'Feedback'
        ])
    else:
        # Create swap details from the Action column and merge with feedback data
        swap_details = []
        for _, row in swaps_df.iterrows():
            orig_item = row['Entity--Item']
            action = row['Action']
            
            # Extract receiving item from action (assuming format like "Swapped to ITEM_ID")
            if 'Swapped to ' in action:
                recv_item = action.replace('Swapped to ', '').strip()
                
                # Get originating item details from current row (which has all the data from analyze_swap_results)
                orig_vendor = row.get('VGN', 'N/A')
                orig_description = row.get('Combined Descriptions', 'N/A')
                orig_attributes = row.get('attributes', '')
                orig_pl_flag = row.get('PL_Flag', 'N/A')
                
                # Get receiving item details from updated_df
                recv_details = updated_df[updated_df['Entity--Item'] == recv_item]
                
                if not recv_details.empty:
                    recv_row = recv_details.iloc[0]
                    recv_vendor = recv_row.get('VGN', 'N/A')
                    recv_description = recv_row.get('Combined Descriptions', 'N/A')
                    recv_attributes = recv_row.get('attributes', '')
                    recv_pl_flag = recv_row.get('PL_Flag', 'N/A')
                    recv_volume = recv_row.get('Final_Volume', 0)
                else:
                    # Fallback values if receiving item not found in updated_df
                    recv_vendor = 'N/A'
                    recv_description = 'N/A'
                    recv_attributes = ''
                    recv_pl_flag = 'N/A'
                    recv_volume = 0
                
                swap_details.append({
                    'Originating SKU ID': orig_item,
                    'Originating SKU Vendor': orig_vendor,
                    'Originating SKU Description': orig_description,
                    'Originating SKU Attributes': orig_attributes,
                    'Originating PL Flag': orig_pl_flag,
                    'Originating Qty': row.get('Original_Volume', 0),
                    'Receiving SKU ID': recv_item,
                    'Receiving SKU Vendor': recv_vendor,
                    'Receiving SKU Description': recv_description,
                    'Receiving SKU Attributes': recv_attributes,
                    'Receiving PL Flag': recv_pl_flag,
                    'Receiving Qty': recv_volume,
                    'Savings': row.get('Savings_From_Swapping_Away', 0),  # Use Savings_From_Swapping_Away for originating item swaps
                    'Reviewed': _determine_review_status_comprehensive(orig_item, recv_item, im_final_with_feedback),
                    'Accept/Reject': 'Accept',
                    'Feedback': ''
                })
        
        empty_df = pd.DataFrame(swap_details) if swap_details else pd.DataFrame(columns=[
            'Originating SKU ID', 'Originating SKU Vendor', 'Originating SKU Description', 'Originating SKU Attributes',
            'Originating PL Flag', 'Originating Qty',
            'Receiving SKU ID', 'Receiving SKU Vendor', 'Receiving SKU Description', 'Receiving SKU Attributes',
            'Receiving PL Flag', 'Receiving Qty',
            'Savings', 'Reviewed', 'Accept/Reject', 'Feedback'
        ])
    
    # Create worksheet and write data
    worksheet = _create_worksheet(writer, workbook, sheet_name, engine_to_use)
    _write_suggested_swaps_data(worksheet, workbook, empty_df, formats, engine_to_use)


def _create_vendor_summary_sheet_new(writer, workbook, updated_df: pd.DataFrame, 
                                    formats: Dict, engine_to_use: str) -> None:
    """Create the Vendor Performance Summary sheet with supplier, original qty, qty swapped to, qty swapped away, final qty, change in qty, change in qty %, and savings."""
    sheet_name = REPORT_SHEETS['vendor_summary']
    print(f"\n--- Processing Sheet: {sheet_name} ---")
    
    # Group by Original_Supplier and calculate metrics
    vendor_summary = updated_df.groupby('Original_Supplier').agg({
        'Original_Volume': 'sum',
        'Volume_Swapped_To': 'sum',
        'Volume_Swapped_Away': 'sum',
        'Savings_From_Swapping_To': 'sum'
    }).reset_index()
    
    # Rename columns for output
    vendor_summary = vendor_summary.rename(columns={
        'Original_Supplier': 'Supplier',
        'Original_Volume': 'Original Qty (Units)',
        'Volume_Swapped_To': 'Qty Swapped To (Units)',
        'Volume_Swapped_Away': 'Qty Swapped Away (Units)',
        'Savings_From_Swapping_To': 'Savings From Swapping To ($)'
    })
    
    # Calculate Final Qty = Original + Swapped To - Swapped Away
    vendor_summary['Final Qty (Units)'] = (vendor_summary['Original Qty (Units)'] + 
                                          vendor_summary['Qty Swapped To (Units)'] - 
                                          vendor_summary['Qty Swapped Away (Units)'])
    
    # Calculate Change in Qty (Units) = Final - Original
    vendor_summary['Change in Qty (Units)'] = (vendor_summary['Final Qty (Units)'] - 
                                              vendor_summary['Original Qty (Units)'])
    
    # Calculate Change in Qty (%) = (Final - Original) / Original * 100
    vendor_summary['Change in Qty (%)'] = np.where(
        vendor_summary['Original Qty (Units)'] > 0,
        ((vendor_summary['Final Qty (Units)'] - vendor_summary['Original Qty (Units)']) /
         vendor_summary['Original Qty (Units)'] * 100).round(2),
        0  # Set to 0% when original quantity is 0
    )
    
    # Sort by final quantity descending
    vendor_summary_sorted = vendor_summary.sort_values(by='Final Qty (Units)', ascending=False)
    
    # Create worksheet and write data
    worksheet = _create_worksheet(writer, workbook, sheet_name, engine_to_use)
    _write_vendor_summary_data(worksheet, workbook, vendor_summary_sorted, formats, engine_to_use)

def _create_customer_summary_sheet_new(writer, workbook, updated_df: pd.DataFrame, 
                                      transactions_grouped_by_item_customer: pd.DataFrame,
                                      formats: Dict, engine_to_use: str, 
                                      redi_customers: Optional[List] = None,
                                      customer_exclusions: Optional[List] = None) -> None:
    """Create the Customer Performance Summary sheet using new DataFrame structure."""
    sheet_name = REPORT_SHEETS['customer_summary']
    print(f"\n--- Processing Sheet: {sheet_name} ---")
    
    # Identify dropped SKUs (Final_Volume = 0)
    dropped_skus = set(updated_df[updated_df['Final_Volume'] == 0]['Entity--Item'].unique())
    
    # Mark affected SKUs in transactions data
    transactions_grouped_by_item_customer['is_affected_sku'] = \
        transactions_grouped_by_item_customer['Entity--Item'].isin(dropped_skus)
    
    # Calculate total and affected performance by customer
    customer_total_performance = transactions_grouped_by_item_customer.groupby('Customer_Name').agg(
        Total_L3M_Qty=('Total_Qty', 'sum'),
        Total_L3M_Gross_Cost=('Total_Gross_Cost', 'sum')
    ).reset_index()
    
    affected_customer_performance = transactions_grouped_by_item_customer[
        transactions_grouped_by_item_customer['is_affected_sku']
    ].groupby('Customer_Name').agg(
        Affected_L3M_Qty=('Total_Qty', 'sum'),
        Affected_L3M_Gross_Cost=('Total_Gross_Cost', 'sum')
    ).reset_index()
    
    # Merge and calculate percentages
    customer_summary_df = pd.merge(customer_total_performance, affected_customer_performance, 
                                 on='Customer_Name', how='left').fillna(0)
    
    customer_summary_df['% Qty Affected'] = np.where(
        customer_summary_df['Total_L3M_Qty'] > 0,
        customer_summary_df['Affected_L3M_Qty'] / customer_summary_df['Total_L3M_Qty'],
        0
    )
    
    # Add Redi and Excluded columns based on customer codes
    customer_summary_df['Redi'] = 'No'
    customer_summary_df['Excluded'] = 'No'
    
    if redi_customers:
        # Find customers who match the redi customer codes
        redi_customer_names = transactions_grouped_by_item_customer[
            transactions_grouped_by_item_customer[TRANSACTION_COLUMNS['customer_code']].isin(redi_customers)
        ]['Customer_Name'].unique()
        customer_summary_df.loc[customer_summary_df['Customer_Name'].isin(redi_customer_names), 'Redi'] = 'Yes'
    
    if customer_exclusions:
        # Find customers who match the excluded customer codes
        excluded_customer_names = transactions_grouped_by_item_customer[
            transactions_grouped_by_item_customer[TRANSACTION_COLUMNS['customer_code']].isin(customer_exclusions)
        ]['Customer_Name'].unique()
        customer_summary_df.loc[customer_summary_df['Customer_Name'].isin(excluded_customer_names), 'Excluded'] = 'Yes'
    
    # Final column selection and renaming
    customer_summary_final = customer_summary_df[[
        'Customer_Name', 'Redi', 'Excluded', 'Total_L3M_Qty', 'Affected_L3M_Qty', '% Qty Affected'
    ]].rename(columns={
        'Customer_Name': 'Customer Name',
        'Total_L3M_Qty': 'Total Qty (Units)',
        'Affected_L3M_Qty': 'Affected Qty (Units)'
    })
    
    customer_summary_sorted = customer_summary_final.sort_values(by='Affected Qty (Units)', ascending=False)
    
    # Create worksheet and write data
    worksheet = _create_worksheet(writer, workbook, sheet_name, engine_to_use)
    _write_customer_summary_data(worksheet, workbook, customer_summary_sorted, formats, engine_to_use)

def create_enhanced_sku_report(
    updated_df: pd.DataFrame,
    im_final_with_feedback: pd.DataFrame,
    transactions_grouped_by_item_customer: pd.DataFrame,
    output_excel_filepath: str,
    item_id_col: str = 'Entity--Item',
    vendor_col: str = TRANSACTION_COLUMNS['vgn'],
    all_descriptions_col: str = 'Combined Descriptions',
    attribute_col: str = 'attributes',
    matches_col: str = 'Matches',
    gross_cost_col: str = TRANSACTION_COLUMNS['gross_cost'],
    net_cost_col: str = TRANSACTION_COLUMNS['net_cost'],
    private_label_col: str = TRANSACTION_COLUMNS['vb_flag'],
    qty_col: str = TRANSACTION_COLUMNS['qty'],
    redi_customers: Optional[List] = None,
    customer_exclusions: Optional[List] = None
) -> bool:
    """
    Creates an Excel file with "Suggested Swaps", "Vendor Performance Summary",
    and "Customer Performance Summary" sheets using the new DataFrame structure.
    
    This function generates a comprehensive report with optimization results, including:
    - Suggested swaps with originating and receiving SKU details
    - Vendor performance summary with quantity metrics
    - Customer performance summary showing impact of changes
    
    Args:
        updated_df (pd.DataFrame): DataFrame with columns ['Entity--Item', 'Original_Volume', 
            'Final_Volume', 'Volume_Change', 'Original_Cost', 'Original_Supplier', 'New_Supplier', 
            'Action', 'Savings']
        im_final_with_feedback (pd.DataFrame): DataFrame with item details and feedback
        transactions_grouped_by_item_customer (pd.DataFrame): DataFrame with columns 
            ['Entity--Item', TRANSACTION_COLUMNS['customer_code'], 'Total_Qty', 'Total_Net_Cogs', 'Total_Gross_Cost', 
            'Matches', TRANSACTION_COLUMNS['vgn'], TRANSACTION_COLUMNS['customer_class'], 'Customer_Name', TRANSACTION_COLUMNS['case_pack']]
        output_excel_filepath (str): Full path for the output Excel file.
        item_id_col (str): Name of the Entity ID column.
        vendor_col (str): Name of the vendor name column.
        all_descriptions_col (str): Name of the column containing Combined Descriptions.
        attribute_col (str): Name of the column for "Attributes".
        matches_col (str): Name of the column with list of matches.
        gross_cost_col (str): Name of the gross cost column.
        net_cost_col (str): Name of the net cost column.
        private_label_col (str): Name of the private label flag column.
        qty_col (str): Name of the quantity column.
        redi_customers (list, optional): List of redistributor customer codes.
        customer_exclusions (list, optional): List of customer codes to exclude.

    Returns:
        bool: True if the Excel file was created successfully, False otherwise.
    """
    print(f"--- Starting to generate Enhanced SKU Report: {output_excel_filepath} ---")

    # Validate inputs
    validate_dataframe(updated_df, "updated_df")
    validate_dataframe(im_final_with_feedback, "im_final_with_feedback")
    validate_dataframe(transactions_grouped_by_item_customer, "transactions_grouped_by_item_customer")
    validate_string_input(output_excel_filepath, "output_excel_filepath")
    

    # Validate required columns in updated_df
    required_updated_cols = ['Entity--Item', 'Original_Volume', 'Final_Volume', 'Volume_Change',
                           'Original_Cost', 'Original_Supplier', 'New_Supplier', 'VGN', 
                           'Combined Descriptions', 'attributes', 'PL_Flag', 'Action', 
                           'Savings_From_Swapping_Away', 'Savings_From_Swapping_To']
    validate_columns_exist(updated_df, required_updated_cols, "updated_df")
    
    # Validate required columns in transactions_grouped_by_item_customer
    required_transaction_cols = ['Entity--Item', TRANSACTION_COLUMNS['customer_code'], 'Total_Qty', 'Total_Net_Cogs',
                               'Total_Gross_Cost', 'Matches', TRANSACTION_COLUMNS['vgn'], TRANSACTION_COLUMNS['customer_class'], 
                               'Customer_Name', TRANSACTION_COLUMNS['case_pack']]
    validate_columns_exist(transactions_grouped_by_item_customer, required_transaction_cols, 
                         "transactions_grouped_by_item_customer")

    try:
        # Create Excel writer
        with pd.ExcelWriter(output_excel_filepath, engine=EXCEL_CONFIG['engine']) as writer:
            workbook = writer.book
            formats = _create_excel_formats(workbook)
            
            # Create sheets with updated data structure
            _create_suggested_swaps_sheet_new(writer, workbook, updated_df, im_final_with_feedback, 
                                            formats, EXCEL_CONFIG['engine'])
            
            _create_vendor_summary_sheet_new(writer, workbook, updated_df, formats, 
                                           EXCEL_CONFIG['engine'])
            
            _create_customer_summary_sheet_new(writer, workbook, updated_df, 
                                             transactions_grouped_by_item_customer,
                                             formats, EXCEL_CONFIG['engine'], 
                                             redi_customers, customer_exclusions)

        print(f"--- Successfully created Enhanced SKU Report: {output_excel_filepath} ---")
        return True

    except Exception as e:
        print(f"Error creating Enhanced SKU Report: {str(e)}")
        return False

def read_and_process_feedback_excel(
    excel_filepath: str,
    suggest_swaps_sheet_name: str = REPORT_SHEETS['suggested_swaps'],
    suggest_swaps_orig_sku_col: str = 'Originating SKU ID',
    suggest_swaps_recv_sku_col: str = 'Receiving SKU ID',
    accept_reject_col: str = 'Accept/Reject',
    feedback_col: str = 'Feedback'
) -> pd.DataFrame:
    """
    Reads the "Suggested Swaps" sheet from a specified Excel file
    and returns a DataFrame with selected columns.

    Args:
        excel_filepath (str): The full path to the Excel file.
        suggest_swaps_sheet_name (str): Name of the sheet containing suggested swaps.
        suggest_swaps_orig_sku_col (str): Name of the originating SKU ID column in "Suggested Swaps".
        suggest_swaps_recv_sku_col (str): Name of the receiving SKU ID column in "Suggested Swaps".
        accept_reject_col (str): Name of the column containing "Accept/Reject" status.
        feedback_col (str): Name of the column containing feedback.

    Returns:
        pd.DataFrame: DataFrame with [suggest_swaps_orig_sku_col, 
                     suggest_swaps_recv_sku_col, accept_reject_col, feedback_col]
                     from the "Suggested Swaps" sheet.
                     Returns None if the file doesn't exist or sheet is not found.
    """
    print(f"--- Reading feedback from Excel file: {excel_filepath} ---")

    if not os.path.exists(excel_filepath):
        print(f"Error: Excel file not found at {excel_filepath}")
        return None

    try:
        xls = pd.ExcelFile(excel_filepath)
    except Exception as e:
        print(f"Error opening Excel file: {e}")
        return None

    # --- Process "Suggested Swaps" sheet ---
    if suggest_swaps_sheet_name in xls.sheet_names:
        print(f"Processing sheet: {suggest_swaps_sheet_name}...")
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

    return suggested_swaps_df

def incorporate_swaps_feedback(
    im_final: pd.DataFrame,
    swap: pd.DataFrame,
    entity_col: str = 'Entity--Item',
    matches_col: str = 'Matches',
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