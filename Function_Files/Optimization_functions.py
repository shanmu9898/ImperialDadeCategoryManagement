"""
Optimization Functions for Category Management

This module provides functions for supplier optimization using Mixed-Integer Linear Programming (MILP).
It includes utilities for creating optimization models, solving supplier selection problems,
and analyzing optimization results.
"""

import pandas as pd
import numpy as np
import pulp
import ast
from typing import List, Dict, Tuple, Optional, Any
import warnings
from datetime import datetime
import os
from pulp import (
    LpProblem, LpVariable, LpMinimize, lpSum, LpBinary, 
    LpContinuous, LpStatus, value, LpMaximize, LpAffineExpression
)
import math
# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Common column names
DEFAULT_COLUMNS = {
    'item_id': 'Entity--Item',
    'vendor': 'vgn_name',
    'description': 'All Descriptions',
    'fallback_description': 'description',
    'attributes': 'openai_response',
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

# Optimization model configuration
OPTIMIZATION_CONFIG = {
    'time_limit': 300,  # 5 minutes
    'gap_rel': 0.01,    # 1% gap
    'gap_abs': 1000     # $1000 absolute gap
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
# DATA PREPARATION FUNCTIONS
# =============================================================================

def prepare_optimization_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for optimization by cleaning and validating data.
    
    Args:
        df: Input DataFrame with supplier data
        
    Returns:
        Cleaned DataFrame ready for optimization
        
    Raises:
        ValidationError: If data validation fails
    """
    validate_dataframe(df, "Input DataFrame")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Fill missing values and ensure numeric types
    numeric_columns = ['sales', 'cogs', 'adj_vol', 'private_label', 'sales_pct']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    return df_clean


def _clean_vendor_names(df: pd.DataFrame, vendor_col: str) -> pd.DataFrame:
    """Clean and standardize vendor names in DataFrame."""
    df_clean = df.copy()
    df_clean[vendor_col] = df_clean[vendor_col].astype(str).str.strip().str.upper().replace({'NAN': 'UNKNOWN'})
    return df_clean


def _filter_valid_matches(df: pd.DataFrame, item_id_col: str, matches_col: str) -> pd.DataFrame:
    """Filter matches to only include valid SKUs that exist in the DataFrame."""
    valid_skus = set(df[item_id_col].dropna().unique())
    df_filtered = df.copy()
    
    for idx, row in df_filtered.iterrows():
        raw_matches_value = row[matches_col]
        filtered_matches = _parse_and_filter_matches(raw_matches_value, valid_skus)
        df_filtered.at[idx, matches_col] = filtered_matches
    
    return df_filtered


def _parse_and_filter_matches(raw_matches_value: Any, valid_skus: set) -> List[str]:
    """Parse and filter matches to only include valid SKUs."""
    if isinstance(raw_matches_value, list):
        return [match for match in raw_matches_value if str(match).strip() in valid_skus]
    elif isinstance(raw_matches_value, str) and raw_matches_value.strip().startswith('[') and raw_matches_value.strip().endswith(']'):
        try:
            parsed_value = ast.literal_eval(raw_matches_value)
            if isinstance(parsed_value, list):
                return [match for match in parsed_value if str(match).strip() in valid_skus]
        except (ValueError, SyntaxError, TypeError):
            pass
    return []


def _create_series_for_optimization(df_indexed_by_item: pd.DataFrame, sorted_skus: List[str], column: str) -> pd.Series:
    """Create a standardized series for optimization calculations."""
    return pd.to_numeric(df_indexed_by_item[column], errors='coerce').reindex(sorted_skus).fillna(0.0).astype(float)


def _apply_rebate_adjustments(cogs_series: pd.Series, df_indexed_by_item: pd.DataFrame, 
                            sorted_skus: List[str], vendor_col: str, rebates: pd.DataFrame,
                            rebate_vendor_col: str, rebate_percent_col: str) -> pd.Series:
    """Apply rebate adjustments to COGS series."""
    if not isinstance(rebates, pd.DataFrame) or rebate_vendor_col not in rebates.columns or rebate_percent_col not in rebates.columns:
        return cogs_series
    
    rb = rebates.copy()
    rb['_VN_'] = rb[rebate_vendor_col].astype(str).str.upper()
    rmap = rb.set_index('_VN_')[rebate_percent_col].to_dict()
    vend_series = df_indexed_by_item[vendor_col].reindex(sorted_skus).fillna('UNKNOWN').astype(str)
    factor = vend_series.map(lambda v: (1 - float(rmap.get(v, 0))) if pd.notna(v) else 1.0)
    return cogs_series * factor.values


def _prepare_base_dataframe(df: pd.DataFrame, item_id_col: str, matches_col: str, adj_vol_col: str, 
                           sales_col: str, cogs_col: str, vendor_col: str) -> pd.DataFrame:
    """Prepare base DataFrame with required columns and Entity column."""
    required_cols = [item_id_col, matches_col, adj_vol_col, sales_col, cogs_col, vendor_col]
    
    if 'Entity' not in df.columns:
        print("Warning: 'Entity' column not found in DataFrame. Assuming all items have Entity == 1.")
        base_df = df[required_cols].copy()
        base_df['Entity'] = 1
    else:
        base_df = df[required_cols + ['Entity']].copy()
    
    return base_df


def _create_empty_optimization_result() -> Tuple[np.ndarray, List[float], List[float], List[float], np.ndarray, List[int], List[str], List[str], Dict[str, int], Dict[str, int], List[int]]:
    """Create empty optimization result structure."""
    return (np.array([]).reshape(0,0), [], [], [], np.array([]).reshape(0,0), [], [], [], {}, {}, [])


def _create_adjacency_matrix_numpy(A_df: pd.DataFrame, sorted_skus: List[str]) -> np.ndarray:
    """Create numpy adjacency matrix from DataFrame."""
    if not A_df.empty:
        A_df = A_df.reindex(index=sorted_skus, columns=sorted_skus, fill_value=0)
        A_np = A_df.to_numpy()
    else:
        A_np = np.zeros((len(sorted_skus), len(sorted_skus)), dtype=int)
    np.fill_diagonal(A_np, 1)
    return A_np


def _create_supplier_matrix(sorted_skus: List[str], sorted_suppliers: List[str], 
                          df_indexed_by_item: pd.DataFrame, vendor_col: str,
                          item_to_idx: Dict[str, int], supplier_to_idx: Dict[str, int]) -> np.ndarray:
    """Create supplier matrix mapping SKUs to suppliers."""
    S_np = np.zeros((len(sorted_skus), len(sorted_suppliers)), dtype=int)
    vendors_for_sorted = df_indexed_by_item[vendor_col].reindex(sorted_skus).fillna('UNKNOWN').astype(str)
    
    for eid, vend in zip(sorted_skus, vendors_for_sorted):
        S_np[item_to_idx[eid], supplier_to_idx.get(vend, 0)] = 1
    
    return S_np


def _process_must_keep_skus(must_keep_item_ids: Optional[List[str]], item_to_idx: Dict[str, int]) -> List[int]:
    """Process must_keep_skus to get their indices."""
    must_keep_sku_indices_list = []
    if must_keep_item_ids:
        for mk in must_keep_item_ids:
            if mk in item_to_idx:
                must_keep_sku_indices_list.append(item_to_idx[mk])
    return must_keep_sku_indices_list


def _create_simple_adjacency_matrix(df: pd.DataFrame, item_id_col: str, matches_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create a simple adjacency matrix for optimization inputs.
    
    This function creates an adjacency matrix that represents which items can be swapped
    with each other based on the matches data. The matrix is symmetric, meaning if item A
    can be swapped with item B, then item B can also be swapped with item A.
    
    Args:
        df: DataFrame with item data containing item_id_col and matches_col
        item_id_col: Column name containing unique item identifiers
        matches_col: Column name containing lists of items that can be swapped
        
    Returns:
        Tuple of (adjacency DataFrame, sorted item IDs)
        
    Raises:
        ValidationError: If input validation fails
    """
    validate_dataframe(df, "df")
    validate_string_input(item_id_col, "item_id_col")
    validate_string_input(matches_col, "matches_col")
    
    # Get all unique items
    all_items = set(df[item_id_col].dropna().unique())
    
    if not all_items:
        print("Warning: No valid items found in DataFrame. Returning empty adjacency matrix.")
        return pd.DataFrame(), []
    
    # Create adjacency matrix
    adjacency_data = {}
    for item in all_items:
        adjacency_data[item] = {}
        for other_item in all_items:
            adjacency_data[item][other_item] = 0
    
    # Fill adjacency matrix based on matches
    for _, row in df.iterrows():
        item = row[item_id_col]
        matches = row[matches_col]
        
        if pd.isna(item) or item not in all_items:
            continue
            
        if isinstance(matches, list):
            for match in matches:
                if str(match).strip() in all_items:
                    adjacency_data[item][str(match).strip()] = 1
                    adjacency_data[str(match).strip()][item] = 1  # Make it symmetric
        elif isinstance(matches, str) and matches.strip().startswith('[') and matches.strip().endswith(']'):
            try:
                parsed_matches = ast.literal_eval(matches)
                if isinstance(parsed_matches, list):
                    for match in parsed_matches:
                        if str(match).strip() in all_items:
                            adjacency_data[item][str(match).strip()] = 1
                            adjacency_data[str(match).strip()][item] = 1  # Make it symmetric
            except (ValueError, SyntaxError, TypeError):
                print(f"Warning: Could not parse matches string for item {item}: {matches}")
                continue
    
    # Create DataFrame
    adjacency_df = pd.DataFrame(adjacency_data)
    sorted_items = sorted(all_items)
    
    return adjacency_df, sorted_items

# =============================================================================
# ANALYSIS AND REPORTING FUNCTIONS
# =============================================================================

def analyze_optimization_results(df: pd.DataFrame, 
                               results: Dict) -> pd.DataFrame:
    """
    Analyze optimization results and create detailed report.
    
    Args:
        df: Original supplier DataFrame
        results: Optimization results dictionary
        
    Returns:
        DataFrame with detailed analysis
    """
    validate_dataframe(df, "Supplier DataFrame")
    
    if not results['selected_suppliers']:
        return pd.DataFrame()
    
    # Filter DataFrame to selected suppliers
    selected_df = df[df['vendor'].isin(results['selected_suppliers'])].copy()
    
    # Calculate additional metrics
    selected_df['optimization_rank'] = range(1, len(selected_df) + 1)
    selected_df['sales_contribution'] = selected_df['sales'] / selected_df['sales'].sum()
    
    return selected_df

def create_optimization_summary(df: pd.DataFrame, 
                              results: Dict) -> Dict:
    """
    Create a summary of optimization results.
    
    Args:
        df: Original supplier DataFrame
        results: Optimization results dictionary
        
    Returns:
        Dictionary with optimization summary
    """
    validate_dataframe(df, "Supplier DataFrame")
    
    summary = {
        'total_suppliers_analyzed': len(df),
        'suppliers_selected': len(results['selected_suppliers']),
        'optimization_status': results['status'],
        'total_sales': results['objective_value'],
        'selected_supplier_list': results['selected_suppliers'],
        'timestamp': datetime.now().isoformat()
    }
    
    return summary

# =============================================================================
# EXCEL EXPORT FUNCTIONS
# =============================================================================

def export_optimization_results(summary: Dict,
                              detailed_results: pd.DataFrame,
                              output_path: str) -> None:
    """
    Export optimization results to Excel file.
    
    Args:
        summary: Optimization summary dictionary
        detailed_results: Detailed results DataFrame
        output_path: Path for output Excel file
        
    Raises:
        ValidationError: If output path is invalid
    """
    validate_string_input(output_path, "Output path")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with pd.ExcelWriter(output_path, **EXCEL_CONFIG) as writer:
        # Write summary to first sheet
        summary_df = pd.DataFrame([summary])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Write detailed results to second sheet
        if not detailed_results.empty:
            detailed_results.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_supplier_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional metrics for suppliers.
    
    Args:
        df: DataFrame with supplier data
        
    Returns:
        DataFrame with additional metrics
    """
    validate_dataframe(df, "Supplier DataFrame")
    
    df_metrics = df.copy()
    
    # Calculate profit margin
    if 'sales' in df_metrics.columns and 'cogs' in df_metrics.columns:
        df_metrics['profit_margin'] = (df_metrics['sales'] - df_metrics['cogs']) / df_metrics['sales']
        df_metrics['profit_margin'] = df_metrics['profit_margin'].fillna(0)
    
    # Calculate sales rank
    if 'sales' in df_metrics.columns:
        df_metrics['sales_rank'] = df_metrics['sales'].rank(ascending=False)
    
    return df_metrics

def filter_suppliers_by_criteria(df: pd.DataFrame,
                               min_sales: float = 0.0,
                               min_profit_margin: float = 0.0,
                               exclude_private_label: bool = False) -> pd.DataFrame:
    """
    Filter suppliers based on specified criteria.
    
    Args:
        df: DataFrame with supplier data
        min_sales: Minimum sales threshold
        min_profit_margin: Minimum profit margin threshold
        exclude_private_label: Whether to exclude private label suppliers
        
    Returns:
        Filtered DataFrame
    """
    validate_dataframe(df, "Supplier DataFrame")
    
    df_filtered = df.copy()
    
    # Apply filters
    if min_sales > 0 and 'sales' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['sales'] >= min_sales]
    
    if min_profit_margin > 0 and 'profit_margin' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['profit_margin'] >= min_profit_margin]
    
    if exclude_private_label and 'private_label' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['private_label'] == 0]
    
    return df_filtered


# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================

def prepare_optimization_inputs(
    im_final_full_df: pd.DataFrame,
    must_keep_item_ids: Optional[List[str]] = None,
    rebates: Optional[pd.DataFrame] = None,
    item_id_col: str = DEFAULT_COLUMNS['item_id'],
    matches_col: str = DEFAULT_COLUMNS['matches'],
    adj_vol_col: str = DEFAULT_COLUMNS['adj_vol'],
    sales_col: str = DEFAULT_COLUMNS['sales'],
    cogs_col: str = DEFAULT_COLUMNS['cogs'],
    vendor_col: str = DEFAULT_COLUMNS['vendor'],
    rebate_vendor_col: str = 'VGN Name',
    rebate_percent_col: str = '%OF GROSS'
) -> Tuple[np.ndarray, List[float], List[float], List[float], np.ndarray, List[int], List[str], List[str], Dict[str, int], Dict[str, int], List[int]]:
    """
    Prepare inputs for supplier optimization problem.
    
    Builds adjacency matrix A, volume list v, sales/COGS lists, supplier matrix S,
    and various mappings and indices needed for optimization.
    
    Args:
        im_final_full_df: Input DataFrame with item data
        must_keep_item_ids: Optional list of item IDs that must retain their volume
        rebates: Optional DataFrame with rebate information
        item_id_col: Column name for item IDs
        matches_col: Column name for matches
        adj_vol_col: Column name for adjusted volume
        sales_col: Column name for sales data
        cogs_col: Column name for COGS data
        vendor_col: Column name for vendor information
        rebate_vendor_col: Column name for vendor in rebates DataFrame
        rebate_percent_col: Column name for rebate percentage
    
    Returns:
        Tuple of optimization inputs: (A_np, v_list, sales_list, cogs_list, S_np, 
        must_keep_sku_indices_list, sorted_skus, sorted_suppliers, item_to_idx, supplier_to_idx, entity_list)
    """
    validate_dataframe(im_final_full_df, "im_final_full_df")

    # Validate required columns
    required = [item_id_col, matches_col, adj_vol_col, sales_col, cogs_col, vendor_col]
    missing = [c for c in required if c not in im_final_full_df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {', '.join(missing)}")

    # Prepare base DataFrame with Entity column
    df = _prepare_base_dataframe(im_final_full_df, item_id_col, matches_col, adj_vol_col, sales_col, cogs_col, vendor_col)
    
    # Clean vendor names
    df = _clean_vendor_names(df, vendor_col)

    # Filter matches to only include valid SKUs
    df_filtered = _filter_valid_matches(df, item_id_col, matches_col)
    
    A_df, sorted_skus = _create_simple_adjacency_matrix(df_filtered, item_id_col, matches_col)
    if not sorted_skus:
        print("Warning: No SKUs found or processed from the DataFrame. Optimization inputs might be empty.")
        return _create_empty_optimization_result()

    # Create adjacency matrix numpy array
    A_np = _create_adjacency_matrix_numpy(A_df, sorted_skus)
    item_to_idx = {item_id: i for i, item_id in enumerate(sorted_skus)}

    # Prepare DataFrame for lookups
    df_indexed_by_item = df.drop_duplicates(subset=[item_id_col]).set_index(item_id_col)

    # Create optimization series
    v_series = _create_series_for_optimization(df_indexed_by_item, sorted_skus, adj_vol_col)
    sales_series = _create_series_for_optimization(df_indexed_by_item, sorted_skus, sales_col)
    cogs_series = _create_series_for_optimization(df_indexed_by_item, sorted_skus, cogs_col)

    # Create entity list
    entity_series = pd.to_numeric(df_indexed_by_item['Entity'], errors='coerce').reindex(sorted_skus).fillna(1).astype(int)
    entity_list = entity_series.tolist()

    # Apply rebate adjustments to COGS
    cogs_series = _apply_rebate_adjustments(cogs_series, df_indexed_by_item, sorted_skus, vendor_col, 
                                          rebates, rebate_vendor_col, rebate_percent_col)

    # Convert to lists
    v_list = v_series.tolist()
    sales_list = sales_series.tolist()
    cogs_list = cogs_series.tolist()

    # Create supplier matrix
    sorted_suppliers = sorted(df[vendor_col].unique().tolist())
    supplier_to_idx = {supplier_name: i for i, supplier_name in enumerate(sorted_suppliers)}
    S_np = _create_supplier_matrix(sorted_skus, sorted_suppliers, df_indexed_by_item, vendor_col, item_to_idx, supplier_to_idx)

    print(f"Supplier matrix S shape: {S_np.shape}")
    
    # Process must_keep_skus
    must_keep_sku_indices_list = _process_must_keep_skus(must_keep_item_ids, item_to_idx)

    return (A_np, v_list, sales_list, cogs_list,
            S_np, must_keep_sku_indices_list, 
            sorted_skus, sorted_suppliers, item_to_idx, supplier_to_idx, entity_list)

def solve_supplier_optimization_looped(
    sku_ids: List[str],
    supplier_ids: List[str],
    sku_volumes_list: List[float],
    swap_feasibility_matrix: np.ndarray,
    sku_supplier_matrix: np.ndarray,
    beta_volume_factor: float,
    sku_total_sales_list: List[float],
    sku_total_cogs_list: List[float],
    must_keep_sku_ids: Optional[List[str]] = None,
    entity_list: Optional[List[int]] = None
) -> Dict:
    """
    Solve supplier optimization iteratively to maximize GM dollars with decreasing supplier limits.
    
    Args:
        sku_ids: List of unique SKU identifiers
        supplier_ids: List of unique supplier identifiers
        sku_volumes_list: List of volumes for each SKU
        swap_feasibility_matrix: 2D array for swap feasibility
        sku_supplier_matrix: 2D array mapping SKUs to suppliers
        beta_volume_factor: Multiplier for volume redistribution limits
        sku_total_sales_list: List of sales for each SKU
        sku_total_cogs_list: List of COGS for each SKU
        must_keep_sku_ids: Optional list of SKUs that must retain their volume
        entity_list: Optional list of Entity values for each SKU
    
    Returns:
        DataFrame with columns: Max_Suppliers_Allowed, Actual_Active_Suppliers, 
        and Achieved_Gross_Margin_Dollars showing the trade-off
    """
    print("--- Initializing Looped Optimization Problem ---")

    # Data Validation and Preparation (similar to original, but using copies)
    sku_ids_copy = sku_ids.copy()
    supplier_ids_copy = supplier_ids.copy()
    sku_volumes_list_copy = sku_volumes_list.copy()
    swap_feasibility_matrix_copy = swap_feasibility_matrix.copy()
    sku_supplier_matrix_copy = sku_supplier_matrix.copy()
    sku_total_sales_list_copy = sku_total_sales_list.copy()
    sku_total_cogs_list_copy = sku_total_cogs_list.copy()
    entity_list_copy = entity_list.copy() if entity_list else None

    output_columns = ['Max_Suppliers_Allowed', 'Actual_Active_Suppliers', 'Achieved_Gross_Margin_Dollars']

    # Initial data validation is crucial for the loop to run correctly
    if not sku_ids_copy or not supplier_ids_copy or not sku_volumes_list_copy:
        print("Error: Input lists cannot be empty.")
        return pd.DataFrame(columns=output_columns)
    if len(sku_volumes_list_copy) != len(sku_ids_copy) or \
       len(sku_total_sales_list_copy) != len(sku_ids_copy) or \
       len(sku_total_cogs_list_copy) != len(sku_ids_copy):
        print("Error: Length mismatch between SKU related lists.")
        return pd.DataFrame(columns=output_columns)
    
    num_total_suppliers = len(supplier_ids_copy)
    results_list = []
    current_max_allowed_suppliers = num_total_suppliers

    print(f"Starting optimization loop. Initial max suppliers: {current_max_allowed_suppliers}")

    while True:
        if current_max_allowed_suppliers < 1:
            print("Supplier limit fell below 1. Stopping.")
            break

        print(f"\n--- Solving for Max Suppliers <= {current_max_allowed_suppliers} ---")

        status, _, actual_active_suppliers, _, achieved_gm_dollars, _ = solve_supplier_optimization_zero(
            sku_ids=sku_ids_copy,
            supplier_ids=supplier_ids_copy,
            sku_volumes_list=sku_volumes_list_copy,
            swap_feasibility_matrix=swap_feasibility_matrix_copy,
            sku_supplier_matrix=sku_supplier_matrix_copy,
            beta_volume_factor=beta_volume_factor,
            sku_total_sales_list=sku_total_sales_list_copy,
            sku_total_cogs_list=sku_total_cogs_list_copy,
            must_keep_sku_ids=must_keep_sku_ids,
            max_suppliers_limit=current_max_allowed_suppliers,
            objective_type='maximize_gm_dollars', # This objective remains constant for the loop
            entity_list=entity_list_copy
        )

        if status == "Optimal":
            results_list.append({
                'Max_Suppliers_Allowed': current_max_allowed_suppliers,
                'Actual_Active_Suppliers': actual_active_suppliers,
                'Achieved_Gross_Margin_Dollars': achieved_gm_dollars
            })

            if current_max_allowed_suppliers == 1:
                print("Reached supplier limit of 1 and found an optimal solution. Stopping loop.")
                break

            # Determine the next supplier limit
            next_max_suppliers_float = current_max_allowed_suppliers * 0.98
            next_max_suppliers = max(1, math.floor(next_max_suppliers_float))

            if next_max_suppliers >= current_max_allowed_suppliers and current_max_allowed_suppliers > 1:
                next_max_suppliers = current_max_allowed_suppliers - 1

            current_max_allowed_suppliers = next_max_suppliers

        else:
            print(f"Optimization became non-optimal (status: {status}) at max supplier limit: {current_max_allowed_suppliers}. Stopping loop.")
            break

    if not results_list:
        print("No optimal solutions found in any iteration.")
        return pd.DataFrame(columns=output_columns)

    results_df = pd.DataFrame(results_list)
    return results_df.sort_values(by='Max_Suppliers_Allowed', ascending=False).reset_index(drop=True)

def solve_supplier_optimization_zero(
    sku_ids: List[str],
    supplier_ids: List[str],
    sku_volumes_list: List[float],
    swap_feasibility_matrix: np.ndarray,
    sku_supplier_matrix: np.ndarray,
    beta_volume_factor: float,
    sku_total_sales_list: List[float],
    sku_total_cogs_list: List[float],
    must_keep_sku_ids: Optional[List[str]] = None,
    max_suppliers_limit: Optional[int] = None,
    objective_type: str = 'maximize_gm_dollars',
    entity_list: Optional[List[int]] = None
) -> Tuple[str, Dict, int, float, float, Dict]:
    """
    Solve supplier optimization using compact MILP with all-or-nothing 1:1 swaps.
    
    Args:
        sku_ids: List of unique SKU identifiers
        supplier_ids: List of unique supplier identifiers
        sku_volumes_list: List of volumes for each SKU
        swap_feasibility_matrix: 2D array for swap feasibility
        sku_supplier_matrix: 2D array mapping SKUs to suppliers
        beta_volume_factor: Multiplier for max allowed final volume per SKU
        sku_total_sales_list: List of sales for each SKU
        sku_total_cogs_list: List of COGS for each SKU
        must_keep_sku_ids: Optional list of SKUs that cannot be zeroed out
        max_suppliers_limit: Optional cap on active suppliers
        objective_type: 'maximize_gm_dollars' or 'minimize_suppliers'
        entity_list: Optional list of Entity values for each SKU
    
    Returns:
        Tuple of (status, active_suppliers_dict, num_active_suppliers_count, 
        achieved_gm_dollars, achieved_gm_percent, volume_redistribution_dict)
    """
    print("--- Initializing Optimization Problem ---")

    # --- Data Validation and Preparation ---
    if not sku_ids:
        print("Error: sku_ids list is empty.")
        return "Error", {}, 0, {}, 0.0, None
    if not supplier_ids:
        print("Error: supplier_ids list is empty.")
        return "Error", {}, 0, {}, 0.0, None
    if not isinstance(sku_volumes_list, list) or not sku_volumes_list:
        print("Error: sku_volumes_list must be a non-empty list.")
        return "Error", {}, 0, {}, 0.0, None
    if len(sku_volumes_list) != len(sku_ids):
        print(f"Error: Length of sku_volumes_list ({len(sku_volumes_list)}) must match length of sku_ids ({len(sku_ids)}).")
        return "Error", {}, 0, {}, 0.0, None
    if len(sku_total_sales_list) != len(sku_ids):
        print(f"Error: Length of sku_total_sales_list ({len(sku_total_sales_list)}) must match length of sku_ids ({len(sku_ids)}).")
        return "Error", {}, 0, {}, 0.0, None
    if len(sku_total_cogs_list) != len(sku_ids):
        print(f"Error: Length of sku_total_cogs_list ({len(sku_total_cogs_list)}) must match length of sku_ids ({len(sku_ids)}).")
        return "Error", {}, 0, {}, 0.0, None

    # Handle entity_list - if not provided, assume all SKUs have Entity == 1
    if entity_list is None:
        print("Warning: entity_list not provided. Assuming all SKUs have Entity == 1.")
        entity_list = [1] * len(sku_ids)
    elif len(entity_list) != len(sku_ids):
        print(f"Error: Length of entity_list ({len(entity_list)}) must match length of sku_ids ({len(sku_ids)}).")
        return "Error", {}, 0, {}, 0.0, None

    num_skus = len(sku_ids)
    num_suppliers = len(supplier_ids)

    # Convert inputs to NumPy arrays for efficiency and consistent access
    v_np = np.array(sku_volumes_list, dtype=float)
    A_np = np.array(swap_feasibility_matrix, dtype=int)
    S_np = np.array(sku_supplier_matrix, dtype=int)
    sales_np = np.array(sku_total_sales_list, dtype=float)
    cogs_np = np.array(sku_total_cogs_list, dtype=float)
    entity_np = np.array(entity_list, dtype=int)

    v_np[np.isnan(v_np)] = 0.0
    sales_np[np.isnan(sales_np)] = 0.0
    cogs_np[np.isnan(cogs_np)] = 0.0

    # Calculate profit per unit for each SKU
    profit_per_unit_array = np.zeros(num_skus, dtype=float)
    for i in range(num_skus):
        if v_np[i] > 1e-6: # Avoid division by zero for original volume
            profit_per_unit_array[i] = (sales_np[i] - cogs_np[i]) / v_np[i]
        else:
            profit_per_unit_array[i] = 0.0

    # Validate matrix dimensions
    if A_np.shape != (num_skus, num_skus):
        print(f"Error: swap_feasibility_matrix dimensions ({A_np.shape}) do not match SKUs ({num_skus}x{num_skus}).")
        return "Error", {}, 0, {}, 0.0, None
    if S_np.shape != (num_skus, num_suppliers):
        print(f"Error: sku_supplier_matrix dimensions ({S_np.shape}) do not match SKUs x Suppliers ({num_skus}x{num_suppliers}).")
        return "Error", {}, 0, {}, 0.0, None

    # Create mappings from string IDs to integer indices
    sku_ids_str = [str(s_id) for s_id in sku_ids]
    supplier_ids_str = [str(s_id) for s_id in supplier_ids]

    sku_to_idx = {sku_id_str: idx for idx, sku_id_str in enumerate(sku_ids_str)}
    supplier_to_idx = {supp_id_str: idx for idx, supp_id_str in enumerate(supplier_ids_str)}

    # In this simplified version supplier ids are positional; S columns align with supplier_ids_str
    def resolve_supplier_col_index(s_id_str: str) -> int:
        return supplier_to_idx[s_id_str]

    # Large constant M for Big-M constraints
    total_system_volume = np.sum(v_np)
    M = total_system_volume * 100 if total_system_volume > 0 else 1.0 # Use a larger M
    if M < 1e6: M = 1e6 # Ensure M is sufficiently large even for small volumes

    print(f"SKUs (count: {num_skus}): {sku_ids_str[:3]}..." if num_skus > 3 else sku_ids_str)
    print(f"Suppliers (count: {num_suppliers}): {supplier_ids_str[:3]}..." if num_suppliers > 3 else supplier_ids_str)
    print(f"Max volume factor (beta): {beta_volume_factor}")
    print(f"Big M for constraints: {M}")

    # --- Define Problem ---
    prob = LpProblem("Supplier_Optimization_Problem", LpMinimize if objective_type == 'minimize_suppliers' else LpMaximize)

    # --- Decision Variables (compact) ---
    feasible_non_self_swap_pairs_ids = []
    for i_idx in range(num_skus):
        for j_idx in range(num_skus):
            if i_idx != j_idx and A_np[i_idx, j_idx] == 1:
                feasible_non_self_swap_pairs_ids.append((sku_ids_str[i_idx], sku_ids_str[j_idx]))

    z = LpVariable.dicts("z_sku_zero_out", sku_ids_str, cat=LpBinary)
    x = LpVariable.dicts("x_swap_one_to_one", feasible_non_self_swap_pairs_ids, cat=LpBinary)
    y = LpVariable.dicts("y_supplier_active", supplier_ids_str, cat=LpBinary)

    # Helper: final volume expression per receiving SKU j
    def final_volume_expr_for(j_str: str) -> LpAffineExpression:
        j_idx = sku_to_idx[j_str]
        retain_part = v_np[j_idx] * (1 - z[j_str])
        inbound_part = lpSum(v_np[sku_to_idx[i_str]] * x[(i_str, j_str)]
                             for i_str in sku_ids_str
                             if i_str != j_str and (i_str, j_str) in x)
        return retain_part + inbound_part

    # --- Objective Function ---
    if objective_type == 'minimize_suppliers':
        prob += lpSum(y[s_id] for s_id in supplier_ids_str), "Minimize_Suppliers"
    elif objective_type == 'maximize_gm_dollars':
        prob += lpSum(profit_per_unit_array[sku_to_idx[j_str]] * final_volume_expr_for(j_str)
                      for j_str in sku_ids_str), "Maximize_Total_Gross_Margin_Dollars"
    else:
        print(f"Error: Invalid objective_type '{objective_type}'. Must be 'minimize_suppliers' or 'maximize_gm_dollars'.")
        return "Error", {}, 0, {}, 0.0, None

    print(f"Objective function set to {objective_type}.")

    # --- Constraints ---
    # 1. 1:1 all-or-nothing swap selection per origin i
    for i_str in sku_ids_str:
        prob += lpSum(x[(i_str, j_str)] for j_str in sku_ids_str if (i_str, j_str) in x) == z[i_str], \
                 f"One_To_One_Swap_If_Zero_Out_SKU_{str(i_str).replace(' ', '_')}"

    # 2. Must-keep SKUs cannot be zeroed out
    if must_keep_sku_ids:
        processed_must_keep_sku_ids_str = [str(s_id) for s_id in must_keep_sku_ids]
        for k_sku_str in processed_must_keep_sku_ids_str:
            if k_sku_str in sku_ids_str:
                prob += z[k_sku_str] == 0, f"Must_Keep_SKU_{str(k_sku_str).replace(' ', '_')}_Not_Zero_Out"
            else:
                print(f"Warning: Must-keep SKU ID '{k_sku_str}' not found in sku_ids. Ignoring.")

    # 3. Volume caps per receiving SKU
    print("Adding Volume Redistribution Limit constraints...")
    max_cap_by_j = {}
    for j_str in sku_ids_str:
        j_idx = sku_to_idx[j_str]
        original_volume_j = v_np[j_idx]
        # Compute cap
        if original_volume_j < 1e-5 and beta_volume_factor > 0:
            non_zero_volumes = v_np[v_np > 1e-5]
            avg_vol = np.mean(non_zero_volumes) if non_zero_volumes.size > 0 else 0
            if avg_vol == 0:
                max_allowed_volume_j = total_system_volume * 0.1 if total_system_volume > 0 else 100
            else:
                max_allowed_volume_j = beta_volume_factor * avg_vol
            max_allowed_volume_j = max(max_allowed_volume_j, 1e-5)
            max_allowed_volume_j = min(max_allowed_volume_j, total_system_volume) if total_system_volume > 0 else max_allowed_volume_j
            print(f"    Note: SKU {j_str} has ~0 original volume. Max redistributed volume capped at {max_allowed_volume_j:.2f}.")
        elif original_volume_j > 0:
            if beta_volume_factor == 0:
                max_allowed_volume_j = original_volume_j
            elif beta_volume_factor > 0:
                max_allowed_volume_j = beta_volume_factor * original_volume_j
            else:
                max_allowed_volume_j = original_volume_j
        else:
            max_allowed_volume_j = 0.0

        prob += final_volume_expr_for(j_str) <= max_allowed_volume_j, \
                 f"Max_Volume_For_SKU_{str(j_str).replace(' ', '_')}"
        max_cap_by_j[j_str] = max_allowed_volume_j

    # 4. Supplier activation linkage
    print("Adding Supplier Activation constraints (per-SKU linking)...")
    # Map each SKU to its supplier column index (assume one-hot S row)
    sku_to_supplier_col = {}
    for j_str in sku_ids_str:
        j_idx = sku_to_idx[j_str]
        supplier_cols = np.where(S_np[j_idx, :] == 1)[0]
        if supplier_cols.size > 0:
            sku_to_supplier_col[j_str] = int(supplier_cols[0])
        else:
            # No supplier assigned; skip linking for this SKU
            continue
        s_idx = sku_to_supplier_col[j_str]
        s_id_str = supplier_ids_str[s_idx]
        # If any volume appears at SKU j, y[s_j] must be 1
        prob += final_volume_expr_for(j_str) <= max_cap_by_j[j_str] * y[s_id_str], \
                 f"Supplier_On_If_Volume_At_SKU_{str(j_str).replace(' ', '_')}"
        # Retention implies supplier active
        prob += (1 - z[j_str]) <= y[s_id_str], \
                 f"Supplier_On_If_Retain_SKU_{str(j_str).replace(' ', '_')}"
        # Any inbound x to j implies supplier active
        for i_str in sku_ids_str:
            if i_str != j_str and (i_str, j_str) in x:
                prob += x[(i_str, j_str)] <= y[s_id_str], \
                         f"Supplier_On_If_Inbound_{str(i_str).replace(' ', '_')}_to_{str(j_str).replace(' ', '_')}"

    # 5. SKU can either give or receive volume, but not both
    print("Adding SKU Give-or-Receive constraints...")
    for i_str in sku_ids_str:
        # Check if SKU i gives volume (z[i] = 1 means it gives all its volume)
        gives_volume = z[i_str]
        # Check if SKU i receives volume (any x[k,i] where k != i)
        receives_volume = lpSum(x[(k_str, i_str)] for k_str in sku_ids_str 
                               if k_str != i_str and (k_str, i_str) in x)
        # Constraint: cannot both give and receive
        prob += gives_volume + receives_volume <= 1, \
                 f"SKU_{str(i_str).replace(' ', '_')}_Cannot_Give_And_Receive"

    # 6. Optional: limit number of active suppliers
    if max_suppliers_limit is not None:
        prob += lpSum(y[s_id] for s_id in supplier_ids_str) <= max_suppliers_limit, \
                 f"Supplier_Count_Limit_{max_suppliers_limit}"

    # 7. Entity constraint: SKUs being swapped TO must have Entity == 1
    print("Adding Entity constraint...")
    for j_str in sku_ids_str:
        j_idx = sku_to_idx[j_str]
        if entity_np[j_idx] != 1:
            # If SKU j has Entity != 1, it cannot receive any swaps
            prob += lpSum(x[(i_str, j_str)] for i_str in sku_ids_str if (i_str, j_str) in x) == 0, \
                     f"Entity_Constraint_SKU_{str(j_str).replace(' ', '_')}"

    # --- Solve Problem ---
    print("\n--- Solving the Optimization Problem ---")
    prob.solve()

    # --- Process and Return Results ---
    solution_status = LpStatus[prob.status]
    print(f"\n--- Optimization Results ---")
    print(f"Status: {solution_status}")

    active_suppliers_dict = {}
    num_active_suppliers_count = 0
    volume_redistribution_dict = {}
    achieved_gm_dollars = 0.0

    if solution_status == "Optimal":
        # Build transfers and final volumes from x and z
        tolerance = 1e-5
        # Transfers
        for (i_str, j_str) in feasible_non_self_swap_pairs_ids:
            x_val = value(x[(i_str, j_str)])
            if x_val is not None and x_val > 0.5:
                transfer_amount = float(v_np[sku_to_idx[i_str]])
                if transfer_amount > tolerance:
                    volume_redistribution_dict[(i_str, j_str)] = transfer_amount

        # Final volumes and GM
        final_volume_at = {}
        for j_str in sku_ids_str:
            z_val = value(z[j_str]) or 0.0
            retain_part = v_np[sku_to_idx[j_str]] * (1 - z_val)
            inbound = 0.0
            for i_str in sku_ids_str:
                if i_str != j_str and (i_str, j_str) in x:
                    x_val = value(x[(i_str, j_str)]) or 0.0
                    if x_val > 0.5:
                        inbound += v_np[sku_to_idx[i_str]]
            final_volume = retain_part + inbound
            final_volume_at[j_str] = final_volume
            achieved_gm_dollars += profit_per_unit_array[sku_to_idx[j_str]] * final_volume

        # Count active suppliers
        for s_id in supplier_ids_str:
            y_val = value(y[s_id]) or 0.0
            active_suppliers_dict[s_id] = 1.0 if y_val > 0.5 else 0.0
            if y_val > 0.5:
                num_active_suppliers_count += 1

        print(f"Total number of active suppliers: {num_active_suppliers_count}")
    else:
        print("No optimal solution found or problem was infeasible/unbounded.")
        for s_id in supplier_ids_str:
            active_suppliers_dict[s_id] = 0.0

    return solution_status, active_suppliers_dict, num_active_suppliers_count, volume_redistribution_dict, achieved_gm_dollars, prob

def create_symmetric_adjacency_matrix(
    df: pd.DataFrame,
    item_id_col: str = DEFAULT_COLUMNS['item_id'],
    matches_col: str = DEFAULT_COLUMNS['matches']
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs a symmetric adjacency matrix from a DataFrame that specifies item-to-item matches.
    The function first identifies all unique items from both the primary item column and all
    the match lists. It then builds a square matrix where a '1' indicates a connection
    between two items. The connection is always symmetric (if A matches B, B is set to match A).
    The function is robust to match lists being stored as either actual lists or as string
    representations of lists.

    Args:
        df (pd.DataFrame): The input DataFrame containing the item data and their corresponding matches.
        item_id_col (str): The name of the column in `df` that contains the primary item identifiers.
        matches_col (str): The name of the column in `df` that contains the lists of matching or connected items.

    Returns:
        tuple: A tuple containing two elements:
               - pd.DataFrame: A symmetric adjacency matrix where the index and columns are the sorted
                               unique item IDs. A value of 1 indicates a connection.
               - list: A sorted list of all unique item IDs found, which corresponds to the matrix's
                       index and columns.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Input DataFrame is empty or not a Pandas DataFrame.")
        return pd.DataFrame(), []
    if item_id_col not in df.columns:
        print(f"Error: Item ID column '{item_id_col}' not found in DataFrame.")
        return pd.DataFrame(), []
    if matches_col not in df.columns:
        print(f"Error: Matches column '{matches_col}' not found in DataFrame.")
        return pd.DataFrame(), []

    all_items = set()
    # Temporarily store processed matches to avoid re-parsing
    processed_matches_for_all_items = {} 

    for index, row in df.iterrows():
        source_item = row[item_id_col]
        if pd.notna(source_item): # Ensure source_item is not NaN
            all_items.add(source_item)
        
        raw_matches_value = row[matches_col]
        current_item_matches_list = []

        if isinstance(raw_matches_value, list): 
            current_item_matches_list = raw_matches_value
        elif isinstance(raw_matches_value, str) and raw_matches_value.strip().startswith('[') and raw_matches_value.strip().endswith(']'):
            try:
                # Safely evaluate the string as a Python literal (e.g., a list)
                parsed_value = ast.literal_eval(raw_matches_value)
                if isinstance(parsed_value, list):
                    current_item_matches_list = parsed_value
            except (ValueError, SyntaxError, TypeError):
                pass # Keep current_item_matches_list as []
        
        processed_matches_for_all_items[source_item] = current_item_matches_list # Store for second pass

        for matched_item in current_item_matches_list:
            if pd.notna(matched_item): # Ensure matched_item is not NaN
                all_items.add(str(matched_item).strip()) # Ensure items added to set are strings and stripped

    if not all_items:
        print("No items found to create an adjacency matrix.")
        return pd.DataFrame(), []

    sorted_unique_items = sorted(list(all_items))
    item_to_index = {item_id: i for i, item_id in enumerate(sorted_unique_items)}
    num_items = len(sorted_unique_items)
    adj_matrix_np = np.zeros((num_items, num_items), dtype=int)

    for source_item, matches_list in processed_matches_for_all_items.items():
        if pd.isna(source_item): # Should have been filtered by all_items logic, but defensive
            continue
            
        idx_source = item_to_index.get(source_item)
        if idx_source is None: 
            continue

        for matched_item_raw in matches_list:
            if pd.isna(matched_item_raw): 
                continue 
            
            matched_item = str(matched_item_raw).strip() # Ensure it's a string and stripped
            idx_target = item_to_index.get(matched_item)
            
            if idx_target is None: 
                continue
            
            if idx_source != idx_target: 
                adj_matrix_np[idx_source, idx_target] = 1
                adj_matrix_np[idx_target, idx_source] = 1
    
    adj_matrix_df = pd.DataFrame(adj_matrix_np, index=sorted_unique_items, columns=sorted_unique_items)
    return adj_matrix_df, sorted_unique_items

# =============================================================================
# POST OPTIMIZATION UPDATE
# =============================================================================

def update_dataframe_with_new_volume(
    im_final_df2: pd.DataFrame,
    V_solution_values: Dict,
    skus_order: List[str],
    item_to_idx_map: Dict[str, int],
    original_volume_col: str = DEFAULT_COLUMNS['adj_vol'],
    item_id_col: str = DEFAULT_COLUMNS['item_id'],
    suggest_volume_col_name: str = 'suggest_volume'
) -> pd.DataFrame:
    """
    Add suggested volume column to DataFrame based on optimization results.
    
    Calculates: suggest_volume = original_volume - volume_transferred_out + volume_transferred_in
    
    Args:
        im_final_df: Original DataFrame with item_id_col and original_volume_col
        V_solution_values: Solved numerical values for volume redistribution V[from_idx, to_idx]
        skus_order: List of Entity--Item IDs defining PuLP index order
        item_to_idx_map: Mapping from Entity--Item ID string to PuLP integer index
        original_volume_col: Column name with original SKU volumes
        item_id_col: Column name containing Entity--Item IDs
        suggest_volume_col_name: Name for new suggest_volume column
    
    Returns:
        DataFrame with suggest_volume column added
        
    Raises:
        ValidationError: If input validation fails
    """
    # Validate inputs
    _validate_volume_update_inputs(im_final_df2, V_solution_values, skus_order, item_to_idx_map, 
                                 item_id_col, original_volume_col)
    
    im_final_df = im_final_df2.copy() # Avoid modifying the original DataFrame
    
    # Initialize suggest_volume column
    im_final_df = _initialize_suggest_volume_column(im_final_df, original_volume_col, suggest_volume_col_name)
    
    tolerance = 1e-5
    idx_to_item_map = {v: k for k, v in item_to_idx_map.items()}

    # Process volume transfers
    for (from_pulp_idx, to_pulp_idx), transferred_amount in V_solution_values.items():
        if transferred_amount is None or transferred_amount <= tolerance:
            continue

        from_item_id = idx_to_item_map.get(int(from_pulp_idx))
        to_item_id = idx_to_item_map.get(int(to_pulp_idx))

        if from_item_id is None or to_item_id is None:
            print(f"  Warning (SuggestVol): Could not map PuLP indices ({from_pulp_idx}, {to_pulp_idx}) to item IDs. Skipping this transfer.")
            continue
        
        if from_pulp_idx != to_pulp_idx: # This is an actual transfer between different SKUs
            im_final_df = _process_volume_transfer(im_final_df, from_item_id, to_item_id, 
                                                 transferred_amount, item_id_col, suggest_volume_col_name)

    # Normalize vendor names to match optimization supplier universe
    if 'vgn_name' in im_final_df.columns:
        im_final_df['vgn_name'] = im_final_df['vgn_name'].apply(lambda v: str(v).strip().upper() if pd.notna(v) else 'UNKNOWN')

    return im_final_df


def _validate_volume_update_inputs(im_final_df: pd.DataFrame, V_solution_values: Dict, 
                                 skus_order: List[str], item_to_idx_map: Dict[str, int],
                                 item_id_col: str, original_volume_col: str) -> None:
    """Validate inputs for volume update function."""
    validate_dataframe(im_final_df, "im_final_df")
    
    if not isinstance(V_solution_values, dict):
        raise ValidationError("'V_solution_values' must be a dictionary of solved numerical values.")
    if not isinstance(skus_order, list):
        raise ValidationError("'skus_order' must be a list.")
    if not isinstance(item_to_idx_map, dict):
        raise ValidationError("'item_to_idx_map' must be a dictionary.")
    
    validate_columns_exist(im_final_df, [item_id_col, original_volume_col], "im_final_df")


def _initialize_suggest_volume_column(df: pd.DataFrame, original_volume_col: str, suggest_volume_col_name: str) -> pd.DataFrame:
    """Initialize the suggest_volume column from the original volume column."""
    if not pd.api.types.is_numeric_dtype(df[original_volume_col]):
        print(f"Warning: Original volume column '{original_volume_col}' is not numeric. Attempting conversion.")
        try:
            df[suggest_volume_col_name] = pd.to_numeric(df[original_volume_col], errors='coerce').fillna(0.0)
        except Exception as e:
            raise ValidationError(f"Could not convert original_volume_col '{original_volume_col}' to numeric: {e}")
    else:
        df[suggest_volume_col_name] = df[original_volume_col].fillna(0.0)
    
    return df


def _process_volume_transfer(df: pd.DataFrame, from_item_id: str, to_item_id: str, 
                           transferred_amount: float, item_id_col: str, suggest_volume_col_name: str) -> pd.DataFrame:
    """Process a single volume transfer between two items."""
    # Subtract from the sender
    from_match_condition = df[item_id_col] == from_item_id
    if from_match_condition.any():
        if df.loc[from_match_condition, suggest_volume_col_name].empty:
            print(f"    Warning (SuggestVol): No rows found for sender SKU ID '{from_item_id}' when trying to get current volume.")
        else:
            df.loc[from_match_condition, suggest_volume_col_name] -= transferred_amount
    else:
        print(f"    Warning (SuggestVol): Sender SKU ID '{from_item_id}' not found in im_final_df for volume subtraction.")

    # Add to the receiver
    to_match_condition = df[item_id_col] == to_item_id
    if to_match_condition.any():
        if df.loc[to_match_condition, suggest_volume_col_name].empty:
            print(f"    Warning (SuggestVol): No rows found for receiver SKU ID '{to_item_id}' when trying to get current volume.")
        else:
            df.loc[to_match_condition, suggest_volume_col_name] += transferred_amount
    else:
        print(f"    Warning (SuggestVol): Receiver SKU ID '{to_item_id}' not found in im_final_df for volume addition.")

    return df