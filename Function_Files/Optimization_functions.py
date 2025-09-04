"""
Optimization Functions for Category Management

This module provides functions for supplier optimization using Mixed-Integer Linear Programming (MILP).
The objective is to maximize cost savings from swapping items while maintaining volume constraints.
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
    LpContinuous, LpStatus, value, LpMaximize, LpAffineExpression, PULP_CBC_CMD
)
import math

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

def get_cost_diff_df(im_final, transactions, rebates):
    rebates = rebates.drop_duplicates(subset=[TRANSACTION_COLUMNS['vgn']])
    df_rebate = transactions.merge(
        rebates[[TRANSACTION_COLUMNS['vgn'], 'Imperial Rebate % 2024']],
        on=TRANSACTION_COLUMNS['vgn'],
        how='left'
    )
    df_rebate.loc[:, 'Imperial Rebate % 2024'] = df_rebate['Imperial Rebate % 2024'].fillna(0)
    df_rebate['po_cost_amt (after rebate) for Non POD'] = df_rebate['po_cost_amt'] * (1 - df_rebate['Imperial Rebate % 2024'])

    # Ensure 'Case Pack' is numeric before aggregation to avoid TypeError
    df_rebate[TRANSACTION_COLUMNS['case_pack']] = pd.to_numeric(df_rebate[TRANSACTION_COLUMNS['case_pack']], errors='coerce')

    df_grouped_by_item_POD = df_rebate.groupby(['Entity--Item', TRANSACTION_COLUMNS['pod']]).agg({
        TRANSACTION_COLUMNS['qty']: 'sum',
        'po_cost_amt (after rebate) for Non POD': lambda x: np.average(x, weights=df_rebate.loc[x.index, TRANSACTION_COLUMNS['qty']]),
        TRANSACTION_COLUMNS['net_cost']: 'sum',
        TRANSACTION_COLUMNS['case_pack']: 'mean'
    }).reset_index().rename(columns={TRANSACTION_COLUMNS['case_pack']: 'Avg Case Pack'})

    df_grouped_by_item_POD.loc[:, 'Acq PO Cost (POD)'] = df_grouped_by_item_POD[TRANSACTION_COLUMNS['net_cost']] / df_grouped_by_item_POD[TRANSACTION_COLUMNS['qty']]

    # Get unique items and their indices
    unique_items = im_final['Entity--Item'].unique()
    n = len(unique_items)

    # Create a mapping from item to its Acq PO Cost (after rebate) where POD == 'N'
    item_Non_POD_cost_dict = df_grouped_by_item_POD[df_grouped_by_item_POD[TRANSACTION_COLUMNS['pod']] == 'N'].set_index('Entity--Item')['po_cost_amt (after rebate) for Non POD'].to_dict()
    item_Non_POD_qty_dict = df_grouped_by_item_POD[df_grouped_by_item_POD[TRANSACTION_COLUMNS['pod']] == 'N'].set_index('Entity--Item')[TRANSACTION_COLUMNS['qty']].to_dict()
    item_case_pack_dict = df_grouped_by_item_POD.groupby('Entity--Item')['Avg Case Pack'].mean().to_dict()
    item_POD_cost_dict = df_grouped_by_item_POD[df_grouped_by_item_POD[TRANSACTION_COLUMNS['pod']] == 'Y'].set_index('Entity--Item')['Acq PO Cost (POD)'].to_dict()
    item_POD_qty_dict = df_grouped_by_item_POD[df_grouped_by_item_POD[TRANSACTION_COLUMNS['pod']] == 'Y'].set_index('Entity--Item')[TRANSACTION_COLUMNS['qty']].to_dict()

    # Fill NaN values in item_case_pack_dict with 1000
    for k, v in item_case_pack_dict.items():
        if pd.isna(v) or v == 0:
            item_case_pack_dict[k] = 1000
    # Initialize the matrix
    cost_diff_matrix = np.zeros((n, n))

    # Fill the matrix
    for i, item_i in enumerate(unique_items):
        for j, item_j in enumerate(unique_items):
            case_pack_j = item_case_pack_dict.get(item_j, np.nan)

            POD_cost_i = item_POD_cost_dict.get(item_i, np.nan)
            POD_qty_i = item_POD_qty_dict.get(item_i, np.nan)

            Non_POD_cost_i = item_Non_POD_cost_dict.get(item_i, np.nan)
            Non_POD_qty_i = item_Non_POD_qty_dict.get(item_i, np.nan)

            case_pack_i = item_case_pack_dict.get(item_i, np.nan)
            case_pack_ratio_i = case_pack_i/case_pack_j

            adj_Non_POD_cost_i = Non_POD_cost_i*case_pack_ratio_i
            adj_POD_cost_i = POD_cost_i*case_pack_ratio_i
            adj_Non_POD_qty_i = Non_POD_qty_i*(1/case_pack_ratio_i)
            adj_POD_qty_i = POD_qty_i*(1/case_pack_ratio_i)


            cost_j = item_Non_POD_cost_dict.get(item_j, np.nan)
            case_pack_ratio_j = 1
            adj_cost_j = cost_j*case_pack_ratio_j

            Non_POD_savings = (adj_Non_POD_cost_i - adj_cost_j)*adj_Non_POD_qty_i
            POD_savings = (adj_POD_cost_i - adj_cost_j)*adj_POD_qty_i

            if np.isnan(Non_POD_savings):
                Non_POD_savings = 0
            if np.isnan(POD_savings):
                POD_savings = 0

            cost_diff_matrix[i, j] = Non_POD_savings + POD_savings

    cost_diff_df = pd.DataFrame(cost_diff_matrix, index=unique_items, columns=unique_items)
    np.fill_diagonal(cost_diff_df.values, 0)
    return cost_diff_df

# =============================================================================
# DATA PREPARATION FUNCTIONS
# =============================================================================

def prepare_optimization_data(
    im_final: pd.DataFrame,
    cost_diff_matrix: pd.DataFrame,
    must_keep_ids: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], Dict[str, int], Dict[str, int]]:
    """
    Prepare data for optimization by cleaning and validating inputs.
    
    Args:
        im_final: DataFrame with item data
        cost_diff_matrix: DataFrame with cost differences for swaps [i,j] = cost to switch from i to j
        must_keep_ids: Optional list of item IDs that must retain their volume
        
    Returns:
        Tuple of (cleaned_im_final, cleaned_cost_matrix, item_ids, supplier_ids, item_to_idx, supplier_to_idx)
        
    Raises:
        ValidationError: If data validation fails
    """
    validate_dataframe(im_final, "im_final")
    validate_dataframe(cost_diff_matrix, "cost_diff_matrix")
    
    # Validate required columns in im_final
    required_columns = ['Entity--Item', TRANSACTION_COLUMNS['qty'], TRANSACTION_COLUMNS['net_cost']]
    validate_columns_exist(im_final, required_columns, "im_final")
    
    # Validate cost_diff_matrix structure
    if cost_diff_matrix.shape[0] != cost_diff_matrix.shape[1]:
        raise ValidationError("cost_diff_matrix must be square (same number of rows and columns)")
    
    # Get unique item IDs
    item_ids = sorted(im_final['Entity--Item'].dropna().unique().tolist())
    
    # Validate that cost_diff_matrix covers all items
    matrix_items = sorted(cost_diff_matrix.index.tolist())
    if set(item_ids) != set(matrix_items):
        missing_in_matrix = set(item_ids) - set(matrix_items)
        missing_in_data = set(matrix_items) - set(item_ids)
        if missing_in_matrix:
            print(f"Warning: Items in im_final but not in cost_diff_matrix: {missing_in_matrix}")
        if missing_in_data:
            print(f"Warning: Items in cost_diff_matrix but not in im_final: {missing_in_data}")
    
    # Clean and prepare im_final
    im_final_clean = im_final.copy()
    
    # Ensure numeric columns are properly typed
    im_final_clean[TRANSACTION_COLUMNS['qty']] = pd.to_numeric(im_final_clean[TRANSACTION_COLUMNS['qty']], errors='coerce').fillna(0)
    im_final_clean[TRANSACTION_COLUMNS['net_cost']] = pd.to_numeric(im_final_clean[TRANSACTION_COLUMNS['net_cost']], errors='coerce').fillna(0)
    
    # Get unique suppliers
    supplier_ids = sorted(im_final_clean[TRANSACTION_COLUMNS['vgn']].dropna().unique().tolist())
    
    # Create mappings
    item_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}
    supplier_to_idx = {supplier_id: i for i, supplier_id in enumerate(supplier_ids)}
    
    # Clean cost_diff_matrix
    cost_matrix_clean = cost_diff_matrix.copy()
    cost_matrix_clean = cost_matrix_clean.reindex(index=item_ids, columns=item_ids, fill_value=0)
    
    # Ensure diagonal is 0 (no cost to keep same item)
    np.fill_diagonal(cost_matrix_clean.values, 0)
    
    return im_final_clean, cost_matrix_clean, item_ids, supplier_ids, item_to_idx, supplier_to_idx

def create_swap_feasibility_matrix(
    im_final: pd.DataFrame,
    item_ids: List[str],
    item_to_idx: Dict[str, int]
) -> np.ndarray:
    """
    Create a swap feasibility matrix based on the Matches column.
    
    Args:
        im_final: DataFrame with item data
        item_ids: List of unique item IDs
        item_to_idx: Mapping from item ID to index
        
    Returns:
        numpy array where [i,j] = 1 if item i can be swapped with item j
    """
    num_items = len(item_ids)
    feasibility_matrix = np.zeros((num_items, num_items), dtype=int)

    # Set diagonal to 1 (items can always stay the same)
    np.fill_diagonal(feasibility_matrix, 1)

    # Process matches column
    for _, row in im_final.iterrows():
        item_id = row['Entity--Item']
        matches = row['Matches']

        if pd.isna(item_id) or item_id not in item_to_idx:
            continue
            
        item_idx = item_to_idx[item_id]

        # Parse matches
        match_list = []
        if isinstance(matches, list):
            match_list = matches
        elif isinstance(matches, str) and matches.strip().startswith('[') and matches.strip().endswith(']'):
            try:
                parsed = ast.literal_eval(matches)
                if isinstance(parsed, list):
                    match_list = parsed
            except (ValueError, SyntaxError, TypeError):
                pass

        # Set feasibility for matches
        for match in match_list:
            match_str = str(match).strip()
            if match_str in item_to_idx:
                match_idx = item_to_idx[match_str]
                feasibility_matrix[item_idx, match_idx] = 1
                feasibility_matrix[match_idx, item_idx] = 1  # Make symmetric

    return feasibility_matrix

def check_basic_feasibility(
    im_final: pd.DataFrame,
    cost_diff_matrix: pd.DataFrame,
    must_keep_ids: Optional[List[str]] = None
) -> Dict:
    """
    Check basic feasibility of the optimization problem before solving.
    
    Args:
        im_final: DataFrame with item data
        cost_diff_matrix: DataFrame with cost differences for swaps
        must_keep_ids: Optional list of item IDs that must retain their volume
        
    Returns:
        Dictionary with feasibility check results
    """
    print("--- Checking Basic Feasibility ---")
    
    # Prepare data
    im_final_clean, cost_matrix_clean, item_ids, supplier_ids, item_to_idx, supplier_to_idx = prepare_optimization_data(
        im_final, cost_diff_matrix, must_keep_ids
    )
    
    # Create swap feasibility matrix
    feasibility_matrix = create_swap_feasibility_matrix(im_final_clean, item_ids, item_to_idx)
    
    # Get volumes
    volumes = {}
    for item_id in item_ids:
        item_data = im_final_clean[im_final_clean['Entity--Item'] == item_id]
        if not item_data.empty:
            volumes[item_id] = item_data[TRANSACTION_COLUMNS['qty']].iloc[0]
        else:
            volumes[item_id] = 0
    
    # Check 1: Are there any feasible swaps?
    feasible_swaps = 0
    for i in item_ids:
        for j in item_ids:
            if i != j and feasibility_matrix[item_to_idx[i], item_to_idx[j]] == 1:
                feasible_swaps += 1
    
    print(f"Total feasible swaps: {feasible_swaps}")
    
    # Check 2: Are there any positive cost savings?
    positive_savings = 0
    total_savings = 0
    for i in item_ids:
        for j in item_ids:
            if i != j:
                savings = cost_matrix_clean.loc[i, j]
                total_savings += savings
                if savings > 0:
                    positive_savings += 1
    
    print(f"Positive cost savings opportunities: {positive_savings}")
    print(f"Total potential savings: ${total_savings:,.2f}")
    
    # Check 3: Volume constraints
    zero_volume_items = sum(1 for v in volumes.values() if v < 1e-6)
    print(f"Items with zero volume: {zero_volume_items}")
    
    # Check 4: Must-keep constraints
    if must_keep_ids:
        must_keep_in_data = [item_id for item_id in must_keep_ids if item_id in item_ids]
        print(f"Must-keep items found in data: {len(must_keep_in_data)}/{len(must_keep_ids)}")
    
    feasibility_info = {
        'feasible_swaps': feasible_swaps,
        'positive_savings': positive_savings,
        'total_savings': total_savings,
        'zero_volume_items': zero_volume_items,
        'total_items': len(item_ids),
        'total_suppliers': len(supplier_ids)
    }
    
    print(f"Feasibility check complete. Problem appears {'feasible' if feasible_swaps > 0 else 'infeasible'}.")
    return feasibility_info

# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================

def solve_swap_optimization(
    im_final: pd.DataFrame,
    cost_diff_matrix: pd.DataFrame,
    must_keep_ids: Optional[List[str]] = None,
    max_swaps: Optional[int] = None,
    volume_constraint_factor: float = 2.0,
    max_suppliers: Optional[int] = None
) -> Dict:
    """
    Solve the swap optimization problem to maximize cost savings.

    This consolidated solver supports optional supplier limit via max_suppliers.
    
    Args:
        im_final: DataFrame with item data
        cost_diff_matrix: DataFrame with cost differences for swaps
        must_keep_ids: Optional list of item IDs that must retain their volume
        max_swaps: Optional maximum number of swaps allowed
        volume_constraint_factor: Factor to limit volume redistribution (default 2.0)
        max_suppliers: Optional maximum number of suppliers allowed (None = no cap)
        
    Returns:
        Dictionary with optimization results
    """
    print("--- Initializing Swap Optimization Problem ---")

    # Prepare data
    im_final_clean, cost_matrix_clean, item_ids, supplier_ids, item_to_idx, supplier_to_idx = prepare_optimization_data(
        im_final, cost_diff_matrix, must_keep_ids
    )

    # Create swap feasibility matrix
    feasibility_matrix = create_swap_feasibility_matrix(im_final_clean, item_ids, item_to_idx)

    # Get volumes, costs, suppliers
    volumes: Dict[str, float] = {}
    costs: Dict[str, float] = {}
    suppliers: Dict[str, str] = {}

    for item_id in item_ids:
        item_data = im_final_clean[im_final_clean['Entity--Item'] == item_id]
        if not item_data.empty:
            volumes[item_id] = float(item_data[TRANSACTION_COLUMNS['qty']].iloc[0])
            costs[item_id] = float(item_data[TRANSACTION_COLUMNS['net_cost']].iloc[0])
            raw_supplier = item_data[TRANSACTION_COLUMNS['vgn']].iloc[0]
            supplier_str = str(raw_supplier) if pd.notna(raw_supplier) else 'UNKNOWN'
            if supplier_str.lower() in ('nan', 'none', ''):
                supplier_str = 'UNKNOWN'
            suppliers[item_id] = supplier_str
        else:
            volumes[item_id] = 0.0
            costs[item_id] = 0.0
            suppliers[item_id] = 'UNKNOWN'

    # Normalize supplier_ids to match normalized suppliers mapping
    supplier_ids = sorted(set(suppliers.values()))

    # Create the optimization problem
    prob = LpProblem("Swap_Optimization", LpMaximize)

    # Decision variables
    feasible_pairs = [(i, j) for i in item_ids for j in item_ids
                      if i != j and feasibility_matrix[item_to_idx[i], item_to_idx[j]] == 1]
    x = LpVariable.dicts("x_swap", feasible_pairs, cat=LpBinary)

    # Supplier activation (only used if needed through constraints)
    y = LpVariable.dicts("y_supplier_active", supplier_ids, cat=LpBinary)

    # Helpers
    def outgoing_sum(i: str) -> LpAffineExpression:
        return lpSum(x[(i, j)] for j in item_ids if (i, j) in x)

    def incoming_sum(j: str) -> LpAffineExpression:
        return lpSum(x[(i, j)] for i in item_ids if (i, j) in x)

    def final_volume_expr_for(j: str) -> LpAffineExpression:
        retain = volumes[j] * (1 - outgoing_sum(j))
        inbound = lpSum(volumes[i] * x[(i, j)] for i in item_ids if (i, j) in x)
        return retain + inbound

    # Objective
    prob += lpSum(cost_matrix_clean.loc[i, j] * x[(i, j)] for (i, j) in feasible_pairs), "Maximize_Cost_Savings"

    # Constraints
    # (1) Each item can swap to at most one other item
    for i in item_ids:
        prob += outgoing_sum(i) <= 1, f"At_Most_One_Outgoing_Swap_{i}"

    # (1b) Cannot give and receive (all-or-nothing): if an item swaps away, it cannot receive
    for i in item_ids:
        prob += outgoing_sum(i) + incoming_sum(i) <= 1, f"Cannot_Give_And_Receive_{i}"

    # (2) Must-keep SKUs cannot be swapped away (but can receive)
    if must_keep_ids:
        for k in must_keep_ids:
            if k in item_ids:
                prob += outgoing_sum(k) == 0, f"Must_Keep_No_Outgoing_{k}"

    # (3) Final volume caps
    cap_by_j = {}
    for j in item_ids:
        original_volume = volumes[j]
        if original_volume < 1e-6:
            cap_by_j[j] = 1000.0
        else:
            cap_by_j[j] = float(volume_constraint_factor) * original_volume
        prob += final_volume_expr_for(j) <= cap_by_j[j], f"Final_Volume_Cap_{j}"
        prob += final_volume_expr_for(j) >= 0, f"Final_Volume_NonNegative_{j}"

    # (4) Supplier activation linkage: if any volume at SKU j, its supplier must be active
    for j in item_ids:
        s = suppliers[j]
        if s in y:
            prob += final_volume_expr_for(j) <= cap_by_j[j] * y[s], f"Supplier_On_If_Volume_At_{j}"

    # (5) Optional: limit total number of swaps
    if max_swaps is not None:
        prob += lpSum(x[(i, j)] for (i, j) in feasible_pairs) <= max_swaps, "Total_swaps_limit"

    # (6) Optional: supplier limit
    if max_suppliers is not None:
        # Constraint: total active suppliers <= max_suppliers
        prob += lpSum(y[s] for s in supplier_ids) <= max_suppliers, f"Supplier_Limit_{max_suppliers}"
        
        # Faster approach: Link supplier activation to actual usage using individual item constraints
        # For each item j: final_volume[j] <= M * y[supplier[j]]
        # This is more efficient than summing all items per supplier
        for j in item_ids:
            supplier_j = suppliers[j]
            final_vol_j = final_volume_expr_for(j)
            if final_vol_j is None:
                continue
            if supplier_j not in y:
                # Skip linking if supplier not present in activation variables (shouldn't happen after normalization)
                continue
            # If item j has volume > 0, its supplier must be active
            # Use original volume as M (tight bound)
            M_j = float(volumes.get(j, 0.0)) * float(volume_constraint_factor)
            if M_j <= 0:
                continue
            prob += final_vol_j <= M_j * y[supplier_j], f"Item_supplier_link_{j}"

    # Solve
    solver = PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    # Results
    solution_status = LpStatus[prob.status]
    print(f"--- Optimization Results ---")
    print(f"Status: {solution_status}")

    results = {
        'status': solution_status,
        'total_savings': 0.0,
        'swaps': [],
        'final_volumes': {},
        'active_suppliers': set(),
        'objective_value': value(prob.objective) if solution_status == "Optimal" else None
    }

    if solution_status == "Optimal":
        # Swaps and savings
        for (i, j) in feasible_pairs:
            if value(x[(i, j)]) == 1:
                savings = float(cost_matrix_clean.loc[i, j])
                results['swaps'].append({
                    'from_item': i,
                    'to_item': j,
                    'savings': savings,
                    'volume_moved': volumes[i]
                })
                results['total_savings'] += savings

        # Final volumes and active suppliers
        for j in item_ids:
            fv = float(value(final_volume_expr_for(j))) if final_volume_expr_for(j) is not None else 0.0
            results['final_volumes'][j] = fv
            if fv > 0:
                results['active_suppliers'].add(suppliers[j])

        print(f"Total yearly cost savings: ${results['total_savings']*2:,.2f}")
        print(f"Number of swaps: {len(results['swaps'])}")
        print(f"Active suppliers: {len(results['active_suppliers'])}")

    return results

def solve_swap_optimization_looped(
    im_final: pd.DataFrame,
    cost_diff_matrix: pd.DataFrame,
    must_keep_ids: Optional[List[str]] = None,
    volume_constraint_factor: float = 2.0,
    supplier_step_factor: float = 0.95
) -> pd.DataFrame:
    """
    Solve swap optimization iteratively with decreasing supplier limits to find the trade-off
    between supplier count and cost savings.
    
    Args:
        im_final: DataFrame with item data
        cost_diff_matrix: DataFrame with cost differences for swaps
        must_keep_ids: Optional list of item IDs that must retain their volume
        volume_constraint_factor: Factor to limit volume redistribution
        supplier_step_factor: Factor to reduce supplier limit in each iteration (default 0.95 = 5% reduction)
    
    Returns:
        DataFrame with columns: Max_Suppliers_Allowed, Actual_Active_Suppliers, 
        and Achieved_Cost_Savings showing the trade-off
    """
    print("--- Initializing Looped Optimization Problem ---")

    # Prepare data
    im_final_clean, cost_matrix_clean, item_ids, supplier_ids, item_to_idx, supplier_to_idx = prepare_optimization_data(
        im_final, cost_diff_matrix, must_keep_ids
    )
    
    # Create swap feasibility matrix
    feasibility_matrix = create_swap_feasibility_matrix(im_final_clean, item_ids, item_to_idx)

    # Get volumes, costs, and suppliers
    volumes = {}
    costs = {}
    suppliers = {}
    
    for item_id in item_ids:
        item_data = im_final_clean[im_final_clean['Entity--Item'] == item_id]
        if not item_data.empty:
            volumes[item_id] = item_data[TRANSACTION_COLUMNS['qty']].iloc[0]
            costs[item_id] = item_data[TRANSACTION_COLUMNS['net_cost']].iloc[0]
            suppliers[item_id] = item_data[TRANSACTION_COLUMNS['vgn']].iloc[0]
        else:
            volumes[item_id] = 0
            costs[item_id] = 0
            suppliers[item_id] = 'UNKNOWN'
    
    num_total_suppliers = len(supplier_ids)
    results_list = []
    current_max_allowed_suppliers = num_total_suppliers

    print(f"Starting optimization loop. Initial max suppliers: {current_max_allowed_suppliers}")

    while True:
        if current_max_allowed_suppliers < 1:
            print("Supplier limit fell below 1. Stopping.")
            break

        print(f"\n--- Solving for Max Suppliers <= {current_max_allowed_suppliers} ---")

        # Solve optimization with current supplier limit
        results = solve_swap_optimization(
            im_final=im_final,
            cost_diff_matrix=cost_diff_matrix,
            must_keep_ids=must_keep_ids,
            max_suppliers=current_max_allowed_suppliers,
            volume_constraint_factor=volume_constraint_factor
        )
        
        if results['status'] == "Optimal":
            results_list.append({
                'Max_Suppliers_Allowed': current_max_allowed_suppliers,
                'Actual_Active_Suppliers': len(results['active_suppliers']),
                'Achieved_Cost_Savings': results['total_savings'],
                'Number_of_Swaps': len(results['swaps']),
                'Total_Volume_Moved': sum(swap['volume_moved'] for swap in results['swaps'])
            })

            if current_max_allowed_suppliers == 1:
                print("Reached supplier limit of 1 and found an optimal solution. Stopping loop.")
                break

            # Determine the next supplier limit
            next_max_suppliers_float = current_max_allowed_suppliers * supplier_step_factor
            next_max_suppliers = max(1, math.floor(next_max_suppliers_float))

            if next_max_suppliers >= current_max_allowed_suppliers and current_max_allowed_suppliers > 1:
                next_max_suppliers = current_max_allowed_suppliers - 1

            current_max_allowed_suppliers = next_max_suppliers

        else:
            print(f"Optimization became non-optimal (status: {results['status']}) at max supplier limit: {current_max_allowed_suppliers}. Stopping loop.")
            break

    if not results_list:
        print("No optimal solutions found in any iteration.")
        return pd.DataFrame(columns=['Max_Suppliers_Allowed', 'Actual_Active_Suppliers', 'Achieved_Cost_Savings', 'Number_of_Swaps', 'Total_Volume_Moved'])

    results_df = pd.DataFrame(results_list)
    return results_df.sort_values(by='Max_Suppliers_Allowed', ascending=False).reset_index(drop=True)

# =============================================================================
# ANALYSIS AND REPORTING FUNCTIONS
# =============================================================================

def analyze_swap_results(
    im_final: pd.DataFrame,
    optimization_results: Dict
) -> pd.DataFrame:
    """
    Analyze optimization results and create detailed report.
    
    Args:
        im_final: Original DataFrame with item data
        optimization_results: Results from optimization
    
    Returns:
        DataFrame with detailed analysis including:
        - Entity--Item: Item identifier
        - Original_Volume, Final_Volume, Volume_Change: Volume metrics
        - Original_Cost, Original_Supplier, New_Supplier: Cost and supplier info
        - VGN: Vendor name
        - All Descriptions: Item descriptions
        - attributes: Product attributes
        - PL_Flag: Private label flag
        - Action: Description of what happened to the item
        - Savings_From_Swapping_Away: Savings when this item was swapped to another
        - Savings_From_Swapping_To: Savings when other items were swapped to this one
    """
    if optimization_results['status'] != "Optimal":
        print("Cannot analyze results: optimization was not optimal")
        return pd.DataFrame()
    
    # Create analysis DataFrame
    analysis_data = []
    
    for item_id in im_final['Entity--Item'].unique():
        if pd.isna(item_id):
            continue

        item_data = im_final[im_final['Entity--Item'] == item_id].iloc[0]

        original_volume = item_data[TRANSACTION_COLUMNS['qty']]
        original_cost = item_data[TRANSACTION_COLUMNS['net_cost']]
        original_supplier = item_data[TRANSACTION_COLUMNS['vgn']]

        final_volume = optimization_results['final_volumes'].get(item_id, original_volume)

        # Find if this item was involved in any swaps
        swaps_from = [s for s in optimization_results['swaps'] if s['from_item'] == item_id]
        swaps_to = [s for s in optimization_results['swaps'] if s['to_item'] == item_id]

        # Calculate savings from swapping away (when this item was swapped to another)
        savings_from_swapping_away = sum(s['savings'] for s in swaps_from)
        
        # Calculate savings from swapping to (when other items were swapped to this one)
        savings_from_swapping_to = sum(s['savings'] for s in swaps_to)
        
        # Determine action and new supplier
        if swaps_from:
            # Item was swapped away
            swap_info = swaps_from[0]
            new_supplier = im_final[im_final['Entity--Item'] == swap_info['to_item']][TRANSACTION_COLUMNS['vgn']].iloc[0]
            action = f"Swapped to {swap_info['to_item']}"
        elif swaps_to:
            # Item received a swap
            swap_info = swaps_to[0]
            new_supplier = original_supplier
            action = f"Received from {swap_info['from_item']}"
        else:
            # No change
            new_supplier = original_supplier
            action = "No change"

        # Get additional item details from im_final
        vendor_name = item_data.get(TRANSACTION_COLUMNS['vgn'], 'N/A')
        all_descriptions = item_data.get('All Descriptions', 'N/A')
        attributes = item_data.get('attributes', 'N/A')
        pl_flag = item_data.get(TRANSACTION_COLUMNS['vb_flag'], 'N/A')
        
        analysis_data.append({
            'Entity--Item': item_id,
            'Original_Volume': original_volume,
            'Final_Volume': final_volume,
            'Volume_Change': final_volume - original_volume,
            'Original_Cost': original_cost,
            'Original_Supplier': original_supplier,
            'New_Supplier': new_supplier,
            'VGN': vendor_name,
            'All Descriptions': all_descriptions,
            'attributes': attributes,
            'PL_Flag': pl_flag,
            'Action': action,
            'Savings_From_Swapping_Away': savings_from_swapping_away,
            'Savings_From_Swapping_To': savings_from_swapping_to
        })

    return pd.DataFrame(analysis_data)

def create_optimization_summary(optimization_results: Dict) -> Dict:
    """
    Create a summary of optimization results.
    
    Args:
        optimization_results: Results from optimization
        
    Returns:
        Dictionary with optimization summary
    """
    summary = {
        'optimization_status': optimization_results['status'],
        'total_cost_savings': optimization_results['total_savings'],
        'number_of_swaps': len(optimization_results['swaps']),
        'active_suppliers': len(optimization_results['active_suppliers']),
        'timestamp': datetime.now().isoformat()
    }
    
    if optimization_results['swaps']:
        summary['average_savings_per_swap'] = optimization_results['total_savings'] / len(optimization_results['swaps'])
        summary['total_volume_moved'] = sum(swap['volume_moved'] for swap in optimization_results['swaps'])
    
    return summary

# =============================================================================
# EXCEL EXPORT FUNCTIONS
# =============================================================================

def export_optimization_results(
    summary: Dict,
    detailed_results: pd.DataFrame,
    output_path: str
) -> None:
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

def calculate_swap_metrics(
    im_final: pd.DataFrame,
    cost_diff_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate metrics for potential swaps.
    
    Args:
        im_final: DataFrame with item data
        cost_diff_matrix: DataFrame with cost differences for swaps
    
    Returns:
        DataFrame with swap metrics
    """
    metrics_data = []
    
    for i in im_final['Entity--Item'].unique():
        if pd.isna(i):
            continue

        for j in im_final['Entity--Item'].unique():
            if pd.isna(j) or i == j:
                continue
            
            cost_diff = cost_diff_matrix.loc[i, j]
            if cost_diff > 0:  # Only positive cost differences (savings)
                i_data = im_final[im_final['Entity--Item'] == i].iloc[0]
                j_data = im_final[im_final['Entity--Item'] == j].iloc[0]
                
                metrics_data.append({
                    'From_Item': i,
                    'To_Item': j,
                    'Cost_Savings': cost_diff,
                    'From_Volume': i_data[TRANSACTION_COLUMNS['qty']],
                    'To_Volume': j_data[TRANSACTION_COLUMNS['qty']],
                    'From_Supplier': i_data[TRANSACTION_COLUMNS['vgn']],
                    'To_Supplier': j_data[TRANSACTION_COLUMNS['vgn']],
                    'From_Cost': i_data[TRANSACTION_COLUMNS['net_cost']],
                    'To_Cost': j_data[TRANSACTION_COLUMNS['net_cost']]
                })
    
    return pd.DataFrame(metrics_data).sort_values('Cost_Savings', ascending=False)

def filter_swaps_by_criteria(
    swaps_df: pd.DataFrame,
    min_savings: float = 0.0,
    min_volume: float = 0.0,
    exclude_same_supplier: bool = True
) -> pd.DataFrame:
    """
    Filter swaps based on specified criteria.
    
    Args:
        swaps_df: DataFrame with swap data
        min_savings: Minimum savings threshold
        min_volume: Minimum volume threshold
        exclude_same_supplier: Whether to exclude swaps within the same supplier
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = swaps_df.copy()
    
    if min_savings > 0:
        filtered_df = filtered_df[filtered_df['Cost_Savings'] >= min_savings]
    
    if min_volume > 0:
        filtered_df = filtered_df[filtered_df['From_Volume'] >= min_volume]
    
    if exclude_same_supplier:
        filtered_df = filtered_df[filtered_df['From_Supplier'] != filtered_df['To_Supplier']]
    
    return filtered_df