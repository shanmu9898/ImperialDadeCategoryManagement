import pandas as pd
import polars as pl
from typing import List, Tuple, Optional, Dict, Any

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Import column configurations from config
from config import PipelineConfig
TRANSACTION_COLUMNS = PipelineConfig.TRANSACTION_COLUMNS

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

# =============================================================================
# CUSTOMER EXCLUSION FUNCTIONS
# =============================================================================

def get_skus_to_redi(
    true_l3m_data: pd.DataFrame,
    customer_class_col: str = TRANSACTION_COLUMNS['customer_class'],
) -> pd.DataFrame:
    """
    Identifies SKUs that serve redi customers (Customer Class = 40).

    A must keep SKU is a SKU that serves a redi customer (Customer Class = 40).
    This function filters transaction data for transactions where the customer has Customer Class = 40.

    Args:
        true_l3m_data (pd.DataFrame): Transaction data with customer class column
        customer_class_col (str): Column name for customer class in transaction data

    Returns:
        pd.DataFrame: Filtered transaction data containing only SKUs that serve redi customers
    """

    # Filter transaction data to only include transactions from redi customers (Customer_Class == 40)
    redi_customers_high_level = true_l3m_data[true_l3m_data[customer_class_col] == 'Redistributor']

    # Hardcode item_id as 'Entity--Item' per requirement
    n_unique_skus = redi_customers_high_level['Entity--Item'].nunique()
    print(f"Found {n_unique_skus} unique SKUs serving redi customers")

    return redi_customers_high_level


def get_skus_to_excl_cust(
    true_l3m: pd.DataFrame,
    customer_exclusions: list
) -> pd.DataFrame:
    """
    Identifies SKUs that are bought by excluded customers.

    This function takes transaction data and a list of customer exclusions,
    then identifies which SKUs are purchased by those excluded customers.

    Args:
        true_l3m (pd.DataFrame): Transaction data with customer name and item ID columns from TRANSACTION_COLUMNS
        customer_exclusions (list): List of customer names to exclude

    Returns:
        pd.DataFrame: DataFrame containing SKUs bought by excluded customers
    """
    # Filter for transactions from excluded customers
    excluded_customer_transactions = true_l3m[true_l3m['Customer_Name'].isin(customer_exclusions)]

    # Hardcode item_id as 'Entity--Item' per requirement
    n_unique_skus = excluded_customer_transactions['Entity--Item'].nunique()
    print(f"Found {n_unique_skus} unique SKUs bought by excluded customers")

    return excluded_customer_transactions