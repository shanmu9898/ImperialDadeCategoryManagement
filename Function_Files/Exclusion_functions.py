import pandas as pd
import polars as pl
from typing import List, Tuple, Optional, Dict, Any

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Column name mappings (only used columns)
DEFAULT_COLUMNS = {
    'item_id': 'Entity--Item'
}

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
# TRANSACTION DATA PROCESSING FUNCTIONS
# =============================================================================

def get_transactions_data(im_final2, true_l3m_file_path, exclude_canada_rows):
    """
    Process transaction data and merge with im_final DataFrame.
    
    Args:
        im_final2: Main DataFrame to merge with
        true_l3m_file_path: Path to transaction data file
        exclude_canada_rows: Whether to exclude Canada region rows
    
    Returns:
        Tuple of (processed_im_final, true_l3m_final)
    """
    transactions_schema_override = {
        "FY_24_sales": pl.Float64,
    }

    true_l3m = pl.scan_csv(
        f"{true_l3m_file_path}",    
        schema_overrides=transactions_schema_override,
        null_values=["NULL", "null", ""],
        infer_schema_length=10000,
    )

    # Step 1: Calculate L3M_sales, L3M_adj_vol, and L3M_net_cost as sums of respective months
    true_l3m = true_l3m.with_columns([
        (
            pl.col("Jan_25_sales").cast(pl.Float64).fill_null(0) +
            pl.col("Feb_25_sales").cast(pl.Float64).fill_null(0) +
            pl.col("Mar_25_sales").cast(pl.Float64).fill_null(0)
        ).alias("L3M_sales"),

        (
            pl.col("Jan_25_adj_vol").cast(pl.Float64).fill_null(0) +
            pl.col("Feb_25_adj_vol").cast(pl.Float64).fill_null(0) +
            pl.col("Mar_25_adj_vol").cast(pl.Float64).fill_null(0)
        ).alias("L3M_adj_vol"),

        (
            pl.col("Jan_25_net_cost").cast(pl.Float64).fill_null(0) +
            pl.col("Feb_25_net_cost").cast(pl.Float64).fill_null(0) +
            pl.col("Mar_25_net_cost").cast(pl.Float64).fill_null(0)
        ).alias("L3M_net_cost"),
    ])

    # Step 2: Calculate GP as sales - net_cost
    true_l3m = true_l3m.with_columns([
        (pl.col("L3M_sales") - pl.col("L3M_net_cost")).alias("L3M_GP")
    ])

    true_l3m = true_l3m.drop([
        "Jan_25_sales", "Feb_25_sales", "Mar_25_sales",
        "Jan_25_adj_vol", "Feb_25_adj_vol", "Mar_25_adj_vol",
        "Jan_25_net_cost", "Feb_25_net_cost", "Mar_25_net_cost"
    ])

    # Clean and cast numeric columns to Float64 (not Int64!)
    true_l3m_cleaned = true_l3m.with_columns([
        pl.col("L3M_sales").fill_null(0).fill_nan(0).cast(pl.Float64),
        pl.col("L3M_adj_vol").fill_null(0).fill_nan(0).cast(pl.Float64),
        pl.col("L3M_GP").fill_null(0).fill_nan(0).cast(pl.Float64),
        pl.col("L3M_net_cost").fill_null(0).fill_nan(0).cast(pl.Float64),
    ])

    # Create Entity--Item column and filter data
    true_l3m_final = true_l3m_cleaned.with_columns(
        (pl.col("entity_code").cast(pl.Utf8) + "--" + pl.col("item_code").cast(pl.Utf8)).alias("Entity--Item")
    )

    true_l3m_final = true_l3m_final.filter(pl.col("vendor") != "One Time only Vendors")

    # --- Exclude rows with region == 'Canada' if flag is set ---
    if exclude_canada_rows:
        true_l3m_final = true_l3m_final.filter(pl.col("region") != "Canada")

    true_l3m_final = true_l3m_final.collect()  # Materialize the LazyFrame before filtering with im_final

    # Convert im_final to polars DataFrame
    im_final_pl = pl.from_pandas(im_final2)

    # Filter based on Entity--Item from im_final
    true_l3m_final = true_l3m_final.filter(
        pl.col("Entity--Item").is_in(im_final_pl["Entity--Item"])
    )

    true_l3m_final = true_l3m_final.filter(pl.col('L3M_sales') > 0)

    # Group by Entity--Item 
    true_l3m_grouped = true_l3m_final.group_by(["Entity--Item"]).agg([
        pl.col("L3M_sales").sum().alias("L3M_Sales"),
        pl.col("L3M_net_cost").sum().alias("L3M_Cogs"),
        pl.col("L3M_adj_vol").sum().alias("L3M_adj_vol"),
    ])

    # Ensure the grouped columns are all Float64
    true_l3m_grouped = true_l3m_grouped.with_columns([
        pl.col("L3M_Sales").cast(pl.Float64),
        pl.col("L3M_Cogs").cast(pl.Float64),
        pl.col("L3M_adj_vol").cast(pl.Float64),
    ])

    # merge L3M_Sales, L3M_Cogs, L3M_adj_vol into im_final2. if those columns are already in im_final2, replace them
    cols_to_replace = ["L3M_Sales", "L3M_Cogs", "L3M_adj_vol"]
    existing_cols = [col for col in cols_to_replace if col in im_final_pl.columns]
    if existing_cols:
        im_final_pl = im_final_pl.drop(existing_cols)
    im_final_pl = im_final_pl.join(
        true_l3m_grouped.select(["Entity--Item", "L3M_Sales", "L3M_Cogs", "L3M_adj_vol"]),
        on="Entity--Item",
        how="left"
    )

    im_final_pl = im_final_pl.filter(pl.col("L3M_Sales") > 0)

    # Ensure the merged columns are also float in the final DataFrame
    for col in ["L3M_Sales", "L3M_Cogs", "L3M_adj_vol"]:
        if col in im_final_pl.columns:
            im_final_pl = im_final_pl.with_columns([
                pl.col(col).cast(pl.Float64)
            ])

    return im_final_pl.to_pandas(), true_l3m_final

# =============================================================================
# CUSTOMER EXCLUSION FUNCTIONS
# =============================================================================

def get_skus_to_redi(
    true_l3m_data: pd.DataFrame,
    customer_data_path: Optional[str] = None,
    account_code_col: str = 'account_code',
    customer_no_col: str = 'Customer_No',
    customer_class_col: str = 'Customer_Class',
    customer_name: str = 'customer'
) -> pd.DataFrame:
    """
    Identifies SKUs that serve redi customers (Customer Class = 40).
    
    A must keep SKU is a SKU that serves a redi customer (Customer Class = 40).
    This function loads customer data, joins it with transaction data, and filters
    for transactions where the customer has Customer Class = 40.
    
    Args:
        true_l3m_data (pl.LazyFrame): Transaction data with account_code column
        customer_data_path (str, optional): Path to customer data file. If None, uses default path.
        account_code_col (str): Column name for account code in transaction data
        customer_no_col (str): Column name for customer number in customer data
        customer_class_col (str): Column name for customer class in customer data
        
    Returns:
        pl.LazyFrame: Filtered transaction data containing only SKUs that serve redi customers
    """
    
    # Default customer data path if not provided
    if customer_data_path is None:
        customer_data_path = r"/mnt/imperialdade/approach_v2/imperial/data/01_raw/Fornax.dbo.vCustomers Entity 1.csv"
    
    # Customer schema override for consistent data types
    customer_schema_override = {
        'Branch No.': pl.String, 
        'Days To Age': pl.String,
        'Customer No.': pl.String,
        'Customer Class': pl.Int32
    }
    
    # Load customer data with complete schema override
    customer = pl.scan_csv(customer_data_path, schema_overrides=customer_schema_override)
    customer = customer.select([
        'Customer No.',
        'Customer Class'
    ])
    customer = customer.rename({"Customer No.": customer_no_col})
    customer = customer.rename({"Customer Class": customer_class_col})
    
    # Collect customer data and find all customers who have any class 40 rows
    customer_df = customer.collect()
    redi_customers = customer_df.filter(pl.col(customer_class_col) == 40)[customer_no_col].unique()
    
    # Filter transaction data to only include transactions from redi customers
    # Get unique customer names from redi customers
    redi_customer_names = true_l3m_data.filter(
        pl.col(account_code_col).is_in(redi_customers)
    ).select(pl.col(customer_name)).unique()

    # Filter for SKUs with those customer names
    redi_customers_high_level = true_l3m_data.filter(
        pl.col(customer_name).is_in(redi_customer_names[customer_name].unique().to_list())
    )
    
    print(f"Found {redi_customers_high_level.select(pl.col('Entity--Item').n_unique()).item()} unique SKUs serving redi customers")
    
    return redi_customers_high_level


def get_skus_to_excl_cust(
    true_l3m: pd.DataFrame,
    customer_exclusions: pd.DataFrame
) -> pd.DataFrame:
    """
    Identifies SKUs that are bought by excluded customers.
    
    This function takes transaction data and a list of customer exclusions,
    then identifies which SKUs are purchased by those excluded customers.
    
    Args:
        true_l3m (pl.LazyFrame): Transaction data with account_code and sales columns
        customer_exclusions (list): List of customer codes to exclude
        
    Returns:
        pd.DataFrame: DataFrame containing SKUs bought by excluded customers with L3M sales
    """
    # Filter for transactions from excluded customers
    excluded_customer_transactions = true_l3m.filter(
        pl.col("customer").is_in(customer_exclusions)
    )
        
    print(f"Found {excluded_customer_transactions.select(pl.col('Entity--Item').n_unique()).item()} unique SKUs bought by excluded customers")
    
    return excluded_customer_transactions