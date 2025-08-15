import pandas as pd
from typing import List, Tuple, Optional, Dict
from pyvent.tools.llm.openai_api import OpenAIAgent

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Column name mappings (only used columns)
DEFAULT_COLUMNS = {
    'item_id': 'Entity--Item'
}

# Processing configuration
PROCESSING_CONFIG = {
    'sample_size': 100,
    'ai_model': 'gpt-4o-mini',
    'ai_chunk_size': 32
}

# Category level mapping
CATEGORY_LEVEL_MAP = {
    2: 'item_category_level2_name',
    3: 'item_category_level3_name'
}

# Required columns for data processing
REQUIRED_COLUMNS = ['UNSPSC', 'UL ECOLOGO Certification']

# Columns for concatenation in search
SEARCH_COLUMNS = [
    'item_category_level1_code', 'item_category_level1_name',
    'item_category_level2_code', 'item_category_level2_name', 
    'item_category_level3_code', 'item_category_level3_name',      
    'description_line1_txt', 'description_line2_txt', 'description_line3_txt'
]

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

# =============================================================================
# DATA FILTERING FUNCTIONS
# =============================================================================

def _validate_s2k_inputs(
    s2k_div: str,
    im_grp: pd.DataFrame,
    sfy: pd.DataFrame,
    level: int
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate input parameters for S2K processing.
    
    Returns:
        Tuple of (is_valid, error_message, category_column_name)
    """
    try:
        validate_string_input(s2k_div, "s2k_div")
        validate_dataframe(im_grp, "im_grp")
        validate_dataframe(sfy, "sfy")
    except ValidationError as e:
        return False, str(e), None
    
    category_col_name = CATEGORY_LEVEL_MAP.get(level)
    if not category_col_name:
        return False, f"Invalid level {level}. Must be 2 or 3.", None
    
    try:
        validate_columns_exist(im_grp, [category_col_name, 'Entity'], "im_grp")
        validate_columns_exist(sfy, [DEFAULT_COLUMNS['item_id']], "sfy")
    except ValidationError as e:
        return False, str(e), None
    
    return True, "", category_col_name


def _filter_data_by_category(
    im_grp: pd.DataFrame,
    s2k_div: str,
    category_col_name: str
) -> pd.DataFrame:
    """Filter im_grp DataFrame by category and entity."""
    return im_grp[
        (im_grp[category_col_name] == s2k_div) & 
        (im_grp['Entity'] == 1)
    ].copy()


def _filter_sfy_by_items(
    sfy: pd.DataFrame,
    im_s2k: pd.DataFrame
) -> pd.DataFrame:
    """Filter sfy DataFrame based on items from im_s2k."""
    item_id_col = DEFAULT_COLUMNS['item_id']
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
    columns_with_coverage: List[str]
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
        
        if len(values) >= PROCESSING_CONFIG['sample_size']:
            sample = values[:PROCESSING_CONFIG['sample_size']]
        else:
            multiplier = (PROCESSING_CONFIG['sample_size'] // len(values)) + 1
            sample = (values * multiplier)[:PROCESSING_CONFIG['sample_size']]
        
        final_data_dict[col] = sample
        columns_actually_populated.append(col)
    
    return final_data_dict, columns_actually_populated

# =============================================================================
# MAIN S2K PROCESSING FUNCTION
# =============================================================================

def get_columns_with_coverage(
    s2k_div: str,
    im_grp: pd.DataFrame,
    sfy: pd.DataFrame,
    coverage_threshold: float,
    level: int = 2
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
    print(f"Processing category '{s2k_div}' at level {level}")
    
    # Validate inputs
    is_valid, error_msg, category_col_name = _validate_s2k_inputs(s2k_div, im_grp, sfy, level)
    if not is_valid:
        print(f"Warning: {error_msg}")
        return pd.DataFrame(), [], pd.DataFrame()
    
    # Filter im_grp by category
    im_s2k = _filter_data_by_category(im_grp, s2k_div, category_col_name)
    if im_s2k.empty:
        print(f"Warning: No items found for '{s2k_div}' at level {level}")
        return pd.DataFrame(), [], pd.DataFrame()
    
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

# =============================================================================
# NON-S2K PROCESSING FUNCTIONS
# =============================================================================

def _validate_non_s2k_inputs(
    im_grp: pd.DataFrame,
    search_str: List[str],
    category_name: str
) -> Tuple[bool, str]:
    """
    Validate inputs for get_non_s2k function.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        validate_dataframe(im_grp, "im_grp")
        validate_list_input(search_str, "search_str")
        validate_string_input(category_name, "category_name")
        
        required_columns = ['Entity'] + SEARCH_COLUMNS
        validate_columns_exist(im_grp, required_columns, "im_grp")
    except ValidationError as e:
        return False, str(e)
    
    return True, ""


def _create_concatenated_search_text(im_grp: pd.DataFrame) -> pd.DataFrame:
    """Create concatenated search text from multiple columns."""
    im_grp_copy = im_grp.copy()
    im_grp_copy['concatenated_str'] = im_grp_copy[SEARCH_COLUMNS].astype(str).agg(' '.join, axis=1)
    return im_grp_copy


def _filter_non_primary_items(
    im_grp: pd.DataFrame,
    search_str: List[str]
) -> pd.DataFrame:
    """Filter non-primary items that match search pattern."""
    pattern = '|'.join(search_str)
    return im_grp[
        (im_grp['Entity'] != 1) &
        (im_grp['concatenated_str'].str.lower().str.contains(pattern, regex=True))
    ].copy()

# =============================================================================
# AI PROCESSING FUNCTIONS
# =============================================================================

def _create_ai_prompts(category_name: str, extra: Optional[str] = None) -> Tuple[str, str]:
    """
    Create system and user prompts for AI processing.
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = f"""
You are to tag if a product is {category_name} related based on the product description.
Just return 1 if you think it is {category_name} related and 0 if you think it is not. 
No reasoning, no explanation, no other text. If you are unsure, return 0.
{extra or ""}
    """.strip()

    user_prompt = (
        f"Is this product a(n) {category_name}: {{concatenated_str}}\n\n"
        f"Please only return 1 if yes, or 0 if no."
    )
    
    return system_prompt, user_prompt


def _process_with_ai(
    im_nons2k: pd.DataFrame,
    system_prompt: str,
    user_prompt: str
) -> pd.DataFrame:
    """Process DataFrame with AI agent for tagging."""
    try:
        agent = OpenAIAgent(
            model=PROCESSING_CONFIG['ai_model'], 
            chunk_size=PROCESSING_CONFIG['ai_chunk_size']
        )
        im_nons2k_tagged = agent.format_df_prompts(im_nons2k, system_prompt, user_prompt)
        return agent.run_df_prompts(im_nons2k_tagged)
    except Exception as e:
        print(f"Error during AI processing: {e}")
        return pd.DataFrame()


def _filter_ai_approved_items(im_nons2k_tagged: pd.DataFrame) -> pd.DataFrame:
    """
    Filter items approved by AI and clean up metadata columns.
    
    Returns:
        Cleaned DataFrame with only AI-approved items
    """
    if im_nons2k_tagged.empty:
        return pd.DataFrame()
    
    # Filter AI-approved items
    filtered_items = im_nons2k_tagged[
        im_nons2k_tagged['openai_response'] == "1"
    ].copy()
    
    # Remove AI agent metadata columns (last 3 columns)
    if len(filtered_items.columns) >= 3:
        filtered_items = filtered_items.iloc[:, :-3]
    
    return filtered_items

# =============================================================================
# MAIN NON-S2K PROCESSING FUNCTION
# =============================================================================

def get_non_s2k(
    im_grp: pd.DataFrame,
    search_str: List[str],
    category_name: str,
    extra: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter non-primary items and use AI to tag category-related ones.
    
    Args:
        im_grp: DataFrame with item information and Entity column
        search_str: List of search terms to match against
        category_name: Category to tag items against
        extra: Additional instructions for AI agent
    
    Returns:
        DataFrame of AI-approved category-related non-primary items
    """
    print(f"Processing non-primary items for category: {category_name}")
    
    # Validate inputs
    is_valid, error_msg = _validate_non_s2k_inputs(im_grp, search_str, category_name)
    if not is_valid:
        print(f"Warning: {error_msg}")
        return pd.DataFrame()
    
    # Create concatenated search text
    im_grp_with_search = _create_concatenated_search_text(im_grp)
    
    # Filter non-primary items matching search pattern
    im_nons2k = _filter_non_primary_items(im_grp_with_search, search_str)
    print(f"Found {len(im_nons2k)} non-primary items matching search pattern")
    
    if im_nons2k.empty:
        print("No matching non-primary items found")
        return pd.DataFrame()
    
    # Create AI prompts
    system_prompt, user_prompt = _create_ai_prompts(category_name, extra)
    
    # Process with AI agent
    im_nons2k_tagged = _process_with_ai(im_nons2k, system_prompt, user_prompt)
    if im_nons2k_tagged.empty:
        print("AI processing failed or returned no results")
        return pd.DataFrame()
    
    # Filter AI-approved items
    filtered_items = _filter_ai_approved_items(im_nons2k_tagged)
    print(f"AI approved {len(filtered_items)} items as {category_name} related")
    
    return filtered_items