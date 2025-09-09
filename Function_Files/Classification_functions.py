import pandas as pd
import os
from typing import List, Tuple, Optional, Dict, Any
from pyvent.tools.llm.openai_api import OpenAIAgent

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Import column configurations from config
from config import PipelineConfig
TRANSACTION_COLUMNS = PipelineConfig.TRANSACTION_COLUMNS

# Processing configuration
PROCESSING_CONFIG = {
    'ai_model': 'gpt-4o-mini',
    'ai_chunk_size': 32
}

# Excel configuration
EXCEL_CONFIG = {
    'sheets': {
        'drop_list': 'Drop List',
        'suggested_swaps': 'Suggested Swaps',
        'customer_performance': 'Customer Performance Summary',
        'vendor_performance': 'Vendor Performance Summary',
        'sku_summary': 'SKU Summary',
        'extracted_attributes': 'Extracted_Attributes'
    },
    'formats': {
        'item_id_color': '#E0E0E0',      # Grey
        'desc_vendor_color': '#DAEEF3',   # Light Blue
        'attribute_color': '#E2EFDA',     # Light Green
    }
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

def validate_columns_exist(df: pd.DataFrame, required_columns: List[str], df_name: str) -> None:
    """Validate that required columns exist in DataFrame."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValidationError(f"Missing required columns in {df_name}: {', '.join(missing_cols)}")

def validate_list_input(value: List[str], name: str) -> None:
    """Validate list input."""
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{name} must be a non-empty list")
    
    if not all(isinstance(item, str) for item in value):
        raise ValidationError(f"All items in {name} must be strings")

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def get_top_values(df: pd.DataFrame, columns: List[str], n: int = 5) -> str:
    """
    Get summary string of top N most frequent values for specified columns.
    
    Args:
        df: DataFrame to analyze
        columns: List of column names to process
        n: Number of top values to return per column
    
    Returns:
        Formatted string with column summaries
    """
    try:
        validate_dataframe(df, "DataFrame")
        validate_list_input(columns, "columns")
    except ValidationError as e:
        print(f"Error: {e}")
        return ""
    
    column_summaries = []
    
    for col_name in columns:
        if col_name not in df.columns:
            column_summaries.append(f"{col_name}: [Column not found]")
            continue

        # Clean and get top values
        series = (
            df[col_name]
            .dropna()
            .astype(str)
            .str.strip()
        )
        series = series[series != '']

        top_values = series.value_counts().nlargest(n).index.tolist()
        
        if not top_values:
            column_summaries.append(f"{col_name}: [No values]")
        else:
            column_summaries.append(f"{col_name}: {', '.join(top_values)}")

    return " \n\n ".join(column_summaries)


def create_description_string(row: pd.Series, columns: List[str]) -> str:
    """
    Create formatted description string from DataFrame row columns.
    
    Args:
        row: DataFrame row to process
        columns: List of column names to include
    
    Returns:
        Formatted description string with column values and Description with Attributes
    """
    try:
        validate_list_input(columns, "columns")
    except ValidationError as e:
        print(f"Error: {e}")
        return ""
    
    description_parts = []
    
    for col_name in columns:
        if col_name in row:
            value = row[col_name]
            if pd.isna(value):
                continue
            description_parts.append(f"{col_name}: {value}, ")

    description_str = " ".join(description_parts).strip()
    all_descriptions = str(row.get('Combined Descriptions', ''))

    return f"{description_str} {all_descriptions}".strip()


def get_most_common_values(prompt_options_string: str, value_separator: str = ", ") -> str:
    """
    Extract most common values from formatted column summaries.
    
    Expected format: "Column A: val1, val2, val3\n\nColumn B: val4, val5"
    
    Args:
        prompt_options_string: String with column summaries separated by "\n\n"
        value_separator: Delimiter between values in each summary
    
    Returns:
        Compact string with most common value per column separated by " | "
    """
    if not prompt_options_string:
        return ""
    
    output_parts = []
    column_parts = prompt_options_string.split("\n\n")

    for part in column_parts:
        if ':' not in part:
            continue

        col_name_str, values_str = part.split(':', 1)
        col_name = col_name_str.strip()
        values_str_stripped = values_str.strip()

        if not values_str_stripped:
            continue
            
        most_common_value = values_str_stripped.split(value_separator)[0].strip()
        output_parts.append(f"{col_name}: {most_common_value}")

    return " | ".join(output_parts)


def parse_openai_response(response_string: str) -> Dict[str, str]:
    """
    Parse OpenAI pipe-delimited response string into dictionary.
    
    Args:
        response_string: Pipe-delimited string from OpenAI
    
    Returns:
        Dictionary of attribute names and values
    """
    if not response_string or pd.isna(response_string):
        return {}
    
    attributes = {}
    try:
        parts = response_string.split('|')
        for part in parts:
            part = part.strip()
            if ':' in part:
                key, value = part.split(':', 1)
                attributes[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error parsing OpenAI response: {e}")
    
    return attributes

# =============================================================================
# AI PROCESSING FUNCTIONS
# =============================================================================

def _create_ai_prompts(output_str: str) -> Tuple[str, str]:
    """Create AI prompts for description analysis."""
    system_prompt = f"""
    You are to pull out the key features and characteristics of a product description. 
    Your output should look like this: {output_str}.
    
    It should keep the exact same structure, with the same features followed by a colon 
    and separated by a pipe (|) character. If a feature is not present in the description, 
    it should be left blank.
    """.strip()

    user_prompt = "Explain this product description:\n\n{description}"
    
    return system_prompt, user_prompt

def explain_top_qty_description(
    df: pd.DataFrame, 
    output_str: str, 
    description_col: str = 'Description with Attributes',
    qty_col: str = None
) -> Tuple[str, str]:
    """
    Use AI to explain the description of the top-selling item.
    
    Args:
        df: DataFrame with sales data and descriptions
        output_str: Expected output format for AI
        description_col: Column name containing descriptions
    
    Returns:
        Tuple of (description_text, ai_explanation)
    """
    try:
        validate_dataframe(df, "DataFrame")
        
        if not qty_col:
            sales_col = PipelineConfig.TRANSACTION_COLUMNS['qty']
        else:
            sales_col = sales_col
        required_cols = [sales_col, description_col]
        validate_columns_exist(df, required_cols, "DataFrame")
    except ValidationError as e:
        raise ValueError(f"Invalid input: {e}")

    # Find top-selling item
    top_row = df.loc[df[sales_col].idxmax()]
    description_text = str(top_row[description_col])

    # Create AI prompts
    system_prompt, user_prompt_template = _create_ai_prompts(output_str)
    user_prompt = user_prompt_template.format(description=description_text)

    # Process with AI agent
    try:
        agent = OpenAIAgent(
            model=PROCESSING_CONFIG['ai_model'], 
            chunk_size=PROCESSING_CONFIG['ai_chunk_size']
        )
        input_df = pd.DataFrame([{'description': description_text}])
        input_df = agent.format_df_prompts(input_df, system_prompt, user_prompt)
        output_df = agent.run_df_prompts(input_df)

        if 'openai_response' not in output_df.columns:
            raise RuntimeError("AI response column 'openai_response' not found in output")

        explanation = output_df.at[0, 'openai_response']
        return description_text, explanation
    except Exception as e:
        print(f"Error during AI processing: {e}")
        return description_text, ""


# =============================================================================
# EXCEL PROCESSING FUNCTIONS
# =============================================================================

def _determine_excel_engine() -> str:
    """Determine which Excel engine to use."""
    try:
        with pd.ExcelWriter('test.xlsx', engine='xlsxwriter') as test_writer:
            pd.DataFrame().to_excel(test_writer)
        os.remove('test.xlsx')
        return 'xlsxwriter'
    except ImportError:
        print("xlsxwriter not found, using openpyxl. Color formatting will not be applied.")
        return 'openpyxl'
    except Exception:
        print("xlsxwriter test failed, falling back to openpyxl.")
        return 'openpyxl'


def _create_xlsxwriter_formats(workbook) -> Dict[str, Any]:
    """Create formatting objects for xlsxwriter."""
    base_cell_properties = {'text_wrap': True, 'valign': 'top', 'border': 1}
    header_base_properties = {'bold': True, 'align': 'center', **base_cell_properties}
    
    return {
        'item_id_header': workbook.add_format({
            **header_base_properties, 
            'fg_color': EXCEL_CONFIG['formats']['item_id_color']
        }),
        'desc_vendor_header': workbook.add_format({
            **header_base_properties, 
            'fg_color': EXCEL_CONFIG['formats']['desc_vendor_color']
        }),
        'attribute_header': workbook.add_format({
            **header_base_properties, 
            'fg_color': EXCEL_CONFIG['formats']['attribute_color']
        }),
        'data_cell': workbook.add_format(base_cell_properties)
    }


def _write_xlsxwriter_excel(
    df: pd.DataFrame, 
    filepath: str, 
    item_id_col: str, 
    description_col: str, 
    vendor_col: str
) -> None:
    """Write DataFrame to Excel using xlsxwriter with formatting."""
    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet(EXCEL_CONFIG['sheets']['extracted_attributes'])
        
        formats = _create_xlsxwriter_formats(workbook)
        
        # Write headers with specific formatting
        for col_num, col_name in enumerate(df.columns):
            if col_name == item_id_col:
                worksheet.write(0, col_num, col_name, formats['item_id_header'])
            elif col_name in [description_col, vendor_col]:
                worksheet.write(0, col_num, col_name, formats['desc_vendor_header'])
            else:  # Attribute columns
                worksheet.write(0, col_num, col_name, formats['attribute_header'])
        
        # Write data cells
        if not df.empty:
            for row_num, row_data in enumerate(df.itertuples(index=False)):
                for col_num, cell_value in enumerate(row_data):
                    if pd.notna(cell_value):
                        worksheet.write(row_num + 1, col_num, cell_value, formats['data_cell'])
                    else:
                        worksheet.write_blank(row_num + 1, col_num, None, formats['data_cell'])
        
        # Auto-adjust column width
        for idx, col in enumerate(df):
            series = df[col]
            header_len = len(str(series.name))
            max_data_len = series.astype(str).map(len).max()
            if pd.isna(max_data_len):
                max_data_len = 0
            max_len = max(header_len, int(max_data_len)) + 2
            worksheet.set_column(idx, idx, max_len)
        
        worksheet.freeze_panes(1, 1)


def write_excel_file(
    df: pd.DataFrame, 
    filepath: str, 
    item_id_col: str, 
    description_col: str, 
    vendor_col: str
) -> None:
    """Write DataFrame to Excel file with appropriate engine."""
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        engine_to_use = _determine_excel_engine()
        
        if engine_to_use == 'xlsxwriter':
            _write_xlsxwriter_excel(df, filepath, item_id_col, description_col, vendor_col)
        else:
            with pd.ExcelWriter(filepath, engine=engine_to_use) as writer:
                df.to_excel(
                    writer, 
                    sheet_name=EXCEL_CONFIG['sheets']['extracted_attributes'], 
                    index=False
                )
        
        print(f"Successfully wrote DataFrame to Excel: {filepath}")
    except Exception as e:
        print(f"Error writing DataFrame to Excel file '{filepath}': {e}")

def extract_case_pack_with_examples(df, agent, default_pack_size=1000):
        example_text = """
        Description: Dome Sip Lid F/10-20Oz Hot Cup Wht 20/50
        Case Pack: 1000

        Description: *Custom* Cp12 - 12Oz American Burger Cup 3 Color Print - 5000/Cs
        Case Pack: 5000

        Description: Pm-Pc3.25Blk - 3.25Oz Black Portion Cup - 20x125/Cs
        Case Pack: 2500

        Description: 64313  Clear Cup Pet Straw Slot Lid For 12-24 98Mm 1000Ca
        Case Pack: 1000

        Description: Kneaders 22Oz Ppr Cold Cup Prnt 1M
        Case Pack: 1000

        Description: Gp Cp10 Dixie 10Oz Cup 20X50Ca Pete Clear Plastic Cold Cup
        Case Pack: 1000
        """

        system_prompt = f"""
        You are to extract the Case Pack quantity from a product description. The Case Pack is the number of items in a case, usually a number like 1000, 500, 250, etc.

        Here are some examples from the data - but these may not be the only ways case pack may be mentioned in the description:
        {example_text}

        Only return the number, no other text. If you cannot find a Case Pack in the description, return 'N/A'.
        """


        user_prompt = (
                "Given this product description: '{Description with Attributes}' "
                "extract the Case Pack number frm the description. Only return the number, no other text. If you cannot find a Case Pack number in the description, return 'N/A'."
        )

            # Use the per-row system prompts
        df = agent.format_df_prompts(df, system_prompt, user_prompt)
        df = agent.run_df_prompts(df)

        # Go through the rows where 'openai_response' is 'N/A' and set it to the value in the 'Case Pack' field (if it exists)
        case_pack_col = PipelineConfig.TRANSACTION_COLUMNS['case_pack']
        if case_pack_col in df.columns:
            # If openai_response is 'N/A' and Case Pack exists, use Case Pack value, removing non-numeric characters
            def clean_case_pack(row):
                if str(row['openai_response']).strip() == 'N/A' and pd.notnull(row.get(case_pack_col)):
                    # Remove all non-numeric characters from Case Pack
                    cleaned = ''.join(filter(str.isdigit, str(row[case_pack_col])))
                    return cleaned if cleaned else row['openai_response']
                else:
                    return row['openai_response']
            df['openai_response'] = df.apply(clean_case_pack, axis=1)

        df = df.drop(columns=[case_pack_col], errors='ignore')
        df = df.rename(columns={'openai_response': case_pack_col})
        df = df.drop(columns=['system_prompt', 'user_prompt'], errors='ignore')
        df[case_pack_col] = df[case_pack_col].replace('N/A', pd.NA)
        df[case_pack_col] = df[case_pack_col].fillna(default_pack_size)

        return df


def attribute_with_ai(im_final, agent, columns_for_description2, prompt_options_string, example_desc, example_output, default_pack_size=1000):
    ps = False
    columns_for_description = columns_for_description2.copy()
    case_pack_col = PipelineConfig.TRANSACTION_COLUMNS['case_pack']
    if 'Pack Size' in columns_for_description2:
        columns_for_description.remove('Pack Size')
        ps = True

    system_prompt = f"""

    You are to pull out information from a product description. 

    Here are the attributes I am looking for:
    {columns_for_description}

    Here is a look at the most common values for each of these attributes. This is not an exhaustive list, but it should help you understand what I am looking for. 
    You probably will need to pull out similar information for an attribute that's not on this list: \n\n
    {prompt_options_string}

    Return in 'Column: Value' format separated by '|'. For example, if the description is "{example_desc}", then return
    {example_output}

    Not everything will be in the description, so you can leave things blank.
    """
    user_prompt = "Pull out the relevant information based on this product description '{Description with Attributes}'"

    im_final = agent.format_df_prompts(im_final, system_prompt, user_prompt)
    im_final = agent.run_df_prompts(im_final)
    im_final = im_final.rename(columns={'openai_response': 'attributes'})
    
    if ps:
        df = extract_case_pack_with_examples(im_final, agent, default_pack_size)

    return df


# =============================================================================
# MAIN ATTRIBUTE EXTRACTION FUNCTION
# =============================================================================

def extract_attributes_to_dataframe(
    df: pd.DataFrame, 
    columns_for_description2: List[str], 
    item_id_col: str = 'Entity--Item',
    description_col: str = "Description with Attributes",
    vendor_col: str = PipelineConfig.TRANSACTION_COLUMNS['vgn'],
    attribute_col: str = 'attributes',
    output_excel_filepath: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]: 
    """
    Convert OpenAI pipe-delimited attribute strings into structured DataFrame columns.
    
    Args:
        df: DataFrame with item data and AI response strings
        columns_for_description: List of attribute names to extract
        item_id_col: Column name for unique item identifiers
        description_col: Column name for item descriptions
        vendor_col: Column name for vendor names
        attribute_col: Column containing OpenAI pipe-delimited strings
        output_excel_filepath: Optional path to save Excel file
    
    Returns:
        Tuple of (extracted_df, merged_df) where extracted_df has structured attributes
        and merged_df is original DataFrame with extracted columns
    """
    # Input validation
    try:
        validate_dataframe(df, "DataFrame")
        validate_list_input(columns_for_description2, "columns_for_description")
        
        required_cols = [item_id_col, description_col, vendor_col, attribute_col]
        validate_columns_exist(df, required_cols, "DataFrame")
    except ValidationError as e:
        print(f"Validation error: {e}")
        empty_df = pd.DataFrame(columns=[item_id_col, description_col, vendor_col] + columns_for_description2)
        return empty_df, pd.DataFrame()

    # Process data
    processed_data = []
    print(f"Processing {len(df)} rows from the input DataFrame...")

    columns_for_description = columns_for_description2.copy()
    case_pack_col = PipelineConfig.TRANSACTION_COLUMNS['case_pack']
    if 'Pack Size' in columns_for_description:
        columns_for_description.remove('Pack Size')
        columns_for_description.append(case_pack_col)
        # add | Case Pack to attributes 
        df[attribute_col] = df[attribute_col] + ' | Case Pack: ' + df[case_pack_col].astype(str)

    
    for index, row in df.iterrows():
        entity_id = row[item_id_col]
        desc_val = row.get(description_col, '')
        vendor_val = row.get(vendor_col, '')
        response_string = row[attribute_col]

        parsed_attributes = parse_openai_response(response_string)
        new_row_data = {
            item_id_col: entity_id,
            description_col: desc_val,
            vendor_col: vendor_val
        }
        
        for attribute_name in columns_for_description:
            new_row_data[attribute_name] = parsed_attributes.get(attribute_name, None) 
        
        processed_data.append(new_row_data)

    output_df = pd.DataFrame(processed_data)
    
    # Ensure all specified columns_for_description are present
    for attribute_name in columns_for_description:
        if attribute_name not in output_df.columns:
            output_df[attribute_name] = None
            
    # Define final column order
    final_columns_order = [item_id_col, description_col, vendor_col]
    final_columns_order.extend([
        col for col in columns_for_description 
        if col in output_df.columns and col not in [item_id_col, description_col, vendor_col]
    ])
    
    if not output_df.empty and final_columns_order:
        existing_final_columns = [col for col in final_columns_order if col in output_df.columns]
        output_df = output_df[existing_final_columns]
    elif output_df.empty: 
        output_df = pd.DataFrame(columns=final_columns_order if final_columns_order else [item_id_col])

    print(f"Successfully created DataFrame with {len(output_df)} rows and {len(output_df.columns)} columns.")

    # Write to Excel if filepath is provided
    if output_excel_filepath:
        write_excel_file(output_df, output_excel_filepath, item_id_col, description_col, vendor_col)
            
    # Clean and merge data
    df_cleaned = df.drop(columns=[col for col in columns_for_description if col in df.columns])
    
    # Merge extracted columns back into df_cleaned
    merged_df = pd.merge(
        df_cleaned,
        output_df[[item_id_col] + columns_for_description],
        on=item_id_col,
        how='left'
    )
    
    return merged_df, output_df

# =============================================================================
# COVERAGE ANALYSIS FUNCTIONS
# =============================================================================

def _calculate_coverage_percentage(
    df: pd.DataFrame, 
    total_rows: int, 
    exclude_nulls: bool = False
) -> Dict[str, float]:
    """
    Calculate coverage percentage for each column in DataFrame.
    
    Args:
        df: DataFrame to calculate coverage for
        total_rows: Total number of rows to use as denominator
        exclude_nulls: Whether to exclude null values from count
    
    Returns:
        Dictionary of column names and their coverage percentages
    """
    coverage = {}
    for col in df.columns:
        non_empty_count = df[col].astype(str).apply(lambda x: x.strip() != '').sum()
        if exclude_nulls:
            non_empty_count -= df[col].isna().sum()
        coverage[col] = (non_empty_count / total_rows) * 100 if total_rows > 0 else 0
    return coverage


def _create_coverage_comparison(
    coverage_post_llm: Dict[str, float],
    coverage_initial: Dict[str, float]
) -> pd.DataFrame:
    """Create comparison DataFrame from coverage dictionaries."""
    coverage_post_series = pd.Series(coverage_post_llm, name='% Coverage Post LLM')
    coverage_initial_series = pd.Series(coverage_initial, name='% Initial Coverage')
    
    # Combine and format results
    comparison_df = pd.concat([coverage_post_series, coverage_initial_series], axis=1)
    comparison_df.index.name = 'Column_Name'
    comparison_df['Difference'] = comparison_df['% Coverage Post LLM'] - comparison_df['% Initial Coverage']
    
    # Format percentages
    for col in ['% Coverage Post LLM', '% Initial Coverage', 'Difference']:
        comparison_df[col] = comparison_df[col].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A'
        )
    
    comparison_df.fillna('0.00%', inplace=True)
    return comparison_df


def coverage_improvement(
    sfy2: pd.DataFrame, 
    im_final: pd.DataFrame, 
    extract_attributes_df: pd.DataFrame, 
    columns_for_description2: List[str]
) -> pd.DataFrame:
    """
    Compare data coverage before and after LLM enrichment.
    
    Args:
        sfy: Original data with partially filled columns
        im_final: Filter reference DataFrame with Entity--Item column
        extract_attributes_df: Output of attribute extraction via LLM
        columns_for_description: List of columns to compare
    
    Returns:
        DataFrame with pre/post coverage percentages and improvement
    """
    print("Computing coverage improvement analysis...")
    
    # Validate inputs
    try:
        validate_dataframe(sfy2, "sfy")
        validate_dataframe(im_final, "im_final")
        validate_dataframe(extract_attributes_df, "extract_attributes_df")
        validate_list_input(columns_for_description2, "columns_for_description")
    except ValidationError as e:
        print(f"Validation error: {e}")
        return pd.DataFrame()

    columns_for_description = columns_for_description2.copy()
    case_pack_col = PipelineConfig.TRANSACTION_COLUMNS['case_pack']
    if 'Pack Size' in columns_for_description:
        columns_for_description.remove('Pack Size')
        columns_for_description.append(case_pack_col)
    
    sfy = sfy2.copy()
    # Rename 'Pack Size' column to 'Case Pack' if it exists in sfy
    if 'Pack Size' in sfy.columns:
        sfy = sfy.rename(columns={'Pack Size': case_pack_col})
    # Get sfy rows that exist in im_final
    item_id_col = 'Entity--Item'
    sfy_in_im_final = sfy[sfy[item_id_col].isin(im_final[item_id_col])].copy()
    sfy_in_im_final = sfy_in_im_final[columns_for_description + [item_id_col]].copy()
    
    # Compute post-LLM coverage
    total_rows_extract_attr = len(extract_attributes_df)
    if total_rows_extract_attr == 0:
        print("Warning: extract_attributes_df is empty")
        return pd.DataFrame()
    
    coverage_post_llm = _calculate_coverage_percentage(extract_attributes_df, total_rows_extract_attr)
    print(f"Computed post-LLM coverage for {len(coverage_post_llm)} columns")
    
    # Compute initial coverage from sfy
    if sfy_in_im_final.empty:
        print("Warning: No matching items found in sfy")
        return pd.DataFrame()
    
    coverage_initial = _calculate_coverage_percentage(
        sfy_in_im_final, total_rows_extract_attr, exclude_nulls=True
    )
    print(f"Computed initial coverage for {len(coverage_initial)} columns")
    
    # Create comparison DataFrame
    comparison_df = _create_coverage_comparison(coverage_post_llm, coverage_initial)
    
    # Remove the first 3 rows (likely metadata) and return
    result_df = comparison_df.tail(-3)
    print(f"Coverage improvement analysis complete. Found {len(result_df)} columns to compare.")
    
    return result_df 