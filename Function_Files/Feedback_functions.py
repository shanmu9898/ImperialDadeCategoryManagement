import pandas as pd
import os
import re
from typing import List, Tuple, Optional, Dict, Any
from pyvent.tools.llm.openai_api import OpenAIAgent
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from jinja2 import Template
import Levenshtein
import numpy as np
import faiss
import json
import ast

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

# =============================================================================
# FEEDBACK CONSOLIDATION FUNCTIONS
# =============================================================================

def consolidate_feedback_from_excel(
    excel_filepaths: List[str], 
    im_final: pd.DataFrame
) -> pd.DataFrame:
    """
    Consolidate feedback data from multiple Excel files into a single DataFrame.
    
    Reads multiple Excel files, processes each sheet (excluding specific ones),
    and consolidates Targets, Subs, Correct, Feedback, Match Type, and Subcategory
    information for substitute items. Stops processing when blank rows are encountered.
    
    Args:
        excel_filepaths: List of file paths to Excel files containing feedback data
        im_final: DataFrame containing item details with Entity--Item, embeddings columns
    
    Returns:
        DataFrame with consolidated substitute data including embeddings
    """
    consolidated_data = []
    sheets_to_ignore = {"Summary", "_VendorCalcHelper"}
    
    total_files = len(excel_filepaths)
    files_processed_count = 0
    sheets_processed_count = 0

    print(f"--- Starting Consolidation Process for {total_files} Excel Files ---")

    for i, excel_filepath in enumerate(excel_filepaths):
        print(f"\n[{i+1}/{total_files}] Processing file: {os.path.basename(excel_filepath)}")
        
        if not os.path.exists(excel_filepath):
            print(f"  [ERROR] File not found. Skipping.")
            continue

        try:
            xls = pd.ExcelFile(excel_filepath)
            files_processed_count += 1
        except Exception as e:
            print(f"  [ERROR] Could not open or read file: {e}. Skipping.")
            continue

        all_sheet_names_in_file = xls.sheet_names
        print(f"  [INFO] Found {len(all_sheet_names_in_file)} sheets.")

        for sheet_name in all_sheet_names_in_file:
            if sheet_name in sheets_to_ignore:
                print(f"    - Skipping ignored sheet: '{sheet_name}'")
                continue
            
            try:
                df_sheet = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                sheets_processed_count += 1
                
                if df_sheet.empty:
                    print(f"    - [WARNING] Sheet '{sheet_name}' is empty. Skipping.")
                    continue

                if 'Entity--Item' not in df_sheet.columns:
                    print(f"    - [WARNING] 'Entity--Item' column not found in sheet '{sheet_name}'. Skipping.")
                    continue
                
                if len(df_sheet) < 2:
                    print(f"    - [WARNING] Sheet '{sheet_name}' has a target but no substitute rows. Skipping.")
                    continue
                
                target_item_id = df_sheet.iloc[0]['Entity--Item']
                subs_found_in_sheet = 0

                for idx, sub_row in df_sheet.iloc[1:].iterrows():
                    sub_item_id_val = sub_row.get('Entity--Item')

                    if pd.isna(sub_item_id_val) or (isinstance(sub_item_id_val, str) and not sub_item_id_val.strip()):
                        print(f"    - [INFO] Blank row found. Concluding substitute processing for '{sheet_name}'.")
                        break
                    
                    subs_found_in_sheet += 1
                    consolidated_data.append({
                        'Targets': target_item_id,
                        'Subs': sub_item_id_val,
                        'Correct': sub_row.get('Accept/Reject'),
                        'Feedback': sub_row.get('Feedback'),
                        'Match Type': sub_row.get('Match Type'),
                        'Subcategory': sub_row.get('Subcategory')
                    })
                
                print(f"    - [SUCCESS] Processed sheet '{sheet_name}', found {subs_found_in_sheet} substitutes for target '{target_item_id}'.")

            except Exception as e:
                print(f"    - [ERROR] Failed to process sheet '{sheet_name}': {e}. Skipping.")
                continue
            
    print("\n--- Consolidation Summary ---")
    if not consolidated_data:
        print("[WARNING] No substitute data was consolidated from any file.")
        return pd.DataFrame()
        
    print(f"[INFO] Files processed: {files_processed_count}/{total_files}")
    print(f"[INFO] Sheets processed: {sheets_processed_count}")
    print(f"[INFO] Total substitute records found: {len(consolidated_data)}")
    
    feedback_summary_df = pd.DataFrame(consolidated_data)
    
    print("[INFO] Merging feedback data with item master for embeddings...")
    details_lookup = im_final[['Entity--Item', 'embeddings', 'for_embedding']].copy()

    target_lookup = details_lookup.rename(columns={'embeddings': 'Target_embeddings', 'for_embedding': 'Target_for_embeddings'})
    sub_lookup = details_lookup.rename(columns={'embeddings': 'Sub_embeddings', 'for_embedding': 'Sub_for_embeddings'})

    merged_df = pd.merge(feedback_summary_df, target_lookup, left_on='Targets', right_on='Entity--Item', how='left').drop(columns=['Entity--Item'], errors='ignore')
    merged_df = pd.merge(merged_df, sub_lookup, left_on='Subs', right_on='Entity--Item', how='left').drop(columns=['Entity--Item'], errors='ignore')
    
    print("[SUCCESS] Merging complete.")
    print("--------------------------------\n")
            
    return merged_df

# =============================================================================
# FEEDBACK ANALYSIS FUNCTIONS
# =============================================================================

def get_feedback_summary(im_final_with_feedback: pd.DataFrame) -> pd.DataFrame:
    """
    Get a summary of the feedback data.
    """
    # Count values in the 'Correct' column across the entire dataset
    counts = im_final_with_feedback['Correct'].value_counts()

    # Convert to DataFrame
    summary_df = counts.to_frame(name='Count').T

    # Ensure all relevant columns are present and numeric, fill missing with 0
    for col in ['Accept', 'Consider', 'Reject']:
        if col not in summary_df.columns:
            summary_df[col] = 0
    summary_df[['Accept', 'Consider', 'Reject']] = summary_df[['Accept', 'Consider', 'Reject']].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Calculate Accept Rate
    total = summary_df[['Accept', 'Consider', 'Reject']].sum(axis=1)
    summary_df['Accept Rate'] = summary_df['Accept'] / total.replace(0, 1)
    summary_df['Accept Rate'] = summary_df['Accept Rate'].apply(lambda x: f"{x:.2%}")

    # Calculate Accept + Consider Rate
    summary_df['Accept/Consider Rate'] = (summary_df['Accept'] + summary_df['Consider']) / total.replace(0, 1)
    summary_df['Accept/Consider Rate'] = summary_df['Accept/Consider Rate'].apply(lambda x: f"{x:.2%}")

    # Export to clipboard
    summary_df
    return summary_df

def get_reviewed_coverage(
    im_final: pd.DataFrame,
    im_final_with_feedback: pd.DataFrame
) -> pd.DataFrame:
    """
    Get a summary of the reviewed coverage using current column structure.
    
    Args:
        im_final: DataFrame with original data including 'Net Cost', 'Qty', 'VB Flag'
        im_final_with_feedback: DataFrame with feedback data including 'Targets' column
        
    Returns:
        DataFrame with coverage metrics for Private Label items
    """
    # Create coverage dataframe using current columns: Net Cost, Qty, Gross Cost
    coverage = pd.DataFrame({
        'Net Cost': [im_final[im_final[TRANSACTION_COLUMNS['vb_flag']] == 'Y - VB'][TRANSACTION_COLUMNS['net_cost']].sum()],
        'Qty': [im_final[im_final[TRANSACTION_COLUMNS['vb_flag']] == 'Y - VB'][TRANSACTION_COLUMNS['qty']].sum()],
        'Gross Cost': [im_final[im_final[TRANSACTION_COLUMNS['vb_flag']] == 'Y - VB'][TRANSACTION_COLUMNS['gross_cost']].sum()],
        'GM': [im_final[im_final[TRANSACTION_COLUMNS['vb_flag']] == 'Y - VB'][TRANSACTION_COLUMNS['gross_cost']].sum() - im_final[im_final[TRANSACTION_COLUMNS['vb_flag']] == 'Y - VB'][TRANSACTION_COLUMNS['net_cost']].sum()]
    })
    # Name the row Private Label
    coverage.index = ['Private Label']

    # Get reviewed PL SKUs metrics
    reviewed_pl_items = im_final[im_final['Entity--Item'].isin(im_final_with_feedback['Targets'].unique())]
    coverage.loc['Reviewed PL SKUs'] = [
        reviewed_pl_items[TRANSACTION_COLUMNS['net_cost']].sum(),
        reviewed_pl_items[TRANSACTION_COLUMNS['qty']].sum(),
        reviewed_pl_items[TRANSACTION_COLUMNS['gross_cost']].sum(),
        reviewed_pl_items[TRANSACTION_COLUMNS['gross_cost']].sum() - reviewed_pl_items[TRANSACTION_COLUMNS['net_cost']].sum()
    ]
    
    coverage = coverage.T
    
    # Calculate percentages
    coverage['% Reviewed PL SKUs'] = coverage['Reviewed PL SKUs'] / coverage['Private Label'].replace(0, 1)
    coverage['% Reviewed PL SKUs'] = coverage['% Reviewed PL SKUs'].apply(lambda x: f"{x:.2%}")
    
    # Add Total column (all items, not just PL)
    coverage['Total'] = [
        im_final[TRANSACTION_COLUMNS['net_cost']].sum(),
        im_final[TRANSACTION_COLUMNS['qty']].sum(),
        im_final[TRANSACTION_COLUMNS['gross_cost']].sum(),
        im_final[TRANSACTION_COLUMNS['gross_cost']].sum() - im_final[TRANSACTION_COLUMNS['net_cost']].sum()
    ]
    
    coverage['% Reviewed of Total'] = coverage['Reviewed PL SKUs'] / coverage['Total'].replace(0, 1)
    coverage['% Reviewed of Total'] = coverage['% Reviewed of Total'].apply(lambda x: f"{x:.2%}")
    
    # Round numeric columns and reorder
    coverage = coverage.round(2)
    coverage = coverage[['Reviewed PL SKUs', 'Private Label', 'Total', '% Reviewed PL SKUs', '% Reviewed of Total']]
    
    return coverage

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def _process_attributes(attr_string: str) -> List[str]:
    """
    Parses a pipe-delimited string of attributes into a list.

    Takes a single string where attributes are separated by a pipe ('|')
    and splits it into a list of individual attribute strings. Each item
    in the resulting list is also stripped of whitespace.

    Args:
        attr_str (str): The pipe-delimited string of attributes to process.

    Returns:
        list: A list of attribute strings. Returns an empty list if the
              input is not a valid string or is empty.
    """
    if not isinstance(attr_string, str) or not attr_string.strip():
        return []

    # First, split the string by the '|' delimiter. This will incorrectly
    # split values that also contain pipes, which we will fix next.
    raw_parts = [part.strip() for part in attr_string.split('|') if part.strip()]

    if not raw_parts:
        return []

    # Iterate through the raw parts and rejoin any that were part of a value.
    rejoined_parts = []
    for part in raw_parts:
        # A part containing a colon is a new key-value pair.
        # A part WITHOUT a colon is a continuation of the previous value.
        if ':' in part:
            rejoined_parts.append(part)
        elif rejoined_parts:
            # This is a value fragment, so append it back to the last attribute.
            rejoined_parts[-1] += '|' + part

    # Now, process the correctly formed attribute strings.
    clean_attributes = []
    for part in rejoined_parts:
        key, value = part.split(':', 1)
        key = key.strip()
        value = value.strip()
        # Check for non-meaningful values
        if value and value.lower() not in ['null', 'none', 'n/a', '']:
            clean_attributes.append(f"{key}: {value}")
    
    return clean_attributes
               
def summarize_prompt_into_rules(prompt_text: str) -> str:
    """
    Summarizes a formatted text block into a concise set of rules.

    This function is intended to take a detailed text block describing an item
    and its feedback examples and distill it into a short, clear set of
    swapping rules (e.g., "Must be the same brand," "Color can be different").
    In a real implementation, this would likely call an external AI model.

    Args:
        prompt_text (str): The formatted text to be summarized.

    Returns:
        str: A string containing the summarized swapping rules.
    """
    model = 'gpt-4o-mini'
    chunk_size = 32
    agent = OpenAIAgent(model=model, chunk_size=chunk_size)
    
    if not prompt_text:
        return ""
    system_prompt = (
        "You are a rule-extraction expert. Based on the provided target item and its list of "
        "substitutes, summarize the criteria for what makes a good or bad substitute. "
        " The rules must be general to apply to other situations where the actual values might be different. So for example, if they say two products are not a match because one is green and the other is blue, "
        "the rule should be something like 'The color of the substitute should match the target item' and absoluterly not 'The substitute should be green'. "
        "The final output must be a single JSON object with one key, \"rules\", "
        "which contains a de-duplicated list of the final rule strings."
    )
    try:
        llm_prompts = agent.generate_prompts(system_prompt, [prompt_text])
        response_json = agent.get_responses(llm_prompts, temperature=0.5, response_format={"type": "json_object"})
        if response_json and isinstance(response_json, list) and len(response_json) > 0:
            final_response = response_json[0]
            if isinstance(final_response, str):
                final_response = json.loads(final_response)
            if isinstance(final_response, dict) and 'rules' in final_response and isinstance(final_response['rules'], list):
                rules_list = list(dict.fromkeys(final_response['rules']))
                return "\n".join(f"- {rule}" for rule in rules_list)
    except Exception as e:
        print(f"Error during LLM call or parsing: {e}")
        return ""
    return ""

# =============================================================================
# AI PROCESSING FUNCTIONS
# =============================================================================

def _format_examples_for_rules(target_desc: str, examples: List[str]) -> str:
    """
    Formats feedback examples into a text block for rule generation.

    This function takes the description of a target item and a list of
    feedback examples (e.g., accepted/rejected swaps) and formats them
    into a single, coherent string. This string is intended to be used
    as input for another function to summarize into a set of swapping rules.

    Args:
        target_desc (str): The description of the target item.
        feedback_examples (list[dict]): A list of dictionaries, with each
                                         dictionary representing a feedback example.

    Returns:
        str: A formatted text block ready for rule summarization.
    """
    prompt_lines = [f"Target Description: {target_desc}", "Potential Sub:"]
    for ex in examples:
        # Assuming examples are dicts from to_dict('records')
        sub_desc = ex.get('Sub_for_embeddings', ex.get('sub_description', 'N/A'))
        status = ex.get('Correct', 'N/A')
        feedback = ex.get('Feedback', '')
        
        prompt_lines.append(f" - {_clean_str(sub_desc)}")
        output_line = f"   - Accept/Reject: {_clean_str(status)}"
        if str(status).lower() == 'reject' and feedback:
            output_line += f" - Reasoning: {_clean_str(feedback)}"
        prompt_lines.append(output_line)
    return "\n".join(prompt_lines)

def _clean_str(text: str) -> str:
    """
    Safely converts an input to a cleaned string.

    This function converts the input value to its string representation and removes
    any leading or trailing whitespace. If the input is null (e.g., None, NaN),
    it returns an empty string.

    Args:
        text (any): The input value to be cleaned.

    Returns:
        str: The cleaned string, or an empty string if the input is null.
    """
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text).strip()

# =============================================================================
# MAIN MATCHING FUNCTIONS
# =============================================================================

def user_prompts_with_rules(
    im_final_full2: pd.DataFrame,
    merged_df: pd.DataFrame,
    n: int = 10,
    vendor_exclusion: bool = False,
    use_faiss: bool = True,
    use_levenshtein: bool = True,
    use_subcategory_rules: bool = False,
    subcategory_col: str = 'Subcategory'
) -> Tuple[List[str], Dict[int, str]]:
    """
    Generates formatted text prompts for an AI model to determine item swappability.
    
    This version can generate either unified rules from all feedback examples or
    subcategory-specific rules based on the use_subcategory_rules parameter.
    
    Args:
        im_final_full2: DataFrame with item data and embeddings
        merged_df: DataFrame with feedback data
        n: Number of similar items to include per prompt
        vendor_exclusion: Whether to exclude same-vendor matches
        use_faiss: Whether to use FAISS for similarity search
        use_levenshtein: Whether to use Levenshtein similarity filtering
        use_subcategory_rules: If True, generate subcategory-specific rules instead of unified rules
        subcategory_col: Column name containing subcategory information
    
    Returns:
        Tuple of (user_prompts, numerical_id_to_entity_id_map)
    """
    im_final_full = im_final_full2.copy()
    
    if use_subcategory_rules:
        print("Starting user_prompts_with_rules function with subcategory-specific rule generation...")
    else:
        print("Starting user_prompts_with_rules function with unified rule generation...")

    # --- Embedding Processing ---
    if 'embeddings' in im_final_full.columns:
        im_final_full['embeddings'] = im_final_full['embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str) else x)
        im_final_full.dropna(subset=['embeddings'], inplace=True)
    if isinstance(merged_df, pd.DataFrame) and 'Target_embeddings' in merged_df.columns:
        merged_df['Target_embeddings'] = merged_df['Target_embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str) else x)
        merged_df.dropna(subset=['Target_embeddings'], inplace=True)

    # --- Create Numerical ID Mappings ---
    im_final_full = im_final_full.reset_index(drop=True)
    im_final_full['Numerical_ID'] = np.arange(1, len(im_final_full) + 1)
    numerical_id_to_entity_id_map = dict(zip(im_final_full['Numerical_ID'], im_final_full['Entity--Item']))

    # --- Data Preparation ---
    # Check if embeddings column exists, otherwise use openai_response
    embedding_col = 'embeddings' if 'embeddings' in im_final_full.columns else 'attributes'
    
    if embedding_col not in im_final_full.columns:
        print(f"Critical: Neither 'embeddings' nor 'attributes' column found in DataFrame.")
        print(f"Available columns: {list(im_final_full.columns)}")
        return [], {}
    
    # Create mask based on column type
    if embedding_col == 'embeddings':
        valid_embeddings_mask = im_final_full[embedding_col].apply(lambda x: isinstance(x, np.ndarray))
    else:
        # For openai_response column, check for non-empty strings
        valid_embeddings_mask = im_final_full[embedding_col].notna() & (im_final_full[embedding_col] != '')
    
    im_final_full = im_final_full[valid_embeddings_mask].copy()
    if im_final_full.empty:
        print(f"Critical: No valid embeddings in '{embedding_col}' column.")
        return [], {}

    # --- FAISS & Levenshtein Preparation ---
    embeddings = np.stack(im_final_full['embeddings'].values).astype(np.float32)
    faiss.normalize_L2(embeddings)
    if use_faiss:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    levenshtein_col = 'for_embedding' if 'for_embedding' in im_final_full.columns else 'description'
    im_final_full[levenshtein_col] = im_final_full.get(levenshtein_col, "").apply(_clean_str)
    all_im_final_levenshtein_texts = im_final_full[levenshtein_col].tolist()

    def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
        if not s1 and not s2: return 1.0
        if not s1 or not s2: return 0.0
        distance = Levenshtein.distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len) if max_len else 1.0

    # --- Rule Generation (MODIFIED BLOCK) ---
    if use_subcategory_rules:
        # Generate subcategory-specific rules
        subcategory_rules_dict = {}
        if isinstance(merged_df, pd.DataFrame) and not merged_df.empty and subcategory_col in merged_df.columns:
            print(f"Generating subcategory-specific rules from feedback examples...")
            
            # Get unique subcategories
            unique_subcategories = merged_df[subcategory_col].dropna().unique()
            print(f"Found {len(unique_subcategories)} unique subcategories: {unique_subcategories}")
            
            for subcategory in unique_subcategories:
                print(f"Processing subcategory: {subcategory}")
                subcategory_feedback = merged_df[merged_df[subcategory_col] == subcategory]
                
                if not subcategory_feedback.empty:
                    # Concatenate all feedback for this subcategory into a single prompt string
                    subcategory_feedback_prompt_lines = []
                    for target_id in subcategory_feedback['Targets'].unique():
                        target_examples = subcategory_feedback[subcategory_feedback['Targets'] == target_id]
                        if not target_examples.empty:
                            target_desc = _clean_str(target_examples['Target_for_embeddings'].iloc[0])
                            feedback_examples = target_examples.to_dict('records')
                            subcategory_feedback_prompt_lines.append(_format_examples_for_rules(target_desc, feedback_examples))
                    
                    if subcategory_feedback_prompt_lines:
                        # Join all formatted examples for this subcategory into one text block
                        subcategory_prompt_text = "\n\n".join(subcategory_feedback_prompt_lines)
                        subcategory_rules = summarize_prompt_into_rules(subcategory_prompt_text)
                        subcategory_rules_dict[subcategory] = subcategory_rules
                        rule_count = len(subcategory_rules.split('\n')) if subcategory_rules else 0
                        print(f"Generated {rule_count} rules for subcategory: {subcategory}")
                else:
                    print(f"No feedback data found for subcategory: {subcategory}")
        else:
            print(f"Warning: No feedback data or '{subcategory_col}' column found. Using empty subcategory rules.")
        
        # Create a mapping from item ID to subcategory for quick lookup
        item_to_subcategory_map = {}
        if subcategory_col in im_final_full.columns:
            item_to_subcategory_map = dict(zip(im_final_full['Entity--Item'], im_final_full[subcategory_col]))
        
        unified_rules = ""  # Not used when use_subcategory_rules=True
    else:
        # Generate unified rules (original behavior)
        unified_rules = ""
        subcategory_rules_dict = {}
        item_to_subcategory_map = {}
        
        if isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
            print("Generating one unified set of rules from all feedback examples...")
            # Concatenate all feedback into a single prompt string
            all_feedback_prompt_lines = []
            for target_id in merged_df['Targets'].unique():
                all_examples = merged_df[merged_df['Targets'] == target_id]
                if not all_examples.empty:
                    target_desc = _clean_str(all_examples['Target_for_embeddings'].iloc[0])
                    feedback_examples = all_examples.to_dict('records')
                    all_feedback_prompt_lines.append(_format_examples_for_rules(target_desc, feedback_examples))
            
            if all_feedback_prompt_lines:
                # Join all formatted examples into one large text block
                unified_prompt_text = "\n\n".join(all_feedback_prompt_lines)
                unified_rules = summarize_prompt_into_rules(unified_prompt_text)
                print("Unified rule generation complete.")

    # --- Jinja2 Template (remains the same) ---
    template_str = """
Original Item:
Entity ID: {{ eid_numerical }}
{% if description_display and description_display.strip() -%}
 - Description: {{ description_display.strip() }}
{% endif -%}
{% if attributes_list %}
{% for attr in attributes_list -%}
 - {{ attr }}
{% endfor %}
{%- endif %}

Top Similar Items from Descriptions:
{% for item in top_items %}
Entity ID:  {{ item['Numerical_ID'] }}
 - Description: {{ item['description_display'].strip() }}
{%- if item['attributes_list'] %}
{% for attr in item['attributes_list'] -%}
 - {{ attr }}
{% endfor %}
{%- endif %}
{% if not loop.last %}

{% endif %}
{% endfor %}
{%- if generated_rules %}

---
**Rules based on feedback:**
{{ generated_rules }}
---
{%- endif %}

Return a list of Entity IDs that are swappable with the given Entity ID from the list of similar items.
If an item doesn't seem like it is swappable you can replace it with None.
"""
    jinja_template = Template(template_str, trim_blocks=True, lstrip_blocks=True)

    # --- Main Loop ---
    user_prompts = []
    for idx, row in tqdm(im_final_full.iterrows(), total=im_final_full.shape[0], desc="Generating Prompts"):
        current_embedding_np = embeddings[idx].reshape(1, -1)
        current_lev_text = all_im_final_levenshtein_texts[idx]
        current_vendor = row.get('vgn_name')
        
        # --- Find Similar Items (remains the same) ---
        if use_faiss:
            _, sim_indices = index.search(current_embedding_np, n + 20)
            sim_indices = sim_indices[0]
        else:
            sims = cosine_similarity(current_embedding_np, embeddings)[0]
            sim_indices = np.argsort(sims)[::-1][:n + 20]
        top_items_data = []
        for sim_idx in sim_indices:
            if len(top_items_data) >= n:
                break
            if sim_idx == idx:
                continue
            candidate_row = im_final_full.iloc[sim_idx]
            if vendor_exclusion and current_vendor is not None and candidate_row.get('vgn_name') == current_vendor:
                continue
            if use_levenshtein:
                lev_score = normalized_levenshtein_similarity(current_lev_text, all_im_final_levenshtein_texts[sim_idx])
                if lev_score < 0.5:
                    continue
            top_items_data.append({
                'Numerical_ID': candidate_row['Numerical_ID'],
                'description_display': _clean_str(candidate_row.get('description', '')),
                'attributes_list': _process_attributes(candidate_row.get('openai_response', ''))
            })
        
        # --- Determine which rules to use for this prompt ---
        if use_subcategory_rules:
            # Get the subcategory for this item
            current_item_id = row['Entity--Item']
            current_subcategory = item_to_subcategory_map.get(current_item_id)
            
            # Get the appropriate rules for this subcategory
            if current_subcategory and current_subcategory in subcategory_rules_dict:
                rules_for_prompt = subcategory_rules_dict[current_subcategory]
                rule_count = len(rules_for_prompt.split('\n')) if rules_for_prompt else 0
                print(f"Using {rule_count} rules for subcategory: {current_subcategory}")
            else:
                rules_for_prompt = ""
                if current_subcategory:
                    print(f"No rules found for subcategory: {current_subcategory}")
                else:
                    print(f"No subcategory found for item: {current_item_id}")
        else:
            # Use unified rules (original behavior)
            rules_for_prompt = unified_rules
        
        # --- Render the template with the appropriate rules ---
        prompt = jinja_template.render(
            eid_numerical=row['Numerical_ID'],
            description_display=_clean_str(row.get('description', '')),
            attributes_list=_process_attributes(row.get('openai_response')),
            top_items=top_items_data,
            generated_rules=rules_for_prompt
        )
        user_prompts.append(prompt)
    
    print(f"Generated {len(user_prompts)} prompts.")
    return user_prompts, numerical_id_to_entity_id_map

# =============================================================================
# MATCH UPDATE FUNCTIONS
# =============================================================================

def update_matches_based_on_feedback_sym(
    im_final_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    feedback_target_col: str = 'Targets',
    feedback_sub_col: str = 'Subs',
    feedback_correct_col: str = 'Accept/Reject',
    main_id_col: str = 'Entity--Item',
    main_matches_col: str = 'Matches',
    accept_value: str = 'Yes',
    consider_value: str = 'Consider',
    reject_value: str = 'No'
) -> pd.DataFrame:
    """
    Update Matches lists in im_final_df based on feedback using strict target-centric rules.
    
    Logic:
    1. For Targets: Matches list contains only Subs with 'Accept' or 'Consider' feedback
    2. For non-Targets: Matches to Targets are kept only if (Target, non-Target) was accepted
    3. Matches between non-Targets are never affected
    
    Args:
        im_final_df: DataFrame with main item data
        feedback_df: DataFrame with feedback data
        feedback_target_col: Column for Target IDs in feedback_df
        feedback_sub_col: Column for Substitute IDs in feedback_df
        feedback_correct_col: Column for Accept/Reject status in feedback_df
        main_id_col: ID column in im_final_df
        main_matches_col: Matches list column in im_final_df
        accept_value: Value for accepted matches
        consider_value: Value for considered matches
        reject_value: Value for rejected matches
    
    Returns:
        New DataFrame with updated Matches lists
    """
    print("Starting update with strict target-centric rules...")
    
    im_final_updated = im_final_df.copy() 

    # --- Input Validation and Preprocessing ---
    if not isinstance(im_final_updated, pd.DataFrame):
        raise ValueError("im_final_df must be a pandas DataFrame.")
    if not isinstance(feedback_df, pd.DataFrame):
        raise ValueError("feedback_df must be a pandas DataFrame.")

    required_main_cols = [main_id_col, main_matches_col]
    for col in required_main_cols:
        if col not in im_final_updated.columns:
            raise ValueError(f"Column '{col}' missing in im_final_df.")

    required_feedback_cols = [feedback_target_col, feedback_sub_col, feedback_correct_col]
    for col in required_feedback_cols:
        if col not in feedback_df.columns:
            raise ValueError(f"Column '{col}' missing in feedback_df.")

    # Ensure all IDs in main_df are strings for consistent dictionary keys
    im_final_updated[main_id_col] = im_final_updated[main_id_col].astype(str)
    
    # Ensure Matches column contains lists of strings from original im_final_df
    def _ensure_list_of_strings(matches_val: Any) -> List[str]:
        if isinstance(matches_val, list):
            return [str(m) for m in matches_val]
        elif isinstance(matches_val, str):
            try:
                if matches_val.strip().startswith('[') and matches_val.strip().endswith(']'):
                    parsed_val = ast.literal_eval(matches_val)
                    if isinstance(parsed_val, list):
                        return [str(m) for m in parsed_val]
                    else:
                        return []
                else:
                    return []
            except (ValueError, SyntaxError, TypeError):
                return []
        elif pd.isna(matches_val): 
            return []
        else:
            return []

    im_final_updated[main_matches_col] = im_final_updated[main_matches_col].apply(_ensure_list_of_strings)

    # Create a working dictionary of current matches (as sets for efficient ops)
    current_matches_dict = {
        sku_id: set(matches_list) 
        for sku_id, matches_list in im_final_updated.set_index(main_id_col)[main_matches_col].items()
    }

    # Prepare feedback DataFrame: drop NaNs in key columns, ensure string types
    feedback_df_cleaned = feedback_df.dropna(subset=[feedback_target_col, feedback_sub_col, feedback_correct_col]).copy()
    feedback_df_cleaned[feedback_target_col] = feedback_df_cleaned[feedback_target_col].astype(str)
    feedback_df_cleaned[feedback_sub_col] = feedback_df_cleaned[feedback_sub_col].astype(str)
    
    # --- Step 1: Identify all unique Targets from feedback_df ---
    all_targets_in_feedback = set(feedback_df_cleaned[feedback_target_col].unique())
    print(f"Identified {len(all_targets_in_feedback)} unique Targets in feedback.")

    # --- Step 2: Build the definitive set of *allowed* symmetric pairs from feedback ---
    # This set stores (ID1, ID2) tuples in canonical order (ID1 < ID2) for allowed matches.
    # This is the "global truth" for which explicit relationships are permitted.
    definitive_symmetric_allowed_pairs = set()

    print(f"Processing {len(feedback_df_cleaned)} feedback entries to build allowed pairs...")
    for _, row in feedback_df_cleaned.iterrows():
        target_id_str = row[feedback_target_col]
        sub_id_str = row[feedback_sub_col]
        feedback_status = row[feedback_correct_col]

        # Canonicalize the pair (min_id, max_id) for consistent lookup, ignoring feedback direction
        canonical_pair = tuple(sorted((target_id_str, sub_id_str)))

        if feedback_status == accept_value or feedback_status == consider_value:
            definitive_symmetric_allowed_pairs.add(canonical_pair)
        elif feedback_status == reject_value:
            # Rejection overrides any previous accept/consider for the same pair
            definitive_symmetric_allowed_pairs.discard(canonical_pair)
            
    # --- Step 3: FIRST PASS - Apply filtering based on current_sku_id's Target Status ---
    # This pass modifies `current_matches_dict` according to the core rules.
    print("Applying conditional filtering (Pass 1 of 2)...")

    # Create a temporary copy of the dict to iterate over while modifying the main one,
    # or iterate through keys.
    # It's safer to iterate through keys and modify the dict.
    for sku_id in list(current_matches_dict.keys()): # Iterate over a copy of keys
        original_matches_for_sku = current_matches_dict[sku_id]
        new_matches_for_sku = set() 

        if sku_id in all_targets_in_feedback:
            # Rule 1: If sku_id IS a Target. Its matches are strictly whitelisted.
            for match_id in original_matches_for_sku:
                # Check if the canonical pair (sku_id, match_id) is explicitly allowed
                if tuple(sorted((sku_id, match_id))) in definitive_symmetric_allowed_pairs:
                    new_matches_for_sku.add(match_id)
        else:
            # Rule 2: If sku_id is NOT a Target.
            # Its matches are unaffected UNLESS they match a Target.
            for match_id in original_matches_for_sku:
                if match_id in all_targets_in_feedback:
                    # This match_id IS a Target. So, this specific pair is affected.
                    # Only keep it if the pair (sku_id, match_id) is explicitly allowed.
                    if tuple(sorted((sku_id, match_id))) in definitive_symmetric_allowed_pairs:
                        new_matches_for_sku.add(match_id)
                else:
                    # This match_id is NOT a Target. Matches between two non-Targets are untouched. Keep it.
                    new_matches_for_sku.add(match_id)
        
        current_matches_dict[sku_id] = new_matches_for_sku # Update the working dictionary

    # --- Step 4: SECOND PASS - Reconcile to enforce global symmetry based on the final filtered state ---
    # The first pass applies the rules from each SKU's perspective. Now we need to ensure
    # that if a pair (A, B) is allowed by `definitive_symmetric_allowed_pairs` and one side
    # of the relationship was successfully filtered to include it, the other side also reflects it.
    # This final pass is crucial for absolute consistency as per rule 2's "replicated" aspect.

    print("Reconciling matches for global symmetry (Pass 2 of 2)...")
    
    # Create a new dictionary to build the truly final consistent state
    final_consistent_matches = {sku_id: set() for sku_id in im_final_updated[main_id_col].unique()}

    for sku_id, matches_set in current_matches_dict.items():
        for match_id in matches_set:
            # For every (sku_id, match_id) pair that survived the first pass filtering,
            # we must ensure it's reciprocated. This relies on the fact that
            # if (A,B) survived A's filter, it implies (A,B) is in allowed_symmetric_pairs,
            # so B must be allowed to have A.
            
            # Add match_id to sku_id's set
            final_consistent_matches[sku_id].add(match_id)
            
            # And add sku_id to match_id's set for perfect symmetry
            # Ensure match_id exists as a key in final_consistent_matches (i.e., it's in im_final_df)
            if match_id in final_consistent_matches:
                final_consistent_matches[match_id].add(sku_id)

    # Update current_matches_dict with the final consistent state
    current_matches_dict = final_consistent_matches


    # --- Step 5: Convert current_matches_dict back to DataFrame column ---
    # Use .map and .apply(list) for efficient assignment.
    im_final_updated[main_matches_col] = im_final_updated[main_id_col].map(
        {sku_id: sorted(list(matches_set)) for sku_id, matches_set in current_matches_dict.items()}
    ).fillna('').apply(list) # fillna('') then apply(list) handles SKUs with no matches elegantly

    print("Finished update with strict target-centric rules.")
    return im_final_updated