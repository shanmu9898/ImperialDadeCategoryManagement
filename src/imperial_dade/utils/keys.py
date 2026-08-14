"""Entity--Item composite-key construction.

The legacy pipeline keys every transaction, item-master row, and Salsify row
on a string ``"<erp_int>--<UPPERCASE_ITEM_CODE>"``. ERP systems are mapped
to integers via `map_erp_system_to_number` so the same item code on different
ERPs gets distinct keys. S2K is always 1.

These helpers MUST be used everywhere the key is built so all dataframes line
up for joins downstream.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

S2K_ERP_NUMBER = 1
S2K_ERP_NAME = "S2K"
KEY_SEPARATOR = "--"

# S2K shows up under different display names across source tables: the
# item-master/Salsify side spells it "S2K", but the sales table spells it
# "Imperial S2K". Both are the same physical ERP and MUST map to the same
# number (1), or the Entity--Item keys won't line up across frames. Compared
# uppercased. Extend this set if another S2K spelling surfaces.
S2K_ERP_ALIASES = {"S2K", "IMPERIAL S2K"}


def _is_s2k(value) -> bool:
    """True if an ERP-system name is any known spelling of S2K."""
    return str(value).strip().upper() in S2K_ERP_ALIASES


def map_erp_system_to_number(series: pd.Series) -> Tuple[dict, pd.Series]:
    """Build a stable ERP-system-name -> int mapping.

    Any S2K spelling (see ``S2K_ERP_ALIASES``) is always 1. All other ERP
    systems are assigned 2..N in their order of first appearance in `series`.

    Returns:
        (mapping_dict, mapped_series) — the dict so callers can reuse the
        same mapping across multiple dataframes; the series so they can
        attach it to a single dataframe. Every S2K alias present in `series`
        is keyed into the dict at 1 so direct ``mapping[name]`` lookups work.
    """
    unique_systems = pd.Series(series.unique())
    # All S2K spellings are special-cased to 1; everything else gets 2+.
    others = unique_systems[~unique_systems.apply(_is_s2k)]
    mapping: dict = {S2K_ERP_NAME: S2K_ERP_NUMBER}
    # Register every S2K alias actually seen so callers indexing by the raw
    # name (e.g. "Imperial S2K") resolve to 1.
    for val in unique_systems[unique_systems.apply(_is_s2k)]:
        mapping[val] = S2K_ERP_NUMBER
    for idx, val in enumerate(others, start=S2K_ERP_NUMBER + 1):
        mapping[val] = idx

    def _lookup(x):
        if pd.isna(x):
            return None
        if _is_s2k(x):
            return S2K_ERP_NUMBER
        return mapping.get(x)

    return mapping, series.apply(_lookup)


def build_entity_item_key(
    df: pd.DataFrame, erp_col: str, item_col: str, mapping_dict: dict
) -> pd.Series:
    """Construct the Entity--Item key for every row.

    Format: ``f"{erp_int}--{item_code.upper()}"``.

    Rows with an ERP system not in `mapping_dict` produce a key starting with
    "None--"; the caller should drop/inspect these before joining.
    """
    def _key(row):
        erp_val = row[erp_col]
        item_val = row[item_col]
        if pd.isna(erp_val) or pd.isna(item_val):
            return None
        if _is_s2k(erp_val):
            erp_int = S2K_ERP_NUMBER
        else:
            erp_int = mapping_dict.get(erp_val)
        if erp_int is None:
            return None
        return f"{erp_int}{KEY_SEPARATOR}{str(item_val).upper()}"

    return df.apply(_key, axis=1)


def build_s2k_key(item_code_series: pd.Series) -> pd.Series:
    """Convenience for Salsify rows, where ERP is always S2K.

    Equivalent to: ``"1--" + item_code.upper()``.
    """
    return f"{S2K_ERP_NUMBER}{KEY_SEPARATOR}" + item_code_series.astype(str).str.upper()
