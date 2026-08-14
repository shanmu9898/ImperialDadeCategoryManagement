"""Tests for the Fornax loader, key construction, and table config."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from imperial_dade.utils.keys import (
    KEY_SEPARATOR,
    S2K_ERP_NAME,
    S2K_ERP_NUMBER,
    build_entity_item_key,
    build_s2k_key,
    map_erp_system_to_number,
)


# ---------------------------------------------------------------------------
# map_erp_system_to_number
# ---------------------------------------------------------------------------


def test_s2k_is_always_mapped_to_1() -> None:
    mapping, _ = map_erp_system_to_number(pd.Series(["SAP", "ORACLE", "S2K"]))
    assert mapping[S2K_ERP_NAME] == S2K_ERP_NUMBER == 1


def test_non_s2k_systems_get_2_onwards() -> None:
    mapping, _ = map_erp_system_to_number(pd.Series(["SAP", "ORACLE", "S2K"]))
    other_values = sorted(v for k, v in mapping.items() if k != S2K_ERP_NAME)
    assert other_values == [2, 3]


def test_mapping_is_case_insensitive_for_s2k() -> None:
    """Every spelling of S2K resolves to 1, however it is cased.

    The mapping intentionally registers each alias spelling it actually saw, not
    just the canonical "S2K": `build_entity_item_key` looks up the raw ERP name
    with `mapping_dict.get(erp_val)`, and the sales table spells it
    "Imperial S2K". So assert the invariant — no spelling gets a number other
    than 1 — rather than an exact dict shape.
    """
    spellings = ["s2k", "S2K", "S2k"]
    mapping, mapped = map_erp_system_to_number(pd.Series(spellings))

    assert mapping[S2K_ERP_NAME] == 1
    assert set(mapping.values()) == {1}, f"a non-S2K number leaked in: {mapping}"
    for spelling in spellings:
        assert mapping[spelling] == 1, f"{spelling!r} did not resolve to 1"
    # all three rows should map to 1 because they're all S2K-ish
    assert mapped.tolist() == [1, 1, 1]


# ---------------------------------------------------------------------------
# build_*_key
# ---------------------------------------------------------------------------


def test_build_s2k_key_uppercase_and_prefix() -> None:
    result = build_s2k_key(pd.Series(["abc", "Xyz", "123"]))
    assert result.tolist() == ["1--ABC", "1--XYZ", "1--123"]


def test_build_entity_item_key_with_multi_erp() -> None:
    df = pd.DataFrame({
        "erp_system_name": ["S2K", "SAP", "ORACLE"],
        "item_code": ["abc", "def", "ghi"],
    })
    mapping, _ = map_erp_system_to_number(df["erp_system_name"])
    keys = build_entity_item_key(df, "erp_system_name", "item_code", mapping)
    assert keys.tolist() == [
        f"1{KEY_SEPARATOR}ABC",
        f"2{KEY_SEPARATOR}DEF",
        f"3{KEY_SEPARATOR}GHI",
    ]


def test_build_entity_item_key_unknown_erp_returns_none() -> None:
    df = pd.DataFrame({"erp": ["UNKNOWN"], "item": ["abc"]})
    mapping = {S2K_ERP_NAME: 1}
    keys = build_entity_item_key(df, "erp", "item", mapping)
    assert keys.tolist() == [None]


# ---------------------------------------------------------------------------
# FornaxTables — env override
# ---------------------------------------------------------------------------


def test_fornax_tables_pick_up_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting FORNAX_TABLE_* must change what get_fornax_tables() returns."""
    from imperial_dade.config import tables as tables_mod

    monkeypatch.setenv("FORNAX_TABLE_SALSIFY", "qa_schema.salsify_items_v2")
    monkeypatch.setenv("FORNAX_TABLE_SALES", "qa_schema.sales_l6m_v2")
    # cache-bust the lru_cache
    tables_mod.get_fornax_tables.cache_clear()
    try:
        t = tables_mod.get_fornax_tables()
        assert t.salsify_items_table == "qa_schema.salsify_items_v2"
        assert t.sales_data_table == "qa_schema.sales_l6m_v2"
    finally:
        tables_mod.get_fornax_tables.cache_clear()


def test_sales_table_composes_from_three_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """When FORNAX_TABLE_SALES is unset, the 3-part vars compose database.schema.table."""
    from imperial_dade.config import tables as tables_mod

    monkeypatch.delenv("FORNAX_TABLE_SALES", raising=False)
    monkeypatch.setenv("FORNAX_SALES_DATA_DATABASE", "my_db")
    monkeypatch.setenv("FORNAX_SALES_DATA_SCHEMA", "my_schema")
    monkeypatch.setenv("FORNAX_SALES_DATA_TABLE", "my_table")
    tables_mod.get_fornax_tables.cache_clear()
    try:
        t = tables_mod.get_fornax_tables()
        assert t.sales_data_table == "my_db.my_schema.my_table"
    finally:
        tables_mod.get_fornax_tables.cache_clear()


def test_explicit_sales_table_overrides_three_part(monkeypatch: pytest.MonkeyPatch) -> None:
    """If both are set, FORNAX_TABLE_SALES wins."""
    from imperial_dade.config import tables as tables_mod

    monkeypatch.setenv("FORNAX_TABLE_SALES", "explicit.schema.table")
    monkeypatch.setenv("FORNAX_SALES_DATA_DATABASE", "should_be_ignored")
    monkeypatch.setenv("FORNAX_SALES_DATA_SCHEMA", "should_be_ignored")
    monkeypatch.setenv("FORNAX_SALES_DATA_TABLE", "should_be_ignored")
    tables_mod.get_fornax_tables.cache_clear()
    try:
        t = tables_mod.get_fornax_tables()
        assert t.sales_data_table == "explicit.schema.table"
    finally:
        tables_mod.get_fornax_tables.cache_clear()


# ---------------------------------------------------------------------------
# CategoryConfig — taxonomy block
# ---------------------------------------------------------------------------


def test_cups_yaml_pins_a_taxonomy_column_list() -> None:
    """Cups was pinned to the legacy curated list."""
    from imperial_dade.categories import load_category

    cfg = load_category("cups")
    assert cfg.taxonomy.columns_for_description is not None
    assert "Beverage Cup Type" in cfg.taxonomy.columns_for_description
    assert cfg.taxonomy.coverage_threshold == 15.0
    assert cfg.taxonomy.default_pack_size == 1000


def test_cutlery_yaml_leaves_taxonomy_columns_null() -> None:
    """Cutlery defers to runtime coverage discovery — list intentionally None."""
    from imperial_dade.categories import load_category

    cfg = load_category("cutlery")
    assert cfg.taxonomy.columns_for_description is None


# ---------------------------------------------------------------------------
# FornaxLoader — connection wiring (no actual DB)
# ---------------------------------------------------------------------------


def test_default_pyodbc_conn_str_uses_windows_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """No UID/PWD env vars → conn string omits them (Windows auth)."""
    for var in ("FORNAX_DRIVER", "FORNAX_SERVER", "FORNAX_DATABASE",
                "FORNAX_UID", "FORNAX_PWD"):
        monkeypatch.delenv(var, raising=False)
    from imperial_dade.io.fornax import _build_pyodbc_conn_str

    cs = _build_pyodbc_conn_str()
    assert cs == "DRIVER=SQL Server;SERVER=ibp-db01;DATABASE=fornax;"


def test_pyodbc_conn_str_overrides_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORNAX_DRIVER", "ODBC Driver 18 for SQL Server")
    monkeypatch.setenv("FORNAX_SERVER", "qa-db.example.com")
    monkeypatch.setenv("FORNAX_DATABASE", "fornax_qa")
    monkeypatch.delenv("FORNAX_UID", raising=False)
    monkeypatch.delenv("FORNAX_PWD", raising=False)
    from imperial_dade.io.fornax import _build_pyodbc_conn_str

    cs = _build_pyodbc_conn_str()
    assert "DRIVER=ODBC Driver 18 for SQL Server" in cs
    assert "SERVER=qa-db.example.com" in cs
    assert "DATABASE=fornax_qa" in cs
    assert "UID=" not in cs and "PWD=" not in cs


def test_pyodbc_conn_str_appends_uid_pwd_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORNAX_UID", "service_account")
    monkeypatch.setenv("FORNAX_PWD", "s3cret")
    from imperial_dade.io.fornax import _build_pyodbc_conn_str

    cs = _build_pyodbc_conn_str()
    assert "UID=service_account" in cs
    assert "PWD=s3cret" in cs


def test_connection_factory_override_is_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    """IMPERIAL_DADE_CONNECTION_FACTORY=module:callable must be honoured."""
    monkeypatch.setenv("IMPERIAL_DADE_CONNECTION_FACTORY", "tests.test_keys_and_fornax:_fake_connection")
    from imperial_dade.io.fornax import get_fornax_connection

    conn = get_fornax_connection()
    assert conn == {"fake": "connection"}


# Module-level helper referenced by the factory-override test above.
def _fake_connection():
    return {"fake": "connection"}
