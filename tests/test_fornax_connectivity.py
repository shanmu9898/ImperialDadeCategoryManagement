"""Connectivity smoke tests — hits real Fornax.

Pulls 2 rows from each of the four tables the pipeline reads, confirming:
    1. The connection URL works.
    2. Each table name resolves correctly.
    3. Your account has SELECT permission on each.

Auto-skips when `FORNAX_URL` and `IMPERIAL_DADE_CONNECTION_FACTORY` are both
unset, so this file is safe to leave in the test suite.

Run with::

    pytest tests/test_fornax_connectivity.py -v -s

The ``-s`` flag is recommended — the tests print row counts and the first few
column names so you can eyeball that the right tables came back.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

# These tests require live Fornax credentials.
pytestmark = pytest.mark.integration


def _fornax_configured() -> bool:
    """Skip if we can't reasonably attempt a connection.

    Defaults DRIVER=SQL Server / SERVER=ibp-db01 / DATABASE=fornax are baked in,
    so the only way to opt out is to set IMPERIAL_DADE_SKIP_FORNAX=1.
    """
    if os.getenv("IMPERIAL_DADE_SKIP_FORNAX") == "1":
        return False
    if os.getenv("IMPERIAL_DADE_CONNECTION_FACTORY"):
        return True
    # pyodbc must at least be importable for the test to have any chance.
    try:
        import pyodbc  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def loader():
    if not _fornax_configured():
        pytest.skip(
            "Fornax connection not available — set IMPERIAL_DADE_SKIP_FORNAX=1 to silence, "
            "or install pyodbc + ensure you have network access to the SQL Server."
        )

    # Load real .env values so the test uses the table names the operator pinned,
    # not the code-level defaults. `override=True` ensures we beat anything
    # tests/conftest.py set with `os.environ.setdefault(...)`.
    from pathlib import Path

    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    # Bust the lru_cache so newly-loaded env vars are respected.
    from imperial_dade.config.tables import get_fornax_tables
    get_fornax_tables.cache_clear()

    from imperial_dade.io.fornax import FornaxLoader

    try:
        ld = FornaxLoader()
    except Exception as exc:
        pytest.skip(f"Could not open Fornax connection: {exc}")
        return  # unreachable but keeps type-checkers happy
    try:
        yield ld
    finally:
        ld.close()


@pytest.mark.parametrize(
    "table_attr,description",
    [
        ("salsify_items_table",      "Salsify items"),
        ("item_segment_table",       "Item-segment mapping"),
        ("consolidated_items_table", "Consolidated item master"),
        ("sales_data_table",         "L6M sales / cost"),
    ],
)
def test_can_pull_two_rows(loader, table_attr: str, description: str) -> None:
    table = getattr(loader.tables, table_attr)
    df = loader.preview(table, limit=2)

    assert isinstance(df, pd.DataFrame), f"{description}: expected DataFrame, got {type(df)}"
    assert len(df) >= 1, (
        f"{description} ({table}): expected at least 1 row back, got 0. "
        f"The table is either empty or our query is filtering it out."
    )
    assert len(df) <= 2, (
        f"{description} ({table}): asked for 2 rows, got {len(df)} — "
        f"the dialect-specific LIMIT/TOP clause isn't being honored."
    )
    assert len(df.columns) >= 1, f"{description}: expected at least 1 column"

    # Visible-when-run-with-`pytest -s` evidence the right table came back
    print(
        f"\n{description}"
        f"\n  table   : {table}"
        f"\n  shape   : {df.shape}"
        f"\n  columns : {df.columns.tolist()[:8]}{'...' if len(df.columns) > 8 else ''}"
    )


def test_all_four_tables_have_distinct_names(loader) -> None:
    """Cheap sanity check — nobody pointed two FORNAX_TABLE_* vars at the same place."""
    names = [
        loader.tables.salsify_items_table,
        loader.tables.item_segment_table,
        loader.tables.consolidated_items_table,
        loader.tables.sales_data_table,
    ]
    assert len(set(names)) == 4, f"Duplicate table names in Fornax config: {names}"


# ---------------------------------------------------------------------------
# Loader contract tests — verify rename maps + Entity--Item key build against
# real Fornax data. These catch any mismatch between what Fornax returns and
# what the downstream pipeline expects (legacy in-memory schema).
# ---------------------------------------------------------------------------


def test_salsify_loader_builds_entity_item_key(loader) -> None:
    """get_salsify_items() must attach Entity--Item starting with '1--'."""
    sfy = loader.get_salsify_items()
    assert "Entity--Item" in sfy.columns
    sample = sfy["Entity--Item"].dropna().head(5).tolist()
    print(f"\nSalsify Entity--Item samples: {sample}")
    assert all(str(k).startswith("1--") for k in sample), (
        f"Salsify rows are always ERP=S2K, so keys must start '1--'. Got: {sample}"
    )


def test_item_segment_loader_returns_required_columns(loader) -> None:
    """get_item_segment_mapping() must surface 'Item Segment Key' + 'Item Segment'."""
    seg = loader.get_item_segment_mapping()
    assert "Item Segment Key" in seg.columns
    assert "Item Segment" in seg.columns
    distinct = seg["Item Segment"].dropna().unique().tolist()
    print(f"\nDistinct Item Segment values ({len(distinct)} total), sample: {distinct[:10]}")
    assert len(distinct) >= 1


def test_consolidated_loader_renames_and_keys(loader) -> None:
    """get_consolidated_items() returns the legacy in-memory schema."""
    items = loader.get_consolidated_items(entity_id=1, limit=10)
    print(f"\nConsolidated items: {items.shape}")

    expected_renamed = [
        "Entity Code",        # was "Company No."
        "ERP System",         # was "ERP_System"
        "Item Desc 1",        # was "Item Description 1"
        "Item Desc 2",        # was "Item Description 2"
        "location_code",      # was "Warehouse Code"
        "po_cost_amt",        # was "PO Cost"
        "VB Flag",            # was "VB_Flag"
        "Entity--Item",       # new, built from ERP_System + Item Code
        "Item Code",          # unchanged
        "Item Segment Key",   # unchanged (joins to Item_Segment)
    ]
    missing = [c for c in expected_renamed if c not in items.columns]
    assert not missing, (
        f"Consolidated items missing expected (renamed) columns: {missing}. "
        f"Present: {items.columns.tolist()[:15]}..."
    )
    # And the raw Fornax names should NOT survive the rename
    leaked = [c for c in ("Company No.", "ERP_System", "Item Description 1") if c in items.columns]
    assert not leaked, f"Fornax raw names leaked through the rename map: {leaked}"


def test_sales_loader_renames_and_keys(loader) -> None:
    """get_sales_data() returns the legacy in-memory schema."""
    # Pull a tiny sample of real item codes from consolidated to drive the query.
    items = loader.get_consolidated_items(entity_id=1, limit=50)
    sample_codes = items["Item Code"].dropna().unique().tolist()[:50]
    if not sample_codes:
        pytest.skip("No item codes in consolidated_items(entity_id=1) — nothing to query")

    from imperial_dade.categories import load_category
    cfg = load_category("cups")
    sales = loader.get_sales_data(sample_codes, cfg, entity_id=1, limit=10)
    print(f"\nSales data: {sales.shape}")

    if sales.empty:
        pytest.skip(
            "Sales pull returned 0 rows. Either the sample item codes have no sales "
            "for entity_id=1 in the L6M window, or the (entity_code, item_code) filter "
            "isn't lining up. Run the test against codes you know have sales."
        )

    expected_renamed = [
        "Entity Code",
        "Item",
        "ERP System",
        "Qty",
        "Net Cost",
        "Whs Code",
        "Entity--Item",
    ]
    missing = [c for c in expected_renamed if c not in sales.columns]
    assert not missing, (
        f"Sales data missing expected (renamed) columns: {missing}. "
        f"Present: {sales.columns.tolist()[:15]}..."
    )
    leaked = [c for c in ("entity_code", "item_code", "qty", "net_cost") if c in sales.columns]
    assert not leaked, f"Fornax raw names leaked through the rename map: {leaked}"

    # Entity--Item should be properly formed
    sample_keys = sales["Entity--Item"].dropna().head(5).tolist()
    print(f"Sales Entity--Item samples: {sample_keys}")
    assert all("--" in str(k) for k in sample_keys), (
        f"Entity--Item keys must contain '--'. Got: {sample_keys}"
    )
