"""Connectivity smoke tests — hits the real Fabric lakehouse.

Confirms:
    1. The Fabric SQL endpoint connection works.
    2. ``src_s2k_r50modsdta.VIOITEM`` is reachable and has rows.
    3. ``FabricLoader.get_salsify_to_s2k_mapping`` returns the normalized
       schema with non-null Salsify-id and S2K-code values.

Auto-skips when ``FABRIC_SQL_ENDPOINT`` isn't set, so this file is safe to
leave in the suite. Auth defaults to ``ActiveDirectoryInteractive`` — the
first run pops a browser sign-in; cached after.

Run with::

    pytest tests/test_fabric_connectivity.py -v -s -m integration
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def _fabric_configured() -> bool:
    """Skip if we can't reasonably attempt a connection.

    Loads ``.env`` first so the operator's real FABRIC_* values are honored
    even though conftest.py only stubs Azure OpenAI keys.
    """
    from pathlib import Path

    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    if os.getenv("IMPERIAL_DADE_SKIP_FABRIC") == "1":
        return False
    if os.getenv("IMPERIAL_DADE_FABRIC_CONNECTION_FACTORY"):
        return True
    if not (os.getenv("FABRIC_SQL_ENDPOINT") and os.getenv("FABRIC_LAKEHOUSE")):
        return False
    try:
        import mssql_python  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def loader():
    if not _fabric_configured():
        pytest.skip(
            "Fabric connection not available — set FABRIC_SQL_ENDPOINT, "
            "FABRIC_LAKEHOUSE in .env, install mssql-python, or set "
            "IMPERIAL_DADE_SKIP_FABRIC=1 to silence."
        )

    # Reload .env so the test honors the operator-pinned endpoint, not the
    # placeholder values conftest.py sets for unit tests.
    from pathlib import Path

    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    # Bust the lru_cache so newly-loaded env vars are respected.
    from imperial_dade.config.tables import get_fabric_tables
    get_fabric_tables.cache_clear()

    from imperial_dade.io.fabric import FabricLoader

    try:
        ld = FabricLoader()
    except Exception as exc:                       # noqa: BLE001
        pytest.skip(f"Could not open Fabric connection: {exc}")
        return  # unreachable
    try:
        yield ld
    finally:
        ld.close()


def test_can_preview_vioitem(loader) -> None:
    """Raw connectivity check — pull 2 rows from VIOITEM."""
    table = loader.tables.vioitem_table
    df = loader.preview(table, limit=2)

    assert isinstance(df, pd.DataFrame), f"expected DataFrame, got {type(df)}"
    assert 1 <= len(df) <= 2, (
        f"asked for 2 rows from {table}, got {len(df)}. "
        f"Empty table or TOP-N clause not honored."
    )
    # VIOITEM has 56 cols in the R50 mirror; we don't pin the exact count
    # because new columns get added upstream, but the key bridge columns
    # must be present.
    for col in ("IOITEM_COMPANY_NUMBER", "IOITEM_ITEM_NUMBER", "IOITEM_UNIQUE_ID_COLUMN"):
        assert col in df.columns, f"VIOITEM missing required column {col!r}"

    print(
        f"\nVIOITEM"
        f"\n  table   : {table}"
        f"\n  shape   : {df.shape}"
        f"\n  bridge  : item_number={df['IOITEM_ITEM_NUMBER'].tolist()}, "
        f"unique_id={df['IOITEM_UNIQUE_ID_COLUMN'].tolist()}"
    )


def test_salsify_to_s2k_mapping_returns_normalized_schema(loader) -> None:
    """get_salsify_to_s2k_mapping must surface the three normalized cols."""
    df = loader.get_salsify_to_s2k_mapping(entity_id=1, limit=20)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "VIOITEM (entity=1) returned no rows — check the filter"

    expected_cols = {"salsify_product_id", "s2k_item_code", "entity_id"}
    assert set(df.columns) == expected_cols, (
        f"Unexpected columns. Got {sorted(df.columns)}, expected {sorted(expected_cols)}"
    )

    # The whole point of this method is that the bridge columns are
    # populated — rows where IOITEM_UNIQUE_ID_COLUMN is NULL must be
    # filtered out server-side.
    assert df["salsify_product_id"].notna().all(), (
        "Some salsify_product_id values are null — the NOT NULL filter is broken"
    )
    assert df["s2k_item_code"].astype(str).str.strip().str.len().gt(0).all(), (
        "Some s2k_item_code values are empty/whitespace-only — strip() may not be applied"
    )

    # entity filter honored
    assert (df["entity_id"] == 1).all(), (
        f"entity_id filter leaked rows for other entities: {df['entity_id'].unique()}"
    )

    # iSeries CHAR strip — no trailing whitespace should survive
    leftover_padded = df["s2k_item_code"].astype(str).map(lambda s: s != s.strip())
    assert not leftover_padded.any(), (
        f"Trailing whitespace on s2k_item_code: {df.loc[leftover_padded, 's2k_item_code'].head().tolist()}"
    )

    print(
        f"\nSalsify->S2K mapping sample (entity=1):"
        f"\n  rows  : {len(df)}"
        f"\n  dtype : {df.dtypes.to_dict()}"
        f"\n  head  :\n{df.head(5).to_string(index=False)}"
    )


def test_limit_param_caps_row_count(loader) -> None:
    """limit must push TOP N server-side, not slice client-side."""
    df = loader.get_salsify_to_s2k_mapping(entity_id=1, limit=3)
    assert len(df) <= 3


def test_item_segment_mapping_schema(loader) -> None:
    """get_item_segment_mapping must surface the two legacy columns."""
    df = loader.get_item_segment_mapping(branch_company_code="1")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "src_reltio.item_segment returned 0 rows for branch_company_code=1"

    assert list(df.columns) == ["Item Segment", "Item Segment Key"], (
        f"Unexpected columns. Got {list(df.columns)}, expected ['Item Segment', 'Item Segment Key']"
    )

    # Keys must follow the legacy branch--division-class shape.
    sample = df["Item Segment Key"].dropna().head(20).tolist()
    assert all(k.startswith("1--") for k in sample), (
        f"All keys for branch_company_code='1' must start '1--'. Got: {sample}"
    )

    # The category we depend on for Stage 1 must be present.
    segments = df["Item Segment"].unique().tolist()
    assert "Cups" in segments, f"Expected 'Cups' in segments. Got: {segments}"

    print(
        f"\nItem-segment mapping:"
        f"\n  total rows         : {len(df)}"
        f"\n  distinct segments  : {df['Item Segment'].nunique()}"
        f"\n  distinct keys      : {df['Item Segment Key'].nunique()}"
        f"\n  Cups keys          : {(df['Item Segment'] == 'Cups').sum()}"
    )


def test_item_segment_keys_join_to_fornax(loader) -> None:
    """The Cups keys from Fabric must actually match items in Fornax.

    Without this we'd have a schema-correct but useless mapping. Hits real
    Fornax — skipped when pyodbc / SQL Server isn't reachable.
    """
    import os

    try:
        import pyodbc
    except ImportError:
        pytest.skip("pyodbc not installed; can't cross-check against Fornax")

    if os.getenv("IMPERIAL_DADE_SKIP_FORNAX") == "1":
        pytest.skip("Fornax explicitly skipped")

    df = loader.get_item_segment_mapping(branch_company_code="1")
    cups_keys = df.loc[df["Item Segment"] == "Cups", "Item Segment Key"].unique().tolist()
    assert cups_keys, "no Cups keys to validate"

    try:
        conn = pyodbc.connect("DRIVER=SQL Server;SERVER=ibp-db01;DATABASE=fornax;")
    except Exception as exc:                       # noqa: BLE001
        pytest.skip(f"Fornax unreachable for cross-check: {exc}")

    try:
        ph = ", ".join(["?"] * len(cups_keys))
        sql = (
            "SELECT [Item Segment Key], COUNT(*) AS items "
            "FROM fornax.dbo.ConsolidatedItemsByLocation "
            f"WHERE [Company No.] = 1 AND [Item Segment Key] IN ({ph}) "
            "GROUP BY [Item Segment Key]"
        )
        verify = pd.read_sql(sql, conn, params=cups_keys)
    finally:
        conn.close()

    assert not verify.empty, (
        f"None of the {len(cups_keys)} Cups segment keys from Fabric matched "
        "any items in Fornax ConsolidatedItemsByLocation. The key-reconstruction "
        "formula in get_item_segment_mapping is broken."
    )
    total_items = int(verify["items"].sum())
    assert total_items >= 1000, (
        f"Only {total_items} Cups items joined via Fabric keys — expected at least "
        "a thousand. Either the source data shifted or the key formula is wrong."
    )
    print(
        f"\nFabric->Fornax join validation:"
        f"\n  Fabric Cups keys     : {len(cups_keys)}"
        f"\n  Keys w/ Fornax items : {len(verify)}"
        f"\n  Total items mapped   : {total_items}"
    )
