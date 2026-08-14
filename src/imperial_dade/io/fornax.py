"""Fornax data loader — single entry point for all source data the pipeline reads.

Replaces the four ``pd.read_excel`` / ``pd.read_csv`` calls in the legacy
``01_Taxonomy.ipynb``:

    legacy file                                      -> FornaxLoader method
    --------------------------------------------     -----------------------------
    All Salsify Items.xlsx (sheet='in', skiprows=1) -> get_salsify_items()
    Item Segment Mapping.csv                         -> get_item_segment_mapping()
    Item_Master_real_time.csv                        -> get_consolidated_items(entity_id=1)
    {Category} Sales Data for Cost Analysis L6M.xlsx -> get_sales_data(item_codes, entity_id, category)

Connection model: **pyodbc to SQL Server**. Mirrors the legacy snippet::

    conn_str = (
        r'DRIVER=SQL Server;'
        r'SERVER=ibp-db01;'
        r'DATABASE=fornax;'
    )
    conn = pyodbc.connect(conn_str)

Column renames: Fornax tables use a mix of column-name styles
(``Company No.``, ``ERP_System``, ``item_code``, ``Item Code``). To keep the
rest of the pipeline working against the legacy in-memory schema
(``Entity Code``, ``ERP System``, ``Item``, ``Item Desc 1``, ...) without a
sweeping refactor, each loader applies a per-table rename map after the
SELECT.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

import pandas as pd

from imperial_dade.categories import CategoryConfig
from imperial_dade.config.tables import (
    CONSOLIDATED_ENTITY_ID,
    CONSOLIDATED_ERP_SYSTEM_NAME,
    CONSOLIDATED_ITEM_CODE,
    CONSOLIDATED_ITEM_SEGMENT_KEY,
    FornaxTables,
    ITEM_SEGMENT_KEY,
    ITEM_SEGMENT_NAME,
    SALES_ENTITY_ID,
    SALES_ERP_SYSTEM,
    SALES_ITEM_CODE,
    SALSIFY_S2K_ITEM_NUMBER,
    get_fornax_tables,
)
from imperial_dade.utils.keys import (
    build_entity_item_key,
    build_s2k_key,
    map_erp_system_to_number,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column rename maps — Fornax column name -> legacy in-memory name
# ---------------------------------------------------------------------------

# Consolidated item master.
_CONSOLIDATED_RENAME: dict[str, str] = {
    "Company No.": "Entity Code",
    "ERP_System": "ERP System",
    "Item Description 1": "Item Desc 1",
    "Item Description 2": "Item Desc 2",
    "Warehouse Code": "location_code",
    "PO Cost": "po_cost_amt",
    "Preferred Vendor name": "Preferred Vendor Name",
    "VB_Flag": "VB Flag",
    "Item Sub Category Name": "Item Sub Category",
    "Item Category Name": "Item Category",
    # left as-is (no rename, here for documentation):
    #   'Item Code', 'VGN', 'VPN', 'Case Pack', 'Region', 'Territory',
    #   'Item Segment Key', 'Preferred Vendor Code'
}

# Salsify products. The Fornax mirror exports column names in PascalCase
# (``ProductCapacity``, ``BeverageCupStyle``) but the per-category YAML
# configs and downstream code reference them as display names with spaces.
# This rename map bridges the two so ``merge_with_salsify`` can find the
# columns the YAML asks for.
#
# Only rename columns that have a direct counterpart. If a YAML name has no
# Salsify column at all (e.g. cups' ``Pack Size``, ``Pattern & Design``),
# leave it absent — the taxonomy LLM will attempt to extract those from the
# item description instead.
_SALSIFY_RENAME: dict[str, str] = {
    "ProductCapacity":   "Product Capacity",
    "UsageTemperature":  "Usage Temperature",
    "BeverageCupStyle":  "Beverage Cup Style",
}


# Sales data.
_SALES_RENAME: dict[str, str] = {
    "entity_code": "Entity Code",
    "item_code": "Item",
    "ERP": "ERP System",
    "qty": "Qty",
    "net_cost": "Net Cost",
    "customer_code": "Customer Code",
    "price_group_name": "Price Group Name",
    "Warehouse No": "Whs Code",
    "branch_location": "Dashboard Location",
    "sales_rep_id": "Updated Salesperson Name",
    "fiscal_period": "Fiscal Period",
    # left as-is: 'Gross Cost', 'Customer Class', 'Ship to Code', 'State', 'City'
}


# ---------------------------------------------------------------------------
# Connection — pyodbc + SQL Server (with factory override escape hatch)
# ---------------------------------------------------------------------------


def _build_pyodbc_conn_str() -> str:
    """Compose the pyodbc connection string from env (or defaults).

    Defaults reproduce the legacy snippet::

        DRIVER=SQL Server;SERVER=ibp-db01;DATABASE=fornax;

    If both ``FORNAX_UID`` and ``FORNAX_PWD`` are set, they're appended; if
    neither is set, the connection uses Windows authentication (the SQL Server
    ODBC driver's default when no credentials are provided).
    """
    parts = [
        f"DRIVER={os.getenv('FORNAX_DRIVER', 'SQL Server')}",
        f"SERVER={os.getenv('FORNAX_SERVER', 'ibp-db01')}",
        f"DATABASE={os.getenv('FORNAX_DATABASE', 'fornax')}",
    ]
    uid = os.getenv("FORNAX_UID")
    pwd = os.getenv("FORNAX_PWD")
    if uid:
        parts.append(f"UID={uid}")
    if pwd:
        parts.append(f"PWD={pwd}")
    return ";".join(parts) + ";"


def get_fornax_connection() -> Any:
    """Open a pyodbc connection to Fornax.

    Override with ``IMPERIAL_DADE_CONNECTION_FACTORY=module:callable`` if your
    team has a centralized helper. The callable must return something
    ``pandas.read_sql`` can use as ``con``.
    """
    factory_spec = os.getenv("IMPERIAL_DADE_CONNECTION_FACTORY")
    if factory_spec:
        module_name, _, attr = factory_spec.partition(":")
        if not module_name or not attr:
            raise ValueError(
                f"IMPERIAL_DADE_CONNECTION_FACTORY={factory_spec!r} must be 'module:callable'"
            )
        import importlib

        module = importlib.import_module(module_name)
        factory = getattr(module, attr)
        return factory()

    try:
        import pyodbc
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pyodbc is not installed. Run `pip install pyodbc` or set "
            "IMPERIAL_DADE_CONNECTION_FACTORY to point at your team's helper."
        ) from exc

    conn_str = _build_pyodbc_conn_str()
    logger.info("Opening Fornax connection (%s)", conn_str.replace("PWD=", "PWD=<redacted>"))
    return pyodbc.connect(conn_str)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _bracket(name: str) -> str:
    """SQL Server identifier quoting — handles spaces, dots, special chars."""
    return f"[{name}]"


class FornaxLoader:
    """Loads source data from Fornax tables, attaches Entity--Item keys."""

    def __init__(
        self,
        connection: Optional[Any] = None,
        tables: Optional[FornaxTables] = None,
    ) -> None:
        self._connection_owned = connection is None
        self.connection = connection or get_fornax_connection()
        self.tables = tables or get_fornax_tables()
        self._erp_mapping: dict | None = None

    # -- Public API ----------------------------------------------------------

    def get_salsify_items(
        self,
        salsify_to_s2k: Optional[pd.DataFrame] = None,
        entity_id: int = 1,
    ) -> pd.DataFrame:
        """Salsify product attributes.

        ``ProductID`` in the Salsify export is an internal id (the iSeries
        ``VIOITEM.IOITEM_UNIQUE_ID_COLUMN``), not the S2K item code. To make
        Salsify rows joinable to ``ConsolidatedItemsByLocation`` we need a
        bridge through that VIOITEM table.

        Args:
            salsify_to_s2k: pre-loaded bridge frame (e.g. from FabricLoader).
                Must contain columns ``salsify_product_id`` and ``s2k_item_code``.
                If ``None``, this method opens a FabricLoader transparently.
            entity_id: only used when this method opens its own FabricLoader.

        Returns:
            The Salsify frame plus an ``Entity--Item`` key built from the
            mapped S2K item code, plus an ``S2K Item Number`` alias column.
            Rows whose ``ProductID`` doesn't appear in the bridge are still
            returned (Entity--Item / S2K Item Number will be NaN) — callers
            usually drop them before joining.
        """
        sql = f"SELECT * FROM {self.tables.salsify_items_table}"
        logger.info("Loading Salsify items from %s", self.tables.salsify_items_table)
        df = pd.read_sql(sql, self.connection)

        # Surface PascalCase attribute columns under the display names the
        # category YAMLs use ("ProductCapacity" -> "Product Capacity", etc.).
        applied = {k: v for k, v in _SALSIFY_RENAME.items() if k in df.columns}
        if applied:
            df = df.rename(columns=applied)
            logger.info("Applied %d Salsify display-name renames: %s",
                        len(applied), list(applied.values()))

        if SALSIFY_S2K_ITEM_NUMBER not in df.columns:
            raise ValueError(
                f"Salsify table {self.tables.salsify_items_table!r} is missing "
                f"id column {SALSIFY_S2K_ITEM_NUMBER!r}. "
                f"Override FORNAX_SALSIFY_S2K_COL to point at the right column. "
                f"Available: {df.columns.tolist()[:15]}..."
            )

        if salsify_to_s2k is None:
            # Local import to avoid a hard dependency on mssql-python when
            # the caller has already loaded the bridge some other way.
            from imperial_dade.io.fabric import FabricLoader

            with FabricLoader() as fabric:
                salsify_to_s2k = fabric.get_salsify_to_s2k_mapping(entity_id=entity_id)

        required = {"salsify_product_id", "s2k_item_code"}
        missing = required - set(salsify_to_s2k.columns)
        if missing:
            raise ValueError(
                f"salsify_to_s2k bridge is missing required columns {missing}. "
                f"Present: {salsify_to_s2k.columns.tolist()}"
            )

        df = df.copy()
        # Normalize the join key on both sides to Int64 so NaN-friendly,
        # type-stable merges work. Salsify's ProductID column may come in
        # as object dtype.
        df["_salsify_product_id_key"] = pd.to_numeric(
            df[SALSIFY_S2K_ITEM_NUMBER], errors="coerce"
        ).astype("Int64")

        bridge = salsify_to_s2k[["salsify_product_id", "s2k_item_code"]].drop_duplicates(
            subset=["salsify_product_id"], keep="first"
        )
        df = df.merge(
            bridge,
            left_on="_salsify_product_id_key",
            right_on="salsify_product_id",
            how="left",
        ).drop(columns=["_salsify_product_id_key", "salsify_product_id"])

        unmatched = df["s2k_item_code"].isna().sum()
        if unmatched:
            logger.warning(
                "get_salsify_items: %d/%d Salsify rows did not map to an S2K item "
                "code via VIOITEM — they will have a null Entity--Item",
                unmatched, len(df),
            )

        df["Entity--Item"] = build_s2k_key(df["s2k_item_code"])
        # Mask the rows whose s2k_item_code was NaN — build_s2k_key turns
        # those into the literal string "1--NAN", which would silently break
        # downstream joins.
        df.loc[df["s2k_item_code"].isna(), "Entity--Item"] = pd.NA

        # Alias for legacy code that references the human-friendly name.
        if "S2K Item Number" not in df.columns:
            df["S2K Item Number"] = df["s2k_item_code"]

        logger.info(
            "Loaded %d Salsify rows (%d mapped to S2K items)",
            len(df), len(df) - unmatched,
        )
        return df

    def get_item_segment_mapping(self) -> pd.DataFrame:
        """Map of (division-class) -> human-readable segment name.

        Returns columns including ``Item Segment Key`` (joins back to
        ConsolidatedItemsByLocation) and ``Item Segment`` (the category name
        you'd see in cfg.name, e.g. "Cups").
        """
        sql = f"SELECT * FROM {self.tables.item_segment_table}"
        logger.info("Loading item-segment mapping from %s", self.tables.item_segment_table)
        df = pd.read_sql(sql, self.connection)

        required = {ITEM_SEGMENT_KEY, ITEM_SEGMENT_NAME}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Item-segment table {self.tables.item_segment_table!r} is missing "
                f"required columns {missing}. Present: {df.columns.tolist()}"
            )

        logger.info(
            "Loaded %d item-segment rows (%d distinct segments)",
            len(df), df[ITEM_SEGMENT_NAME].nunique(),
        )
        return df

    def get_consolidated_items(
        self, entity_id: int = 1, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Consolidated item master, filtered server-side to one entity.

        Args:
            entity_id: server-side filter on ``Company No.``.
            limit: optional ``TOP N`` cap. Useful for tests / debugging —
                production runs leave this ``None`` to load the full table.

        Renames Fornax column names to the legacy in-memory schema and builds
        ``Entity--Item`` using the same ERP-system->int mapping the sales
        loader will reuse.
        """
        top_clause = f"TOP {int(limit)} " if limit else ""
        sql = (
            f"SELECT {top_clause}* FROM {self.tables.consolidated_items_table} "
            f"WHERE {_bracket(CONSOLIDATED_ENTITY_ID)} = ?"
        )
        logger.info(
            "Loading consolidated items from %s (entity_id=%d)",
            self.tables.consolidated_items_table, entity_id,
        )
        df = pd.read_sql(sql, self.connection, params=(entity_id,))

        required = {CONSOLIDATED_ERP_SYSTEM_NAME, CONSOLIDATED_ITEM_CODE}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Consolidated items table missing required columns {missing}. "
                f"Present: {df.columns.tolist()}"
            )

        # Build ERP-int mapping BEFORE renaming (the constant points at the
        # Fornax column name, not the renamed in-memory one).
        mapping, _ = map_erp_system_to_number(df[CONSOLIDATED_ERP_SYSTEM_NAME])
        df = df.copy()
        df["Entity--Item"] = build_entity_item_key(
            df, CONSOLIDATED_ERP_SYSTEM_NAME, CONSOLIDATED_ITEM_CODE, mapping
        )
        self._erp_mapping = mapping

        df = df.rename(columns=_CONSOLIDATED_RENAME)
        logger.info("Loaded %d item-master rows (entity_id=%d)", len(df), entity_id)
        return df

    def get_sales_data(
        self,
        item_codes: Iterable[str],
        category: CategoryConfig,
        entity_id: int = 1,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Sales / cost data filtered server-side to one entity + an item-code list.

        Filtering pushes (``entity_code``, ``item_code IN (...)``) into SQL.
        After load, Fornax columns are renamed to the legacy in-memory schema
        and ``Entity--Item`` is attached using the cached ERP mapping (or a
        fresh one if ``get_consolidated_items`` hasn't run yet).

        Args:
            item_codes: raw item codes (NOT pre-built Entity--Item keys) to
                filter on. SQL Server caps IN-clause params at ~2100, so we
                chunk above 1000.
            category: only used for logging.
            entity_id: which entity to pull. Sales is multi-entity; we filter
                here to match the consolidated_items load.
        """
        codes = list(item_codes)
        if not codes:
            logger.warning("get_sales_data called with empty item_codes for %s", category.name)
            return pd.DataFrame()

        chunk_size = 1000
        top_clause = f"TOP {int(limit)} " if limit else ""
        frames: list[pd.DataFrame] = []
        for offset in range(0, len(codes), chunk_size):
            chunk = codes[offset:offset + chunk_size]
            placeholders = ", ".join(["?"] * len(chunk))
            sql = (
                f"SELECT {top_clause}* FROM {self.tables.sales_data_table} "
                f"WHERE {_bracket(SALES_ENTITY_ID)} = ? "
                f"AND {_bracket(SALES_ITEM_CODE)} IN ({placeholders})"
            )
            logger.info(
                "Loading sales rows %d-%d/%d for %s (entity_id=%d)",
                offset, offset + len(chunk), len(codes), category.name, entity_id,
            )
            frames.append(
                pd.read_sql(sql, self.connection, params=(entity_id, *chunk))
            )
            # If a row cap was requested, stop chunking once we've hit it.
            if limit and sum(len(f) for f in frames) >= limit:
                break

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            logger.warning("get_sales_data returned 0 rows for %s", category.name)
            return df

        # Build Entity--Item BEFORE renaming (uses raw Fornax column names).
        mapping = self._erp_mapping or map_erp_system_to_number(df[SALES_ERP_SYSTEM])[0]
        df = df.copy()
        df["Entity--Item"] = build_entity_item_key(
            df, SALES_ERP_SYSTEM, SALES_ITEM_CODE, mapping
        )

        df = df.rename(columns=_SALES_RENAME)
        logger.info("Loaded %d sales rows for %s", len(df), category.name)
        return df

    # -- Diagnostics ---------------------------------------------------------

    def preview(self, table: str, limit: int = 5) -> pd.DataFrame:
        """Fetch a small sample from any fully-qualified table.

        Useful as a smoke test before running a full stage: confirms the
        connection works, the table name is right, and your account has
        SELECT permission. Uses SQL Server's ``TOP N`` syntax.
        """
        limit = int(limit)
        if limit <= 0:
            raise ValueError("limit must be > 0")
        sql = f"SELECT TOP {limit} * FROM {table}"
        logger.info("preview: %s (limit=%d)", table, limit)
        return pd.read_sql(sql, self.connection)

    # -- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        if not self._connection_owned:
            return
        if hasattr(self.connection, "close"):
            try:
                self.connection.close()
            except Exception:  # pragma: no cover - already closed
                pass
        elif hasattr(self.connection, "dispose"):
            self.connection.dispose()

    def __enter__(self) -> "FornaxLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
