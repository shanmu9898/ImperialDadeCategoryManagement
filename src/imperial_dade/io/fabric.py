"""Microsoft Fabric lakehouse loader.

Sibling to ``io/fornax.py``. Reads from the ``lh_idedw_business`` lakehouse
**SQL analytics endpoint** (TDS over TCP 1433) using the ``mssql-python``
package.

Connection model — worth being precise, because the layers explain several
otherwise-baffling failures:

  * ``mssql-python`` is NOT a pure-Python driver. It vendors its own native
    stack inside the site-packages directory: ``msodbcsql18.dll`` (ODBC Driver
    18 for SQL Server), ``mssql-auth.dll`` (Entra ID), and a
    ``ddbc_bindings`` C-extension. So errors surface as
    ``[Microsoft][ODBC Driver 18 for SQL Server]...`` even though no
    system-level Driver 18 is installed — this box only registers the legacy
    "SQL Server" ODBC driver, which ``io/fornax.py`` uses via pyodbc.
  * Because the bundled layer is DDBC rather than plain ODBC, connection-string
    keywords differ from what ADO.NET/pyodbc accept: ``Connection Timeout`` is
    rejected as an unknown keyword, and the handshake timeout has to be passed
    as ``connect(..., timeout=N)`` instead.
  * We read through the SQL endpoint, NOT the ``abfss://`` OneLake path. Tables
    in other lakehouses of the same workspace are reachable by three-part name
    (see ``FABRIC_TABLE_ITEM_CLUSTER``); pointing ``Database=`` at another
    lakehouse instead just hangs, since this host is a warehouse endpoint.

Currently exposes the Salsify <-> S2K item-master bridge built from
``src_s2k_r50modsdta.VIOITEM``:

    Salsify ``ProductID`` == VIOITEM ``IOITEM_UNIQUE_ID_COLUMN``
    S2K item code        == VIOITEM ``IOITEM_ITEM_NUMBER``
    Entity id            == VIOITEM ``IOITEM_COMPANY_NUMBER``

...the Reltio item-segment mapping, and the Sancus item clusters that let Stage 1
read every description Imperial Dade holds for one physical product.

Two hard-won constraints govern reads against this endpoint:

  * **Three-part naming, not a direct database.** Sancus lives in its own
    lakehouse (``lh_idedw_sancus``). Connecting with
    ``Database=lh_idedw_sancus`` hangs, because the configured host is a
    warehouse endpoint; qualifying the table as
    ``[lh_idedw_sancus].[src_sancus].[item_cluster]`` while connected to
    ``lh_idedw_business`` answers in under a second.
  * **No bound parameters in predicates.** ``WHERE col IN (?, ?, ...)`` never
    returns against these tables, while the same query with inlined literals is
    immediate. See ``_sql_literal``.

Auth defaults to ``ActiveDirectoryInteractive`` (browser sign-in on first
call; token cached after). Override with ``FABRIC_AUTH`` for non-interactive
deployments (e.g. ``ActiveDirectoryServicePrincipal`` + ``FABRIC_UID`` /
``FABRIC_PWD``).
"""
from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Any, Optional, Sequence

import pandas as pd

from imperial_dade.config.tables import (
    FabricTables,
    ITEM_CLUSTER_ATTRIBUTE_COLUMNS,
    ITEM_CLUSTER_DESCRIPTION,
    ITEM_CLUSTER_ENTITY,
    ITEM_CLUSTER_ENTITY_ITEM_CODE,
    ITEM_CLUSTER_ENTITY_NAME,
    ITEM_CLUSTER_ERP_INSTANCE,
    ITEM_CLUSTER_ID,
    ITEM_CLUSTER_ITEM_CODE,
    ITEM_CLUSTER_S2K_INSTANCE_PATTERN,
    ITEM_CLUSTER_VB_FLAG,
    ITEM_CLUSTER_VGN,
    ITEM_SEGMENT_BRANCH_COMPANY,
    ITEM_SEGMENT_CATEGORY_TYPE,
    ITEM_SEGMENT_CODE,
    ITEM_SEGMENT_GROUP_CODE,
    VIOITEM_COMPANY_NUMBER,
    VIOITEM_ITEM_NUMBER,
    VIOITEM_UNIQUE_ID_COLUMN,
    get_fabric_tables,
)

logger = logging.getLogger(__name__)


# Normalized output column names — what downstream code joins on. Picked to
# stay neutral of the iSeries IOITEM_* prefix while still being explicit
# about what each one means.
NORMALIZED_ENTITY_ID = "entity_id"
NORMALIZED_S2K_ITEM_CODE = "s2k_item_code"
NORMALIZED_SALSIFY_PRODUCT_ID = "salsify_product_id"

# Normalized output columns for the Sancus cluster reads.
NORMALIZED_CLUSTER_ID = "cluster_id"
NORMALIZED_CLUSTER_DESCRIPTION = "cluster_description"


def to_sancus_key(entity_item: str) -> Optional[str]:
    """Translate our ``Entity--Item`` key to the Sancus ``entity_item_code``.

    ``'1--12HDQW'`` -> ``'1_12hdqw'``. Sancus stores the key as
    ``<entity>_<item_code>`` using the same entity numbering we do, lowercased.
    Returns None for anything that isn't a well-formed key.

    Lowercasing is not cosmetic: the lakehouse endpoint's collation is
    case-sensitive, so an uppercase key matches zero rows without erroring.
    """
    cleaned = _to_clean_str(entity_item)
    if not cleaned or "--" not in cleaned:
        return None
    entity, _, item_code = cleaned.partition("--")
    entity, item_code = entity.strip(), item_code.strip()
    if not entity or not item_code:
        return None
    return f"{entity}_{item_code}".lower()


def _sql_literal(value) -> str:
    """Render a single-quoted SQL string literal, escaping embedded quotes.

    Cluster reads inline their IN-list values instead of binding ``?`` params.
    That is deliberate: bound parameters in a predicate against this lakehouse
    make the query hang indefinitely (a 5-value IN never returned in 10+
    minutes), while the identical query with inlined literals answers in under a
    second. Every value routed through here is an item code or cluster id read
    back out of the lakehouse itself, and quotes are escaped regardless.
    """
    return "'" + str(value).replace("'", "''") + "'"


def _to_clean_str(value):
    """Strip whitespace, decoding latin-1 fallback for legacy iSeries bytes.

    Some VIOITEM rows store non-UTF-8 codepoints (notably 0xA0 — Latin-1
    non-breaking space). pandas's strict ``.astype(str)`` path raises a
    UnicodeDecodeError on those. Doing the conversion ourselves keeps the
    rest of the column usable.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, float) and pd.isna(value):
        return None
    return str(value).strip()


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def _build_fabric_conn_str() -> str:
    """Build the Fabric SQL endpoint connection string from env.

    Required env:
        FABRIC_SQL_ENDPOINT     hostname (without `,1433`)
        FABRIC_LAKEHOUSE        database name (e.g. ``lh_idedw_business``)

    Optional env:
        FABRIC_AUTH             default ``ActiveDirectoryInteractive``
        FABRIC_UID / FABRIC_PWD service-principal creds (only used when the
                                auth mode actually needs them)
        FABRIC_CONNECT_TIMEOUT  seconds to wait for the connection handshake,
                                default 60. When the workspace's capacity is
                                exhausted the endpoint stops accepting
                                connections and the driver otherwise hangs
                                indefinitely — a health check then costs ten
                                minutes instead of failing fast. This bounds the
                                handshake only; query duration is unaffected, so
                                long reads still work.
    """
    server = os.getenv("FABRIC_SQL_ENDPOINT")
    database = os.getenv("FABRIC_LAKEHOUSE")
    if not server or not database:
        raise RuntimeError(
            "FabricLoader requires FABRIC_SQL_ENDPOINT and FABRIC_LAKEHOUSE to "
            "be set in the environment (see .env.example)."
        )

    auth = os.getenv("FABRIC_AUTH", "ActiveDirectoryInteractive")
    uid = os.getenv("FABRIC_UID")
    pwd = os.getenv("FABRIC_PWD")

    parts = [
        f"Server={server},1433",
        f"Database={database}",
        f"Authentication={auth}",
        "Encrypt=yes",
        "TrustServerCertificate=no",
    ]
    if uid:
        parts.append(f"UID={uid}")
    if pwd:
        parts.append(f"PWD={pwd}")
    return ";".join(parts) + ";"


def get_fabric_connection() -> Any:
    """Open a Fabric SQL connection via mssql-python.

    Override with ``IMPERIAL_DADE_FABRIC_CONNECTION_FACTORY=module:callable``
    if your team already has a centralized helper.
    """
    factory_spec = os.getenv("IMPERIAL_DADE_FABRIC_CONNECTION_FACTORY")
    if factory_spec:
        module_name, _, attr = factory_spec.partition(":")
        if not module_name or not attr:
            raise ValueError(
                f"IMPERIAL_DADE_FABRIC_CONNECTION_FACTORY={factory_spec!r} "
                "must be 'module:callable'"
            )
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, attr)()

    try:
        import mssql_python
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "mssql-python is not installed. Run `pip install mssql-python`."
        ) from exc

    conn_str = _build_fabric_conn_str()
    redacted = conn_str.replace(os.getenv("FABRIC_PWD", ""), "<redacted>") if os.getenv("FABRIC_PWD") else conn_str
    # Bound the handshake. When the workspace's capacity is exhausted the
    # endpoint stops accepting connections and the driver hangs indefinitely,
    # turning a health check into a ten-minute wait. `timeout` is a connect()
    # argument here — 'Connection Timeout' is NOT a recognized keyword for this
    # driver's connection string and raises on parse. 0 means wait forever.
    timeout = int(os.getenv("FABRIC_CONNECT_TIMEOUT", "0") or 0)
    logger.info("Opening Fabric connection (%s) timeout=%ss", redacted, timeout or "none")
    return mssql_python.connect(conn_str, timeout=timeout)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class FabricLoader:
    """Reads from the Fabric lakehouse SQL endpoint.

    Use as a context manager so the underlying connection closes cleanly::

        with FabricLoader() as loader:
            mapping = loader.get_salsify_to_s2k_mapping(entity_id=1)
    """

    def __init__(
        self,
        connection: Optional[Any] = None,
        tables: Optional[FabricTables] = None,
    ) -> None:
        self._connection_owned = connection is None
        self.connection = connection or get_fabric_connection()
        self.tables = tables or get_fabric_tables()

    # -- Public API ----------------------------------------------------------

    def get_salsify_to_s2k_mapping(
        self,
        entity_id: int = 1,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return the Salsify -> S2K item-code bridge from VIOITEM.

        Args:
            entity_id: server-side filter on ``IOITEM_COMPANY_NUMBER``. Imperial
                Dade's primary US entity is ``1``.
            limit: optional ``TOP N`` cap, mostly for tests.

        Returns columns:
            ``salsify_product_id`` (Int64, nullable) — int Salsify uses as ProductID
            ``s2k_item_code``      (str, stripped)   — joins to ConsolidatedItemsByLocation.[Item Code]
            ``entity_id``          (int)             — from IOITEM_COMPANY_NUMBER

        Rows where ``IOITEM_UNIQUE_ID_COLUMN`` is NULL are dropped — those S2K
        items have no Salsify counterpart and can't bridge.
        """
        top_clause = f"TOP {int(limit)} " if limit else ""
        sql = (
            f"SELECT {top_clause}"
            f"  [{VIOITEM_COMPANY_NUMBER}] AS company_number,"
            f"  [{VIOITEM_ITEM_NUMBER}]    AS item_number,"
            f"  [{VIOITEM_UNIQUE_ID_COLUMN}] AS unique_id "
            f"FROM {self.tables.vioitem_table} "
            f"WHERE [{VIOITEM_COMPANY_NUMBER}] = ? "
            f"  AND [{VIOITEM_UNIQUE_ID_COLUMN}] IS NOT NULL"
        )
        logger.info(
            "Loading Salsify->S2K mapping from %s (entity_id=%d, limit=%s)",
            self.tables.vioitem_table, entity_id, limit,
        )
        df = self._read_sql(sql, params=(entity_id,))

        if df.empty:
            logger.warning(
                "VIOITEM returned 0 rows for entity_id=%d — check filter and table contents",
                entity_id,
            )
            return pd.DataFrame(
                columns=[
                    NORMALIZED_SALSIFY_PRODUCT_ID,
                    NORMALIZED_S2K_ITEM_CODE,
                    NORMALIZED_ENTITY_ID,
                ]
            )

        # iSeries CHAR columns come back with trailing spaces — strip them on
        # the S2K code so downstream joins work. The Salsify id is numeric.
        #
        # Some VIOITEM rows have non-UTF-8 bytes (e.g. 0xA0 — Latin-1
        # non-breaking space) in IOITEM_ITEM_NUMBER. Pandas's strict
        # .astype(str) blows up on those, so route through a tolerant
        # decoder.
        df["item_number"] = df["item_number"].apply(_to_clean_str)
        df["unique_id"] = pd.to_numeric(df["unique_id"], errors="coerce").astype("Int64")

        out = df.rename(
            columns={
                "unique_id": NORMALIZED_SALSIFY_PRODUCT_ID,
                "item_number": NORMALIZED_S2K_ITEM_CODE,
                "company_number": NORMALIZED_ENTITY_ID,
            }
        )[[NORMALIZED_SALSIFY_PRODUCT_ID, NORMALIZED_S2K_ITEM_CODE, NORMALIZED_ENTITY_ID]]

        logger.info("Loaded %d Salsify->S2K mapping rows", len(out))
        return out

    def get_item_segment_mapping(
        self,
        branch_company_code: str = "1",
    ) -> pd.DataFrame:
        """Return the legacy ``Item Segment`` <-> ``Item Segment Key`` mapping.

        Pulls from ``src_reltio.item_segment`` (lakehouse). ``fornax.dbo.Item_Segment``
        is empty in production, so the pipeline now sources the same shape of
        data from Reltio instead.

        Args:
            branch_company_code: server-side filter on the entity id. ``'1'``
                is the US entity. Passed as a string because the source column
                is varchar (it also stores category names for non-US rows).

        Returns:
            DataFrame with the legacy two-column schema:
                * ``Item Segment``      — category name (e.g. "Cups")
                * ``Item Segment Key``  — reconstructed ``branch--division-class``
                  key that joins to ``ConsolidatedItemsByLocation.[Item Segment Key]``.
            Distinct, non-null, ready to feed into the existing notebook code.

        Notes:
            The source's ``item_segment_group_code`` is pipe-delimited:
            ``<branch>|<branch_company>|NA|<division>|<class>``. The legacy
            key uses positions 1, 3, 4. Rows where the class is empty/null
            (``NA``) are still emitted — the downstream ``.isin()`` filter
            will simply not match them.
        """
        sql = (
            f"SELECT DISTINCT "
            f"  [{ITEM_SEGMENT_CATEGORY_TYPE}]  AS category_type, "
            f"  [{ITEM_SEGMENT_BRANCH_COMPANY}] AS branch_company_code, "
            f"  [{ITEM_SEGMENT_GROUP_CODE}]     AS group_code, "
            f"  [{ITEM_SEGMENT_CODE}]           AS segment_code "
            f"FROM {self.tables.item_segment_table} "
            f"WHERE [{ITEM_SEGMENT_BRANCH_COMPANY}] = ? "
            f"  AND [{ITEM_SEGMENT_CATEGORY_TYPE}] IS NOT NULL "
            f"  AND [{ITEM_SEGMENT_GROUP_CODE}] IS NOT NULL"
        )
        logger.info(
            "Loading item-segment mapping from %s (branch_company_code=%s)",
            self.tables.item_segment_table, branch_company_code,
        )
        df = self._read_sql(sql, params=(branch_company_code,))

        if df.empty:
            logger.warning(
                "item_segment returned 0 rows for branch_company_code=%s — "
                "check the filter and table contents",
                branch_company_code,
            )
            return pd.DataFrame(columns=["Item Segment", "Item Segment Key"])

        # Strip CHAR padding, then derive the legacy key from group_code parts.
        for col in ("category_type", "branch_company_code", "group_code"):
            df[col] = df[col].apply(_to_clean_str)

        parts = df["group_code"].str.split("|")

        def _build_key(row_parts, branch_company):
            if not isinstance(row_parts, list) or len(row_parts) < 5:
                return None
            division, klass = row_parts[3], row_parts[4]
            if not division or not klass:
                return None
            return f"{branch_company}--{division}-{klass}"

        df["Item Segment Key"] = [
            _build_key(p, bc) for p, bc in zip(parts, df["branch_company_code"])
        ]
        df["Item Segment"] = df["category_type"]

        out = df[["Item Segment", "Item Segment Key"]].dropna(
            subset=["Item Segment Key"]
        ).drop_duplicates()

        logger.info(
            "Loaded %d item-segment rows (%d distinct keys, %d distinct segments)",
            len(out),
            out["Item Segment Key"].nunique(),
            out["Item Segment"].nunique(),
        )
        return out.reset_index(drop=True)

    def get_item_cluster_table(
        self,
        attribute_columns: Sequence[str] = ITEM_CLUSTER_ATTRIBUTE_COLUMNS,
        s2k_only: bool = False,
        chunksize: Optional[int] = None,
        keep_entity_item_codes: Optional[Sequence[str]] = None,
        keep_cluster_ids: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Bulk-read the cluster table in ONE query, for local joining.

        This is the preferred entry point. The endpoint tolerates a single wide
        scan far better than many small predicated queries: a 640k-row narrow
        projection streams back in ~5 minutes, whereas repeated ``IN`` lookups
        degrade sharply and can exhaust the workspace's capacity until it
        throttles new connections outright. Callers should cache the result
        (see the ``cached()`` helpers in the run scripts) — the join itself is
        then pure pandas and needs no connection at all.

        Args:
            attribute_columns: Sancus's extracted attribute columns to include.
            s2k_only: restrict to S2K ERP instances (~640k of 3.5M rows). Enough
                to resolve which cluster our items belong to, but NOT enough to
                read every sibling description, since clusters span branch
                companies on other ERPs. Leave False to get all members.
            chunksize: batch size for a streamed read. DEFAULT None, and leave it
                there: ``pd.read_sql(..., chunksize=N)`` hangs against this
                lakehouse via mssql-python — a bounded ``TOP 5000`` query that
                answers instantly unchunked did not return in minutes when
                chunked. The unchunked read of all 3.5M rows completes in ~25
                minutes and is what the run scripts rely on.
            keep_entity_item_codes: reduce the result to rows whose
                ``entity_item_code`` is in this set ("which cluster is my item
                in?"). Applied after the read.
            keep_cluster_ids: reduce the result to rows in these clusters
                ("give me the members"). Applied after the read.

        Returns the same normalized schema as ``get_cluster_members``, plus
        ``erp_system_instance``.
        """
        attr_cols = list(attribute_columns)
        keep_keys = (
            {k for k in (_to_clean_str(v) for v in keep_entity_item_codes) if k}
            if keep_entity_item_codes is not None
            else None
        )
        keep_ids = (
            {c for c in (_to_clean_str(v) for v in keep_cluster_ids) if c}
            if keep_cluster_ids is not None
            else None
        )
        if keep_keys is not None:
            keep_keys = {k.lower() for k in keep_keys}
        attr_select = "".join(f", [{c}]" for c in attr_cols)
        where = (
            f"WHERE LOWER([{ITEM_CLUSTER_ERP_INSTANCE}]) LIKE "
            f"{_sql_literal(ITEM_CLUSTER_S2K_INSTANCE_PATTERN)} "
            if s2k_only
            else "WHERE 1 = 1 "
        )
        sql = (
            f"SELECT [{ITEM_CLUSTER_ID}]               AS cluster_id, "
            f"       [{ITEM_CLUSTER_ENTITY}]           AS entity, "
            f"       [{ITEM_CLUSTER_ITEM_CODE}]        AS item_code, "
            f"       [{ITEM_CLUSTER_ENTITY_ITEM_CODE}] AS entity_item_code, "
            f"       [{ITEM_CLUSTER_DESCRIPTION}]      AS description, "
            f"       [{ITEM_CLUSTER_ENTITY_NAME}]      AS entity_name, "
            f"       [{ITEM_CLUSTER_ERP_INSTANCE}]     AS erp_system_instance, "
            f"       [{ITEM_CLUSTER_VB_FLAG}]          AS vb_flag, "
            f"       [{ITEM_CLUSTER_VGN}]              AS vgn"
            f"{attr_select} "
            f"FROM {self.tables.item_cluster_table} "
            f"{where}"
            f"  AND [{ITEM_CLUSTER_ID}] IS NOT NULL"
        )
        logger.info(
            "Bulk-loading %s (s2k_only=%s, chunksize=%s, filter=%s) — expect "
            "minutes, cache the result",
            self.tables.item_cluster_table, s2k_only, chunksize,
            "keys" if keep_keys is not None else
            ("clusters" if keep_ids is not None else "none"),
        )

        if not chunksize:
            t0 = time.time()
            df = self._read_sql(sql)
            logger.info("Read %s rows in %.1f min", f"{len(df):,}", (time.time() - t0) / 60)
            if df.empty:
                logger.warning("Cluster table returned 0 rows")
                return df
            if keep_keys is not None:
                df = df[df["entity_item_code"].map(_to_clean_str).str.lower().isin(keep_keys)]
            if keep_ids is not None:
                df = df[df["cluster_id"].map(_to_clean_str).isin(keep_ids)]
            if df.empty:
                logger.warning("No cluster rows survived the requested filter")
                return df
            return self._normalize_cluster_frame(df, attr_cols)

        kept: list[pd.DataFrame] = []
        scanned = 0
        t0 = time.time()
        for chunk in self._read_sql_chunked(sql, chunksize):
            scanned += len(chunk)
            if keep_keys is not None:
                chunk = chunk[
                    chunk["entity_item_code"].map(_to_clean_str)
                    .str.lower().isin(keep_keys)
                ]
            if keep_ids is not None:
                chunk = chunk[chunk["cluster_id"].map(_to_clean_str).isin(keep_ids)]
            if len(chunk):
                kept.append(chunk)
            logger.info(
                "  scanned %s rows, kept %s (%.1f min)",
                f"{scanned:,}", f"{sum(len(k) for k in kept):,}",
                (time.time() - t0) / 60,
            )

        if not kept:
            logger.warning("Cluster table returned 0 usable rows out of %s scanned",
                           f"{scanned:,}")
            return pd.DataFrame()

        df = pd.concat(kept, ignore_index=True)
        logger.info("Retained %s of %s scanned rows", f"{len(df):,}", f"{scanned:,}")
        return self._normalize_cluster_frame(df, attr_cols)

    def _normalize_cluster_frame(
        self, df: pd.DataFrame, attr_cols: Sequence[str]
    ) -> pd.DataFrame:
        """Shared cleanup for cluster reads.

        Sancus stores non-UTF-8 bytes (0xA0, Latin-1 non-breaking space) in text
        columns, so pandas's strict ``.astype(str)`` raises UnicodeDecodeError on
        them. Route every string column through the tolerant decoder instead.
        """
        text_cols = [
            "cluster_id", "item_code", "entity_item_code", "description",
            "entity_name", "erp_system_instance", "vb_flag", "vgn",
        ] + list(attr_cols)
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(_to_clean_str)

        out = df.rename(
            columns={
                "cluster_id": NORMALIZED_CLUSTER_ID,
                "entity": NORMALIZED_ENTITY_ID,
                "item_code": NORMALIZED_S2K_ITEM_CODE,
                "description": NORMALIZED_CLUSTER_DESCRIPTION,
            }
        ).drop_duplicates()

        logger.info(
            "Normalized %d cluster rows (%d clusters, %d distinct descriptions)",
            len(out), out[NORMALIZED_CLUSTER_ID].nunique(),
            out[NORMALIZED_CLUSTER_DESCRIPTION].nunique(),
        )
        return out.reset_index(drop=True)

    def get_item_cluster_ids(
        self,
        entity_item_keys: Sequence[str],
        chunk_size: int = 100,
    ) -> pd.DataFrame:
        """Resolve our ``Entity--Item`` keys to their Sancus cluster ids.

        Joins on ``entity_item_code``, which is exactly ``<entity>_<item_code>``
        and shares our entity numbering — ``'1--12HDQW'`` becomes ``'1_12hdqw'``.
        This is an exact 1:1 key, so no ERP filtering or disambiguation is
        needed. Do NOT match on bare ``item_code``: the same code under another
        branch company is often a different product entirely.

        Args:
            entity_item_keys: our ``Entity--Item`` values, e.g. ``'1--12HDQW'``.
            chunk_size: keys per IN clause. Kept small deliberately — large IN
                lists against this endpoint degrade badly. For a whole category
                prefer ``get_item_cluster_table``.

        Returns columns:
            ``entity_item_code`` (lowercased sancus key), ``s2k_item_code``,
            ``cluster_id``, ``entity_id``.

        Comparison is done with ``LOWER()`` on both sides because the stored key
        is lowercased and this endpoint's collation is case-sensitive — an
        uppercase lookup silently matches nothing.
        """
        keys = [k for k in (to_sancus_key(k) for k in entity_item_keys) if k]
        keys = list(dict.fromkeys(keys))  # de-dupe, preserve order
        empty = pd.DataFrame(
            columns=[
                ITEM_CLUSTER_ENTITY_ITEM_CODE,
                NORMALIZED_S2K_ITEM_CODE,
                NORMALIZED_CLUSTER_ID,
                NORMALIZED_ENTITY_ID,
            ]
        )
        if not keys:
            logger.warning("get_item_cluster_ids called with no Entity--Item keys")
            return empty

        frames: list[pd.DataFrame] = []
        for offset in range(0, len(keys), chunk_size):
            chunk = keys[offset:offset + chunk_size]
            in_list = ", ".join(_sql_literal(k) for k in chunk)
            sql = (
                f"SELECT [{ITEM_CLUSTER_ENTITY_ITEM_CODE}] AS entity_item_code, "
                f"       [{ITEM_CLUSTER_ITEM_CODE}]        AS item_code, "
                f"       [{ITEM_CLUSTER_ID}]               AS cluster_id, "
                f"       [{ITEM_CLUSTER_ENTITY}]           AS entity "
                f"FROM {self.tables.item_cluster_table} "
                f"WHERE LOWER([{ITEM_CLUSTER_ENTITY_ITEM_CODE}]) IN ({in_list}) "
                f"  AND [{ITEM_CLUSTER_ID}] IS NOT NULL"
            )
            logger.info(
                "Resolving cluster ids for keys %d-%d/%d",
                offset, offset + len(chunk), len(keys),
            )
            frames.append(self._read_sql(sql))

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            logger.warning("No Sancus cluster rows matched %d Entity--Item keys", len(keys))
            return empty

        for col in ("entity_item_code", "item_code", "cluster_id"):
            df[col] = df[col].apply(_to_clean_str)

        out = df.rename(
            columns={
                "item_code": NORMALIZED_S2K_ITEM_CODE,
                "cluster_id": NORMALIZED_CLUSTER_ID,
                "entity": NORMALIZED_ENTITY_ID,
            }
        ).drop_duplicates()

        logger.info(
            "Resolved %d/%d keys to %d distinct clusters",
            out[ITEM_CLUSTER_ENTITY_ITEM_CODE].nunique(), len(keys),
            out[NORMALIZED_CLUSTER_ID].nunique(),
        )
        return out.reset_index(drop=True)

    def get_cluster_members(
        self,
        cluster_ids: Sequence[str],
        attribute_columns: Sequence[str] = ITEM_CLUSTER_ATTRIBUTE_COLUMNS,
        chunk_size: int = 100,
    ) -> pd.DataFrame:
        """Return every member of the given clusters, across all branch companies.

        Deliberately unfiltered by ERP or entity: the point of reading the
        cluster is to collect every description Imperial Dade holds for one
        physical product, wherever it's stocked.

        Args:
            cluster_ids: values from ``get_item_cluster_ids``.
            attribute_columns: Sancus's own extracted attribute columns to carry
                along as prompt hints. Pass ``()`` to skip them.
            chunk_size: cluster ids per IN clause.

        Returns one row per cluster member:
            ``cluster_id``, ``entity_id``, ``s2k_item_code``,
            ``entity_item_code``, ``cluster_description``, ``vb_flag``, ``vgn``,
            plus one column per requested attribute.
        """
        ids = [c for c in (_to_clean_str(c) for c in cluster_ids) if c]
        ids = list(dict.fromkeys(ids))
        attr_cols = [c for c in attribute_columns]
        base_cols = [
            NORMALIZED_CLUSTER_ID,
            NORMALIZED_ENTITY_ID,
            NORMALIZED_S2K_ITEM_CODE,
            ITEM_CLUSTER_ENTITY_ITEM_CODE,
            NORMALIZED_CLUSTER_DESCRIPTION,
            ITEM_CLUSTER_VB_FLAG,
            ITEM_CLUSTER_VGN,
        ]
        if not ids:
            logger.warning("get_cluster_members called with no cluster ids")
            return pd.DataFrame(columns=base_cols + attr_cols)

        attr_select = "".join(f", [{c}]" for c in attr_cols)
        frames: list[pd.DataFrame] = []
        for offset in range(0, len(ids), chunk_size):
            chunk = ids[offset:offset + chunk_size]
            in_list = ", ".join(_sql_literal(c) for c in chunk)
            sql = (
                f"SELECT [{ITEM_CLUSTER_ID}]               AS cluster_id, "
                f"       [{ITEM_CLUSTER_ENTITY}]           AS entity, "
                f"       [{ITEM_CLUSTER_ITEM_CODE}]        AS item_code, "
                f"       [{ITEM_CLUSTER_ENTITY_ITEM_CODE}] AS entity_item_code, "
                f"       [{ITEM_CLUSTER_DESCRIPTION}]      AS description, "
                f"       [{ITEM_CLUSTER_ENTITY_NAME}]      AS entity_name, "
                f"       [{ITEM_CLUSTER_VB_FLAG}]          AS vb_flag, "
                f"       [{ITEM_CLUSTER_VGN}]              AS vgn"
                f"{attr_select} "
                f"FROM {self.tables.item_cluster_table} "
                f"WHERE [{ITEM_CLUSTER_ID}] IN ({in_list})"
            )
            logger.info(
                "Loading cluster members %d-%d/%d", offset, offset + len(chunk), len(ids)
            )
            frames.append(self._read_sql(sql))

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            logger.warning("No cluster members found for %d cluster ids", len(ids))
            return pd.DataFrame(columns=base_cols + attr_cols)

        return self._normalize_cluster_frame(df, attr_cols)

    # -- Diagnostics ---------------------------------------------------------

    def preview(self, table: str, limit: int = 5) -> pd.DataFrame:
        """Fetch a small sample from any fully-qualified table.

        Mirrors ``FornaxLoader.preview`` — handy for smoke-checking that a
        new env-configured table name resolves before wiring it into a loader.
        """
        limit = int(limit)
        if limit <= 0:
            raise ValueError("limit must be > 0")
        sql = f"SELECT TOP {limit} * FROM {table}"
        logger.info("preview: %s (limit=%d)", table, limit)
        return self._read_sql(sql)

    # -- Internals -----------------------------------------------------------

    def _read_sql_chunked(self, sql: str, chunksize: int):
        """Yield the result in row batches so memory stays bounded.

        Materializing the whole 3.5M-row cluster table costs several GB of
        Python string objects on top of the item-master and sales frames the run
        script already holds, and the OS OOM-kills the process without a
        traceback. Streaming keeps peak memory at one chunk.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pandas only supports SQLAlchemy connectable.*",
                category=UserWarning,
            )
            yield from pd.read_sql(sql, self.connection, chunksize=chunksize)

    def _read_sql(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """pd.read_sql wrapper that silences the noisy non-SQLAlchemy warning.

        mssql-python is a DB-API 2.0 driver but not a SQLAlchemy connectable,
        so pandas emits a UserWarning every call. Cosmetic — suppress it here
        so logs stay readable. Drop this once we switch to SQLAlchemy.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pandas only supports SQLAlchemy connectable.*",
                category=UserWarning,
            )
            return pd.read_sql(sql, self.connection, params=params)

    # -- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        if not self._connection_owned:
            return
        if hasattr(self.connection, "close"):
            try:
                self.connection.close()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self) -> "FabricLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
