"""Fornax table + column name configuration.

All table names and the column names read from those tables are sourced here
so the rest of the code never embeds a bare string literal. Tables come from
env vars (so QA / prod can point at different schemas without code changes).
Column names are constants because they should be stable per dataset.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


# ---------------------------------------------------------------------------
# Table name configuration (from env)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FornaxTables:
    salsify_items_table: str
    item_segment_table: str
    consolidated_items_table: str
    sales_data_table: str


# Sales lives at a different addressing depth (database.schema.table) than
# the other three tables (schema.table). Compose it from three env vars.
_SALES_DATA_DATABASE_DEFAULT = "fornax_datamart"
_SALES_DATA_SCHEMA_DEFAULT = "dm_sales"
_SALES_DATA_TABLE_DEFAULT = "cda_bcg_sales_database_all_years"


def _resolve_sales_table() -> str:
    """Build the fully-qualified sales table reference.

    Precedence:
        1. ``FORNAX_TABLE_SALES`` — explicit override; used as-is.
        2. ``FORNAX_SALES_DATA_{DATABASE,SCHEMA,TABLE}`` — joined with dots.
    """
    explicit = os.getenv("FORNAX_TABLE_SALES")
    if explicit:
        return explicit
    return ".".join([
        os.getenv("FORNAX_SALES_DATA_DATABASE", _SALES_DATA_DATABASE_DEFAULT),
        os.getenv("FORNAX_SALES_DATA_SCHEMA", _SALES_DATA_SCHEMA_DEFAULT),
        os.getenv("FORNAX_SALES_DATA_TABLE", _SALES_DATA_TABLE_DEFAULT),
    ])


@lru_cache(maxsize=1)
def get_fornax_tables() -> FornaxTables:
    """Resolve table names from environment variables.

    Defaults are placeholders — every deployment must set these explicitly
    in `.env` to point at the right Fornax schema.
    """
    return FornaxTables(
        # Salsify, item-segment and consolidated-items all live in the same
        # database/schema (`fornax.dbo`). Sales is in a separate db/schema —
        # composed by `_resolve_sales_table()` from its own 3-part env vars.
        salsify_items_table=os.getenv("FORNAX_TABLE_SALSIFY", "fornax.dbo.products"),
        item_segment_table=os.getenv("FORNAX_TABLE_ITEM_SEGMENT", "fornax.dbo.Item_Segment"),
        consolidated_items_table=os.getenv(
            "FORNAX_TABLE_CONSOLIDATED_ITEMS", "fornax.dbo.ConsolidatedItemsByLocation"
        ),
        sales_data_table=_resolve_sales_table(),
    )


# ---------------------------------------------------------------------------
# Column-name constants — REAL names used in the Fornax tables.
#
# These names appear in the SQL itself (WHERE clauses, joins) so they MUST
# match the Fornax schema exactly. Bulk column renaming for the rest of the
# pipeline is done by the loader after the SELECT — see _RENAME_MAP_* dicts
# in `io/fornax.py`.
# ---------------------------------------------------------------------------

# fornax.salsify.products
# (Salsify exports use ProductID for the customer's product key. If your
#  Salsify deployment loads a different column as the S2K item code, override
#  with FORNAX_SALSIFY_S2K_COL.)
SALSIFY_S2K_ITEM_NUMBER = os.getenv("FORNAX_SALSIFY_S2K_COL", "ProductID")

# fornax.dbo.Item_Segment
ITEM_SEGMENT_KEY = "Item Segment Key"   # 'entity--division-class', joins back to ConsolidatedItemsByLocation
ITEM_SEGMENT_NAME = "Item Segment"      # human-readable category name ("Cups", "Cutlery", ...)

# fornax.dbo.ConsolidatedItemsByLocation
CONSOLIDATED_ENTITY_ID = "Company No."      # int — used in WHERE filter, requires bracket-quoting in SQL
CONSOLIDATED_ERP_SYSTEM_NAME = "ERP_System" # used to build Entity--Item
CONSOLIDATED_ITEM_CODE = "Item Code"        # used to build Entity--Item
CONSOLIDATED_ITEM_SEGMENT_KEY = "Item Segment Key"  # joins to Item_Segment

# fornax_datamart.dm_sales.cda_bcg_sales_database_all_years
SALES_ENTITY_ID = "entity_code"
SALES_ITEM_CODE = "item_code"
SALES_ERP_SYSTEM = "ERP"


# ---------------------------------------------------------------------------
# Fabric Lakehouse table configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FabricTables:
    """Tables read from the Microsoft Fabric lakehouse SQL endpoint."""

    vioitem_table: str          # S2K item master mirror — Salsify <-> S2K bridge
    item_segment_table: str     # Reltio item segment mapping (replaces empty fornax.dbo.Item_Segment)
    item_cluster_table: str     # Sancus item clustering — sibling descriptions per product


@lru_cache(maxsize=1)
def get_fabric_tables() -> FabricTables:
    return FabricTables(
        vioitem_table=os.getenv(
            "FABRIC_TABLE_VIOITEM", "src_s2k_r50modsdta.VIOITEM"
        ),
        item_segment_table=os.getenv(
            "FABRIC_TABLE_ITEM_SEGMENT", "src_reltio.item_segment"
        ),
        # Sancus lives in its OWN lakehouse (lh_idedw_sancus), not the
        # lh_idedw_business database the connection targets. It must therefore
        # be referenced with three-part naming. Connecting directly to
        # `Database=lh_idedw_sancus` hangs — the configured host is a warehouse
        # endpoint, so cross-database naming is the supported path.
        item_cluster_table=os.getenv(
            "FABRIC_TABLE_ITEM_CLUSTER",
            "[lh_idedw_sancus].[src_sancus].[item_cluster]",
        ),
    )


# Column names on src_s2k_r50modsdta.VIOITEM. These are the raw DB2/iSeries
# names mirrored as-is into the lakehouse; the loader renames them to a
# normalized schema after the SELECT.
VIOITEM_COMPANY_NUMBER = "IOITEM_COMPANY_NUMBER"  # entity id (filtered server-side)
VIOITEM_ITEM_NUMBER = "IOITEM_ITEM_NUMBER"        # S2K item code (joins to ConsolidatedItemsByLocation.[Item Code])
VIOITEM_UNIQUE_ID_COLUMN = "IOITEM_UNIQUE_ID_COLUMN"  # int — what Salsify uses as ProductID


# Column names on src_reltio.item_segment.
ITEM_SEGMENT_CATEGORY_TYPE = "ItemCategory_type"        # human-readable category (e.g. "Cups") -- matches cfg.name
ITEM_SEGMENT_BRANCH_COMPANY = "branch_company_code"     # entity id as string ('1' for US)
ITEM_SEGMENT_GROUP_CODE = "item_segment_group_code"     # pipe-delimited segment key (e.g. "1|1|NA|509|3")
ITEM_SEGMENT_CODE = "item_segment_code"                 # human-readable hierarchical code (e.g. "FSP-CUPS-CUPS-PAPER")


# ---------------------------------------------------------------------------
# Column names on src_sancus.item_cluster.
#
# Sancus groups near-identical items across branch companies into clusters, so
# every description Imperial Dade holds for one physical product can be read
# together. Three key facts about this table:
#
#   * `entity_item_code` is exactly '<entity>_<item_code>' (verified on 100.00%
#     of rows) and uses the SAME entity numbering as our `Entity--Item` prefix.
#     Entity 1 is Imperial US S2K — 494,784 of the 640,197 S2K rows, matching
#     the 'imperial s2k' + 'Imperial S2K' instance counts exactly. So our
#     '1--12HDQW' maps to '1_12hdqw' and that is the correct join key.
#   * It is stored LOWERCASED while `item_code` preserves case, and this
#     endpoint's collation is CASE-SENSITIVE — an uppercase lookup silently
#     returns zero rows. Always compare with LOWER() on both sides.
#   * Never join on bare `item_code`: the same code exists under other branch
#     companies and can be a different product there. Cups item 'CC' is a
#     coffee-cup sleeve under entity 1 but reaches freight-charge and
#     repair-labor rows under other entities.
# ---------------------------------------------------------------------------

ITEM_CLUSTER_ENTITY = "entity"                       # branch company id (85 values)
ITEM_CLUSTER_ENTITY_NAME = "entity_name"             # e.g. "Mailender S2K"
ITEM_CLUSTER_ITEM_CODE = "item_code"                 # S2K item code, case-preserving
ITEM_CLUSTER_ENTITY_ITEM_CODE = "entity_item_code"   # '<entity>_<item_code>', lowercased — unique per row
ITEM_CLUSTER_ID = "final_cluster_id"                 # the cluster key (1,991,739 distinct)
ITEM_CLUSTER_LENGTH = "cluster_length"               # sancus's own member count
ITEM_CLUSTER_DESCRIPTION = "orig_desc"               # raw description — preferred over cleaned_desc
ITEM_CLUSTER_CLEANED_DESCRIPTION = "cleaned_desc"    # lossy: strips punctuation, so 1.5" -> 1 5
ITEM_CLUSTER_ERP_INSTANCE = "erp_system_instance"    # e.g. "imperial s2k", "Western Paper P21"
ITEM_CLUSTER_VB_FLAG = "vb_flag"                     # 'N' / 'Y - VB' / 'Y - Other'
ITEM_CLUSTER_VGN = "vgn"                             # vendor group name

# Attribute columns Sancus has already extracted. Passed to the Stage-1 prompt
# as hints so the LLM can confirm or override them. `diameter` and `thickness`
# have no counterpart in cups.yaml's columns_for_description.
ITEM_CLUSTER_ATTRIBUTE_COLUMNS = (
    "color",
    "shape",
    "material",
    "size",
    "volume",
    "units",
    "thickness",
    "diameter",
    "dimensions",
    "case_pack",
)

# Only these `erp_system_instance` values represent items keyed the same way as
# our `1--` (S2K) Entity--Item values. Matched case-insensitively as a LIKE.
ITEM_CLUSTER_S2K_INSTANCE_PATTERN = "%s2k%"
