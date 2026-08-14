"""Attach Sancus cluster descriptions to the Stage-1 taxonomy frame.

Imperial Dade stocks the same physical product under many branch companies, each
with its own hand-typed description. Any single description is usually missing
something the others carry::

    12oz paper hot cup coat. white 12hdqw          <- coated, but no brand/pack
    victoria bay 12 oz paper hot cup white 1000/ca <- brand + case pack
    12oz vb white paper hot cups 1m/cs             <- pack in another notation
    12oz hot cup paper 1000cs                      <- no color

Sancus already groups these into clusters, so reading a whole cluster gives the
Stage-1 extractor strictly more evidence than the item's own description does.
This module resolves each item to its cluster, collects the sibling
descriptions, and renders the block the LLM sees.

The item's own description always comes first and is labelled as primary, so the
model has an anchor when siblings disagree.

Join contract: our ``Entity--Item`` maps to Sancus ``entity_item_code`` by
replacing ``--`` with ``_`` and lowercasing (``'1--12HDQW'`` -> ``'1_12hdqw'``).
Both sides use the same entity numbering, and the key is unique per row, so this
is an exact 1:1 match. Never fall back to matching on bare ``item_code``: the
same code under a different branch company is frequently a different product,
and doing so pulled freight charges into a coffee-cup sleeve's attributes.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import pandas as pd

from imperial_dade.config.tables import ITEM_CLUSTER_ENTITY_ITEM_CODE
from imperial_dade.io.fabric import (
    NORMALIZED_CLUSTER_DESCRIPTION,
    NORMALIZED_CLUSTER_ID,
    NORMALIZED_S2K_ITEM_CODE,
    to_sancus_key,
)

logger = logging.getLogger(__name__)

# Columns this module adds to the taxonomy frame.
CLUSTER_ID_COL = "Cluster ID"
CLUSTER_SIZE_COL = "Cluster Size"
CLUSTER_DESCRIPTIONS_COL = "Cluster Descriptions"
CLUSTER_HINTS_COL = "Cluster Attribute Hints"

# Descriptions that carry no product information and would only waste tokens
# (and risk misleading the model). Matched case-insensitively as substrings.
_NOISE_MARKERS = ("do not use", "donotuse", "obsolete", "discontinued", "dead item")

# Minimum description overlap required before a cluster is trusted.
#
# This guard is not optional. Short, generic item codes collide across branch
# companies, so a bare item_code match can land on an unrelated cluster: real
# example, cups item '1--CC' ("VB SLEEVE COFFEE CUP HOT KFT 92MM SERIES 12/100")
# matched two clusters whose other members were "upsg freight charge",
# "labor to repair tool" and "drop ship freight at no charge to customer".
# Feeding those to the extractor as "descriptions of the same product" would be
# strictly worse than using no cluster at all.
DEFAULT_SIMILARITY_THRESHOLD = 0.30

_TOKEN_MIN_LEN = 2


def _tokens(text: str) -> set[str]:
    """Normalized alphanumeric tokens for overlap scoring."""
    if not isinstance(text, str):
        return set()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    return {t for t in cleaned.split() if len(t) >= _TOKEN_MIN_LEN}


def _similarity(left: str, right: str) -> float:
    """Containment overlap of two descriptions, in [0, 1].

    Divides by the SMALLER token set rather than the union (plain Jaccard), so a
    terse description still scores well against a verbose one describing the same
    product — which is exactly the asymmetry across branch companies.
    """
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _is_noise(description: str) -> bool:
    lowered = description.lower()
    return any(marker in lowered for marker in _NOISE_MARKERS)


def _dedupe_descriptions(descriptions: Iterable[str]) -> list[str]:
    """Drop blanks, noise and case-insensitive duplicates, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in descriptions:
        if not isinstance(raw, str):
            continue
        text = " ".join(raw.split())  # collapse internal whitespace
        if not text or _is_noise(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def build_description_block(
    own_description: str,
    sibling_descriptions: Sequence[str],
    max_descriptions: int = 8,
) -> str:
    """Render the multi-description block for one item.

    The item's own description is always entry 1 and labelled PRIMARY. Siblings
    follow, longest first — a longer description carries more attributes, so when
    the cap bites we keep the most informative ones.

    Returns an empty string when there is nothing but the item's own
    description, so callers can fall back to the single-description prompt.
    """
    own = " ".join(str(own_description or "").split())
    ordered = _dedupe_descriptions([own, *sibling_descriptions])
    if not ordered:
        return ""

    primary, siblings = ordered[0], ordered[1:]
    if not siblings:
        return ""

    # Cap the siblings, keeping the longest (most attribute-dense) ones. Sort by
    # length for selection, then restore the original order for stable prompts.
    if max_descriptions and len(siblings) > max_descriptions - 1:
        keep = set(sorted(siblings, key=len, reverse=True)[: max_descriptions - 1])
        siblings = [s for s in siblings if s in keep]

    lines = [
        "PRIMARY DESCRIPTION (this item): " + primary,
        f"ADDITIONAL DESCRIPTIONS OF THE SAME PRODUCT ({len(siblings)}):",
    ]
    lines.extend(f"  - {s}" for s in siblings)
    return "\n".join(lines)


def build_hint_string(
    member_rows: pd.DataFrame,
    attribute_columns: Sequence[str],
) -> str:
    """Summarize Sancus's own extracted attributes for a cluster.

    Emits ``name=value`` pairs for attributes where the cluster agrees. Where
    members disagree, every distinct value is listed separated by ``/`` so the
    model can see the ambiguity rather than trusting one arbitrary row.
    """
    parts: list[str] = []
    for col in attribute_columns:
        if col not in member_rows.columns:
            continue
        values = _dedupe_descriptions(member_rows[col].tolist())
        if not values:
            continue
        parts.append(f"{col}={' / '.join(values[:4])}")
    return ", ".join(parts)


def attach_cluster_descriptions(
    cat_data: pd.DataFrame,
    cluster_ids: pd.DataFrame,
    cluster_members: pd.DataFrame,
    attribute_columns: Sequence[str] = (),
    max_descriptions: int = 8,
    entity_item_col: str = "Entity--Item",
    own_description_col: str = "Combined Descriptions",
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """Add cluster id, size, sibling-description block and attribute hints.

    Args:
        cat_data: the taxonomy frame, carrying ``entity_item_col`` and
            ``own_description_col``.
        cluster_ids: output of ``FabricLoader.get_item_cluster_ids``.
        cluster_members: output of ``FabricLoader.get_cluster_members``.
        attribute_columns: Sancus attribute columns to render as hints. Empty
            disables hints entirely.
        max_descriptions: total descriptions per prompt, including the item's own.
        similarity_threshold: minimum description overlap for a cluster to be
            trusted, and for an individual sibling to be kept. See
            ``DEFAULT_SIMILARITY_THRESHOLD`` for why this guard is required.

    Returns a copy of ``cat_data`` with four columns added. Items with no cluster
    match, an unconvincing cluster, or no surviving siblings get empty strings —
    the caller then falls back to the existing single-description behaviour.
    """
    out = cat_data.copy()
    for col in (CLUSTER_ID_COL, CLUSTER_DESCRIPTIONS_COL, CLUSTER_HINTS_COL):
        out[col] = ""
    out[CLUSTER_SIZE_COL] = 0

    if cluster_ids.empty or cluster_members.empty:
        logger.warning(
            "attach_cluster_descriptions: no cluster data supplied "
            "(ids=%d, members=%d) — Stage 1 will use item descriptions only",
            len(cluster_ids), len(cluster_members),
        )
        return out

    # Map our Entity--Item -> Sancus entity_item_code -> cluster id(s). This is
    # an exact 1:1 key including the entity, so an item can only pick up the
    # cluster of its own branch's row — never a same-coded item elsewhere.
    ids = cluster_ids.copy()
    ids["_key"] = ids[ITEM_CLUSTER_ENTITY_ITEM_CODE].astype(str).str.strip().str.lower()
    key_to_clusters: dict[str, list[str]] = (
        ids.groupby("_key")[NORMALIZED_CLUSTER_ID]
        .apply(lambda s: list(dict.fromkeys(v for v in s if v)))
        .to_dict()
    )

    members = cluster_members.copy()
    members[NORMALIZED_CLUSTER_ID] = members[NORMALIZED_CLUSTER_ID].astype(str)
    members_by_cluster = dict(tuple(members.groupby(NORMALIZED_CLUSTER_ID)))

    hint_cols = [c for c in attribute_columns if c in members.columns]

    matched = 0
    enriched = 0
    multi_cluster = 0
    rejected_cluster = 0
    dropped_siblings = 0
    total_extra = 0

    cluster_id_values: list[str] = []
    size_values: list[int] = []
    block_values: list[str] = []
    hint_values: list[str] = []

    for _, row in out.iterrows():
        raw_key = str(row.get(entity_item_col, ""))
        sancus_key = to_sancus_key(raw_key)
        clusters = key_to_clusters.get(sancus_key, []) if sancus_key else []
        own_desc = str(row.get(own_description_col, "") or "")

        if not clusters:
            cluster_id_values.append("")
            size_values.append(0)
            block_values.append("")
            hint_values.append("")
            continue

        matched += 1
        if len(clusters) > 1:
            multi_cluster += 1

        # Pick ONE cluster, never the union. When a code maps to several
        # clusters they are usually different products (the same code means
        # different things at different branches), so unioning them imports
        # unrelated descriptions. Score each candidate by how well its members'
        # descriptions overlap this item's own description and take the best.
        best_cluster: Optional[str] = None
        best_members: Optional[pd.DataFrame] = None
        best_score = 0.0
        for candidate in clusters:
            candidate_members = members_by_cluster.get(candidate)
            if candidate_members is None or candidate_members.empty:
                continue
            score = max(
                (
                    _similarity(own_desc, d)
                    for d in candidate_members[NORMALIZED_CLUSTER_DESCRIPTION].tolist()
                ),
                default=0.0,
            )
            if score > best_score:
                best_score, best_cluster, best_members = score, candidate, candidate_members

        if best_cluster is None or best_score < similarity_threshold:
            # The cluster does not describe this product — better no context
            # than misleading context.
            rejected_cluster += 1
            cluster_id_values.append("")
            size_values.append(0)
            block_values.append("")
            hint_values.append("")
            continue

        member_rows = best_members

        # Exclude only this item's OWN Sancus row — its description is already
        # the primary entry. Rows for the same item_code under other branch
        # companies are kept: they are the same product described differently,
        # which is precisely the signal we came for.
        siblings = member_rows
        if sancus_key and ITEM_CLUSTER_ENTITY_ITEM_CODE in member_rows.columns:
            is_self = (
                member_rows[ITEM_CLUSTER_ENTITY_ITEM_CODE]
                .astype(str).str.strip().str.lower()
                == sancus_key
            )
            siblings = member_rows[~is_self]

        # Even inside a well-matched cluster, individual members can be junk.
        # Hold every sibling to the same overlap bar.
        sibling_descs = []
        for desc in siblings[NORMALIZED_CLUSTER_DESCRIPTION].tolist():
            if _similarity(own_desc, desc) >= similarity_threshold:
                sibling_descs.append(desc)
            else:
                dropped_siblings += 1

        block = build_description_block(
            own_desc, sibling_descs, max_descriptions=max_descriptions
        )
        hints = build_hint_string(member_rows, hint_cols) if hint_cols else ""

        if block:
            enriched += 1
            total_extra += block.count("\n  - ")

        cluster_id_values.append(best_cluster)
        size_values.append(len(member_rows))
        block_values.append(block)
        hint_values.append(hints)

    out[CLUSTER_ID_COL] = cluster_id_values
    out[CLUSTER_SIZE_COL] = size_values
    out[CLUSTER_DESCRIPTIONS_COL] = block_values
    out[CLUSTER_HINTS_COL] = hint_values

    logger.info(
        "attach_cluster_descriptions: %d/%d items matched a cluster; "
        "%d rejected as off-topic (below %.2f overlap); %d gained sibling "
        "descriptions (%d extra descriptions, %d junk siblings dropped, "
        "%d items spanned multiple clusters)",
        matched, len(out), rejected_cluster, similarity_threshold,
        enriched, total_extra, dropped_siblings, multi_cluster,
    )
    return out


def attach_from_cluster_table(
    cat_data: pd.DataFrame,
    cluster_table: pd.DataFrame,
    attribute_columns: Sequence[str] = (),
    max_descriptions: int = 8,
    entity_item_col: str = "Entity--Item",
    own_description_col: str = "Combined Descriptions",
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """Attach cluster context from one bulk-read cluster frame.

    This is the preferred path: ``FabricLoader.get_item_cluster_table`` is read
    once and cached, then everything here is local pandas. It derives the
    item->cluster mapping and the member list from the same frame, so no further
    round-trips are needed.

    Our items are matched on the exact ``entity_item_code`` key (our
    ``Entity--Item`` with ``--`` -> ``_``, lowercased), so an item can only take
    the cluster of its own branch's row. Sibling descriptions are then read from
    the whole cluster regardless of entity or ERP — the cluster is the vetted
    grouping, and other branches' wording is the signal we want.
    """
    if cluster_table.empty:
        logger.warning("attach_from_cluster_table: empty cluster table")
        return attach_cluster_descriptions(
            cat_data, pd.DataFrame(), pd.DataFrame(),
            attribute_columns=attribute_columns,
            max_descriptions=max_descriptions,
            entity_item_col=entity_item_col,
            own_description_col=own_description_col,
            similarity_threshold=similarity_threshold,
        )

    table = cluster_table
    wanted = {
        key
        for key in (to_sancus_key(str(v)) for v in cat_data[entity_item_col].dropna())
        if key
    }

    keys_lc = (
        table[ITEM_CLUSTER_ENTITY_ITEM_CODE].astype(str).str.strip().str.lower()
    )
    is_ours = keys_lc.isin(wanted)

    cluster_ids = table.loc[
        is_ours, [ITEM_CLUSTER_ENTITY_ITEM_CODE, NORMALIZED_S2K_ITEM_CODE,
                  NORMALIZED_CLUSTER_ID]
    ].drop_duplicates()
    if cluster_ids.empty:
        logger.warning(
            "attach_from_cluster_table: none of %d Entity--Item keys matched a "
            "cluster row", len(wanted),
        )

    members = table[table[NORMALIZED_CLUSTER_ID].isin(set(cluster_ids[NORMALIZED_CLUSTER_ID]))]
    logger.info(
        "attach_from_cluster_table: %d keys -> %d clusters -> %d members",
        len(wanted), cluster_ids[NORMALIZED_CLUSTER_ID].nunique(), len(members),
    )

    return attach_cluster_descriptions(
        cat_data, cluster_ids, members,
        attribute_columns=attribute_columns,
        max_descriptions=max_descriptions,
        entity_item_col=entity_item_col,
        own_description_col=own_description_col,
        similarity_threshold=similarity_threshold,
    )


def load_cluster_context(
    cat_data: pd.DataFrame,
    loader,
    attribute_columns: Sequence[str] = (),
    max_descriptions: int = 8,
    entity_item_col: str = "Entity--Item",
    own_description_col: str = "Combined Descriptions",
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """Convenience wrapper: resolve clusters via ``loader`` and attach them.

    Issues one query per ~100 item codes, so it suits small or incremental runs.
    For a full category prefer ``FabricLoader.get_item_cluster_table`` plus
    ``attach_from_cluster_table``, which needs a single (cacheable) read.

    Kept separate from ``attach_cluster_descriptions`` so the joining logic stays
    unit-testable without a Fabric connection.
    """
    keys = [str(v) for v in cat_data[entity_item_col].dropna() if to_sancus_key(str(v))]
    if not keys:
        logger.warning("load_cluster_context: no usable Entity--Item keys on %s",
                       entity_item_col)
        return attach_cluster_descriptions(
            cat_data, pd.DataFrame(), pd.DataFrame(),
            attribute_columns=attribute_columns,
            max_descriptions=max_descriptions,
            entity_item_col=entity_item_col,
            own_description_col=own_description_col,
            similarity_threshold=similarity_threshold,
        )

    ids = loader.get_item_cluster_ids(keys)
    members = (
        loader.get_cluster_members(ids[NORMALIZED_CLUSTER_ID].dropna().unique().tolist())
        if not ids.empty
        else pd.DataFrame()
    )
    return attach_cluster_descriptions(
        cat_data, ids, members,
        attribute_columns=attribute_columns,
        max_descriptions=max_descriptions,
        entity_item_col=entity_item_col,
        own_description_col=own_description_col,
        similarity_threshold=similarity_threshold,
    )
