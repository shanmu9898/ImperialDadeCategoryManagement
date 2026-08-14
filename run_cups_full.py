r"""FULL Cups run — Stage 1 (Taxonomy) + Stage 2 (Matching) for ALL cups SKUs,
stopping at the substitutes Excel that humans review (i.e. before Stage 3
feedback).

Differs from run_cups_stages.py: NO row cap, ALL S2K cups item codes (~7,779),
and a fresh FULL sales pull (cached to sales_Cups_full.pkl so it's pulled once).
Salsify / item-master / bridge / segment come from the existing full caches, so
the only live calls are the Fornax sales pull (token auth) + Azure OpenAI (key
auth) — no interactive login, runs unattended.

Outputs (Data/Cups/Output/):
    Stage 1: Cups_Attributed.csv, Cups_taxonomy.xlsx, Cups_coverage_improvement.csv
    Stage 2: Cups_Subs_<date>.xlsx  (<- review this), Cups_matches.csv (no embeddings col)
Log: Data/Cups/Output/run_cups_full_<date>.log

Run:
    .\.venv312\Scripts\python.exe run_cups_full.py
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env", override=True)
warnings.filterwarnings(
    "ignore", message="pandas only supports SQLAlchemy connectable.*", category=UserWarning,
)

CFG_NAME = "cups"
OUTPUT_DIR = Path("Data") / "Cups" / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / f"run_cups_full_{date.today():%Y-%m-%d}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
              logging.StreamHandler()],
    force=True,
)
log = logging.getLogger("run_cups_full")

CACHE_DIR = Path(os.environ.get(
    "IMPERIAL_DADE_SMOKE_CACHE", str(Path.home() / ".imperial_dade" / "smoke_cache")))


def cached(name: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        df = pd.read_pickle(path)
        log.info("CACHE HIT  %-26s shape=%s", name, df.shape)
        return df
    log.info("CACHE MISS %s — pulling fresh (live source)", name)
    t0 = time.time()
    df = fn()
    log.info("CACHE FILL %-26s shape=%s (%.1f min)", name, df.shape, (time.time() - t0) / 60)
    df.to_pickle(path)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-stage2", action="store_true", dest="skip_stage2",
        help="Stop after Stage 1. Stage 2's embedding calls have failed with "
             "'Connection error' when issued seconds after Stage 1 saturates the "
             "Azure endpoint; running run_cups_stage2.py afterwards in a fresh "
             "process reads Cups_Attributed.csv off disk and works reliably.",
    )
    args = ap.parse_args()

    t_start = time.time()
    log.info("=" * 80)
    log.info("FULL Cups run — Stage 1 + Stage 2 (ALL cups SKUs), real Azure LLM")
    log.info("Log: %s", LOG_PATH.resolve())
    log.info("=" * 80)

    from imperial_dade.categories import load_category
    from imperial_dade.config.tables import get_fabric_tables, get_fornax_tables
    get_fabric_tables.cache_clear()
    get_fornax_tables.cache_clear()
    cfg = load_category(CFG_NAME)

    # ---- Phase 1: cached source loads --------------------------------------
    log.info("--- Phase 1: source loads (cache) ---")

    def _bridge():
        from imperial_dade.io.fabric import FabricLoader
        with FabricLoader() as f:
            return f.get_salsify_to_s2k_mapping(entity_id=1)

    def _segment():
        from imperial_dade.io.fabric import FabricLoader
        with FabricLoader() as f:
            return f.get_item_segment_mapping(branch_company_code="1")

    def _salsify():
        from imperial_dade.io.fornax import FornaxLoader
        with FornaxLoader() as f:
            return f.get_salsify_items(salsify_to_s2k=bridge)

    def _item_master():
        from imperial_dade.io.fornax import FornaxLoader
        with FornaxLoader() as f:
            return f.get_consolidated_items(entity_id=1)

    bridge = cached("salsify_bridge", _bridge)
    segment_map = cached("item_segment", _segment)
    sfy = cached("salsify_items", _salsify)
    item_master_raw = cached("item_master_raw", _item_master)

    # ---- Phase 2: cups filter + FULL sales pull ----------------------------
    log.info("--- Phase 2: cups segment filter + FULL sales pull ---")
    segment_keys = (segment_map.loc[segment_map["Item Segment"] == cfg.name, "Item Segment Key"]
                    .dropna().unique().tolist())
    item_master = item_master_raw[item_master_raw["Item Segment Key"].isin(segment_keys)].copy()
    s2k_items = item_master[item_master["Entity--Item"].str.startswith("1--", na=False)]
    category_item_codes = s2k_items["Item Code"].dropna().unique().tolist()
    log.info("Cups item-master rows: %d ; S2K item codes (ALL): %d",
             len(item_master), len(category_item_codes))

    def _sales():
        from imperial_dade.io.fornax import FornaxLoader
        log.info("Pulling FULL sales for %d codes — expect ~2.5-3 h...", len(category_item_codes))
        with FornaxLoader() as f:
            return f.get_sales_data(category_item_codes, cfg, entity_id=1)

    cat_data = cached(f"sales_{cfg.name}_full", _sales)
    log.info("Full sales rows: %d", len(cat_data))

    # ---- Phase 3: attach item attrs, group, coverage, salsify merge --------
    log.info("--- Phase 3: group_data + merge_with_salsify ---")
    from imperial_dade.stages import taxonomy_load

    item_attrs = ["Item Desc 1", "Item Desc 2", "Case Pack", "VB Flag", "VGN", "VPN"]
    attrs_to_merge = [c for c in item_attrs if c in item_master.columns]
    item_master_attrs = (item_master[["Entity--Item"] + attrs_to_merge]
                         .drop_duplicates(subset=["Entity--Item"], keep="first"))
    cat_data = cat_data.merge(item_master_attrs, on="Entity--Item", how="left")

    cat_data_grouped = taxonomy_load.group_data(cat_data)
    _, columns_with_coverage, _ = taxonomy_load.get_columns_with_coverage(
        cat_data_grouped, sfy, cfg.taxonomy.coverage_threshold)
    columns_for_description = cfg.taxonomy.columns_for_description or columns_with_coverage
    cat_data_final = taxonomy_load.merge_with_salsify(cat_data_grouped, sfy, columns_for_description)
    log.info("cat_data_final: %s ; attribute cols: %d",
             cat_data_final.shape, len(columns_for_description))

    # ---- Phase 4: Sancus cluster descriptions (cached) ---------------------
    # Each item gets the other descriptions of the same physical product, so
    # Stage 1 can recover attributes this branch's wording omits. Cached because
    # the lakehouse scans are slow (minutes), not because the data is large.
    if cfg.taxonomy.use_cluster_descriptions:
        log.info("--- Phase 4: Sancus cluster descriptions ---")
        import gc

        from imperial_dade.stages import clusters, taxonomy_classify as _tc

        # Cluster matching scores siblings against the item's OWN description,
        # so that column must exist BEFORE attaching. taxonomy_classify.run()
        # builds it at Step 11, which is too late — without this every item
        # scores 0.0 and all clusters are silently rejected.
        cat_data_final = _tc.build_combined_descriptions(cat_data_final)
        _have_desc = cat_data_final["Combined Descriptions"].str.strip().ne("").sum()
        log.info("Combined Descriptions built: %d/%d non-empty",
                 _have_desc, len(cat_data_final))
        if _have_desc == 0:
            raise RuntimeError(
                "Combined Descriptions is empty for every row — cluster matching "
                "would reject everything. Check Item Desc 1/2 on the merged frame."
            )

        # Release the multi-GB source frames before reading the cluster table.
        # Everything downstream works off cat_data_final (4.5k rows), while
        # item-master + sales + salsify hold ~8 GB and the cluster read adds
        # several more on top. Rebinding is what frees them — `del locals()[...]`
        # is a no-op for function locals in CPython.
        item_master_raw = item_master = cat_data = sfy = None
        cat_data_grouped = s2k_items = item_master_attrs = None
        segment_map = bridge = None
        gc.collect()
        log.info("Released source frames before the cluster scan")

        def _clusters():
            from imperial_dade.io.fabric import FabricLoader

            # Escape hatch for when the lakehouse endpoint is throttled: point
            # IMPERIAL_DADE_CLUSTER_TABLE_PKL at a previously-pulled cluster
            # table (normalized schema) and skip the live read entirely.
            pkl = os.environ.get("IMPERIAL_DADE_CLUSTER_TABLE_PKL")
            if pkl:
                log.info("Reading cluster table from %s (no live query)", pkl)
                table = pd.read_pickle(pkl)
                log.info("Cluster table from disk: %s", table.shape)
                return clusters.attach_from_cluster_table(
                    cat_data_final,
                    table,
                    attribute_columns=cfg.taxonomy.cluster_attribute_hints,
                    max_descriptions=cfg.taxonomy.max_cluster_descriptions,
                    similarity_threshold=cfg.taxonomy.cluster_similarity_threshold,
                )[["Entity--Item", clusters.CLUSTER_ID_COL, clusters.CLUSTER_SIZE_COL,
                   clusters.CLUSTER_DESCRIPTIONS_COL, clusters.CLUSTER_HINTS_COL]]

            with FabricLoader() as f:
                # Targeted, NOT a full-table scan. We only need the ~4.2k
                # clusters our items belong to (~27k member rows), which is two
                # rounds of small chunked IN lookups. Reading all 3.5M rows
                # instead took 25+ min and is what kept exhausting the
                # workspace's capacity until the endpoint throttled outright.
                log.info("Resolving clusters for %d items (targeted lookups)",
                         len(cat_data_final))
                return clusters.load_cluster_context(
                    cat_data_final,
                    f,
                    attribute_columns=cfg.taxonomy.cluster_attribute_hints,
                    max_descriptions=cfg.taxonomy.max_cluster_descriptions,
                    similarity_threshold=cfg.taxonomy.cluster_similarity_threshold,
                )[["Entity--Item", clusters.CLUSTER_ID_COL, clusters.CLUSTER_SIZE_COL,
                   clusters.CLUSTER_DESCRIPTIONS_COL, clusters.CLUSTER_HINTS_COL]]

        cluster_cols = cached(f"clusters_{cfg.name}_full", _clusters)
        cat_data_final = cat_data_final.merge(cluster_cols, on="Entity--Item", how="left")
        for col in (clusters.CLUSTER_ID_COL, clusters.CLUSTER_DESCRIPTIONS_COL,
                    clusters.CLUSTER_HINTS_COL):
            cat_data_final[col] = cat_data_final[col].fillna("")
        cat_data_final[clusters.CLUSTER_SIZE_COL] = (
            cat_data_final[clusters.CLUSTER_SIZE_COL].fillna(0).astype(int)
        )
        with_siblings = cat_data_final[clusters.CLUSTER_DESCRIPTIONS_COL].str.strip().ne("").sum()
        log.info("Cluster context attached: %d/%d items have sibling descriptions",
                 with_siblings, len(cat_data_final))

    # ---- Stage 1: Taxonomy classify (real LLM, full) -----------------------
    log.info("=" * 80)
    log.info("STAGE 1 — Taxonomy classify (real LLM) on %d items", len(cat_data_final))
    log.info("=" * 80)
    from imperial_dade.llm.client import OpenAIAgent
    from imperial_dade.stages import taxonomy_classify

    agent = OpenAIAgent(model="gpt-4.1", chunk_size=32)
    t0 = time.time()
    attributed, taxonomy_df = taxonomy_classify.run(
        cat_data_final, agent, cfg, columns_for_description, output_dir=OUTPUT_DIR)
    log.info("STAGE 1 done in %.1f min — %d rows attributed (LLM cost $%.4f)",
             (time.time() - t0) / 60, len(attributed), agent.get_cost())

    if args.skip_stage2:
        log.info("=" * 80)
        log.info("--skip-stage2 set — Stage 1 complete, stopping here.")
        log.info("Next: .\\.venv312\\Scripts\\python.exe run_cups_stage2.py")
        log.info("=" * 80)
        for f in ("Cups_Attributed.csv", "Cups_taxonomy.xlsx",
                  "Cups_coverage_improvement.csv"):
            p = OUTPUT_DIR / f
            log.info("  %s  (%s)", p, f"{p.stat().st_size:,} B" if p.exists() else "MISSING")
        log.info("TOTAL %.1f min (LLM cost $%.4f)",
                 (time.time() - t_start) / 60, agent.get_cost())
        return 0

    # ---- Stage 2: Matching (real LLM, full) --------------------------------
    log.info("=" * 80)
    log.info("STAGE 2 — Matching (real LLM, top_n=%d) on %d items",
             cfg.matching.top_n, len(attributed))
    log.info("=" * 80)
    from imperial_dade.stages import matching

    t0 = time.time()
    im_final = matching.run(
        attributed, agent, category=cfg, output_dir=OUTPUT_DIR,
        batch_model=False, pl=True, enable_vpn_exclusion=True)
    # Save matches without the giant embeddings columns (keeps the CSV usable).
    # Drop the giant embedding vectors and the cluster prompt scaffolding (the
    # sibling-description block is multi-line text that bloats the CSV and is
    # already in Cups_Attributed.csv). Cluster ID / Size stay for auditing.
    matches_csv = OUTPUT_DIR / f"{cfg.name}_matches.csv"
    im_final.drop(
        columns=["embeddings", "for_embedding",
                 "Cluster Descriptions", "Cluster Attribute Hints"],
        errors="ignore",
    ).to_csv(matches_csv, index=False)
    log.info("STAGE 2 done in %.1f min — wrote %s (LLM cost so far $%.4f)",
             (time.time() - t0) / 60, matches_csv.name, agent.get_cost())

    # ---- Summary -----------------------------------------------------------
    subs = sorted(OUTPUT_DIR.glob(f"{cfg.name}_Subs_*.xlsx"))
    subs = [p for p in subs if "post_feedback" not in p.name
            and "Comparison" not in p.name and not p.name.startswith("~$")]
    log.info("=" * 80)
    log.info("DONE in %.1f min total (LLM cost $%.4f)",
             (time.time() - t_start) / 60, agent.get_cost())
    for f in ("Cups_Attributed.csv", "Cups_taxonomy.xlsx",
              "Cups_coverage_improvement.csv", "Cups_matches.csv"):
        p = OUTPUT_DIR / f
        log.info("  %s  (%s)", p, f"{p.stat().st_size:,} B" if p.exists() else "MISSING")
    if subs:
        log.info("  REVIEW THIS -> %s", subs[-1].resolve())
    log.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
