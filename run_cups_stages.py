r"""Run Cups Stage 1 (Taxonomy) + Stage 2 (Matching) on the cached ~200-item
sample with the REAL Azure LLM, then stop.

The Stage 2 workbook (Cups_Subs_<date>.xlsx) is the human-review file: mark up
its Accept/Reject + Feedback columns and feed it back to Stage 3 later.

Data loads reuse ~/.imperial_dade/smoke_cache/*.pkl (full salsify / item-master
/ bridge / segment + the 200-code cups sales sample), so NO interactive Fabric/
Fornax auth is needed. Only Azure OpenAI is called live (key auth).

Outputs (under Data/Cups/Output/):
    Stage 1:  Cups_Attributed.csv, Cups_taxonomy.xlsx, Cups_coverage_improvement.csv
    Stage 2:  Cups_Subs_<date>.xlsx  (<- review this), Cups_matches.csv
Log:  Data/Cups/Output/run_cups_stages_<date>.log

Run:
    .\.venv312\Scripts\python.exe run_cups_stages.py
"""
from __future__ import annotations

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
    "ignore",
    message="pandas only supports SQLAlchemy connectable.*",
    category=UserWarning,
)

CFG_NAME = "cups"
LLM_ROW_CAP = 200          # cap classify/matching to the ~200-item sample
OUTPUT_DIR = Path("Data") / "Cups" / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / f"run_cups_stages_{date.today():%Y-%m-%d}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
              logging.StreamHandler()],
    force=True,
)
log = logging.getLogger("run_cups")

CACHE_DIR = Path(
    os.environ.get("IMPERIAL_DADE_SMOKE_CACHE",
                   str(Path.home() / ".imperial_dade" / "smoke_cache"))
)


def cached(name: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        df = pd.read_pickle(path)
        log.info("CACHE HIT  %-28s shape=%s", name, df.shape)
        return df
    log.info("CACHE MISS %s — pulling fresh (this needs DB auth!)", name)
    t0 = time.time()
    df = fn()
    log.info("CACHE FILL %-28s shape=%s (%.1fs)", name, df.shape, time.time() - t0)
    df.to_pickle(path)
    return df


def main() -> int:
    t_start = time.time()
    log.info("=" * 80)
    log.info("Cups Stage 1 + Stage 2 run — cached sample, real Azure LLM")
    log.info("Log file: %s", LOG_PATH.resolve())
    log.info("=" * 80)

    from imperial_dade.categories import load_category
    from imperial_dade.config.tables import get_fabric_tables, get_fornax_tables
    get_fabric_tables.cache_clear()
    get_fornax_tables.cache_clear()

    cfg = load_category(CFG_NAME)

    # ---- Phase 1: cached source loads (no DB auth on cache hit) -------------
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

    # ---- Phase 2: filter item-master to cups + cached sales -----------------
    log.info("--- Phase 2: cups segment filter + cached sales ---")
    segment_keys = (
        segment_map.loc[segment_map["Item Segment"] == cfg.name, "Item Segment Key"]
        .dropna().unique().tolist()
    )
    item_master = item_master_raw[item_master_raw["Item Segment Key"].isin(segment_keys)].copy()
    log.info("item_master filtered to %d cups rows", len(item_master))

    s2k_items = item_master[item_master["Entity--Item"].str.startswith("1--", na=False)]
    category_item_codes = (
        s2k_items["Item Code"].dropna().unique().tolist()
        or item_master["Item Code"].dropna().unique().tolist()
    )[:200]

    def _sales():
        from imperial_dade.io.fornax import FornaxLoader
        with FornaxLoader() as f:
            return f.get_sales_data(category_item_codes, cfg, entity_id=1)

    cat_data = cached(f"sales_{cfg.name}_smoke_s2k", _sales)

    # ---- Phase 3: attach item attrs, group, coverage, salsify merge --------
    log.info("--- Phase 3: group_data + merge_with_salsify ---")
    from imperial_dade.stages import taxonomy_load

    item_attrs = ["Item Desc 1", "Item Desc 2", "Case Pack", "VB Flag", "VGN", "VPN"]
    attrs_to_merge = [c for c in item_attrs if c in item_master.columns]
    item_master_attrs = (
        item_master[["Entity--Item"] + attrs_to_merge]
        .drop_duplicates(subset=["Entity--Item"], keep="first")
    )
    cat_data = cat_data.merge(item_master_attrs, on="Entity--Item", how="left")

    cat_data_grouped = taxonomy_load.group_data(cat_data)
    _, columns_with_coverage, _ = taxonomy_load.get_columns_with_coverage(
        cat_data_grouped, sfy, cfg.taxonomy.coverage_threshold,
    )
    columns_for_description = cfg.taxonomy.columns_for_description or columns_with_coverage
    cat_data_final = taxonomy_load.merge_with_salsify(
        cat_data_grouped, sfy, columns_for_description,
    )
    log.info("cat_data_final: %s ; attribute cols: %d",
             cat_data_final.shape, len(columns_for_description))

    # ---- Stage 1: Taxonomy classify (real LLM) -----------------------------
    log.info("=" * 80)
    log.info("STAGE 1 — Taxonomy classify (real LLM, capped at %d rows)", LLM_ROW_CAP)
    log.info("=" * 80)
    from imperial_dade.llm.client import OpenAIAgent
    from imperial_dade.stages import taxonomy_classify

    smoke_input = cat_data_final.head(LLM_ROW_CAP).copy()

    # Sancus cluster descriptions. This runner is row-capped, so the targeted
    # per-item lookup is cheaper here than the full-table read run_cups_full.py
    # uses — it only resolves the codes actually being processed.
    cluster_loader = None
    if cfg.taxonomy.use_cluster_descriptions:
        from imperial_dade.io.fabric import FabricLoader
        cluster_loader = FabricLoader()
        log.info("Cluster descriptions ON — resolving %d item codes", len(smoke_input))

    agent1 = OpenAIAgent(model="gpt-4.1", chunk_size=32)
    t0 = time.time()
    try:
        attributed, taxonomy_df = taxonomy_classify.run(
            smoke_input, agent1, cfg, columns_for_description, output_dir=OUTPUT_DIR,
            cluster_loader=cluster_loader,
        )
    finally:
        if cluster_loader is not None:
            cluster_loader.close()
    log.info("STAGE 1 done in %.1fs — %d rows attributed (LLM cost $%.4f)",
             time.time() - t0, len(attributed), agent1.get_cost())

    # ---- Stage 2: Matching (real LLM) --------------------------------------
    log.info("=" * 80)
    log.info("STAGE 2 — Matching (real LLM, top_n=%d)", cfg.matching.top_n)
    log.info("=" * 80)
    from imperial_dade.stages import matching

    t0 = time.time()
    im_final = matching.run(
        attributed, agent1, category=cfg, output_dir=OUTPUT_DIR,
        batch_model=False, pl=True, enable_vpn_exclusion=True,
    )
    matches_csv = OUTPUT_DIR / f"{cfg.name}_matches.csv"
    im_final.to_csv(matches_csv, index=False)
    log.info("STAGE 2 done in %.1fs — wrote %s", time.time() - t0, matches_csv.name)

    # ---- Summary -----------------------------------------------------------
    subs = sorted(OUTPUT_DIR.glob(f"{cfg.name}_Subs_*.xlsx"))
    subs = [p for p in subs if "post_feedback" not in p.name and not p.name.startswith("~$")]
    log.info("=" * 80)
    log.info("DONE in %.1f min total", (time.time() - t_start) / 60)
    for f in ("Cups_Attributed.csv", "Cups_taxonomy.xlsx",
              "Cups_coverage_improvement.csv", "Cups_matches.csv"):
        p = OUTPUT_DIR / f
        log.info("  %s  (%s)", p, f"{p.stat().st_size:,} B" if p.exists() else "MISSING")
    if subs:
        log.info("  REVIEW THIS -> %s", subs[-1].resolve())
    log.info("Next: mark up Accept/Reject + Feedback in that workbook, save as "
             "%s_Subs*_with_Feedback.xlsx, then run Stage 3.", cfg.name)
    log.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
