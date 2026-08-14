r"""Re-run Stage 2 (Matching) ONLY, from the cached Stage 1 output.

Stage 1 (taxonomy) + the full sales pull already completed and are on disk, so
this skips straight to embeddings + matching over Cups_Attributed.csv and writes
the substitutes workbook for human review. Real Azure LLM (key auth).

Outputs (Data/Cups/Output/):
    Cups_Subs_<date>.xlsx   (<- review this)
    Cups_matches.csv        (matches, embeddings column dropped)
Log: Data/Cups/Output/run_cups_stage2_<date>.log

Run:
    .\.venv312\Scripts\python.exe run_cups_stage2.py
"""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env", override=True)

CFG_NAME = "cups"
OUTPUT_DIR = Path("Data") / "Cups" / "Output"
ATTRIBUTED_CSV = OUTPUT_DIR / "Cups_Attributed.csv"
LOG_PATH = OUTPUT_DIR / f"run_cups_stage2_{date.today():%Y-%m-%d}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
              logging.StreamHandler()],
    force=True,
)
log = logging.getLogger("run_cups_stage2")


def main() -> int:
    t0 = time.time()
    log.info("=" * 80)
    log.info("Stage 2 ONLY — Matching from %s", ATTRIBUTED_CSV)
    log.info("Log: %s", LOG_PATH.resolve())
    log.info("=" * 80)

    if not ATTRIBUTED_CSV.exists():
        raise FileNotFoundError(f"Stage 1 output not found: {ATTRIBUTED_CSV}")

    from imperial_dade.categories import load_category
    from imperial_dade.llm.client import OpenAIAgent
    from imperial_dade.stages import matching

    cfg = load_category(CFG_NAME)
    attributed = pd.read_csv(ATTRIBUTED_CSV)
    log.info("Loaded attributed frame: %s", attributed.shape)
    assert "Entity--Item" in attributed.columns, "attributed frame missing Entity--Item"
    assert "attributes" in attributed.columns, "attributed frame missing attributes"

    agent = OpenAIAgent(model="gpt-4.1", chunk_size=32)
    im_final = matching.run(
        attributed, agent, category=cfg, output_dir=OUTPUT_DIR,
        batch_model=False, pl=True, enable_vpn_exclusion=True,
    )

    # Drop the giant embedding vectors, and the cluster prompt scaffolding —
    # the sibling-description block is multi-line text that bloats the CSV and
    # is already preserved in Cups_Attributed.csv. Cluster ID / Size stay, since
    # they're small and useful when auditing why an item matched.
    matches_csv = OUTPUT_DIR / f"{cfg.name}_matches.csv"
    im_final.drop(
        columns=["embeddings", "for_embedding",
                 "Cluster Descriptions", "Cluster Attribute Hints"],
        errors="ignore",
    ).to_csv(matches_csv, index=False)

    subs = sorted(OUTPUT_DIR.glob(f"{cfg.name}_Subs_*.xlsx"))
    subs = [p for p in subs if "post_feedback" not in p.name
            and "Comparison" not in p.name and not p.name.startswith("~$")]
    log.info("=" * 80)
    log.info("STAGE 2 done in %.1f min — wrote %s", (time.time() - t0) / 60, matches_csv.name)
    if subs:
        log.info("  REVIEW THIS -> %s", subs[-1].resolve())
    log.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
