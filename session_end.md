# Session handoff — 2026-06-16

## This session: rule-aware Stage 1 extraction + single-VB substitutes

Two feature changes landed (all in the working tree, **nothing committed**):

### Change 1 — deterministic dict extractor folded into the Stage 1 LLM prompt
A pasted regex/dictionary cups extractor (TYPE_MAP, MATERIAL_MAP, COLOR_MAP,
capacity/diameter patterns, wall/slot/hole/sip/foam, etc.) was turned into
**prompt guidance + YAML data** — no new columns, no separate extractor, the
existing `columns_for_description` set is still what's extracted.
- `categories/base.py`: new frozen `AttributeRule` model (`synonyms`, `values`,
  `pattern`, `notes`) + `TaxonomyConfig.extraction_rules: dict[str, AttributeRule]`
  (default `{}` → other categories unchanged).
- `categories/cups.yaml`: all pasted dicts moved into `taxonomy.extraction_rules`,
  keyed to existing attribute names (Beverage Cup Type, Material, Usage
  Temperature, Product Capacity, Color, Foodservice Global Attributes, Beverage
  Cup Style, Pattern & Design). Edge-case logic from the script encoded as
  `notes` (type specificity precedence; `strawless` = No Slot + Sip Through;
  abbreviation handling; oz vs mm/series).
- `llm/prompts/attribute_extraction.j2`: renders a normalization-rules block
  when `extraction_rules` present. NOTE: uses **bracket access** (`rule['values']`)
  not `rule.values` — dicts expose `.values` as a method, which crashes Jinja.
- `stages/taxonomy_classify.py`: `run()` passes `cfg.taxonomy.extraction_rules`
  (model_dump'd) through `attribute_with_ai_optimized` → `render_prompt`.

### Change 2 — non-VB targets get all non-VB subs + at most ONE Victoria Bay sub
Edited the shared `stages/matching.py:write_top_matches` (Stage 3 imports the
same fn, so both stages get it):
- Targets stay non-VB only (unchanged). Matches split by raw VB Flag: every
  `'N'`/`'Y - Other'` match stays a regular substitute; only the single
  highest-ranked `'Y - VB'` match is kept. Empty is allowed.
- New **"Recommendation"** fixed column per row: `Target` / `Substitute` /
  `VB Substitute`. Raw flag carried on row dicts via `_raw_vb_flag`/`_row_type`
  (set in `_create_target_data`/`_create_match_data`).

### Decisions taken (from the user's answers)
- Rules **supplement** the LLM via the prompt, **no new columns**.
- VB: keep subs, **flag the one VB separately** (its own column), empty if none.
- Keep the **current matching attribute logic** (critical/directional in
  cups.yaml unchanged); rule logic integrated into the description/extraction
  guidance only.

### Open follow-ups
- Attribute mapping in `cups.yaml` `extraction_rules` is a first pass — the
  user may want to retune which existing column each rule concept maps to
  (esp. Wall/Slot/Diameter, which lack a dedicated column).
- Real LLM end-to-end on live Fornax/Fabric data not run this session (needs DB
  + Azure). Verified instead via a deterministic stub (see below).

### How it was verified
- `_tmp_stage123_smoke.py` — offline stub-agent run of Stage 1→2→3 on a 5-row
  cups sample (no network/DB). All three stages pass; each non-VB target shows
  Target + 2 non-VB subs + exactly 1 VB substitute, in both the matching and
  post-feedback workbooks.
- `tests/test_smoke.py` — 3 new permanent tests (extraction rules render into
  the prompt; no-rules category unchanged; `write_top_matches` single-VB split).
  `pytest tests/test_smoke.py` → **13 passed**.
- Full `pytest`: 38 passed, 1 pre-existing unrelated failure
  (`test_keys_and_fornax.py::test_mapping_is_case_insensitive_for_s2k`, in
  `utils.keys` — not touched by this work).

---

# Session handoff — 2026-05-20

## Where we are

**Goal of the multi-session project:** get Stage 1 (Taxonomy) of the
category-substitution pipeline working end-to-end against live data
(Fornax + Microsoft Fabric lakehouse). The legacy pipeline depended on
`fornax.dbo.Item_Segment` (which is empty in production) and a
silent-wrong `Entity--Item = '1--' + ProductID` join that never actually
matched item-master rows.

**Status:** Stage 1 ran end-to-end on a 200-item / 50-LLM-row smoke. The
five caches survive in `~/.imperial_dade/smoke_cache/` (3.85 GB total) so
the next run picks up from there without re-pulling from Fornax. Three
real bugs were fixed in the smoke loop but the run that proves all fixes
together was rejected — see "Resume here" below.

**Nothing is committed.** All session work is in the working tree.

---

## Cache locations (persistent across sessions)

| File | Rows | Size | Source |
|---|---|---|---|
| `~/.imperial_dade/smoke_cache/salsify_bridge.pkl` | 495,044 | 16 MB | Fabric `src_s2k_r50modsdta.VIOITEM` |
| `~/.imperial_dade/smoke_cache/item_segment.pkl` | 7,193 | 281 KB | Fabric `src_reltio.item_segment` |
| `~/.imperial_dade/smoke_cache/salsify_items.pkl` | 434,580 | 818 MB | Fornax `fornax.salsify.products` |
| `~/.imperial_dade/smoke_cache/item_master_raw.pkl` | 3,210,186 | 3.2 GB | Fornax `fornax.dbo.ConsolidatedItemsByLocation` |
| `~/.imperial_dade/smoke_cache/sales_Cups_smoke.pkl` | 256,121 | 93 MB | Fornax sales — 200 codes |

Cache hits on the smoke take **< 5 seconds total**. Bypass with
`--refresh` or by setting `IMPERIAL_DADE_SMOKE_CACHE` to a different
directory.

**Note:** the old sales cache `sales_Cups_smoke.pkl` was pulled BEFORE we
added the S2K-only item filter. The smoke script now references a new key
`sales_Cups_smoke_s2k` — first run after resume will pull a fresh ~4-min
sales chunk for 200 S2K-keyed codes.

---

## What was completed this session

### C1 — Fabric `get_item_segment_mapping()` (done, tested)

`fornax.dbo.Item_Segment` is empty; replaced the lookup with the
lakehouse view `lh_idedw_business.src_reltio.item_segment`. Pipe-delimited
`item_segment_group_code` (`1|1|NA|509|3`) is split and reformatted into
the legacy `branch_company--division-class` key (`1--509-3`) so the
downstream `.isin()` filter on `ConsolidatedItemsByLocation.[Item Segment
Key]` just works.

- Cups → 600 candidate keys → 23 productive → **61,216 items** in Fornax
- Integration tests pass against live data

### C2 — Removed `add_po_cost` and related dead code (done)

Stages 2 (`matching.py`) and 3 (`feedback.py`) grep clean — zero
references. Removed the function (~200 lines), the `po_cost_amt_col`
branch in `group_data`, and the `add_po_cost` call from the notebook.

**Left alone:** `PipelineConfig.TRANSACTION_COLUMNS["pod"]` and the
`PO Cost → po_cost_amt` rename in `_CONSOLIDATED_RENAME`. Stage 4
(optimization) and Stage 5 (report) still reference them; per user
direction, those stages are being rebuilt later.

### C3 — Salsify column rename map for cups (done)

Fornax's `salsify.products` exports PascalCase (`ProductCapacity`,
`BeverageCupStyle`); cups.yaml uses display names with spaces.
`_SALSIFY_RENAME` in `io/fornax.py` translates 3 columns, applied
inside `get_salsify_items()` after the SQL pull:

```
ProductCapacity   -> Product Capacity
UsageTemperature  -> Usage Temperature
BeverageCupStyle  -> Beverage Cup Style
```

Result: 5 of 10 cups attributes now flow directly from Salsify (was 2).
The other 5 (`Beverage Cup Type`, `Product Type Collapse`, `Pack Size`,
`Foodservice Global Attributes`, `Pattern & Design`) have no Salsify
equivalent — kept in YAML per user direction so the LLM tries to extract
them from the item description.

### Coverage CSV persist (done)

`taxonomy_classify.run` now writes `<Category>_coverage_improvement.csv`
alongside the other outputs.

### Smoke run (mostly working — needs one more re-run)

`_tmp_stage1_smoke.py` exercises Stage 1 end-to-end with disk cache.
**Latest run output (rejected at end):**

- All caches hit cleanly
- Item-master merge attached `Item Desc 1/2 + Case Pack + VB Flag + VGN + VPN` to 256k sales rows
- `merge_with_salsify` attached 5 attribute cols, logged 5 LLM-only ones
- 74 rows survived to the LLM stage
- LLM extraction: 50 rows, 4 OpenAI calls, **$0.0012 total cost** in 6 seconds
- Wrote `Cups_Attributed.csv` (50 rows, 18 KB)

**But output values were empty** because the first 200 item codes happened
to be on a non-S2K ERP (`Entity--Item = "2--..."`). Salsify has only S2K
items (`"1--..."`), so no rows matched.

This session ended right after the three fixes for that:

1. **Smoke now filters to S2K items first** (line ~165 in `_tmp_stage1_smoke.py`)
2. **`coverage_improvement` handles LLM-only columns** (intersects with
   sfy.columns instead of KeyError; persists the CSV correctly)
3. **xlsxwriter NaN crash fixed** in `_write_xlsxwriter_excel` — column
   auto-sizing now stringifies values defensively
4. **Unicode print at end-of-smoke** replaced with ASCII (`[ok]`/`[MISSING]`)

The next run with these three fixes is the one that needs to happen.

---

## Files modified this session

### Stage 1 production code

- **`src/imperial_dade/io/fabric.py`** — added `FabricLoader.get_item_segment_mapping()`. Pipe-delimited group code → legacy branch--division-class format.
- **`src/imperial_dade/config/tables.py`** — added `FabricTables.item_segment_table`, `ITEM_SEGMENT_*` column constants.
- **`src/imperial_dade/stages/taxonomy_load.py`** — deleted `add_po_cost` (~200 lines); simplified `group_data` to drop the `po_cost_amt_col` branch and to only preserve extra columns that exist on the input frame.
- **`src/imperial_dade/stages/taxonomy_classify.py`** — (a) persist `Cups_coverage_improvement.csv`; (b) fix `coverage_improvement` to gracefully skip LLM-only columns; (c) defensive stringification in `_write_xlsxwriter_excel` to avoid NaN→`len()` crash.
- **`src/imperial_dade/io/fornax.py`** — added `_SALSIFY_RENAME` map (3 cups columns) applied inside `get_salsify_items()` after the SQL pull.

### Notebook

- **`notebooks/pipeline_template.ipynb`** — taxonomy cell rewritten to: (1) use `FabricLoader.get_item_segment_mapping()` instead of `loader.get_item_segment_mapping()`; (2) use `FabricLoader.get_salsify_to_s2k_mapping()` for the Salsify-S2K bridge; (3) drop the `add_po_cost` call.

### Tests

- **`tests/test_fabric_connectivity.py`** — added `test_item_segment_mapping_schema` and `test_item_segment_keys_join_to_fornax`. Both pass against live data.

### Config / env

- **`.env.example`** — added `FABRIC_TABLE_ITEM_SEGMENT` entry with default `src_reltio.item_segment`.
- **`.gitignore`** — added `_smoke_cache/` and `_tmp_*.py`.

### Smoke runner (local, not for commit)

- **`_tmp_stage1_smoke.py`** — disk-cached Stage 1 smoke runner. Caches at `~/.imperial_dade/smoke_cache/`. LLM capped at 50 rows; sales pull capped at 200 S2K-filtered item codes.

---

## Resume here

The current smoke script has all the fixes in place but **was never run after the last edits.** To pick up:

```powershell
# Working dir: C:\Users\MGadupudi\PycharmProjects\ImperialDadeCategoryManagement

# 1. Sanity check the venv + imports still work
.\.venv312\Scripts\python.exe -c "from imperial_dade.io.fabric import FabricLoader; print('ok')"

# 2. Run the smoke. All caches except sales_Cups_smoke_s2k will hit.
#    Expect: ~4 min for the new S2K-filtered sales pull, then everything
#    runs in seconds. LLM extraction is ~6s. Total: ~5 min on cold sales.
"" | Out-File -Encoding utf8 _smoke_cache\smoke.log
"" | Out-File -Encoding utf8 _smoke_cache\smoke.err
$proc = Start-Process -FilePath .\.venv312\Scripts\python.exe -ArgumentList '-u', '_tmp_stage1_smoke.py' -RedirectStandardOutput _smoke_cache\smoke.log -RedirectStandardError _smoke_cache\smoke.err -NoNewWindow -PassThru
Write-Output ("Started PID: " + $proc.Id)

# 3. Tail the log
Get-Content _smoke_cache\smoke.err -Wait
```

### What to verify in the output

1. `Cups_Attributed.csv` has **non-empty** values for `Product Capacity`, `Usage Temperature`, `Material`, `Color`, `Beverage Cup Style` (the 5 Salsify-direct cols).
2. The LLM-only columns (`Beverage Cup Type`, `Product Type Collapse`, `Pack Size`, `Foodservice Global Attributes`, `Pattern & Design`) get filled in by gpt-4.1 from the description.
3. `Cups_taxonomy.xlsx` writes without the `'float' has no len()` error.
4. `Cups_coverage_improvement.csv` is produced and has rows for every YAML attribute.
5. Total LLM cost stays in cents (< $0.01).

### If something's wrong

The most likely failure mode: the S2K filter (`item_master["Entity--Item"].str.startswith("1--")`) might filter to an empty list if Cups items in segments 1--509-* and 1--506-* are all on non-S2K ERPs. The script falls back to all codes in that case (`s2k_codes or category_item_codes_all`) — that fallback shouldn't fire for cups (we saw S2K Cups items in the bridge probe), but watch the `S2K-keyed Cups item codes: NN` log line.

---

## What's left after the smoke passes

### Immediate (Stage 1 production readiness)

1. **Propagate the item-master merge to the notebook** — the smoke does an inline `cat_data.merge(item_master[attrs])` between sales pull and `group_data`. The notebook taxonomy cell needs the same merge. Could also live as a helper in `taxonomy_load.py` (e.g. `enrich_with_item_master_attrs(cat_data, item_master, attrs)`).
2. **Full Stage 1 run with all 7,793 item codes** — user wants to do this once the smoke is verified. **Heads-up:** the sales pull at full size is ~2.5 hours at the current chunk-of-1000 IN clause rate. Options before launching: (a) just let it run overnight, (b) reduce chunk_size to 200 codes per chunk to keep memory bounded, (c) push more filtering server-side.
3. **`columns_dump.txt`** at the repo root is a leftover debug artifact — fine to delete.

### Medium term (other Stage 1 blockers we noted but deferred)

4. **Cutlery YAML reconciliation** — `cutlery.yaml` has `columns_for_description: null` (auto-discover). When we run cutlery, the auto-discovery via `get_columns_with_coverage` currently errors with `"'UL ECOLOGO Certification' is not in list"` because that column doesn't exist on Salsify anymore. The function silently falls back to the YAML-pinned list for cups (which is fine since it's pinned). For cutlery we need to either pin a list in cutlery.yaml or fix the auto-discovery to handle the missing-anchor case.
5. **C3 follow-up for cutlery** — only 3 PascalCase→spaced renames are in `_SALSIFY_RENAME`. Cutlery may need more (e.g. `FoodserviceTablewareProductType → Foodservice Tableware Product Type`). Defer until cutlery is the focus.

### Longer term (after Stage 1 is fully working)

6. **Stages 4/5 (optimization + report)** were stubbed `TODO` in the notebook. They reference `po_cost_amt` and the `POD` column — when they're rebuilt, they should source those from `item_master` directly, not via the removed `add_po_cost`.
7. **Lakehouse migration for item_master + sales** — the lakehouse has `business.item_location` (item-master) and `business.invoice_line_24m` (sales). Switching sources is a separate project but worth doing — would fix the slow Fornax pulls (12 min for item_master, hours for sales).
8. **Salsify source switch** — the lakehouse has `src_salsify.*` schemas that may have fresher data than `fornax.salsify.products`. Worth investigating once everything else is stable.

---

## Open questions / decisions to make

1. **Where does the item-master-attr merge belong long-term?** The smoke does it inline. Options:
   - A: leave it in the notebook (simplest, mirrors the legacy pattern)
   - B: add a helper to `taxonomy_load.py` (e.g. `enrich_transactions_with_item_attrs`) — cleaner, testable
   - C: do it inside `FornaxLoader.get_sales_data` — most magical, hides what's happening from the notebook
2. **Full Stage 1 run scheduling** — 2.5 hours is a lot. Is overnight OK, or do we want to optimize the sales loader first?
3. **`columns_dump.txt`** — delete?

---

## Open issues worth knowing about

- The Azure OpenAI API key lives only in `.env` (key prefix redacted from this note - do not quote real key material in tracked files). Verified 2026-08: `.env` is gitignored and has never appeared in git history, and `_legacy/.../pyvent.py` reads the key from the environment rather than hardcoding it. No rotation is required on account of this repo.
- Fornax `pd.read_sql` emits a `UserWarning` about not being a SQLAlchemy connectable. Cosmetic; suppressed in `FabricLoader._read_sql` but not yet in `FornaxLoader`. Fix later by wrapping the pyodbc conn in `sqlalchemy.create_engine`.
- `add_po_cost` deletion removed the negative-row cleanup (rows with `Qty <= 0 OR Gross Cost <= 0 OR Net Cost <= 0`). The smoke doesn't seem to be impacted, but if Stage 1 produces weird aggregates with negative values, we may need to reintroduce that cleanup separately.

---

## Tasks state at session end

```
#12  [completed] C1: Fabric get_item_segment_mapping + test
#13  [completed] C2: Remove add_po_cost (after confirming Stages 2/3 don't use it)
#14  [completed] C3: Salsify column rename for cups
#15  [completed] Persist coverage_improvement output
#16  [in_progress] Stage 1 smoke run with disk cache + 50-row LLM cap
                  ^- needs one final run with the S2K filter + coverage_improvement +
                     xlsxwriter + unicode print fixes
```

When the smoke produces the three expected outputs with real attribute values, close #16 and we're done with Stage 1 for cups.
