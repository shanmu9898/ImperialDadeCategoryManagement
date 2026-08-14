# CLAUDE.md

Project-specific guidance for Claude Code working in this repository.
For what the pipeline does and how to run it, read `README.md` first.
This file records the things that are **not** discoverable from the code, and the traps that fail silently.

Team: `datascience`. Contact this team for issues, not central platform.

---

## Orientation

Category-agnostic substitution and vendor-consolidation pipeline.
One YAML per product category drives everything; there is no per-category Python.

Five stages: Taxonomy (`taxonomy_load` + `taxonomy_classify`), Matching, Feedback, Optimization, Report.
Stages 4 and 5 are stubbed pending a rebuild.

Entry points, in order of how much they are actually used:

| Script | Purpose |
|---|---|
| `run_cups_full.py` | Full cups run, Stage 1 (+ Stage 2). `--skip-stage2` stops after Stage 1. |
| `run_cups_stage2.py` | Stage 2 alone, reading `Cups_Attributed.csv` off disk. |
| `run_cups_stages.py` | Row-capped smoke over the ~200-item cached sample. |
| `build_comparison_output.py` | Post-hoc primary/substitute comparison workbook. |
| `imperial-dade` CLI | Papermill-executes `notebooks/pipeline_template.ipynb`. |

---

## Run Stage 1 and Stage 2 as separate processes

Stage 2's embedding calls fail with `Connection error` on every batch when issued seconds after Stage 1 saturates the Azure endpoint.
The same code succeeds in a fresh process.
Use `run_cups_full.py --skip-stage2`, then `run_cups_stage2.py`.
Note that `make_embeddings` swallows those failures at INFO level and fills `None`, so a failed run still looks like it succeeded: always grep the log for `Batch .* failed`.

Reference timings and cost on the full 4,568-item cups universe:

| Phase | Time | Cost |
|---|---|---|
| Phases 1-3 (all cached) | ~2 min | - |
| Sancus cluster resolution | ~2.5 min | - |
| Stage 1 (LLM extraction) | ~16 min | ~$15 |
| Stage 2 (embeddings + matching) | ~42 min | included above |

---

## Caches

Every slow source pull is cached as a pickle under `~/.imperial_dade/smoke_cache/` (~6 GB).
The `cached()` helper is duplicated in each run script.
Keys are **filenames only** - no content hash, no TTL - so a cache never invalidates itself.
Bust it by deleting the file, pointing `IMPERIAL_DADE_SMOKE_CACHE` elsewhere, or `--refresh` (only `_tmp_stage1_smoke.py` has that flag).

A poisoned cache is a real failure mode: an early cluster run wrote all-empty columns and the next run happily reused them.
If a cached artifact looks wrong, delete it rather than reasoning around it.

There is **no LLM response cache**.
The only saving is within-run deduplication of identical descriptions, which recovers almost nothing (12 of 4,568).
Any prompt edit re-spends the full Stage 1 cost.

---

## Fabric lakehouse access

Use `imperial_dade.io.fabric`; do not hand-roll a connection.
The stack is pandas -> `mssql_python` 1.6.0 -> `ddbc_bindings` C extension -> `msodbcsql18.dll` **vendored inside the pip wheel** -> TDS over TCP 1433.
The driver is not registered with the Windows ODBC Driver Manager and needed no admin rights; it is per-venv, so a rebuilt venv needs `pip install mssql-python` again.
Do not put `DRIVER={ODBC Driver 18 for SQL Server}` in the connection string.
(`io/fornax.py` is different: pyodbc against the registered legacy `DRIVER=SQL Server` on `ibp-db01`.)

Six traps, all of which fail silently or misleadingly:

1. **Three-part naming, never a different `Database=`.**
   The host is a warehouse endpoint. Cross-lakehouse tables work as `[db].[schema].[table]`; setting `Database=lh_idedw_sancus` hangs forever.
2. **No bound parameters in predicates.**
   `WHERE col IN (?, ?)` never returns. Inline escaped literals via `_sql_literal`.
3. **Collation is case-sensitive.**
   `entity_item_code` is stored lowercased; an uppercase lookup returns zero rows without erroring. Always `LOWER()` both sides.
4. **`pd.read_sql(..., chunksize=N)` hangs.** Read unchunked.
5. **Non-UTF-8 bytes (0xA0) in text columns** break `.astype(str)`. Use `_to_clean_str`.
6. **`Connection Timeout=N` is an invalid keyword.** Pass `connect(..., timeout=N)`; exposed as `FABRIC_CONNECT_TIMEOUT`.

Auth is Entra ID interactive: first call opens a browser, then the cached token works headlessly until it expires.
**A non-interactive shell cannot recover from expiry** - it fails with `User canceled sign in`, and a human must run one command themselves.

### Capacity throttling

Heavy full-table scans exhaust the workspace's Fabric capacity.
Once exhausted the endpoint refuses **connections**, not just queries, and a trivial `SELECT 1` hangs; recovery takes roughly an hour.
Always set `FABRIC_CONNECT_TIMEOUT` so health checks fail fast instead of hanging ten minutes.

Prefer targeted lookups over full scans. Measured on `src_sancus.item_cluster` (3,529,408 rows):

- full-table read: ~25 min, and it is what caused the throttling
- 100 clusters via chunked `IN` (chunk 100): 9.7s, so ~7 min for 4,164 clusters
- small `IN` lookups when healthy: 0.4-1.3s

We read through the SQL endpoint, never the `abfss://` OneLake path.
To resolve which lakehouse a GUID means, list `sys.databases` then search `[db].INFORMATION_SCHEMA.TABLES`.

---

## How a category is filtered

The category name never touches the item master.
It is resolved to a set of keys first:

```
cups.yaml `name: Cups`
  -> matched against ItemCategory_type in src_reltio.item_segment
  -> 600 candidate `Item Segment Key` values
  -> item_master_raw["Item Segment Key"].isin(keys)   -> 61,216 rows
  -> Entity--Item startswith "1--" (S2K only)         -> 48,076 rows
  -> 7,779 distinct Item Code                         -> feeds the sales pull
```

`Item Segment Key` does not exist in Reltio; `FabricLoader.get_item_segment_mapping()` reconstructs the legacy Fornax key from the pipe-delimited `item_segment_group_code` as `f"{branch_company}--{division}-{klass}"`.
This exists because `fornax.dbo.Item_Segment` is empty in production.

Two things to know: the coupling is a bare string equality on the category name, so a typo yields zero keys and an empty run rather than an error; and only 23 of the 600 Cups keys are productive, so that 600 is not a coverage measure.

---

## Sancus cluster descriptions (Stage 1)

Join on `entity_item_code`, which is exactly `<entity>_<item_code>` (verified on 100.00% of rows) and shares our `Entity--Item` numbering, so `1--12HDQW` -> `1_12hdqw`. Use `fabric.to_sancus_key`.
Entity 1 is Imperial US S2K.

**Never join on bare `item_code`.**
The same code under another branch company is frequently a different product: item `CC` is a coffee-cup sleeve under entity 1 but reaches "upsg freight charge" and "labor to repair tool" rows under others.
The similarity threshold in `stages/clusters.py` is a second line of defence against junk cluster members, not a tuning nicety.

Ordering matters: `Combined Descriptions` must exist **before** clusters are attached, because siblings are scored against it.
`taxonomy_classify.run()` builds it at Step 11, which is too late for a caller doing its own attachment - call `build_combined_descriptions()` first.
Getting this wrong rejects every cluster silently while the run still reports success.

Clusters can mix variants (one real cups cluster groups a translucent and a black portion cup), so extraction runs per item, never once per cluster, and the prompt tells the model the primary description wins on conflict.

---

## Prompt templates

`llm/prompts/*.j2` render with `StrictUndefined`, so any new variable must be defaulted (`| default(false)`) or every existing caller breaks.

Access rule fields with **brackets** (`rule['values']`), never `rule.values` - dicts expose `.values` as a method and Jinja crashes.

`format_df_prompts` calls `.format(**row)` on the rendered system prompt, so a literal `{` or `}` reaching the template will raise `KeyError`.
Keep row-level free text in the user prompt.

---

## Testing

`pytest tests/test_smoke.py` is the fast offline suite and should stay green.
`tests/test_fabric_connectivity.py` and `tests/test_fornax_connectivity.py` hit live databases, take ~16 minutes, and need a valid Entra token.

Validate data changes against real cached data rather than synthetic fixtures where possible.
Two defects in the cluster work were invisible to unit tests and only appeared against the real 640k-row Sancus slice.
