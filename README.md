# Imperial Dade Category Management

Category-agnostic substitution & vendor-consolidation pipeline.

The same pipeline runs for **any product category** (cups, cutlery, plates, …)
driven by a single YAML config per category. There is no per-category code —
only per-category data.

> **Team:** `datascience`. For issues or support, contact this team. Do not
> contact central platform.

---

## What it does

Five stages, one parameterized notebook:

| # | Stage          | What it produces |
|---|----------------|------------------|
| 1 | Taxonomy       | Cleaned item master + LLM-extracted attributes |
| 2 | Matching       | Top-N substitution candidates per SKU (LLM + embeddings) |
| 3 | Feedback       | Human-feedback-aware refined matches |
| 4 | Optimization   | MILP-driven vendor consolidation |
| 5 | Report         | Final Excel deliverable |

---

## Installation

```powershell
# 1. Python 3.12 venv
py -V:3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1

# 2. Install the package in editable mode
pip install -e .

# 3. (Optional) Install Jupyter UI if you want to edit notebooks interactively.
#    This is NOT a required dependency because the jupyterlab extension files
#    hit Windows' 260-char path limit on default installs. Enable long paths in
#    the registry first, or install in a shorter venv path:
pip install jupyterlab
```

---

## Configuration

Copy `.env.example` to `.env` at the repo root and fill in real values:

| Variable                       | Required | Purpose |
|--------------------------------|:--------:|---------|
| `AZURE_OPENAI_ENDPOINT`        | yes      | Azure OpenAI resource URL |
| `AZURE_OPENAI_API_KEY`         | yes      | API key for the resource above |
| `AZURE_BATCH_OPENAI_ENDPOINT`  | no       | Separate batch deployment; falls back to primary if unset |
| `AZURE_BATCH_OPENAI_KEY`       | no       | Batch key; falls back to primary if unset |
| `IMPERIAL_DADE_DATA_DIR`       | no       | Root for input/output files (default: repo-local `Data/`) |
| `IMPERIAL_DADE_LOG_LEVEL`      | no       | `DEBUG` / `INFO` (default) / `WARNING` / `ERROR` |

`.env` is gitignored. **Never** commit credentials.

---

## Running the pipeline

### CLI

```powershell
# What categories are available?
imperial-dade list-categories

# Inspect the system prompt the LLM will see for a category
imperial-dade render-prompt -c cups

# Run all stages for cups
imperial-dade run --category cups --stage all

# Run only the matching stage for cutlery
imperial-dade run --category cutlery --stage matching
```

The CLI invokes `notebooks/pipeline_template.ipynb` via Papermill with
`category` and `stage` as injected parameters, and writes the executed
notebook to `notebooks/runs/<category>_<stage>_run.ipynb`.

### Notebook (interactive)

Open `notebooks/pipeline_template.ipynb` in Jupyter. Edit the first cell
(tagged `parameters`) — change `category = "cups"` to whatever you want —
and run cell-by-cell.

---

## Adding a new category

**No Python code changes needed.** Just drop a YAML:

```yaml
# src/imperial_dade/categories/plates.yaml
name: Plates
data_dir: Plates
sales_file: "Plates Sales Data.xlsx"

matching:
  critical_attributes:
    - Plate Diameter
    - Material
    - Color
  directional_attributes:
    - Pattern
    - Sustainable
  exclusion_rules: []
  vendor_hierarchies:
    Paper Plates: [Dixie, Hefty, Solo]
    Plastic Plates: [Dart, Fabrikal]
  vb_brand_name: Victoria Bay
  top_n: 30

feedback:
  accept_label: "Accept"
  reject_label: "Reject"

optimization:
  max_suppliers: 150
  extra_vendor_exclusions: []
```

Then:

```powershell
imperial-dade list-categories        # plates should appear
imperial-dade render-prompt -c plates   # sanity-check the rendered prompt
imperial-dade run --category plates
```

All category-specific values — matching attributes, vendor hierarchies,
exclusion rules, `top_n`, optimization limits — live in this YAML and are
validated by `CategoryConfig` (Pydantic) at load time.

---

## Cluster descriptions (Stage 1)

Imperial Dade stocks the same physical product under many branch companies, each
with its own hand-typed description, and any single one is usually missing
something the others carry.
Stage 1 can therefore read the whole Sancus cluster for an item instead of just
that item's own description.

```
PRIMARY DESCRIPTION (this item): CUP COLD CLR PLA STOCK PRINT 16 OZ ECO GREENWARE
ADDITIONAL DESCRIPTIONS OF THE SAME PRODUCT (3):
  - cup cold clr 16 oz eco pla unprinted greenware 98mm 9509106
  - district gelato gc16s ptd 16 oz clr cup 1000/cs 109453501
  - cup pla 16/18 oz sqt clear stock print 1m 950922601
```

The siblings above contribute the 98mm diameter, the 1000/cs case pack and the
squat form factor, none of which appear in the item's own description.

Enable it per category in the YAML:

```yaml
taxonomy:
  use_cluster_descriptions: true
  max_cluster_descriptions: 8       # cap per item, including its own
  cluster_similarity_threshold: 0.30
  cluster_attribute_hints: [color, material, volume, diameter, case_pack]
```

Four things to know:

- **The join key is `entity_item_code`, not `item_code`.** Sancus stores
  `<entity>_<item_code>` using the same entity numbering we do, lowercased, so
  `1--12HDQW` maps to `1_12hdqw` (`fabric.to_sancus_key`). It is unique per row,
  making this an exact 1:1 match. Never fall back to bare `item_code`: the same
  code under another branch company is often a different product — matching that
  way pulled "upsg freight charge" and "labor to repair tool" into the
  attributes of cups item `CC`.
- **The endpoint's collation is case-sensitive.** `entity_item_code` is stored
  lowercased, and an uppercase lookup returns zero rows *without erroring*.
  Compare with `LOWER()` on both sides.
- **Clusters can mix variants.** One real cups cluster groups a translucent and a
  black portion cup, so extraction runs per item (never once per cluster) and the
  prompt instructs the model that the primary description wins on conflict. The
  similarity threshold is a second line of defence against junk cluster members.
- **Cache the read.** `FabricLoader.get_item_cluster_table()` is a single bulk
  scan; the lakehouse endpoint handles that far better than many `IN` lookups,
  which degrade badly and can throttle the workspace. `run_cups_full.py` caches
  it alongside the other source pulls.

Measured on the 4,568-item cups universe: 99.7% of items resolve to a cluster,
30% gain at least one extra description (2,421 extra descriptions in total), and
prompt tokens grow ~33% (projecting roughly $11 → $15 for a full Stage 1 run).

---

## Repository layout

```
src/imperial_dade/
├── categories/           # YAML configs + Pydantic schema
│   ├── base.py           # CategoryConfig
│   ├── cups.yaml
│   └── cutlery.yaml
├── config/               # Column names + env-driven settings
├── llm/
│   ├── azure_config.py   # Reads env, exposes Azure creds
│   ├── client.py         # OpenAIAgent + batching
│   └── prompts/          # Jinja2 templates — NO category names hardcoded
├── stages/
│   ├── taxonomy_load.py
│   ├── taxonomy_classify.py
│   ├── clusters.py         # Sancus cluster -> sibling descriptions for Stage 1
│   ├── matching.py
│   ├── feedback.py
│   ├── optimization.py
│   ├── report.py
│   └── exclusions.py
├── utils/validation.py   # Single source of ValidationError
└── cli.py                # imperial-dade entry point

notebooks/
└── pipeline_template.ipynb   # ONE notebook, parameterized

tests/
├── conftest.py           # 100-row fixture, env stubs
└── test_smoke.py         # Category YAML + prompt-rendering smoke tests

_legacy/                  # (post Phase 7) Original notebooks + Function_Files
```

---

## Running tests

```powershell
pytest                    # all
pytest tests/test_smoke.py -v
```

The smoke tests confirm:
- Every category YAML loads and validates.
- The rendered matching prompt mentions every critical attribute in its YAML.
- Cups-specific rules (Perfect Touch) survive in the cups render.
- Cutlery does not leak cups-specific content.
- The Taxonomy stage's `group_data` runs end-to-end on a 100-row fixture.
- `ValidationError` is a single class — every stage imports the canonical one.

---

## Migration notes

This package is the rewrite of the prior `Function_Files/` + `OpenAI Code/` +
`Pipeline Notebooks/Categories/{Cups,Cutlery}/` layout. The originals were
moved to `_legacy/` (kept for reference). Highlights:

- Secrets removed from source — `AZURE_OPENAI_API_KEY` now env-only.
- `pyvent.py` → `imperial_dade.llm.azure_config`.
- `openai_api.py` → `imperial_dade.llm.client`.
- 7 duplicate `ValidationError` definitions → 1 in `utils/validation.py`.
- 229 `print()` calls → standard `logging` (warning/error keywords detected
  and upgraded automatically).
- 90-line hardcoded Cups matching prompt → 1 Jinja template driven by YAML.

If you previously ran `Pipeline Notebooks/Categories/Cups/01_Taxonomy.ipynb`,
the equivalent is now:

```powershell
imperial-dade run --category cups --stage taxonomy
```
