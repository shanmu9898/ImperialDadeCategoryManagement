# _legacy

Pre-rewrite codebase, preserved for reference. **Do not run anything from
here in production** — the rewritten package under `src/imperial_dade/` is
the source of truth.

Contents:
- `Function_Files/` — original stage files (now under `src/imperial_dade/stages/`)
- `OpenAI Code/`    — original Azure OpenAI client + pyvent.py
- `Pipeline Notebooks/` — original per-category notebooks (now `notebooks/pipeline_template.ipynb`)

Maps from old paths to new:
- `Function_Files/config.py`                  -> `src/imperial_dade/config/pipeline.py`
- `Function_Files/Load_Isolate_functions.py`  -> `src/imperial_dade/stages/taxonomy_load.py`
- `Function_Files/Classification_functions.py`-> `src/imperial_dade/stages/taxonomy_classify.py`
- `Function_Files/Matching_functions.py`      -> `src/imperial_dade/stages/matching.py`
- `Function_Files/Feedback_functions.py`      -> `src/imperial_dade/stages/feedback.py`
- `Function_Files/Optimization_functions.py`  -> `src/imperial_dade/stages/optimization.py`
- `Function_Files/Final_report_functions.py`  -> `src/imperial_dade/stages/report.py`
- `Function_Files/Exclusion_functions.py`     -> `src/imperial_dade/stages/exclusions.py`
- `OpenAI Code/openai_api.py`                 -> `src/imperial_dade/llm/client.py`
- `OpenAI Code/pyvent.py`                     -> `src/imperial_dade/llm/azure_config.py`

When confident the new pipeline produces equivalent outputs, this directory
can be removed in one commit.
