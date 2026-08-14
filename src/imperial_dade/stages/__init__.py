"""Pipeline stages.

Stage order:
    1. taxonomy_load    — load + group transaction / item-master data
    2. taxonomy_classify — LLM-driven attribute extraction
    3. matching          — generate substitution candidates
    4. feedback          — incorporate human feedback into matches
    5. optimization      — MILP-based vendor consolidation
    6. report            — final Excel deliverable
    7. exclusions        — shared customer / vendor exclusion helpers
"""
