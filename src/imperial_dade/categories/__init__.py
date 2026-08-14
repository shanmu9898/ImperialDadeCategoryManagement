"""Per-category configuration.

Add a new category by dropping a YAML file in this directory matching the
schema defined in `imperial_dade.categories.base.CategoryConfig`. No Python
changes are needed — the YAML drives the matching prompt, optimization
parameters, and file naming.
"""

from imperial_dade.categories.base import CategoryConfig, load_category

__all__ = ["CategoryConfig", "load_category"]
