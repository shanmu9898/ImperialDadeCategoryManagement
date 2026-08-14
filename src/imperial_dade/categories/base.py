"""Schema + loader for per-category configuration.

A `CategoryConfig` is loaded from a YAML file next to this module
(e.g. `cups.yaml`, `cutlery.yaml`). The model is frozen — config is immutable
during a pipeline run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

_CATEGORY_DIR = Path(__file__).parent


class ExclusionRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str


class MatchingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    critical_attributes: list[str] = Field(
        ..., description="Attributes that MUST match exactly for a valid substitution."
    )
    directional_attributes: list[str] = Field(
        default_factory=list,
        description="Attributes that influence ranking but don't disqualify.",
    )
    exclusion_rules: list[ExclusionRule] = Field(
        default_factory=list,
        description="Category-specific absolute exclusions (e.g. 'Perfect Touch').",
    )
    vendor_hierarchies: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Preferred vendors per subcategory, ordered most-to-least preferred.",
    )
    vb_brand_name: str = Field(
        default="Victoria Bay",
        description="Brand whose VB Flag drives the same-brand exclusion rule.",
    )
    top_n: int = Field(default=20, description="Top-N matches written per item.")


class AttributeRule(BaseModel):
    """Category-specific normalization guidance for a single attribute.

    These rules are injected into the Stage-1 attribute-extraction prompt so the
    LLM canonicalizes synonyms/abbreviations the same way a deterministic
    dictionary extractor would, without adding any new output columns. Every
    field is optional — supply only what's relevant for the attribute.
    """

    model_config = ConfigDict(frozen=True)

    synonyms: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Canonical value -> list of variants/abbreviations that map to it.",
    )
    values: list[str] = Field(
        default_factory=list,
        description="Closed list of allowed values for this attribute.",
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Human-readable format rule (e.g. 'a number followed by oz').",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Free-form disambiguation guidance for the LLM.",
    )


class TaxonomyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    columns_for_description: Optional[list[str]] = Field(
        default=None,
        description=(
            "Salsify attribute columns to enrich on top of transactions. "
            "If unset, the taxonomy stage falls back to whichever columns "
            "hit the coverage threshold automatically."
        ),
    )
    coverage_threshold: float = Field(
        default=15.0,
        description="Minimum percent non-null required to keep a Salsify column.",
    )
    default_pack_size: int = Field(
        default=1000,
        description="Fallback Case Pack value when LLM extraction returns N/A.",
    )
    extraction_rules: dict[str, AttributeRule] = Field(
        default_factory=dict,
        description=(
            "Per-attribute normalization rules (keyed by an attribute name in "
            "columns_for_description). Rendered into the extraction prompt. "
            "Leave empty to use the category-agnostic LLM extraction unchanged."
        ),
    )
    use_cluster_descriptions: bool = Field(
        default=False,
        description=(
            "Read Sancus item clusters and feed every description of the same "
            "physical product into the extraction prompt. Off by default so "
            "categories that haven't been validated against Sancus are unchanged."
        ),
    )
    max_cluster_descriptions: int = Field(
        default=8,
        ge=1,
        description=(
            "Total descriptions shown per item, including its own. Bounds prompt "
            "size — clusters can run to 36+ members. Siblings are kept "
            "longest-first, since longer descriptions carry more attributes."
        ),
    )
    cluster_attribute_hints: list[str] = Field(
        default_factory=list,
        description=(
            "Sancus attribute columns (color, material, volume, diameter, ...) "
            "passed to the prompt as hints for the LLM to confirm or override. "
            "Empty disables hints."
        ),
    )
    cluster_similarity_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum description overlap before a cluster (or an individual "
            "sibling within it) is trusted. Guards against short, generic item "
            "codes colliding across branch companies — raise it if off-topic "
            "descriptions still leak in, lower it to recover more siblings."
        ),
    )


class FeedbackConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    accept_label: str = "Accept"
    reject_label: str = "Reject"
    use_subcategory_rules: bool = False


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_suppliers: int
    extra_vendor_exclusions: list[str] = Field(default_factory=list)


class CategoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    data_dir: str = Field(..., description="Path under the configured data root.")
    sales_file: str = Field(..., description="Sales-history filename inside data_dir.")
    item_master_file: Optional[str] = Field(
        default=None, description="Item-master filename inside data_dir (if separate)."
    )

    matching: MatchingConfig
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    optimization: OptimizationConfig


def load_category(name: str) -> CategoryConfig:
    """Load a category by its YAML filename stem (e.g. 'cups', 'cutlery')."""
    path = _CATEGORY_DIR / f"{name.lower()}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in _CATEGORY_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"Category config {path.name!r} not found. "
            f"Available: {available or '(none)'}."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return CategoryConfig.model_validate(data)
