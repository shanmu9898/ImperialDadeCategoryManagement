"""Smoke tests — confirm the new category-agnostic substrate is wired up.

Each category YAML must:
  1. Load and validate against the CategoryConfig schema.
  2. Render the matching system prompt without error.
  3. Produce a prompt that mentions every critical attribute defined in its YAML.

Plus one Taxonomy-stage check: `group_data` must run on a 100-row fixture.
"""
from __future__ import annotations

import pytest

from imperial_dade.categories import CategoryConfig, load_category
from imperial_dade.llm.prompts import render_prompt


# ---------------------------------------------------------------------------
# CategoryConfig + Jinja rendering
# ---------------------------------------------------------------------------


def test_at_least_one_category_yaml_exists(available_categories: list[str]) -> None:
    assert available_categories, "no category YAMLs found under src/imperial_dade/categories/"


def test_every_category_yaml_loads(available_categories: list[str]) -> None:
    for name in available_categories:
        cfg = load_category(name)
        assert isinstance(cfg, CategoryConfig)
        assert cfg.name
        assert cfg.matching.critical_attributes, f"{name} has no critical attributes"
        assert cfg.optimization.max_suppliers > 0


@pytest.mark.parametrize("category_name", ["cups", "cutlery"])
def test_matching_prompt_renders_with_category_attributes(category_name: str) -> None:
    """The rendered system prompt must mention every critical attribute from the YAML.

    This is the contract that makes the pipeline category-agnostic — changing a
    category's YAML changes the LLM's system prompt without any code changes.
    """
    cfg = load_category(category_name)
    rendered = render_prompt("matching_system.j2", category=cfg, hard_rules=None)

    for attr in cfg.matching.critical_attributes:
        assert attr in rendered, (
            f"{category_name}: critical attribute {attr!r} missing from rendered prompt"
        )


def test_cups_prompt_contains_perfect_touch_exclusion() -> None:
    """Cups-specific behavior survives the refactor."""
    cfg = load_category("cups")
    rendered = render_prompt("matching_system.j2", category=cfg)
    assert "Perfect Touch" in rendered, "Perfect Touch exclusion rule lost in cups.yaml"


def test_cutlery_prompt_does_not_leak_cups_attributes() -> None:
    """The cutlery rendering must NOT reference cup-specific attributes."""
    cfg = load_category("cutlery")
    rendered = render_prompt("matching_system.j2", category=cfg)
    for forbidden in ("Beverage Cup", "Perfect Touch", "Plastic Cold Cup"):
        assert forbidden not in rendered, (
            f"cutlery prompt leaks cups-specific content: {forbidden!r}"
        )


def test_top_n_differs_by_category() -> None:
    cups = load_category("cups")
    cutlery = load_category("cutlery")
    assert cups.matching.top_n != cutlery.matching.top_n, (
        "cups and cutlery should have different top_n values; otherwise the YAML differentiation isn't working"
    )


def test_unknown_category_raises_useful_error() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_category("doesnotexist")


# ---------------------------------------------------------------------------
# Stage 1 — extraction_rules render into the attribute-extraction prompt
# ---------------------------------------------------------------------------


def _render_attr_prompt(cfg):
    return render_prompt(
        "attribute_extraction.j2",
        attributes=cfg.taxonomy.columns_for_description or [],
        prompt_options_string="",
        example_desc="d",
        example_output="o",
        extraction_rules={
            a: r.model_dump() for a, r in cfg.taxonomy.extraction_rules.items()
        },
    )


def test_cups_extraction_rules_render_into_attribute_prompt() -> None:
    """Cups normalization rules (ported from the dict extractor) reach the LLM prompt."""
    cfg = load_category("cups")
    assert cfg.taxonomy.extraction_rules, "cups.yaml lost its taxonomy.extraction_rules"
    rendered = _render_attr_prompt(cfg)

    assert "Apply these category-specific normalization rules" in rendered
    for needle in (
        "Polypropylene (PP)",  # Material synonym canonical value
        "Cup Lid Dome",        # ordered keyword-subset type
        "MOST specific",       # disambiguation note (type precedence)
        "strawless",           # slot/sip conflict note
    ):
        assert needle in rendered, f"extraction prompt missing rule text: {needle!r}"


def test_category_without_rules_has_no_normalization_block() -> None:
    """A category with no extraction_rules renders the original prompt unchanged."""
    cfg = load_category("cutlery")
    assert not cfg.taxonomy.extraction_rules
    rendered = _render_attr_prompt(cfg)
    assert "Apply these category-specific normalization rules" not in rendered


# ---------------------------------------------------------------------------
# Stage 1 — Sancus cluster descriptions
# ---------------------------------------------------------------------------


def _cluster_frames():
    """A 3-member cluster where each description carries something the others lack.

    Mirrors the real 12HDQW cluster: one branch records the coating, another the
    brand and case pack, a third neither.
    """
    import pandas as pd

    cat_data = pd.DataFrame(
        {
            "Entity--Item": ["1--12HDQW", "1--NOCLUSTER"],
            "Combined Descriptions": [
                "12oz paper hot cup coat. white 12hdqw",
                "some unmatched item",
            ],
        }
    )
    cluster_ids = pd.DataFrame(
        {
            "s2k_item_code": ["12HDQW"],
            "cluster_id": ["6306599"],
            "entity_id": [1],
            "entity_item_code": ["1_12hdqw"],
        }
    )
    cluster_members = pd.DataFrame(
        {
            "cluster_id": ["6306599"] * 4,
            "entity_id": [1, 46, 14, 58],
            "s2k_item_code": ["12HDQW", "12HDQW", "12HDQWC", "12HDQWD"],
            "entity_item_code": ["1_12hdqw", "46_12hdqw", "14_12hdqwc", "58_12hdqwd"],
            "cluster_description": [
                "12oz paper hot cup coat. white 12hdqw",
                "victoria bay 12 oz paper hot cup white 1000/ca vbclhp12w",
                "12oz hot cup paper 1000cs",
                "DO NOT USE 12oz cup",
            ],
            "vb_flag": ["Y - VB", "Y - VB", "N", "N"],
            "vgn": ["Imperial Dade"] * 4,
            "color": ["white", "white", "", ""],
            "volume": ["12 oz", "12 oz", "", ""],
        }
    )
    return cat_data, cluster_ids, cluster_members


def test_entity_item_translates_to_sancus_key() -> None:
    """'1--12HDQW' -> '1_12hdqw'.

    Sancus stores entity_item_code as '<entity>_<item_code>' using the SAME
    entity numbering we do, lowercased. Lowercasing is load-bearing: the
    lakehouse endpoint's collation is case-sensitive, so an uppercase key
    matches zero rows without raising — which is exactly how this join was
    first mis-diagnosed.
    """
    from imperial_dade.io.fabric import to_sancus_key

    assert to_sancus_key("1--12HDQW") == "1_12hdqw"
    assert to_sancus_key("1--CC") == "1_cc"
    assert to_sancus_key("2--VBCLHP12W") == "2_vbclhp12w"
    assert to_sancus_key("  1--325PC  ") == "1_325pc"
    # Malformed keys must not silently produce a lookup value.
    for bad in ("", "1--", "--ABC", "no-separator", None):
        assert to_sancus_key(bad) is None, f"{bad!r} should not yield a key"


def test_sibling_from_another_branch_with_same_code_is_kept() -> None:
    """The same item_code under a different entity is a sibling, not the item.

    Only the item's OWN row (exact entity_item_code) is excluded from the
    sibling list. Other branches' rows for the same code are the whole point —
    they describe the same product in different words.
    """
    import pandas as pd

    from imperial_dade.stages.clusters import (
        CLUSTER_DESCRIPTIONS_COL,
        attach_cluster_descriptions,
    )

    cat_data = pd.DataFrame(
        {
            "Entity--Item": ["1--20HDCCF"],
            "Combined Descriptions": ["vb cup hot paper 20 oz serenity 1000/cs 20hdccf"],
        }
    )
    ids = pd.DataFrame(
        {
            "s2k_item_code": ["20HDCCF"],
            "cluster_id": ["6308005"],
            "entity_item_code": ["1_20hdccf"],
        }
    )
    members = pd.DataFrame(
        {
            "cluster_id": ["6308005"] * 3,
            "s2k_item_code": ["20HDCCF", "20HDCCF", "20HDCCF"],
            # Same item_code, three different branch companies.
            "entity_item_code": ["1_20hdccf", "46_20hdccf", "12_20hdccf"],
            "cluster_description": [
                "vb cup hot paper 20 oz serenity 1000/cs 20hdccf",
                "20oz vb serenity paper hot cups 1m/cs temporary substitute",
                "20hdccf 20 oz vb coffee design hot paper cup 1m/cs",
            ],
        }
    )

    block = attach_cluster_descriptions(cat_data, ids, members).loc[
        0, CLUSTER_DESCRIPTIONS_COL
    ]
    assert "1m/cs" in block, "same-code sibling from another branch was dropped"
    assert block.count("\n  - ") == 2, f"expected 2 siblings, got:\n{block}"
    # The item's own row must appear once, as the primary, not repeated below.
    assert block.count("vb cup hot paper 20 oz serenity") == 1


def test_missing_own_description_rejects_every_cluster() -> None:
    """Regression: attaching clusters before 'Combined Descriptions' exists.

    Siblings are scored against the item's own description. If that column is
    absent the score is 0.0 for every row and all clusters are rejected —
    silently, with the run still 'succeeding'. This happened on a real run and
    cost a full Stage 1 pass. build_combined_descriptions must come first.
    """
    import pandas as pd

    from imperial_dade.stages.clusters import (
        CLUSTER_DESCRIPTIONS_COL,
        attach_cluster_descriptions,
    )
    from imperial_dade.stages.taxonomy_classify import build_combined_descriptions

    cat_data, ids, members = _cluster_frames()
    without = cat_data.drop(columns=["Combined Descriptions"])

    out = attach_cluster_descriptions(without, ids, members)
    assert (out[CLUSTER_DESCRIPTIONS_COL] == "").all(), (
        "expected total rejection without the description column — if this now "
        "passes, the guard changed and the ordering trap may be masked"
    )

    # And with the column built the same input yields siblings.
    rebuilt = build_combined_descriptions(
        without.assign(**{"Item Desc 1": cat_data["Combined Descriptions"],
                          "Item Desc 2": ""})
    )
    out2 = attach_cluster_descriptions(rebuilt, ids, members)
    assert out2.loc[0, CLUSTER_DESCRIPTIONS_COL] != "", "siblings lost after rebuild"


def test_build_combined_descriptions_is_idempotent() -> None:
    """Calling it twice (runner, then run()) must not change the result."""
    import pandas as pd

    from imperial_dade.stages.taxonomy_classify import build_combined_descriptions

    df = pd.DataFrame({"Item Desc 1": ["12OZ CUP"], "Item Desc 2": ["WHITE PAPER"]})
    once = build_combined_descriptions(df)
    twice = build_combined_descriptions(once)
    assert once["Combined Descriptions"].tolist() == ["12OZ CUP WHITE PAPER"]
    assert twice["Combined Descriptions"].tolist() == once["Combined Descriptions"].tolist()


def test_cluster_descriptions_union_sibling_wording() -> None:
    """Sibling descriptions are attached, primary first, noise rows dropped."""
    from imperial_dade.stages.clusters import (
        CLUSTER_DESCRIPTIONS_COL,
        CLUSTER_HINTS_COL,
        CLUSTER_ID_COL,
        attach_cluster_descriptions,
    )

    cat_data, ids, members = _cluster_frames()
    out = attach_cluster_descriptions(
        cat_data, ids, members, attribute_columns=["color", "volume"]
    )

    block = out.loc[0, CLUSTER_DESCRIPTIONS_COL]
    assert block.startswith("PRIMARY DESCRIPTION (this item): 12oz paper hot cup coat.")
    # The brand + case pack only exist on a sibling — that is the point.
    assert "victoria bay" in block
    assert "1000/ca" in block
    # 'DO NOT USE' rows carry no product information and must be filtered out.
    assert "DO NOT USE" not in block.upper()

    assert out.loc[0, CLUSTER_ID_COL] == "6306599"
    assert "color=white" in out.loc[0, CLUSTER_HINTS_COL]

    # An item with no cluster match stays empty so the caller falls back.
    assert out.loc[1, CLUSTER_DESCRIPTIONS_COL] == ""
    assert out.loc[1, CLUSTER_ID_COL] == ""


def test_off_topic_cluster_is_rejected() -> None:
    """Regression: a generic item code must not import an unrelated cluster.

    Real case from the cups universe — item '1--CC' matched clusters whose other
    members were freight charges and repair labor, because short codes collide
    across branch companies. Feeding those in as "the same product" is worse than
    using no cluster at all.
    """
    import pandas as pd

    from imperial_dade.stages.clusters import (
        CLUSTER_DESCRIPTIONS_COL,
        CLUSTER_ID_COL,
        attach_cluster_descriptions,
    )

    cat_data = pd.DataFrame(
        {
            "Entity--Item": ["1--CC"],
            "Combined Descriptions": ["VB SLEEVE COFFEE CUP HOT KFT 92MM SERIES 12/100"],
        }
    )
    ids = pd.DataFrame(
        {
            "s2k_item_code": ["CC"],
            "cluster_id": ["2145269"],
            "entity_item_code": ["1_cc"],
        }
    )
    members = pd.DataFrame(
        {
            "cluster_id": ["2145269"] * 3,
            "s2k_item_code": ["CC", "FREIGHTCHA", "REPAIRLABO"],
            "entity_item_code": ["1_cc", "1_freightcha", "1_repairlabo"],
            "cluster_description": [
                "upsg freight charge freightcha",
                "labor to repair tool repairlabo",
                "drop ship freight at no charge to customer",
            ],
        }
    )

    out = attach_cluster_descriptions(cat_data, ids, members)
    assert out.loc[0, CLUSTER_DESCRIPTIONS_COL] == ""
    assert out.loc[0, CLUSTER_ID_COL] == ""


def test_best_matching_cluster_wins_over_union() -> None:
    """An item in two clusters takes the better match, never the union."""
    import pandas as pd

    from imperial_dade.stages.clusters import (
        CLUSTER_DESCRIPTIONS_COL,
        CLUSTER_ID_COL,
        attach_cluster_descriptions,
    )

    cat_data = pd.DataFrame(
        {
            "Entity--Item": ["1--CC"],
            "Combined Descriptions": ["coffee cup sleeve hot kraft 92mm"],
        }
    )
    ids = pd.DataFrame(
        {
            "s2k_item_code": ["CC", "CC"],
            "cluster_id": ["good", "bad"],
            "entity_item_code": ["1_cc", "1_cc"],
        }
    )
    members = pd.DataFrame(
        {
            "cluster_id": ["good", "good", "bad"],
            "s2k_item_code": ["CC", "CCB", "CCX"],
            "entity_item_code": ["1_cc", "46_ccb", "1_ccx"],
            "cluster_description": [
                "coffee cup sleeve hot kraft 92mm",
                "sleeve coffee cup hot kraft compostable 1200/cs",
                "cycle counter unrelated device",
            ],
        }
    )

    out = attach_cluster_descriptions(cat_data, ids, members)
    assert out.loc[0, CLUSTER_ID_COL] == "good"
    block = out.loc[0, CLUSTER_DESCRIPTIONS_COL]
    assert "1200/cs" in block, "lost the genuine sibling"
    assert "cycle counter" not in block, "unioned in the wrong cluster"


def test_cluster_description_cap_keeps_longest_siblings() -> None:
    """max_descriptions bounds prompt size, retaining the most attribute-dense text."""
    from imperial_dade.stages.clusters import build_description_block

    block = build_description_block(
        "short primary",
        ["tiny", "a much longer and more detailed sibling description", "mid length one"],
        max_descriptions=3,
    )
    assert block.startswith("PRIMARY DESCRIPTION (this item): short primary")
    assert "a much longer and more detailed sibling description" in block
    assert "mid length one" in block
    assert "tiny" not in block.split("ADDITIONAL")[1]


def test_single_description_item_gets_no_cluster_block() -> None:
    """A cluster of one adds nothing, so the prompt stays in the original format."""
    from imperial_dade.stages.clusters import build_description_block

    assert build_description_block("only one", []) == ""
    # Case-insensitive duplicates of the item's own description are not siblings.
    assert build_description_block("Only One", ["only one", "ONLY ONE"]) == ""


def test_cluster_prompt_block_renders_only_when_enabled() -> None:
    """The multi-description instructions appear iff cluster_descriptions is set."""
    kwargs = dict(
        attributes=["Color"],
        prompt_options_string="",
        example_desc="d",
        example_output="o",
        extraction_rules={},
    )
    on = render_prompt("attribute_extraction.j2", cluster_descriptions=True, **kwargs)
    off = render_prompt("attribute_extraction.j2", cluster_descriptions=False, **kwargs)

    for needle in ("ADDITIONAL DESCRIPTIONS OF THE SAME PRODUCT", "UNION", "PRIMARY DESCRIPTION wins"):
        assert needle in on, f"cluster prompt missing: {needle!r}"
    assert "ADDITIONAL DESCRIPTIONS OF THE SAME PRODUCT" not in off
    assert "Never invent" not in off


def test_cups_enables_clusters_and_cutlery_does_not() -> None:
    """Clustering is opt-in per category, so unvalidated categories are unchanged."""
    cups = load_category("cups")
    cutlery = load_category("cutlery")
    assert cups.taxonomy.use_cluster_descriptions is True
    assert cups.taxonomy.cluster_attribute_hints, "cups lost its cluster hint columns"
    assert cutlery.taxonomy.use_cluster_descriptions is False
    assert cutlery.taxonomy.cluster_attribute_hints == []


def test_create_description_string_unchanged_without_cluster_data() -> None:
    """The single-description format is byte-for-byte preserved."""
    import pandas as pd

    from imperial_dade.stages.taxonomy_classify import create_description_string

    row = pd.Series({"Color": "White", "Combined Descriptions": "12oz cup"})
    plain = create_description_string(row, ["Color"], use_cluster_descriptions=False)
    assert plain == "Color: White, 12oz cup"
    # Enabled but with no block on the row -> identical output.
    assert create_description_string(row, ["Color"], use_cluster_descriptions=True) == plain


# ---------------------------------------------------------------------------
# Stage 2/3 — single Victoria Bay substitute in write_top_matches
# ---------------------------------------------------------------------------


def test_write_top_matches_emits_single_vb_substitute(tmp_path) -> None:
    """A non-VB target keeps its non-VB subs but at most ONE 'Y - VB' substitute,
    flagged in the Recommendation column. (Shared by Stage 2 and Stage 3.)"""
    import pandas as pd

    from imperial_dade.stages.matching import write_top_matches

    rows = [
        # target (non-VB) ranks two VB candidates + one non-VB + one Y-Other
        ("1--T", "N", ["1--S1", "1--VB1", "1--VB2", "1--O1"]),
        ("1--S1", "N", []),
        ("1--VB1", "Y - VB", []),
        ("1--VB2", "Y - VB", []),
        ("1--O1", "Y - Other", []),
    ]
    df = pd.DataFrame({
        "Entity--Item": [r[0] for r in rows],
        "VB Flag": [r[1] for r in rows],
        "Matches": [r[2] for r in rows],
        "Combined Descriptions": ["12oz paper hot cup" for _ in rows],
        "Description with Attributes": ["12oz paper hot cup" for _ in rows],
        "attributes": ["Material: Paper" for _ in rows],
        "reasoning": ["" for _ in rows],
        "VGN": ["V" for _ in rows],
        "Qty": [100, 1, 1, 1, 1],
        "Net Cost": [10.0, 1.0, 1.0, 1.0, 1.0],
    })

    out = tmp_path / "subs.xlsx"
    write_top_matches(df, str(out), n=10, pl=True, enable_vpn_exclusion=False)
    assert out.exists(), "write_top_matches produced no workbook"

    xls = pd.ExcelFile(out)
    assert "1--T" in xls.sheet_names, "non-VB target sheet missing"

    sheet = pd.read_excel(xls, sheet_name="1--T", header=0)
    assert "Recommendation" in sheet.columns

    data_rows = []
    for _, r in sheet.iterrows():
        v = r.get("Entity--Item")
        if pd.isna(v) or (isinstance(v, str) and not v.strip()):
            break
        data_rows.append(r)

    recs = [str(r["Recommendation"]) for r in data_rows]
    assert recs[0] == "Target"
    assert recs.count("VB Substitute") == 1, f"expected exactly one VB substitute, got {recs}"
    # The two non-VB subs (N + Y-Other) survive as regular substitutes.
    assert recs.count("Substitute") == 2, f"expected two regular substitutes, got {recs}"


# ---------------------------------------------------------------------------
# Taxonomy stage on a 100-row fixture
# ---------------------------------------------------------------------------


def test_taxonomy_group_data_runs_on_fixture(taxonomy_fixture) -> None:
    """`group_data` must aggregate the 100-row fixture without error."""
    from imperial_dade.stages.taxonomy_load import group_data

    grouped = group_data(taxonomy_fixture)
    assert len(grouped) <= len(taxonomy_fixture), (
        "grouped output cannot have more rows than input"
    )
    assert "Entity--Item" in grouped.columns


# ---------------------------------------------------------------------------
# Validation utility consolidation
# ---------------------------------------------------------------------------


def test_validation_error_is_single_source() -> None:
    """Every stage that uses ValidationError must import the canonical one."""
    from imperial_dade.utils.validation import ValidationError as Canonical

    from imperial_dade.stages import (
        exclusions,
        feedback,
        matching,
        optimization,
        report,
        taxonomy_classify,
        taxonomy_load,
    )

    for mod in (exclusions, feedback, matching, optimization, report, taxonomy_classify, taxonomy_load):
        assert getattr(mod, "ValidationError", None) is Canonical, (
            f"{mod.__name__} uses a stale ValidationError class"
        )
