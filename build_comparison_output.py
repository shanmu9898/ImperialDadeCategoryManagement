r"""Build the primary<->substitute comparison workbook from a matches CSV.

Reads a Stage 2 / Stage 3 matches file (default Cups_matches.csv) and emits a
workbook with two DISJOINT sheets, both sharing the same primary_/substitute_
schema. Every non-VB primary appears on exactly one of them:

  Primary_vs_VB   primaries (VB Flag == 'N') that HAVE a Victoria Bay ('Y - VB')
                  substitute -> one row each, against the single best VB sub.
  Primary_no_VB   primaries with NO Victoria Bay substitute at all -> one row per
                  non-VB alternative, so you can see what they could move to.
                  Primaries with no substitutes whatsoever still get a row, with
                  the substitute_ columns left blank, so nothing is dropped.

NOTE the change in meaning. Before 2026-08, sheet 2 was named Primary_vs_NonVB
and held EVERY non-VB pairing, so a primary that already had a VB substitute was
listed on both sheets (1,004 of 1,051 in the 2026-06-17 workbook). The sheet is
renamed alongside the semantic change so old and new files can't be confused.
Pass --overlapping-nonvb to reproduce the old behaviour.

Only attributes actually present in the matches file are compared. Run:

    .\.venv312\Scripts\python.exe build_comparison_output.py
    .\.venv312\Scripts\python.exe build_comparison_output.py --category cups \
        --matches "Data/Cups/Output/Cups_matches.csv"
"""
from __future__ import annotations

import argparse
import ast
from datetime import date
from pathlib import Path

import pandas as pd
from imperial_dade.categories import load_category

# (output label, source column in the matches file). Order drives column order.
ATTRS: list[tuple[str, str]] = [
    ("Description", "Combined Descriptions"),
    ("VB Flag", "VB Flag"),
    ("VGN", "VGN"),
    ("Case Pack", "Case Pack"),
    ("Beverage Cup Style", "Beverage Cup Style"),
    ("Beverage Cup Type", "Beverage Cup Type"),
    ("Color", "Color"),
    ("Foodservice Global Attributes", "Foodservice Global Attributes"),
    ("Material", "Material"),
    ("Pattern & Design", "Pattern & Design"),
    ("Product Capacity", "Product Capacity"),
    ("Product Type Collapse", "Product Type Collapse"),
    ("Usage Temperature", "Usage Temperature"),
]

_CUPS_ATTRS = ATTRS.copy()


def configure_attrs(category: str, available: set[str] | None = None) -> None:
    """Use the legacy Cups layout, or derive the layout from category YAML."""
    global ATTRS, SUBS_PAIRS
    if category.casefold() == "cups":
        ATTRS = _CUPS_ATTRS.copy()
    else:
        cfg = load_category(category)
        names = ["Description", "VB Flag", "VGN", "VPN", "Case Pack"]
        names += cfg.matching.critical_attributes
        names += cfg.matching.directional_attributes
        seen: set[str] = set()
        ordered = [name for name in names if not (name in seen or seen.add(name))]
        source_map = {"Description": "Combined Descriptions"}
        ATTRS = [
            (name, source_map.get(name, name))
            for name in ordered
            if available is None or source_map.get(name, name) in available
        ]
    SUBS_PAIRS = [(label, label) for label, _ in ATTRS]

ITEM_COL = "Entity--Item"
VB_FLAG_COL = "VB Flag"
MATCHES_COL = "Matches"
VPN_COL = "VPN"
VB_VALUE = "Y - VB"


def _vpn_excluded(match_vpn, target_vpn: str, target_item_code: str) -> bool:
    """Mirror matching._should_exclude_vpn_match: drop a match whose VPN equals
    the target's VPN, or equals the target's item_code. The matches CSV has no
    item_code column, so target_item_code == '' (same as write_top_matches,
    where the column is injected empty)."""
    return (pd.notna(match_vpn) and match_vpn == target_vpn) or \
           (pd.notna(match_vpn) and match_vpn == target_item_code)


def column_order() -> list[str]:
    cols = ["primary_item", "substitute_item"]
    for label, _ in ATTRS:
        cols += [f"primary_{label}", f"substitute_{label}"]
    return cols


def parse_matches(raw) -> list[str]:
    """Matches is stored as a stringified list in the CSV (or already a list)."""
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text or text in ("[]", "nan", "None"):
        return []
    try:
        val = ast.literal_eval(text)
        if isinstance(val, (list, tuple)):
            return [str(x).strip() for x in val if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return []


def _cell(row: pd.Series, col: str):
    if col not in row.index:
        return ""
    val = row.get(col)
    return "" if pd.isna(val) else val


def make_pair(primary: pd.Series, sub: pd.Series) -> dict:
    rec = {
        "primary_item": _cell(primary, ITEM_COL),
        "substitute_item": _cell(sub, ITEM_COL),
    }
    for label, src in ATTRS:
        rec[f"primary_{label}"] = _cell(primary, src)
        rec[f"substitute_{label}"] = _cell(sub, src)
    return rec


def _blank_pair(primary: pd.Series, labels) -> dict:
    """A row for a primary that has no substitute at all — substitute side blank."""
    rec = {"primary_item": _cell(primary, ITEM_COL), "substitute_item": ""}
    for label, src in labels:
        rec[f"primary_{label}"] = _cell(primary, src)
        rec[f"substitute_{label}"] = ""
    return rec


def build(
    matches_path: Path,
    enable_vpn_exclusion: bool = False,
    overlapping_nonvb: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Read only what we need (skip the giant embeddings column on full runs).
    header = pd.read_csv(matches_path, nrows=0).columns.tolist()
    needed = {ITEM_COL, VB_FLAG_COL, MATCHES_COL, VPN_COL} | {src for _, src in ATTRS}
    usecols = [c for c in header if c in needed]
    df = pd.read_csv(matches_path, usecols=usecols)

    # Mirror write_top_matches' column cleaning so VPN comparisons match exactly.
    if enable_vpn_exclusion and VPN_COL in df.columns:
        df[VPN_COL] = df[VPN_COL].astype(str).str.strip()

    by_id = {str(r[ITEM_COL]): r for _, r in df.iterrows()}

    vb_rows: list[dict] = []
    nonvb_rows: list[dict] = []

    # Primaries are non-VB items only (matches the Stage 2 target convention).
    primaries = df[df[VB_FLAG_COL] == "N"]
    for _, primary in primaries.iterrows():
        match_ids = parse_matches(primary.get(MATCHES_COL))
        subs = [by_id[m] for m in match_ids if m in by_id]

        if enable_vpn_exclusion:
            target_vpn = str(primary.get(VPN_COL, "")).strip()
            target_item_code = ""  # no item_code column in the matches CSV
            subs = [s for s in subs
                    if not _vpn_excluded(s.get(VPN_COL), target_vpn, target_item_code)]

        vb_subs = [s for s in subs if str(s.get(VB_FLAG_COL)) == VB_VALUE]
        non_vb_subs = [s for s in subs if str(s.get(VB_FLAG_COL)) != VB_VALUE]

        if vb_subs:
            # Has a VB option: sheet 1, against the single best VB sub
            # (Matches is rank-ordered).
            vb_rows.append(make_pair(primary, vb_subs[0]))
            if overlapping_nonvb:         # legacy: also list every non-VB pair
                for s in non_vb_subs:
                    nonvb_rows.append(make_pair(primary, s))
            continue

        # No VB option at all: sheet 2, with whatever non-VB alternatives exist.
        if non_vb_subs:
            for s in non_vb_subs:
                nonvb_rows.append(make_pair(primary, s))
        else:
            nonvb_rows.append(_blank_pair(primary, ATTRS))

    cols = column_order()
    vb_df = pd.DataFrame(vb_rows, columns=cols).fillna("")
    nonvb_df = pd.DataFrame(nonvb_rows, columns=cols).fillna("")
    return vb_df, nonvb_df


# In the Cups_Subs review workbook the per-row columns already carry the same
# attribute labels (and a "Description" column), so the source column == label.
SUBS_IGNORE = {"Summary", "_VendorCalcHelper"}
SUBS_PAIRS = [(label, label) for label, _ in ATTRS]


def make_pair_subs(primary: pd.Series, sub: pd.Series) -> dict:
    rec = {
        "primary_item": _cell(primary, ITEM_COL),
        "substitute_item": _cell(sub, ITEM_COL),
    }
    for label, src in SUBS_PAIRS:
        rec[f"primary_{label}"] = _cell(primary, src)
        rec[f"substitute_{label}"] = _cell(sub, src)
    return rec


def build_from_subs(subs_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the comparison straight from the Cups_Subs review workbook, so it
    mirrors exactly the targets/substitutes in that file (top-N, VPN-excluded,
    single best VB). Each target sheet = one primary (the 'Target' row) + its
    substitute rows, split by the 'Recommendation' column."""
    xls = pd.ExcelFile(subs_path)
    vb_rows: list[dict] = []
    nonvb_rows: list[dict] = []

    for sheet in xls.sheet_names:
        if sheet in SUBS_IGNORE:
            continue
        df = pd.read_excel(xls, sheet_name=sheet, header=0)
        if "Entity--Item" not in df.columns or "Recommendation" not in df.columns:
            continue

        rows = []
        for _, r in df.iterrows():            # stop at the blank row before "Reasoning:"
            v = r.get(ITEM_COL)
            if pd.isna(v) or (isinstance(v, str) and not v.strip()):
                break
            rows.append(r)
        if len(rows) < 2:
            continue

        primary = rows[0]                     # the highlighted 'Target' row
        subs = rows[1:]
        vb_subs = [s for s in subs if str(s.get("Recommendation")) == "VB Substitute"]
        non_vb_subs = [s for s in subs if str(s.get("Recommendation")) != "VB Substitute"]

        # Same disjoint split as build(): a target with a VB substitute belongs
        # on sheet 1 only.
        if vb_subs:
            vb_rows.append(make_pair_subs(primary, vb_subs[0]))
            continue
        if non_vb_subs:
            for s in non_vb_subs:
                nonvb_rows.append(make_pair_subs(primary, s))
        else:
            nonvb_rows.append(_blank_pair(primary, SUBS_PAIRS))

    cols = column_order()
    vb_df = pd.DataFrame(vb_rows, columns=cols).fillna("")
    nonvb_df = pd.DataFrame(nonvb_rows, columns=cols).fillna("")
    return vb_df, nonvb_df


def write_workbook(
    vb_df: pd.DataFrame,
    nonvb_df: pd.DataFrame,
    out_path: Path,
    nonvb_sheet: str = "Primary_no_VB",
) -> None:
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        sheets = {"Primary_vs_VB": vb_df, nonvb_sheet: nonvb_df}
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            book = writer.book
            hdr = book.add_format({"bold": True, "bg_color": "#D9E1F2",
                                   "border": 1, "text_wrap": True, "valign": "top"})
            for c, col in enumerate(frame.columns):
                ws.write(0, c, col, hdr)
                width = 40 if col.endswith("Description") else 20
                ws.set_column(c, c, width)
            ws.freeze_panes(1, 2)
            if len(frame):
                ws.autofilter(0, 0, len(frame), len(frame.columns) - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Cups")
    ap.add_argument("--matches", default=None,
                    help="Path to the matches CSV (default: Data/<cat>/Output/<cat>_matches.csv)")
    ap.add_argument("--out", default=None, help="Output xlsx path (default alongside source)")
    ap.add_argument("--from-subs", default=None, dest="from_subs",
                    help="Build from a Cups_Subs review workbook (top-N, VPN-excluded) "
                         "instead of the full matches CSV.")
    ap.add_argument("--vpn-exclusion", action="store_true", dest="vpn_exclusion",
                    help="Apply the same VPN same-vendor exclusion the Stage 2 workbook "
                         "uses (matches-CSV mode only).")
    ap.add_argument("--overlapping-nonvb", action="store_true", dest="overlapping_nonvb",
                    help="Legacy pre-2026-08 behaviour: sheet 2 lists EVERY non-VB "
                         "pairing, so primaries that already have a VB substitute "
                         "appear on both sheets. Default is the disjoint split.")
    args = ap.parse_args()

    out_dir = Path("Data") / args.category / "Output"

    if args.from_subs:
        configure_attrs(args.category)
        subs_path = Path(args.from_subs)
        if not subs_path.exists():
            raise FileNotFoundError(f"Subs workbook not found: {subs_path}")
        source = subs_path
        out_path = Path(args.out) if args.out else (
            out_dir / f"{args.category}_Subs_Comparison_fromSubs_{date.today():%Y-%m-%d}.xlsx"
        )
        vb_df, nonvb_df = build_from_subs(subs_path)
    else:
        source = Path(args.matches) if args.matches else out_dir / f"{args.category}_matches.csv"
        if not source.exists():
            raise FileNotFoundError(f"Matches file not found: {source}")
        configure_attrs(args.category, set(pd.read_csv(source, nrows=0).columns))
        out_path = Path(args.out) if args.out else (
            out_dir / f"{args.category}_Subs_Comparison_{date.today():%Y-%m-%d}.xlsx"
        )
        vb_df, nonvb_df = build(
            source,
            enable_vpn_exclusion=args.vpn_exclusion,
            overlapping_nonvb=args.overlapping_nonvb,
        )

    nonvb_sheet = "Primary_vs_NonVB" if args.overlapping_nonvb else "Primary_no_VB"
    write_workbook(vb_df, nonvb_df, out_path, nonvb_sheet=nonvb_sheet)

    vb_p = set(vb_df["primary_item"]) if len(vb_df) else set()
    nv_p = set(nonvb_df["primary_item"]) if len(nonvb_df) else set()
    no_alt = int((nonvb_df["substitute_item"] == "").sum()) if len(nonvb_df) else 0

    print(f"Source : {source}")
    print(f"Output : {out_path.resolve()}")
    print(f"  Primary_vs_VB   : {len(vb_df):>5} rows  "
          f"({len(vb_p)} primaries WITH a VB substitute)")
    print(f"  {nonvb_sheet:<15} : {len(nonvb_df):>5} rows  "
          f"({len(nv_p)} primaries"
          f"{'' if args.overlapping_nonvb else ' WITHOUT any VB substitute'}"
          f"{f', {no_alt} of them with no substitute at all' if no_alt else ''})")
    overlap = vb_p & nv_p
    if overlap and not args.overlapping_nonvb:
        print(f"  WARNING: {len(overlap)} primaries appear on both sheets — "
              f"the split should be disjoint")
    elif not args.overlapping_nonvb:
        print(f"  sheets are disjoint; {len(vb_p) + len(nv_p)} non-VB primaries covered in total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
