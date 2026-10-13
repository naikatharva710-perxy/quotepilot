"""Regression tests for review-queue safety and empty-cell display. Offline: no model calls."""
import io
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import analyst, award, normalize, pipeline, review  # noqa: E402
from tests.live_shapes import build  # noqa: E402

AMBIGUOUS_8 = ["L23", "L24", "L25", "L26", "L27", "L28", "L29", "L30"]
CLEAR_PER_KG = ["L01", "L02", "L03", "L04", "L05", "L08", "L09", "L10", "L11", "L12", "L13", "L14", "L19"]
LAST_YEAR = ["L06", "L07", "L18", "L20"]


@pytest.fixture()
def data():
    rfx, ex = build()
    hist = pipeline.load_history()
    df = normalize.normalise(rfx, ex, hist, normalize.Assumptions())
    q = df[df.status == "REVIEW"].copy()
    return rfx, ex, hist, df, q


def per_kg_key(q):
    g = review.bulk_groups(q)
    return g[(g.vendor_id == "V5") & g.reason_key.str.startswith("Per-kg rate converted")].reason_key.iloc[0]


def chakan(df, lid):
    return df[(df.line_id == lid) & (df.vendor_id == "V5")].iloc[0]


# ---------------------------------------------------------------- 1. review queue safety

def test_all_30_chakan_lines_await_review(data):
    _, _, _, df, _ = data
    assert (df[df.vendor_id == "V5"].status == "REVIEW").all()


def test_ambiguous_eight_are_blocked_from_bulk(data):
    _, _, _, _, q = data
    q = review.annotate(q)
    for lid in AMBIGUOUS_8:
        r = q[(q.line_id == lid) & (q.vendor_id == "V5")].iloc[0]
        assert r.bulk_block, f"{lid} should need an individual decision"


def test_bulk_groups_never_list_ambiguous_lines(data):
    _, _, _, _, q = data
    g = review.bulk_groups(q)
    listed = [lid for row in g[g.vendor_id == "V5"].line_ids for lid in row]
    assert not set(listed) & set(AMBIGUOUS_8)
    assert set(CLEAR_PER_KG) <= set(listed)


def test_bulk_accept_skips_ambiguous_and_accepts_eligible(data):
    _, _, _, _, q = data
    decisions, logs = {}, []
    accepted, skipped = review.bulk_accept(q, "V5", per_kg_key(q), decisions, lambda a, d: logs.append((a, d)))
    assert sorted(accepted) == sorted(CLEAR_PER_KG)
    skipped_ids = {lid for lid, _ in skipped}
    assert set(AMBIGUOUS_8) <= skipped_ids
    assert not any((lid, "V5") in decisions for lid in AMBIGUOUS_8)
    # one log line per accepted cell, each with its source evidence
    assert len(logs) == len(CLEAR_PER_KG)
    assert all("source V5_file, line 8" in d for _, d in logs)
    assert all("estimated box weight" in d and "not supplied or confirmed" in d for _, d in logs)


def test_eligible_bulk_accept_confirms_cells_with_evidence_intact(data):
    rfx, ex, hist, _, q = data
    decisions = {}
    review.bulk_accept(q, "V5", per_kg_key(q), decisions)
    df2 = normalize.normalise(rfx, ex, hist, normalize.Assumptions(), decisions)
    for lid in CLEAR_PER_KG:
        r = chakan(df2, lid)
        assert r.status == "CONFIRMED" and r.usable
        assert r.source_quote and r.source_ref == "line 8"
    for lid in AMBIGUOUS_8:
        assert chakan(df2, lid).status == "REVIEW"


def test_data_layer_refuses_bulk_accept_of_ambiguous_cells(data):
    """Bypass attempt: a bulk-tagged decision written straight into state must not confirm the cell."""
    rfx, ex, hist, _, _ = data
    forged = {(lid, "V5"): {"action": "accept", "value": 1.0, "via": "bulk", "note": "forged"} for lid in AMBIGUOUS_8}
    df2 = normalize.normalise(rfx, ex, hist, normalize.Assumptions(), forged)
    for lid in AMBIGUOUS_8:
        r = chakan(df2, lid)
        assert r.status == "REVIEW" and not r.usable
        assert r.decision == "blocked bulk accept"


def test_individual_decision_still_works_for_ambiguous_cell(data):
    rfx, ex, hist, _, _ = data
    d = {("L23", "V5"): {"action": "accept", "value": None, "via": "individual", "note": "checked with vendor"}}
    r = chakan(normalize.normalise(rfx, ex, hist, normalize.Assumptions(), d), "L23")
    assert r.status == "CONFIRMED" and r.usable and r.decision_note == "checked with vendor"


def test_other_held_back_cases(data):
    _, _, _, _, q = data
    q = review.annotate(q)
    m09 = q[(q.line_id == "L09") & (q.vendor_id == "V1")].iloc[0]
    k14 = q[(q.line_id == "L14") & (q.vendor_id == "V4")].iloc[0]
    assert m09.bulk_block == "vendor quoted a different specification"
    assert k14.bulk_block == "reader flagged it as ambiguous"


# ---------------------------------------------------------------- 2. empty-cell display

def _app_grid_helpers():
    """Load GRID_MISSING, matrix_display and style_matrix from app.py without running the Streamlit page."""
    import ast
    src = open(os.path.join(ROOT, "app.py")).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in ("matrix_display", "style_matrix"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("GRID_MISSING", "STATUS_COLOR") for t in n.targets))]
    ns = {"pd": pd}
    exec(compile(ast.Module(keep, []), "app.py", "exec"), ns)
    return ns


def _app_matrix_display():
    return _app_grid_helpers()["matrix_display"]


@pytest.mark.parametrize("lid", ["L06", "L07", "L28"])
def test_ambad_missing_cells_display_dash(data, lid):
    _, _, _, df, _ = data
    h = _app_grid_helpers()
    grid, _ = h["matrix_display"](df, "landed_price_inr")
    assert pd.isna(grid.xs(lid, level="line_id").iloc[0]["Ambad Kraftline LLP"])   # blank in the data
    html = h["style_matrix"](df, "landed_price_inr").to_html()
    assert h["GRID_MISSING"] == "-"
    assert "None" not in html and "nan" not in html.lower()
    # the rendered row shows '-' in the Ambad column
    styler = h["style_matrix"](df, "landed_price_inr")
    ctx = styler._translate(False, False)
    col = list(grid.columns).index("Ambad Kraftline LLP")
    row = [i for i, (l, _) in enumerate(grid.index) if l == lid][0]
    cell = [c for c in ctx["body"][row] if c.get("type") == "td"][col]
    assert cell["display_value"] == "-"


def test_grid_sorts_numerically_not_as_text(data):
    """Regression: the grid used to hold formatted strings, so '3.67' sorted after '24.55'."""
    _, _, _, df, _ = data
    grid, _ = _app_matrix_display()(df, "landed_price_inr")
    for v in grid.columns:
        assert pd.api.types.is_float_dtype(grid[v]), (v, grid[v].dtype)
    amb = grid["Ambad Kraftline LLP"].sort_values()        # what a header-click sort does: by value
    vals = amb.dropna().tolist()
    assert vals == sorted(vals)
    assert vals.index(min(vals)) == 0 and min(vals) < 10 < max(vals)   # single-digit prices first, not after '24.55'
    as_text = sorted(f"{x:,.2f}" for x in vals)
    assert as_text != [f"{x:,.2f}" for x in vals], "fixture should contain values where text and numeric order differ"
    assert amb.isna().sum() == 3   # the three Ambad gaps stay genuinely blank (NaN), never a string


@pytest.mark.parametrize("lid", ["L06", "L07", "L28"])
def test_ambad_missing_cells_stay_missing_in_data(data, lid):
    _, _, _, df, _ = data
    r = df[(df.line_id == lid) & (df.vendor_id == "V3")].iloc[0]
    assert r.status == "MISSING" and not r.usable
    assert pd.isna(r.unit_price_inr) and pd.isna(r.landed_price_inr) and pd.isna(r.annual_value_inr)


def test_missing_cells_do_not_enter_calculations_or_exports(data):
    rfx, ex, hist, df, _ = data
    vdf = pd.DataFrame([{"vendor_id": v, "vendor": n, "cleared_quality": True} for v, n in
                        df[["vendor_id", "vendor"]].drop_duplicates().values])
    for s in award.scenarios(df, vdf, hist, rfx, [], False):
        picks = s["picks"]
        bad = picks[(picks.vendor == "Ambad Kraftline LLP") & picks.line_id.isin(["L06", "L07", "L28"])]
        assert bad.empty
        assert picks.landed_price_inr.notna().all()
    # CSV export (same call the app uses) leaves them blank, never '-'
    csv = df.drop(columns=["bbox"]).to_csv(index=False)
    out = pd.read_csv(io.StringIO(csv))
    sel = out[(out.vendor_id == "V3") & out.line_id.isin(["L06", "L07", "L28"])]
    assert sel.unit_price_inr.isna().all() and sel.landed_price_inr.isna().all()
    assert "-" not in set(sel.unit_price_inr.astype(str))
    # analyst database: usable table excludes them
    con = analyst.build_db(df, rfx, vdf.assign(gate_summary="", freight="", payment_terms="", lead_time="",
                                               validity="", q1_iso="", q2_lab="", q4_lead=""),
                           pd.DataFrame(columns=["vendor_id", "q_id", "question", "answer"]),
                           pd.DataFrame(columns=["vendor_id", "description", "percent", "condition"]), hist)
    n = con.execute("SELECT count(*) FROM quotes WHERE vendor_id='V3' AND line_id IN ('L06','L07','L28')").fetchone()[0]
    assert n == 0
