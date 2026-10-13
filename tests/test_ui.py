"""UI-level checks with Streamlit's AppTest, using the live-shaped run (no model calls)."""
import os
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.live_shapes import build  # noqa: E402

AMBIGUOUS_8 = ["L23", "L24", "L25", "L26", "L27", "L28", "L29", "L30"]


@pytest.fixture()
def app():
    os.environ.pop("GEMINI_API_KEY", None)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120).run()
    rfx, ex = build()
    at.session_state["rfx"] = rfx
    at.session_state["run"] = {"ran_at": "test", "bid_date": "2026-10-13", "extractions": ex,
                               "certificates": {}, "gates": {}, "errors": []}
    at.session_state["inbox"] = []
    return at


def goto(at, page):
    at.radio(key="page").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_compare_grid_shows_dash_for_ambad_missing(app):
    at = goto(app, "Compare")
    el = at.dataframe[0]
    assert el.proto.placeholder == "-"          # missing values render as '-', not 'None'
    grid = el.value
    grid = grid.data if hasattr(grid, "data") else grid
    for lid in ("L06", "L07", "L28"):
        row = grid.xs(lid, level="line_id").iloc[0]
        assert pd.isna(row["Ambad Kraftline LLP"]), (lid, row["Ambad Kraftline LLP"])
    assert all(pd.api.types.is_float_dtype(grid[c]) for c in grid.columns)   # numeric, so it sorts as numbers


def test_review_bulk_button_excludes_ambiguous_and_logs_each_cell(app):
    at = goto(app, "Review queue")
    labels = [b.label for b in at.button]
    chakan_md = [m.value for m in at.markdown if "Chakan Boxcraft Solutions" in m.value and "cells" in m.value]
    per_kg = [m for m in chakan_md if "Per-kg rate converted" in m][0]
    assert not any(f"{lid}," in per_kg or per_kg.rstrip(".").endswith(lid) for lid in AMBIGUOUS_8), per_kg
    assert "need individual decisions and are not included" in per_kg
    caps = [c.value for c in at.caption if "estimated box weight" in c.value]
    assert caps and "not supplied or confirmed" in caps[0]
    btn = [b for b in at.button if b.label == "Accept 13 at est. weights"]
    assert btn, labels
    assert "estimated box weights" in btn[0].help and "not supplied or confirmed" in btn[0].help
    btn[0].click().run()
    assert not at.exception
    dec = at.session_state["decisions"]
    assert not any((lid, "V5") in dec for lid in AMBIGUOUS_8)
    assert sum(1 for k in dec if k[1] == "V5") == 13
    audit = at.session_state["audit"]
    bulk_logs = [a for a in audit if a["action"] == "Cell accept (bulk)"]
    assert len(bulk_logs) == 13 and all("source V5_file, line 8" in a["detail"] for a in bulk_logs)
    assert all("estimated box weight" in a["detail"] and "not supplied or confirmed" in a["detail"] for a in bulk_logs)
    assert all(d.get("via") == "bulk" for k, d in dec.items() if k[1] == "V5")


def test_ambiguous_cells_show_individual_decision_chip(app):
    at = goto(app, "Review queue")
    chips = [m.value for m in at.markdown if "Needs your individual decision" in m.value]
    assert len(chips) >= 8
    assert not any("nan" in c for c in chips)


# ---------------------------------------------------------------- individual decisions: audit log wording

EST = "(price uses our estimated box weight; the vendor has not supplied or confirmed it)"


def _decide(at, lid, vid, decision, note, value=None):
    """Make one individual decision on the Review queue card for (lid, vid); return its audit log entry."""
    key = f"{lid}_{vid}"
    at.radio(key=f"a_{key}").set_value(decision).run()
    if value is not None:
        at.number_input(key=f"v_{key}").set_value(value).run()
    at.text_input(key=f"n_{key}").input(note).run()
    at.button(key=f"s_{key}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    entries = [a for a in at.session_state["audit"] if a["detail"].startswith(f"{lid}/")]
    assert len(entries) == 1, entries
    return entries[0]


def test_individual_accept_of_per_kg_price_discloses_estimated_weight(app):
    at = goto(app, "Review queue")
    e = _decide(at, "L23", "V5", "Accept", "checked partition spec")
    assert e["action"] == "Cell accept"
    assert e["detail"].startswith("L23/Chakan Boxcraft Solutions: checked partition spec; was ₹")   # note + previous price kept
    assert EST in e["detail"]
    assert e["detail"].endswith('; source V5_file, line 8: "38 for the 3-ply"')                     # source + quote kept
    assert at.session_state["decisions"][("L23", "V5")]["via"] == "individual"


def test_ordinary_individual_decisions_have_no_estimated_weight_disclosure(app):
    at = goto(app, "Review queue")
    # Not a per-kg price: Krishnai's smudged L14 and Meridian's spec-swap L09
    k = _decide(at, "L14", "V4", "Accept", "confirmed 17.00 by phone")
    assert k["detail"].startswith("L14/Krishnai Pack Industries: confirmed 17.00 by phone; was ₹")
    assert k["detail"].endswith('; source V4_file, x: "q"') and "estimated box weight" not in k["detail"]
    m = _decide(at, "L09", "V1", "Reject", "3-ply not acceptable")
    assert m["action"] == "Cell reject" and "estimated box weight" not in m["detail"]
    # A per-kg cell that is corrected or rejected no longer relies on our weight estimate: no disclosure
    c = _decide(at, "L24", "V5", "Correct", "vendor confirmed price", value=4.10)
    assert c["action"] == "Cell override" and "→ ₹4.10" in c["detail"] and "estimated box weight" not in c["detail"]
    r = _decide(at, "L25", "V5", "Reject", "not covered by 3-ply rate")
    assert r["action"] == "Cell reject" and "estimated box weight" not in r["detail"]
    # and the previous-price / source format is unchanged for all of them
    for e in (k, m, c, r):
        assert "; was ₹" in e["detail"] and "; source " in e["detail"]


# ---------------------------------------------------------------- award card: direction of change vs last year

def _vs_last_year():
    import ast
    tree = ast.parse(open(os.path.join(ROOT, "app.py")).read())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("vs_last_year", "lakh")]
    ns = {}
    exec(compile(ast.Module(keep, []), "app.py", "exec"), ns)
    return ns["vs_last_year"]


def test_award_card_says_more_expensive_when_cost_rises():
    # cleared-vendors scenario on the demo run: ₹7,55,73,300 vs ₹7,28,35,500 last year, saving field -3.8
    text = _vs_last_year()(75_573_300, 72_835_500, -3.8)
    assert text == "₹27.4 L (3.8%) more expensive than last year on the same lines"
    assert "-" not in text.split("(")[1]


def test_award_card_says_cheaper_when_cost_falls_and_same_when_flat():
    f = _vs_last_year()
    assert f(72_133_600, 72_835_500, 1.0) == "₹7.0 L (1.0%) cheaper than last year on the same lines"
    assert f(100.0, 100.0, 0.0) == "About the same as last year on the same lines"
