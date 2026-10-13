"""UI-level checks with Streamlit's AppTest, using the live-shaped run (no model calls)."""
import os
import sys

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
    grid = at.dataframe[0].value
    grid = grid.data if hasattr(grid, "data") else grid   # Styler -> its display data
    for lid in ("L06", "L07", "L28"):
        row = grid.xs(lid, level="line_id").iloc[0]
        assert row["Ambad Kraftline LLP"] == "-", (lid, row["Ambad Kraftline LLP"])
    assert "None" not in grid.to_string()


def test_review_bulk_button_excludes_ambiguous_and_logs_each_cell(app):
    at = goto(app, "Review queue")
    labels = [b.label for b in at.button]
    chakan_md = [m.value for m in at.markdown if "Chakan Boxcraft Solutions" in m.value and "cells" in m.value]
    per_kg = [m for m in chakan_md if "Per-kg rate converted" in m][0]
    assert not any(f"{lid}," in per_kg or per_kg.rstrip(".").endswith(lid) for lid in AMBIGUOUS_8), per_kg
    assert "need individual decisions and are not included" in per_kg
    btn = [b for b in at.button if b.label == "Accept these 13"]
    assert btn, labels
    btn[0].click().run()
    assert not at.exception
    dec = at.session_state["decisions"]
    assert not any((lid, "V5") in dec for lid in AMBIGUOUS_8)
    assert sum(1 for k in dec if k[1] == "V5") == 13
    audit = at.session_state["audit"]
    bulk_logs = [a for a in audit if a["action"] == "Cell accept (bulk)"]
    assert len(bulk_logs) == 13 and all("source V5_file, line 8" in a["detail"] for a in bulk_logs)


def test_ambiguous_cells_show_individual_decision_chip(app):
    at = goto(app, "Review queue")
    chips = [m.value for m in at.markdown if "Needs your individual decision" in m.value]
    assert len(chips) >= 8
    assert not any("nan" in c for c in chips)
