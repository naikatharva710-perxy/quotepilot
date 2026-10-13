"""Ask-the-data analyst: evidence, code-written exclusions, number check, failure handling. Offline: Gemini is mocked."""
import os
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import analyst, award, llm, normalize, pipeline, review  # noqa: E402
from tests.live_shapes import build  # noqa: E402

VP_Q = "What if we split it, cheapest per line, but only among vendors who cleared the quality questionnaire?"
VP_SQL = """WITH c AS (SELECT q.* FROM quotes q JOIN vendors v USING (vendor_id) WHERE v.cleared_quality)
SELECT line_id, vendor_id, vendor, ROUND(landed_price_inr, 2) AS landed_price_inr, annual_value_inr
FROM c QUALIFY ROW_NUMBER() OVER (PARTITION BY line_id ORDER BY landed_price_inr) = 1 ORDER BY line_id"""
CLEARED = {"V2", "V4"}
DISCOUNTS = pd.DataFrame([{"vendor_id": "V2", "description": "volume discount", "percent": 2.5,
                           "condition": "award value above ₹75 lakh"}])


# ---------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network model call attempted in an offline test")
    monkeypatch.setattr(llm, "client", boom)


def _data(decisions=None):
    rfx, ex = build()
    hist = pipeline.load_history()
    df = normalize.normalise(rfx, ex, hist, normalize.Assumptions(), decisions or {})
    vdf = pd.DataFrame([{"vendor_id": v, "vendor": n, "cleared_quality": v in CLEARED, "gate_summary": "", "freight": "",
                         "payment_terms": "", "lead_time": "", "validity": "", "q1_iso": "", "q2_lab": "", "q4_lead": ""}
                        for v, n in df[["vendor_id", "vendor"]].drop_duplicates().values])
    adf = pd.DataFrame(columns=["vendor_id", "q_id", "question", "answer"])
    con = analyst.build_db(df, rfx, vdf, adf, DISCOUNTS, hist)
    return rfx, hist, df, vdf, con, analyst.facts(df, vdf, DISCOUNTS)


def _mock(monkeypatch, plan=None, explain=None, plan_error=None, explain_error=None, repair=None):
    calls = []

    def fake(task, prompt, **kw):
        calls.append(task)
        if task == "analyst:plan":
            if plan_error:
                raise plan_error
            return plan
        if task == "analyst:repair":
            return repair or {"sql": None}
        if task == "analyst:explain":
            if explain_error:
                raise explain_error
            return explain(prompt) if callable(explain) else explain
        raise AssertionError(task)
    monkeypatch.setattr(llm, "generate", fake)
    return calls


# ---------------------------------------------------------------- the VP's question

def test_vp_cleared_vendors_split_matches_award_and_carries_evidence(monkeypatch):
    rfx, hist, df, vdf, con, fp = _data()
    expected = next(s for s in award.scenarios(df, vdf, hist, rfx, [], False) if "cleared vendors only" in s["name"])

    def explain(prompt):
        total = con.execute(f"SELECT SUM(annual_value_inr) FROM ({VP_SQL})").fetchone()[0]
        return f"Split across cleared vendors costs ₹{total / 1e7:.2f} cr a year."
    _mock(monkeypatch, plan={"sql": VP_SQL, "title": "Cleared split", "chart": {"type": "none"}}, explain=explain)
    out = analyst.ask(VP_Q, con, [], fp, df=df, discounts_df=DISCOUNTS)

    t = out["table"]
    assert set(t.vendor_id) <= CLEARED                                     # only cleared vendors
    assert round(t.annual_value_inr.sum(), 0) == expected["total_inr"]    # same as the Award page
    assert len(t) == expected["lines_covered"]
    ev = out["evidence"]                                                   # one source row per result row
    assert list(zip(ev.line_id, ev.vendor)) == list(zip(t.line_id, t.vendor))
    assert ev.matches_result.all() and ev.source_file.notna().all() and ev.source_quote.notna().all()
    assert set(ev.status) <= {"OK", "CONFIRMED"}
    assert out["figures_found"] and not out["figures_unverified"]
    assert any("conditional discount not applied (2.5%" in n for n in out["notes"])
    assert not out["explain_failed"]


# ---------------------------------------------------------------- SQL safety

@pytest.mark.parametrize("sql", ["DROP TABLE quotes", "INSERT INTO quotes VALUES (1)", "SELECT 1; DROP TABLE quotes",
                                 "COPY quotes TO 'x.csv'", "UPDATE quotes SET landed_price_inr = 0",
                                 "WITH x AS (SELECT 1) DELETE FROM quotes", "ATTACH 'evil.db'", "PRAGMA version"])
def test_sql_guard_rejects_anything_but_one_select(sql):
    _, _, _, _, con, _ = _data()
    with pytest.raises(ValueError):
        analyst.run_sql(con, sql)
    assert con.execute("SELECT COUNT(*) FROM quotes").fetchone()[0] > 0


def test_unsafe_plan_computes_nothing_and_leaves_data_intact(monkeypatch):
    _, _, df, _, con, fp = _data()
    before = con.execute("SELECT COUNT(*) FROM quotes").fetchone()[0]
    calls = _mock(monkeypatch, plan={"sql": "DROP TABLE quotes"}, repair={"sql": "DELETE FROM quotes"})
    out = analyst.ask("delete everything", con, [], fp, df=df, discounts_df=DISCOUNTS)
    assert out["table"] is None and out["failed_step"] == "query"
    assert "analyst:explain" not in calls
    assert con.execute("SELECT COUNT(*) FROM quotes").fetchone()[0] == before


# ---------------------------------------------------------------- cannot answer / failures

def test_cannot_answer_is_passed_through_without_a_table(monkeypatch):
    _, _, df, _, con, fp = _data()
    calls = _mock(monkeypatch, plan={"sql": None, "cannot_answer": "The data has no delivery history."})
    out = analyst.ask("Which vendor delivers late most often?", con, [], fp, df=df, discounts_df=DISCOUNTS)
    assert out["answer"] == "The data has no delivery history."
    assert out["table"] is None and out["evidence"] is None and out["failed_step"] is None
    assert calls == ["analyst:plan"]


def test_planner_failure_computes_nothing(monkeypatch):
    _, _, df, _, con, fp = _data()
    _mock(monkeypatch, plan_error=llm.LLMUnavailable("down"))
    out = analyst.ask(VP_Q, con, [], fp, df=df, discounts_df=DISCOUNTS)
    assert out["failed_step"] == "plan" and out["table"] is None and "nothing was computed" in out["answer"]


def test_narrator_failure_keeps_table_sql_evidence_and_notes(monkeypatch):
    _, _, df, _, con, fp = _data()
    _mock(monkeypatch, plan={"sql": VP_SQL}, explain_error=llm.LLMUnavailable("503"))
    out = analyst.ask(VP_Q, con, [], fp, df=df, discounts_df=DISCOUNTS)
    assert out["explain_failed"] is True and out["answer"] is None
    assert out["table"] is not None and len(out["table"]) and out["sql"] == VP_SQL
    assert out["evidence"] is not None and len(out["evidence"]) == len(out["table"])
    assert out["notes"]


# ---------------------------------------------------------------- number check

def test_invented_figures_are_flagged_and_real_ones_pass(monkeypatch):
    _, _, df, _, con, fp = _data()
    real = con.execute(f"SELECT landed_price_inr FROM ({VP_SQL}) WHERE line_id = 'L01'").fetchone()[0]
    _mock(monkeypatch, plan={"sql": VP_SQL},
          explain=f"L01 goes at ₹{real:.2f}. The total is ₹9.87 cr, 12.3% below last year, saving ₹45.6 lakh across 30 lines.")
    out = analyst.ask(VP_Q, con, [], fp, df=df, discounts_df=DISCOUNTS)
    assert f"₹{real:.2f}" in out["figures_found"]
    assert set(out["figures_unverified"]) == {"₹9.87 cr", "12.3%", "₹45.6 lakh"}


def test_number_check_ignores_counts_and_line_ids():
    found, missing = analyst.check_numbers("30 lines, 5-ply, L23 and 2 vendors", pd.DataFrame({"x": [1.0]}), {}, "")
    assert found == [] and missing == []


# ---------------------------------------------------------------- exclusions written by code

def _ids(note: str) -> set[str]:
    """Expand 'L06–L07, L20–L30' in a note to the individual line ids."""
    import re
    out = set()
    for a, b in re.findall(r"L(\d+)(?:–L(\d+))?", note.split("(", 1)[1]):
        out |= {f"L{i:02d}" for i in range(int(a), int(b or a) + 1)}
    return out


def test_chakan_l23_l30_listed_as_not_counted_even_after_bulk_accept():
    rfx, ex = build()
    hist = pipeline.load_history()
    df0 = normalize.normalise(rfx, ex, hist, normalize.Assumptions())
    q = df0[df0.status == "REVIEW"]
    key = next(k for k in review.bulk_groups(q).reason_key if review.uses_estimated_weight(k))
    decisions = {}
    review.bulk_accept(q, "V5", key, decisions)
    _, _, df, _, _, _ = _data(decisions)
    result = df[df.usable & df.line_id.isin([f"L{i}" for i in range(23, 31)])][["line_id", "vendor_id", "vendor", "landed_price_inr"]]
    notes = analyst.exclusion_notes(df, result, DISCOUNTS)
    chakan = [n for n in notes if n.startswith("Chakan Boxcraft Solutions") and "awaiting buyer review, not counted" in n]
    assert chakan and _ids(chakan[0]) == {f"L{i}" for i in range(23, 31)} and "8 cells" in chakan[0]
    # whole-comparison answer: the 13 bulk-accepted cells are counted, so they are not listed as excluded
    full = analyst.exclusion_notes(df, None, DISCOUNTS)
    listed = next(n for n in full if n.startswith("Chakan") and "awaiting" in n)
    assert "17 cells" in listed and {f"L{i}" for i in range(23, 31)} <= _ids(listed)
    assert not _ids(listed) & {"L01", "L02", "L03", "L04", "L05", "L08", "L09", "L10", "L11", "L12", "L13", "L14", "L19"}


def test_assumptions_are_labelled_as_assumptions_not_vendor_facts():
    _, _, df, _, _, _ = _data()
    result = df[df.line_id.isin(["L01", "L02"])][["line_id", "vendor_id", "vendor", "landed_price_inr"]]
    notes = " ".join(analyst.exclusion_notes(df, result, DISCOUNTS))
    assert "our 4% freight assumption" in notes and "not a vendor-quoted freight cost" in notes
    assert "USD prices converted at ₹88.50, our editable exchange-rate assumption" in notes
    assert "estimated box weights" in notes and "has not supplied or confirmed" in notes
    assert "conditional discount not applied" in notes


def test_aggregate_answer_has_no_fake_evidence_and_scopes_to_its_vendors():
    _, _, df, _, _, _ = _data()
    agg = pd.DataFrame({"vendor_id": ["V2"], "vendor": ["Bhimthadi Corrugation Works"], "total": [1.0]})
    ev, note = analyst.evidence_for(agg, df)
    assert ev is None and "aggregates" in note
    notes = analyst.exclusion_notes(df, agg, DISCOUNTS)
    assert all(n.startswith("Bhimthadi") for n in notes)          # only the vendor in scope


# ---------------------------------------------------------------- the page

PAGE_SQL = "SELECT line_id, vendor_id, vendor, landed_price_inr FROM quotes WHERE line_id IN ('L01', 'L02') ORDER BY line_id, vendor"


def test_ask_page_shows_evidence_and_notes_when_explanation_fails(monkeypatch):
    _mock(monkeypatch, plan={"sql": PAGE_SQL, "title": "L01-L02 prices", "chart": {"type": "none"}},
          explain_error=llm.LLMUnavailable("503"))
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-placeholder")     # enables the page; the model itself is mocked
    rfx, ex = build()
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120).run()
    at.session_state["rfx"] = rfx
    at.session_state["run"] = {"ran_at": "test", "bid_date": "2026-10-13", "extractions": ex,
                               "certificates": {}, "gates": {}, "errors": []}
    at.session_state["inbox"] = []
    at.radio(key="page").set_value("Ask the data").run()
    at.chat_input[0].set_value("Show usable prices for L01 and L02").run()
    assert not at.exception, [e.value for e in at.exception]
    msg = at.session_state["chat"][-1]
    assert msg["explain_failed"] and msg["table"] and msg["evidence"] and msg["notes"]
    assert any("explanation step failed" in w.value for w in at.warning)
    assert any("Not counted, and assumptions inside these numbers" in m.value for m in at.markdown)
    assert any(e.label.startswith("Sources for this answer") for e in at.expander)


# ---------------------------------------------------------------- the real verified run (local only, never committed)

REAL_FILE = os.environ.get("QP_DEMO_RUN_FILE")


@pytest.mark.skipif(not (REAL_FILE and os.path.exists(REAL_FILE)), reason="set QP_DEMO_RUN_FILE to the verified backup to run")
def test_real_run_vp_question_matches_award_page(monkeypatch):
    import json
    run = json.load(open(REAL_FILE))
    rfx, hist = pipeline.load_rfx(), pipeline.load_history()
    df = normalize.normalise(rfx, run["extractions"], hist, normalize.Assumptions())
    vdf, adf = pipeline.vendor_tables(rfx, run)
    ddf = pd.DataFrame(normalize.discounts(run["extractions"]))
    con = analyst.build_db(df, rfx, vdf, adf, ddf, hist)
    expected = next(s for s in award.scenarios(df, vdf, hist, rfx, [], False) if "cleared vendors only" in s["name"])
    _mock(monkeypatch, plan={"sql": VP_SQL}, explain_error=llm.LLMUnavailable("not needed"))
    out = analyst.ask(VP_Q, con, [], analyst.facts(df, vdf, ddf), df=df, discounts_df=ddf)
    assert round(out["table"].annual_value_inr.sum(), 0) == expected["total_inr"]
    assert set(out["table"].vendor_id) == set(vdf[vdf.cleared_quality].vendor_id)
    assert out["evidence"].matches_result.all() and out["evidence"].source_quote.notna().all()
