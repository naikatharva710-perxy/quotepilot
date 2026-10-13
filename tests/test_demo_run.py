"""Verified demo run loaded from Streamlit Secrets. Offline, no model calls.

Fixture payloads are built from the test fixture (tests/live_shapes.py), never from the real run, so no run data
is committed. Tests marked `real_backup` run only when QP_DEMO_RUN_FILE points at the verified backup locally.
"""
import ast
import base64
import gzip
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import normalize, pipeline, review  # noqa: E402
from tests.live_shapes import build  # noqa: E402

REAL_FILE = os.environ.get("QP_DEMO_RUN_FILE")
real_backup = pytest.mark.skipif(not (REAL_FILE and os.path.exists(REAL_FILE)),
                                 reason="set QP_DEMO_RUN_FILE to the verified backup to run")
AMBIGUOUS_8 = [f"L{i}" for i in range(23, 31)]


# ---------------------------------------------------------------- helpers

def _loader_ns():
    """demo_run_from_secrets and its constants from app.py, without running the page."""
    tree = ast.parse(open(os.path.join(ROOT, "app.py")).read())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in ("demo_run_from_secrets", "demo_run_shape_ok", "_typed"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "").startswith(("DEMO_", "_NUM", "_OPT_STR")) for t in n.targets))]
    ns = {"base64": base64, "gzip": gzip, "hashlib": hashlib, "json": json, "st": SimpleNamespace(secrets={})}
    exec(compile(ast.Module(keep, []), "app.py", "exec"), ns)
    return ns


def _fixture_run() -> dict:
    rfx, ex = build()
    return {"ran_at": "2026-10-13T09:00:00", "bid_date": "2026-10-13", "extractions": ex,
            "certificates": {}, "gates": {}, "errors": []}


def _pack(raw: bytes) -> str:
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode()


def _secrets(blob):
    return {"saved_run": {"gz_b64": blob}}


# ---------------------------------------------------------------- loading and integrity

def test_valid_payload_loads_byte_exact_and_unchanged():
    ns = _loader_ns()
    raw = json.dumps(_fixture_run(), indent=1).encode()
    run, problem = ns["demo_run_from_secrets"](_secrets(_pack(raw)), hashlib.sha256(raw).hexdigest())
    assert problem is None
    assert run == json.loads(raw)                       # identical, extractions included


def test_extractions_not_modified_by_normalise():
    ns = _loader_ns()
    raw = json.dumps(_fixture_run()).encode()
    run, _ = ns["demo_run_from_secrets"](_secrets(_pack(raw)), hashlib.sha256(raw).hexdigest())
    before = json.dumps(run["extractions"], sort_keys=True)
    normalize.normalise(pipeline.load_rfx(), run["extractions"], pipeline.load_history(), normalize.Assumptions())
    assert json.dumps(run["extractions"], sort_keys=True) == before


def test_label_and_fingerprint_constants():
    ns = _loader_ns()
    assert ns["DEMO_RUN_LABEL"] == "Saved demo run — 13 Oct 2026, 10:52"
    assert ns["DEMO_RUN_SHA256"] == "1a39b331062208f9101cc375df2bbe0b9ffc4f0b38c4f0f8a48603e6a7c7dcc4"


# ---------------------------------------------------------------- missing and invalid payloads

@pytest.mark.parametrize("secrets", [{}, {"saved_run": {}}, {"saved_run": {"gz_b64": ""}}, {"other": 1}])
def test_missing_payload_is_silent(secrets):
    assert _loader_ns()["demo_run_from_secrets"](secrets, "x") == (None, None)


def test_no_secrets_file_is_silent():
    class Broken:
        def get(self, *_):
            raise FileNotFoundError("no secrets.toml")
    assert _loader_ns()["demo_run_from_secrets"](Broken(), "x") == (None, None)


def _bad_cases():
    good = json.dumps(_fixture_run()).encode()
    not_json = b"MARKER_PAYLOAD_TEXT not json at all"
    wrong_shape = json.dumps({"ran_at": "x", "MARKER_PAYLOAD_TEXT": 1}).encode()
    no_lines = json.dumps({**_fixture_run(), "extractions": {"V1": {"MARKER_PAYLOAD_TEXT": []}}}).encode()

    def mutate(fn):
        r = _fixture_run(); fn(r); return json.dumps(r).encode()
    no_currency = mutate(lambda r: r["extractions"]["V2"].pop("currency"))
    empty_lines = mutate(lambda r: r["extractions"]["V3"].__setitem__("lines", []))
    line_no_id = mutate(lambda r: r["extractions"]["V4"]["lines"][5].pop("rfx_line_id"))
    conf_text = mutate(lambda r: r["extractions"]["V5"]["lines"][0].__setitem__("confidence", "MARKER_PAYLOAD_TEXT"))
    price_text = mutate(lambda r: r["extractions"]["V1"]["lines"][2].__setitem__("price_value", "MARKER_PAYLOAD_TEXT"))
    quoted_str = mutate(lambda r: r["extractions"]["V1"]["lines"][0].__setitem__("quoted", "yes"))
    bad_quote = mutate(lambda r: r["extractions"]["V2"]["lines"][0].__setitem__("source_quote", 123))
    q_no_id = mutate(lambda r: r["extractions"]["V2"]["questionnaire"].append({"answer": "MARKER_PAYLOAD_TEXT"}))
    return [
        ("not base64", "%%%MARKER_PAYLOAD_TEXT%%%", None, "could not be decoded"),
        ("base64 but not gzip", base64.b64encode(b"MARKER_PAYLOAD_TEXT").decode(), None, "could not be decoded"),
        ("wrong fingerprint", _pack(good), "0" * 64, "failed its integrity check"),
        ("not JSON", _pack(not_json), hashlib.sha256(not_json).hexdigest(), "is not valid JSON"),
        ("wrong structure", _pack(wrong_shape), hashlib.sha256(wrong_shape).hexdigest(), "does not have the expected structure"),
        ("extraction without lines", _pack(no_lines), hashlib.sha256(no_lines).hexdigest(), "does not have the expected structure"),
    ] + [(name, _pack(raw), hashlib.sha256(raw).hexdigest(), "does not have the expected structure") for name, raw in [
        ("vendor missing currency", no_currency), ("vendor with no lines", empty_lines),
        ("line missing rfx_line_id", line_no_id), ("confidence not a number", conf_text),
        ("price_value not a number", price_text), ("quoted not a boolean", quoted_str),
        ("source_quote not text", bad_quote), ("questionnaire answer without q_id", q_no_id)]]


@pytest.mark.parametrize("name,blob,sha,expected", _bad_cases(), ids=[c[0] for c in _bad_cases()])
def test_invalid_payload_refused_with_fixed_message(name, blob, sha, expected):
    run, problem = _loader_ns()["demo_run_from_secrets"](_secrets(blob), sha or "0" * 64)
    assert run is None and problem == expected
    assert "MARKER" not in problem and blob[:20] not in problem


# ---------------------------------------------------------------- the page

def _send_page(monkeypatch, tmp_path, secrets=None, last_run=None):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cache = tmp_path / "last_run.json"
    if last_run is not None:
        cache.write_text(json.dumps(last_run))
    monkeypatch.setattr(pipeline, "CACHE", cache)
    monkeypatch.setattr(pipeline, "SAVED", tmp_path / "saved_run.json")
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120)
    if secrets is not None:
        at.secrets["saved_run"] = secrets["saved_run"]
    at.run()
    at.session_state["rfx"] = pipeline.load_rfx()
    at.session_state["inbox"] = pipeline.sample_inbox()
    at.session_state["sent"] = True                      # same state as "Skip to replies received"
    at.radio(key="page").set_value("Send and collect").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Read every reply with AI" in _text(at)       # the collect section actually rendered
    return at


def _text(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def test_page_without_demo_run_or_last_read_offers_nothing(monkeypatch, tmp_path):
    at = _send_page(monkeypatch, tmp_path)
    assert "Load last AI read" not in [b.label for b in at.button]
    assert "Saved demo run unavailable" not in _text(at)


def test_page_with_bad_secret_falls_back_to_last_read_and_says_why(monkeypatch, tmp_path):
    bad = _secrets(_pack(b'{"MARKER_PAYLOAD_TEXT": 1}'))           # fingerprint will not match
    at = _send_page(monkeypatch, tmp_path, secrets=bad, last_run=_fixture_run())
    text = _text(at)
    assert "Saved demo run unavailable: it failed its integrity check." in text
    assert "MARKER" not in text
    assert "Last AI read on this server, 2026-10-13T09:00:00" in text
    assert "Load last AI read" in [b.label for b in at.button]


def test_temporary_download_button_is_gone(monkeypatch, tmp_path):
    at = _send_page(monkeypatch, tmp_path, last_run=_fixture_run())
    at.radio(key="page").set_value("Assumptions and audit").run()
    assert not at.exception
    assert not at.get("download_button")
    assert "Back up the saved AI run" not in _text(at)


# ---------------------------------------------------------------- the real verified backup (local only)

def _real_secrets():
    raw = open(REAL_FILE, "rb").read()
    return raw, _secrets(_pack(raw))


@real_backup
def test_real_backup_round_trips_and_passes_pinned_fingerprint():
    raw, sec = _real_secrets()
    ns = _loader_ns()
    run, problem = ns["demo_run_from_secrets"](sec)               # default = pinned fingerprint
    assert problem is None and run == json.loads(raw) and run["ran_at"] == "2026-10-13T10:52:40"


@real_backup
def test_demo_run_wins_over_newer_live_read_and_keeps_counts_and_safeguards(monkeypatch, tmp_path):
    raw, sec = _real_secrets()
    at = _send_page(monkeypatch, tmp_path, secrets=sec, last_run=_fixture_run())   # a "live read" exists too
    assert "Saved demo run — 13 Oct 2026, 10:52" in _text(at)
    [b for b in at.button if b.label == "Load last AI read"][0].click().run()
    assert not at.exception
    assert at.session_state["run"]["ran_at"] == "2026-10-13T10:52:40"
    assert at.session_state["run"] == json.loads(raw)
    df = normalize.normalise(pipeline.load_rfx(), at.session_state["run"]["extractions"],
                             pipeline.load_history(), normalize.Assumptions())
    c = df.status.value_counts().to_dict()
    assert (c.get("OK", 0) + c.get("CONFIRMED", 0), c.get("REVIEW", 0), c.get("MISSING", 0) + c.get("REJECTED", 0)) == (115, 32, 3)
    g = review.bulk_groups(df[df.status == "REVIEW"])
    listed = {lid for row in g.line_ids for lid in row}
    assert not listed & set(AMBIGUOUS_8)
    per_kg = g[g.reason_key.apply(review.uses_estimated_weight)]
    assert len(per_kg) == 1 and per_kg.n.iloc[0] == 13
    for lid in ("L06", "L07", "L28"):
        r = df[(df.line_id == lid) & (df.vendor_id == "V3")].iloc[0]
        assert r.status == "MISSING" and pd.isna(r.unit_price_inr)


# ---------------------------------------------------------------- live AI read still works (model mocked, no network)

def _no_network(monkeypatch):
    from core import llm

    def boom(*a, **k):
        raise AssertionError("a model call was attempted in an offline test")
    for name in ("client", "generate", "available_models", "models"):
        monkeypatch.setattr(llm, name, boom)


def _live_read(monkeypatch, tmp_path, secrets=None):
    """Click 'Read all replies now' with pipeline.process mocked to return a fresh run."""
    _no_network(monkeypatch)
    live = {**_fixture_run(), "ran_at": "2026-10-14T08:00:00"}
    monkeypatch.setattr(pipeline, "process", lambda rfx, inbox, progress=None: json.loads(json.dumps(live)))
    at = _send_page(monkeypatch, tmp_path, secrets=secrets)
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-placeholder")        # only so the button is enabled
    at.run()
    btn = [b for b in at.button if b.label == "Read all replies now"][0]
    assert not btn.disabled
    btn.click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at, live


def test_live_read_opens_compare_and_saves_last_run(monkeypatch, tmp_path):
    at, live = _live_read(monkeypatch, tmp_path)
    assert at.session_state["run"]["ran_at"] == live["ran_at"]
    assert at.session_state["page"] == "Compare"
    assert json.loads((tmp_path / "last_run.json").read_text())["ran_at"] == live["ran_at"]


@real_backup
def test_live_read_does_not_replace_the_demo_run(monkeypatch, tmp_path):
    raw, sec = _real_secrets()
    at, live = _live_read(monkeypatch, tmp_path, secrets=sec)
    assert at.session_state["run"]["ran_at"] == live["ran_at"]               # the live read is what you see now
    at.radio(key="page").set_value("Send and collect").run()
    assert "Saved demo run — 13 Oct 2026, 10:52" in _text(at)                # ...but the offer is still the demo run
    [b for b in at.button if b.label == "Load last AI read"][0].click().run()
    assert at.session_state["run"] == json.loads(raw)
    assert open(REAL_FILE, "rb").read() == raw
