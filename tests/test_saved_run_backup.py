"""Temporary saved-run backup: read-only, exact bytes, correct fingerprint, clear message when missing. Offline."""
import ast
import hashlib
import os
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import pipeline  # noqa: E402

SAMPLE = b'{\n "ran_at": "2026-10-13T10:52:40",\n "extractions": {"V1": {"lines": []}}\n}'


def _backup_fn():
    tree = ast.parse(open(os.path.join(ROOT, "app.py")).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "saved_run_backup")
    ns = {"Path": Path, "hashlib": hashlib}
    exec(compile(ast.Module([fn], []), "app.py", "exec"), ns)
    return ns["saved_run_backup"]


def test_backup_returns_exact_bytes_and_fingerprint_without_touching_file(tmp_path):
    f = tmp_path / "last_run.json"
    f.write_bytes(SAMPLE)
    before = (f.stat().st_mtime_ns, f.stat().st_size)
    b = _backup_fn()(f)
    assert b["bytes"] == SAMPLE
    assert b["size"] == len(SAMPLE)
    assert b["sha256"] == hashlib.sha256(SAMPLE).hexdigest()
    assert (f.stat().st_mtime_ns, f.stat().st_size) == before and f.read_bytes() == SAMPLE


def test_backup_missing_file_returns_none(tmp_path):
    assert _backup_fn()(tmp_path / "nope.json") is None
    assert not (tmp_path / "nope.json").exists()       # never created


def _audit_page(monkeypatch, cache_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(pipeline, "CACHE", cache_path)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120).run()
    at.radio(key="page").set_value("Assumptions and audit").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_audit_page_shows_fingerprint_of_served_bytes(monkeypatch, tmp_path):
    f = tmp_path / "last_run.json"
    f.write_bytes(SAMPLE)
    at = _audit_page(monkeypatch, f)
    code = "\n".join(c.value for c in at.code)
    assert hashlib.sha256(SAMPLE).hexdigest() in code
    assert f"bytes   {len(SAMPLE)}" in code
    assert [d.proto.label for d in at.get("download_button")] == ["Download saved run (JSON)"]
    assert f.read_bytes() == SAMPLE


def test_audit_page_says_clearly_when_no_saved_run(monkeypatch, tmp_path):
    at = _audit_page(monkeypatch, tmp_path / "missing.json")
    assert any("No saved run on this server" in i.value for i in at.info)
    assert not at.get("download_button")
    assert not (tmp_path / "missing.json").exists()
