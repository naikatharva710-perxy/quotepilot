"""Orchestration: read every response, read certificates, check quality gates, cache the run."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd

from . import extract, ingest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache" / "last_run.json"

SAMPLE_FILES = {
    "V1": "V1_Meridian_Fold_commercial_offer.xlsx",
    "V2": "V2_Bhimthadi_quotation.pdf",
    "V3": "V3_Ambad_Kraftline_offer.docx",
    "V4": "V4_Krishnai_Pack_ratecard_photo.jpg",
    "V5": "V5_Chakan_Boxcraft_email.txt",
}
SAMPLE_CERTS = {
    "V1": "V1_Meridian_ISO9001.pdf", "V2": "V2_Bhimthadi_ISO9001.pdf",
    "V3": "V3_Ambad_ISO9001.pdf", "V5": "V5_Chakan_ISO9001.pdf",
}


def load_rfx() -> dict:
    return json.loads((DATA / "rfx_sample.json").read_text())


def load_history() -> pd.DataFrame:
    return pd.read_csv(DATA / "history_last_year.csv")


def load_ground_truth() -> dict:
    return json.loads((DATA / "ground_truth.json").read_text())


def sample_inbox() -> list[dict]:
    out = []
    for vid, fn in SAMPLE_FILES.items():
        p = DATA / "vendor_responses" / fn
        cert = SAMPLE_CERTS.get(vid)
        out.append({"vendor_id": vid, "file": fn, "bytes": p.read_bytes(),
                    "cert": cert, "cert_bytes": (DATA / "attachments" / cert).read_bytes() if cert else None})
    return out


def _date(s):
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def process(rfx: dict, inbox: list[dict], progress=None, bid_date: dt.date | None = None) -> dict:
    bid_date = bid_date or dt.date.today()
    vendors = {v["vendor_id"]: v for v in rfx["vendors"]}
    run = {"ran_at": dt.datetime.now().isoformat(timespec="seconds"), "bid_date": bid_date.isoformat(),
           "extractions": {}, "certificates": {}, "gates": {}, "errors": []}
    steps = len(inbox) * 2 + 1
    k = 0
    for item in inbox:
        vid = item["vendor_id"]
        k += 1
        if progress: progress(k / steps, f"Reading {item['file']}")
        try:
            doc = ingest.load(item["file"], item["bytes"])
            run["extractions"][vid] = extract.extract_response(doc, rfx, vendors.get(vid, {}).get("name"))
        except Exception as e:
            run["errors"].append(f"{vid} extraction: {e}")
        k += 1
        if item.get("cert_bytes"):
            if progress: progress(k / steps, f"Reading certificate {item['cert']}")
            try:
                mime = "application/pdf" if item["cert"].lower().endswith(".pdf") else "image/jpeg"
                c = extract.read_certificate(item["cert"], item["cert_bytes"], mime)
                vu = _date(c.get("valid_until"))
                c["valid_on_bid_date"] = None if vu is None else vu >= bid_date
                run["certificates"][vid] = c
            except Exception as e:
                run["errors"].append(f"{vid} certificate: {e}")
        time.sleep(1.5)  # stay polite with free-tier rate limits
    if progress: progress(1.0, "Checking quality gates")
    run["gates"] = gate_check(rfx, run)
    return run


def gate_check(rfx: dict, run: dict) -> dict:
    payload = []
    for vid, ex in run["extractions"].items():
        answers = [{"q_id": a.get("q_id"), "answered": a.get("answered"), "answer": a.get("answer")}
                   for a in ex.get("questionnaire", [])]
        payload.append({"vendor_id": vid, "answers": answers, "lead_time_in_terms": (ex.get("terms") or {}).get("lead_time"),
                        "certificate": run["certificates"].get(vid) or "no certificate attached"})
    gates = {}
    try:
        res = extract.assess_gates(rfx, payload)
        for v in res.get("vendors", []):
            gates[v["vendor_id"]] = {g["q_id"]: {"result": g["result"], "reason": g["reason"]} for g in v.get("gates", [])}
    except Exception as e:
        run["errors"].append(f"gate assessment: {e}")
    # Deterministic guardrail: the certificate date decides Q1, whatever the vendor's answer says.
    for vid in run["extractions"]:
        g = gates.setdefault(vid, {})
        c = run["certificates"].get(vid)
        if not c:
            if g.get("Q1", {}).get("result") == "pass":
                g["Q1"] = {"result": "unknown", "reason": "Vendor claims certification but attached no certificate"}
        elif c.get("valid_on_bid_date") is False:
            g["Q1"] = {"result": "fail", "reason": f"Certificate expired on {c.get('valid_until')} (bid date {run['bid_date']}), "
                                                    "although the vendor's answer says certified"}
    return gates


def vendor_tables(rfx: dict, run: dict, overrides: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    overrides = overrides or {}
    qtext = {q["q_id"]: q["question"] for q in rfx["questionnaire"]}
    gate_ids = [q["q_id"] for q in rfx["questionnaire"] if q.get("gate")]
    vrows, arows = [], []
    names = {v["vendor_id"]: v["name"] for v in rfx["vendors"]}
    for vid, ex in run["extractions"].items():
        g = dict(run["gates"].get(vid, {}))
        for qid, res in (overrides.get(vid) or {}).items():
            g[qid] = res
        results = {q: (g.get(q) or {}).get("result", "unknown") for q in gate_ids}
        cleared = all(r == "pass" for r in results.values())
        summary = "; ".join(f"{q}: {results[q]} ({(g.get(q) or {}).get('reason', '')})" for q in gate_ids)
        t = ex.get("terms") or {}
        vrows.append(dict(vendor_id=vid, vendor=names.get(vid, ex.get("vendor_name")), cleared_quality=cleared,
                          gate_summary=summary, freight=t.get("freight"), payment_terms=t.get("payment_terms"),
                          lead_time=t.get("lead_time"), validity=t.get("validity"),
                          q1_iso=results.get("Q1"), q2_lab=results.get("Q2"), q4_lead=results.get("Q4")))
        for a in ex.get("questionnaire", []):
            arows.append(dict(vendor_id=vid, q_id=a.get("q_id"), question=qtext.get(a.get("q_id"), ""),
                              answer=a.get("answer") if a.get("answered") else "(not answered)"))
    return pd.DataFrame(vrows), pd.DataFrame(arows, columns=["vendor_id", "q_id", "question", "answer"])


def save_cache(run: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(run, indent=1, default=str))


SAVED = DATA / "cache" / "saved_run.json"   # a real run committed to the repo so the demo survives rate limits


def load_cache() -> dict | None:
    for p in (CACHE, SAVED):
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return None
