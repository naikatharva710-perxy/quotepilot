"""QuotePilot: draft an RFx, read whatever vendors send back, interrogate the comparison.

Built for the Aerchain "Kill the Quote Spreadsheet" assignment.
"""
from __future__ import annotations

import base64
import copy
import datetime as dt
import gzip
import hashlib
import io
import json

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from core import analyst, award, copilot, ingest, llm, normalize, pipeline, review

st.set_page_config(page_title="QuotePilot", page_icon="📦", layout="wide")

# ------------------------------------------------------------------ style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, input, textarea { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Archivo', sans-serif !important; letter-spacing: -0.01em; color: #1F2B30; }
h1 { font-weight: 800 !important; font-size: 2.1rem !important; }
h2 { font-weight: 700 !important; font-size: 1.45rem !important; }
h3 { font-weight: 700 !important; font-size: 1.1rem !important; }
.block-container { padding-top: 2.2rem; max-width: 1320px; }
[data-testid="stSidebar"] { background: #1F2B30; }
[data-testid="stSidebar"] * { color: #E7EAE4 !important; }
[data-testid="stSidebar"] .stRadio label p { font-size: 0.98rem; }
.qd-lede { font-size: 1.05rem; color: #46555B; max-width: 68ch; line-height: 1.55; }
.qd-chip { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.82rem; font-weight: 600; margin-right: 6px; }
.ok { background:#DCEFE5; color:#1E6446; } .review { background:#F8EBCB; color:#7A5208; }
.missing { background:#F3DADA; color:#8E2B2B; } .absent { background:#F1F1EE; color:#5E6360; border:1px solid #DADBD6; } .info { background:#E3E7EA; color:#33434A; }
.qd-stat { border-left: 4px solid #9A6631; padding: 4px 0 4px 12px; margin-bottom: 8px; }
.qd-stat b { font-family: 'Archivo', sans-serif; font-size: 1.6rem; display:block; line-height: 1.1; }
.qd-stat span { color:#56656B; font-size: 0.85rem; }
.qd-quote { background:#FFFFFF; border:1px solid #D9DDD6; border-left:4px solid #9A6631; padding:10px 14px; font-size:0.95rem; }
.qd-small { color:#5D6B70; font-size:0.86rem; }
div[data-testid="stDataFrame"] { border: 1px solid #D9DDD6; }
</style>
""", unsafe_allow_html=True)

STATUS_COLOR = {"OK": "#DCEFE5", "CONFIRMED": "#C7E6D5", "REVIEW": "#F8EBCB", "MISSING": "#F1F1EE", "REJECTED": "#F3DADA"}
BUYER = {"name": "Priya Kulkarni", "role": "Category Buyer - Packaging", "company": "Konkanveda Foods Pvt Ltd (fictional)"}


def inr(x: float | None, dec: int = 2) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"₹{x:,.{dec}f}"


def lakh(x: float) -> str:
    if x >= 1e7:
        return f"₹{x/1e7:.2f} cr"
    return f"₹{x/1e5:.1f} L"


# ------------------------------------------------------------------ state
S = st.session_state
if "init" not in S:
    S.init = True
    S.master = pipeline.load_rfx()           # item master + forecast + standard questionnaire
    S.rfx = None                             # issued RFx
    S.draft = None
    S.copilot_msgs = []
    S.sent = False
    S.inbox = []
    S.run = None
    S.decisions = {}
    S.gate_overrides = {}
    S.audit = []
    S.assume = normalize.Assumptions()
    S.chat = []
    S.clar = {}
    S.memo = None
    S.history = pipeline.load_history()

AI_ON = bool(llm.api_key())
if S.get("nav_to"):
    S.page = S.nav_to
    S.nav_to = None


def go(page_name: str):
    S.nav_to = page_name
    st.rerun()


def log(action: str, detail: str):
    S.audit.append({"time": dt.datetime.now().strftime("%H:%M:%S"), "action": action, "detail": detail})


def draft_to_rfx(draft: dict) -> dict:
    m = S.master
    items = {l["line_id"]: l for l in m["line_items"]}
    ids = [i for i in draft.get("line_ids") or list(items) if i in items]
    lines = []
    for i in ids:
        l = dict(items[i])
        if str(i) in (draft.get("qty_overrides") or {}):
            l["annual_qty"] = int(draft["qty_overrides"][str(i)])
        lines.append(l)
    vids = draft.get("vendor_ids") or [v["vendor_id"] for v in m["vendors"]]
    return {
        "rfx_id": m["rfx_id"], "title": draft.get("title") or m["title"], "buyer": m["buyer"],
        "category": draft.get("category") or m["category"], "currency": "INR", "line_items": lines,
        "questionnaire": draft.get("questionnaire") or m["questionnaire"],
        "terms": {**m["terms"], **(draft.get("terms") or {})},
        "vendors": [v for v in m["vendors"] if v["vendor_id"] in vids],
    }


def compute():
    """Everything downstream of the AI run is recomputed deterministically from state."""
    if not S.run or not S.rfx:
        return None
    if not S.run.get("extractions"):
        return None
    df = normalize.normalise(S.rfx, S.run["extractions"], S.history, S.assume, S.decisions)
    if df.empty:
        return None
    vdf, adf = pipeline.vendor_tables(S.rfx, S.run, S.gate_overrides)
    disc = normalize.discounts(S.run["extractions"])
    ddf = pd.DataFrame(disc) if disc else pd.DataFrame(columns=["vendor_id", "description", "percent", "condition"])
    for c in ["vendor_id", "description", "percent", "condition"]:
        if c not in ddf:
            ddf[c] = None
    return df, vdf, adf, ddf


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## QuotePilot")
    st.caption("Corrugated packaging · Konkanveda Foods (fictional)")
    steps = ["Start here", "Draft the RFx", "Send and collect", "Compare", "Review queue", "Ask the data", "Award", "Assumptions and audit"]
    page = st.radio("Go to", steps, label_visibility="collapsed", key="page")
    st.divider()
    st.markdown(f"AI reader: **{'on' if AI_ON else 'off - no key'}**")
    _ok = [t for t in llm.trace() if t.ok]
    if _ok:
        st.caption(f"Last model used: {_ok[-1].model}")
    if S.run:
        st.markdown(f"Responses read: **{len(S.run['extractions'])}**")
    st.caption("Plumbing is stubbed (no real email). Extraction, reasoning and answers are live model calls.")


# ------------------------------------------------------------------ pages
def page_start():
    st.markdown("# Kill the quote spreadsheet")
    st.markdown('<p class="qd-lede">Five vendors replied to a 30-line corrugated packaging RFx in five different shapes: '
                'an Excel that ignores the template and quotes in USD per 1,000, a PDF with the discount in a footnote, '
                'a Word letter with prices in paragraphs and "per box" meaning 100 pieces, a phone photo of a rate card, '
                'and a one-line email. QuotePilot reads all of it, puts it on one comparable grid, and lets the buyer ask '
                'questions all the way to an award she can defend.</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown("### The model reads, code does the maths")
    c1.write("The AI only reports what each document says, with a pointer to where it saw it. Unit and currency "
             "conversions are visible rules you can edit. No number on screen was produced by a model doing arithmetic.")
    c2.markdown("### Nothing uncertain is used silently")
    c2.write("Every cell is OK, needs review, or missing. Analysis and awards use only confirmed cells, and every "
             "answer says what it left out. A ₹4 crore decision should never rest on a guess.")
    c3.markdown("### Every number has a receipt")
    c3.write("Click any price to see the quote it came from, the cell or paragraph, the conversion chain and the reader's "
             "confidence. For the photo, you see the crop.")
    st.divider()
    st.markdown("### Drive it")
    a, b = st.columns(2)
    with a:
        st.write("**Full flow.** Talk the RFx into existence with the co-pilot, send it, then read the replies.")
        if st.button("Start with the co-pilot", type="primary"):
            go("Draft the RFx")
    with b:
        st.write("**Jump ahead.** Issue the sample RFx and load the five vendor replies straight into the inbox.")
        if st.button("Skip to replies received"):
            S.rfx = copy.deepcopy(S.master); S.sent = True; S.inbox = pipeline.sample_inbox()
            log("RFx issued", "Sample RFx issued (skip-ahead)")
            go("Send and collect")
    st.caption("All companies, people, prices and certificates in this demo are fictional.")
    if not AI_ON:
        st.warning("No Gemini API key is configured, so the AI steps are switched off. Add GEMINI_API_KEY to the app's secrets.")


def page_draft():
    st.markdown("# Draft the RFx")
    st.markdown('<p class="qd-lede">Tell the co-pilot what you need in your own words. It works from your item master, '
                'this year\'s forecast and your standard quality questionnaire, and keeps the draft on the right up to date.</p>',
                unsafe_allow_html=True)
    left, right = st.columns([1.05, 1])
    with left:
        if not S.copilot_msgs:
            S.copilot_msgs = [{"role": "assistant", "content": "Hi Priya. What are we sourcing, and for which plant?"}]
        for m in S.copilot_msgs:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        starter = None
        if len(S.copilot_msgs) == 1:
            if st.button("Try: annual corrugated contract for Ranjangaon, usual five vendors, quality gate"):
                starter = ("Annual rate contract for all our corrugated packaging at Ranjangaon - same 30 items as last year, "
                           "this year's forecast quantities. Quality matters this time: I want ISO 9001, in-house BCT/ECT testing "
                           "and lead time within 15 days as hard gates. Invite our usual five vendors. Quotes in 9 days.")
        msg = st.chat_input("Describe what you need, or ask the co-pilot to change something") or starter
        if msg:
            if not AI_ON:
                st.error("The co-pilot needs the AI reader. Add a Gemini key, or use the sample RFx.")
            else:
                S.copilot_msgs.append({"role": "user", "content": msg})
                with st.spinner("Co-pilot is drafting"):
                    try:
                        master = copy.deepcopy(S.master)
                        ctx_vendors = master["vendors"]
                        res = copilot.copilot_turn(S.copilot_msgs, master["line_items"] + [], ctx_vendors,
                                                   S.draft or {"standard_questionnaire": master["questionnaire"],
                                                               "standard_terms": master["terms"]})
                        S.copilot_msgs.append({"role": "assistant", "content": res.get("reply", "")})
                        if res.get("draft"):
                            S.draft = res["draft"]
                            log("Co-pilot draft updated", f"{len(S.draft.get('line_ids') or [])} lines")
                        if res.get("missing"):
                            S.copilot_msgs[-1]["content"] += "\n\n_Still open: " + "; ".join(res["missing"]) + "_"
                    except llm.LLMUnavailable as e:
                        S.copilot_msgs.append({"role": "assistant", "content": f"I couldn't reach the model: {e}"})
                st.rerun()
    with right:
        rfx_view = draft_to_rfx(S.draft) if S.draft else None
        if rfx_view is None:
            st.info("The draft appears here once the co-pilot has enough to start. Or issue the sample RFx.")
            if st.button("Use the sample RFx"):
                S.draft = {"line_ids": [l["line_id"] for l in S.master["line_items"]]}
                st.rerun()
            return
        st.markdown(f"### {rfx_view['title']}")
        st.caption(f"{rfx_view['rfx_id']} · {len(rfx_view['line_items'])} lines · {len(rfx_view['vendors'])} vendors")
        t1, t2, t3, t4 = st.tabs(["Line items", "Questionnaire", "Terms", "Vendors"])
        with t1:
            st.dataframe(pd.DataFrame(rfx_view["line_items"])[["line_id", "item", "size_mm", "spec", "printing", "annual_qty"]],
                         hide_index=True, height=330, width="stretch")
        with t2:
            st.dataframe(pd.DataFrame(rfx_view["questionnaire"]), hide_index=True, width="stretch")
        with t3:
            st.json(rfx_view["terms"])
        with t4:
            st.dataframe(pd.DataFrame(rfx_view["vendors"]), hide_index=True, width="stretch")
        if st.button("Issue this RFx", type="primary"):
            S.rfx = rfx_view
            log("RFx issued", f"{len(rfx_view['line_items'])} lines to {len(rfx_view['vendors'])} vendors")
            go("Send and collect")


def preview_doc(item: dict):
    name, data = item["file"], item["bytes"]
    if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        st.image(data, width="stretch")
    elif name.lower().endswith(".xlsx"):
        x = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
        for sh, d in x.items():
            st.caption(f"Sheet: {sh}")
            st.dataframe(d.fillna("").astype(str), hide_index=True, width="stretch", height=260)
    else:
        d = ingest.load(name, data)
        st.text(d.text[:6000])
    st.download_button("Download original", data, file_name=name, key=f"dl_{item['vendor_id']}")


# Verified demo run, kept in Streamlit Secrets (not in git) so it survives reboots. Pinned by fingerprint:
# the hash below says nothing about the contents, but any other payload is refused.
DEMO_RUN_SHA256 = "1a39b331062208f9101cc375df2bbe0b9ffc4f0b38c4f0f8a48603e6a7c7dcc4"
DEMO_RUN_LABEL = "Saved demo run — 13 Oct 2026, 10:52"
DEMO_RUN_KEYS = {"ran_at": str, "bid_date": str, "extractions": dict, "certificates": dict, "gates": dict, "errors": list}
# Fields the app reads from each vendor's extraction and each quoted line (core/normalize.py, core/pipeline.py).
_NUM = (int, float)
_OPT_STR = (str, type(None))
DEMO_VENDOR_KEYS = {"currency": str, "terms": dict, "lines": list, "questionnaire": list, "_source_file": str}
DEMO_LINE_REQUIRED = {"rfx_line_id": str, "quoted": bool, "price_basis": str, "confidence": _NUM,
                      "price_value": _NUM + (type(None),)}
DEMO_LINE_OPTIONAL = {"price_currency": _OPT_STR, "basis_pack_size": (int, type(None)), "source_ref": _OPT_STR,
                      "source_quote": _OPT_STR, "spec_deviation": _OPT_STR, "uncertainty_reason": _OPT_STR,
                      "vendor_description": _OPT_STR, "bbox": (list, type(None))}


def _typed(v, t) -> bool:
    return isinstance(v, t) and not (isinstance(v, bool) and bool not in (t if isinstance(t, tuple) else (t,)))


def demo_run_shape_ok(run) -> bool:
    """Structure check for the demo run: top level, every vendor's extraction, every line and questionnaire answer."""
    if not (isinstance(run, dict) and all(_typed(run.get(k), t) for k, t in DEMO_RUN_KEYS.items()) and run["extractions"]):
        return False
    for ex in run["extractions"].values():
        if not (isinstance(ex, dict) and all(_typed(ex.get(k), t) for k, t in DEMO_VENDOR_KEYS.items()) and ex["lines"]):
            return False
        for ln in ex["lines"]:
            if not (isinstance(ln, dict) and all(k in ln and _typed(ln[k], t) for k, t in DEMO_LINE_REQUIRED.items())
                    and all(_typed(ln[k], t) for k, t in DEMO_LINE_OPTIONAL.items() if k in ln)):
                return False
        if not all(isinstance(q, dict) and isinstance(q.get("q_id"), str) for q in ex["questionnaire"]):
            return False
    return True


def demo_run_from_secrets(secrets=None, expected_sha256: str = DEMO_RUN_SHA256) -> tuple[dict | None, str | None]:
    """Return (run, problem). The run is returned only after decompression, fingerprint check and schema check,
    and is never changed or written to disk. `problem` is a fixed message that never quotes the stored data;
    (None, None) means no demo run is configured."""
    try:
        sec = st.secrets if secrets is None else secrets
        blob = (sec.get("saved_run") or {}).get("gz_b64")
    except Exception:
        return None, None                      # no secrets configured at all (local runs, tests)
    if not blob:
        return None, None
    try:
        raw = gzip.decompress(base64.b64decode(str(blob).strip(), validate=True))
    except Exception:
        return None, "could not be decoded"
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        return None, "failed its integrity check"
    try:
        run = json.loads(raw)
    except ValueError:
        return None, "is not valid JSON"
    if not demo_run_shape_ok(run):
        return None, "does not have the expected structure"
    return run, None


def page_send():
    st.markdown("# Send and collect")
    if not S.rfx:
        st.info("Issue an RFx first."); return
    if not S.sent:
        st.markdown('<p class="qd-lede">Each vendor gets the RFx by email with the line items as an Excel template. '
                    'They are free to ignore it, and most will. Sending is stubbed: no email leaves this app.</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(S.rfx["vendors"]), hide_index=True, width="stretch")
        with st.expander("Invitation email"):
            st.text(f"Subject: {S.rfx['rfx_id']} - {S.rfx['title']}\n\nDear supplier,\n\nPlease quote for the "
                    f"{len(S.rfx['line_items'])} items in the attached template, {S.rfx['terms']['price_basis']}. "
                    f"Answer the {len(S.rfx['questionnaire'])}-question quality questionnaire and attach certificates. "
                    f"Quotes are due {S.rfx['terms']['quote_due']}.\n\nRegards,\n{BUYER['name']}")
        if st.button("Send to all vendors (stubbed)", type="primary"):
            S.sent = True; log("RFx sent", "Stubbed email to all vendors"); st.rerun()
        return
    if not S.inbox:
        st.success("Sent. Nine days later, the replies are in.")
        if st.button("Open the inbox", type="primary"):
            S.inbox = pipeline.sample_inbox(); st.rerun()
        return
    st.markdown('<p class="qd-lede">What actually came back. Open any reply to see it exactly as the vendor sent it.</p>',
                unsafe_allow_html=True)
    names = {v["vendor_id"]: v["name"] for v in S.rfx["vendors"]}
    for item in S.inbox:
        kind = ingest.load(item["file"], item["bytes"]).kind if not item["file"].lower().endswith((".jpg", ".png")) else "image"
        with st.expander(f"{names.get(item['vendor_id'], item['vendor_id'])}  ·  {item['file']}  ·  {kind}"
                         + (f"  ·  attachment: {item['cert']}" if item.get("cert") else "")):
            preview_doc(item)
    with st.expander("Add a reply of your own (try your own quote file)"):
        up = st.file_uploader("Excel, PDF, Word, photo or text", type=["xlsx", "pdf", "docx", "jpg", "jpeg", "png", "txt", "eml"])
        vname = st.text_input("Vendor name", value="New vendor")
        cert = st.file_uploader("Certificate (optional)", type=["pdf", "jpg", "png"], key="cert_up")
        if up and st.button("Add to inbox"):
            vid = f"V{len(S.rfx['vendors']) + 1}"
            S.rfx["vendors"].append({"vendor_id": vid, "name": vname, "email": ""})
            S.inbox.append({"vendor_id": vid, "file": up.name, "bytes": up.getvalue(),
                            "cert": cert.name if cert else None, "cert_bytes": cert.getvalue() if cert else None})
            log("Reply added", f"{vname}: {up.name}"); st.rerun()
    st.divider()
    if S.get("run_errors"):
        st.error("The last AI read had problems:\n\n" + "\n".join(f"- {e}" for e in S.run_errors))
    # Verified demo run first. A live read still opens directly and is saved to last_run.json,
    # but it never replaces the demo run offered below.
    demo, demo_problem = demo_run_from_secrets()
    cached = demo or pipeline.load_cache()
    cached_label = DEMO_RUN_LABEL if demo else (f"Last AI read on this server, {cached['ran_at']}" if cached else None)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Read every reply with AI")
        st.caption(f"{len(S.inbox)} files plus certificates. Roughly one model call per file; about a minute on the free tier.")
        if st.button("Read all replies now", type="primary", disabled=not AI_ON):
            bar = st.progress(0.0, "Starting")
            try:
                run = pipeline.process(S.rfx, S.inbox, progress=lambda p, t: bar.progress(min(p, 1.0), t))
                S.run_errors = run["errors"]
                log("Replies read", f"{len(run['extractions'])} extracted, {len(run['errors'])} errors")
                if run["extractions"]:
                    S.run = run; S.decisions = {}; S.chat = []
                    pipeline.save_cache(run)
                    go("Compare")
                else:
                    st.rerun()
            except llm.LLMUnavailable as e:
                st.error(f"The model could not be reached: {e}")
    with c2:
        if cached and set(cached.get("extractions", {})) >= {i["vendor_id"] for i in S.inbox}:
            st.markdown("### Or reuse the last AI read")
            st.markdown(f"**{cached_label}**")
            st.caption("A real AI read, saved so the demo doesn't depend on free-tier rate limits. "
                       "Same output format; you can re-read live any time.")
            if st.button("Load last AI read"):
                S.run = cached; S.decisions = {}; log("Replies loaded", cached_label)
                go("Compare")
        if demo_problem:
            st.caption(f"Saved demo run unavailable: it {demo_problem}.")


GRID_MISSING = "-"   # what a missing price looks like on screen; the data itself stays NaN


def matrix_display(df: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numeric grid (NaN for missing) plus a matching CSS grid. Keeping the values numeric lets the grid
    sort and right-align them as numbers; the '-' is applied at display time only (Styler na_rep and
    st.dataframe placeholder), so calculations, analysis and exports never see it.
    Streamlit's grid renders background colour, text colour and bold from the CSS; it ignores italics
    and borders, so neither is used."""
    piv = df.pivot_table(index=["line_id", "item"], columns="vendor", values=value_col, aggfunc="first", dropna=False)
    piv = piv.reindex(columns=sorted(df.vendor.unique()))
    piv = piv.reindex(pd.MultiIndex.from_frame(df[["line_id", "item"]].drop_duplicates().sort_values("line_id")))
    grid = piv.apply(pd.to_numeric, errors="coerce").astype("float64")
    status = {(r.line_id, r.vendor): r.status for r in df[["line_id", "vendor", "status"]].itertuples()}
    usable_min = df[df.usable].groupby("line_id")[value_col].min().to_dict()
    css = pd.DataFrame("", index=grid.index, columns=grid.columns)
    for (lid, item) in grid.index:
        for v in grid.columns:
            val = grid.at[(lid, item), v]
            s_ = status.get((lid, v))
            c = f"background-color:{STATUS_COLOR.get(s_, '#FFFFFF')};"
            if s_ in ("OK", "CONFIRMED") and pd.notna(val) and abs(val - usable_min.get(lid, -1)) < 1e-6:
                c += "font-weight:700;"
            if s_ == "REVIEW":
                c += "color:#6B4A0B;"
            if pd.isna(val):
                c += "color:#8A8F8C;"
            css.at[(lid, item), v] = c
    return grid, css


def style_matrix(df: pd.DataFrame, value_col: str):
    grid, css = matrix_display(df, value_col)
    return grid.style.format("{:,.2f}", na_rep=GRID_MISSING).apply(lambda _: css, axis=None)


def cell_inspector(df: pd.DataFrame):
    st.markdown("### Where did this number come from?")
    c1, c2 = st.columns(2)
    lid = c1.selectbox("Line", sorted(df.line_id.unique()), format_func=lambda x: f"{x} · {df[df.line_id == x].item.iloc[0]}")
    vend = c2.selectbox("Vendor", sorted(df.vendor.unique()))
    r = df[(df.line_id == lid) & (df.vendor == vend)]
    if r.empty:
        st.info("No entry"); return
    r = r.iloc[0]
    chip = {"OK": "ok", "CONFIRMED": "ok", "REVIEW": "review"}.get(r.status, "missing")
    st.markdown(f'<span class="qd-chip {chip}">{r.status}</span><span class="qd-chip info">reader confidence {r.confidence:.0%}</span>',
                unsafe_allow_html=True)
    a, b = st.columns([1.2, 1])
    with a:
        st.markdown(f"**Landed price:** {inr(r.landed_price_inr)} per piece  ·  **Unit price:** {inr(r.unit_price_inr)}")
        st.markdown(f"**As quoted:** {r.quoted_value if r.quoted_value is not None else '-'} {r.quoted_currency or ''}, basis `{r.quoted_basis}`")
        if r["method"]:
            st.markdown("**Conversion:** " + " → ".join(r["method"]))
        st.markdown(f"**Source:** {r.source_file}, {r.source_ref or 'location not given'}")
        if r.source_quote:
            st.markdown(f'<div class="qd-quote">{r.source_quote}</div>', unsafe_allow_html=True)
        for f in r["flags"]:
            st.markdown(f"- {f}")
    with b:
        item = next((i for i in S.inbox if i["file"] == r.source_file), None)
        if item and r.bbox and item["file"].lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                im = Image.open(io.BytesIO(item["bytes"]))
                y0, x0, y1, x1 = r.bbox
                W, H = im.size
                pad = 60
                box = (max(0, x0 * W / 1000 - pad * 4), max(0, y0 * H / 1000 - pad), min(W, x1 * W / 1000 + pad), min(H, y1 * H / 1000 + pad))
                st.image(im.crop(box), caption="Where the reader saw it on the photo", width="stretch")
            except Exception:
                pass
        elif r.vendor_description:
            st.markdown(f"**Vendor's description:** {r.vendor_description}")


def page_compare():
    st.markdown("# Compare")
    res = compute()
    if res is None:
        st.info("Read the replies first (Send and collect)."); return
    df, vdf, adf, ddf = res
    if S.get("run_errors"):
        with st.expander(f"{len(S.run_errors)} step(s) failed during the AI read"):
            for e in S.run_errors:
                st.markdown(f"- {e}")
    counts = df.status.value_counts().to_dict()
    total = len(df)
    c = st.columns(5)
    c[0].markdown(f'<div class="qd-stat"><b>{counts.get("OK", 0) + counts.get("CONFIRMED", 0)}</b><span>usable cells of {total}</span></div>', unsafe_allow_html=True)
    c[1].markdown(f'<div class="qd-stat"><b>{counts.get("REVIEW", 0)}</b><span>need your review</span></div>', unsafe_allow_html=True)
    c[2].markdown(f'<div class="qd-stat"><b>{counts.get("MISSING", 0) + counts.get("REJECTED", 0)}</b><span>missing or rejected</span></div>', unsafe_allow_html=True)
    c[3].markdown(f'<div class="qd-stat"><b>{int(vdf.cleared_quality.sum())} of {len(vdf)}</b><span>vendors cleared the quality gate</span></div>', unsafe_allow_html=True)
    c[4].markdown(f'<div class="qd-stat"><b>{len(ddf)}</b><span>conditional discounts found (not applied)</span></div>', unsafe_allow_html=True)

    st.markdown("### All vendors, same lines, same unit: ₹ per piece ex-GST")
    basis = st.segmented_control("Price shown", ["Landed (incl. freight)", "Unit (before freight)"], default="Landed (incl. freight)")
    col = "landed_price_inr" if basis != "Unit (before freight)" else "unit_price_inr"
    st.markdown('<span class="qd-chip ok">usable</span><span class="qd-chip review">needs review (not used in analysis)</span>'
                '<span class="qd-chip absent">missing (-)</span><span class="qd-chip missing">rejected</span>'
                '<span class="qd-small">Bold: lowest usable price on the line.</span>',
                unsafe_allow_html=True)
    st.dataframe(style_matrix(df, col), width="stretch", height=560, placeholder=GRID_MISSING)
    st.download_button("Export comparison (CSV)", df.drop(columns=["bbox"]).to_csv(index=False), "comparison.csv")
    st.divider()
    cell_inspector(df)
    st.divider()
    st.markdown("### Quality gate by vendor")
    st.caption("Gating questions decide who can be awarded. Certificates are read and their dates checked against the bid date; "
               "a vendor's 'yes, certified' does not override an expired certificate.")
    show = vdf[["vendor", "cleared_quality", "q1_iso", "q2_lab", "q4_lead", "freight", "payment_terms", "lead_time"]].rename(
        columns={"cleared_quality": "cleared", "q1_iso": "Q1 ISO", "q2_lab": "Q2 in-house lab", "q4_lead": "Q4 lead time"})
    st.dataframe(show, hide_index=True, width="stretch")
    for _, v in vdf.iterrows():
        with st.expander(f"{v.vendor}: {'cleared' if v.cleared_quality else 'not cleared'}"):
            for part in v.gate_summary.split("; "):
                st.markdown(f"- {part}")
            cert = S.run["certificates"].get(v.vendor_id)
            if cert:
                st.markdown(f"Certificate read: **{cert.get('standard')}**, no. {cert.get('certificate_no')}, valid until "
                            f"**{cert.get('valid_until')}** → {'valid' if cert.get('valid_on_bid_date') else 'NOT valid'} on bid date")
            st.dataframe(adf[adf.vendor_id == v.vendor_id][["q_id", "answer"]], hide_index=True, width="stretch")
            gq = st.selectbox("Override a gate result (logged)", ["-", "Q1", "Q2", "Q4"], key=f"gq_{v.vendor_id}")
            if gq != "-":
                gr = st.radio("Result", ["pass", "fail", "unknown"], horizontal=True, key=f"gr_{v.vendor_id}")
                note = st.text_input("Reason", key=f"gn_{v.vendor_id}")
                if st.button("Save override", key=f"gs_{v.vendor_id}") and note:
                    S.gate_overrides.setdefault(v.vendor_id, {})[gq] = {"result": gr, "reason": f"Buyer override: {note}"}
                    log("Gate override", f"{v.vendor} {gq} → {gr}: {note}"); st.rerun()


def page_review():
    st.markdown("# Review queue")
    res = compute()
    if res is None:
        st.info("Read the replies first."); return
    df, vdf, adf, ddf = res
    q = df[df.status == "REVIEW"].copy()
    q["value_at_stake"] = q.landed_price_inr * q.annual_qty
    q = q.sort_values("value_at_stake", ascending=False)
    st.markdown(f'<p class="qd-lede">{len(q)} prices are held back from analysis until you decide. Sorted by annual value at stake, '
                'so the decisions that matter most come first. Accept, correct the number, or reject it.</p>', unsafe_allow_html=True)
    if q.empty:
        st.success("Nothing waiting for review.")
    else:
        # Same reason, same vendor: decide once. Only cells the reader was sure about are eligible;
        # ambiguous, low-confidence, deviating or outlier cells always need an individual decision.
        groups = review.bulk_groups(q)
        if len(groups):
            st.markdown("### Decide in bulk")
            for _, g in groups.iterrows():
                cc = st.columns([2.4, 1.6])
                held = f" {g.held_back} similar cells need individual decisions and are not included." if g.held_back else ""
                est = review.uses_estimated_weight(g.reason_key)
                cc[0].markdown(f"**{g.vendor}**: {g.n} cells, *{g.reason_key}*. {', '.join(g.line_ids)}.{held}")
                if est:
                    cc[0].caption("Each price is the vendor's per-kg rate × our estimated box weight from the RFx spec. "
                                  "The vendor has not supplied or confirmed these weights. Accepting uses our estimate as-is; "
                                  "ask the vendor for actual box weights before award on lines that matter.")
                label = f"Accept {g.n} at est. weights" if est else f"Accept these {g.n}"
                tip = ("Accepts these prices as converted with our estimated box weights. "
                       "The vendor has not supplied or confirmed the weights.") if est else None
                if cc[1].button(label, key=f"bulk_{g.vendor_id}_{abs(hash(g.reason_key))}", help=tip):
                    review.bulk_accept(q, g.vendor_id, g.reason_key, S.decisions, log)
                    st.rerun()
            st.markdown("### One by one")
        q = review.annotate(q)
    for _, r in q.head(40).iterrows():
        with st.container(border=True):
            a, b, c = st.columns([2.2, 1.2, 1.6])
            a.markdown(f"**{r.line_id} · {r['item']}**  \n{r.vendor}")
            a.markdown("<br>".join(f"<span class='qd-small'>• {f}</span>" for f in r["flags"]), unsafe_allow_html=True)
            if review.blocked(r.get("bulk_block")) and r["bulk_block"] != "not awaiting review":
                a.markdown(f"<span class='qd-chip review'>Needs your individual decision: {r['bulk_block']}</span>", unsafe_allow_html=True)
            b.markdown(f"**{inr(r.landed_price_inr)}** /pc  \n<span class='qd-small'>{lakh(r.value_at_stake) if pd.notna(r.value_at_stake) else '-'} a year</span>",
                       unsafe_allow_html=True)
            if r.source_quote:
                b.caption(f"“{str(r.source_quote)[:110]}”")
            key = f"{r.line_id}_{r.vendor_id}"
            act = c.radio("Decision", ["Accept", "Correct", "Reject"], horizontal=True, key=f"a_{key}", label_visibility="collapsed")
            val = c.number_input("Correct price ₹/pc (before freight)", value=float(r.unit_price_inr or 0), key=f"v_{key}",
                                 disabled=act != "Correct")
            note = c.text_input("Note", key=f"n_{key}", placeholder="Why (goes in the audit log)")
            if c.button("Save", key=f"s_{key}"):
                action = {"Accept": "accept", "Correct": "override", "Reject": "reject"}[act]
                S.decisions[(r.line_id, r.vendor_id)] = {"action": action, "value": val, "note": note, "via": "individual"}
                # Accepting a per-kg price accepts our weight estimate: disclose it, same wording as bulk accept.
                est = action == "accept" and any(review.uses_estimated_weight(f) for f in (r["flags"] or []))
                log(f"Cell {action}", f"{r.line_id}/{r.vendor}: {note or '(no note)'}" + (f" → ₹{val:.2f}" if action == "override" else "")
                    + f"; was {inr(r.unit_price_inr)}" + (f" ({review.ESTIMATED_WEIGHT_NOTE})" if est else "")
                    + f"; source {r.source_file}, {r.source_ref}: \"{str(r.source_quote or '')[:80]}\"")
                st.rerun()
    st.divider()
    st.markdown("### Ask the vendor instead of guessing")
    st.caption("For gaps and ambiguities, the fastest fix is usually a targeted question to the vendor. Drafts below are written by "
               "the model from the open issues; sending is stubbed.")
    open_df = df[df.status.isin(["REVIEW", "MISSING"])]
    for vid, g in open_df.groupby("vendor_id"):
        vname = g.vendor.iloc[0]
        issues = [f"{r.line_id} {r['item']}: {'; '.join(r['flags'])}" for _, r in g.iterrows()]
        with st.expander(f"{vname}: {len(issues)} open issues"):
            st.markdown("\n".join(f"- {i}" for i in issues[:15]))
            if st.button("Draft clarification email", key=f"cl_{vid}", disabled=not AI_ON):
                try:
                    S.clar[vid] = copilot.clarification_email(BUYER, {"name": vname}, S.rfx["rfx_id"], issues)
                    log("Clarification drafted", vname)
                except llm.LLMUnavailable as e:
                    st.error(str(e))
            if vid in S.clar:
                st.text_input("Subject", S.clar[vid].get("subject", ""), key=f"cs_{vid}")
                st.text_area("Body", S.clar[vid].get("body", ""), height=240, key=f"cb_{vid}")


SUGGESTED = [
    "What if we split it, cheapest per line, but only among vendors who cleared the quality questionnaire?",
    "How does that split compare with giving everything to one cleared vendor?",
    "Which quoted prices are uncertain, incomplete, or missing freight?",
    "How much more does the quality gate cost us versus the cheapest price from anyone?",
    "Where are we paying more than last year?",
]


def page_ask():
    st.markdown("# Ask the data")
    res = compute()
    if res is None:
        st.info("Read the replies first."); return
    df, vdf, adf, ddf = res
    st.markdown('<p class="qd-lede">Ask in plain language. The model writes a query, the database computes the answer, and the '
                'model explains it. Only usable prices are counted; each answer says what it left out.</p>', unsafe_allow_html=True)
    con = analyst.build_db(df, S.rfx, vdf, adf, ddf, S.history)
    fp = analyst.facts(df, vdf, ddf)
    for i, m in enumerate(S.chat):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("chart") and m.get("table") is not None:
                ch = m["chart"]
                try:
                    t = pd.DataFrame(m["table"])
                    if ch.get("type") in ("bar", "line", "scatter") and ch.get("x") in t and ch.get("y") in t:
                        fn = {"bar": px.bar, "line": px.line, "scatter": px.scatter}[ch["type"]]
                        fig = fn(t, x=ch["x"], y=ch["y"], color=ch.get("color") if ch.get("color") in t else None,
                                 color_discrete_sequence=["#9A6631", "#2F6B5A", "#46555B", "#C98A12", "#7B3F3F"])
                        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#FFFFFF",
                                          paper_bgcolor="rgba(0,0,0,0)", font_family="IBM Plex Sans")
                        st.plotly_chart(fig, width="stretch", key=f"ch_{i}")
                except Exception:
                    pass
            if m.get("table") is not None:
                t = pd.DataFrame(m["table"])
                with st.expander(f"Data behind this answer ({len(t)} rows) and how it was computed"):
                    st.dataframe(t, hide_index=True, width="stretch")
                    st.code(m.get("sql") or "", language="sql")
                    st.download_button("Export (CSV)", t.to_csv(index=False), f"answer_{i}.csv", key=f"ex_{i}")
    if not S.chat:
        st.caption("Try one of these, or ask your own:")
    picked = None
    cols = st.columns(len(SUGGESTED))
    for c, sq in zip(cols, SUGGESTED):
        if c.button(sq, key=f"sq_{sq[:20]}", width="stretch"):
            picked = sq
    q = st.chat_input("Ask about prices, vendors, gaps, scenarios") or picked
    if q:
        if not AI_ON:
            st.error("The analyst needs the AI reader (Gemini key)."); return
        S.chat.append({"role": "user", "content": q})
        with st.spinner("Working it out"):
            try:
                out = analyst.ask(q, con, S.chat[:-1], fp)
                S.chat.append({"role": "assistant", "content": out["answer"], "sql": out.get("sql"),
                               "table": out["table"].to_dict("records") if out.get("table") is not None else None,
                               "chart": out.get("chart")})
                log("Question asked", q)
            except Exception as e:
                S.chat.append({"role": "assistant", "content": f"That failed: {e}. Try rephrasing."})
        st.rerun()


def page_award():
    st.markdown("# Award")
    res = compute()
    if res is None:
        st.info("Read the replies first."); return
    df, vdf, adf, ddf = res
    st.markdown('<p class="qd-lede">Scenarios are computed from usable prices only. Pick one, check what is still open, '
                'and export the award pack with every number traceable to its source.</p>', unsafe_allow_html=True)
    apply_disc = st.toggle("Apply conditional discounts where the award meets the vendor's condition", value=False)
    disc = normalize.discounts(S.run["extractions"])
    scen = award.scenarios(df, vdf, S.history, S.rfx, disc, apply_disc)
    cols = st.columns(len(scen))
    for c, s in zip(cols, scen):
        with c.container(border=True):
            st.markdown(f"**{s['name']}**")
            st.markdown(f'<div class="qd-stat"><b>{lakh(s["total_inr"])}</b><span>{s["lines_covered"]} of {len(S.rfx["line_items"])} lines covered</span></div>',
                        unsafe_allow_html=True)
            if s["saving_vs_last_year_pct"] is not None:
                st.caption(f"{s['saving_vs_last_year_pct']:+.1f}% vs last year on the same lines")
            st.caption(s["note"])
            for d in s["discounts_applied"]:
                st.caption(f"Discount applied: {d}")
    names = [s["name"] for s in scen]
    pick = st.selectbox("Inspect scenario", names)
    s = scen[names.index(pick)]
    a, b = st.columns([1.3, 1])
    with a:
        p = s["picks"].copy()
        st.dataframe(p.rename(columns={"landed_price_inr": "₹/pc landed", "annual_value_inr": "annual ₹"}),
                     hide_index=True, width="stretch", height=420)
    with b:
        sp = pd.DataFrame(s["split"])
        if len(sp):
            fig = px.bar(sp, x="value", y="vendor", orientation="h", text="lines", color_discrete_sequence=["#9A6631"])
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="annual value ₹", yaxis_title="",
                              plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        if s["lines_uncovered"]:
            st.warning(f"Not covered by usable prices: {', '.join(s['lines_uncovered'])}. Resolve in the review queue or ask the vendors.")
    open_items = {
        "unconfirmed_cells": int((df.status == "REVIEW").sum()),
        "missing_cells": int(df.status.isin(["MISSING", "REJECTED"]).sum()),
        "conditional_discounts": disc, "vendors_not_cleared": vdf[~vdf.cleared_quality].vendor.tolist(),
        "freight_assumption_pct": S.assume.freight_pct_when_extra, "usd_inr": S.assume.usd_inr,
    }
    if st.button("Draft the award recommendation memo", disabled=not AI_ON):
        payload = {"SCENARIOS": [{k: v for k, v in x.items() if k != "picks"} for x in scen], "OPEN_ITEMS": open_items,
                   "rfx": S.rfx["title"]}
        try:
            S.memo = copilot.award_memo(payload); log("Memo drafted", pick)
        except llm.LLMUnavailable as e:
            st.error(str(e))
    if S.memo:
        st.markdown(S.memo)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        s["picks"].to_excel(xw, sheet_name="Award (selected)", index=False)
        out = df.drop(columns=["bbox"]).copy()
        out["flags"] = out["flags"].apply("; ".join); out["method"] = out["method"].apply(" → ".join)
        out.to_excel(xw, sheet_name="All prices with sources", index=False)
        vdf.to_excel(xw, sheet_name="Vendor quality", index=False)
        pd.DataFrame([vars(S.assume)]).to_excel(xw, sheet_name="Assumptions", index=False)
        pd.DataFrame(S.audit).to_excel(xw, sheet_name="Audit log", index=False)
        pd.DataFrame([vars(t) for t in llm.trace()]).to_excel(xw, sheet_name="AI calls", index=False)
    st.download_button("Export award pack (Excel)", buf.getvalue(), "award_pack.xlsx", type="primary")


def page_audit():
    st.markdown("# Assumptions and audit")
    st.markdown("### Assumptions (edit and everything recomputes)")
    a, b, c = st.columns(3)
    usd = a.number_input("USD → INR", value=float(S.assume.usd_inr), step=0.25)
    fr = b.number_input("Freight added when vendor says 'freight extra' (%)", value=float(S.assume.freight_pct_when_extra), step=0.5)
    c.markdown("<br>", unsafe_allow_html=True)
    if c.button("Apply"):
        S.assume.usd_inr, S.assume.freight_pct_when_extra = usd, fr
        log("Assumptions changed", f"USD {usd}, freight {fr}%"); st.rerun()
    st.markdown("### Decisions log")
    st.dataframe(pd.DataFrame(S.audit), hide_index=True, width="stretch")
    st.markdown("### Every AI call this session")
    st.caption("Task, model, latency, and the start of the prompt and output. Nothing on screen is hardcoded.")
    st.dataframe(pd.DataFrame([vars(t) for t in llm.trace()]), hide_index=True, width="stretch")
    res = compute()
    if res is not None:
        df = res[0]
        st.markdown("### How good is the reader? Check against hand-verified values")
        st.caption("This dataset is fabricated, so we know the right answer for every cell. This compares the system's ₹/pc "
                   "(before freight, before review) with it. A miss within 1% counts as correct.")
        gt = pipeline.load_ground_truth()
        rows = []
        for _, r in df.iterrows():
            exp = gt["lines"].get(r.line_id, {}).get(r.vendor_id)
            got = r.unit_price_inr
            if exp is None and (got is None or pd.isna(got)):
                verdict = "correct (no price)"
            elif exp is None:
                verdict = "should be empty"
            elif got is None or pd.isna(got):
                verdict = "missed"
            else:
                verdict = "correct" if abs(got - exp) / exp <= 0.01 else "wrong"
            rows.append({"line_id": r.line_id, "vendor": r.vendor, "expected": exp, "got": got, "status": r.status, "verdict": verdict})
        ev = pd.DataFrame(rows)
        good = ev.verdict.str.startswith("correct").mean()
        wrong_ok = ev[(ev.verdict.isin(["wrong", "should be empty"])) & (ev.status.isin(["OK", "CONFIRMED"]))]
        x, y = st.columns(2)
        x.markdown(f'<div class="qd-stat"><b>{good:.0%}</b><span>cells correct before any human review</span></div>', unsafe_allow_html=True)
        y.markdown(f'<div class="qd-stat"><b>{len(wrong_ok)}</b><span>wrong values that slipped through as usable (the number that matters)</span></div>',
                   unsafe_allow_html=True)
        st.dataframe(ev[~ev.verdict.str.startswith("correct")], hide_index=True, width="stretch")


{"Start here": page_start, "Draft the RFx": page_draft, "Send and collect": page_send, "Compare": page_compare,
 "Review queue": page_review, "Ask the data": page_ask, "Award": page_award, "Assumptions and audit": page_audit}[page]()
