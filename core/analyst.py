"""Ask-the-comparison analyst.

The model plans, the database computes, the model explains. It never does arithmetic on
numbers it made up: every figure in an answer comes from a SQL result the buyer can open.
By default analysis only sees usable cells (status OK or CONFIRMED); the answer must say
what was excluded and why.
"""
from __future__ import annotations

import json
import re
import duckdb
import pandas as pd
from . import llm

SCHEMA_DOC = """
Tables (DuckDB):

quotes  -- one row per (line, vendor). ONLY usable prices (status OK or CONFIRMED).
  line_id VARCHAR, item VARCHAR, ply INT, annual_qty INT, vendor_id VARCHAR, vendor VARCHAR,
  unit_price_inr DOUBLE  -- INR per piece ex-GST, before freight
  landed_price_inr DOUBLE -- INR per piece ex-GST, including freight (use this for comparisons and awards)
  annual_value_inr DOUBLE -- landed_price_inr * annual_qty
  status VARCHAR, confidence DOUBLE (reader confidence 0-1),
  source_file VARCHAR, source_ref VARCHAR, source_quote VARCHAR  -- where in the vendor's document the price came from

quotes_all -- same columns for EVERY (line, vendor) pair plus:
  status VARCHAR ('OK','CONFIRMED','REVIEW','MISSING','REJECTED'), flags VARCHAR (semicolon-separated reasons)

lines -- the RFx: line_id, item, kind, size_mm, ply, flute, bf, printing, annual_qty, est_weight_kg

vendors -- vendor_id, vendor, cleared_quality BOOLEAN (passed every gating question),
  gate_summary VARCHAR, freight VARCHAR, payment_terms VARCHAR, lead_time VARCHAR, validity VARCHAR,
  q1_iso VARCHAR, q2_lab VARCHAR, q4_lead VARCHAR  -- gate results 'pass'/'fail'/'unknown'

answers -- vendor_id, q_id, question, answer   (raw questionnaire answers)

discounts -- vendor_id, description, percent DOUBLE, condition VARCHAR   (NOT applied to prices; conditional)

last_year -- line_id, awarded_supplier, unit_price_inr, annual_qty_ly
"""

PLANNER = """You are the analyst behind a procurement comparison tool. A category buyer asks questions in plain language.
Write ONE DuckDB SQL query (SELECT or WITH only) that answers the question from the tables below.
Guidance:
- Use quotes (usable prices only) for any ranking, cheapest, award or spend question. Use landed_price_inr.
- "Cheapest per line" / "split award": for each line pick the vendor with minimum landed_price_inr (use QUALIFY or ROW_NUMBER).
- "Vendors who cleared the quality questionnaire" means vendors.cleared_quality = TRUE.
- If the question is about coverage, gaps, data quality or why something is missing, use quotes_all.
- Filter vendors by vendor_id (see VENDORS below), never by an exact or partial vendor name.
- A question about one line or vendor that is not yet usable must use quotes_all, so REVIEW/MISSING cells are visible.
- Always include line_id, vendor_id and vendor columns when rows are per line (the app attaches source evidence by them).
  Round money to 2 decimals.
- Keep result to at most 60 rows; aggregate when sensible.
- If the question cannot be answered from the data, set sql to null and explain in "cannot_answer".
Return JSON: {"sql": str|null, "chart": {"type": "bar"|"line"|"scatter"|"none", "x": str|null, "y": str|null, "color": str|null},
"title": str, "cannot_answer": str|null}
""" + SCHEMA_DOC

NARRATOR = """You explain query results to a category buyer who has ₹ crores at stake.
Rules: use only numbers present in RESULT, TOTALS or FACTS (TOTALS are column totals and per-vendor subtotals computed by
the database layer); never compute new totals, differences or percentages in your head;
if you need a number that is not there, say what you would need. Lead with the answer in one or two sentences, then at
most 4 short bullets of supporting detail. Do not list exclusions or assumptions: the app adds those itself from the
data, so do not restate or contradict them. Indian number formatting (lakh/crore) for totals. Plain text with simple
markdown, no tables (the table is shown separately)."""


def build_db(df: pd.DataFrame, rfx: dict, vendors_df: pd.DataFrame, answers_df: pd.DataFrame,
             discounts_df: pd.DataFrame, history: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    cols = ["line_id", "item", "ply", "annual_qty", "vendor_id", "vendor", "unit_price_inr", "landed_price_inr", "annual_value_inr",
            "status", "confidence", "source_file", "source_ref", "source_quote"]
    allq = df.copy()
    allq["flags"] = allq["flags"].apply(lambda f: "; ".join(f) if isinstance(f, list) else "")
    q_usable = allq[allq.usable][cols].copy()
    q_all = allq[cols + ["flags"]].copy()
    lines = pd.DataFrame(rfx["line_items"])[["line_id", "item", "kind", "size_mm", "ply", "flute", "bf", "printing", "annual_qty", "est_weight_kg"]]
    con.register("quotes_df", q_usable); con.execute("CREATE TABLE quotes AS SELECT * FROM quotes_df")
    con.register("quotes_all_df", q_all); con.execute("CREATE TABLE quotes_all AS SELECT * FROM quotes_all_df")
    con.register("lines_df", lines); con.execute("CREATE TABLE lines AS SELECT * FROM lines_df")
    con.register("vendors_df", vendors_df); con.execute("CREATE TABLE vendors AS SELECT * FROM vendors_df")
    con.register("answers_df", answers_df); con.execute("CREATE TABLE answers AS SELECT * FROM answers_df")
    dd = discounts_df if len(discounts_df) else pd.DataFrame(columns=["vendor_id", "description", "percent", "condition"])
    con.register("disc_df", dd[["vendor_id", "description", "percent", "condition"]]); con.execute("CREATE TABLE discounts AS SELECT * FROM disc_df")
    con.register("ly_df", history[["line_id", "awarded_supplier", "unit_price_inr", "annual_qty_ly"]])
    con.execute("CREATE TABLE last_year AS SELECT * FROM ly_df")
    return con


_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|copy|pragma|install|load|export)\b", re.I)


def run_sql(con, sql: str) -> pd.DataFrame:
    s = sql.strip().rstrip(";")
    if not re.match(r"^\s*(select|with)\b", s, re.I) or _FORBIDDEN.search(s) or ";" in s:
        raise ValueError("Only single read-only SELECT queries are allowed")
    return con.execute(s).df()


def facts(df: pd.DataFrame, vendors_df: pd.DataFrame, discounts_df: pd.DataFrame) -> dict:
    by_status = df.groupby(["vendor", "status"]).size().unstack(fill_value=0).to_dict("index")
    review = df[df.status == "REVIEW"][["line_id", "vendor"]].values.tolist()
    missing = df[df.status.isin(["MISSING", "REJECTED"])][["line_id", "vendor"]].values.tolist()
    return {
        "cells_by_vendor_and_status": by_status,
        "unconfirmed_cells_excluded": [f"{a}/{b}" for a, b in review][:40],
        "missing_cells": [f"{a}/{b}" for a, b in missing][:40],
        "vendors_quality": vendors_df[["vendor", "cleared_quality", "gate_summary", "freight"]].to_dict("records"),
        "conditional_discounts_not_applied": discounts_df.to_dict("records") if len(discounts_df) else [],
    }


# ------------------------------------------------------------------ evidence, exclusions, number check (no model)

EVIDENCE_COLS = ["line_id", "item", "vendor", "status", "landed_price_inr", "confidence", "source_file", "source_ref", "source_quote"]
ESTIMATED_WEIGHT_FLAG = "Per-kg rate converted with our weight estimate"


def _pairs(result: pd.DataFrame, df: pd.DataFrame) -> list[tuple[str, str]] | None:
    """(line_id, vendor_id) pairs behind a per-line result, in result order; None if the result is not per line+vendor."""
    if result is None or "line_id" not in result.columns:
        return None
    if "vendor_id" in result.columns:
        vids = result["vendor_id"].astype(str).tolist()
    elif "vendor" in result.columns:
        name_to_id = dict(df[["vendor", "vendor_id"]].drop_duplicates().values)
        vids = [name_to_id.get(v) for v in result["vendor"]]
    else:
        return None
    seen, out = set(), []
    for lid, vid in zip(result["line_id"].astype(str), vids):
        if vid and (lid, vid) not in seen:
            seen.add((lid, vid)); out.append((lid, vid))
    return out


def _pairs_all(result: pd.DataFrame, df: pd.DataFrame) -> list[tuple[str, str]]:
    if "vendor_id" in result.columns:
        vids = result["vendor_id"].astype(str).tolist()
    else:
        name_to_id = dict(df[["vendor", "vendor_id"]].drop_duplicates().values)
        vids = [name_to_id.get(v) for v in result["vendor"]]
    return list(zip(result["line_id"].astype(str), vids))


def evidence_for(result: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """Source evidence for exactly the (line, vendor) rows in the result, looked up in the comparison itself."""
    pairs = _pairs(result, df)
    if pairs is None:
        return None, "This answer aggregates rows, so there is no single source per figure; ask for the per-line breakdown to see sources."
    idx = df.set_index(["line_id", "vendor_id"])
    rows = [idx.loc[p] for p in pairs if p in idx.index]
    if not rows:
        return None, "No matching cells in the comparison."
    ev = pd.DataFrame(rows).reset_index(drop=True)
    ev.insert(0, "line_id", [p[0] for p in pairs if p in idx.index])
    ev = ev[EVIDENCE_COLS].copy()
    # the evidence price must be the price the result used
    if "landed_price_inr" in result.columns:
        res_price: dict = {}
        for key, price in zip(_pairs_all(result, df), result["landed_price_inr"]):
            res_price.setdefault(key, price)
        keyed = [p for p in pairs if p in idx.index]
        ev["matches_result"] = [None if pd.isna(res_price.get(k)) else (pd.notna(v) and abs(float(res_price[k]) - float(v)) < 0.01)
                                for k, v in zip(keyed, ev["landed_price_inr"])]
    return ev, f"Source for each of the {len(ev)} line/vendor rows in this answer."


def _compact(ids: list[str]) -> str:
    """L15, L16, L17, L20 -> L15–L17, L20"""
    nums = sorted({int(i[1:]) for i in ids if i[1:].isdigit()})
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"L{nums[i]:02d}" if i == j else f"L{nums[i]:02d}–L{nums[j]:02d}")
        i = j + 1
    return ", ".join(out)


def scope(result: pd.DataFrame | None, df: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Lines and vendors an answer is about. Per-line answers: their lines, across every vendor (a held-back cell on
    those lines could change the answer). Vendor-only answers: those vendors, all lines. Otherwise: everything."""
    lines, vendors = set(df.line_id), set(df.vendor_id)
    if result is None or not len(result):
        return lines, vendors
    if "line_id" in result.columns:
        lines = set(result["line_id"].astype(str)) & lines
    elif "vendor_id" in result.columns or "vendor" in result.columns:
        if "vendor_id" in result.columns:
            vendors = set(result["vendor_id"].astype(str)) & vendors
        else:
            vendors = set(df[df.vendor.isin(result["vendor"])].vendor_id)
    return lines, vendors


def exclusion_notes(df: pd.DataFrame, result: pd.DataFrame | None, discounts_df: pd.DataFrame | None) -> list[str]:
    """What this answer did not count, and which of our assumptions are inside its numbers. Written by code, not the model."""
    lines, vendors = scope(result, df)
    sc = df[df.line_id.isin(lines) & df.vendor_id.isin(vendors)]
    notes = []
    labels = {"REVIEW": "awaiting buyer review, not counted", "MISSING": "no usable price (not quoted or unreadable), not counted",
              "REJECTED": "rejected by the buyer, not counted"}
    for vend, g in sc[~sc.usable].groupby("vendor", sort=True):
        for status in ("REVIEW", "MISSING", "REJECTED"):
            ids = g[g.status == status].line_id.tolist()
            if ids:
                notes.append(f"{vend}: {len(ids)} cell{'s' if len(ids) != 1 else ''} {labels[status]} ({_compact(ids)}).")
    pairs = _pairs(result, df) if result is not None else None
    used = sc[sc.set_index(["line_id", "vendor_id"]).index.isin(pairs)] if pairs else sc[sc.usable]
    for vend, g in used.groupby("vendor", sort=True):
        methods = [m for ms in g["method"] for m in (ms or [])]
        fr = sorted({m for m in methods if "freight (vendor: freight extra)" in m})
        if fr:
            pct = fr[0].split("%")[0].lstrip("+")
            notes.append(f"{vend}: landed prices include our {pct}% freight assumption. The vendor wrote 'freight extra' "
                         f"without a figure, so this is not a vendor-quoted freight cost.")
        fx = sorted({m for m in methods if m.startswith(("USD converted at", "EUR converted at"))})
        if fx:
            notes.append(f"{vend}: {fx[0].replace(' converted at', ' prices converted at')}, our editable exchange-rate assumption.")
        est = int(sum(any(str(f).startswith(ESTIMATED_WEIGHT_FLAG) for f in (fl or [])) for fl in g["flags"]))
        if est:
            notes.append(f"{vend}: {est} price{'s' if est != 1 else ''} here use our estimated box weights from the RFx spec; "
                         f"the vendor has not supplied or confirmed them.")
    if discounts_df is not None and len(discounts_df):
        names = dict(df[["vendor_id", "vendor"]].drop_duplicates().values)
        for d in discounts_df.to_dict("records"):
            if d.get("vendor_id") not in vendors:
                continue
            pct = d.get("percent")
            terms = [f"{pct:g}%"] if isinstance(pct, (int, float)) and pct == pct else []
            terms.append(str(d.get("condition") or "condition not stated"))
            notes.append(f"{names.get(d['vendor_id'], d['vendor_id'])}: conditional discount not applied ({', '.join(terms)}).")
    return notes


_FIG = re.compile(r"(?<![A-Za-z0-9.])(₹\s?)?(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(crores?|cr\b|lakhs?|lacs?|L\b|%)?")


def result_totals(result: pd.DataFrame | None) -> dict:
    """Totals computed in code from the full result, so the narrator never has to add numbers up itself."""
    if result is None or not len(result):
        return {}
    num = result.select_dtypes("number")
    num = num[[c for c in num.columns if re.search(r"annual|value|total|spend|qty|quantity|saving|count|lines", c, re.I)]]
    out: dict = {"rows": len(result), "column_totals": {c: round(float(num[c].sum()), 2) for c in num.columns}}
    key = "vendor" if "vendor" in result.columns else ("vendor_id" if "vendor_id" in result.columns else None)
    if key and len(num.columns):
        g = result.groupby(key)
        out["by_vendor"] = {str(k): {"rows": int(len(v)), **{c: round(float(v[c].sum()), 2) for c in num.columns}}
                            for k, v in g}
    return out


def _known_numbers(result: pd.DataFrame | None, fact_pack: dict | None, question: str) -> list[float]:
    vals: list[float] = []
    if result is not None:
        for v in result.select_dtypes("number").to_numpy().ravel().tolist():
            if v == v:
                vals.append(float(v))
        vals.append(float(len(result)))

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(k); walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, bool):
            return
        elif isinstance(x, (int, float)) and x == x:
            vals.append(float(x))
        elif isinstance(x, str):
            for m in re.findall(r"\d+(?:\.\d+)?", x.replace(",", "")):
                vals.append(float(m))
    walk(fact_pack or {})
    walk(result_totals(result))
    for m in re.findall(r"\d+(?:\.\d+)?", (question or "").replace(",", "")):
        vals.append(float(m))
    return vals


def check_numbers(text: str, result: pd.DataFrame | None, fact_pack: dict | None, question: str = "") -> tuple[list[str], list[str]]:
    """Money, decimal and percentage figures in the explanation, split into (found in the data, not found).
    Plain small integers without a unit (counts, line numbers, ply) are not checked."""
    known = _known_numbers(result, fact_pack, question)
    found, missing = [], []
    for m in _FIG.finditer(text or ""):
        rupee, num, unit = m.group(1), m.group(2), (m.group(3) or "").lower()
        if not (rupee or unit or "." in num or "," in num):
            continue
        x = float(num.replace(",", ""))
        d = len(num.split(".")[1]) if "." in num else 0
        tol = 0.5 * 10 ** -d + 1e-9
        if unit.startswith("cr"):
            cands = [k / 1e7 for k in known]
        elif unit.startswith(("lakh", "lac")) or unit == "l":
            cands = [k / 1e5 for k in known]
        elif unit == "%":
            cands = known + [k * 100 for k in known]
        else:
            cands = known
        (found if any(abs(c - x) <= tol for c in cands) else missing).append(m.group(0).strip())
    return found, missing


def ask(question: str, con, history_msgs: list[dict], fact_pack: dict, df: pd.DataFrame | None = None,
        discounts_df: pd.DataFrame | None = None) -> dict:
    """Plan (model) -> SQL (database) -> explain (model). Evidence, exclusions and the number check are computed in
    code, so they survive a failed explanation call."""
    out = {"answer": None, "sql": None, "table": None, "chart": None, "title": None, "evidence": None, "evidence_note": "",
           "notes": [], "figures_found": [], "figures_unverified": [], "explain_failed": False, "failed_step": None}
    convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history_msgs[-6:])
    try:
        vendors = "" if df is None else "\n".join(f"{i} = {n}" for i, n in df[["vendor_id", "vendor"]].drop_duplicates().values)
        plan = llm.generate("analyst:plan", PLANNER + f"\n\nVENDORS (vendor_id = vendor):\n{vendors}\n\nRecent conversation:\n{convo}\n\nQUESTION: {question}")
    except Exception as e:
        out.update(answer="The model could not be reached to plan this question, so nothing was computed. Try again shortly.",
                   failed_step="plan")
        return out
    plan = plan if isinstance(plan, dict) else {}
    result, sql, error = None, plan.get("sql"), None
    if sql:
        try:
            result = run_sql(con, sql)
        except Exception as e:
            error = str(e)[:300]
            try:
                retry = llm.generate("analyst:repair", PLANNER + f"\n\nQUESTION: {question}\nYour previous SQL:\n{sql}\nfailed with:\n{error}\nFix it.")
                retry = retry if isinstance(retry, dict) else {}
                sql = retry.get("sql")
                plan["chart"] = retry.get("chart", plan.get("chart"))
                if sql:
                    result = run_sql(con, sql)
                    error = None
            except Exception as e2:
                error = str(e2)[:300]
    out.update(sql=sql, title=plan.get("title"), chart=plan.get("chart"))
    if result is None:
        out["answer"] = plan.get("cannot_answer") or f"I couldn't answer that from the data. {error or ''}".strip()
        out["failed_step"] = None if plan.get("cannot_answer") else "query"
        return out
    out["table"] = result
    if not len(result):
        out["answer"] = ("The query matched no rows. That is not evidence that a vendor did not quote or that a value is missing: "
                         "the filter may be wrong. Check the SQL below, or rephrase with the line id and vendor.")
        out["failed_step"] = "empty"
        return out
    if df is not None:
        out["evidence"], out["evidence_note"] = evidence_for(result, df)
        out["notes"] = exclusion_notes(df, result, discounts_df)
    payload = {"question": question, "sql": sql, "row_count": len(result),
               "RESULT": json.loads(result.head(60).to_json(orient="records")), "TOTALS": result_totals(result),
               "FACTS": fact_pack}
    try:
        answer = llm.generate("analyst:explain", NARRATOR + "\n\n" + json.dumps(payload, default=str), as_json=False, temperature=0.2)
        out["answer"] = str(answer)
        out["figures_found"], out["figures_unverified"] = check_numbers(out["answer"], result, fact_pack, question)
    except Exception:
        out["explain_failed"] = True
    return out
