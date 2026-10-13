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
- Always include line_id and vendor columns when rows are per line. Round money to 2 decimals.
- Keep result to at most 60 rows; aggregate when sensible.
- If the question cannot be answered from the data, set sql to null and explain in "cannot_answer".
Return JSON: {"sql": str|null, "chart": {"type": "bar"|"line"|"scatter"|"none", "x": str|null, "y": str|null, "color": str|null},
"title": str, "cannot_answer": str|null}
""" + SCHEMA_DOC

NARRATOR = """You explain query results to a category buyer who has ₹ crores at stake.
Rules: use only numbers present in RESULT or FACTS; never compute new totals in your head; if you need a number that
is not there, say what you would need. Lead with the answer in one or two sentences, then at most 4 short bullets
of supporting detail. Then a line starting "Excluded:" naming any data-quality limits from FACTS that affect THIS
answer (unconfirmed cells, missing lines, conditional discounts not applied, freight assumptions). Indian number
formatting (lakh/crore) for totals. Plain text with simple markdown, no tables (the table is shown separately)."""


def build_db(df: pd.DataFrame, rfx: dict, vendors_df: pd.DataFrame, answers_df: pd.DataFrame,
             discounts_df: pd.DataFrame, history: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    cols = ["line_id", "item", "ply", "annual_qty", "vendor_id", "vendor", "unit_price_inr", "landed_price_inr", "annual_value_inr"]
    allq = df.copy()
    allq["flags"] = allq["flags"].apply(lambda f: "; ".join(f) if isinstance(f, list) else "")
    q_usable = allq[allq.usable][cols].copy()
    q_all = allq[cols + ["status", "flags"]].copy()
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


def ask(question: str, con, history_msgs: list[dict], fact_pack: dict) -> dict:
    convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history_msgs[-6:])
    plan = llm.generate("analyst:plan", PLANNER + f"\n\nRecent conversation:\n{convo}\n\nQUESTION: {question}")
    result, sql, error = None, plan.get("sql"), None
    if sql:
        try:
            result = run_sql(con, sql)
        except Exception as e:
            error = str(e)
            retry = llm.generate("analyst:repair", PLANNER + f"\n\nQUESTION: {question}\nYour previous SQL:\n{sql}\nfailed with:\n{error}\nFix it.")
            sql = retry.get("sql")
            plan["chart"] = retry.get("chart", plan.get("chart"))
            if sql:
                result = run_sql(con, sql)
                error = None
    if result is None:
        text = plan.get("cannot_answer") or f"I couldn't answer that from the data. {error or ''}"
        return {"answer": text, "sql": sql, "table": None, "chart": None, "title": plan.get("title")}
    payload = {"question": question, "sql": sql, "row_count": len(result),
               "RESULT": json.loads(result.head(60).to_json(orient="records")), "FACTS": fact_pack}
    answer = llm.generate("analyst:explain", NARRATOR + "\n\n" + json.dumps(payload, default=str), as_json=False, temperature=0.2)
    return {"answer": answer, "sql": sql, "table": result, "chart": plan.get("chart"), "title": plan.get("title")}
