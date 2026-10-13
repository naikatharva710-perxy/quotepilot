"""Deterministic award scenarios. The model narrates these; it never computes them."""
from __future__ import annotations

import re
import pandas as pd


def _threshold_inr(condition: str | None) -> float | None:
    if not condition:
        return None
    c = condition.lower().replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lac|crore|cr)\b", c)
    if m:
        n = float(m.group(1))
        return n * (1e5 if m.group(2) in ("lakh", "lac") else 1e7)
    m = re.search(r"(?:rs\.?|inr|₹)\s*(\d{5,})", c)
    return float(m.group(1)) if m else None


def _pick(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("landed_price_inr").groupby("line_id", as_index=False).first()


def scenarios(df: pd.DataFrame, vendors_df: pd.DataFrame, history: pd.DataFrame, rfx: dict,
              discounts: list[dict], apply_conditional: bool) -> list[dict]:
    usable = df[df.usable].copy()
    cleared = set(vendors_df[vendors_df.cleared_quality].vendor_id)
    all_lines = [l["line_id"] for l in rfx["line_items"]]
    qty = {l["line_id"]: l["annual_qty"] for l in rfx["line_items"]}
    ly = history.set_index("line_id").unit_price_inr.to_dict()

    def summarise(name: str, picks: pd.DataFrame, note: str) -> dict:
        picks = picks.copy()
        applied = []
        if apply_conditional and len(picks):
            for d in discounts:
                vid = d["vendor_id"]
                pct = d.get("percent")
                if not pct:
                    continue
                vt = (picks[picks.vendor_id == vid].landed_price_inr * picks[picks.vendor_id == vid].annual_qty).sum()
                thr = _threshold_inr(d.get("condition"))
                if vt > 0 and (thr is None or vt > thr):
                    m = picks.vendor_id == vid
                    picks.loc[m, "landed_price_inr"] = (picks.loc[m, "landed_price_inr"] * (1 - pct / 100)).round(2)
                    applied.append(f"{vid}: {pct}% (award value ₹{vt/1e5:.1f} L vs threshold "
                                   f"{'n/a' if thr is None else f'₹{thr/1e5:.0f} L'})")
        picks["annual_value_inr"] = picks.landed_price_inr * picks.annual_qty
        covered = set(picks.line_id)
        total = float(picks.annual_value_inr.sum())
        ly_cov = sum(ly.get(l, 0) * qty[l] for l in covered)
        split = picks.groupby("vendor").agg(lines=("line_id", "count"), value=("annual_value_inr", "sum")).reset_index()
        return {
            "name": name, "note": note, "total_inr": round(total, 0), "lines_covered": len(covered),
            "lines_uncovered": [l for l in all_lines if l not in covered],
            "last_year_same_lines_inr": round(ly_cov, 0),
            "saving_vs_last_year_pct": round((ly_cov - total) / ly_cov * 100, 1) if ly_cov else None,
            "split": split.to_dict("records"), "picks": picks[["line_id", "item", "vendor", "landed_price_inr", "annual_value_inr"]],
            "discounts_applied": applied,
        }

    out = [
        summarise("Cheapest per line - cleared vendors only", _pick(usable[usable.vendor_id.isin(cleared)]),
                  "Each line to the lowest landed price among vendors who passed every quality gate"),
        summarise("Cheapest per line - all vendors", _pick(usable),
                  "Ignores the quality gate; shown to price the cost of quality"),
    ]
    for vid in sorted(cleared):
        sub = usable[usable.vendor_id == vid]
        if len(sub):
            out.append(summarise(f"Single vendor - {sub.vendor.iloc[0]}", sub,
                                 "Simplest to manage; only lines this vendor quoted with usable prices"))
    return out
