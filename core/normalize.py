"""Deterministic normalisation: vendor's number -> INR per piece, ex-GST.

The model never does this maths. Every conversion is a visible, editable rule, and every
cell carries a status that decides whether analysis is allowed to use it:

  OK        read with high confidence, converted by a known rule
  REVIEW    usable only after the buyer confirms (derived, ambiguous, deviating, outlier, low confidence)
  MISSING   no usable price
A buyer decision turns REVIEW into CONFIRMED (usable) or REJECTED (treated as missing).
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, asdict
import pandas as pd

CONF_OK = 0.8
OUTLIER_BAND = 0.35


@dataclass
class Assumptions:
    usd_inr: float = 88.5
    eur_inr: float = 96.0
    freight_pct_when_extra: float = 4.0    # added to landed price when vendor says freight extra
    apply_conditional_discounts: bool = False


def _fx(currency: str | None, a: Assumptions) -> tuple[float, str | None]:
    c = (currency or "INR").upper().strip()
    if c in ("INR", "RS", "RS.", "₹", "RUPEES"):
        return 1.0, None
    if c == "USD":
        return a.usd_inr, f"USD converted at ₹{a.usd_inr:.2f}"
    if c == "EUR":
        return a.eur_inr, f"EUR converted at ₹{a.eur_inr:.2f}"
    return float("nan"), f"Unknown currency '{currency}'"


def _rate_from_quote(q: str | None, ply: int | None = None) -> float | None:
    """Per-kg rate for this item's ply from the quoted text.
    'Rs 42/kg for the 5-ply, 38 for the 3-ply' -> 42 for 5-ply, 38 for 3-ply, None for 7-ply."""
    import re
    if not q:
        return None
    by_ply = {int(p): float(r) for r, p in re.findall(r"(\d+(?:\.\d+)?)\s*(?:/\s*kg)?\s*for\s+the\s+(\d)\s*-?\s*ply", q, re.I)}
    if by_ply:
        return by_ply.get(ply) if ply is not None else None
    if re.search(r"\d\s*-?\s*ply", q, re.I):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*kg", q, re.I)
    return float(m.group(1)) if m else None


def _same_vendor(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    return a in b or b in a or difflib.SequenceMatcher(None, a, b).ratio() > 0.7


def normalise(rfx: dict, extractions: dict[str, dict], history: pd.DataFrame, a: Assumptions,
              decisions: dict[tuple[str, str], dict] | None = None) -> pd.DataFrame:
    decisions = decisions or {}
    vendors = {v["vendor_id"]: v["name"] for v in rfx["vendors"]}
    hist = history.set_index("line_id").to_dict("index") if history is not None and len(history) else {}
    items = {l["line_id"]: l for l in rfx["line_items"]}
    rows = []
    for vid, ex in extractions.items():
        vname = vendors.get(vid) or ex.get("vendor_name") or vid
        doc_currency = ex.get("currency")
        freight = (ex.get("terms") or {}).get("freight", "unclear")
        for ln in ex["lines"]:
            lid = ln["rfx_line_id"]
            item = items.get(lid)
            if not item:
                continue
            flags, method = [], []
            status = "OK"
            val = ln.get("price_value")
            basis = ln.get("price_basis", "unclear")
            conf = float(ln.get("confidence") or 0)
            price = None
            q_text = str(ln.get("source_quote") or "").lower()
            if (basis == "per_kg" and val is None and _rate_from_quote(q_text, item.get("ply")) is None
                    and "last year" in q_text):
                basis = "reference_last_year"   # no per-kg rate for this ply; the vendor's catch-all is "same as last year"
                flags.append("No per-kg rate quoted for this ply; treated as 'same as last year'")
            if not ln.get("quoted", True) and val is None and basis != "reference_last_year":
                status = "MISSING"
                flags.append(ln.get("uncertainty_reason") or "Not quoted")
            elif basis == "reference_last_year":
                h = hist.get(lid)
                if h and _same_vendor(str(h["awarded_supplier"]), vname):
                    price = float(h["unit_price_inr"])
                    method.append(f"'Same as last year' → last year's PO price ₹{price:.2f}")
                    flags.append("Resolved from last year's price list, confirm the vendor meant this")
                    status = "REVIEW"
                else:
                    status = "MISSING"
                    who = h["awarded_supplier"] if h else "nobody"
                    flags.append(f"Vendor said 'same as last year' but last year this line went to {who}; no price to carry over")
            elif val is None and basis == "per_kg" and _rate_from_quote(ln.get("source_quote"), item.get("ply")) is not None:
                # Reader saw a per-kg rate but left the value blank (seen live with a lite model).
                rate = _rate_from_quote(ln.get("source_quote"), item.get("ply"))
                w = item.get("est_weight_kg")
                price = round(rate * float(w), 2)
                method.append(f"Rate ₹{rate:g}/kg taken from the quoted text by rule (reader left it blank)")
                method.append(f"× {w} kg estimated box weight (from spec)")
                flags.append("Per-kg rate converted with our weight estimate; vendor's actual weight may differ")
                status = "REVIEW"
                val = rate
            elif val is None:
                status = "MISSING"
                flags.append(ln.get("uncertainty_reason") or "Price not readable")
            else:
                cur = ln.get("price_currency") or doc_currency
                fx, fx_note = _fx(cur, a)
                if fx_note:
                    method.append(fx_note)
                    flags.append(fx_note)
                v = float(val) * fx
                if basis == "per_piece" or basis == "per_set":
                    pass
                elif basis == "per_100":
                    v /= 100; method.append("÷100 (quoted per 100)")
                elif basis == "per_1000":
                    v /= 1000; method.append("÷1000 (quoted per 1000)")
                elif basis == "per_box_of_n":
                    n = ln.get("basis_pack_size")
                    if n:
                        v /= float(n); method.append(f"÷{int(n)} (quoted per box of {int(n)})")
                    else:
                        status = "REVIEW"; flags.append("Quoted 'per box' but pack size unknown")
                elif basis == "per_kg":
                    w = item.get("est_weight_kg")
                    v *= float(w); method.append(f"× {w} kg estimated box weight (from spec)")
                    flags.append("Per-kg rate converted with our weight estimate; vendor's actual weight may differ")
                    status = "REVIEW"
                else:
                    status = "REVIEW"; flags.append(f"Unclear price basis ('{basis}')")
                price = round(v, 2)
                if conf < CONF_OK:
                    status = "REVIEW"
                    flags.append(f"Reader confidence {conf:.0%}: {ln.get('uncertainty_reason') or 'not fully sure'}")
                if ln.get("spec_deviation"):
                    status = "REVIEW"; flags.append(f"Spec deviation: {ln['spec_deviation']}")
            landed = price
            if price is not None and freight == "extra":
                landed = round(price * (1 + a.freight_pct_when_extra / 100), 2)
                method.append(f"+{a.freight_pct_when_extra:g}% freight (vendor: freight extra)")
            elif price is not None and freight == "unclear":
                flags.append("Freight terms unclear")
            rows.append(dict(
                line_id=lid, item=item["item"], ply=item["ply"], annual_qty=item["annual_qty"],
                vendor_id=vid, vendor=vname, quoted_value=val, quoted_basis=basis,
                quoted_currency=ln.get("price_currency") or doc_currency,
                unit_price_inr=price, landed_price_inr=landed, status=status,
                confidence=conf, flags=flags, method=method,
                source_file=ex.get("_source_file"), source_ref=ln.get("source_ref"),
                source_quote=ln.get("source_quote"), bbox=ln.get("bbox"),
                spec_deviation=ln.get("spec_deviation"), vendor_description=ln.get("vendor_description"),
                uncertainty=ln.get("uncertainty_reason"),
            ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Outlier guard: catches unit mistakes (per 100 read as per piece) and misreads.
    flags_col = df["flags"].tolist()
    status_col = df["status"].tolist()
    prices = df["landed_price_inr"].tolist()
    by_line: dict[str, list[int]] = {}
    for i, (lid, p) in enumerate(zip(df["line_id"].tolist(), prices)):
        if p is not None and p == p:
            by_line.setdefault(lid, []).append(i)
    for lid, idxs in by_line.items():
        if len(idxs) < 3:
            continue
        for i in idxs:
            others = sorted(prices[j] for j in idxs if j != i)
            n = len(others)
            med = others[n // 2] if n % 2 else (others[n // 2 - 1] + others[n // 2]) / 2
            if med and abs(prices[i] - med) / med > OUTLIER_BAND:
                status_col[i] = "REVIEW"
                flags_col[i] = list(flags_col[i]) + [f"Outlier: {(prices[i] - med) / med:+.0%} vs other vendors' median ₹{med:.2f}"]
    df["status"] = status_col
    df["flags"] = flags_col
    # Buyer decisions
    df["decision"] = None
    df["decision_note"] = None
    from .review import bulk_block_reason
    for (lid, vid), d in decisions.items():
        m = (df.line_id == lid) & (df.vendor_id == vid)
        if not m.any():
            continue
        idx = df[m].index[0]
        if d.get("via") == "bulk" and bulk_block_reason(df.loc[idx].to_dict()):
            # A bulk accept can never confirm a cell that needs an individual decision, whatever the UI sent.
            df.at[idx, "decision"] = "blocked bulk accept"
            continue
        df.at[idx, "decision"] = d["action"]
        df.at[idx, "decision_note"] = d.get("note")
        if d["action"] == "accept" and df.at[idx, "unit_price_inr"] is not None and pd.notna(df.at[idx, "unit_price_inr"]):
            df.at[idx, "status"] = "CONFIRMED"
        elif d["action"] == "override":
            v = float(d["value"])
            df.at[idx, "unit_price_inr"] = v
            freight_extra = (extractions[vid].get("terms") or {}).get("freight") == "extra"
            df.at[idx, "landed_price_inr"] = round(v * (1 + a.freight_pct_when_extra / 100), 2) if freight_extra else v
            df.at[idx, "status"] = "CONFIRMED"
            df.at[idx, "method"] = list(df.at[idx, "method"]) + [f"Buyer override ₹{v:.2f}"]
        elif d["action"] == "reject":
            df.at[idx, "status"] = "REJECTED"
    df["usable"] = df.status.isin(["OK", "CONFIRMED"])
    df["annual_value_inr"] = (df.landed_price_inr * df.annual_qty).where(df.usable)
    return df


def discounts(extractions: dict[str, dict]) -> list[dict]:
    out = []
    for vid, ex in extractions.items():
        for d in (ex.get("terms") or {}).get("discounts") or []:
            out.append({"vendor_id": vid, **d})
    return out
