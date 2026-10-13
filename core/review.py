"""Review-queue rules. Kept out of the UI so they can be tested, and enforced again in normalise()
so no screen can bypass them.

Bulk accept exists for one honest case: many cells flagged for the same mechanical reason
(e.g. a per-kg rate converted with our weight estimate). It must never sweep in a cell the
reader itself was unsure about. Those need a buyer to look at them one by one.
"""
from __future__ import annotations

import pandas as pd

LOW_CONFIDENCE = 0.5   # below this, the reader is guessing
AMBIGUITY_WORDS = ("ambig", "unclear", "not clear", "unsure", "not sure", "uncertain", "smudg", "illegible",
                   "blur", "obscur", "cannot read", "can't read", "could be", "either ")


def _text(row) -> str:
    parts = [row.get("uncertainty") or "", row.get("spec_deviation") or ""]
    parts += list(row.get("flags") or [])
    return " ".join(str(p) for p in parts if p == p).lower()


def bulk_block_reason(row) -> str | None:
    """Why this cell may NOT be accepted in bulk, or None if it may."""
    if row.get("status") != "REVIEW":
        return "not awaiting review"
    price = row.get("unit_price_inr")
    if price is None or price != price:
        return "no price to accept"
    conf = float(row.get("confidence") or 0)
    if conf < LOW_CONFIDENCE:
        return f"low reader confidence ({conf:.0%})"
    sd = row.get("spec_deviation")
    if sd and sd == sd:
        return "vendor quoted a different specification"
    if any(str(f).startswith("Outlier") for f in (row.get("flags") or [])):
        return "price is an outlier against other vendors"
    if any(w in _text(row) for w in AMBIGUITY_WORDS):
        return "reader flagged it as ambiguous"
    return None


def blocked(reason) -> bool:
    """True if a bulk_block value means 'needs an individual decision'. Pandas may store 'no reason' as NaN."""
    return isinstance(reason, str) and bool(reason)


def uses_estimated_weight(key: str) -> bool:
    """Per-kg groups: the price is the vendor's per-kg rate × OUR estimated box weight, not a vendor-supplied weight."""
    return str(key).lower().startswith("per-kg")


ESTIMATED_WEIGHT_NOTE = "price uses our estimated box weight; the vendor has not supplied or confirmed it"


def reason_key(flags) -> str:
    return flags[0].split(":")[0][:70] if flags else "Other"


def annotate(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    q["reason_key"] = q["flags"].apply(reason_key)
    q["bulk_block"] = [bulk_block_reason(r) for r in q.to_dict("records")]
    return q


def bulk_groups(q: pd.DataFrame, min_size: int = 3) -> pd.DataFrame:
    """Groups of bulk-eligible cells sharing vendor and reason, plus how many similar cells are held back."""
    q = annotate(q)
    elig = q[q.bulk_block.isna()]
    held = q[q.bulk_block.notna()].groupby(["vendor_id", "reason_key"]).size().to_dict()
    g = elig.groupby(["vendor_id", "vendor", "reason_key"]).agg(n=("line_id", "count"),
                                                               line_ids=("line_id", list)).reset_index()
    g["held_back"] = [held.get((r.vendor_id, r.reason_key), 0) for r in g.itertuples()]
    return g[g.n >= min_size].reset_index(drop=True)


def bulk_accept(q: pd.DataFrame, vendor_id: str, key: str, decisions: dict, log=None) -> tuple[list[str], list[tuple[str, str]]]:
    """Accept every ELIGIBLE cell in the group. Returns (accepted line ids, [(skipped line id, reason)])."""
    q = annotate(q)
    sel = q[(q.vendor_id == vendor_id) & (q.reason_key == key)]
    accepted, skipped = [], []
    caveat = f" ({ESTIMATED_WEIGHT_NOTE})" if uses_estimated_weight(key) else ""
    for r in sel.to_dict("records"):
        if blocked(r["bulk_block"]):
            skipped.append((r["line_id"], r["bulk_block"]))
            continue
        decisions[(r["line_id"], vendor_id)] = {"action": "accept", "value": r["unit_price_inr"], "via": "bulk",
                                                "note": f"Bulk accept: {key}{caveat}"}
        accepted.append(r["line_id"])
        if log:
            log("Cell accept (bulk)", f"{r['line_id']}/{r['vendor']} at ₹{r['unit_price_inr']:.2f}{caveat}; "
                                      f"source {r.get('source_file')}, {r.get('source_ref')}: \"{str(r.get('source_quote') or '')[:80]}\"")
    return accepted, skipped
