"""Offline fixture: reader output shaped exactly like live run 4 (13 Oct 2026, gemini-3.5-flash-lite for Chakan).

Used only by the tests. The app never imports this. Values for the four non-Chakan vendors come from the
hand-verified ground truth; Chakan's lines reproduce what the live model returned, including its
confidence scores and its own words about ambiguity.
"""
from __future__ import annotations

from core import pipeline

USD = 88.5

CHAKAN_AMBIGUOUS = {
    "L15": (0.5, "Rate quoted per kg for 3-ply; ambiguity whether mailers are covered by box ply rate."),
    "L16": (0.5, "Rate quoted per kg for 3-ply; ambiguity whether mailers are covered by box ply rate."),
    "L17": (0.5, "Rate quoted per kg for 5-ply; ambiguity whether mailers are covered by box ply rate."),
    "L21": (0.5, "Rate quoted per kg for 3-ply; ambiguity whether retail-ready boxes are covered."),
    "L22": (0.5, "Rate quoted per kg for 5-ply; ambiguity whether gift hamper box is covered."),
    "L23": (0.4, "Ambiguous whether partitions are covered by the 3-ply box rate."),
    "L24": (0.4, "Ambiguous whether partitions are covered by the 3-ply box rate."),
    "L25": (0.4, "Ambiguous whether partitions are covered by the 3-ply box rate."),
    "L26": (0.4, "Ambiguous whether layer pads are covered by the 3-ply box rate."),
    "L27": (0.4, "Ambiguous whether layer pads are covered by the 3-ply box rate."),
    "L28": (0.5, "Rate quoted per kg for 5-ply; ambiguity whether display trays are covered."),
    "L29": (0.5, "Rate quoted per kg for 3-ply; ambiguity whether PDQ trays are covered."),
    "L30": (0.5, "Rate quoted per kg for 3-ply; ambiguity whether sleeves are covered."),
}


def _line(lid, **kw):
    base = dict(rfx_line_id=lid, quoted=True, price_value=None, price_basis="per_piece", price_currency=None,
                confidence=0.95, source_ref="x", source_quote="q", bbox=None, spec_deviation=None,
                uncertainty_reason=None, vendor_description=None)
    base.update(kw)
    return base


def build():
    rfx = pipeline.load_rfx()
    gt = pipeline.load_ground_truth()["lines"]
    ex = {v: {"currency": "INR", "terms": {"freight": "included", "discounts": []}, "lines": [], "questionnaire": [],
              "_source_file": f"{v}_file"} for v in ("V1", "V2", "V3", "V4", "V5")}
    ex["V1"]["currency"] = "USD"
    ex["V5"]["terms"]["freight"] = "extra"
    for l in rfx["line_items"]:
        lid, ply = l["line_id"], l["ply"]
        e = gt[lid]
        ex["V1"]["lines"].append(_line(lid, price_value=round(e["V1"] * 1000 / USD, 1), price_basis="per_1000",
                                       price_currency="USD",
                                       **({"spec_deviation": "3-ply offered instead of 5-ply", "confidence": 0.7}
                                          if lid == "L09" else {})))
        ex["V2"]["lines"].append(_line(lid, price_value=e["V2"]))
        if e["V3"] is None:
            ex["V3"]["lines"].append(_line(lid, quoted=False, price_basis="unclear", confidence=0.95,
                                           uncertainty_reason="Not quoted by vendor"))
        elif lid in ("L26", "L27"):
            ex["V3"]["lines"].append(_line(lid, price_value=e["V3"]))
        else:
            ex["V3"]["lines"].append(_line(lid, price_value=round(e["V3"] * 100), price_basis="per_box_of_n",
                                           basis_pack_size=100, source_ref="para 7"))
        if lid == "L14":
            ex["V4"]["lines"].append(_line(lid, price_value=17.0, confidence=0.7,
                                           uncertainty_reason="Price is slightly smudged/stamped over, read as 17.00."))
        else:
            ex["V4"]["lines"].append(_line(lid, price_value=e["V4"], confidence=0.85))
        # Chakan, as the live model returned it
        if l["kind"] == "RSC" and ply == 7:
            ex["V5"]["lines"].append(_line(lid, price_basis="reference_last_year", confidence=0.8,
                                           source_ref="line 8", source_quote="rest same as last year"))
        else:
            rate = 42 if ply == 5 else 38
            conf, why = CHAKAN_AMBIGUOUS.get(lid, (0.5, f"Rate quoted per kg for {ply}-ply; box weight is not provided in RFx."))
            ex["V5"]["lines"].append(_line(lid, price_value=rate, price_basis="per_kg", confidence=conf,
                                           uncertainty_reason=why, source_ref="line 8",
                                           source_quote=f"Rs 42/kg for the 5-ply" if ply == 5 else "38 for the 3-ply"))
    return rfx, ex
