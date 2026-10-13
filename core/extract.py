"""AI extraction: read one vendor response (any shape) against the RFx.

Design choices:
- The model only *reads*. It reports what the document says, in the vendor's own units and
  currency, with a pointer to where it saw it and how sure it is. It never converts units,
  never fills gaps, never does arithmetic. All maths happens in normalize.py (deterministic).
- Mapping vendor lines to RFx lines is done by meaning (item, size, ply), never by position.
"""
from __future__ import annotations

import json
from . import llm
from .ingest import Doc

SYSTEM = """You are a meticulous procurement analyst reading supplier quotations for corrugated packaging.
You extract exactly what the document says. You never guess, never infer a number that is not visible,
and never convert units or currencies. When something is unclear you say so and lower your confidence.
Return only JSON."""

PRICE_BASIS = ["per_piece", "per_100", "per_1000", "per_box_of_n", "per_set", "per_kg",
               "reference_last_year", "lump_sum", "unclear"]


def _rfx_brief(rfx: dict) -> str:
    rows = [f"{l['line_id']} | {l['item']} | {l['size_mm']} | {l['ply']}-ply {l['flute']} {l['bf']}BF | {l['printing']} | qty {l['annual_qty']}"
            for l in rfx["line_items"]]
    qs = [f"{q['q_id']}: {q['question']}" for q in rfx["questionnaire"]]
    return "RFX LINE ITEMS (id | item | size | board | print | annual qty):\n" + "\n".join(rows) + \
        "\n\nQUESTIONNAIRE:\n" + "\n".join(qs)


INSTRUCTIONS = """
Read the supplier response and return JSON with this exact shape:
{
  "vendor_name": str,
  "document_date": str|null,
  "currency": "INR"|"USD"|"EUR"|other|null,
  "unit_statements": [ {"text": verbatim statement about units/pack size/basis, "source_ref": str} ],
  "terms": {
     "freight": "included"|"extra"|"unclear",
     "freight_quote": str|null,
     "gst": str|null,
     "payment_terms": str|null,
     "validity": str|null,
     "lead_time": str|null,
     "discounts": [ {"description": str, "percent": number|null, "condition": str|null, "source_ref": str, "source_quote": str} ],
     "other_conditions": [str]
  },
  "lines": [
    {
      "rfx_line_id": "L01".."L30",
      "quoted": bool,
      "vendor_ref": str|null,            // vendor's own code or row number
      "vendor_description": str|null,
      "price_value": number|null,        // the number exactly as printed, in the vendor's unit and currency
      "price_currency": str|null,
      "price_basis": one of PRICE_BASIS,
      "basis_pack_size": number|null,    // e.g. 100 when "1 box = 100 nos"
      "spec_as_quoted": str|null,
      "spec_deviation": str|null,        // only if the quoted board/size/print differs from the RFx line
      "source_ref": str,                 // cell ref, paragraph, page, line number, or row number on the photo
      "source_quote": str,               // verbatim snippet (max 25 words) that contains the price
      "bbox": [ymin, xmin, ymax, xmax]|null,  // ONLY for images: box around the price on a 0-1000 scale
      "confidence": number,              // 0-1: how sure you are of BOTH the mapping and the value
      "uncertainty_reason": str|null
    }
  ],
  "questionnaire": [ {"q_id": "Q1".., "answered": bool, "answer": str|null, "source_quote": str|null, "confidence": number} ],
  "attachments_mentioned": [str],
  "reader_notes": [str]   // anything a careful buyer should know that does not fit above
}

Rules:
1. Return exactly one entry in "lines" for EVERY RFx line id, in RFx order. If the vendor did not quote it, set quoted=false,
   price_value=null and put the vendor's stated reason (if any) in uncertainty_reason.
2. Map vendor items to RFx lines by meaning: product, size, ply. Vendors use their own codes, wording and order.
3. Price basis:
   - If the document states a global rule (e.g. "all rates per box, 1 box = 100 nos, unless otherwise mentioned"),
     apply it to every line it governs and record basis_pack_size; but respect per-line exceptions ("pads per piece").
   - "per 1000 pcs" -> per_1000. "Rs/kg" -> per_kg. "same as last year" -> reference_last_year with price_value=null.
   - A rate per kg that applies to a class of items (e.g. "42/kg for the 5-ply") applies only to lines of that class;
     if it is ambiguous whether it covers non-box items (partitions, pads, trays, sleeves), say so in uncertainty_reason
     and set confidence <= 0.6.
4. If a digit is blurred, smudged, cut off or ambiguous, set price_value to your best reading ONLY if you are fairly sure,
   otherwise null; either way set confidence below 0.5 and explain. Never invent a value.
5. If the vendor offered a different specification than asked, fill spec_deviation and set confidence <= 0.7.
6. Discounts, freight and conditions are often buried in footnotes, asterisks or the last paragraph. Look for them.
7. Questionnaire answers may be in prose or another sheet. If not answered at all, answered=false. Do not judge pass/fail.
"""


def extract_response(doc: Doc, rfx: dict, vendor_hint: str | None = None) -> dict:
    prompt = _rfx_brief(rfx) + "\n\nPRICE_BASIS options: " + ", ".join(PRICE_BASIS) + "\n" + INSTRUCTIONS
    if vendor_hint:
        prompt += f"\nThe sender is expected to be: {vendor_hint}\n"
    files = []
    if doc.binary is not None:
        files.append((doc.binary, doc.mime, doc.name))
        prompt += f"\nThe supplier response is the attached {doc.kind} file '{doc.name}'."
        if doc.text:
            prompt += "\nA text layer extracted from it (may be incomplete) follows:\n" + doc.text[:20000]
    else:
        prompt += f"\nSUPPLIER RESPONSE ({doc.kind}, '{doc.name}'), addressed text follows:\n" + doc.text[:30000]
    out = llm.generate(f"extract:{doc.name}", prompt, system=SYSTEM, files=files)
    out = _repair(out, rfx)
    out["_source_file"] = doc.name
    out["_doc_kind"] = doc.kind
    return out


def _repair(out: dict, rfx: dict) -> dict:
    """Guarantee one entry per RFx line; anything the model dropped becomes an explicit gap."""
    by_id = {}
    for ln in out.get("lines", []) or []:
        lid = str(ln.get("rfx_line_id", "")).strip().upper()
        if lid and lid not in by_id:
            ln["rfx_line_id"] = lid
            by_id[lid] = ln
    fixed = []
    for l in rfx["line_items"]:
        ln = by_id.get(l["line_id"]) or {
            "rfx_line_id": l["line_id"], "quoted": False, "price_value": None, "price_basis": "unclear",
            "confidence": 0.0, "source_ref": "", "source_quote": "",
            "uncertainty_reason": "Reader returned nothing for this line"}
        if ln.get("price_basis") not in PRICE_BASIS:
            ln["price_basis"] = "unclear"
        try:
            ln["confidence"] = float(ln.get("confidence") or 0)
        except (TypeError, ValueError):
            ln["confidence"] = 0.0
        fixed.append(ln)
    out["lines"] = fixed
    out.setdefault("terms", {})
    out.setdefault("questionnaire", [])
    return out


CERT_INSTRUCTIONS = """Read this certificate and return JSON:
{"holder": str|null, "standard": str|null, "certificate_no": str|null, "issued": "YYYY-MM-DD"|null,
 "valid_until": "YYYY-MM-DD"|null, "issuer": str|null, "source_quote": str|null, "confidence": number}
Copy dates exactly as printed and convert to ISO format. If a field is not visible, use null."""


def read_certificate(name: str, data: bytes, mime: str) -> dict:
    out = llm.generate(f"certificate:{name}", CERT_INSTRUCTIONS, files=[(data, mime, name)])
    out["_source_file"] = name
    return out


GATE_INSTRUCTIONS = """You are checking supplier questionnaire answers against the buyer's pass rules.
For each vendor and each gating question, decide "pass", "fail" or "unknown" using ONLY the answer text and the
certificate facts given. Unknown means not answered or too vague to decide. Do not be generous: "third-party lab"
is not in-house; "15-20 days" does not meet "15 days or less". Return JSON:
{"vendors": [{"vendor_id": str, "gates": [{"q_id": str, "result": "pass"|"fail"|"unknown", "reason": str}]}]}
"""


def assess_gates(rfx: dict, vendors: list[dict]) -> dict:
    gates = [q for q in rfx["questionnaire"] if q.get("gate")]
    payload = {"gating_questions": [{"q_id": q["q_id"], "question": q["question"], "pass_rule": q["pass_rule"]} for q in gates],
               "vendors": vendors}
    return llm.generate("assess_gates", GATE_INSTRUCTIONS + "\nINPUT:\n" + json.dumps(payload, indent=1))
