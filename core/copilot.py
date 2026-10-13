"""RFx co-pilot and vendor clarification drafts."""
from __future__ import annotations

import json
from . import llm

COPILOT_SYSTEM = """You are an RFx co-pilot for a category buyer at an Indian FMCG manufacturer.
You help the buyer talk an RFx into existence: scope, line items, questionnaire, commercial terms, vendor list.
You have the buyer's item master with specs and this year's demand forecast. Use it; never invent specs.
Be brief and practical, like a sharp colleague. Ask at most one question per turn, and only if it matters.
Always return JSON:
{"reply": str,                     // what you say to the buyer (2-6 sentences, can use simple markdown)
 "draft": null | {                 // the FULL current RFx draft whenever anything changed, else null
    "title": str, "category": str, "currency": "INR",
    "line_ids": [str],             // which item-master line ids are in scope
    "qty_overrides": {line_id: int},
    "questionnaire": [{"q_id": str, "question": str, "gate": bool, "pass_rule": str}],
    "terms": {"delivery": str, "price_basis": str, "contract_period": str, "payment_terms": str, "quote_due": str},
    "vendor_ids": [str]
 },
 "missing": [str]                  // what is still undecided before it can be issued
}
Good RFx practice you should nudge toward: ask for prices per piece ex-GST with freight to plant (so quotes are comparable),
a gating quality questionnaire (certification, testing, lead time), validity and payment terms."""


def copilot_turn(messages: list[dict], item_master: list[dict], vendors: list[dict], current_draft: dict | None) -> dict:
    master = [{k: l[k] for k in ("line_id", "item", "size_mm", "ply", "printing", "annual_qty")} for l in item_master]
    ctx = {"item_master_and_forecast": master, "approved_vendors": vendors, "current_draft": current_draft}
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-12:])
    prompt = f"CONTEXT:\n{json.dumps(ctx)}\n\nCONVERSATION SO FAR:\n{convo}\n\nRespond to the buyer's last message."
    return llm.generate("rfx_copilot", prompt, system=COPILOT_SYSTEM, temperature=0.3)


CLARIFY = """Draft a short, polite, specific clarification email from the buyer to this supplier.
Only ask about the ISSUES listed; group them; reference line ids and item names; ask for a reply in a simple
"line id: price per piece" format; mention the reply deadline of 2 working days. No fluff, no marketing.
Return JSON: {"subject": str, "body": str}"""


def clarification_email(buyer: dict, vendor: dict, rfx_id: str, issues: list[str]) -> dict:
    payload = {"buyer": buyer, "vendor": vendor, "rfx_id": rfx_id, "ISSUES": issues}
    return llm.generate(f"clarify:{vendor.get('name')}", CLARIFY + "\n" + json.dumps(payload), temperature=0.3)


MEMO = """Write a one-page award recommendation memo for the VP of procurement from the SCENARIOS (computed by the system).
Recommend one scenario and say why in terms of cost, risk and quality. State the savings vs last year ONLY using
numbers given. List what must be resolved before PO release (from OPEN_ITEMS). Indian number formatting.
Plain markdown, under 300 words. Never invent numbers."""


def award_memo(payload: dict) -> str:
    return llm.generate("award_memo", MEMO + "\n\n" + json.dumps(payload, default=str), as_json=False, temperature=0.2)
