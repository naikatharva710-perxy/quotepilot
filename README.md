# QuotePilot

A prototype for Aerchain's "Kill the Quote Spreadsheet" assignment. A category buyer drafts an RFx with an AI co-pilot, five vendors reply in five messy shapes, the system reads every reply onto one comparable grid, and the buyer interrogates it in plain language up to an award she can defend.

Category: corrugated packaging, 30 line items, 5 vendors, for a fictional FMCG company (Konkanveda Foods, Pune).

## How it is built

- **The model reads; code does the maths.** Gemini extracts what each document says, in the vendor's own units and currency, with a pointer to where it saw it (cell, paragraph, page, photo region) and a confidence score. Unit, currency, per-kg and freight conversions are deterministic, visible rules in `core/normalize.py`.
- **Nothing uncertain is used silently.** Every cell is OK, needs review, or missing. Analysis and awards only use OK or buyer-confirmed cells, and every answer states what it excluded.
- **The analyst writes SQL, the database computes.** Questions become a single read-only DuckDB query; the model then explains the result using only numbers in it.
- **Certificates are checked, not trusted.** ISO certificates are read and their expiry checked against the bid date; an expired certificate fails the gate even if the vendor says "certified".

## Run locally

```
pip install -r requirements.txt
export GEMINI_API_KEY=...        # free key from aistudio.google.com
streamlit run app.py
```

On Streamlit Community Cloud, add `GEMINI_API_KEY` under the app's Secrets. Optional: `GEMINI_MODEL` to pin a model.

## Files

- `app.py` the interface
- `core/` ingest, extraction prompts, normalisation, award scenarios, analyst, co-pilot
- `data/` the fabricated RFx, last year's prices, five vendor replies, certificates, and hand-verified ground truth (used only by the accuracy check)
- `scripts/make_dataset.py` how the dataset was fabricated
