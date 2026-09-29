# Facts-Only MF Assistant

A small RAG-based FAQ assistant for factual mutual-fund questions.

## Product selected
**Groww** (the product/platform selected for the assignment).

## Corpus scope
**AMC:** HDFC Mutual Fund / HDFC AMC

**Schemes:**
1. HDFC Large Cap Fund
2. HDFC Flexi Cap Fund
3. HDFC ELSS Tax Saver
4. HDFC Balanced Advantage Fund

**Supporting authorities:** selected SEBI Investor and AMFI investor/disclosure pages.

## What the assistant answers
- Expense ratio / TER (when present in the approved source)
- Minimum SIP
- Exit load
- ELSS lock-in
- Riskometer
- Benchmark
- Account / capital-gains / CAS statement process
- Official document availability

Every answer includes one official source link and a "Last updated from sources" note.

## What it refuses
- Buy/sell/switch recommendations
- Portfolio allocation and "best fund" questions
- Return/performance comparison or calculations
- Requests containing PAN, Aadhaar, folio/account numbers, OTPs, phone numbers or emails

## RAG design
1. Approved public pages are represented as a small curated corpus in `data/knowledge_base.json`.
2. Each chunk has a source ID mapped to an official URL in `data/sources.csv`.
3. `sentence-transformers` creates embeddings.
4. Cosine similarity retrieves the most relevant chunks.
5. A conservative threshold rejects weakly grounded queries.
6. Response rules generate short, extractive factual answers and attach exactly one source link.

This intentionally prioritizes grounding and citation accuracy over open-ended generation.

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deploy the repository on Streamlit Community Cloud and set the main file to `app.py`. No API key is required for this prototype because retrieval and response generation are intentionally lightweight and grounded in the curated corpus.

## Disclaimer used in UI
> Facts-only. No investment advice. Do not enter PAN, Aadhaar, folio/account numbers, OTPs, phone numbers or email addresses.

## Known limits
- The corpus is intentionally small and limited to the four selected HDFC schemes plus SEBI/AMFI support pages.
- Values such as TER, riskometer and minimum SIP can change; users should open the cited official source for the latest value.
- The prototype does not calculate or compare returns.
- The current response layer is extractive/rule-based after embedding retrieval. A production version could add an LLM generation layer constrained to retrieved chunks.

## Files
- `app.py` — Streamlit prototype
- `data/sources.csv` — 18 approved official URLs
- `data/knowledge_base.json` — retrieval chunks
- `sample_qa.md` — demo questions and expected answers
