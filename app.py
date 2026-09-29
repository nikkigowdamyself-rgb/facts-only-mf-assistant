import os, re, json, csv
from pathlib import Path
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

BASE = Path(__file__).parent
chunks = json.loads((BASE/"data/knowledge_base.json").read_text(encoding="utf-8"))
sources = {r["id"]: r for r in csv.DictReader(open(BASE/"data/sources.csv", encoding="utf-8"))}

st.set_page_config(page_title="Facts-Only MF Assistant", page_icon="📘", layout="centered")
st.title("📘 Facts-Only MF Assistant")
st.caption("HDFC Mutual Fund corpus • RAG prototype • Facts-only. No investment advice.")
st.info("Facts-only. No investment advice. Do not enter PAN, Aadhaar, folio/account numbers, OTPs, phone numbers or email addresses.")
st.markdown("**Try:** `What is the minimum SIP for HDFC ELSS Tax Saver?` · `What is the exit load of HDFC Flexi Cap Fund?` · `How can I get a consolidated account statement?`")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
texts = [c["text"] for c in chunks]
emb = model.encode(texts, normalize_embeddings=True)

PII = re.compile(r"\b(?:\d{10}|[A-Z]{5}\d{4}[A-Z]|(?:\+91[\s-]?)?\d{10})\b", re.I)
ADVICE = re.compile(r"\b(should i|should we|buy|sell|invest|worth investing|best fund|which fund|recommend|recommendation|portfolio|allocate|switch to|good investment|better fund|where should)\b", re.I)
RETURNS = re.compile(r"\b(return|returns|cagr|xirr|performance|outperform|beat the benchmark|highest return|best performing)\b", re.I)

def retrieve(q, k=4):
    qv = model.encode([q], normalize_embeddings=True)[0]
    scores = emb @ qv
    idx = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in idx]

def intent(q):
    if PII.search(q):
        return "pii"
    if ADVICE.search(q):
        return "advice"
    if RETURNS.search(q):
        return "returns"
    return "facts"

def answer(q):
    typ = intent(q)
    if typ == "pii":
        return ("For privacy, please do not share PAN, Aadhaar, folio/account numbers, OTPs, phone numbers or email addresses here. "
                "I can explain the general statement/download process without personal information.", "S09")
    if typ == "advice":
        return ("I can provide facts about the schemes, but I can’t recommend buying, selling, switching or choosing a fund. "
                "You can ask for a scheme’s expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark or official documents.", "S15")
    if typ == "returns":
        return ("I don’t calculate or compare investment returns in this facts-only assistant. "
                "I can point you to the official factsheet or scheme documents for the scheme’s disclosed performance information.", "S07")
    hits = retrieve(q)
    top, score = hits[0]
    # Conservative grounding threshold
    if score < 0.32:
        return ("I couldn’t find a sufficiently grounded fact in the approved HDFC/SEBI/AMFI corpus. "
                "Please ask about a supported scheme or fact such as expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark or statements.", "S15")
    # Extractive answer generation: avoids unsupported claims.
    t = top["text"]
    ql = q.lower()
    if "minimum" in ql and "sip" in ql:
        m = re.search(r"minimum SIP[^₹]*₹[\d,]+", t, re.I)
        ans = m.group(0) if m else t.split(".")[0]
    elif "exit load" in ql:
        sent = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if "exit load" in s.lower()]
        ans = " ".join(sent[:2]) if sent else t
    elif "lock" in ql:
        sent = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if "lock" in s.lower()]
        ans = " ".join(sent[:2]) if sent else t
    elif "riskometer" in ql or "risk" in ql:
        sent = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if "risk" in s.lower()]
        ans = " ".join(sent[:2]) if sent else t
    elif "benchmark" in ql:
        sent = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if "benchmark" in s.lower()]
        ans = " ".join(sent[:2]) if sent else t
    elif "statement" in ql or "capital gain" in ql or "cas" in ql:
        ans = t
    else:
        ans = t[:450]
    return ans, top["source_id"]

q = st.chat_input("Ask a factual mutual-fund question…")
if q:
    st.chat_message("user").write(q)
    ans, sid = answer(q)
    st.chat_message("assistant").write(ans)
    st.markdown(f"**Source:** [{sources[sid]['title']}]({sources[sid]['url']})")
    st.caption("Last updated from sources: verify the linked official page for the latest scheme-specific value.")

with st.expander("Scope & limitations"):
    st.write("Corpus: HDFC Mutual Fund, four schemes (Large Cap, Flexi Cap, ELSS Tax Saver, Balanced Advantage), plus selected SEBI and AMFI educational/statement pages.")
    st.write("The prototype uses embedding retrieval over a small curated corpus and extractive response rules. It does not accept/store PII and does not provide investment advice or return comparisons.")
