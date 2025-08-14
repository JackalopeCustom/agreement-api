import os, re
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# Load env
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OCR_DOC_ID = os.environ.get("OCR_DOC_ID")  # lock to OCR doc

if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and OPENAI_API_KEY):
    raise SystemExit("Missing SUPABASE_URL / SERVICE_ROLE_KEY / OPENAI_API_KEY")

# Clients
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
oa = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI + CORS for WordPress
app = FastAPI(title="Agreement Q&A API")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Models -----------------
class AskBody(BaseModel):
    question: str
    top_k: int = 12

# ----------------- Topic / query helpers -----------------
VACATION_KEYS = [
    "vacation", "annual vacation", "vacation days", "vacation allowance",
    "vacation schedule", "paid vacation", "vacation pay", "weeks of vacation"
]
WAGE_KEYS = [
    # broader so "how much do they make" hits wages
    "rate of pay","rates of pay","wage","wages","hourly rate","hourly pay","per hour",
    "pay rate","how much pay","how much do","how much does","make","salary","appendix bb"
]
HOLIDAY_KEYS = ["holiday","holidays","paid holiday","holiday pay"]
OVERTIME_KEYS = ["overtime","time and one-half","time & one-half","double time","time-and-a-half"]

# Known titles from Appendix BB (add more here if your table grows)
KNOWN_TITLES = [
    "Signal Maintenance Foreman","Signal Gang Foreman","Signal Safety Foreman","TSC Coordinator",
    "CDC Electronic Technician Level 1","CDC Electronic Technician Level 2","CDC Electronic Technician Level 3",
    "Electronic Technician","Signal Inspector","Interlocking Repairman","Retarder Yard Maintainer",
    "Signal Maintainer","Lead Signalman","Relief Signal Maintainer","Signalman",
    "Assistant Signal Candidate","Assistant Signalman Step 1","Assistant Signalman Step 2",
    "Assistant Signalman Step 3","Assistant Signalman Step 4"
]

# Simple mapping so short wording still matches the right row
ROLE_SYNONYMS = [
    (re.compile(r"\bsignal\s+maintainer\b", re.I), "Signal Maintainer"),
    (re.compile(r"\brelief\s+signal\s+maintainer\b", re.I), "Relief Signal Maintainer"),
    (re.compile(r"\bretarder\b.*maintainer\b", re.I), "Retarder Yard Maintainer"),
    (re.compile(r"\b(interlocking)\b.*\brepair(man|men)\b", re.I), "Interlocking Repairman"),
    (re.compile(r"\bsignal\s+inspector\b", re.I), "Signal Inspector"),
    (re.compile(r"\belectronic\s+technician\b", re.I), "Electronic Technician"),
    (re.compile(r"\bcdc\b.*level\s*1\b", re.I), "CDC Electronic Technician Level 1"),
    (re.compile(r"\bcdc\b.*level\s*2\b", re.I), "CDC Electronic Technician Level 2"),
    (re.compile(r"\bcdc\b.*level\s*3\b", re.I), "CDC Electronic Technician Level 3"),
    (re.compile(r"\blead\s+signalman\b", re.I), "Lead Signalman"),
    (re.compile(r"\bsignalman\b", re.I), "Signalman"),
]

def guess_topic(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in VACATION_KEYS): return "vacation"
    if any(k in ql for k in WAGE_KEYS): return "wage"
    if any(k in ql for k in HOLIDAY_KEYS): return "holiday"
    if any(k in ql for k in OVERTIME_KEYS): return "overtime"
    # loose heuristics
    if re.search(r"\b(per hour|hourly|how much|pay|wage|rate)\b", ql): return "wage"
    if re.search(r"\bvacation|annual leave\b", ql): return "vacation"
    return "general"

def expand_queries(q: str) -> List[str]:
    topic = guess_topic(q)
    extras: List[str] = []
    if topic == "vacation":
        extras = [
            f"{q} vacation days", "vacation days by years of service",
            "vacation schedule years of service", "annual vacation entitlement",
            "vacation allowance table",
        ]
    elif topic == "wage":
        extras = [
            f"{q} rates of pay","Appendix BB rates of pay",
            "RATES OF PAY table", "hourly rate schedule", "pay rate per hour"
        ]
    elif topic == "holiday":
        extras = [f"{q} paid holiday","holiday pay rules","holiday qualification"]
    elif topic == "overtime":
        extras = [f"{q} overtime","time and one-half","double time rules"]
    return [q] + extras

def embed_query(text: str) -> List[float]:
    return oa.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding

def rpc_hybrid(query_text: str, emb: List[float], top_k: int, doc_id: Optional[str]) -> List[Dict]:
    payload = {"query_text": query_text, "query_embedding": emb, "match_count": top_k}
    if doc_id: payload["in_document_id"] = doc_id
    res = sb.rpc("hybrid_match_agreement_chunks", payload).execute()
    return res.data or []

def retrieve_hybrid_smart(query_text: str, top_k: int = 20, doc_id: Optional[str] = None) -> List[Dict]:
    # 1) Run hybrid for original + synonyms
    queries = expand_queries(query_text)
    all_hits: Dict[str, Dict] = {}
    for q in queries:
        emb = embed_query(q)
        hits = rpc_hybrid(q, emb, top_k=top_k, doc_id=doc_id)
        for h in hits:
            key = h.get("chunk_id") or f"{h.get('page_start')}-{h.get('page_end')}-{h.get('chunk_index')}"
            prev = all_hits.get(key)
            if not prev or (h.get("hybrid_score", 0) > prev.get("hybrid_score", 0)):
                all_hits[key] = h
    results = list(all_hits.values())

    # 2) Soft keyword filter by topic
    topic = guess_topic(query_text)
    if topic == "vacation":
        kw = re.compile(r"\b(vacation|annual leave|weeks of vacation|vacation days)\b", re.I)
    elif topic == "wage":
        kw = re.compile(r"\b(rate of pay|rates of pay|wage|hourly|per hour|pay rate|RATES OF PAY|appendix\s*bb)\b", re.I)
    elif topic == "holiday":
        kw = re.compile(r"\b(holiday|holidays|holiday pay|paid holiday)\b", re.I)
    elif topic == "overtime":
        kw = re.compile(r"\b(overtime|time[- ]?and[- ]?one[- ]?half|double[- ]?time)\b", re.I)
    else:
        kw = None

    if kw:
        filtered = [r for r in results if kw.search(r.get("content", ""))]
        if filtered:
            results = filtered

    # 3) sort by hybrid/ts_rank/similarity
    results.sort(key=lambda r: (r.get("hybrid_score", 0), r.get("ts_rank", 0), r.get("similarity", 0)), reverse=True)
    return results[:max(top_k, 12)]

# ----------------- Pay table extractor (deterministic) -----------------
def normalize_spaces(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()

def guess_role_from_question(q: str) -> Optional[str]:
    for pat, title in ROLE_SYNONYMS:
        if pat.search(q):
            return title
    # last resort: try to match any known title words loosely
    ql = q.lower()
    for title in KNOWN_TITLES:
        if all(t.lower() in ql for t in title.split()[:2]):  # first two words present
            return title
    return None

def extract_pay_from_chunks(role: str, chunks: List[Dict]) -> Optional[Tuple[str, str, Dict]]:
    """
    Look for 'RATES OF PAY' header and the role's row in the same chunk.
    Returns (rate_str, effective_str, source_chunk)
    """
    role_pat = re.compile(rf"^\s*{re.escape(role)}\*?\s+(\d+\.\d{{2}})\s*$", re.I | re.M)
    eff_pat = re.compile(r"\(Effective:\s*([^)]+)\)", re.I)

    for ch in chunks:
        text = ch.get("content", "")
        # Normalize columns by collapsing internal spacing (keeps newlines)
        lines = [normalize_spaces(x) for x in text.splitlines()]
        norm = "\n".join(lines)

        # Must mention a rates table to avoid false matches
        if re.search(r"\bRATES OF PAY\b", norm, re.I):
            m_rate = role_pat.search(norm)
            if m_rate:
                rate = m_rate.group(1)
                m_eff = eff_pat.search(norm)
                eff = m_eff.group(1).strip() if m_eff else "unspecified"
                return rate, eff, ch
    return None

# ----------------- Context + Answer -----------------
def format_context(chunks: List[Dict]) -> str:
    """
    Keep the best 3 chunks full (tables/schedules often sit there),
    then show head+tail for the rest.
    """
    out: List[str] = []
    for i, c in enumerate(chunks, start=1):
        txt = c["content"]
        if i > 3 and len(txt) > 1800:
            txt = txt[:900] + "\n...\n" + txt[-900:]
        page_note = f"Pages {c['page_start']}–{c['page_end']}"
        section_note = f" | Section: {c['section']}" if c.get("section") else ""
        out.append(f"[Chunk {i} | {page_note}{section_note}]\n{txt}")
    return "\n\n".join(out)

def answer_with_citations_llm(question: str, chunks: List[Dict]) -> str:
    context = format_context(chunks)
    system = (
        "You are a contracts assistant. Answer ONLY from the provided excerpts.\n"
        "If the answer truly is not present, say 'Not found in the agreement excerpts provided.'\n"
        "If you see a schedule/table (e.g., 'RATES OF PAY'), extract the correct row for the requested classification.\n"
        "Always add page-range citations like (Pages X–Y) at the end of each sentence."
    )
    user = (
        f"Agreement excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "1) Be concise and factual.\n"
        "2) For pay questions, state the hourly rate and the effective date from the header.\n"
        "3) Include (Pages X–Y) after each sentence. Quote exact lines when possible."
    )
    resp = oa.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content.strip()

# ----------------- API -----------------
@app.get("/")
def root():
    return {"ok": True, "message": "Agreement Q&A API running."}

@app.post("/ask")
def ask(body: AskBody):
    hits = retrieve_hybrid_smart(body.question, top_k=min(max(body.top_k, 14), 28), doc_id=OCR_DOC_ID)
    if not hits:
        return {"answer": "Not found in the agreement excerpts provided.", "citations": [], "chunks": []}

    topic = guess_topic(body.question)

    # Deterministic path for pay table: parse Appendix BB without relying on the model
    if topic == "wage":
        role = guess_role_from_question(body.question) or "Signal Maintainer"
        extracted = extract_pay_from_chunks(role, hits[:10])
        if extracted:
            rate, effective, src = extracted
            pages = f"Pages {src['page_start']}–{src['page_end']}"
            answer = f"- {role}: ${rate} per hour. ({pages})\n- Effective date: {effective}. ({pages})"
            cites = [{"page_start": src["page_start"], "page_end": src["page_end"]}]
            return {"answer": answer, "citations": cites, "chunks": hits[:6]}

    # Fallback to LLM answer for everything else
    answer = answer_with_citations_llm(body.question, hits[:10])
    cites = [{"page_start": h["page_start"], "page_end": h["page_end"]} for h in hits[:8]]
    return {"answer": answer, "citations": cites, "chunks": hits[:6]}
