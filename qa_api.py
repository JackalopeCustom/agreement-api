import os, re
from typing import List, Dict, Optional
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
OCR_DOC_ID = os.environ.get("OCR_DOC_ID")  # lock queries to OCR doc (recommended)

if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and OPENAI_API_KEY):
    raise SystemExit("Missing SUPABASE_URL / SERVICE_ROLE_KEY / OPENAI_API_KEY")

# Clients
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
oa = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI + CORS (so WordPress can call this)
app = FastAPI(title="Agreement Q&A API")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Helpers --------
class AskBody(BaseModel):
    question: str
    top_k: int = 12

VACATION_KEYS = [
    "vacation", "annual vacation", "vacation days", "vacation allowance",
    "vacation schedule", "paid vacation", "vacation pay", "leave of absence (vacation)",
    "days of vacation", "weeks of vacation"
]
WAGE_KEYS = [
    "rate of pay", "wage", "hourly rate", "rates of pay", "appendix bb", "pay rate"
]
HOLIDAY_KEYS = ["holiday", "holidays", "paid holiday", "holiday pay"]
OVERTIME_KEYS = ["overtime", "time and one-half", "double time", "time & one-half"]

def guess_topic(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in VACATION_KEYS): return "vacation"
    if any(k in ql for k in WAGE_KEYS): return "wage"
    if any(k in ql for k in HOLIDAY_KEYS): return "holiday"
    if any(k in ql for k in OVERTIME_KEYS): return "overtime"
    # heuristic by words
    if re.search(r"\bvacation|vacay|annual leave\b", ql): return "vacation"
    return "general"

def expand_queries(q: str) -> List[str]:
    topic = guess_topic(q)
    extras: List[str] = []
    if topic == "vacation":
        extras = [
            f"{q} vacation days",
            "vacation days by years of service",
            "vacation schedule years of service",
            "annual vacation entitlement",
            "vacation allowance table",
        ]
    elif topic == "wage":
        extras = [f"{q} rates of pay", "Appendix BB rates of pay", "hourly rate schedule"]
    elif topic == "holiday":
        extras = [f"{q} paid holiday", "holiday pay rules", "holiday qualification"]
    elif topic == "overtime":
        extras = [f"{q} overtime", "time and one-half", "double time rules"]
    return [q] + extras

def embed_query(text: str) -> List[float]:
    return oa.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding

def rpc_hybrid(query_text: str, emb: List[float], top_k: int, doc_id: Optional[str]) -> List[Dict]:
    payload = {"query_text": query_text, "query_embedding": emb, "match_count": top_k}
    if doc_id:
        payload["in_document_id"] = doc_id
    res = sb.rpc("hybrid_match_agreement_chunks", payload).execute()
    return res.data or []

def retrieve_hybrid_smart(query_text: str, top_k: int = 18, doc_id: Optional[str] = None) -> List[Dict]:
    # 1) run hybrid for original + expanded synonyms
    queries = expand_queries(query_text)
    all_hits: Dict[str, Dict] = {}
    for q in queries:
        emb = embed_query(q)
        hits = rpc_hybrid(q, emb, top_k=top_k, doc_id=doc_id)
        for h in hits:
            key = h.get("chunk_id") or f"{h.get('page_start')}-{h.get('page_end')}-{h.get('chunk_index')}"
            # keep the better score
            prev = all_hits.get(key)
            if not prev or (h.get("hybrid_score", 0) > prev.get("hybrid_score", 0)):
                all_hits[key] = h

    results = list(all_hits.values())

    # 2) soft keyword filter if topic is clear (keeps relevance high for short questions)
    topic = guess_topic(query_text)
    if topic == "vacation":
        kw = re.compile(r"\b(vacation|annual leave|vacation (days?|allowance|schedule)|weeks of vacation)\b", re.I)
    elif topic == "wage":
        kw = re.compile(r"\b(rate of pay|wage|hourly rate|rates of pay|appendix\s*bb)\b", re.I)
    elif topic == "holiday":
        kw = re.compile(r"\b(holiday|holidays|holiday pay|paid holiday)\b", re.I)
    elif topic == "overtime":
        kw = re.compile(r"\b(overtime|time[- ]?and[- ]?one[- ]?half|double[- ]?time)\b", re.I)
    else:
        kw = None

    if kw:
        filtered = [r for r in results if kw.search(r.get("content", ""))]
        # if filtering left us empty, fall back to unfiltered
        if filtered:
            results = filtered

    # 3) sort by hybrid_score desc, then ts_rank, then similarity
    results.sort(key=lambda r: (r.get("hybrid_score", 0), r.get("ts_rank", 0), r.get("similarity", 0)), reverse=True)

    # 4) return more for context; the answer function will trim smartly
    return results[:max(top_k, 12)]

def format_context(chunks: List[Dict]) -> str:
    """
    Keep the best 3 chunks full (tables/schedules often sit in there),
    then show head+tail for the rest so totals stay reasonable.
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

def answer_with_citations(question: str, chunks: List[Dict]) -> str:
    context = format_context(chunks)
    system = (
        "You are a contracts assistant. Answer ONLY from the provided excerpts.\n"
        "If the answer truly is not present, say 'Not found in the agreement excerpts provided.'\n"
        "If you see a schedule/table (e.g., vacation by years of service), extract the correct row for the tenure asked.\n"
        "Always add page-range citations like (Pages X–Y) at the end of each sentence."
    )
    user = (
        f"Agreement excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "1) Be concise and factual.\n"
        "2) For vacation questions, find the tenure (e.g., 7 years) and state the exact entitlement.\n"
        "3) Include (Pages X–Y) after each sentence. Quote exact lines when possible."
    )
    resp = oa.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    return resp.choices[0].message.content.strip()

@app.get("/")
def root():
    return {"ok": True, "message": "Agreement Q&A API running."}

@app.post("/ask")
def ask(body: AskBody):
    hits = retrieve_hybrid_smart(body.question, top_k=min(max(body.top_k, 12), 24), doc_id=OCR_DOC_ID)
    if not hits:
        return {"answer": "Not found in the agreement excerpts provided.", "citations": [], "chunks": []}
    answer = answer_with_citations(body.question, hits[:10])
    cites = [{"page_start": h["page_start"], "page_end": h["page_end"]} for h in hits[:8]]
    return {"answer": answer, "citations": cites, "chunks": hits[:6]}
