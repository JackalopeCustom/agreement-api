import os, re
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# ==== env / clients ====
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OCR_DOC_ID = os.environ.get("OCR_DOC_ID")
if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and OPENAI_API_KEY):
    raise SystemExit("Missing SUPABASE_URL / SERVICE_ROLE_KEY / OPENAI_API_KEY")

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
oa = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Agreement Q&A API")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== request model ====
class AskBody(BaseModel):
    question: str
    top_k: int = 12

# ==== topic & synonyms ====
VACATION_KEYS = [
    "vacation","annual vacation","vacation days","vacation allowance",
    "vacation schedule","paid vacation","weeks of vacation"
]
WAGE_KEYS = [
    "rate of pay","rates of pay","wage","wages","hourly rate","hourly pay","per hour",
    "pay rate","how much pay","how much do","how much does","make","salary","appendix bb",
    "rates of pay table","pay table","r o p","rates"
]
HOLIDAY_KEYS = ["holiday","holidays","paid holiday","holiday pay"]
OVERTIME_KEYS = ["overtime","time and one-half","time & one-half","double time","time-and-a-half"]

def guess_topic(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in VACATION_KEYS): return "vacation"
    if any(k in ql for k in WAGE_KEYS): return "wage"
    if any(k in ql for k in HOLIDAY_KEYS): return "holiday"
    if any(k in ql for k in OVERTIME_KEYS): return "overtime"
    if re.search(r"\b(per hour|hourly|pay|wage|rate)\b", ql): return "wage"
    if re.search(r"\bvacation|annual leave\b", ql): return "vacation"
    return "general"

# Job-title aliases → normalized title (covers most ways people ask)
ALIAS_TO_TITLE = {
    "signal maintainer": "Signal Maintainer",
    "relief signal maintainer": "Relief Signal Maintainer",
    "lead signalman": "Lead Signalman",
    "signalman": "Signalman",
    "signal inspector": "Signal Inspector",
    "interlocking repairman": "Interlocking Repairman",
    "retarder": "Retarder Yard Maintainer",
    "retarder yard": "Retarder Yard Maintainer",
    "gang foreman": "Signal Gang Foreman",
    "maintenance foreman": "Signal Maintenance Foreman",
    "safety foreman": "Signal Safety Foreman",
    "tsc": "TSC Coordinator",
    "coordinator": "TSC Coordinator",
    "electronic technician": "Electronic Technician",
    "cdc level 1": "CDC Electronic Technician Level 1",
    "cdc level 2": "CDC Electronic Technician Level 2",
    "cdc level 3": "CDC Electronic Technician Level 3",
    "assistant signalman step 1": "Assistant Signalman Step 1",
    "assistant signalman step 2": "Assistant Signalman Step 2",
    "assistant signalman step 3": "Assistant Signalman Step 3",
    "assistant signalman step 4": "Assistant Signalman Step 4",
    "assistant signal candidate": "Assistant Signal Candidate",
}

# ==== retrieval (hybrid) ====
def embed_query(text: str) -> List[float]:
    return oa.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding

def rpc_hybrid(query_text: str, emb: List[float], top_k: int, doc_id: Optional[str]) -> List[Dict]:
    payload = {"query_text": query_text, "query_embedding": emb, "match_count": top_k}
    if doc_id: payload["in_document_id"] = doc_id
    return (sb.rpc("hybrid_match_agreement_chunks", payload).execute().data) or []

def expanded_queries(q: str) -> List[str]:
    topic = guess_topic(q)
    extras = []
    if topic == "vacation":
        extras = ["vacation days by years of service","vacation schedule years of service","annual vacation entitlement"]
    elif topic == "wage":
        extras = ["Appendix BB rates of pay","RATES OF PAY table","hourly rate schedule","pay table per hour"]
    elif topic == "holiday":
        extras = ["holiday pay rules","holiday qualification"]
    elif topic == "overtime":
        extras = ["time and one-half","double time rules"]
    return [q] + extras

def retrieve_hybrid_smart(query_text: str, top_k: int = 20, doc_id: Optional[str] = None) -> List[Dict]:
    queries = expanded_queries(query_text)
    best: Dict[str, Dict] = {}
    for q in queries:
        hits = rpc_hybrid(q, embed_query(q), top_k=top_k, doc_id=doc_id)
        for h in hits:
            k = h.get("chunk_id") or f"{h.get('page_start')}-{h.get('page_end')}-{h.get('chunk_index')}"
            if (k not in best) or (h.get("hybrid_score", 0) > best[k].get("hybrid_score", 0)):
                best[k] = h
    results = list(best.values())

    # soft keyword filter by topic
    topic = guess_topic(query_text)
    if topic == "vacation":
        kw = re.compile(r"\b(vacation|annual leave|weeks of vacation|vacation days)\b", re.I)
    elif topic == "wage":
        kw = re.compile(r"\b(rate of pay|rates of pay|wage|hourly|per hour|pay rate|RATES OF PAY|appendix\s*bb|pay table)\b", re.I)
    elif topic == "holiday":
        kw = re.compile(r"\b(holiday|holidays|holiday pay|paid holiday)\b", re.I)
    elif topic == "overtime":
        kw = re.compile(r"\b(overtime|time[- ]?and[- ]?one[- ]?half|double[- ]?time)\b", re.I)
    else:
        kw = None
    if kw:
        filtered = [r for r in results if kw.search(r.get("content",""))]
        if filtered: results = filtered

    results.sort(key=lambda r: (r.get("hybrid_score",0), r.get("ts_rank",0), r.get("similarity",0)), reverse=True)
    return results[:max(top_k, 12)]

# ==== deterministic pay-table parse ====
def norm_spaces(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()

def parse_pay_table(chunks: List[Dict]) -> Tuple[Dict[str,str], str, Optional[Dict]]:
    """
    Returns (title_to_rate, effective_date, source_chunk)
    Searches chunks that contain 'RATES OF PAY'.
    """
    title_to_rate: Dict[str, str] = {}
    effective = ""
    src_chunk: Optional[Dict] = None

    header_pat = re.compile(r"\bRATES OF PAY\b", re.I)
    line_pat = re.compile(r"^([A-Za-z][A-Za-z0-9 ./\-]*?)(?:\*+)?\s+(\d+\.\d{2})\s*$")
    eff_pat  = re.compile(r"\(Effective:\s*([^)]+)\)", re.I)

    for ch in chunks:
        text = ch.get("content", "")
        if not header_pat.search(text): 
            continue

        src_chunk = ch  # keep first table-y chunk
        lines = [norm_spaces(x) for x in text.splitlines()]
        for ln in lines:
            m = line_pat.match(ln)
            if m:
                title = m.group(1).strip()
                rate  = m.group(2).strip()
                title_to_rate[title] = rate
        m_eff = eff_pat.search(text)
        if m_eff:
            effective = m_eff.group(1).strip()
        # stop after first good table
        if title_to_rate:
            break

    return title_to_rate, effective, src_chunk

def choose_role_from_query(q: str, available_titles: List[str]) -> Optional[str]:
    ql = q.lower()

    # 1) alias direct match
    for alias, norm_title in ALIAS_TO_TITLE.items():
        if alias in ql and norm_title in available_titles:
            return norm_title

    # 2) token overlap scoring against available titles
    def tokens(s: str) -> set:
        return set(re.findall(r"[a-z0-9]+", s.lower()))
    qtok = tokens(ql)
    best = None
    best_score = 0.0
    for title in available_titles:
        ttok = tokens(title)
        inter = qtok & ttok
        score = len(inter) / max(1, len(ttok))
        # small boost for "level" number matches
        lvl_q = re.search(r"\blevel\s*(\d)\b", ql)
        lvl_t = re.search(r"\blevel\s*(\d)\b", title.lower())
        if lvl_q and lvl_t and lvl_q.group(1) == lvl_t.group(1):
            score += 0.5
        if score > best_score:
            best_score, best = score, title

    # require minimal overlap; else None
    return best if best_score >= 0.25 else None

# ==== context + LLM answer (fallback / non-wage) ====
def format_context(chunks: List[Dict]) -> str:
    out = []
    for i, c in enumerate(chunks, start=1):
        txt = c["content"]
        if i > 3 and len(txt) > 1800:
            txt = txt[:900] + "\n...\n" + txt[-900:]
        page = f"Pages {c['page_start']}–{c['page_end']}"
        sec  = f" | Section: {c['section']}" if c.get("section") else ""
        out.append(f"[Chunk {i} | {page}{sec}]\n{txt}")
    return "\n\n".join(out)

def answer_with_citations_llm(question: str, chunks: List[Dict]) -> str:
    context = format_context(chunks)
    system = (
        "You are a contracts assistant. Answer ONLY from the provided excerpts.\n"
        "If the answer truly is not present, say 'Not found in the agreement excerpts provided.'\n"
        "If you see a schedule/table (vacation or 'RATES OF PAY'), select the correct row for the requested case.\n"
        "Always add page-range citations like (Pages X–Y) at the end of each sentence."
    )
    user = (
        f"Agreement excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "1) Be concise and factual.\n"
        "2) Quote exact lines when possible.\n"
        "3) Add (Pages X–Y) after each sentence."
    )
    resp = oa.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content.strip()

# ==== API ====
@app.get("/")
def root():
    return {"ok": True, "message": "Agreement Q&A API running."}

@app.post("/ask")
def ask(body: AskBody):
    hits = retrieve_hybrid_smart(body.question, top_k=min(max(body.top_k, 16), 28), doc_id=OCR_DOC_ID)
    if not hits:
        return {"answer": "Not found in the agreement excerpts provided.", "citations": [], "chunks": []}

    topic = guess_topic(body.question)

    # --- Deterministic pay table path ---
    if topic == "wage":
        table, effective, src = parse_pay_table(hits[:10])
        if table:
            role = choose_role_from_query(body.question, list(table.keys()))
            # if user didn't name a role, but asked generic pay -> default to maintainer if present
            if not role and "Signal Maintainer" in table:
                role = "Signal Maintainer"
            if role:
                rate = table.get(role)
                if rate:
                    pages = f"Pages {src['page_start']}–{src['page_end']}" if src else "Pages N/A"
                    answer = f"- {role}: ${rate} per hour. ({pages})\n- Effective date: {effective or 'unspecified'}. ({pages})"
                    cites = [{"page_start": src["page_start"], "page_end": src["page_end"]}] if src else []
                    return {"answer": answer, "citations": cites, "chunks": hits[:6]}
        # if table not found in first 10 chunks, just fall back to LLM below

    # --- Fallback (vacation/other) ---
    answer = answer_with_citations_llm(body.question, hits[:10])
    cites = [{"page_start": h["page_start"], "page_end": h["page_end"]} for h in hits[:8]]
    return {"answer": answer, "citations": cites, "chunks": hits[:6]}
