import os
from typing import List, Dict, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# Load environment
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OCR_DOC_ID = os.environ.get("OCR_DOC_ID")  # set this in Railway env vars

if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and OPENAI_API_KEY):
    raise SystemExit("Missing SUPABASE_URL / SERVICE_ROLE_KEY / OPENAI_API_KEY in environment vars")

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
oa = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Agreement Q&A API")

# 🔓 CORS so your WordPress site can call this API
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskBody(BaseModel):
    question: str
    top_k: int = 12

def embed_query(text: str) -> List[float]:
    return oa.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding

def retrieve_hybrid(query_text: str, top_k: int = 12, doc_id: Optional[str] = None) -> List[Dict]:
    emb = embed_query(query_text)
    payload = {
        "query_text": query_text,
        "query_embedding": emb,
        "match_count": top_k,
    }
    if doc_id:
        payload["in_document_id"] = doc_id
    res = sb.rpc("hybrid_match_agreement_chunks", payload).execute()
    return res.data or []

def format_context(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        page_note = f"Pages {c['page_start']}–{c['page_end']}"
        section_note = f" | Section: {c['section']}" if c.get("section") else ""
        text = c["content"]
        if len(text) > 1400:
            text = text[:1400] + "…"
        lines.append(f"[Chunk {i} | {page_note}{section_note}]\n{text}")
    return "\n\n".join(lines)

def answer_with_citations(question: str, chunks: List[Dict]) -> str:
    context = format_context(chunks)
    system = (
        "You are a contracts assistant. Answer ONLY using the provided agreement excerpts. "
        "If the answer is not in the excerpts, say 'Not found in the agreement excerpts provided.' "
        "Always include citations like (Pages X–Y) after each sentence. Quote exact lines when possible."
    )
    user = (
        f"Agreement excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "1) Be concise and factual.\n"
        "2) Use bullets if more than one rule applies.\n"
        "3) Add (Pages X–Y) after each sentence."
    )
    resp = oa.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content.strip()

@app.get("/")
def root():
    return {"ok": True, "message": "Agreement Q&A API running."}

@app.post("/ask")
def ask(body: AskBody):
    hits = retrieve_hybrid(body.question, top_k=min(max(body.top_k, 4), 20), doc_id=OCR_DOC_ID)
    if not hits:
        return {"answer": "Not found in the agreement excerpts provided.", "citations": [], "chunks": []}
    answer = answer_with_citations(body.question, hits[:8])
    cites = [{"page_start": h["page_start"], "page_end": h["page_end"]} for h in hits[:8]]
    return {"answer": answer, "citations": cites, "chunks": hits[:6]}
