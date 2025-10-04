import os
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx


from .models import SearchRequest, SearchResponse, CodeItem
from . import clients, rank, summarize



PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")  

app = FastAPI(title="Clinical Codes Finder")

@app.on_event("startup")
def _log_api_key_presence():
    print(f"OPENAI_API_KEY present? {bool(os.getenv('OPENAI_API_KEY'))}", flush=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    query = req.query.strip()
    async with httpx.AsyncClient() as session:
        results = []
        # Fan-out queries
        tasks = [
            clients.search_icd10cm(session, query),
            clients.search_loinc_items(session, query),
            clients.search_rxterms(session, query),
            clients.search_hcpcs(session, query),
            clients.search_ucum(session, query),
            clients.search_hpo(session, query),
        ]
        for coro in tasks:
            try:
                results.extend(await coro)
            except Exception:
                pass
    top = rank.rank_top(query, results, k=10)
    summary = await summarize.llm_summary(query, top)
    return SearchResponse(results=top, summary=summary)

@app.get("/health/env")
def health_env():
    k = os.getenv("OPENAI_API_KEY","")
    return {"has_key": bool(k), "tail": k[-6:]}
