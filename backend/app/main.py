import os
import time
import logging
from pathlib import Path
from collections import defaultdict
import asyncio
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from .models import SearchRequest, SearchResponse, CodeItem
from . import clients, rank, summarize


# Load environment variables 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

TOP_K = int(os.getenv("TOP_K", "10"))  # configurable number of top results

# Logging 
logger = logging.getLogger("clinical_codes_finder")
# enable a config:
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# FastAPI setup
app = FastAPI(title="Clinical Codes Finder")

@app.on_event("startup")
def _log_api_key_presence():
    logger.info(f"OPENAI_API_KEY present? {bool(os.getenv('OPENAI_API_KEY'))}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper to group results by coding system
def _group_by_system(items):
    grouped = defaultdict(list)
    for it in items:
        grouped[it.system].append(it)
    return dict(grouped)


# Lightweight retry helper for flaky upstreams 
async def _with_retry(fn, session, query, attempts: int = 2, base_delay: float = 0.5):
    """
    Calls one client function (like clients.search_icd10cm) with small retries.
    Retries on timeouts, connect errors, and 5xx server errors.
    """
    for i in range(attempts):
        try:
            return await fn(session, query)
        except httpx.HTTPStatusError as e:
            # retry on 5xx only
            if 500 <= e.response.status_code < 600 and i < attempts - 1:
                await asyncio.sleep(base_delay * (2 ** i))
                continue
            raise
        except (httpx.TimeoutException, httpx.ConnectError):
            if i < attempts - 1:
                await asyncio.sleep(base_delay * (2 ** i))
                continue
            raise
    return []


# Main search endpoint 
@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    query = req.query.strip()
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=15) as session:
        sources = [
            clients.search_icd10cm,
            clients.search_loinc_items,
            clients.search_rxterms,
            clients.search_hcpcs,
            clients.search_ucum,
            clients.search_hpo,
        ]

        # Run ALL source calls in PARALLEL with small retries
        tasks = [
            _with_retry(fn, session, query, attempts=2, base_delay=0.5)
            for fn in sources
        ]
        results = []
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results_lists:
            if isinstance(r, list):
                results.extend(r)
            

    # Rank
    top = rank.rank_top(query, results, k=TOP_K)

    # Summarize 
    try:
        summary = await summarize.llm_summary(query, top)
    except Exception:
        summary = "Summary unavailable right now; showing top matching codes."

    # Group results by system for easier UI use
    grouped = _group_by_system(top)

    # Log one concise line with latency + counts
    elapsed = time.perf_counter() - start
    logger.info(f"query={query!r} results={len(results)} top={len(top)} elapsed={elapsed:.2f}s")

    return SearchResponse(results=top, summary=summary, grouped=grouped)


# Health check endpoint
@app.get("/health/env")
def health_env():
    key = os.getenv("OPENAI_API_KEY", "")
    return {"has_key": bool(key), "tail": key[-6:]}
