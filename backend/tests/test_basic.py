import sys
import asyncio
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import httpx
from fastapi.testclient import TestClient

from backend.app.models import CodeItem
from backend.app import rank
from backend.app.main import app, _with_retry


def test_rank():
    items = [
        CodeItem(system="ICD-10-CM", code="E11.9", display="Type 2 diabetes mellitus without complications"),
        CodeItem(system="LOINC", code="2345-7", display="Glucose [Mass/volume] in Blood"),
    ]
    picked = rank.rank_top("blood sugar test", items)
    assert picked and picked[0].display.lower().find("glucose") >= 0


def _ci(sys, code, display):
    return CodeItem(system=sys, code=code, display=display, extras={})


@pytest.fixture(autouse=True)
def patch_clients_and_summary(monkeypatch):
    # Minimal, deterministic mocks
    async def _icd(session, q):   return [_ci("ICD-10-CM", "R07.9", "Chest pain, unspecified")]
    async def _loinc(session, q): return [_ci("LOINC", "58259-3", "Chest pain")]
    async def _rx(session, q):    return []
    async def _hcpcs(session, q): return []
    async def _ucum(session, q):  return []
    async def _hpo(session, q):   return [_ci("HPO", "HP:0100749", "Chest pain")]

    monkeypatch.setattr("backend.app.clients.search_icd10cm", _icd)
    monkeypatch.setattr("backend.app.clients.search_loinc_items", _loinc)
    monkeypatch.setattr("backend.app.clients.search_rxterms", _rx)
    monkeypatch.setattr("backend.app.clients.search_hcpcs", _hcpcs)
    monkeypatch.setattr("backend.app.clients.search_ucum", _ucum)
    monkeypatch.setattr("backend.app.clients.search_hpo", _hpo)

  
    async def _fake_summary(query, items):
        return f"Summary for {query} with {len(items)} items."
    monkeypatch.setattr("backend.app.summarize.llm_summary", _fake_summary)


def test_search_endpoint_returns_grouped_and_summary():
    client = TestClient(app)
    resp = client.post("/search", json={"query": "chest pain"})
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data.get("results"), list)
    assert isinstance(data.get("summary"), str)
    assert isinstance(data.get("grouped"), dict)

    # basic content check
    systems = set(data["grouped"].keys())
    assert {"ICD-10-CM", "LOINC", "HPO"}.issubset(systems)

    # summary came from our fake summarizer
    assert data["summary"].startswith("Summary for chest pain")



class _Flaky:
    def __init__(self):
        self.calls = 0
    async def __call__(self, session, query):
        self.calls += 1
        if self.calls == 1:
            # first attempt fails with timeout
            raise httpx.TimeoutException("simulated timeout")
        return ["ok"]


def test_with_retry_recovers_on_timeout():
    f = _Flaky()
    out = asyncio.run(_with_retry(f, session=None, query="x", attempts=2, base_delay=0.0))
    assert out == ["ok"]
    assert f.calls == 2
