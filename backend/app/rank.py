
from typing import List
from difflib import SequenceMatcher
from .models import CodeItem
from typing import Dict, Tuple, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json


def score(query: str, item: CodeItem) -> float:
    q = query.lower().strip()
    text = f"{item.code} {item.display}".lower()
    return SequenceMatcher(None, q, text).ratio()

def rank_top(query: str, items: List[CodeItem], k: int = 10) -> List[CodeItem]:
    dedup = {}
    for it in items:
        key = (it.system, it.code)
        if key not in dedup:
            dedup[key] = it
    scored = sorted(dedup.values(), key=lambda it: score(query, it), reverse=True)
    return scored[:k]

class RankingSelection(BaseModel):
    system: str = Field(..., description="Coding system")
    code: str = Field(..., description="Code identifier")

RANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a clinical coding assistant. Choose the MOST relevant Top-{k} codes for the user query. "
     "Return STRICT JSON like {\"selected\": [{\"system\": str, \"code\": str}, ...]} with exactly {k} items. "
     "Prefer precise clinical matches over vague ones. Avoid duplicates."),
    ("user",
     "Query: {query}\n\nCandidates (JSON list):\n{candidates_json}\n\n"
     "Select the {k} best codes by (system, code) only. No extra text.")
])

def llm_rank_top(query: str, items: List[CodeItem], k: int = 10) -> List[CodeItem]:
    """
    Uses an LLM to choose Top-K most relevant codes.
    Falls back to heuristic rank_top on any error or malformed output.
    """
    # de-duplicate by (system, code)
    dedup: Dict[Tuple[str, str], CodeItem] = {}
    for it in items:
        key = (it.system, it.code)
        if key not in dedup:
            dedup[key] = it
    uniques = list(dedup.values())
    if not uniques:
        return []
    if len(uniques) <= k:
        return uniques

    candidates = [
        {"system": it.system, "code": it.code, "display": it.display[:300]}
        for it in uniques
    ]
    candidates_json = json.dumps(candidates, ensure_ascii=False)

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt_msgs = RANK_PROMPT.format_messages(query=query, candidates_json=candidates_json, k=k)
        resp = llm.invoke(prompt_msgs)
        content = getattr(resp, "content", "") or str(resp)
        data = json.loads(content)
        selected = data.get("selected", [])
        # map back to CodeItem
        index = {(it.system, it.code): it for it in uniques}
        out, seen = [], set()
        for sel in selected:
            key = (sel.get("system"), sel.get("code"))
            if key in index and key not in seen:
                out.append(index[key]); seen.add(key)
        # top-up if fewer than k
        if len(out) < k:
            from .rank import rank_top as _heur
            topup = [x for x in _heur(query, uniques, k=k*2) if (x.system, x.code) not in seen][: k - len(out)]
            out.extend(topup)
        return out[:k]
    except Exception:
        # fallback: your existing heuristic
        from .rank import rank_top as _heur
        return _heur(query, uniques, k=k)
