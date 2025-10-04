from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os, json, re
from langchain_openai import ChatOpenAI

def _kw_score(item: Dict[str, Any], query: str) -> Tuple[int, int]:
    """Heuristic: count matching tokens; tie-break by shorter display."""
    text = " ".join([
        item.get("system",""), item.get("code",""),
        item.get("display",""), json.dumps(item.get("extras", {}))
    ]).lower()
    toks = set(re.findall(r"\w+", query.lower()))
    hits = sum(tok in text for tok in toks)
    return (hits, -len(item.get("display","")))

def _heuristic_top_k(query: str, items: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    ranked = sorted(items, key=lambda it: _kw_score(it, query), reverse=True)
    seen = set()
    dedup = []
    for it in ranked:
        key = (it.get("system"), it.get("code"))
        if key not in seen:
            seen.add(key)
            dedup.append(it)
        if len(dedup) >= k:
            break
    return dedup

def rank_top_k(query: str, items: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
    """
    Use the LLM to select the K most relevant codes across all systems.
    Falls back to a deterministic heuristic if LLM is unavailable or errors.
    """
    if not items:
        return []

    # Try LLM selection
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model_name, temperature=0)

        # keep prompt bounded
        subset = items[:120]
        compact = [
            {"system": it.get("system",""), "code": it.get("code",""), "display": it.get("display","")}
            for it in subset
        ]

        prompt = (
            "You are selecting the most clinically relevant codes for a user's query.\n"
            f"Query: {query}\n\n"
            f"From the list below, return up to {k} selections that best match the query.\n"
            "Return ONLY a JSON array of objects with keys 'system' and 'code'. No extra text.\n"
            "Prefer standard/common codes, exact term matches, and high clinical relevance.\n\n"
            f"Items:\n{json.dumps(compact, ensure_ascii=False)}\n"
        )

        raw = llm.invoke(prompt).content
        chosen = json.loads(raw)  # must be a JSON array
        want = {(x.get("system"), x.get("code")) for x in chosen if isinstance(x, dict)}

        picked = [it for it in subset if (it.get("system"), it.get("code")) in want]

        # pad if LLM returns fewer than k
        if len(picked) < k:
            pad = [it for it in _heuristic_top_k(query, subset, k)
                   if (it.get("system"), it.get("code")) not in want]
            picked = (picked + pad)[:k]

        return picked
    except Exception:
        # deterministic fallback
        return _heuristic_top_k(query, items, k)
