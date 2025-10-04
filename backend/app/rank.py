from __future__ import annotations

import os
import json
import difflib
from typing import List, Tuple, Optional, Set

from .models import CodeItem

# Model can be overridden via .env 
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    _HAS_LLM = bool(os.getenv("OPENAI_API_KEY"))
except Exception:
    _HAS_LLM = False




def _dedupe(items: List[CodeItem]) -> List[CodeItem]:
    """Remove duplicates by (system, code) while preserving order."""
    seen: Set[Tuple[str, str]] = set()
    out: List[CodeItem] = []
    for it in items:
        key = (it.system, it.code)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _heuristic_score(query: str, item: CodeItem) -> float:
    """Simple lexical similarity score as fallback."""
    q = (query or "").lower().strip()
    fields = " ".join([item.system or "", item.code or "", item.display or ""]).lower()
    return difflib.SequenceMatcher(None, q, fields).ratio()


def _heuristic_top_k(query: str, items: List[CodeItem], k: int) -> List[CodeItem]:
    """Deterministic heuristic ranking with dedupe and stable tie-breaks."""
    deduped = _dedupe(items)
    scored = []
    for idx, it in enumerate(deduped):
        scored.append((_heuristic_score(query, it), idx, it))
    # Sort by score desc, then by original order asc
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [it for _, _, it in scored[:k]]




_PROMPT = None
if _HAS_LLM:
    _PROMPT = ChatPromptTemplate.from_template(
        """
You are selecting the most relevant clinical codes for a user query.

Return ONLY a JSON array of UNIQUE integer indices, no text before/after.
Rules:
- Consider semantic relevance to the query.
- Deduplicate by (system, code).
- Select at most {k} items.
- If multiple items are near-duplicates, prefer ICD-10-CM for diagnoses, LOINC for labs, RxTerms for meds, HPO for phenotypes.
- Output strictly: [0, 3, 5]  (JSON array of integers)

User query: "{query}"

Items (index · system · code · display):
{lines}
""".strip()
    )


def _format_items_for_prompt(items: List[CodeItem]) -> str:
    lines = []
    for i, it in enumerate(items):
        disp = (it.display or "").replace("\n", " ").strip()
        lines.append(f"{i} · {it.system} · {it.code} · {disp}")
    return "\n".join(lines)


def _llm_select_indices(query: str, items: List[CodeItem], k: int) -> Optional[List[int]]:
    """Ask the LLM to return a JSON array of indices; validate strictly."""
    if not _HAS_LLM or not items or _PROMPT is None:
        return None

    llm = ChatOpenAI(model=_OPENAI_MODEL, temperature=0)
    msg = _PROMPT.format_messages(
        query=query,
        k=k,
        lines=_format_items_for_prompt(items),
    )
    resp = llm.invoke(msg)
    text = (getattr(resp, "content", None) or "").strip()

    # try to parse JSON array safely
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return None
        out: List[int] = []
        seen: Set[int] = set()
        for v in parsed:
            if isinstance(v, int) and 0 <= v < len(items) and v not in seen:
                out.append(v)
                seen.add(v)
            if len(out) >= k:
                break
        return out or None
    except Exception:
        return None




def rank_top(query: str, items: List[CodeItem], k: int = 10) -> List[CodeItem]:
    """
    Returns top-k CodeItem using LLM selection with strict JSON parsing,
    falling back to a deterministic heuristic ranking.
    """
    # Deduplicate first so both LLM and heuristic see clean candidates
    base = _dedupe(items)

    # Try LLM selection 
    try:
        idxs = _llm_select_indices(query, base, k)
        if idxs:
            chosen = [base[i] for i in idxs]
            # Dedupe once more in case indices still include near-dupes
            return _dedupe(chosen)[:k]
    except Exception:
        pass

    # Fallback heuristic 
    return _heuristic_top_k(query, base, k)
