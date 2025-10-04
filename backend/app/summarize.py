import os
import json
from typing import List, Dict

from .models import CodeItem
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Prompt used by the LLM to produce a per-system summary with "why relevant"
SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """You are a clinical coding assistant. Summarize the selected Top codes grouped by system.
For EACH system, provide 1–2 concise bullets. For EACH bullet, include a short WHY it matches the user’s query.
Style: plain English, clinical tone, no marketing. Keep it tight (~100–150 words total).

Query: {query}
Grouped Items JSON: {items}
"""
)


def simple_summary(query: str, picked: List[CodeItem]) -> str:
    """Fallback summary if no API key or LLM error."""
    if not picked:
        return "No codes found."
    parts = [f"Query: {query}"]
    groups: Dict[str, List[CodeItem]] = {}
    for it in picked:
        groups.setdefault(it.system, []).append(it)
    for sys, items in groups.items():
        parts.append(f"\n{sys}:")
        for it in items:
            parts.append(f"  - {it.code}: {it.display}")
    return "\n".join(parts)


async def llm_summary(query: str, picked: List[CodeItem]) -> str:
    """
    Generate a concise, per-system summary using the LLM.
    Falls back to a readable simple summary if there's no API key or any error.
    """
    if not picked:
        return "No codes found."

    # If no key present, skip LLM and return readable fallback
    if not os.getenv("OPENAI_API_KEY"):
        return simple_summary(query, picked)

    # Group items by system and pass as JSON to the prompt
    grouped = {}
    for it in picked:
        grouped.setdefault(it.system, []).append({"code": it.code, "display": it.display})
    grouped_json = json.dumps(grouped, ensure_ascii=False)

    try:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model_name, temperature=0)  # deterministic output
        chain = SUMMARY_PROMPT | llm
        resp = await chain.ainvoke({"query": query, "items": grouped_json})
        text = (resp.content or "").strip()
        return text or simple_summary(query, picked)
    except Exception:
        return simple_summary(query, picked)
