
import httpx
from typing import List, Dict
from .models import CodeItem

BASE = "https://clinicaltables.nlm.nih.gov/api"

async def _fetch(session: httpx.AsyncClient, url: str, params: Dict[str, str]) -> list:
    r = await session.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def _zip_ctss(rows) -> List[Dict]:
    # rows: [total, codes[], extras{}, display_rows[], (optional) code_systems[]]
    if not rows or len(rows) < 4: 
        return []
    total, codes, extras, displays = rows[0], rows[1], rows[2], rows[3]
    out = []
    for idx, code in enumerate(codes):
        dr = displays[idx] if idx < len(displays) else []
        display = " – ".join([str(x) for x in dr if x is not None])
        item = {"code": str(code), "display": display, "extras": {}}
        # Copy simple extras
        if isinstance(extras, dict):
            for k, arr in extras.items():
                if isinstance(arr, list) and idx < len(arr):
                    item.setdefault("extras", {})[k] = arr[idx]
        out.append(item)
    return out

async def search_icd10cm(session: httpx.AsyncClient, terms: str, limit: int = 25) -> List[CodeItem]:
    url = f"{BASE}/icd10cm/v3/search"
    data = await _fetch(session, url, {"terms": terms, "sf": "code,name", "df": "code,name", "count": str(limit)})
    items = _zip_ctss(data)
    return [CodeItem(system="ICD-10-CM", **it) for it in items]

async def search_loinc_items(session: httpx.AsyncClient, terms: str, limit: int = 25) -> List[CodeItem]:
    # LOINC questions/items
    url = f"{BASE}/loinc_items/v3/search"
    data = await _fetch(session, url, {"terms": terms, "type": "question", "df": "text,LOINC_NUM", "count": str(limit)})
    items = _zip_ctss(data)
    # rename fields for consistency
    for it in items:
        parts = it["display"].split(" – ")
        if len(parts) == 2:
            it["display"] = parts[0]
            it["code"] = parts[1]
    return [CodeItem(system="LOINC", **it) for it in items]

async def search_rxterms(session: httpx.AsyncClient, terms: str, limit: int = 25) -> List[CodeItem]:
    url = f"{BASE}/rxterms/v3/search"
    data = await _fetch(session, url, {"terms": terms, "df": "DISPLAY_NAME", "ef": "STRENGTHS_AND_FORMS,RXCUIS", "count": str(limit)})
    # For RxTerms, code is the DISPLAY_NAME by default; expand strengths as separate items when available
    items = []
    total, names, extras, displays = data[0], data[1], data[2], data[3]
    strengths = extras.get("STRENGTHS_AND_FORMS", []) if isinstance(extras, dict) else []
    rxcuis = extras.get("RXCUIS", []) if isinstance(extras, dict) else []
    for i, name in enumerate(names):
        base = {
            "system": "RxTerms",
            "code": str(name),
            "display": displays[i][0] if i < len(displays) else str(name),
            "extras": {}
        }
        # expand strengths
        s_list = strengths[i] if i < len(strengths) else []
        c_list = rxcuis[i] if i < len(rxcuis) else []
        if s_list:
            for j, s in enumerate(s_list):
                code = c_list[j] if j < len(c_list) else name
                items.append(CodeItem(system="RxTerms", code=str(code), display=f"{name} – {s}", extras={"strength": s}))
        else:
            items.append(CodeItem(**base))
    return items

async def search_hcpcs(session: httpx.AsyncClient, terms: str, limit: int = 25) -> List[CodeItem]:
    url = f"{BASE}/hcpcs/v3/search"
    data = await _fetch(session, url, {"terms": terms, "df": "code,display", "ef": "long_desc,obsolete", "count": str(limit)})
    items = _zip_ctss(data)
    return [CodeItem(system="HCPCS", **it) for it in items]

async def search_ucum(session: httpx.AsyncClient, terms: str, limit: int = 25) -> List[CodeItem]:
    url = f"{BASE}/ucum/v3/search"
    data = await _fetch(session, url, {"terms": terms, "count": str(limit)})
    items = _zip_ctss(data)
    return [CodeItem(system="UCUM", **it) for it in items]

async def search_hpo(session: httpx.AsyncClient, terms: str, limit: int = 25) -> List[CodeItem]:
    url = f"{BASE}/hpo/v3/search"
    data = await _fetch(session, url, {"terms": terms, "df": "id,name", "sf": "id,name,synonym.term", "count": str(limit)})
    items = _zip_ctss(data)
    return [CodeItem(system="HPO", **it) for it in items]
