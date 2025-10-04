
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class CodeItem(BaseModel):
    system: str
    code: str
    display: str
    extras: Dict[str, object] = Field(default_factory=dict)

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    results: List[CodeItem]
    summary: str
