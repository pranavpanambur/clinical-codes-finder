
import pytest
from backend.app.rank import score, rank_top
from backend.app.models import CodeItem

def test_rank():
    items = [
        CodeItem(system="ICD-10-CM", code="E11.9", display="Type 2 diabetes mellitus without complications"),
        CodeItem(system="LOINC", code="2345-7", display="Glucose [Mass/volume] in Blood"),
    ]
    picked = rank_top("blood sugar test", items)
    assert picked and picked[0].display.lower().find("glucose") >= 0
