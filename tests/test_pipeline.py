# tests/test_pipeline.py
import pytest
from app.pipeline import query

def test_query_returns_dict():
    result = query("What is GitLab anti-harassment policy?")
    assert isinstance(result, dict)

def test_query_has_required_keys():
    result = query("What is GitLab anti-harassment policy?")
    assert "answer" in result
    assert "sources" in result
    assert "num_chunks" in result

def test_query_retrieves_chunks():
    result = query("What is GitLab anti-harassment policy?")
    assert result["num_chunks"] > 0

def test_query_sources_not_empty():
    result = query("What is GitLab anti-harassment policy?")
    assert len(result["sources"]) > 0

def test_query_correct_source():
    result = query("What is GitLab anti-harassment policy?")
    sources = [s.replace("\\", "/") for s in result["sources"]]
    assert any("anti-harassment" in s for s in sources)

def test_query_out_of_context():
    result = query("What is the stock price of GitLab today?")
    answer = result["answer"].lower()
    # LLM harus jawab tidak tahu, bukan hallucinate
    assert any(phrase in answer for phrase in [
        "don't have information",
        "not in the context",
        "tidak memiliki informasi",
        "tidak ada informasi"
    ])