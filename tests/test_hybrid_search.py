from app.hybrid_search import reciprocal_rank_fusion
from langchain_core.documents import Document

def make_doc(content: str, source: str = "test.md") -> Document:
    return Document(page_content=content, metadata={"source": source})

def test_rrf_combines_results():
    vector_docs = [make_doc("anti-harassment policy"), make_doc("leave types")]
    bm25_docs = [make_doc("leave types"), make_doc("offboarding process")]
    result = reciprocal_rank_fusion(vector_docs, bm25_docs, top_n=3)
    assert len(result) <= 3

def test_rrf_boosts_overlap():
    """Doc yang muncul di kedua list harus dapat score lebih tinggi."""
    shared = make_doc("shared document")
    vector_docs = [shared, make_doc("only in vector")]
    bm25_docs = [shared, make_doc("only in bm25")]
    result = reciprocal_rank_fusion(vector_docs, bm25_docs, top_n=3)
    # shared doc harus di posisi pertama karena ada di kedua list
    assert result[0].page_content == "shared document"

def test_rrf_top_n_limit():
    vector_docs = [make_doc(f"doc {i}") for i in range(10)]
    bm25_docs = [make_doc(f"doc {i}") for i in range(10)]
    result = reciprocal_rank_fusion(vector_docs, bm25_docs, top_n=5)
    assert len(result) == 5

def test_rrf_empty_vector():
    bm25_docs = [make_doc("only bm25 result")]
    result = reciprocal_rank_fusion([], bm25_docs, top_n=3)
    assert len(result) == 1
    assert result[0].page_content == "only bm25 result"

def test_rrf_empty_bm25():
    vector_docs = [make_doc("only vector result")]
    result = reciprocal_rank_fusion(vector_docs, [], top_n=3)
    assert len(result) == 1
    assert result[0].page_content == "only vector result"