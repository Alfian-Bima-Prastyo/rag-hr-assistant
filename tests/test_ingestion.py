import pytest
from app.ingestion import load_documents, chunk_documents, strip_frontmatter

def test_strip_frontmatter_removes_yaml():
    text = "---\ntitle: Test\ndate: 2024\n---\nActual content"
    result = strip_frontmatter(text)
    assert result == "Actual content"

def test_strip_frontmatter_no_yaml():
    text = "No frontmatter"
    result = strip_frontmatter(text)
    assert result == "No frontmatter"

def test_load_documents_returns_list():
    docs = load_documents()
    assert isinstance(docs, list)
    assert len(docs) > 0

def test_load_documents_count():
    docs = load_documents()
    assert len(docs) > 0
    assert len(docs) >= 100  

def test_chunk_documents_returns_chunks():
    docs = load_documents()
    chunks = chunk_documents(docs)
    assert isinstance(chunks, list)
    assert len(chunks) > 0

def test_chunk_size_within_limit():
    docs = load_documents()
    chunks = chunk_documents(docs)
    from app.config import CHUNK_SIZE, CHUNK_OVERLAP
    max_allowed = CHUNK_SIZE + CHUNK_OVERLAP + 200  
    for chunk in chunks:
        assert len(chunk.page_content) <= max_allowed


def test_chunks_have_source_metadata():
    docs = load_documents()
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] != ""