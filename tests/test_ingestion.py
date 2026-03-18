# tests/test_ingestion.py
import pytest
from app.ingestion import load_documents, chunk_documents, strip_frontmatter

def test_strip_frontmatter_removes_yaml():
    text = "---\ntitle: Test\ndate: 2024\n---\nActual content here"
    result = strip_frontmatter(text)
    assert result == "Actual content here"

def test_strip_frontmatter_no_yaml():
    text = "No frontmatter here"
    result = strip_frontmatter(text)
    assert result == "No frontmatter here"

def test_load_documents_returns_list():
    docs = load_documents()
    assert isinstance(docs, list)
    assert len(docs) > 0

def test_load_documents_count():
    docs = load_documents()
    assert len(docs) == 70

def test_chunk_documents_returns_chunks():
    docs = load_documents()
    chunks = chunk_documents(docs)
    assert isinstance(chunks, list)
    assert len(chunks) > 0

def test_chunk_size_within_limit():
    docs = load_documents()
    chunks = chunk_documents(docs)
    # Semua chunks harus <= chunk_size + overlap (toleransi splitter)
    for chunk in chunks:
        assert len(chunk.page_content) <= 800

def test_chunks_have_source_metadata():
    docs = load_documents()
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] != ""