import pytest
from fastapi.testclient import TestClient
from app.main import app
import io

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["qdrant"] == "connected"

def test_health_has_collections():
    response = client.get("/health")
    data = response.json()
    assert "gitlab_handbook" in data["collections"]

def test_chat_endpoint_returns_200():
    response = client.post("/chat", json={"question": "What is GitLab anti-harassment policy?"})
    assert response.status_code == 200

def test_chat_response_structure():
    response = client.post("/chat", json={"question": "What is GitLab anti-harassment policy?"})
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "num_chunks" in data

def test_chat_answer_not_empty():
    response = client.post("/chat", json={"question": "What is GitLab anti-harassment policy?"})
    data = response.json()
    assert len(data["answer"]) > 0

def test_chat_num_chunks_positive():
    response = client.post("/chat", json={"question": "What is GitLab anti-harassment policy?"})
    data = response.json()
    assert data["num_chunks"] > 0

def test_ingest_valid_md_file():
    content = b"# Test Document\n\nThis is a test markdown file for ingestion."
    file = io.BytesIO(content)
    response = client.post(
        "/ingest",
        files={"file": ("test.md", file, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_indexed"] > 0

def test_ingest_invalid_file_type():
    content = b"some content"
    file = io.BytesIO(content)
    response = client.post(
        "/ingest",
        files={"file": ("test.exe", file, "application/octet-stream")}
    )
    assert response.status_code == 400