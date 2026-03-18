# RAG HR Assistant

> **Portfolio Project** — Production RAG system built with FastAPI, Qdrant, LangChain, and Docker.

---

## Overview

HR Document Assistant berbasis **Retrieval-Augmented Generation (RAG)** yang menjawab pertanyaan seputar kebijakan HR GitLab menggunakan [GitLab Handbook](https://handbook.gitlab.com/) sebagai knowledge base.

Sistem ini mendukung query dalam **Bahasa Indonesia dan English** dengan automatic query translation, serta menggunakan **Hybrid Search (Vector + BM25 + RRF)** untuk retrieval yang lebih akurat.

---

## Job Requirements Coverage

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| FastAPI | ✅ | REST API dengan `/health`, `/chat`, `/ingest` |
| Vector DB (Qdrant) | ✅ | 1505 chunks ter-index dengan cosine similarity |
| LangChain | ✅ | Pipeline, retriever, prompt template |
| Ollama local LLM | ✅ | `qwen2.5:7b-instruct` via `langchain_ollama` |
| Docker | ✅ | Docker Compose: FastAPI + Qdrant + Prometheus + Grafana |
| Ingestion pipeline | ✅ | DirectoryLoader + chunking + `/ingest` endpoint |
| Unit testing | ✅ | 26/26 Pytest passed |
| Monitoring | ✅ | Prometheus metrics + Grafana dashboard |
| Hybrid Search | ✅ | BM25 + Vector + Reciprocal Rank Fusion (RRF) |
| Chat UI | ✅ | Chainlit interface |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│                  Chainlit (port 8001)                    │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────────┐
│                   FastAPI (port 8000)                    │
│         /health    /chat    /ingest    /metrics          │
└──────┬──────────────┬───────────────────────────────────┘
       │              │
┌──────▼──────┐ ┌─────▼──────────────────────────────────┐
│   Qdrant    │ │           RAG Pipeline                   │
│ (port 6333) │ │  1. Query Translation (ID → EN)          │
│  1505 chunks│ │  2. Hybrid Search (Vector + BM25 + RRF)  │
└─────────────┘ │  3. Context Assembly                     │
                │  4. LLM Generation (Ollama)              │
                └────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│              Monitoring Stack                            │
│     Prometheus (port 9090) + Grafana (port 3000)        │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Komponen | Tool | Versi/Notes |
|----------|------|-------------|
| API Framework | FastAPI | dengan uvicorn |
| Vector DB | Qdrant | Docker: `qdrant/qdrant` |
| Vector Store | QdrantVectorStore | `langchain_qdrant` |
| LLM | OllamaLLM | `langchain_ollama` |
| Embedding | HuggingFaceEmbeddings | `all-MiniLM-L6-v2` (384 dim) |
| Document Loader | DirectoryLoader + TextLoader | `langchain_community` |
| Text Splitter | RecursiveCharacterTextSplitter | `langchain_text_splitters` |
| BM25 | BM25Retriever | `langchain_community.retrievers` |
| Ollama Model | qwen2.5:7b-instruct | local, port 11434 |
| Monitoring | Prometheus + Grafana | metrics via `/metrics` endpoint |
| Chat UI | Chainlit | terhubung ke FastAPI |
| Testing | Pytest | 26/26 passed |
| Containerization | Docker Compose | FastAPI + Qdrant + Prometheus + Grafana |

---

## Project Structure

```
HR-Assistant/
├── app/
│   ├── config.py           # Konfigurasi (URL, model, chunk size)
│   ├── ingestion.py        # Pipeline ingestion dokumen
│   ├── pipeline.py         # RAG pipeline + query translation
│   ├── ingest_api.py       # Handler untuk /ingest endpoint
│   └── main.py             # FastAPI app + Prometheus metrics
├── documents/
│   └── people-group/       # GitLab Handbook (70 files, 1505 chunks)
├── tests/
│   ├── test_api.py         # Test endpoint FastAPI
│   ├── test_ingestion.py   # Test ingestion pipeline
│   ├── test_pipeline.py    # Test retrieval & generation
│   └── test_hybrid_search.py # Test RRF algorithm
├── monitoring/
│   ├── prometheus.yml      # Prometheus scrape config
│   └── grafana/            # Grafana provisioning
├── chainlit_app.py         # Chainlit UI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Cara Menjalankan

### Prerequisites
- Docker Desktop
- Ollama dengan model `qwen2.5:7b-instruct`
- Python 3.10+

### 1. Clone Repository
```bash
git clone https://github.com/username/hr-assistant.git
cd hr-assistant
```

### 2. Jalankan Ollama
```bash
ollama run qwen2.5:7b-instruct
```

### 3. Jalankan dengan Docker Compose
```bash
docker compose up --build
```


### 4. Jalankan Chat UI (terminal terpisah)
```bash
pip install chainlit
chainlit run chainlit_app.py --port 8001
```

### 5. Akses Services
| Service | URL |
|---------|-----|
| Chat UI | http://localhost:8001 |
| FastAPI Docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

---

## API Endpoints

### GET /health
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "qdrant": "connected", "collections": ["gitlab_handbook"]}
```

### POST /chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is GitLab anti-harassment policy?"}'
```
```json
{
  "answer": "GitLab's anti-harassment policy...",
  "sources": ["documents/people-group/anti-harassment.md"],
  "num_chunks": 5
}
```

### POST /ingest
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@new_policy.md"
```
```json
{
  "filename": "new_policy.md",
  "chunks_indexed": 42,
  "status": "success"
}
```

---

## RAG Pipeline Details

### Knowledge Base
- **Source**: [GitLab Handbook](https://handbook.gitlab.com/) — folder `people-group`
- **License**: Creative Commons — legal untuk portofolio non-komersial
- **Total files**: 70 markdown files
- **Total chunks**: 1505 chunks

### Chunking Strategy
```python
RecursiveCharacterTextSplitter(
    chunk_size=1200,      # Ditentukan berdasarkan analisis dokumen:
    chunk_overlap=200,    # rata-rata section GitLab Handbook ~800-1500 karakter
    separators=["\n## ", "\n### ", "\n\n", "\n", " "]
)
```
> **Insight**: Chunk size 1200 dipilih berdasarkan analisis struktur dokumen GitLab Handbook yang memiliki rata-rata section sekitar 800-1500 karakter. Jika menggunakan chunk kecil akan menghasilkan context yang tidak lengkap; c

### Hybrid Search (RRF)
```
Query → Vector Search (Qdrant) ──┐
                                  ├── RRF Fusion → Top 5 chunks
Query → BM25 Search ─────────────┘
```
Reciprocal Rank Fusion score: `1 / (k + rank + 1)` untuk setiap list, dan kemudian dijumlahkan.

### Bilingual Support
Query dalam Bahasa Indonesia diterjemahkan ke English sebelum retrieval, kemudian LLM menjawab dalam bahasa asli pertanyaan.

---

## Monitoring

Metrics yang di-track via Prometheus:
- `http_requests_total` — total request per endpoint
- `http_request_duration_seconds` — latency per endpoint
- `hr_assistant_chunks_retrieved` — jumlah chunks per query
- `hr_assistant_query_errors_total` — total error
- `hr_assistant_ingest_chunks_total` — chunks per ingestion

---

## Testing

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

```
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_chat_endpoint_returns_200 PASSED
tests/test_api.py::test_ingest_valid_md_file PASSED
tests/test_ingestion.py::test_load_documents_count PASSED
tests/test_ingestion.py::test_chunk_size_within_limit PASSED
tests/test_pipeline.py::test_query_correct_source PASSED
tests/test_hybrid_search.py::test_rrf_boosts_overlap PASSED
... 26 passed
```

---

## Known Limitations

| Issue | Status | Notes |
|-------|--------|-------|
| Query broad ("what should I do") kurang akurat | Known | Query spesifik memberikan hasil lebih baik |
| Knowledge base hanya terbatas folder `people-group` | Planned | Akan ditambahkan folder lain (hiring, total-rewards) |

---

## License

Dataset: [GitLab Handbook](https://handbook.gitlab.com/) — Creative Commons License
