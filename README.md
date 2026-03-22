# RAG HR Assistant

---

## Overview

HR Document Assistant berbasis **Retrieval-Augmented Generation (RAG)** yang menjawab pertanyaan seputar kebijakan HR GitLab menggunakan dokumen dari [GitLab Handbook](https://handbook.gitlab.com/) sebagai knowledge base.

Fitur utama:
- Bisa tanya dalam **Bahasa Indonesia atau English** — query otomatis ditranslate sebelum retrieval
- **Hybrid Search** (Vector + BM25 + RRF) untuk retrieval yang lebih akurat dari vector-only
- **Streaming response** — jawaban muncul bertahap, tidak perlu tunggu
- **Conversation memory** — mendukung follow-up question dalam satu sesi

---

## Job Requirements Coverage

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| FastAPI | ✅ | REST API dengan `/health`, `/chat`, `/chat/stream`, `/ingest` |
| Vector DB (Qdrant) | ✅ | 3177 chunks ter-index dengan cosine similarity |
| LangChain | ✅ | Pipeline, retriever, prompt template |
| Ollama local LLM | ✅ | `qwen2.5:7b-instruct` via `langchain_ollama` |
| Docker | ✅ | Docker Compose: FastAPI + Qdrant + Prometheus + Grafana |
| Ingestion pipeline | ✅ | DirectoryLoader + chunking + `/ingest` endpoint |
| Unit testing | ✅ | 26/26 Pytest passed |
| Monitoring | ✅ | Prometheus metrics + Grafana dashboard |
| Hybrid Search | ✅ | BM25 + Vector + Reciprocal Rank Fusion (RRF) |
| Chat UI | ✅ | Chainlit dengan streaming dan conversation memory |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│          Chainlit (port 8001) — streaming + memory      │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────────┐
│                   FastAPI (port 8000)                    │
│      /health   /chat   /chat/stream   /ingest  /metrics  │
└──────┬──────────────┬───────────────────────────────────┘
       │              │
┌──────▼──────┐ ┌─────▼──────────────────────────────────┐
│   Qdrant    │ │           RAG Pipeline                   │
│ (port 6333) │ │  1. Query Translation (ID → EN)          │
│  3177 chunks│ │  2. Hybrid Search (Vector + BM25 + RRF)  │
└─────────────┘ │  3. Context Assembly + History           │
                │  4. LLM Generation (Ollama) — streaming  │
                └────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│              Monitoring Stack                            │
│     Prometheus (port 9090) + Grafana (port 3000)        │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Komponen | Tool | Notes |
|----------|------|-------|
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
| Chat UI | Chainlit | streaming + conversation memory |
| Testing | Pytest | 26/26 passed |
| Containerization | Docker Compose | FastAPI + Qdrant + Prometheus + Grafana |

---

## Project Structure

```
HR-Assistant/
├── app/
│   ├── config.py              # Konfigurasi (URL, model, chunk size)
│   ├── ingestion.py           # Pipeline ingestion dokumen
│   ├── pipeline.py            # RAG pipeline + query translation + streaming
│   ├── hybrid_search.py       # BM25 + Vector + RRF implementation
│   ├── ingest_api.py          # Handler untuk /ingest endpoint
│   └── main.py                # FastAPI app + Prometheus metrics
├── documents/
│   ├── people-group/          # HR policies, onboarding, offboarding
│   ├── hiring/                # Interview process, talent acquisition
│   ├── leadership/            # 1-1, underperformance, coaching
│   ├── total-rewards/         # Compensation, benefits, stock options
│   └── communication/         # Confidentiality levels, async communication
├── tests/
│   ├── test_api.py            # Test endpoint FastAPI
│   ├── test_ingestion.py      # Test ingestion pipeline
│   ├── test_pipeline.py       # Test retrieval & generation
│   ├── test_hybrid_search.py  # Test RRF algorithm
|   └── test_ragas.py          # RAGAS evaluation — faithfulness, relevancy, recal
├── monitoring/
│   ├── prometheus.yml         # Prometheus scrape config
│   └── grafana/               # Grafana provisioning + dashboard
├── chainlit_app.py            # Chainlit UI
├── Dockerfile
├── docker-compose.yml
├── ragas_results.json
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
git clone https://github.com/Alfian-Bima-Prastyo/rag-hr-assistant.git
cd rag-hr-assistant
```

### 2. Jalankan Ollama
```bash
ollama run qwen2.5:7b-instruct
```

### 3. Jalankan dengan Docker Compose
```bash
docker compose up --build
```

Tunggu sampai muncul:
```
hr-assistant | Ingestion complete — 3177 chunks indexed
hr-assistant | Application startup complete.
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

### POST /chat/stream
Server-Sent Events endpoint untuk streaming response. Digunakan oleh Chainlit UI.

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
- **Source**: [GitLab Handbook](https://handbook.gitlab.com/)
- **License**: Creative Commons — legal untuk portofolio non-komersial
- **Total files**: 142 markdown files
- **Total chunks**: 3177 chunks
- **Folders**: people-group, hiring, leadership, total-rewards, communication

### Chunking Strategy
```python
RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "]
)
```

 Chunk size 1200 dipilih berdasarkan analisis struktur dokumen GitLab Handbook yang memiliki rata-rata section sekitar 800-1500 karakter. Jika menggunakan chunk kecil akan menghasilkan context yang tidak lengkap.
 
### Hybrid Search (RRF)
```
Query → Vector Search (Qdrant) ──┐
                                  ├── RRF Fusion → Top 7 chunks
Query → BM25 Search ─────────────┘
```
Reciprocal Rank Fusion score: `1 / (k + rank + 1)` untuk setiap list, dan kemudian dijumlahkan.

### Bilingual Support
Query Bahasa Indonesia ditranslate ke English sebelum retrieval menggunakan LLM, lalu jawaban digenerate dalam bahasa asli pertanyaan.

### Conversation Memory
History percakapan disimpan per session di Chainlit (max 5 turn terakhir) dan dikirim ke pipeline sebagai context tambahan. Ini memungkinkan follow-up question tanpa perlu repeat context.

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

## Evaluation (RAGAS)

RAG pipeline dievaluasi menggunakan RAGAS dengan 5 test cases dan `qwen2.5:7b-instruct` sebagai evaluator.

| Metrik | Score |
|--------|-------|
| **Faithfulness** | 0.97 | 
| **Answer Relevancy** | 0.96 |
| **Context Precision** | 0.68 |
| **Context Recall** | 0.75 | 

---

## Known Limitations

| Issue | Notes |
|-------|-------|
| Broad/generic query kadang tidak retrieve dokumen yang tepat | Query spesifik memberikan hasil lebih baik. Contoh: "What are warning signs of underperformance?" lebih akurat dari "How does GitLab manage underperformance?" |
| Knowledge base hanya GitLab Handbook | Dataset ini dipilih karena lisensi Creative Commons — legal untuk portofolio non-komersial |

---

## License

Dataset dari [GitLab Handbook](https://handbook.gitlab.com/), digunakan untuk keperluan non-komersial (Creative Commons).
