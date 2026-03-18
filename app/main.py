# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter
from app.config import QDRANT_URL, COLLECTION_NAME

app = FastAPI(title="GitLab Handbook HR Assistant")

# Setup Prometheus instrumentator
Instrumentator().instrument(app).expose(app)

# Custom metrics
CHUNKS_RETRIEVED = Histogram(
    "hr_assistant_chunks_retrieved",
    "Number of chunks retrieved per query",
    buckets=[1, 2, 3, 5, 8, 10, 15, 20]
)

QUERY_ERRORS = Counter(
    "hr_assistant_query_errors_total",
    "Total number of query errors"
)

INGEST_CHUNKS = Histogram(
    "hr_assistant_ingest_chunks_total",
    "Number of chunks indexed per ingest",
    buckets=[1, 10, 50, 100, 200, 500, 1000]
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list
    num_chunks: int

class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    status: str

@app.get("/health")
def health():
    from qdrant_client import QdrantClient
    try:
        client = QdrantClient(url=QDRANT_URL)
        collections = [c.name for c in client.get_collections().collections]
        return {"status": "ok", "qdrant": "connected", "collections": collections}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    from app.pipeline import query
    result = query(request.question)
    # Record metrics
    CHUNKS_RETRIEVED.observe(result["num_chunks"])
    if result["answer"].startswith("Sorry"):
        QUERY_ERRORS.inc()
    return ChatResponse(**result)

@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    from app.ingest_api import ingest_file
    try:
        result = await ingest_file(file)
        INGEST_CHUNKS.observe(result["chunks_indexed"])
        return IngestResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")