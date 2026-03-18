# app/config.py
import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "gitlab_handbook"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DOCS_DIR = "documents"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200