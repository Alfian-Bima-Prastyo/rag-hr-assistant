import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "gitlab_handbook"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DOCS_DIR = "documents"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Groq — ganti Ollama
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Ollama tetap ada sebagai fallback lokal
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")