import os
import re
import tempfile
from pathlib import Path
from fastapi import UploadFile
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.config import *

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

def strip_frontmatter(text: str) -> str:
    if text.startswith('---'):
        end = text.find('---', 3)
        if end != -1:
            return text[end+3:].strip()
    return text

async def ingest_file(file: UploadFile) -> dict:
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load Doc
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file.filename
                if ext == ".md":
                    doc.page_content = strip_frontmatter(doc.page_content)

        # Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "]
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No content could be extracted from the file.")

        # Embed & upsert to Qdrant
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        client = QdrantClient(url=QDRANT_URL)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
        vector_store.add_documents(chunks)

        return {
            "filename": file.filename,
            "chunks_indexed": len(chunks),
            "status": "success"
        }

    finally:
        os.unlink(tmp_path)