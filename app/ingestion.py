import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.config import *
import re

def strip_frontmatter(text):
    if text.startswith('---'):
        end = text.find('---', 3)
        if end != -1:
            return text[end+3:].strip()
    return text

def load_documents():
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        recursive=True,
        show_progress=True
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "]
    )
    chunks = []
    for doc in docs:
        doc.page_content = strip_frontmatter(doc.page_content)
        split = splitter.split_documents([doc])
        chunks.extend(split)
    print(f"Created {len(chunks)} chunks")
    return chunks

def index_documents():
    print("Starting ingestion pipeline...")

    docs   = load_documents()
    chunks = chunk_documents(docs)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    # client = QdrantClient(url=QDRANT_URL)
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        timeout = 60
    )
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' created")

    # print("Embedding and upserting chunks...")
    # QdrantVectorStore.from_documents(
    #     documents=chunks,
    #     embedding=embeddings,
    #     client=client,
    #     collection_name=COLLECTION_NAME,
    # )
    print("Embedding and upserting chunks...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    vector_store.add_documents(chunks)

    batch_size = 50
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"Uploaded {min(i + batch_size, total)}/{total} chunks")

    print(f"Ingestion complete — {total} chunks indexed")
    return total

if __name__ == "__main__":
    index_documents()