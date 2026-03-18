# app/pipeline.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from app.config import *
from app.hybrid_search import HybridRetriever
import logging

logger = logging.getLogger(__name__)

GENERATION_PROMPT = PromptTemplate.from_template("""You are a helpful HR assistant for GitLab.

Answer the question ONLY based on the context provided below.

Do NOT use any prior knowledge. If the answer is not in the context, say
"I don't have information about that in the handbook."

Respond in the same language as the question (English or Bahasa Indonesia).

Context:
{context}

Question: {question}

Answer:""")

TRANSLATION_PROMPT = PromptTemplate.from_template("""Translate the following question to English.
If it is already in English, return it as is. Return ONLY the translated question, nothing else.

Question: {question}
Translation:""")

def load_pipeline():
    """Initialize hybrid retriever and LLM once at startup."""
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME, embedding=embeddings
    )

    print("Loading chunks for BM25 index...")
    from app.ingestion import load_documents, chunk_documents
    docs = load_documents()
    chunks = chunk_documents(docs)

    print("Building hybrid retriever...")
    retriever = HybridRetriever(vector_store=vector_store, chunks=chunks, top_k=10)

    print("Loading LLM...")
    llm = OllamaLLM(
        model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.3
    )

    translation_chain = TRANSLATION_PROMPT | llm | StrOutputParser()
    generation_chain = GENERATION_PROMPT | llm | StrOutputParser()

    print("Hybrid pipeline ready.")
    return retriever, translation_chain, generation_chain

retriever, translation_chain, generation_chain = load_pipeline()

def query(question: str) -> dict:
    """Translate query if needed, retrieve with hybrid search, generate answer."""
    try:
        # Translate ke English untuk retrieval
        translated = translation_chain.invoke({"question": question}).strip()
        logger.info(f"Original: '{question}' | Translated: '{translated}'")

        # Hybrid search pakai translated query
        docs = retriever.invoke(translated)
        context = "\n\n".join([d.page_content for d in docs])
        sources = list(set([d.metadata.get("source", "") for d in docs]))

        # Generate jawaban dalam bahasa asli pertanyaan
        answer = generation_chain.invoke({"context": context, "question": question})
        logger.info(f"Sources: {sources}")
        return {"answer": answer, "sources": sources, "num_chunks": len(docs)}
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {"answer": "Sorry, an error occurred. Please try again.",
                "sources": [], "num_chunks": 0}