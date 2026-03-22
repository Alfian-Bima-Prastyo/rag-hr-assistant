from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import AsyncGenerator
from qdrant_client import QdrantClient
from app.config import *
from app.hybrid_search import HybridRetriever
import logging
import json

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

CONVERSATION_PROMPT = PromptTemplate.from_template("""You are a helpful HR assistant for GitLab.

Answer the question ONLY based on the context provided below.
Do NOT use any prior knowledge. If the answer is not in the context, say
"I don't have information about that in the handbook."

Respond in the same language as the question (English or Bahasa Indonesia).

Previous conversation:
{history}

Context:
{context}

Question: {question}

Answer:""")

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
    retriever = HybridRetriever(vector_store=vector_store, chunks=chunks, top_k=20)

    print("Loading LLM...")
    llm = OllamaLLM(
        model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.3
    )

    translation_chain = TRANSLATION_PROMPT | llm | StrOutputParser()
    generation_chain = GENERATION_PROMPT | llm | StrOutputParser()

    print("Hybrid pipeline ready.")
    return retriever, translation_chain, generation_chain, llm

retriever, translation_chain, generation_chain, llm = load_pipeline()

def query(question: str) -> dict:
    try:
        translated = translation_chain.invoke({"question": question}).strip()
        logger.info(f"Translated query: {translated}")
        docs = retriever.invoke(translated)
        
        context = "\n\n".join([d.page_content for d in docs])
        sources = list(set([d.metadata.get("source", "") for d in docs]))
        answer = generation_chain.invoke({"context": context, "question": question})
        
        return {"answer": answer, "sources": sources, "num_chunks": len(docs)}
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {"answer": "Sorry, an error occurred. Please try again.",
                "sources": [], "num_chunks": 0}
    
async def query_stream(question: str, history: list = []) -> AsyncGenerator[str, None]:
    try:
        # Format history
        history_text = ""
        if history:
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        # Translate
        translated = translation_chain.invoke({"question": question}).strip()

        # Retrieve
        docs = retriever.invoke(translated)
        context = "\n\n".join([d.page_content for d in docs])
        sources = list(set([d.metadata.get("source", "") for d in docs]))
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'num_chunks': len(docs)})}\n\n"

        if history_text:
            prompt_text = CONVERSATION_PROMPT.format(
                history=history_text,
                context=context,
                question=question
            )
        else:
            prompt_text = GENERATION_PROMPT.format(
                context=context,
                question=question
            )

        # Stream tokens
        async for chunk in llm.astream(prompt_text):
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'type': 'token', 'content': 'Sorry, an error occurred.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"