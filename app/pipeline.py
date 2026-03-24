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
from langchain_groq import ChatGroq

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
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )

    print("Connecting to Qdrant...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
    )
    vector_store = QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME, embedding=embeddings
    )

    print("Loading chunks from Qdrant for BM25 index...")
    # Ganti load dari documents/ dengan scroll dari Qdrant
    all_points = []
    offset = None
    while True:
        result, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        all_points.extend(result)
        if offset is None:
            break

    from langchain_core.documents import Document
    chunks = [
        Document(
            page_content=point.payload.get("page_content", ""),
            metadata=point.payload.get("metadata", {})
        )
        for point in all_points
        if point.payload.get("page_content")
    ]
    print(f"Loaded {len(chunks)} chunks from Qdrant for BM25")

    print("Building hybrid retriever...")
    retriever = HybridRetriever(vector_store=vector_store, chunks=chunks, top_k=20)

    print("Loading LLM...")
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.3
    )

    translation_chain = TRANSLATION_PROMPT | llm | StrOutputParser()
    generation_chain = GENERATION_PROMPT | llm | StrOutputParser()

    print("Hybrid pipeline ready.")
    return retriever, translation_chain, generation_chain, llm

retriever, translation_chain, generation_chain, llm = load_pipeline()

# ─── Tambahkan di bawah load_pipeline() yang sudah ada ──────────────

from app.agentic_pipeline import build_agentic_graph
from app.agent_state import AgentState

# Build graph sekali saat startup
agentic_graph = build_agentic_graph(retriever=retriever, llm=llm)

async def query_stream_agentic(
    question: str,
    history: list = []
) -> AsyncGenerator[str, None]:
    try:
        initial_state: AgentState = {
            "question": question,
            "translated_query": "",
            "domains": [],
            "is_broad": False,
            "docs": [],
            "retry_count": 0,
            "answer": "",
            "history": history,
            "sources": [],
        }

        # Invoke graph untuk retrieve + classify
        final_state = agentic_graph.invoke(initial_state)
        
        sources = list(set([
            s.replace("\\", "/") for s in final_state['sources']
        ]))
        # Emit sources
        
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'num_chunks': len(final_state['docs'])})}\n\n"
        # Build prompt dari hasil graph
        context = "\n\n".join([d.page_content for d in final_state["docs"]])
        history_text = ""
        if history:
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        if history_text:
            from app.agentic_pipeline import CONVERSATION_PROMPT
            prompt_text = CONVERSATION_PROMPT.format(
                history=history_text,
                context=context,
                question=question
            )
        else:
            from app.agentic_pipeline import GENERATION_PROMPT
            prompt_text = GENERATION_PROMPT.format(
                context=context,
                question=question
            )

        # Stream tokens — handle AIMessageChunk dari ChatGroq
        async for chunk in llm.astream(prompt_text):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Agentic stream error: {e}")
        yield f"data: {json.dumps({'type': 'token', 'content': 'Sorry, an error occurred.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

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

        # # Stream tokens
        # async for chunk in llm.astream(prompt_text):
        #     yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

        async for chunk in llm.astream(prompt_text):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'type': 'token', 'content': 'Sorry, an error occurred.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"