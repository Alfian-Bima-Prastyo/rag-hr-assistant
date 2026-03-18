# app/hybrid_search.py
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(
    vector_docs: List[Document],
    bm25_docs: List[Document],
    k: int = 60,
    top_n: int = 5
) -> List[Document]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion.
    RRF score = 1/(k + rank + 1) for each result list, then sum scores.
    """
    scores = {}

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content
        if key not in scores:
            scores[key] = {"doc": doc, "score": 0.0}
        scores[key]["score"] += 1.0 / (k + rank + 1)

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        if key not in scores:
            scores[key] = {"doc": doc, "score": 0.0}
        scores[key]["score"] += 1.0 / (k + rank + 1)

    sorted_docs = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs[:top_n]]


class HybridRetriever:
    """
    Hybrid retriever combining Qdrant vector search and BM25Retriever dari LangChain.
    """

    def __init__(self, vector_store, chunks: List[Document], top_k: int = 10):
        self.vector_store = vector_store
        self.top_k = top_k

        # Build BM25 index pakai LangChain BM25Retriever
        logger.info("Building BM25 index...")
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = top_k
        logger.info(f"BM25 index built with {len(chunks)} documents")

    def invoke(self, query: str) -> List[Document]:
        """Run hybrid search with RRF fusion."""
        # Vector search
        vector_docs = self.vector_store.similarity_search(query, k=self.top_k)

        # BM25 search
        bm25_docs = self.bm25_retriever.invoke(query)

        # Fuse dengan RRF
        fused = reciprocal_rank_fusion(
            vector_docs=vector_docs,
            bm25_docs=bm25_docs,
            k=60,
            top_n=5
        )

        logger.info(f"Hybrid search: vector={len(vector_docs)}, bm25={len(bm25_docs)}, fused={len(fused)}")
        return fused
