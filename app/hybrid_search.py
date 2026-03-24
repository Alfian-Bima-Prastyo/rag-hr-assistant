import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)

DOMAIN_PATHS = {
    "people_group": "people-group",
    "hiring": "hiring",
    "leadership": "leadership",
    "total_rewards": "total-rewards",
    "communication": "communication",
}

def reciprocal_rank_fusion(
    vector_docs: List[Document],
    bm25_docs: List[Document],
    k: int = 60,
    top_n: int = 7
) -> List[Document]:
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
    def __init__(self, vector_store, chunks: List[Document], top_k: int = 10):
        self.vector_store = vector_store
        self.chunks = chunks
        self.top_k = top_k

        logger.info("Building BM25 index...")
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = top_k
        logger.info(f"BM25 index built with {len(chunks)} documents")

    def _source_matches_domain(self, source: str, domain_paths: list) -> bool:
        """Handle Windows backslash dan Unix forward slash."""
        normalized = source.replace("\\", "/")
        return any(f"/{path}/" in normalized for path in domain_paths)

    def invoke(self, query: str) -> List[Document]:
        """Hybrid search tanpa filter — sama seperti sebelumnya."""
        vector_docs = self.vector_store.similarity_search(query, k=self.top_k)
        bm25_docs = self.bm25_retriever.invoke(query)
        fused = reciprocal_rank_fusion(vector_docs, bm25_docs, k=60, top_n=7)
        logger.info(f"Hybrid search: vector={len(vector_docs)}, bm25={len(bm25_docs)}, fused={len(fused)}")
        return fused

    def invoke_with_filter(
        self,
        query: str,
        domains: Optional[List[str]] = None
    ) -> List[Document]:
        if not domains:
            return self.invoke(query)

        domain_paths = [DOMAIN_PATHS[d] for d in domains if d in DOMAIN_PATHS]
        if not domain_paths:
            return self.invoke(query)

        # Vector search lebih banyak, filter post-hoc di Python
        vector_docs_all = self.vector_store.similarity_search(query, k=self.top_k * 3)
        vector_docs = [
            doc for doc in vector_docs_all
            if self._source_matches_domain(doc.metadata.get("source", ""), domain_paths)
        ][:self.top_k]

        # BM25 filter manual
        filtered_chunks = [
            c for c in self.chunks
            if self._source_matches_domain(c.metadata.get("source", ""), domain_paths)
        ]

        if filtered_chunks:
            bm25_filtered = BM25Retriever.from_documents(filtered_chunks)
            bm25_filtered.k = self.top_k
            bm25_docs = bm25_filtered.invoke(query)
        else:
            bm25_docs = []

        fused = reciprocal_rank_fusion(vector_docs, bm25_docs, k=60, top_n=7)
        logger.info(
            f"Filtered search domains={domains}: "
            f"vector_all={len(vector_docs_all)}, vector_filtered={len(vector_docs)}, "
            f"bm25_chunks={len(filtered_chunks)}, fused={len(fused)}"
        )

        if not fused:
            logger.warning("Filter returned 0 docs, fallback to unfiltered")
            return self.invoke(query)

        return fused