from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class AgentState(TypedDict):
    question: str
    translated_query: str
    domains: List[str]          # domain yang dipilih LLM
    is_broad: bool              # apakah query broad/generic
    docs: List[Document]        # hasil retrieve
    retry_count: int            # jumlah retry self-reflect
    answer: str
    history: List[dict]
    sources: List[str]