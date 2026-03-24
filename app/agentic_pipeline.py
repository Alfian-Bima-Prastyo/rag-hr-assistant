import json
import logging
from typing import AsyncGenerator
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.agent_state import AgentState

logger = logging.getLogger(__name__)

MAX_RETRIES = 1  # hemat GPU, 1x retry

# ─── Prompt Templates ────────────────────────────────────────────────

TRANSLATION_PROMPT = PromptTemplate.from_template("""Translate the following question to English.
If it is already in English, return it as is. Return ONLY the translated question, nothing else.

Question: {question}
Translation:""")

CLASSIFY_PROMPT = PromptTemplate.from_template("""You are an HR assistant classifier. Analyze the question and respond with ONLY a JSON object.

Available domains:
- people_group: HR policies, onboarding, offboarding, anti-harassment, promotions, time-off
- hiring: interview process, talent acquisition, job descriptions, referrals
- leadership: 1-1 meetings, underperformance, coaching, delegation, managing conflict
- total_rewards: compensation, benefits, stock options, incentives
- communication: confidentiality, async communication, Slack usage

Respond with ONLY this JSON, no other text:
{{
  "domains": ["domain1"],
  "is_broad": true or false,
  "reformulated_query": "more specific version if broad, else same query"
}}

Rules:
- domains: pick 1-2 most relevant domains
- is_broad: true if the query uses vague verbs like "manage", "handle", "deal with", "approach to", "how does gitlab do X" without asking for a specific policy or process. Example broad: "how does gitlab manage underperformance", "what is gitlab approach to hiring". Example specific: "what is a PIP at gitlab", "what are warning signs of underperformance"
- reformulated_query: if is_broad=true, rewrite as a specific policy/process question. Example: "how does gitlab manage underperformance" → "GitLab performance improvement plan warning signs underperformance process"
- If out of context (stock prices, weather, etc), use domains=[]

Question: {question}
JSON:""")

EVALUATE_PROMPT = PromptTemplate.from_template("""You are evaluating if retrieved context is sufficient to answer a question.
Respond with ONLY "sufficient" or "insufficient". Nothing else.

Question: {question}

Retrieved context ({num_chunks} chunks):
{context_preview}

Is the context sufficient to answer the question?
Answer:""")

GENERATION_PROMPT = PromptTemplate.from_template("""You are a helpful HR assistant for GitLab.

Answer the question ONLY based on the context provided below.
Do NOT use any prior knowledge. If the answer is not in the context, say
"I don't have information about that in the handbook."

Respond in the same language as the question (English or Bahasa Indonesia).

Context:
{context}

Question: {question}

Answer:""")

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


# ─── Graph Builder ────────────────────────────────────────────────────

def build_agentic_graph(retriever, llm):
    """
    Build LangGraph StateGraph.
    Dipanggil sekali saat startup, sama seperti load_pipeline() lama.
    """

    translation_chain = TRANSLATION_PROMPT | llm | StrOutputParser()
    classify_chain = CLASSIFY_PROMPT | llm | StrOutputParser()
    evaluate_chain = EVALUATE_PROMPT | llm | StrOutputParser()
    generation_chain = GENERATION_PROMPT | llm | StrOutputParser()

    # ── Node functions ──────────────────────────────────────────────

    def translate_query(state: AgentState) -> AgentState:
        translated = translation_chain.invoke(
            {"question": state["question"]}
        ).strip()
        logger.info(f"[translate_query] '{state['question']}' → '{translated}'")
        return {**state, "translated_query": translated}

    def classify_query(state: AgentState) -> AgentState:
        raw = classify_chain.invoke(
            {"question": state["translated_query"]}
        ).strip()
        logger.info(f"[classify_query] raw output: {raw}")
        try:
            # Bersihkan jika LLM wrap dengan ```json
            clean = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            domains = parsed.get("domains", [])
            is_broad = parsed.get("is_broad", False)
            reformulated = parsed.get("reformulated_query", state["translated_query"])
        except Exception as e:
            logger.warning(f"[classify_query] JSON parse failed: {e}, fallback no filter")
            domains = []
            is_broad = False
            reformulated = state["translated_query"]

        # Kalau broad, pakai reformulated query untuk retrieve
        query_to_use = reformulated if is_broad else state["translated_query"]
        logger.info(f"[classify_query] domains={domains}, is_broad={is_broad}, query='{query_to_use}'")
        return {**state, "domains": domains, "is_broad": is_broad, "translated_query": query_to_use}

    # def retrieve(state: AgentState) -> AgentState:
    #     docs = retriever.invoke_with_filter(
    #         query=state["translated_query"],
    #         domains=state["domains"] if state["domains"] else None
    #     )
    #     sources = list(set([d.metadata.get("source", "") for d in docs]))
    #     logger.info(f"[retrieve] got {len(docs)} docs from domains={state['domains']}")
    #     return {**state, "docs": docs, "sources": sources}

    def retrieve(state: AgentState) -> AgentState:
        docs = retriever.invoke_with_filter(
            query=state["translated_query"],
            domains=state["domains"] if state["domains"] else None
        )
        # Normalisasi path sebelum deduplicate
        sources = list(set([
            d.metadata.get("source", "").replace("\\", "/") for d in docs
        ]))
        logger.info(f"[retrieve] got {len(docs)} docs from domains={state['domains']}")
        return {**state, "docs": docs, "sources": sources}

    def evaluate_context(state: AgentState) -> AgentState:
        # Kalau sudah retry atau tidak ada docs, langsung lanjut
        if state["retry_count"] >= MAX_RETRIES or not state["docs"]:
            return {**state}

        context_preview = "\n\n".join(
            [d.page_content[:300] for d in state["docs"][:3]]
        )
        verdict = evaluate_chain.invoke({
            "question": state["translated_query"],
            "num_chunks": len(state["docs"]),
            "context_preview": context_preview
        }).strip().lower()

        logger.info(f"[evaluate_context] verdict='{verdict}', retry_count={state['retry_count']}")
        return {**state, "eval_verdict": verdict}

    def generate_answer(state: AgentState) -> AgentState:
        context = "\n\n".join([d.page_content for d in state["docs"]])
        history_text = ""
        if state.get("history"):
            for msg in state["history"]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        if history_text:
            prompt_text = CONVERSATION_PROMPT.format(
                history=history_text,
                context=context,
                question=state["question"]
            )
        else:
            prompt_text = GENERATION_PROMPT.format(
                context=context,
                question=state["question"]
            )

        # answer = llm.invoke(prompt_text)
        # logger.info(f"[generate_answer] answer length={len(answer)}")
        # return {**state, "answer": answer}
        result = llm.invoke(prompt_text)
    # ChatGroq return AIMessage, ambil .content
        answer = result.content if hasattr(result, 'content') else str(result)
        return {**state, "answer": answer}

    # ── Conditional edge function ───────────────────────────────────

    def should_retry(state: AgentState) -> str:
        """
        Return 'retrieve' untuk retry, 'generate_answer' untuk lanjut.
        """
        verdict = state.get("eval_verdict", "sufficient")
        retry_count = state.get("retry_count", 0)

        if verdict == "insufficient" and retry_count < MAX_RETRIES:
            logger.info(f"[should_retry] → retry (count={retry_count})")
            # Broadening query untuk retry: hapus domain filter
            return "retry_retrieve"
        else:
            logger.info(f"[should_retry] → generate")
            return "generate_answer"

    def retry_retrieve(state: AgentState) -> AgentState:
        """Retrieve ulang tanpa domain filter, query lebih broad."""
        docs = retriever.invoke(state["translated_query"])
        sources = list(set([d.metadata.get("source", "") for d in docs]))
        logger.info(f"[retry_retrieve] broadened search, got {len(docs)} docs")
        return {
            **state,
            "docs": docs,
            "sources": sources,
            "retry_count": state["retry_count"] + 1,
            "domains": []  # reset domain untuk retry
        }

    # ── Build graph ─────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("translate_query", translate_query)
    graph.add_node("classify_query", classify_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("evaluate_context", evaluate_context)
    graph.add_node("retry_retrieve", retry_retrieve)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("translate_query")

    graph.add_edge("translate_query", "classify_query")
    graph.add_edge("classify_query", "retrieve")
    graph.add_edge("retrieve", "evaluate_context")

    graph.add_conditional_edges(
        "evaluate_context",
        should_retry,
        {
            "retry_retrieve": "retry_retrieve",
            "generate_answer": "generate_answer"
        }
    )

    graph.add_edge("retry_retrieve", "evaluate_context")
    graph.add_edge("generate_answer", END)

    return graph.compile()