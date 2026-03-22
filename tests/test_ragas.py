import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from app.pipeline import retriever, generation_chain, translation_chain
import json

evaluator_llm = LangchainLLMWrapper(
    OllamaLLM(model="qwen2.5:7b-instruct", base_url="http://localhost:11434", temperature=0)
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
)

TEST_CASES = [
    {
        "question": "What is GitLab anti-harassment policy?",
        "ground_truth": "GitLab prohibits harassment including sexual harassment, power harassment, and harassment related to pregnancy or family care leave. Team members can report to Chief People Officer, Team Member Relations, or People Business Partner."
    },
    {
        "question": "What types of leave are available at GitLab?",
        "ground_truth": "GitLab offers Flexible PTO, Parental Leave (16 weeks paid), Military Leave, Emergency Leave, and Sick Leave among others."
    },
    {
        "question": "How does the promotion process work at GitLab?",
        "ground_truth": "GitLab handles promotions through twice-per-year calibration processes. Managers work with People Business Partner to submit promotions through Workday with a promotion document."
    },
    {
        "question": "What is the offboarding process at GitLab?",
        "ground_truth": "The offboarding process involves Team Member Relations notifying Payroll, Security, and Stock Administration. An offboarding issue is created at end of last working day. Critical systems must be deprovisioned within 24 hours."
    },
    {
        "question": "What is the interview process at GitLab?",
        "ground_truth": "GitLab interview process includes Application, Screening Call, Assessment, Technical Interview, Team Interviews, References and TMRG Connection, and Offer and Background Screening."
    },
]

def run_pipeline(question: str):
    translated = translation_chain.invoke({"question": question}).strip()
    docs = retriever.invoke(translated)
    context = "\n\n".join([d.page_content for d in docs])
    sources = [d.page_content for d in docs]
    answer = generation_chain.invoke({"context": context, "question": question})
    return answer, sources

def build_dataset():
    questions, answers, contexts, ground_truths = [], [], [], []
    print("Running pipeline for each test case...")
    for i, tc in enumerate(TEST_CASES):
        print(f"  [{i+1}/{len(TEST_CASES)}] {tc['question'][:50]}...")
        answer, retrieved_contexts = run_pipeline(tc["question"])
        questions.append(tc["question"])
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(tc["ground_truth"])
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

def test_ragas_evaluation():
    """RAGAS evaluation — jalankan dengan: pytest tests/test_ragas.py -v -s"""
    print("\nBuilding evaluation dataset...")
    dataset = build_dataset()

    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(timeout=120, max_workers=1)
    )

    print("\nRAGAS Evaluation Results ")
    print(results)

    results_dict = results.to_pandas().to_dict(orient="records")
    with open("ragas_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print("\nResults saved to ragas_results.json")

    df = results.to_pandas()
    faithfulness_score = df["faithfulness"].mean()
    answer_relevancy_score = df["answer_relevancy"].mean()
    context_recall_score = df["context_recall"].mean()

    print(f"\nFinal scores:")
    print(f"  Faithfulness:     {faithfulness_score:.4f}")
    print(f"  Answer Relevancy: {answer_relevancy_score:.4f}")
    print(f"  Context Recall:   {context_recall_score:.4f}")

    assert faithfulness_score >= 0.8, f"Faithfulness too low: {faithfulness_score}"
    assert answer_relevancy_score >= 0.8, f"Answer relevancy too low: {answer_relevancy_score}"
    assert context_recall_score >= 0.6, f"Context recall too low: {context_recall_score}"

if __name__ == "__main__":
    test_ragas_evaluation()