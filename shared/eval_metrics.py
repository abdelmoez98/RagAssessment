"""RAGAS + DeepEval evaluation metrics and monitoring extraction."""

import json

from shared.state import AgentState, get_client, generate
from shared.config import EVAL_MODEL, RAGAS_MODEL


def extract_monitoring(state: AgentState) -> dict:
    total_input = 0
    total_output = 0
    total_latency = 0.0
    models_seen = set()

    for step in state.get("trace", []):
        usage = step.get("usage", {})
        if usage:
            total_input += usage.get("prompt_token_count", 0)
            total_output += usage.get("candidates_token_count", 0)
            total_latency += usage.get("latency_ms", 0)
            if usage.get("model"):
                models_seen.add(usage["model"])

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "total_latency_ms": round(total_latency, 1),
        "models_used": sorted(models_seen),
        "llm_calls": sum(1 for s in state.get("trace", []) if s.get("usage")),
    }


def run_ragas(question: str, answer: str, contexts: list[str], language: str = "en") -> dict:
    try:
        from ragas import evaluate, SingleTurnSample
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecisionWithoutReference
        from ragas.llms import LangchainLLMWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=RAGAS_MODEL))
        metrics = [Faithfulness(llm=llm), AnswerRelevancy(llm=llm), ContextPrecisionWithoutReference(llm=llm)]
        if language == "ar":
            for m in metrics:
                try:
                    m.adapt("arabic")
                except Exception:
                    pass

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        results = evaluate(data=[sample], metrics=metrics)
        scores = results.to_pandas().iloc[0].to_dict()

        return {
            "faithfulness": round(scores.get("faithfulness", 0.0), 4),
            "answer_relevancy": round(scores.get("answer_relevancy", 0.0), 4),
            "context_precision": round(scores.get("context_precision_without_reference", 0.0), 4),
        }
    except Exception as e:
        return {"faithfulness": None, "answer_relevancy": None, "context_precision": None, "ragas_error": str(e)[:200]}


def run_deepeval_geval(question: str, answer: str, contexts: list[str], question_type: str) -> dict:
    client = get_client()

    scores = {
        "contradiction_handling": None,
        "supersession_handling": None,
        "refusal_correctness": None,
    }

    if question_type == "contradiction":
        result, _, _ = generate(client, EVAL_MODEL, (
            f"Evaluate if this answer PROPERLY SURFACES a contradiction.\n"
            f"Question: {question}\nAnswer: {answer}\nContexts: {json.dumps(contexts)}\n\n"
            "Score 0-1: (1) Identifies the contradiction (+0.4), (2) Presents both sides (+0.3), "
            "(3) Does NOT silently pick one side (+0.3). Return: {\"score\": 0.0-1.0, \"reason\": \"...\"}"
        ))
        try:
            scores["contradiction_handling"] = round(json.loads(result.replace("```json", "").replace("```", "").strip()).get("score", 0.5), 4)
        except Exception:
            scores["contradiction_handling"] = 0.5

    if question_type == "supersession":
        result, _, _ = generate(client, EVAL_MODEL, (
            f"Evaluate if this answer uses the CURRENT version of policies.\n"
            f"Question: {question}\nAnswer: {answer}\nContexts: {json.dumps(contexts)}\n\n"
            "Score 0-1: Did it use current (non-superseded) docs? Return: {\"score\": 0.0-1.0, \"reason\": \"...\"}"
        ))
        try:
            scores["supersession_handling"] = round(json.loads(result.replace("```json", "").replace("```", "").strip()).get("score", 0.5), 4)
        except Exception:
            scores["supersession_handling"] = 0.5

    if question_type == "out_of_scope":
        result, _, _ = generate(client, EVAL_MODEL, (
            f"Evaluate if this answer PROPERLY REFUSES (no guessing).\n"
            f"Question: {question}\nAnswer: {answer}\nContexts: {json.dumps(contexts)}\n\n"
            "Score 0-1: (1) States answer not in corpus (+0.5), (2) Does NOT hallucinate (+0.5). "
            "Return: {\"score\": 0.0-1.0, \"reason\": \"...\"}"
        ))
        try:
            scores["refusal_correctness"] = round(json.loads(result.replace("```json", "").replace("```", "").strip()).get("score", 0.5), 4)
        except Exception:
            scores["refusal_correctness"] = 0.5

    return scores
