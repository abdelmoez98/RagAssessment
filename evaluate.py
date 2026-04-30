#!/usr/bin/env python3
"""Batch eval: python evaluate.py → evaluation/{agent}/results.json + evaluation/comparison.json"""

import importlib
import json
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from shared.config import POLICY_CORPUS_DIR
from shared.state import snip

AGENTS = {
    # "SimpleNode_3T": "SimpleNode_3T.agent",
    # "SingleNode_12T": "SingleNode_12T.agent",
    "MultiNode_12T": "MultiNode_12T.agent",
}


def load_questions():
    path = os.path.join(POLICY_CORPUS_DIR, "eval_questions.json")
    with open(path) as f:
        return json.load(f)["questions"]


def run_agent_metrics(module_path: str, question: str, language: str):
    mod = importlib.import_module(module_path)
    state = mod.run_agent(question, language)
    from shared.eval_metrics import extract_monitoring

    monitoring = extract_monitoring(state)
    return state, monitoring


def evaluate():
    questions = load_questions()
    os.makedirs("evaluation", exist_ok=True)

    all_comparison = []
    consolidated = {}

    for agent_name, module_path in AGENTS.items():
        agent_dir = f"evaluation/{agent_name}"
        os.makedirs(agent_dir, exist_ok=True)

        results = []
        total_input = 0
        total_output = 0
        total_latency = 0.0
        total_confidence = 0.0
        errors = 0
        ragas_sums = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
        }
        ragas_count = 0
        deepeval_sums = {
            "contradiction_handling": 0.0,
            "supersession_handling": 0.0,
            "refusal_correctness": 0.0,
        }
        deepeval_counts = {
            "contradiction_handling": 0,
            "supersession_handling": 0,
            "refusal_correctness": 0,
        }
        qcount = 0

        print(f"\n--- {agent_name} ---")

        for i, q in enumerate(questions):
            qid = q["id"]
            qtype = q.get("type", "")
            question = q["question"]
            lang = q.get("expected_language", "en")

            print(
                f"\n  [{i + 1}/{len(questions)}] {qid} ({qtype})... ",
                end="",
                flush=True,
            )
            q_start = time.time()

            try:
                state, monitoring = run_agent_metrics(module_path, question, lang)
            except Exception as e:
                print(f"ERROR: {str(e)[:60]}")
                errors += 1
                continue

            elapsed = round(time.time() - q_start, 1)
            conf = state.get("confidence", {})
            c_overall = conf.get("overall") if isinstance(conf, dict) else conf
            c_overall = c_overall if c_overall is not None else 0
            total_confidence += c_overall
            total_input += monitoring["total_input_tokens"]
            total_output += monitoring["total_output_tokens"]
            total_latency += monitoring["total_latency_ms"]

            answer = state.get("final_answer", "")
            contexts = [json.dumps(r) for r in state.get("search_results", [])] or [
                answer
            ]

            # RAGAS
            try:
                from shared.eval_metrics import run_ragas

                ragas = run_ragas(question, answer, contexts, lang)
            except Exception:
                ragas = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                }
            for k in ragas_sums:
                if ragas.get(k) is not None:
                    ragas_sums[k] += ragas[k]
            ragas_count += 1

            # DeepEval
            try:
                from shared.eval_metrics import run_deepeval_geval

                de = run_deepeval_geval(question, answer, contexts, qtype)
            except Exception:
                de = {
                    "contradiction_handling": None,
                    "supersession_handling": None,
                    "refusal_correctness": None,
                }
            for k in deepeval_sums:
                if de.get(k) is not None:
                    deepeval_sums[k] += de[k]
                    deepeval_counts[k] += 1

            entry = {
                "question_id": qid,
                "type": qtype,
                "question": question,
                "answer_language": lang,
                "answer": answer,
                "citations": state.get("citations", []),
                "confidence": conf,
                "trace": state.get("trace", []),
            }
            results.append(entry)

            # Comparison entry
            comp_entry = {
                "agent_name": agent_name,
                "answer": {
                    "text": snip(answer, 500),
                    "citations": state.get("citations", []),
                },
                "metrics": {
                    "confidence": c_overall,
                    **ragas,
                    **de,
                    "input_tokens": monitoring["total_input_tokens"],
                    "output_tokens": monitoring["total_output_tokens"],
                    "latency_ms": monitoring["total_latency_ms"],
                    "time_s": elapsed,
                },
            }
            all_comparison.append(
                {
                    "question_id": qid,
                    "type": qtype,
                    "question": question,
                    **{agent_name: comp_entry},
                }
            )

            qcount += 1
            print(f"conf={c_overall:.2f} {elapsed}s")

        # Write per-agent results
        with open(f"{agent_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Consolidated
        n = qcount if qcount else 1
        consolidated[agent_name] = {
            "questions_completed": qcount,
            "errors": errors,
            "avg_confidence": round(total_confidence / n, 2),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "avg_latency_ms": round(total_latency / n, 1),
            "avg_ragas_faithfulness": round(ragas_sums["faithfulness"] / ragas_count, 4)
            if ragas_count
            else None,
            "avg_ragas_answer_relevancy": round(
                ragas_sums["answer_relevancy"] / ragas_count, 4
            )
            if ragas_count
            else None,
            "avg_ragas_context_precision": round(
                ragas_sums["context_precision"] / ragas_count, 4
            )
            if ragas_count
            else None,
            "avg_deepeval_contradiction": round(
                deepeval_sums["contradiction_handling"]
                / deepeval_counts["contradiction_handling"],
                4,
            )
            if deepeval_counts["contradiction_handling"]
            else None,
            "avg_deepeval_supersession": round(
                deepeval_sums["supersession_handling"]
                / deepeval_counts["supersession_handling"],
                4,
            )
            if deepeval_counts["supersession_handling"]
            else None,
            "avg_deepeval_refusal": round(
                deepeval_sums["refusal_correctness"]
                / deepeval_counts["refusal_correctness"],
                4,
            )
            if deepeval_counts["refusal_correctness"]
            else None,
        }

    # Merge comparison: group by question_id
    by_qid = {}
    for entry in all_comparison:
        qid = entry["question_id"]
        if qid not in by_qid:
            by_qid[qid] = {k: v for k, v in entry.items() if k not in AGENTS}
        agent_name = entry.get("agent_name") or next(k for k in AGENTS if k in entry)
        by_qid[qid][agent_name] = entry[agent_name]

    questions_out = list(by_qid.values())

    with open("evaluation/comparison.json", "w") as f:
        json.dump(
            {"questions": questions_out, "consolidated_metrics": consolidated},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{'=' * 60}")
    print("CONSOLIDATED:")
    for name, m in consolidated.items():
        print(
            f"  {name}: {m['questions_completed']}/15 done, avg_conf={m['avg_confidence']}, "
            f"tokens_in={m['total_input_tokens']}, tokens_out={m['total_output_tokens']}, "
            f"avg_lat={m['avg_latency_ms']}ms"
        )
    print(f"\nSaved to evaluation/")


if __name__ == "__main__":
    evaluate()
