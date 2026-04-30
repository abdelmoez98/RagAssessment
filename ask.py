#!/usr/bin/env python3
"""CLI: python ask.py "question" [--version SimpleNode_3T|SingleNode_12T|MultiNode_12T]"""

import argparse
import importlib
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from shared.state import snip

AGENT_MAP = {
    "SimpleNode_3T": "SimpleNode_3T.agent",
    "SingleNode_12T": "SingleNode_12T.agent",
    "MultiNode_12T": "MultiNode_12T.agent",
}


def main():
    default = os.getenv("DEFAULT_AGENT", "MultiNode_12T")
    parser = argparse.ArgumentParser(description="Policy RAG Assistant")
    parser.add_argument("question", nargs="+", help="Policy question")
    parser.add_argument("--version", "-v", choices=list(AGENT_MAP), default=default,
                        help=f"Agent version (default from .env: {default})")
    args = parser.parse_args()

    question = " ".join(args.question)
    module = importlib.import_module(AGENT_MAP[args.version])
    runner = module.run_agent

    print(f"\n[{args.version}] Question: {question}\n")

    start = time.time()
    state = runner(question)
    elapsed = round(time.time() - start, 1)

    print(f"ANSWER:\n{state.get('final_answer', 'No answer')}")

    conf = state.get("confidence", {})
    overall = conf.get("overall") if isinstance(conf, dict) else conf
    if overall:
        print(f"\nCONFIDENCE: {overall}")

    citations = state.get("citations", [])
    if citations:
        print(f"\nCITATIONS ({len(citations)}):")
        for c in citations:
            if isinstance(c, dict):
                doc = c.get("doc_id", "?")
                snippet = c.get("snippet", "")
            elif isinstance(c, str):
                doc = c
                snippet = ""
            else:
                doc = str(c)
                snippet = ""
            s = snip(snippet, 80)
            print(f"  - {doc}: {s}")

    trace = state.get("trace", [])
    if trace:
        print(f"\nTRACE ({len(trace)} steps):")
        for step in trace[-15:]:
            node = step.get("node", "") or step.get("tool", "")
            action = step.get("action", "") or step.get("verdict", "") or step.get("type", "")
            summary = snip(step.get("result_summary", ""), 80)
            if node:
                print(f"  [{node}] {action} {summary}")

    errors = state.get("errors", [])
    if errors:
        print(f"\nERRORS:")
        for e in errors:
            print(f"  - {e}")

    print(f"\n⏱  {elapsed}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
