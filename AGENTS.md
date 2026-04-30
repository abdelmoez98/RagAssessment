# AGENTS.md — Meridian Policy RAG Assistant

## Setup & run order

```bash
cp .env.example .env          # Set GOOGLE_API_KEY, optionally DEFAULT_AGENT
pip install -r requirements.txt
python ingest.py              # Once. Idempotent — clears & rebuilds index
python ask.py "your question"
python evaluate.py            # Runs all 3 agents against 15 questions → evaluation/
```

## Architecture — 3 LangGraph versions

### SimpleNode_3T (V3)
Single node, 3 tools (`retrieve`, `get_document_metadata`, `check_contradictions`), self-critique in prompt.

### SingleNode_12T (V2)  
Single node, all 12 tools, guided prompt walks LLM through retrieve→classify→compose→evaluate cycle.

### MultiNode_12T (V1, default)
5 LangGraph nodes: **Supervisor** → Retriever → Classifier → Composer → Critic. Supervisor maintains pending queue, Critic reports issues → Supervisor rebuilds queue and reroutes to any previous stage.

| Node | Model | Does |
|---|---|---|
| Supervisor | `gemma-4-31b-it` | Routes based on pending queue, rebuilds on Critic feedback |
| Retriever | `gemma-4-26b-a4b-it` | Search + metadata + superseded filtering |
| Classifier | `gemma-4-26b-a4b-it` | Contradiction detection + question classification |
| Composer | `gemma-4-26b-a4b-it` | Answer composition (single/multi/contradiction/refusal) |
| Critic | `gemma-4-31b-it` | Grounding + compliance check → `back_to_retriever|classifier|composer|approved` |

## Evaluation

`python evaluate.py` produces:

```
evaluation/
├── SimpleNode_3T/results.json   ← 15 questions, assessment format
├── SingleNode_12T/results.json
├── MultiNode_12T/results.json
└── comparison.json              ← side-by-side + consolidated metrics
```

Each `results.json` entry: `answer`, `citations`, `confidence` (6 dimensions), `trace`.

Metrics per question: RAGAS (faithfulness, relevancy, context precision), DeepEval GEval (contradiction/supersession/refusal), token usage (from Gemini `usage_metadata`), latency.

## Project structure

```
RagAssessment/
├── shared/                      # Shared across all versions
│   ├── config.py, state.py, tools.py, eval_metrics.py
├── SimpleNode_3T/agent.py       # V3: 3 tools
├── SingleNode_12T/agent.py      # V2: 12 tools, single node
├── MultiNode_12T/agent.py       # V1: Supervisor + 4 workers
├── ingestion/                   # Parser, chunker, indexer
├── ask.py                       # CLI, reads DEFAULT_AGENT from .env
├── evaluate.py                  # Full pipeline
├── ingest.py                    # One-shot ingestion
├── Dockerfile
├── .env.example
└── requirements.txt
```

## Model assignments

| Model | RPM | Used by |
|---|---|---|
| `gemma-4-31b-it` | 15 | Supervisor, Critic, check_contradictions tool |
| `gemma-4-26b-a4b-it` | 15 | Retriever, Classifier, Composer (+ their tools) |
| `gemini-3.1-flash-lite` | 10 | RAGAS + DeepEval evaluation |

## Gotchas

- `policy_corpus/` is gitignored. Copy from `Meridian_Policy_Corpus/policy_corpus/` or symlink.
- First query is 2-3× slower (embedding model cold start).
- All LLM calls wrapped: 429 → `time.sleep(60)` → retry (×3).
- Monitoring: Gemini `usage_metadata` gives exact token counts per call — no extra dependency.
- `evaluate.py` runs RAGAS + DeepEval as LLM judges using the EVAL_MODEL.
- Never commit `.env` or API keys.
