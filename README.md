# Meridian Policy RAG Assistant

An agentic RAG assistant that answers employee questions about Meridian Consulting's corporate policy documents. Built in 3 LangGraph versions with increasing sophistication: a baseline 3-tool agent, a single-agent 12-tool system, and a multi-agent supervisor architecture with 5 specialized nodes.

## What it does

Given a natural-language question about corporate policy (in English or Arabic), the system:

- **Retrieves** relevant policy chunks using hybrid search (dense + BM25)
- **Classifies** the question type (single-document, composition, contradiction, out-of-scope)
- **Detects contradictions** between documents and surfaces them rather than picking one side
- **Respects supersession chains** — answers from current policy versions by default
- **Composes** answers with sentence-level citations to source documents
- **Self-critiques** every answer for grounding, citation accuracy, completeness, and source currency
- **Refuses** gracefully when the answer isn't in the corpus
- **Handles cross-lingual** queries — Arabic questions are auto-detected, translated for retrieval, and answers are returned in Arabic

The corpus contains 43 fictional policy documents (markdown, PDF, DOCX) with deliberate contradictions, supersession chains, and cross-document composition requirements.

## Quick start

```bash
cp .env.example .env                    # Set GOOGLE_API_KEY
source .venv/bin/activate               # Activate uv virtual environment
pip install -r requirements.txt         # Install dependencies
python ingest.py                        # One-shot — parses, chunks, indexes
python ask.py "What is the standard notice period at Meridian?"
python evaluate.py                      # Runs all 3 agents vs. 15 test questions
```

The default agent is `MultiNode_12T`. Switch via `.env`:

```env
DEFAULT_AGENT=SimpleNode_3T    # or SingleNode_12T or MultiNode_12T
```

Or via CLI: `python ask.py --version SimpleNode_3T "question"`

## Docker

```bash
# Build the image
docker compose build

# Start interactive container
docker compose up -d

# Attach to the container
docker compose exec rag bash

# Inside the container:
python ingest.py
python ask.py "What is the standard notice period at Meridian?"
python evaluate.py
# Results appear in ./evaluation/ on the host machine

# Stop when done
docker compose down
```

**Volume mounts:**
| Host path | Container path | Purpose |
|---|---|---|
| `.env` | `/app/.env` | API key and model config (via `env_file`) |
| `./policy_corpus` | `/app/policy_corpus` | Policy documents to index |
| `./evaluation` | `/app/evaluation` | Evaluation results output |
| `./store` | `/app/store` | ChromaDB + BM25 index (persisted across restarts) |

**First-time setup in Docker:**
```bash
docker compose up -d
docker compose exec rag bash
# Your corpus should be in ./policy_corpus/ on the host (volume-mounted)
python ingest.py
python ask.py "What is the standard notice period?"
```
Dependencies are installed in the image — no venv activation needed inside the container.

## Architecture

The system has **3 versions** implemented as independent LangGraph graphs, all sharing the same tools, state model, and ingestion pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                      SHARED LAYER                                │
│  shared/tools.py   — 14 tools (retrieve, metadata, compose...)  │
│  shared/state.py   — AgentState TypedDict, LLM client, retry    │
│  shared/config.py  — 12 model env vars, paths, tuning knobs     │
│  shared/eval_metrics.py — RAGAS + DeepEval + token tracking     │
│  ingestion/        — Parser (md/pdf/docx), chunker, indexer     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│ SimpleNode_3T    │ │ SingleNode_12T   │ │ MultiNode_12T        │
│                  │ │                  │ │                      │
│ 1 node           │ │ 1 node           │ │ 5 nodes              │
│ 3 tools          │ │ 14 tools         │ │ 14 tools             │
│ self-critique in │ │ guided prompt    │ │ Supervisor routes    │
│ prompt only      │ │ walks through    │ │ via pending queue    │
│                  │ │ all 6 steps      │ │ Critic → any stage   │
└──────────────────┘ └──────────────────┘ └──────────────────────┘
```

### MultiNode Architecture (Primary)

```
                         ┌──────────────┐
                         │  SUPERVISOR  │ ← Manages pending queue
                         └──┬───┬───┬──┘
          ┌─────────────────┘   │   └─────────────────┐
          ▼                     ▼                      ▼
    ┌──────────┐         ┌───────────┐          ┌──────────┐
    │RETRIEVER │         │CLASSIFIER │          │ COMPOSER │
    │          │         │           │          │          │
    │ retrieve │         │ check_    │          │ compose  │
    │ metadata │         │ contra-   │          │ single/  │
    │ filter   │         │ dictions  │          │ multi/   │
    │ super-   │         │ classify  │          │ contra/  │
    │ seded    │         │ question  │          │ refusal  │
    └────┬─────┘         └─────┬─────┘          └────┬─────┘
         │                     │                     │
         └──────────┬──────────┴──────────┬──────────┘
                    │                     │
                    ▼                     │
              ┌──────────┐                │
              │  CRITIC  │◄───────────────┘
              │          │
              │ evaluate │─── approved ──→ END
              │ answer + │
              │ evaluate │─── rejected ──→ SUPERVISOR
              │ comply   │     (rebuilds queue)
              └──────────┘
```

**Supervisor** — Rule-based router. Maintains a `pending_steps` queue. On Critic rejection, rebuilds the queue:
- `back_to_retriever` → retriever → classifier → composer → critic
- `back_to_classifier` → classifier → composer → critic
- `back_to_composer` → composer → critic

**Retriever** — Hybrid search (ChromaDB + BM25, RRF merge). Checks metadata, marks superseded docs (doesn't remove them). If the question is in Arabic, translates to English first.

**Classifier** — Calls `check_contradictions` (if 2+ docs overlap) and `classify_question` to determine: `single_doc`, `composition`, `contradiction`, or `out_of_scope`.

**Composer** — Based on classification, calls the appropriate compose tool. All outputs include citations. If original question was Arabic, translates the final answer back to Arabic.

**Critic** — Runs `evaluate_answer` (grounding, citations, completeness, language match, thinking-text detection) and `evaluate_compliance` (source currency, contradiction handling, refusal correctness). Returns a verdict: `approved`, `back_to_retriever`, `back_to_classifier`, or `back_to_composer` with actionable fix instructions.

### State Model

```python
class AgentState(TypedDict):
    question: str              # Original question
    language: str              # "en" | "ar"
    question_en: str           # English translation (if Arabic)
    search_results: list[dict] # Retrieved chunks with metadata
    contradictions: dict       # check_contradictions output
    question_type: str         # single_doc | composition | contradiction | out_of_scope
    draft_answer: str          # Composer output
    citations: list[dict]      # [{doc_id, snippet}]
    critique: dict             # Critic evaluation results
    critic_verdict: str        # approved | back_to_retriever | back_to_classifier | back_to_composer
    final_answer: str          # Final answer text
    confidence: dict           # Multi-dimension confidence scores
    pending_steps: list[str]   # Supervisor queue
    iteration: int             # Revision loop counter
    trace: list[dict]          # Full execution trace
```

## Model Assignments

All models are configurable via `.env`. Defaults use separate rate limit pools:

| Env Var | Default | RPM Pool | Used By |
|---|---|---|---|
| `CONTRADICTION_MODEL` | `gemma-4-31b-it` | 15 | `check_contradictions` tool (all 3 agents) |
| `SINGLE_12T_CRITIC_MODEL` | `gemma-4-31b-it` | 15 | SingleNode_12T evaluate tools |
| `MULTI_CRITIC_MODEL` | `gemma-4-31b-it` | 15 | MultiNode critic node |
| `SIMPLE_3T_MODEL` | `gemma-4-26b-a4b-it` | 15 | SimpleNode agent loop |
| `SINGLE_12T_MODEL` | `gemma-4-26b-a4b-it` | 15 | SingleNode agent loop |
| `SINGLE_12T_CLASSIFY_MODEL` | `gemma-4-26b-a4b-it` | 15 | SingleNode classify tool |
| `SINGLE_12T_COMPOSE_MODEL` | `gemma-4-26b-a4b-it` | 15 | SingleNode compose tools |
| `MULTI_CLASSIFY_MODEL` | `gemma-4-26b-a4b-it` | 15 | MultiNode classifier node |
| `MULTI_COMPOSE_MODEL` | `gemma-4-26b-a4b-it` | 15 | MultiNode composer node |
| `TRANSLATION_MODEL` | `gemma-4-26b-a4b-it` | 15 | Arabic ↔ English translation |
| `EVAL_MODEL` | `gemini-3.1-flash-lite-preview` | 10 | DeepEval GEval custom metrics |
| `RAGAS_MODEL` | `gemini-3.1-flash-lite-preview` | 10 | RAGAS LLM judge |

**Why 31b for routing/critique?** Higher reasoning capacity for multi-document comparison, contradiction detection nuance, and the critique→routing feedback loop. These are the most complex LLM tasks.

**Why 26b for composition/classification?** Good quality for structured composition and classification. Saves tokens vs 31b for high-volume calls (composition runs on every question).

**Why Flash-Lite for eval?** RAGAS and DeepEval judges score simple dimensions — no complex reasoning needed. Lower quality acceptable for quantitative metrics.

### Switch to Gemini Free Tier

Set all model env vars to Gemini free-tier models:

```env
# Replace all gemma-4-* with gemini-2.5-flash (10 RPM, 250 RPD)
SIMPLE_3T_MODEL=gemini-2.5-flash
SINGLE_12T_MODEL=gemini-2.5-flash
SINGLE_12T_CLASSIFY_MODEL=gemini-2.5-flash
SINGLE_12T_COMPOSE_MODEL=gemini-2.5-flash
MULTI_CLASSIFY_MODEL=gemini-2.5-flash
MULTI_COMPOSE_MODEL=gemini-2.5-flash
SINGLE_12T_CRITIC_MODEL=gemini-2.5-flash
MULTI_CRITIC_MODEL=gemini-2.5-flash
CONTRADICTION_MODEL=gemini-2.5-flash
TRANSLATION_MODEL=gemini-2.5-flash

# Flash-Lite for eval (15 RPM, 1000 RPD)
EVAL_MODEL=gemini-2.5-flash-lite-preview
RAGAS_MODEL=gemini-2.5-flash-lite-preview
```

Note: Gemini free tier has tighter rate limits. Expect `evaluate.py` to take 2-3× longer.

## Running evaluate.py

```bash
python evaluate.py
```

This runs **all 3 agents** against the 15 questions in `eval_questions.json` and produces:

```
evaluation/
├── SimpleNode_3T/
│   └── results.json       # 15 entries: answer, citations, confidence, trace
├── SingleNode_12T/
│   └── results.json
├── MultiNode_12T/
│   └── results.json
└── comparison.json        # Side-by-side comparison + consolidated metrics
```

**Metrics collected per question:**

| Metric | Source |
|---|---|
| Agent confidence (6 dimensions) | Critic (MultiNode) or self-score (single-node) |
| `ragas_faithfulness` | RAGAS LLM judge — is answer grounded in contexts? |
| `ragas_answer_relevancy` | RAGAS LLM judge — does answer address the question? |
| `ragas_context_precision` | RAGAS LLM judge — are retrieved chunks useful? |
| `deepeval_contradiction` | DeepEval GEval — did it surface conflict? (only contradiction Qs) |
| `deepeval_supersession` | DeepEval GEval — did it use current version? (only supersession Qs) |
| `deepeval_refusal` | DeepEval GEval — did it refuse properly? (only Q15) |
| `input_tokens` | Gemini `usage_metadata.prompt_token_count` |
| `output_tokens` | Gemini `usage_metadata.candidates_token_count` |
| `latency_ms` | Wall clock per LLM call |

**Token tracking**: Gemini's `usage_metadata` is available on every response — no extra dependency needed. Each `generate()` call captures this and appends it to the trace. `extract_monitoring()` in `evaluate.py` sums them per question.

**Arabic eval**: RAGAS uses `metric.adapt("arabic")` for Arabic-language questions. DeepEval GEval includes a language-match check.

## Walk-through: Contradiction Question (Q10)

> "How many days of paternity leave am I entitled to?"

```
Step 1 — RETRIEVER
  Search: "paternity leave" → 5 docs found
  Key docs: POL-HR-002 (Parental Leave Policy: 5 days)
            POL-HR-003 (Employee Handbook: 7 days)
  Metadata check: both current, no supersession

Step 2 — CLASSIFIER
  check_contradictions(POL-HR-002, POL-HR-003)
  → CONTRADICTION FOUND: 5 days vs 7 days
  classify_question → type: contradiction

Step 3 — COMPOSER
  compose_contradiction(contradiction_report, question)
  → "Policy POL-HR-002 states 5 working days of paid paternity leave,
     but Policy POL-HR-003 states 7 working days.
     Please consult HR for the authoritative answer."

Step 4 — CRITIC
  evaluate_answer: grounding ✓, citations ✓
  evaluate_compliance: contradiction handling ✓ (surfaced, didn't pick)
  → VERDICT: approved

Step 5 — FINAL
  Confidence: 1.0
  Answer returned with both citations
```

## Walk-through: Composition Question (Q06)

> "If I travel from Dubai to Abu Dhabi for a meeting with a UAE government client, what approvals and expense rules apply?"

```
Step 1 — RETRIEVER
  Search 1: "UAE inter-emirate travel government client" → POL-TRAVEL-003
  Search 2: "per diem UAE expenses" → POL-TRAVEL-002
  Search 3: "hotel expense reimbursement" → POL-FIN-003
  Total: 5 unique documents retrieved
  filter_superseded: POL-TRAVEL-001 is superseded → use POL-TRAVEL-001-v2

Step 2 — CLASSIFIER
  check_contradictions: no contradiction found
  classify_question → type: composition (needs 3 documents)

Step 3 — COMPOSER
  compose_multi_doc(5 chunks, question)
  → Synthesized answer with 3 sections:
    1. Approvals: same-day (email) vs overnight (Travel Request Form)
    2. UAE government client: Emirates ID, 30-min early arrival, security clearance
    3. Expenses: hotel cap (800 AED/night), per diem (per POL-TRAVEL-002)
  Citations: POL-TRAVEL-003, POL-TRAVEL-002, POL-FIN-003

Step 4 — CRITIC
  evaluate_answer: 4 sentences, all grounded
  evaluate_compliance: source currency ✓, no contradictions
  → VERDICT: approved

Step 5 — FINAL
  Confidence: 0.95
  Answer with cross-document citations
```

## Retrieval Design

### Hybrid Search

| Component | Technology | Role |
|---|---|---|
| Dense (semantic) | ChromaDB + `all-MiniLM-L6-v2` | Captures conceptual similarity, handles rephrasing |
| Sparse (keyword) | `rank-bm25` | Captures exact policy names, document IDs, threshold values |
| Merge | Reciprocal Rank Fusion (k=60) | Combines both rankings, penalizes documents ranked low in either |

**Why hybrid?** Pure semantic search misses exact matches on policy names (e.g., "POL-HR-007"). Pure keyword misses rephrased questions ("notice period" vs "how long do I have to give notice"). RRF ensures both signals contribute.

### Chunking

`RecursiveCharacterTextSplitter` with separators: `["\n## ", "\n### ", "\n# ", "\n\n", "\n", " "]`

- Splits on markdown headers first (preserves section structure)
- Falls back to paragraph breaks for PDF/DOCX
- 500 characters, 100-character overlap
- Most documents produce 2-4 chunks

### Metadata Filtering

Applied in Python (not ChromaDB `where` — avoids single-operator limitation):

| Filter | How |
|---|---|
| `current_only=true` (default) | Exclude docs with `superseded_by` set |
| `category` | Match against `VALID_CATEGORIES` list, invalid values silently ignored |
| `department` | Match against `VALID_DEPARTMENTS` list |
| `effective_date` | Parse "2024" → "2024-01-01", "2024-03" → "2024-03-01", exclude docs with earlier `effective_date` |

All filters applied to both dense and BM25 results before RRF merge.

## Known Weaknesses

### 1. Arabic Retrieval Quality

The embedding model (`all-MiniLM-L6-v2`) is English-only. Arabic questions must be translated to English before retrieval. Translation errors cascade into retrieval errors — if the translation loses nuance or picks wrong terms, the wrong documents are retrieved.

**Fix**: Use a multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) and index documents in both English and Arabic, or use a cross-lingual retriever.

### 2. MultiNode Hangs on ~40% of Questions

Classifier and composer nodes make internal LLM calls. When rate limits hit or the model returns errors, exceptions can propagate and leave the state machine in an incomplete state. The Supervisor then can't route forward.

**Fix**: Already partially addressed with try/except in classifier and composer nodes. Next: add timeouts, more granular retry, and a "best effort" finalizer that returns partial results rather than nothing.

### 3. Contradiction Detection Misses Cross-Category Documents

`check_contradictions` only compares documents the agent explicitly retrieves. If two conflicting documents live in different categories (e.g., training budget in POL-HR-003 Benefits Handbook vs POL-TRAIN-001 Training Policy), the agent may only retrieve one and miss the contradiction entirely.

**Fix**: Add a "second-pass" retrieval after initial classification — search for the same topic with broader terms and in adjacent categories. Implement structured fact extraction (NER + relation extraction) to identify key-value pairs and cross-reference them.

### 4. Confidence Scores Not Honest in Single-Node Agents

SimpleNode_3T and SingleNode_12T self-report confidence without a dedicated critic. They tend to output `1.0` even for partially correct or incomplete answers. The MultiNode Critic provides more granular scoring but still has blind spots.

**Fix**: Add a lightweight external fact-checker step to single-node agents. Use the `evaluate_answer` tool as a post-hoc check even for SimpleNode_3T.

### 5. PDF/DOCX Chunking Loses Section Structure

PDF and DOCX documents are parsed to plain text, losing heading hierarchy. The `RecursiveCharacterTextSplitter` falls back to paragraph-based splitting, which can cut mid-concept.

**Fix**: Pre-process PDFs with an LLM to add markdown headers, or use a document-aware parser like Unstructured. For DOCX, improve heading detection by also checking font size and bold/italic styles.

## What I'd Build Next (Another Week)

1. **Fix MultiNode hangs** — Add per-node timeouts, circuit breakers, and a degraded-mode finalizer that returns partial answers. This alone would bring MultiNode from 60% completion to 90%+.

2. **Multilingual embeddings** — Swap to `paraphrase-multilingual-MiniLM-L12-v2` for direct Arabic→English cross-lingual retrieval without translation. Removes the translation failure cascade.

3. **Structured fact extraction** — Extract key claims (entity, predicate, value) from each document chunk using an LLM pass at ingestion time. Store as structured metadata alongside embeddings. Enable exact-value contradiction detection without another LLM call.

4. **Full RAGAS + DeepEval in evaluate.py** — Currently both are wrapped in try/except to prevent crashes. Next: configure proper Gemini LLM judges, fix RAGAS LLM wrapper compatibility, and surface evaluation scores in the CLI.

5. **Streaming answers** — Stream the composer output token-by-token to reduce perceived latency. Particularly important for the single-node agents where the user waits for the entire tool-calling loop to complete.

6. **Query expansion** — Given a question, generate 3-5 rephrased search queries to increase recall. Helps with the cross-category contradiction issue and improves retrieval for ambiguous queries.

## Tuning Knobs (.env)

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Required. Google Gemini API key |
| `DEFAULT_AGENT` | `MultiNode_12T` | Which agent to use: `SimpleNode_3T`, `SingleNode_12T`, `MultiNode_12T` |
| `MAX_TOOL_ITERATIONS` | `15` | Max LLM→tool loops per question (single-node agents) |
| `TRACE_SNIPPET_LENGTH` | `-1` | `-1` = full traces, `N` = truncate to N chars |
| `MAX_RETRIES` | `3` | Retry failed API calls |
| `RATE_LIMIT_SLEEP` | `60` | Seconds to wait on API errors |
| `MAX_ITERATIONS` | `5` | Max revision loops in MultiNode |

See `.env.example` for all 12 model configuration variables.
