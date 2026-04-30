import os

from dotenv import load_dotenv

load_dotenv()

# Agent LLMs — Gemma 4 26b (15 RPM)
SIMPLE_3T_MODEL = os.getenv("SIMPLE_3T_MODEL", "gemma-4-26b-a4b-it")
SINGLE_12T_MODEL = os.getenv("SINGLE_12T_MODEL", "gemma-4-26b-a4b-it")
SINGLE_12T_CLASSIFY_MODEL = os.getenv("SINGLE_12T_CLASSIFY_MODEL", "gemma-4-26b-a4b-it")
SINGLE_12T_COMPOSE_MODEL = os.getenv("SINGLE_12T_COMPOSE_MODEL", "gemma-4-26b-a4b-it")
MULTI_CLASSIFY_MODEL = os.getenv("MULTI_CLASSIFY_MODEL", "gemma-4-26b-a4b-it")
MULTI_COMPOSE_MODEL = os.getenv("MULTI_COMPOSE_MODEL", "gemma-4-26b-a4b-it")

# Manager / Critic / Contradiction — Gemma 4 31b (15 RPM)
SINGLE_12T_CRITIC_MODEL = os.getenv("SINGLE_12T_CRITIC_MODEL", "gemma-4-31b-it")
MULTI_CRITIC_MODEL = os.getenv("MULTI_CRITIC_MODEL", "gemma-4-31b-it")
CONTRADICTION_MODEL = os.getenv("CONTRADICTION_MODEL", "gemma-4-31b-it")

# Translation — Gemma 4 26b (15 RPM)
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "gemma-4-26b-a4b-it")

# Evaluation — Gemini Flash-Lite (10 RPM)
EVAL_MODEL = os.getenv("EVAL_MODEL", "gemini-3.1-flash-lite-preview")
RAGAS_MODEL = os.getenv("RAGAS_MODEL", "gemini-3.1-flash-lite-preview")

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Paths
POLICY_CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "policy_corpus")
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "store", "chroma"))
BM25_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "store", "bm25")

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
RETRIEVAL_TOP_K = 10
RRF_K = 60

# Rate limit
RATE_LIMIT_SLEEP = 60
MAX_RETRIES = 3
MAX_ITERATIONS = 5

# Tool & trace tuning
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "15"))
TRACE_SNIPPET_LENGTH = int(os.getenv("TRACE_SNIPPET_LENGTH", "-1"))

# Default agent
DEFAULT_AGENT = os.getenv("DEFAULT_AGENT", "MultiNode_12T")
