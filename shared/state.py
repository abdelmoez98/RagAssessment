import os
import time
from typing import Annotated, TypedDict, Optional

import google.genai as genai
from google.genai import types as genai_types

from shared.config import RATE_LIMIT_SLEEP, MAX_RETRIES, TRACE_SNIPPET_LENGTH


class AgentState(TypedDict):
    question: str
    language: str
    question_en: str
    search_results: Annotated[list[dict], "merge_append"]
    metadata_checked: Annotated[list[dict], "merge_append"]
    contradictions: dict | None
    question_type: str
    draft_answer: str
    citations: list[dict]
    critique: dict | None
    critic_verdict: str
    final_answer: str
    confidence: dict
    pending_steps: list[str]
    iteration: int
    trace: list[dict]
    errors: list[str]


def initial_state(question: str, language: str = "en") -> AgentState:
    return AgentState(
        question=question,
        language=language,
        question_en=question,
        search_results=[],
        metadata_checked=[],
        contradictions=None,
        question_type="",
        draft_answer="",
        citations=[],
        critique=None,
        critic_verdict="",
        final_answer="",
        confidence={},
        pending_steps=None,
        iteration=0,
        trace=[],
        errors=[],
    )


def retry(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RATE_LIMIT_SLEEP)
            else:
                raise


def get_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Copy .env.example to .env and set your key.")
    return genai.Client(api_key=api_key)


def generate(client, model: str, prompt: str, config: Optional[dict] = None) -> tuple[str, dict, float]:
    start = time.time()
    
    # Pack the core arguments
    kwargs = {
        "model": model, 
        "contents": prompt
    }
    
    # Inject the config only if provided, keeping it clean for standard text calls
    if config:
        kwargs["config"] = config

    # Pass the unpacked arguments to your retry wrapper
    response = retry(client.models.generate_content, **kwargs)
    
    latency_ms = round((time.time() - start) * 1000, 1)
    text = response.text or ""
    usage = {}
    
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        u = response.usage_metadata
        usage = {
            "prompt_token_count": getattr(u, "prompt_token_count", 0),
            "candidates_token_count": getattr(u, "candidates_token_count", 0),
            "total_token_count": getattr(u, "total_token_count", 0),
            "model": model,
            "latency_ms": latency_ms,
        }
        
    return text, usage, latency_ms


def generate_with_tools(client, model: str, messages: list, tools: list[dict]):
    """Generate with tools. messages is list of {role, content} dicts.
    Returns (response, usage, latency)."""
    declarations = []
    for t in tools:
        props = {}
        reqs = t.get("parameters", {}).get("required", [])
        for pname, pinfo in t.get("parameters", {}).get("properties", {}).items():
            props[pname] = genai_types.Schema(type=genai_types.Type.STRING)
        declarations.append(genai_types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties=props,
                required=reqs,
            ),
        ))

    cfg = genai_types.GenerateContentConfig(
        tools=[genai_types.Tool(function_declarations=declarations)],
    )

    # Convert messages to Content list
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        role = "model" if role == "assistant" else role
        content = msg.get("content", "")
        parts = []
        if isinstance(content, list):
            # Already a list of Part objects or similar
            parts = content
        elif isinstance(content, str):
            parts = [genai_types.Part(text=content)]
        else:
            parts = [genai_types.Part(text=str(content))]
        contents.append(genai_types.Content(role=role, parts=parts))

    start = time.time()
    response = retry(client.models.generate_content, model=model, contents=contents, config=cfg)
    latency_ms = round((time.time() - start) * 1000, 1)

    usage = {}
    if hasattr(response, "usage_metadata"):
        u = response.usage_metadata
        usage = {
            "prompt_token_count": u.prompt_token_count,
            "candidates_token_count": u.candidates_token_count,
            "total_token_count": u.total_token_count,
            "model": model,
            "latency_ms": latency_ms,
        }
    return response, usage, latency_ms


def snip(text, limit: int = None) -> str:
    if limit is None:
        limit = TRACE_SNIPPET_LENGTH
    if limit == -1:
        return str(text)
    text = str(text)
    return text[:limit] + ("..." if len(text) > limit else "")
