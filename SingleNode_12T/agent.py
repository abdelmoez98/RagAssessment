"""SingleNode_12T: Single node, all 12 tools, guided prompt."""

import json

from google.genai import types as genai_types
from langgraph.graph import StateGraph, END

from shared.state import AgentState, initial_state, get_client, generate, generate_with_tools, snip
from shared.config import (
    SINGLE_12T_MODEL, SINGLE_12T_CLASSIFY_MODEL, SINGLE_12T_COMPOSE_MODEL,
    SINGLE_12T_CRITIC_MODEL, CONTRADICTION_MODEL, TRANSLATION_MODEL, MAX_TOOL_ITERATIONS,
)
from shared.tools import (
    tool_retrieve, tool_get_document_metadata,
    tool_check_contradictions, tool_classify_question,
    tool_compose_single_doc, tool_compose_multi_doc, tool_compose_contradiction, tool_compose_refusal,
    tool_evaluate_answer, tool_evaluate_compliance,
    tool_translate_to_english, tool_translate_to_arabic, tool_detect_language,
)

TOOLS = [
    {"name": "retrieve", "description": "Search docs. current_only=true excludes superseded. Params: query (str), current_only (bool), filters (str optional JSON). Valid categories: Client, Communications, Finance, HR, IT, Legal, Operations, Travel.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "current_only": {"type": "boolean"}, "filters": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_document_metadata", "description": "Doc metadata + version chain. Param: doc_id (str).", "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]}},
    {"name": "translate_to_english", "description": "Translate Arabic text to English for retrieval. Param: text (str). Use when question is in Arabic.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "translate_to_arabic", "description": "Translate English answer to Arabic. Param: text (str). Use when original question was in Arabic.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "check_contradictions", "description": "Detect contradictions. Only call when 2+ docs overlap on same topic. Params: doc_ids (comma-sep), question (str).", "parameters": {"type": "object", "properties": {"doc_ids": {"type": "string"}, "question": {"type": "string"}}, "required": ["doc_ids", "question"]}},
    {"name": "classify_question", "description": "Classify question type: single_doc|composition|contradiction|out_of_scope. Params: docs_json (str), question (str), contradiction_report_json (str).", "parameters": {"type": "object", "properties": {"docs_json": {"type": "string"}, "question": {"type": "string"}, "contradiction_report_json": {"type": "string"}}, "required": ["docs_json", "question"]}},
    {"name": "compose_single_doc", "description": "Answer from one doc. Params: chunks_json (str), question (str), language (str).", "parameters": {"type": "object", "properties": {"chunks_json": {"type": "string"}, "question": {"type": "string"}, "language": {"type": "string"}}, "required": ["chunks_json", "question"]}},
    {"name": "compose_multi_doc", "description": "Synthesize from multiple docs. Params: chunks_json (str), question (str), language (str).", "parameters": {"type": "object", "properties": {"chunks_json": {"type": "string"}, "question": {"type": "string"}, "language": {"type": "string"}}, "required": ["chunks_json", "question"]}},
    {"name": "compose_contradiction", "description": "Surface contradiction. Params: contradiction_report_json (str), question (str), language (str).", "parameters": {"type": "object", "properties": {"contradiction_report_json": {"type": "string"}, "question": {"type": "string"}, "language": {"type": "string"}}, "required": ["contradiction_report_json", "question"]}},
    {"name": "compose_refusal", "description": "Refusal message. Params: question (str), language (str).", "parameters": {"type": "object", "properties": {"question": {"type": "string"}, "language": {"type": "string"}}, "required": ["question"]}},
    {"name": "evaluate_answer", "description": "Evaluate grounding/citations/completeness/language/thinking-text. Params: answer_json (str), question (str), chunks_json (str), language (str).", "parameters": {"type": "object", "properties": {"answer_json": {"type": "string"}, "question": {"type": "string"}, "chunks_json": {"type": "string"}, "language": {"type": "string"}}, "required": ["answer_json", "question", "chunks_json"]}},
    {"name": "evaluate_compliance", "description": "Evaluate currency/contradiction/refusal. Params: answer_json (str), chunks_json (str), classification (str), language (str).", "parameters": {"type": "object", "properties": {"answer_json": {"type": "string"}, "chunks_json": {"type": "string"}, "classification": {"type": "string"}, "language": {"type": "string"}}, "required": ["answer_json", "chunks_json", "classification"]}},
]


def _execute_tool(name: str, args: dict, client) -> str:
    lang = args.get("language", "en")
    m = {
        "retrieve": lambda a: tool_retrieve(a.get("query", ""), a.get("current_only", True), a.get("filters")),
        "get_document_metadata": lambda a: tool_get_document_metadata(a.get("doc_id", "")),
        "translate_to_english": lambda a: tool_translate_to_english(a.get("text", ""), client, TRANSLATION_MODEL),
        "translate_to_arabic": lambda a: tool_translate_to_arabic(a.get("text", ""), client, TRANSLATION_MODEL),
        "check_contradictions": lambda a: tool_check_contradictions(a.get("doc_ids", ""), a.get("question", ""), client, CONTRADICTION_MODEL),
        "classify_question": lambda a: tool_classify_question(a.get("docs_json", ""), a.get("question", ""), a.get("contradiction_report_json", "none"), client, SINGLE_12T_CLASSIFY_MODEL),
        "compose_single_doc": lambda a: tool_compose_single_doc(a.get("chunks_json", ""), a.get("question", ""), client, SINGLE_12T_COMPOSE_MODEL, lang),
        "compose_multi_doc": lambda a: tool_compose_multi_doc(a.get("chunks_json", ""), a.get("question", ""), client, SINGLE_12T_COMPOSE_MODEL, lang),
        "compose_contradiction": lambda a: tool_compose_contradiction(a.get("contradiction_report_json", ""), a.get("question", ""), client, SINGLE_12T_COMPOSE_MODEL, lang),
        "compose_refusal": lambda a: tool_compose_refusal(a.get("question", ""), client, SINGLE_12T_COMPOSE_MODEL, lang),
        "evaluate_answer": lambda a: tool_evaluate_answer(a.get("answer_json", ""), a.get("question", ""), a.get("chunks_json", ""), client, SINGLE_12T_CRITIC_MODEL, lang),
        "evaluate_compliance": lambda a: tool_evaluate_compliance(a.get("answer_json", ""), a.get("chunks_json", ""), a.get("classification", ""), client, SINGLE_12T_CRITIC_MODEL, lang),
    }
    fn = m.get(name)
    return fn(args) if fn else json.dumps({"error": f"Unknown tool: {name}"})


def agent_node(state: AgentState) -> AgentState:
    client = get_client()
    question = state["question_en"]
    lang = state["language"]
    state["trace"].append({"step": 1, "node": "single_agent", "action": "started"})

    messages = [{
        "role": "user",
        "content": (
            f"Answer this policy question.\n\n"
            f"Question: {question}\n"
            f"Answer MUST be in {lang} language.\n\n"
            "Process:\n"
            "0. If the question is in Arabic: call translate_to_english first, then use the English "
            "translation for retrieval. After composing the final answer, call translate_to_arabic to "
            "translate it back before returning.\n"
            "1. retrieve(query, current_only=true) → current policies by default. "
            "Only use current_only=false if user asks for old/previous policies.\n"
            "2. get_document_metadata → verify versions.\n"
            "3. If 2+ docs overlap on same topic, check_contradictions → classify_question → type.\n"
            "4. After retrieving, do a SECOND broader search to catch docs that might disagree.\n"
            "5. Compose: single_doc→compose_single_doc, composition→compose_multi_doc, "
            "contradiction→compose_contradiction, out_of_scope→compose_refusal.\n"
            "6. evaluate_answer + evaluate_compliance → fix issues → re-evaluate.\n\n"
            "CRITICAL RULES:\n"
            "- Surface contradictions, refuse if no answer, cite every claim.\n"
            "- Answer ONLY in {lang} language.\n"
            "- Be honest about confidence. Set < 1.0 if uncertain.\n"
            "- NO reasoning or commentary. Return ONLY the JSON.\n"
            "- Your response must START with {{ and END with }}.\n\n"
            "Return FINAL JSON: {{\"answer\":\"...\", \"citations\":[...], \"confidence\":{{...}}}}"
        ).format(lang=lang),
    }]

    for _ in range(MAX_TOOL_ITERATIONS):
        response, usage, latency = generate_with_tools(client, SINGLE_12T_MODEL, messages, TOOLS)
        state["trace"].append({"step": len(state["trace"]) + 1, "type": "llm_call", "usage": usage})

        if not response.candidates:
            break
        parts = response.candidates[0].content.parts
        final_text = ""
        fn_calls = []

        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                fn_calls.append(part.function_call)
            elif hasattr(part, "text") and part.text:
                final_text += part.text

        if fn_calls:
            messages.append({"role": "model", "content": parts})
            fn_responses = []
            for fn in fn_calls:
                result = _execute_tool(fn.name, dict(fn.args) if fn.args else {}, client)
                state["trace"].append({"step": len(state["trace"]) + 1, "type": "tool_call", "tool": fn.name, "args": snip(str(fn.args)), "result_summary": snip(result)})
                fn_responses.append(genai_types.Part.from_function_response(name=fn.name, response={"result": result}))
            messages.append({"role": "user", "content": fn_responses})
        elif final_text:
            break

    ft = final_text.strip()
    if ft and not ft.startswith("{"):
        ft_lower = ft[:100].lower()
        if any(marker in ft_lower for marker in ("wait,", "let me", "i need to", "i should", "i will", "first,", "now,")):
            idx = ft.find("{")
            if idx > 0:
                ft = ft[idx:]

    try:
        cleaned = ft.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        state["final_answer"] = parsed.get("answer", ft)
        state["citations"] = parsed.get("citations", [])
        conf = parsed.get("confidence", {})
        if isinstance(conf, (int, float)):
            conf = {"overall": conf}
        state["confidence"] = conf
    except (json.JSONDecodeError, AttributeError):
        state["final_answer"] = ft
        state["confidence"] = {"overall": 0.5}

    state["trace"].append({"step": len(state["trace"]) + 1, "node": "single_agent", "action": "finalized"})
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.set_entry_point("agent")
    g.add_edge("agent", END)
    return g.compile()


def run_agent(question: str, language: str = "en") -> AgentState:
    client = get_client()
    try:
        detected = tool_detect_language(question, client, TRANSLATION_MODEL)
        if detected in ("ar", "arabic"):
            language = "ar"
    except Exception:
        pass

    state = initial_state(question, language)
    if language == "ar":
        en, _, _ = generate(client, TRANSLATION_MODEL, f"Translate to English: {question}")
        state["question_en"] = en.strip()
    return build_graph().invoke(state)
