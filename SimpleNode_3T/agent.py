"""SimpleNode_3T: Single node, 3 tools + self-critique prompt."""

import json

from google.genai import types as genai_types
from langgraph.graph import StateGraph, END

from shared.state import AgentState, initial_state, get_client, generate, generate_with_tools, snip
from shared.config import SIMPLE_3T_MODEL, CONTRADICTION_MODEL, TRANSLATION_MODEL, MAX_TOOL_ITERATIONS
from shared.tools import tool_retrieve, tool_get_document_metadata, tool_check_contradictions, tool_translate_to_arabic, tool_detect_language

TOOLS = [
    {
        "name": "retrieve",
        "description": (
    "Search policy documents. Returns top-6 docs with snippets. "
    "Params: query (str), current_only (bool, default=true to exclude superseded), "
    "filters (str optional JSON with category/department/effective_date). "
    "Valid categories: Client, Communications, Finance, HR, IT, Legal, Operations, Travel. "
    "Set current_only=false only if user asks for old/previous policies."
),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "current_only": {"type": "boolean"},
                "filters": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_document_metadata",
        "description": "Get metadata and version chain for a doc. Param: doc_id (str).",
        "parameters": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
    },
    {
        "name": "check_contradictions",
        "description": "Check if 2+ documents contradict on a topic. Only call when multiple docs overlap. "
                       "Params: doc_ids (comma-sep), question (str).",
        "parameters": {
            "type": "object",
            "properties": {"doc_ids": {"type": "string"}, "question": {"type": "string"}},
            "required": ["doc_ids", "question"],
        },
    },
]


def _execute_tool(name: str, args: dict, client) -> str:
    if name == "retrieve":
        return tool_retrieve(args.get("query", ""), args.get("current_only", True), args.get("filters"))
    elif name == "get_document_metadata":
        return tool_get_document_metadata(args.get("doc_id", ""))
    elif name == "check_contradictions":
        return tool_check_contradictions(args.get("doc_ids", ""), args.get("question", ""), client, CONTRADICTION_MODEL)
    return json.dumps({"error": f"Unknown tool: {name}"})


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
            "Steps:\n"
            "1. Call retrieve(query, current_only=true) to search current policies by default. "
            "Only use current_only=false if the user explicitly asks about old/previous policies. "
            "Use filters for category or department narrowing.\n"
            "2. Call get_document_metadata(doc_id) for each retrieved doc to check versions.\n"
            "3. If 2+ documents overlap on the same topic with different values, "
            "call check_contradictions(doc_ids, question).\n"
            "4. After retrieving, do a SECOND broader search to catch any documents "
            "that might disagree (e.g., search 'benefits summary' after 'paternity leave').\n"
            "5. Compose a CONCISE answer with [doc_id] citations. "
            "Every claim must cite a source document.\n"
            "6. Self-critique: Is every claim backed by a chunk? If not, fix it.\n\n"
            "CRITICAL RULES:\n"
            "- If documents contradict, surface BOTH sides — never pick one silently.\n"
            "- If no relevant documents found, refuse politely without guessing.\n"
            "- Answer in {lang} language. If lang='ar', you MUST output the answer in Arabic. "
            "After composing in English, translate the final answer to Arabic. "
            "Do NOT return an English answer when the question is in Arabic.\n"
            "- Be honest about confidence. Set < 1.0 if uncertain.\n"
            "- NO reasoning or commentary. Return ONLY the JSON.\n"
            "- Your response must START with {{ and END with }}.\n\n"
            "Return ONLY this JSON (nothing else):\n"
            '{{"answer":"...", "citations":[{{"doc_id":"...", "snippet":"..."}}], '
            '"confidence":{{"overall":0.0-1.0, "grounding":0.0-1.0, '
            '"completeness":0.0-1.0}}}}'
        ).format(lang=lang),
    }]

    for _ in range(MAX_TOOL_ITERATIONS):
        response, usage, latency = generate_with_tools(client, SIMPLE_3T_MODEL, messages, TOOLS)
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
                name = fn.name
                args = dict(fn.args) if fn.args else {}
                result = _execute_tool(name, args, client)
                state["trace"].append({
                    "step": len(state["trace"]) + 1, "type": "tool_call",
                    "tool": name, "args": snip(str(args)),
                    "result_summary": snip(result),
                })
                fn_responses.append(
                    genai_types.Part.from_function_response(name=name, response={"result": result})
                )
            messages.append({"role": "user", "content": fn_responses})
        elif final_text:
            break

    # Strip thinking text
    ft = final_text.strip()
    if ft and not ft.startswith("{"):
        ft_lower = ft[:100].lower()
        if any(marker in ft_lower for marker in ("wait,", "let me", "i need to", "i should", "i will", "first,", "now,")):
            idx = max(ft.find("{"), ft.find("{"))
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
    state = build_graph().invoke(state)
    if language == "ar" and state.get("final_answer"):
        try:
            client = get_client()
            ar = tool_translate_to_arabic(state["final_answer"], client, TRANSLATION_MODEL)
            state["final_answer"] = ar.strip()
        except Exception:
            pass
    return state
