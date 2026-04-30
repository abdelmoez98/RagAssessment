"""MultiNode_12T: Supervisor + Retriever + Classifier + Composer + Critic.
Supervisor maintains pending queue, Critic reports → Supervisor rebuilds queue."""

import json

from langgraph.graph import StateGraph, END

from shared.state import AgentState, initial_state, get_client, generate, snip
from shared.config import (
    MULTI_CLASSIFY_MODEL, MULTI_COMPOSE_MODEL, MULTI_CRITIC_MODEL,
    CONTRADICTION_MODEL, TRANSLATION_MODEL, MAX_ITERATIONS,
)
from shared.tools import (
    tool_retrieve, tool_get_document_metadata, tool_filter_superseded,
    tool_check_contradictions, tool_classify_question,
    tool_compose_single_doc, tool_compose_multi_doc, tool_compose_contradiction, tool_compose_refusal,
    tool_evaluate_answer, tool_evaluate_compliance,
    tool_translate_to_english, tool_translate_to_arabic, tool_detect_language,
)


def supervisor(state: AgentState) -> AgentState:
    state["trace"].append({"step": len(state["trace"]) + 1, "node": "supervisor", "action": "routing"})

    if state["pending_steps"] is None:
        state["pending_steps"] = ["retriever", "classifier", "composer", "critic"]
        state["iteration"] = 0

    verdict = state.get("critic_verdict", "")
    if verdict:
        if state["iteration"] >= MAX_ITERATIONS:
            state["critic_verdict"] = "approved"
        if verdict == "approved":
            return state
        state["iteration"] += 1
        if verdict == "back_to_retriever":
            state["pending_steps"] = ["retriever", "classifier", "composer", "critic"]
        elif verdict == "back_to_classifier":
            state["pending_steps"] = ["classifier", "composer", "critic"]
        elif verdict == "back_to_composer":
            state["pending_steps"] = ["composer", "critic"]
        state["critic_verdict"] = ""
        state["critique"] = None

    return state


def retriever_node(state: AgentState) -> AgentState:
    client = get_client()
    lang = state.get("language", "en")
    
    # 1. Determine the search query
    if lang == "ar":
        # Ensure we don't re-translate if we already have it from a previous iteration
        if not state.get("question_en"):
            translated_query = tool_translate_to_english(state["question"], client, TRANSLATION_MODEL)
            state["question_en"] = translated_query.strip()
            state["trace"].append({
                "step": len(state["trace"]) + 1, 
                "node": "retriever", 
                "action": "translated_to_en"
            })
        search_query = state["question_en"]
    else:
        search_query = state["question"]

    # 2. Execute retrieval using the determined search_query
    state["trace"].append({
        "step": len(state["trace"]) + 1, 
        "node": "retriever", 
        "action": f"searching with query: {search_query[:50]}..."
    })
    
    # CRITICAL: Ensure tool_retrieve uses the 'search_query' variable
    result = tool_retrieve(search_query, current_only=False)
    state["search_results"] = json.loads(result)

    # 3. Metadata and Filtering
    for doc in state["search_results"]:
        meta = tool_get_document_metadata(doc["doc_id"])
        state["metadata_checked"].append(json.loads(meta))

    if state["search_results"]:
        dids = ",".join([d["doc_id"] for d in state["search_results"]])
        filt = tool_filter_superseded(dids)
        state["trace"].append({
            "step": len(state["trace"]) + 1, 
            "node": "retriever", 
            "filter_status": "applied"
        })

    state["trace"].append({
        "step": len(state["trace"]) + 1, 
        "node": "retriever", 
        "action": f"found {len(state['search_results'])} docs"
    })
    
    return state


def classifier_node(state: AgentState) -> AgentState:
    client = get_client()
    state["trace"].append({"step": len(state["trace"]) + 1, "node": "classifier", "action": "classifying"})

    docs = state["search_results"]
    dids = ",".join([d["doc_id"] for d in docs])
    question = state["question_en"]

    try:
        contra = tool_check_contradictions(dids, question, client, CONTRADICTION_MODEL)
        state["contradictions"] = json.loads(contra)
    except Exception:
        state["contradictions"] = {"contradiction_found": False}
        contra = json.dumps(state["contradictions"])

    state["trace"].append({"step": len(state["trace"]) + 1, "node": "classifier", "contra": snip(contra)})

    try:
        cls = tool_classify_question(json.dumps(docs), question, contra, client, MULTI_CLASSIFY_MODEL)
        state["question_type"] = json.loads(cls).get("question_type", "single_doc")
    except Exception:
        state["question_type"] = "single_doc"

    state["trace"].append({"step": len(state["trace"]) + 1, "node": "classifier", "type": state["question_type"]})
    return state


def composer_node(state: AgentState) -> AgentState:
    client = get_client()
    qtype = state["question_type"]
    question = state["question_en"]
    lang = state["language"]
    docs = json.dumps(state["search_results"])
    state["trace"].append({"step": len(state["trace"]) + 1, "node": "composer", "type": qtype})

    try:
        if qtype == "contradiction" and state.get("contradictions"):
            result = tool_compose_contradiction(json.dumps(state["contradictions"]), question, client, MULTI_COMPOSE_MODEL, lang)
        elif qtype == "composition":
            result = tool_compose_multi_doc(docs, question, client, MULTI_COMPOSE_MODEL, lang)
        elif qtype == "out_of_scope":
            result = tool_compose_refusal(question, client, MULTI_COMPOSE_MODEL, lang)
        else:
            result = tool_compose_single_doc(docs, question, client, MULTI_COMPOSE_MODEL, lang)

        parsed = json.loads(result)
        state["draft_answer"] = parsed.get("answer", result)
        state["citations"] = parsed.get("citations", [])
    except Exception:
        state["draft_answer"] = "Error composing answer. Please try again."
        state["citations"] = []

    state["trace"].append({"step": len(state["trace"]) + 1, "node": "composer", "done": True})

    # Translate answer back to Arabic if input was Arabic
    if lang == "ar" and state["draft_answer"]:
        try:
            ar_answer = tool_translate_to_arabic(state["draft_answer"], client, TRANSLATION_MODEL)
            state["draft_answer"] = ar_answer.strip()
            state["trace"].append({"step": len(state["trace"]) + 1, "node": "composer", "action": "translated_to_arabic"})
        except Exception:
            pass

    return state


def critic_node(state: AgentState) -> AgentState:
    client = get_client()
    state["trace"].append({"step": len(state["trace"]) + 1, "node": "critic", "action": "evaluating"})

    question = state["question_en"]
    lang = state["language"]
    ans_json = json.dumps({"answer": state["draft_answer"], "citations": state["citations"]})
    docs_json = json.dumps(state["search_results"])
    qtype = state["question_type"]

    try:
        ev = tool_evaluate_answer(ans_json, question, docs_json, client, MULTI_CRITIC_MODEL, lang)
        evd = json.loads(ev)
    except Exception:
        state["critic_verdict"] = "approved"
        return state

    try:
        co = tool_evaluate_compliance(ans_json, docs_json, qtype, client, MULTI_CRITIC_MODEL, lang)
        cod = json.loads(co)
    except Exception:
        cod = {"overall_pass": True, "source_currency": {"score": 1.0}}

    verdict = "approved"
    currency = cod.get("source_currency", {})
    contra = cod.get("contradiction_handling", {})
    ground = evd.get("grounding", {})
    complete = evd.get("completeness", {})
    lang_check = evd.get("language_match", {})
    thinking = evd.get("thinking_text", {})

    if thinking.get("detected"):
        verdict = "back_to_composer"
    elif lang_check.get("pass") is False:
        verdict = "back_to_composer"
    elif currency.get("score", 1.0) < 0.5 and currency.get("issues"):
        verdict = cod.get("primary_target", "back_to_retriever")
    elif contra.get("applicable") and contra.get("score", 1.0) < 0.5:
        verdict = "back_to_classifier"
    elif [s for s in ground.get("sentences", []) if not s.get("grounded")]:
        verdict = "back_to_composer"
    elif complete.get("missing_parts"):
        verdict = "back_to_retriever"

    state["critic_verdict"] = verdict
    state["critique"] = {"eval": evd, "compliance": cod}
    state["trace"].append({"step": len(state["trace"]) + 1, "node": "critic", "verdict": verdict})

    gs = ground.get("score", 0.8)
    cs = evd.get("citation_accuracy", {}).get("score", 0.8)
    ss = currency.get("score", 0.8)
    ms = complete.get("score", 0.8)
    state["confidence"] = {
        "overall": round((gs + cs + ss + ms) / 4, 2),
        "grounding": gs, "citation_accuracy": cs,
        "source_currency": ss, "completeness": ms,
    }
    state["final_answer"] = state["draft_answer"]

    return state


def route_supervisor(state: AgentState) -> str:
    if state["pending_steps"] is None:
        supervisor(state)
    if not state["pending_steps"]:
        return "finalize"
    return state["pending_steps"].pop(0)


def route_critic(state: AgentState) -> str:
    return "finalize" if state["critic_verdict"] == "approved" else "supervisor"


def finalize_node(state: AgentState) -> AgentState:
    state["trace"].append({"step": len(state["trace"]) + 1, "node": "finalize", "action": "done"})
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor)
    g.add_node("retriever", retriever_node)
    g.add_node("classifier", classifier_node)
    g.add_node("composer", composer_node)
    g.add_node("critic", critic_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route_supervisor, {
        "retriever": "retriever", "classifier": "classifier",
        "composer": "composer", "critic": "critic", "finalize": END,
    })
    g.add_edge("retriever", "supervisor")
    g.add_edge("classifier", "supervisor")
    g.add_edge("composer", "supervisor")
    g.add_conditional_edges("critic", route_critic, {"finalize": END, "supervisor": "supervisor"})
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
