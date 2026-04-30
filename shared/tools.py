import json
import os
from pydantic import BaseModel

from ingestion.indexer import Indexer
from shared.config import POLICY_CORPUS_DIR, RRF_K
from shared.state import snip as _snip

VALID_CATEGORIES = ["Client", "Communications", "Finance", "HR", "IT", "Legal", "Operations", "Travel"]
VALID_DEPARTMENTS = ["Compliance", "Finance", "HR", "IT", "Legal", "Marketing", "Operations"]


def _load_metadata():
    with open(os.path.join(POLICY_CORPUS_DIR, "metadata.json")) as f:
        return json.load(f)["documents"]


_metadata_index = _load_metadata()
_indexer = Indexer()


def _parse_date(date_str: str) -> str | None:
    """Parse a natural date reference to ISO format. '2024' → '2024-01-01'."""
    import re
    date_str = date_str.strip()
    # Already ISO
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str
    # Year only
    m = re.match(r"^(\d{4})$", date_str)
    if m:
        return f"{m.group(1)}-01-01"
    # Year-Month (2024-03 or March 2024)
    m = re.match(r"^(\d{4})-(\d{2})$", date_str)
    if m:
        return f"{m.group(1)}-{m.group(2)}-01"
    return None


def tool_retrieve(query: str, current_only: bool = True, filters: str = None) -> str:
    bm25, bm25_chunks = _indexer.load_bm25()
    top_k = 6

    where_filter = {}
    date_filter = None

    if filters:
        try:
            raw = json.loads(filters)
        except json.JSONDecodeError:
            raw = {}
        if raw.get("category") and raw["category"] in VALID_CATEGORIES:
            where_filter["category"] = raw["category"]
        if raw.get("department") and raw["department"] in VALID_DEPARTMENTS:
            where_filter["department"] = raw["department"]
        if raw.get("effective_date"):
            date_filter = _parse_date(str(raw["effective_date"]))

    if current_only:
        where_filter["is_current"] = True

    collection = _indexer.get_chroma_collection()
    dense_results = collection.query(
        query_texts=[query], n_results=top_k * 3,
        include=["documents", "metadatas", "distances"],
    )

    tokenized = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized)

    bm25_top = []
    for i in sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True):
        chunk = bm25_chunks[i]
        if where_filter:
            skip = False
            if where_filter.get("is_current") and not chunk.get("is_current", True):
                skip = True
            if where_filter.get("category") and chunk.get("category", "") != where_filter["category"]:
                skip = True
            if where_filter.get("department") and chunk.get("department", "") != where_filter["department"]:
                skip = True
            if date_filter and chunk.get("effective_date", "") < date_filter:
                skip = True
            if skip:
                continue
        bm25_top.append(i)
        if len(bm25_top) >= top_k * 3:
            break

    rrf = {}
    dense_ids = dense_results.get("ids", [[]])[0] or []
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, i in enumerate(bm25_top):
        cid = bm25_chunks[i].get("chunk_id", "")
        rrf[cid] = rrf.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    sorted_ids = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]

    dense_lookup = {}
    if dense_ids:
        for j, cid in enumerate(dense_ids):
            dl = dense_results.get("documents", [[]])[0]
            ml = dense_results.get("metadatas", [[]])[0]
            if j < len(dl) and j < len(ml):
                meta = ml[j]
                if where_filter:
                    skip = False
                    if where_filter.get("is_current") and not meta.get("is_current", True):
                        skip = True
                    if where_filter.get("category") and meta.get("category", "") != where_filter["category"]:
                        skip = True
                    if where_filter.get("department") and meta.get("department", "") != where_filter["department"]:
                        skip = True
                    if date_filter and meta.get("effective_date", "") < date_filter:
                        skip = True
                    if skip:
                        continue
                dense_lookup[cid] = (dl[j], meta)

    results = []
    seen = set()
    for cid, score in sorted_ids:
        if cid in dense_lookup:
            text, meta = dense_lookup[cid]
        else:
            for chunk in bm25_chunks:
                if chunk.get("chunk_id") == cid:
                    text = chunk.get("text", "")
                    meta = {
                        "doc_id": chunk.get("doc_id", ""), "title": chunk.get("title", ""),
                        "category": chunk.get("category", ""), "department": chunk.get("department", ""),
                        "effective_date": chunk.get("effective_date", ""),
                        "is_current": chunk.get("is_current", True),
                        "superseded_by": chunk.get("superseded_by") or "",
                    }
                    break
            else:
                continue
        doc_id = meta.get("doc_id", "")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        results.append({
            "chunk_id": cid, "doc_id": doc_id, "title": meta.get("title", ""),
            "text_snippet": _snip(text.strip() if text else "", 300),
            "category": meta.get("category", ""), "department": meta.get("department", ""),
            "effective_date": meta.get("effective_date", ""),
            "is_current": meta.get("is_current", True),
            "superseded_by": meta.get("superseded_by") or None,
            "rrf_score": round(score, 4),
        })
    return json.dumps(results, ensure_ascii=False, indent=2)


def tool_get_document_metadata(doc_id: str) -> str:
    doc_id = doc_id.strip()
    if doc_id not in _metadata_index:
        return json.dumps({"error": f"Document {doc_id} not found"})
    meta = _metadata_index[doc_id]
    is_current = meta.get("superseded_by") is None
    chain = [{"doc_id": doc_id, "effective_date": meta["effective_date"], "current": is_current}]
    cursor = meta
    while cursor.get("superseded_by") and cursor["superseded_by"] in _metadata_index:
        nid = cursor["superseded_by"]
        nm = _metadata_index[nid]
        chain.append({"doc_id": nid, "effective_date": nm["effective_date"], "current": nm.get("superseded_by") is None})
        cursor = nm
    cv = None
    if not is_current:
        cv = {"doc_id": chain[-1]["doc_id"], "effective_date": chain[-1]["effective_date"]}
    return json.dumps({
        "doc_id": doc_id, "title": meta.get("title", ""), "category": meta.get("category", ""),
        "department": meta.get("department", ""), "effective_date": meta.get("effective_date", ""),
        "superseded_by": meta.get("superseded_by"), "supersedes": meta.get("supersedes"),
        "is_current": is_current, "current_version": cv, "version_chain": chain,
    }, ensure_ascii=False, indent=2)


def tool_filter_superseded(doc_ids: str) -> str:
    results = []
    for doc_id in (d.strip() for d in doc_ids.split(",")):
        if doc_id not in _metadata_index:
            results.append({"doc_id": doc_id, "error": "Not found"})
            continue
        meta = _metadata_index[doc_id]
        is_current = meta.get("superseded_by") is None
        if is_current:
            results.append({"doc_id": doc_id, "is_current": True, "status": "current", "action": "keep"})
        else:
            sid = meta["superseded_by"]
            available = sid in _metadata_index
            results.append({
                "doc_id": doc_id, "is_current": False, "status": "superseded",
                "superseded_by": sid, "successor_available": available,
                "action": "prefer_successor" if available else "refuse",
                "current_version": {"doc_id": sid} if available else None,
                "message": (f"Use {sid}" if available else f"Replacement {sid} not available. Consult department."),
            })
    return json.dumps(results, ensure_ascii=False, indent=2)


def tool_check_contradictions(doc_ids: str, question: str, client, model: str) -> str:
    ids = [d.strip() for d in doc_ids.split(",")]
    if len(ids) < 2:
        return json.dumps({"contradiction_found": False, "reason": "Need at least 2 documents. Only call when multiple docs overlap on the same topic."})
    docs_text = ""
    for did in ids:
        try:
            chunks = _indexer.get_chunks_for_doc(did)
            if chunks:
                docs_text += f"\n### {did}\n" + "\n".join([c["text"] for c in chunks]) + "\n"
        except Exception:
            docs_text += f"\n### {did}\n[Not available]\n"
    if not docs_text.strip():
        return json.dumps({"contradiction_found": False, "reason": "No text retrieved"})
    from shared.state import generate
    result, _, _ = generate(client, model, (
        f"Analyze these policy documents for contradictions regarding: \"{question}\"\n\n"
        f"Documents:{docs_text}\n\n"
        "Return ONLY JSON: {\"contradiction_found\": bool, \"contradictions\": [{\"topic\": \"...\", "
        "\"positions\": [{\"doc_id\": \"...\", \"claim\": \"...\"}], \"explanation\": \"...\"}]}"
    ))
    return result.replace("```json", "").replace("```", "").strip()


def tool_classify_question(docs_json: str, question: str, contra_json: str, client, model: str) -> str:
    docs = json.loads(docs_json) if isinstance(docs_json, str) else docs_json
    contra = json.loads(contra_json) if isinstance(contra_json, str) and contra_json != "none" else {}
    docs_text = "\n".join([
        f"{d.get('doc_id')}: {d.get('title')} [current={d.get('is_current', True)}]"
        for d in docs
    ])
    from shared.state import generate
    result, _, _ = generate(client, model, (
        f"Classify this policy question.\nQuestion: {question}\n\n"
        f"Docs: {docs_text}\nContradiction: {'FOUND' if contra.get('contradiction_found') else 'None'}\n\n"
        "Type: single_doc|composition|contradiction|out_of_scope.\n"
        "single_doc: answerable from one document.\n"
        "composition: needs info from 2+ documents.\n"
        "contradiction: documents contain conflicting facts.\n"
        "out_of_scope: no relevant documents found.\n"
        "Return JSON: {\"question_type\":\"...\", \"relevant_docs\":[...], \"contradictions_found\":bool, \"reasoning\":\"...\"}"
    ))
    return result.replace("```json", "").replace("```", "").strip()


def tool_compose_single_doc(chunks_json: str, question: str, client, model: str, language: str = "en") -> str:
    chunks = json.loads(chunks_json) if isinstance(chunks_json, str) else chunks_json
    ct = "\n\n".join([f"[{c['doc_id']}]: {c.get('text_snippet', c.get('text', ''))}" for c in chunks])
    from shared.state import generate
    result, _, _ = generate(client, model, (
        f"Answer using ONLY this policy text.\n"
        f"Answer in {language} language.\n"
        f"Question: {question}\n\nPolicy text:\n{ct}\n\n"
        "Cite every claim as [doc_id]. Return ONLY JSON: "
        "{\"answer\":\"...\", \"citations\":[{\"doc_id\":\"...\", \"snippet\":\"...\"}]}"
    ))
    return result.replace("```json", "").replace("```", "").strip()


def tool_compose_multi_doc(chunks_json: str, question: str, client, model: str, language: str = "en") -> str:
    chunks = json.loads(chunks_json) if isinstance(chunks_json, str) else chunks_json
    ct = "\n\n".join([f"[{c['doc_id']}]: {c.get('text_snippet', c.get('text', ''))}" for c in chunks])
    from shared.state import generate
    result, _, _ = generate(client, model, (
        f"Synthesize from multiple policies.\n"
        f"Answer in {language} language.\n"
        f"Question: {question}\n\nPolicy texts:\n{ct}\n\n"
        "Cite every claim as [doc_id]. Return ONLY JSON: "
        "{\"answer\":\"...\", \"citations\":[{\"doc_id\":\"...\", \"snippet\":\"...\"}], \"sources\":[...]}"
    ))
    return result.replace("```json", "").replace("```", "").strip()


def tool_compose_contradiction(contra_json: str, question: str, client, model: str, language: str = "en") -> str:
    from shared.state import generate
    result, _, _ = generate(client, model, (
        f"Surface a contradiction without picking a side.\n"
        f"Answer in {language} language.\n"
        f"Question: {question}\nReport: {contra_json}\n\n"
        "State: 'Policy [A] says [X], but Policy [B] says [Y]'. Do NOT pick one. "
        "Advise consulting relevant department. Return ONLY JSON: "
        "{\"answer\":\"...\", \"citations\":[{\"doc_id\":\"...\", \"snippet\":\"...\"}]}"
    ))
    return result.replace("```json", "").replace("```", "").strip()


def tool_compose_refusal(question: str, client, model: str, language: str = "en") -> str:
    from shared.state import generate
    result, _, _ = generate(client, model, (
        f"Write polite refusal.\n"
        f"Answer in {language} language.\n"
        f"Question: {question}\n\n"
        "State: 'I cannot find an authoritative answer in the policy corpus.' "
        "No guessing. Suggest contacting relevant department. "
        "Return ONLY JSON: {\"answer\":\"...\", \"citations\":[]}"
    ))
    return result.replace("```json", "").replace("```", "").strip()


def tool_evaluate_answer(answer_json: str, question: str, chunks_json: str, client, model: str, language: str = "en") -> str:
    ad = json.loads(answer_json) if isinstance(answer_json, str) else answer_json
    chunks = json.loads(chunks_json) if isinstance(chunks_json, str) else chunks_json
    ct = "\n---\n".join([f"[{c['doc_id']}]: {c.get('text_snippet', c.get('text', ''))}" for c in chunks])
    from shared.state import generate
    result, _, _ = generate(client, model, (
        f"Evaluate answer against chunks.\n"
        f"Question: {question}\n\nAnswer: {ad.get('answer', '')}\n\nChunks:\n{ct}\n\n"
        "Check:\n"
        "1. GROUNDING: Is every factual sentence in the answer backed by a retrieved chunk? Sentence-by-sentence.\n"
        "2. CITATIONS: Do citations reference real document IDs from the chunks?\n"
        "3. COMPLETENESS: Does the answer address ALL parts of the question?\n"
        f"4. LANGUAGE: The answer MUST be in {language} language. If not, flag as a major issue.\n"
        "5. THINKING TEXT: If the answer contains reasoning patterns like 'Wait,', 'Let me', 'I should', "
        "'I need to', 'Let me check' — this is NOT an answer, flag as critical.\n\n"
        "Return ONLY JSON: {\"overall_pass\": bool, "
        "\"grounding\": {\"score\":0-1, \"explanation\":\"...\", \"sentences\":[{\"index\":1,\"text\":\"...\", \"grounded\":bool, \"source\":\"...\"}]}, "
        "\"citation_accuracy\": {\"score\":0-1, \"explanation\":\"...\", \"issues\":[]}, "
        "\"completeness\": {\"score\":0-1, \"explanation\":\"...\", \"missing_parts\":[]}, "
        "\"language_match\": {\"pass\": bool, \"expected\": \"" + language + "\", \"actual\": \"...\"}, "
        "\"thinking_text\": {\"detected\": bool, \"explanation\": \"...\"}, "
        "\"actionable_fixes\": [{\"severity\":\"critical|major|minor\", \"location\":\"...\", \"problem\":\"...\", "
        "\"suggested_action\":\"remove|re_search|re_compose|replace\", \"detail\":\"...\"}]}"
    ))
    return result.replace("```json", "").replace("```", "").strip()


def tool_evaluate_compliance(answer_json: str, chunks_json: str, classification: str, client, model: str, language: str = "en") -> str:
    ad = json.loads(answer_json) if isinstance(answer_json, str) else answer_json
    chunks = json.loads(chunks_json) if isinstance(chunks_json, str) else chunks_json
    ct = "\n---\n".join([
        f"[{c['doc_id']}] current={c.get('is_current', True)} superseded_by={c.get('superseded_by')}: {c.get('text_snippet', c.get('text', ''))}"
        for c in chunks
    ])
    from shared.state import generate
    result, _, _ = generate(client, model, (
        f"Evaluate compliance. Classification: {classification}\n\n"
        f"Answer: {ad.get('answer', '')}\n\nChunks:\n{ct}\n\n"
        f"Answer should be in {language} language.\n\n"
        "Check:\n"
        "1. SOURCE CURRENCY: Are cited documents current (not superseded)? "
        "If a citation references a document with superseded_by set AND the question does NOT "
        "explicitly ask for an old/previous version, flag as critical and target retriever.\n"
        "2. CONTRADICTION HANDLING (only if classification=contradiction): "
        "Did the answer properly surface the conflict without picking one side?\n"
        "3. REFUSAL CORRECTNESS (only if classification=out_of_scope): "
        "Did the answer properly refuse without hallucinating?\n"
        "Return ONLY JSON: {\"overall_pass\": bool, "
        "\"source_currency\": {\"score\":0-1, \"explanation\":\"...\", \"issues\":[]}, "
        "\"contradiction_handling\": {\"applicable\":bool, \"score\":0-1|null, \"explanation\":\"...\"}, "
        "\"refusal_correctness\": {\"applicable\":bool, \"score\":0-1|null, \"explanation\":\"...\"}, "
        "\"primary_target\": \"retriever|classifier|composer|approved\", "
        "\"actionable_fixes\": [{\"severity\":\"...\", \"location\":\"...\", \"problem\":\"...\", "
        "\"target\":\"retriever|classifier|composer\", \"detail\":\"...\"}]}"
    ))
    return result.replace("```json", "").replace("```", "").strip()


def get_full_text(doc_ids: list[str]) -> str:
    texts = []
    for did in doc_ids:
        try:
            chunks = _indexer.get_chunks_for_doc(did)
            if chunks:
                texts.extend([c["text"] for c in chunks])
        except Exception:
            pass
    return "\n\n".join(texts)


class TranslationSchema(BaseModel):
    translation: str

def tool_translate_to_english(text: str, client, model: str) -> str:
    from shared.state import generate
    
    prompt = f"""Translate the following Arabic text to English professionally for HR manuals.
Input: {text}"""

    result, _, _ = generate(
        client=client, 
        model=model, 
        prompt=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": TranslationSchema,
            "temperature": 0.1 # Low temperature for more deterministic translation
        }
    )
    
    # 3. Safely load the guaranteed JSON string
    try:
        parsed = json.loads(result)
        return parsed.get("translation", "").strip()
    except json.JSONDecodeError:
        return "" # Fallback in case of a severe API error

def tool_translate_to_arabic(text: str, client, model: str) -> str:
    from shared.state import generate
    
    prompt = f"""Translate the following English text to Arabic professionally for HR manuals.
Input: {text}"""

    result, _, _ = generate(
        client=client, 
        model=model, 
        prompt=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": TranslationSchema,
            "temperature": 0.1
        }
    )
    
    try:
        parsed = json.loads(result)
        return parsed.get("translation", "").strip()
    except json.JSONDecodeError:
        return ""


def tool_detect_language(text: str, client, model: str) -> str:
    from shared.state import generate
    result, _, _ = generate(client, model,
        f"Detect the language of this text. Return ONLY 'ar' if Arabic, 'en' if English, or the ISO 639-1 code:\n\n{text}"
    )
    return result.strip().lower()
