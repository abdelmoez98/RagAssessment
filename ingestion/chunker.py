from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_document(text: str, doc_id: str, metadata: dict) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = splitter.split_text(text)
    result = []

    for i, chunk_text in enumerate(chunks):
        result.append(
            {
                "chunk_id": f"{doc_id}:chunk_{i}",
                "doc_id": doc_id,
                "title": metadata.get("title", ""),
                "text": chunk_text.strip(),
                "category": metadata.get("category", ""),
                "department": metadata.get("department", ""),
                "effective_date": metadata.get("effective_date", ""),
                "superseded_by": metadata.get("superseded_by"),
                "is_current": not metadata.get("superseded_by"),
                "format": metadata.get("format", ""),
            }
        )

    return result
