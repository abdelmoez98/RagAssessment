#!/usr/bin/env python3
"""One-shot ingestion: parses corpus, chunks documents, indexes into ChromaDB + BM25.
Idempotent — clears and rebuilds on each run."""

import json
import os
import sys

from shared.config import POLICY_CORPUS_DIR
from ingestion.parser import parse_document
from ingestion.chunker import chunk_document
from ingestion.indexer import Indexer


def ingest():
    metadata_path = os.path.join(POLICY_CORPUS_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"ERROR: metadata.json not found at {metadata_path}")
        sys.exit(1)

    with open(metadata_path, "r") as f:
        data = json.load(f)
    documents = data["documents"]

    indexer = Indexer()
    print("Clearing existing index...")
    indexer.clear()

    total_chunks = 0
    total_docs = 0

    for doc_id, meta in sorted(documents.items()):
        filename = f"{doc_id}"
        for ext in [".md", ".pdf", ".docx"]:
            filepath = os.path.join(POLICY_CORPUS_DIR, filename + ext)
            if os.path.exists(filepath):
                break
        else:
            print(f"  SKIP {doc_id}: file not found")
            continue

        try:
            text, fmt = parse_document(filepath)
        except Exception as e:
            print(f"  SKIP {doc_id}: parse error — {e}")
            continue

        meta["format"] = fmt

        try:
            chunks = chunk_document(text, doc_id, meta)
        except Exception as e:
            print(f"  SKIP {doc_id}: chunk error — {e}")
            continue

        indexer.index_chunks(chunks)
        total_chunks += len(chunks)
        total_docs += 1
        print(f"  OK  {doc_id} ({fmt}): {len(chunks)} chunks — {meta.get('title', '')}")

    print(f"\nIndexed {total_chunks} chunks from {total_docs} documents into {len(documents)} metadata entries")
    print("Done.")


if __name__ == "__main__":
    ingest()
