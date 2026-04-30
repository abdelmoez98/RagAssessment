import os
import pickle
import shutil

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

from shared.config import BM25_PATH, CHROMA_PATH, EMBEDDING_MODEL


class Indexer:
    def __init__(self):
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.chunks_for_bm25 = []

    def clear(self):
        try:
            self.chroma_client.delete_collection("policy_chunks")
        except Exception:
            pass

        if os.path.exists(BM25_PATH):
            shutil.rmtree(BM25_PATH)

        self.chunks_for_bm25 = []

    def index_chunks(self, chunks: list[dict]):
        if not chunks:
            return

        collection = self.chroma_client.get_or_create_collection(
            name="policy_chunks",
            embedding_function=self.embedding_fn,
        )

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "doc_id": chunk["doc_id"],
                    "title": chunk["title"],
                    "category": chunk["category"],
                    "department": chunk["department"],
                    "effective_date": chunk["effective_date"],
                    "superseded_by": chunk.get("superseded_by") or "",
                    "is_current": chunk["is_current"],
                    "format": chunk["format"],
                }
            )

        batch_size = 50
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        self.chunks_for_bm25.extend(chunks)
        self._save_bm25()

    def _save_bm25(self):
        os.makedirs(BM25_PATH, exist_ok=True)

        tokenized = [chunk["text"].lower().split() for chunk in self.chunks_for_bm25]
        bm25 = BM25Okapi(tokenized)
        bm25_data = {
            "bm25": bm25,
            "chunks": self.chunks_for_bm25,
        }

        with open(os.path.join(BM25_PATH, "bm25_index.pkl"), "wb") as f:
            pickle.dump(bm25_data, f)

    def load_bm25(self) -> tuple[BM25Okapi, list[dict]]:
        bm25_path = os.path.join(BM25_PATH, "bm25_index.pkl")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError("BM25 index not found. Run python ingest.py first.")

        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        return data["bm25"], data["chunks"]

    def get_chroma_collection(self):
        try:
            return self.chroma_client.get_collection(
                "policy_chunks", embedding_function=self.embedding_fn
            )
        except Exception:
            raise FileNotFoundError(
                "ChromaDB collection not found. Run python ingest.py first."
            )

    def get_chunks_for_doc(self, doc_id: str) -> list[dict]:
        collection = self.get_chroma_collection()
        result = collection.get(where={"doc_id": doc_id})
        if not result or not result["documents"]:
            return []

        chunks = []
        for i, text in enumerate(result["documents"]):
            meta = result["metadatas"][i] if result["metadatas"] else {}
            chunks.append(
                {
                    "doc_id": meta.get("doc_id", doc_id),
                    "title": meta.get("title", ""),
                    "text": text,
                }
            )
        return chunks
