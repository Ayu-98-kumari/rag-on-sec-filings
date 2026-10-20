"""Vector store behind a thin interface.

Chroma is the implementation, but every Chroma-specific call lives in this one
module. Swapping in pgvector later means writing a second class with the same
four methods and changing one line in src/index.py.

Why Chroma and not a database server: the corpus is ~2,300 vectors of 384
dimensions, about 3.5 MB. That does not justify running Postgres, and Chroma
persists to a directory that ships inside the deployed app with no external
service to connect to. pgvector would be the right call past roughly a million
vectors, or as soon as the vectors needed to be joined against relational data.
"""

import json
from pathlib import Path
from typing import Any, Protocol

import chromadb

# Chroma metadata values must be str / int / float / bool. Lists are not
# allowed, so list-valued fields are JSON-encoded on the way in and decoded
# on the way out.
LIST_FIELDS = ("sections_touched", "source_block_ids")


class VectorStore(Protocol):
    def add(self, ids: list[str], texts: list[str],
            embeddings: list[list[float]], metadatas: list[dict]) -> None: ...
    def query(self, embedding: list[float], k: int,
              where: dict | None = None) -> list[dict]: ...
    def count(self) -> int: ...


class ChromaStore:
    def __init__(self, path: Path, collection: str, reset: bool = False):
        self.client = chromadb.PersistentClient(path=str(path))
        if reset:
            try:
                self.client.delete_collection(collection)
            except Exception:
                pass
        self.col = self.client.get_or_create_collection(
            name=collection,
            # cosine similarity: the bge embeddings are L2-normalised, so
            # cosine distance is the right metric rather than Chroma's L2 default.
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _encode_meta(m: dict) -> dict:
        out = {}
        for k, v in m.items():
            if k in LIST_FIELDS:
                out[k] = json.dumps(v)
            elif v is None:
                continue                      # Chroma rejects None values
            else:
                out[k] = v
        return out

    @staticmethod
    def _decode_meta(m: dict) -> dict:
        out = dict(m)
        for k in LIST_FIELDS:
            if k in out and isinstance(out[k], str):
                try:
                    out[k] = json.loads(out[k])
                except json.JSONDecodeError:
                    pass
        return out

    def add(self, ids, texts, embeddings, metadatas, batch: int = 500) -> None:
        for i in range(0, len(ids), batch):
            self.col.add(
                ids=ids[i:i + batch],
                documents=texts[i:i + batch],
                embeddings=[list(map(float, e)) for e in embeddings[i:i + batch]],
                metadatas=[self._encode_meta(m) for m in metadatas[i:i + batch]],
            )

    def query(self, embedding, k: int = 5, where: dict | None = None) -> list[dict]:
        res = self.col.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            hits.append({
                "chunk_id": cid,
                "text": doc,
                "metadata": self._decode_meta(meta),
                "distance": dist,
                "score": 1.0 - dist,        # cosine distance -> similarity
            })
        return hits

    def count(self) -> int:
        return self.col.count()


def get_store(path: Path, collection: str, reset: bool = False) -> VectorStore:
    """The single place the concrete backend is chosen."""
    return ChromaStore(path, collection, reset=reset)
