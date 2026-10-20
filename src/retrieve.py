"""Step 9 — three retrievers behind one interface: dense, lexical, hybrid.

    Dense   : the bge embeddings in Chroma. Searches by meaning.
    Lexical : BM25 over the same chunk text. Searches by words.
    Hybrid  : both, merged with reciprocal rank fusion.

Why lexical retrieval earns a place next to a perfectly good embedding model:
the questions this corpus has to answer are full of exact accounting line-item
names -- "net cash provided by operating activities", "share-based
compensation" -- and a dense retriever has no notion of an exact string. It
matches meaning, which is why it finds a paragraph about cash generation when
asked about cash flow, and why it can miss the statement line that carries the
figure. BM25 has the opposite blindness: it cannot tell that "revenues
declined" and "sales decreased" are the same claim.

Fusion is by RANK, not by score. A cosine similarity of 0.78 and a BM25 score
of 14.2 are not on comparable scales and normalising them means inventing a
relationship between them. Reciprocal rank fusion only asks where each
retriever placed a document:

    score(d) = sum over retrievers of  1 / (RRF_K + rank(d))

All three expose the same search(query, k, where) signature, so the Step 10
router composes with any of them rather than being rewritten per retriever.
"""

import json
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from src import config, embed
from src.config import CHROMA, CHUNKS
from src.store import get_store

RRF_K = 60          # the constant from the original RRF paper; damps rank 1's weight
CANDIDATES = 200    # per-retriever depth before fusion

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. No stemming: these are proper accounting terms,
    and stemming 'operating' to 'oper' loses more than it gains on a corpus
    this small."""
    return TOKEN_RE.findall(text.lower())


class Dense:
    name = "dense"

    def __init__(self, strategy: str):
        self.store = get_store(CHROMA, config.collection_name(strategy))

    def search(self, query: str, k: int, where: dict | None = None) -> list[dict]:
        return self.store.query(embed.embed_query(query), k=k, where=where)


@lru_cache(maxsize=4)
def _load_chunks(strategy: str) -> tuple[list[dict], BM25Okapi]:
    """Build the BM25 index from the same chunk file the dense index was built
    from, so the two retrievers are searching identical text."""
    path = CHUNKS / f"{strategy}.jsonl"
    chunks = [json.loads(l) for l in path.open()]
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    return chunks, bm25


class Lexical:
    name = "bm25"

    def __init__(self, strategy: str):
        self.chunks, self.bm25 = _load_chunks(strategy)

    def search(self, query: str, k: int, where: dict | None = None) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            c = self.chunks[i]
            # Metadata filtering has to work here too, or hybrid could not be
            # used inside Step 10's per-company searches.
            if where and any(c.get(f) != v for f, v in where.items()):
                continue
            if scores[i] <= 0:
                break
            out.append({
                "chunk_id": c["chunk_id"], "text": c["text"],
                "metadata": {f: c.get(f) for f in c if f not in ("text",)},
                "distance": None, "score": float(scores[i]),
            })
            if len(out) >= k:
                break
        return out


class Hybrid:
    name = "hybrid"

    def __init__(self, strategy: str):
        self.dense = Dense(strategy)
        self.lexical = Lexical(strategy)

    def search(self, query: str, k: int, where: dict | None = None) -> list[dict]:
        runs = [self.dense.search(query, CANDIDATES, where),
                self.lexical.search(query, CANDIDATES, where)]
        fused: dict[str, float] = {}
        best: dict[str, dict] = {}
        for run in runs:
            for rank, h in enumerate(run):
                cid = h["chunk_id"]
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
                # keep the dense copy where both found it: its score is the one
                # a reader can interpret, and ask.py prints it
                if cid not in best or h["distance"] is not None:
                    best[cid] = h
        ranked = sorted(fused, key=lambda c: -fused[c])[:k]
        out = []
        for cid in ranked:
            h = dict(best[cid])
            h["rrf_score"] = round(fused[cid], 6)
            out.append(h)
        return out


def get_retriever(kind: str, strategy: str):
    return {"dense": Dense, "bm25": Lexical, "hybrid": Hybrid}[kind](strategy)
