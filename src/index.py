"""Step 4 — embed every chunk and load it into the vector store.

    python -m src.index                 # build from data/chunks/baseline.jsonl
    python -m src.index --strategy improved

Run once per chunking strategy. The question asked at query time is embedded
live (milliseconds); the corpus is not re-embedded per query.
"""

import argparse
import json
import time

from src import config, embed
from src.config import CHROMA, CHUNKS, ROOT
from src.store import get_store

# Fields copied from the chunk into the store as filterable metadata. These are
# what make Step 10's comparative route possible: without them we could only
# search everything and hope.
META_FIELDS = ("ticker", "company", "cik", "form", "period", "fiscal_year",
               "section", "sections_touched", "source_block_ids",
               "body_tokens", "total_tokens", "strategy")

SMOKE_QUERIES = [
    "what risks does the company face from rising interest rates",
    "total revenues for the fiscal year",
    "how does the company account for title insurance loss reserves",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=config.DEFAULT_STRATEGY,
                    help=f"chunk file stem in data/chunks "
                         f"(default: {config.DEFAULT_STRATEGY})")
    args = ap.parse_args()

    src_file = CHUNKS / f"{args.strategy}.jsonl"
    if not src_file.exists():
        raise SystemExit(f"no chunk file at {src_file}")

    chunks = [json.loads(l) for l in src_file.open()]
    print(f"loaded {len(chunks)} chunks from {src_file.relative_to(ROOT)}")
    print(f"embedding model: {config.EMBED_MODEL} on {embed.device()}  "
          f"({config.EMBED_DIM} dimensions)")

    t0 = time.time()
    vectors = embed.embed_passages([c["text"] for c in chunks], show_progress=True)
    t_embed = time.time() - t0
    print(f"embedded {len(vectors)} chunks in {t_embed:.1f}s "
          f"({len(vectors)/max(t_embed,1e-9):.0f}/s), shape {vectors.shape}")

    collection = config.collection_name(args.strategy)
    store = get_store(CHROMA, collection, reset=True)
    t0 = time.time()
    store.add(
        ids=[c["chunk_id"] for c in chunks],
        texts=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[{k: c.get(k) for k in META_FIELDS} for c in chunks],
    )
    print(f"stored in collection '{collection}' in {time.time()-t0:.1f}s")

    # --- verification ---
    print("\n=== verification ===")
    print(f"  vectors in store: {store.count()}  (expected {len(chunks)})")
    assert store.count() == len(chunks), "count mismatch"

    print("\n=== smoke queries (eyeball only; real measurement is Step 7) ===")
    for q in SMOKE_QUERIES:
        hits = store.query(embed.embed_query(q), k=3)
        print(f"\n  Q: {q}")
        for h in hits:
            m = h["metadata"]
            snippet = h["text"].split("\n\n", 1)[-1][:90].replace("\n", " ")
            print(f"     {h['score']:.3f}  {m['ticker']} {m['form']} FY{m['fiscal_year']} "
                  f"Item {m.get('section') or '-':3s} | {snippet}...")

    print("\n=== metadata filtering works (needed for Step 10) ===")
    q = "total revenues for the fiscal year"
    for tk in config.TICKERS:
        hits = store.query(embed.embed_query(q), k=1, where={"ticker": tk})
        m = hits[0]["metadata"]
        print(f"  filter ticker={tk}: got {m['ticker']} {m['form']} "
              f"FY{m['fiscal_year']} Item {m.get('section') or '-'}")


if __name__ == "__main__":
    main()
