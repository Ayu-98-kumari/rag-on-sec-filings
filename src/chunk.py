"""Steps 3 and 8 — cut the parsed blocks into chunks. Two strategies.

    python -m src.chunk                      # baseline  (Step 3)
    python -m src.chunk --strategy improved  # structure-aware (Step 8)

BASELINE is deliberately naive: fixed 384-token windows, document structure
ignored, tables flattened to a single line. It exists so the improved strategy
has an honest baseline to be measured against.

IMPROVED is the Step 8 response to what Step 7 measured. Lookup recall@5 was
46.7 per cent, and the failures were ordered by where the figure lives: revenue
and net income scored 3 of 3 because they also appear in MD&A prose, while
operating cash flow and share-based compensation scored 0 of 3 because they
exist only inside tables. The reason is visible in First American's segment
revenue table, which the model was asked for total revenues from. Flattened, it
becomes an unbroken run of digits; the model summed three of five revenue
columns and reported $6,810.3m against a true $7,452.2m. But that table HAS a
"Total Revenues" column whose values sum to exactly 7,452.2. The information
was present and the structure that made it readable was destroyed.

So the improved strategy does four things:
  1. One table per chunk, never split mid-table. Oversized tables split on row
     boundaries with the column header repeated on every piece.
  2. Column headers and row labels preserved, spacer and currency cells
     dropped, negatives split across cells ('( 0.3' + ')') rejoined. Nil
     markers are KEPT, because dropping them shifts every later column.
  3. The caption and the unit scale written into the chunk text in words, so
     "millions" is searchable and the magnitude travels with the figures.
  4. Prose never spans two Items; a chunk is flushed at a section boundary.

The document header prepended to every chunk is held CONSTANT across both
strategies, so the only variable between them is structural awareness.
Otherwise the comparison would conflate two changes.
"""

import argparse
import json
import re
import warnings

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer

from src.config import (CHUNKS as OUT, CHUNK_TOKENS, EMBED_MODEL,
                        OVERLAP_TOKENS, PARSED, ROOT)

FORM_LABEL = {"10-K": "10-K", "10-Q": "10-Q"}

# Cells that carry no information once the grid is linearised. The nil markers
# are NOT here on purpose: an em dash in a financial table means zero, and
# dropping it shifts every subsequent column, which silently reassigns figures
# to the wrong heading -- the exact failure this step exists to fix.
NOISE_CELLS = {"", "$", "%", "(", ")"}
MIN_CHUNK_TOKENS = 48         # below this a chunk is a fragment, not a passage
CAPTION_LOOKBACK = 3          # blocks to search backwards for a table caption
CAPTION_MAX_CHARS = 300
SCALE_PHRASE = {"thousand": "Figures in thousands.",
                "million": "Figures in millions.",
                "billion": "Figures in billions."}


def flatten_table(rows: list[list[str]]) -> str:
    """Baseline table handling: flatten to a single line, deliberately lossy.

    This is what a naive pipeline does, and it is what Step 8 will fix.
    """
    return " ".join(" ".join(c for c in row if c) for row in rows)


def clean_row(row: list[str]) -> list[str]:
    """Strip grid noise while keeping column alignment intact.

    Filing tables are laid out for print, so a single logical column arrives as
    three physical cells: a blank spacer, a currency symbol, and the figure. A
    negative is split further still, as '( 0.3' followed by ')'. Both are
    rejoined here. Nil markers survive, because they hold a column position.
    """
    out, i = [], 0
    while i < len(row):
        c = (row[i] or "").strip()
        if c.startswith("(") and not c.endswith(")") and len(c) > 1:
            num = c.lstrip("( ").strip()
            j = i + 1
            while j < len(row) and (row[j] or "").strip() == "":
                j += 1
            out.append(f"({num})")
            i = (j + 1) if j < len(row) and (row[j] or "").strip() == ")" else j
            continue
        if c not in NOISE_CELLS:
            out.append(c)
        i += 1
    return out


def find_caption(blocks: list[dict], i: int) -> str:
    """A table's caption is usually the paragraph just above it, not a cell."""
    for prev in reversed(blocks[max(0, i - CAPTION_LOOKBACK):i]):
        if prev["type"] == "paragraph":
            t = prev.get("text", "").strip()
            if 0 < len(t) <= CAPTION_MAX_CHARS:
                return t
            return ""
    return ""


def serialize_table(b: dict, caption: str, rows: list[list[str]] | None = None,
                    part: tuple[int, int] | None = None) -> str:
    """Render a table as text that still says which figure sits under which heading."""
    rows = rows if rows is not None else [clean_row(r) for r in b["rows"]]
    rows = [r for r in rows if r]
    head = ["[TABLE]" + (f" part {part[0]} of {part[1]}" if part else "")]
    sec = b.get("section")
    if sec:
        head.append(f"Section: Item {sec} - {b.get('section_title') or ''}".rstrip(" -"))
    if caption:
        head.append(f"Caption: {caption}")
    if b.get("unit_scale"):
        head.append(SCALE_PHRASE.get(b["unit_scale"], ""))
    body = []
    if rows:
        body.append("Columns: " + " | ".join(rows[0]))
        body += [" | ".join(r) for r in rows[1:]]
    return "\n".join([h for h in head if h] + body)


def block_text(b: dict) -> str:
    if b["type"] == "table":
        return flatten_table(b["rows"])
    return b.get("text", "")


def doc_header(meta: dict) -> str:
    """Identity prepended to every chunk so retrieval can tell the filers apart."""
    return (f"{meta['company']} | {FORM_LABEL.get(meta['form'], meta['form'])} "
            f"FY{meta['fiscal_year']} (period {meta['period']})")


def chunk_filing(doc: dict, tok) -> list[dict]:
    meta = doc["meta"]
    header = doc_header(meta)
    header_tokens = len(tok.encode(header, add_special_tokens=False))
    body_budget = CHUNK_TOKENS - header_tokens - 2   # 2 for the blank-line join
    assert body_budget > 100, f"header too long ({header_tokens} tokens)"

    # Build one text stream, remembering which block each character came from.
    spans, texts, cursor = [], [], 0
    for b in doc["blocks"]:
        t = block_text(b).strip()
        if not t:
            continue
        texts.append(t)
        spans.append((cursor, cursor + len(t), b["block_id"], b.get("section")))
        cursor += len(t) + 2            # the "\n\n" join
    full = "\n\n".join(texts)
    if not full:
        return []

    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    stride = body_budget - OVERLAP_TOKENS

    chunks = []
    for start in range(0, n, stride):
        end = min(start + body_budget, n)
        if end - start < 20 and chunks:      # trailing scrap, already covered
            break
        c0, c1 = offsets[start][0], offsets[end - 1][1]
        body = full[c0:c1].strip()
        if not body:
            continue

        touched = [(bid, sec) for (s0, s1, bid, sec) in spans if s0 < c1 and s1 > c0]
        sections = [s for _, s in touched if s]
        primary = sections[0] if sections else None

        chunks.append({
            "chunk_id": f"{meta['ticker']}_{meta['form'].replace('-', '')}_{meta['period']}_c{len(chunks):05d}",
            "text": f"{header}\n\n{body}",
            "body_tokens": end - start,
            "total_tokens": header_tokens + 2 + (end - start),
            "ticker": meta["ticker"],
            "company": meta["company"],
            "cik": meta["cik"],
            "form": meta["form"],
            "period": meta["period"],
            "fiscal_year": meta["fiscal_year"],
            "section": primary,
            "sections_touched": sorted(set(sections), key=str),
            "source_block_ids": [bid for bid, _ in touched],
            "strategy": "baseline_fixed_384",
        })
        if end == n:
            break
    return chunks



def chunk_filing_improved(doc: dict, tok) -> list[dict]:
    """Step 8: respect the document's structure instead of ignoring it.

    Prose accumulates until the budget is reached OR the Item changes, so a
    chunk never straddles two sections. Tables are emitted whole, one per
    chunk, with their caption, section and unit scale attached; a table larger
    than the budget is split on row boundaries with the column header repeated
    on every part, so no part is a headerless block of digits.
    """
    meta = doc["meta"]
    header = doc_header(meta)
    header_tokens = len(tok.encode(header, add_special_tokens=False))
    budget = CHUNK_TOKENS - header_tokens - 2
    assert budget > 100, f"header too long ({header_tokens} tokens)"

    chunks: list[dict] = []
    blocks = doc["blocks"]

    def emit(body: str, bids: list[str], sections: list, n_tok: int) -> None:
        secs = [x for x in sections if x]
        chunks.append({
            "chunk_id": f"{meta['ticker']}_{meta['form'].replace('-', '')}_{meta['period']}_c{len(chunks):05d}",
            "text": f"{header}\n\n{body}",
            "body_tokens": n_tok,
            "total_tokens": header_tokens + 2 + n_tok,
            "ticker": meta["ticker"], "company": meta["company"],
            "cik": meta["cik"], "form": meta["form"], "period": meta["period"],
            "fiscal_year": meta["fiscal_year"],
            "section": secs[0] if secs else None,
            "sections_touched": sorted(set(secs), key=str),
            "source_block_ids": bids,
            "strategy": "improved_structure_aware",
        })

    def emit_windowed(text: str, bid: str, sec, prefix: str = "") -> None:
        """Emit text that exceeds the budget as overlapping token windows.

        Needed because a single unit can be bigger than a chunk: FNF has
        two-column tables whose one cell is a whole accounting policy, 700-plus
        tokens in a single row, and a row cannot be split on row boundaries. An
        over-budget chunk is not a cosmetic problem -- the embedding model has a
        hard 512-token limit and truncates past it silently, so the tail of the
        text would be indexed as if it did not exist.
        """
        n = len(tok.encode(text, add_special_tokens=False))
        if n <= budget:
            emit(text, [bid], [sec], n)
            return
        pre_tok = len(tok.encode(prefix, add_special_tokens=False)) if prefix else 0
        room = budget - pre_tok - 1
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        stride = max(room - OVERLAP_TOKENS, 1)
        for st in range(0, len(offs), stride):
            en = min(st + room, len(offs))
            piece = text[offs[st][0]:offs[en - 1][1]].strip()
            body = f"{prefix}\n{piece}" if prefix else piece
            emit(body, [bid], [sec],
                 len(tok.encode(body, add_special_tokens=False)))
            if en == len(offs):
                break

    buf, buf_bids, buf_secs, buf_toks, buf_section = [], [], [], [], "__none__"

    def flush(force: bool = False, overlap: bool = False,
              next_tok: int = 0) -> None:
        """Write the prose buffer out, unless it is too small to be a passage.

        Flushing at every section boundary produced fragments: the first run of
        this strategy emitted a seven-token chunk reading "For the fiscal year
        ended December 31, 2025", and on a short query that fragment beat the
        statement it was a heading for -- it took rank 1 for Stewart's revenue
        question. A chunk that short carries no evidence but still occupies one
        of the five slots the model gets, so small buffers are carried into the
        next chunk instead of being emitted. `force` is used at end of document,
        where there is nothing left to carry into.
        """
        nonlocal buf, buf_bids, buf_secs, buf_toks, buf_section
        if not buf:
            return
        if sum(buf_toks) < MIN_CHUNK_TOKENS and not force:
            buf_section = "__none__"      # keep the text, drop the boundary
            return
        emit("\n\n".join(buf), buf_bids, buf_secs, sum(buf_toks))

        # Carry a tail of the buffer into the next chunk when the flush was
        # caused by a full budget. The baseline had 48 tokens of overlap and
        # this strategy lost it in the rewrite, which is the likeliest cause of
        # its one regression: a narrative answer often straddles a paragraph
        # break, with the risk named at the end of one paragraph and its
        # consequence at the start of the next, so without overlap neither
        # chunk holds the whole thought.
        #
        # No overlap across a SECTION boundary, which is the whole point of the
        # strategy, and never carry the entire buffer -- leaving at least one
        # paragraph behind is what guarantees forward progress.
        # `next_tok` is the paragraph about to be appended by the caller. The
        # tail has to leave room for it, or the carried overlap would push the
        # next chunk over the budget -- which it did on three chunks before this
        # argument existed, and an over-budget chunk is truncated by the
        # embedding model without raising anything.
        keep: list[int] = []
        if overlap:
            run = 0
            for j in range(len(buf) - 1, 0, -1):
                if (buf_toks[j] > OVERLAP_TOKENS
                        or run + buf_toks[j] > OVERLAP_TOKENS
                        or run + buf_toks[j] + next_tok > budget):
                    break
                run += buf_toks[j]
                keep.insert(0, j)
        buf = [buf[j] for j in keep]
        buf_bids = [buf_bids[j] for j in keep]
        buf_secs = [buf_secs[j] for j in keep]
        buf_toks = [buf_toks[j] for j in keep]
        buf_section = buf_secs[-1] if buf_secs else "__none__"

    for i, b in enumerate(blocks):
        if b["type"] == "table":
            # A table is emitted as its own chunk, but it does NOT flush the
            # prose buffer. Flushing here was the first design and it cost 25
            # points of conceptual recall: tables are 38 per cent of blocks and
            # are interleaved through the narrative, so every one of them cut
            # the surrounding prose into a short fragment. Narrative questions
            # need a passage, not a sentence. The table is lifted out and the
            # prose on either side of it stays together.
            caption = find_caption(blocks, i)
            rows = [r for r in (clean_row(r) for r in b["rows"]) if r]
            if not rows:
                continue
            text = serialize_table(b, caption, rows)
            n = len(tok.encode(text, add_special_tokens=False))
            if n < MIN_CHUNK_TOKENS:
                # a one-cell "table" is layout, not data; let it ride with prose
                if buf and sum(buf_toks) + n > budget:
                    flush(force=True, overlap=True, next_tok=n)
                buf.append(text); buf_bids.append(b["block_id"])
                buf_secs.append(b.get("section")); buf_toks.append(n)
                continue
            if n <= budget:
                emit(text, [b["block_id"]], [b.get("section")], n)
                continue

            # Oversized: split on row boundaries, repeating the column header.
            col_header, data = rows[0], rows[1:]
            parts, cur = [], []
            for r in data:
                trial = serialize_table(b, caption, [col_header] + cur + [r])
                if len(tok.encode(trial, add_special_tokens=False)) > budget and cur:
                    parts.append(cur); cur = [r]
                else:
                    cur.append(r)
            if cur:
                parts.append(cur)
            for pi, rs in enumerate(parts, 1):
                text = serialize_table(b, caption, [col_header] + rs,
                                       part=(pi, len(parts)))
                head, _, rest = text.partition("Columns: ")
                emit_windowed(text, b["block_id"], b.get("section"),
                              prefix=(head + "Columns: " +
                                      rest.split("\n", 1)[0]).strip())
            continue

        t = (b.get("text") or "").strip()
        if not t:
            continue
        sec = b.get("section")
        if buf and sec != buf_section:
            flush()                       # never straddle an Item boundary
        n = len(tok.encode(t, add_special_tokens=False))
        if buf and sum(buf_toks) + n > budget:
            # force, even for a small buffer: the carry-forward rule exists to
            # avoid fragments, and must not be allowed to overflow the budget.
            # Without force, a 20-token heading held in the buffer would accept
            # a 380-token paragraph on top of it and the chunk would be
            # truncated by the embedding model with no error raised.
            flush(force=True, overlap=True, next_tok=n)
        if n > budget:                    # single huge paragraph: window it
            emit_windowed(t, b["block_id"], sec)
            continue
        buf.append(t); buf_bids.append(b["block_id"]); buf_secs.append(sec)
        buf_toks.append(n)
        buf_section = sec
    flush(force=True)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="baseline",
                    choices=["baseline", "improved"])
    args = ap.parse_args()
    cut = chunk_filing if args.strategy == "baseline" else chunk_filing_improved

    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    print(f"strategy: {args.strategy}\n")

    all_chunks, straddling = [], 0
    print(f"{'filing':30s} {'blocks':>7s} {'chunks':>7s} {'avg tok':>8s} {'max tok':>8s} {'multi-sec':>10s}")
    print("-" * 76)

    for f in sorted(PARSED.glob("*.json")):
        doc = json.loads(f.read_text())
        chunks = cut(doc, tok)
        all_chunks.extend(chunks)
        multi = sum(1 for c in chunks if len(c["sections_touched"]) > 1)
        straddling += multi
        avg = sum(c["total_tokens"] for c in chunks) / max(len(chunks), 1)
        mx = max((c["total_tokens"] for c in chunks), default=0)
        print(f"{f.stem[:30]:30s} {len(doc['blocks']):7d} {len(chunks):7d} "
              f"{avg:8.0f} {mx:8d} {multi:10d}")

    dest = OUT / f"{args.strategy}.jsonl"
    with dest.open("w") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c) + "\n")

    print("-" * 76)
    print(f"{'TOTAL':30s} {'':7s} {len(all_chunks):7d}")
    print(f"\nwrote {len(all_chunks)} chunks -> {dest.relative_to(ROOT)}")

    # --- verification ---
    over = [c for c in all_chunks if c["total_tokens"] > CHUNK_TOKENS]
    no_meta = [c for c in all_chunks if not (c["ticker"] and c["period"] and c["form"])]
    no_trace = [c for c in all_chunks if not c["source_block_ids"]]
    print("\n=== verification ===")
    print(f"  chunks over the {CHUNK_TOKENS}-token budget : {len(over)}  (must be 0)")
    print(f"  chunks missing company/form/period     : {len(no_meta)}  (must be 0)")
    print(f"  chunks with no traceable source block  : {len(no_trace)}  (must be 0)")
    print(f"  chunks straddling >1 section           : {straddling} "
          f"({100*straddling/max(len(all_chunks),1):.0f}%) <- what Step 8 should reduce")
    tbl = [c for c in all_chunks if c["text"].split("\n\n", 1)[-1].startswith("[TABLE]")]
    if tbl:
        print(f"  table chunks                           : {len(tbl)} "
              f"({100*len(tbl)/len(all_chunks):.0f}% of chunks)")
    per_co = {}
    for c in all_chunks:
        per_co[c["ticker"]] = per_co.get(c["ticker"], 0) + 1
    tot = len(all_chunks)
    print("\n=== corpus balance (retrieval bias risk) ===")
    for k, v in sorted(per_co.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:5d} chunks ({100*v/tot:4.1f}%)")


if __name__ == "__main__":
    main()
