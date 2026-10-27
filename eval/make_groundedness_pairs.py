"""Step 14 — build the claim/passage pairs that a human will label.

Groundedness asks one question per claim: *is this sentence supported by the
passage the model cited for it?* Answering that needs the two things paired up,
and nothing in `results/` holds them together -- the answer CSV stores the
answer text and a chunk count, not the chunk text. So the pairs are rebuilt
here.

No model is called. The answers already exist in `results/`, and retrieval is
local (bge-small on this machine plus BM25 over the parsed filings), so this
script costs nothing and can be re-run freely. Retrieval is deterministic at
temperature 0 with a fixed index, and the run's own metadata records the
configuration that produced the answers, so the passages rebuilt here are the
passages the model actually saw. That is checked rather than assumed: every
question re-asserts the stored `retrieved_tickers` and `top1_score`, and the
script refuses to write a file if any question's retrieval has drifted.

How a claim is defined, and why it is not a model's judgement:

  Rule 2 of the answering system prompt (src/ask.py) is "cite the passage
  number in square brackets after every claim, like [2]". So the model has
  already told us where its claims end. A claim is the span of text from the
  previous citation group up to and including the next one. Splitting on
  sentences instead would be a guess about what a claim is; splitting on
  citations uses the boundary the generator itself declared, and it costs no
  API call and introduces no second model's opinion into the gold labels.

  A trailing span with no citation is kept as a claim with an empty `cited`
  column. On the configuration this is built from there are none, but the
  Step 12 measurement found 10.1 per cent of numeric claims uncited on an
  earlier configuration, so the case is handled rather than assumed away.

What is deliberately NOT in the output file:

  Noul's score, and any pre-filled guess. The point of the exercise is to
  measure an automated scorer against an independent human judgement, and a
  label shown next to the question is a label the human partly agrees with by
  anchoring. The file therefore carries an empty `supported` column and
  nothing that hints at an answer. A second opinion is recorded separately,
  before the human's labels are opened, so that agreement between them is a
  measurement rather than a claim.
"""

import argparse
import collections
import csv
import json
import re
from pathlib import Path

from src import ask as ask_mod
from src import config, retrieve as retrieve_mod, route as route_mod

ANSWERS = config.ROOT / "results" / "hybrid_routed_b3830_ans.csv"
OUT = config.ROOT / "eval" / "groundedness_pairs.csv"
READABLE = config.ROOT / "eval" / "groundedness_pairs.txt"

# One or more bracketed numbers, together: "[2]", "[1][2]", "[1][2][3][4]".
CITE = re.compile(r"(?:\[\d+\])+")

# Rule 3 of the answering prompt specifies this refusal exactly, so an
# abstention is identified by the string it was told to produce rather than by
# guessing at phrasing. An abstention asserts nothing and has nothing to ground.
REFUSAL = "Not in the provided filings."


def split_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Split an answer into (claim text, cited passage numbers).

    The model's own citation placement is the boundary; see the module
    docstring for why that is better than a sentence splitter here.
    """
    out, pos = [], 0
    for m in CITE.finditer(answer):
        # A continuation claim starts where the previous citation ended, so it
        # opens with the punctuation that closed the sentence before it: ". The
        # company notes...". Stripped, because a claim that begins with a full
        # stop reads as a fragment to whoever labels it and to any scorer.
        text = answer[pos:m.start()].strip().lstrip(".,;: ").strip()
        if text:
            out.append((text, [int(n) for n in re.findall(r"\d+", m.group())]))
        pos = m.end()
    tail = answer[pos:].strip().lstrip(".,;: ").strip(" .")
    if tail:
        out.append((tail, []))
    return out


def passage_text(hit: dict) -> str:
    """One passage as the model saw it: the identity line, then the body."""
    return (f"{ask_mod.passage_label(hit['metadata'])}\n"
            f"{ask_mod.body_of(hit['text']).strip()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, default=ANSWERS,
                    help="the answer run to build pairs from")
    args = ap.parse_args()

    meta = json.loads(args.answers.with_suffix(".meta.json").read_text())
    rows = list(csv.DictReader(args.answers.open()))

    # Rebuilt from the run's own metadata rather than from defaults, so a later
    # change to config.K or the default retriever cannot silently pair claims
    # against passages the model never saw.
    searcher = retrieve_mod.get_retriever(meta["retriever"], meta["strategy"])
    k, routed = meta["k"], meta["route"]
    print(f"rebuilding passages: retriever={meta['retriever']} "
          f"strategy={meta['strategy']} k={k} route={routed}\n")

    pairs, drift, skipped = [], [], []
    for r in rows:
        # An abstention is identified by having nothing to ground, not by
        # containing the refusal string. M004 appends the refusal AFTER four
        # fully cited claims -- including a figure the model computed rather
        # than read -- so testing `REFUSAL in answer` dropped the single most
        # informative answer in the run. The real test is whether any claim
        # survives the split.
        claims = [(c, n) for c, n in split_claims(r["answer"])
                  if c.strip(" .") != REFUSAL.strip(" .")]
        if not claims:
            skipped.append(r["qid"])
            continue

        hits, _ = route_mod.retrieve(searcher, r["question"], k=k, route=routed)

        # The integrity check: same tickers in the top 5, same top-1 score.
        # If either moved, the index or the retriever has changed under us and
        # these passages are not the ones that produced the stored answer.
        tick = ",".join(sorted({h["metadata"]["ticker"] for h in hits[:5]}))
        top1 = round(hits[0]["score"], 4) if hits else 0.0
        # Compared as a number: the CSV drops trailing zeros, so 0.8250 is
        # stored as "0.825" and a string comparison reports drift that is
        # really a formatting difference.
        moved = abs(top1 - float(r["top1_score"] or 0)) > 1e-9
        if tick != r["retrieved_tickers"] or moved:
            drift.append((r["qid"], r["retrieved_tickers"], tick,
                          r["top1_score"], top1))
            continue

        for i, (claim, cited) in enumerate(claims, 1):
            # Cite numbers are 1-indexed over hits, as build_prompt writes them.
            texts = [passage_text(hits[n - 1]) for n in cited
                     if 1 <= n <= len(hits)]
            pairs.append({
                "pair_id": f"{r['qid']}-{i}",
                "qid": r["qid"],
                "type": r["type"],
                "question": r["question"],
                "claim_idx": i,
                "n_claims": len(claims),
                "claim": claim,
                "cited": ";".join(str(n) for n in cited),
                "n_passages": len(texts),
                "passages": "\n\n---\n\n".join(texts),
                # The one column for the human. Left empty on purpose: see the
                # module docstring on anchoring.
                "supported": "",
            })

    if drift:
        print("!! retrieval drifted -- these passages are NOT the ones the "
              "stored answers were produced from:")
        for qid, want_t, got_t, want_s, got_s in drift:
            print(f"   {qid}  tickers {want_t!r} -> {got_t!r}   "
                  f"top1 {want_s} -> {got_s}")
        raise SystemExit("refusing to write a file that would mislabel passages")

    by_type = collections.Counter(p["type"] for p in pairs)
    print(f"{len(rows)} answers, {len(skipped)} with nothing to ground "
          f"({', '.join(skipped)})")
    print(f"{len(pairs)} claim/passage pairs to label: {dict(by_type)}")
    uncited = [p["pair_id"] for p in pairs if not p["cited"]]
    if uncited:
        print(f"{len(uncited)} claim(s) carry no citation, so there is no "
              f"single passage to judge them against: {uncited}")

    chars = sorted(len(p["passages"]) for p in pairs)
    print(f"passage text per row: median {chars[len(chars) // 2]:,} chars, "
          f"max {chars[-1]:,}")

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)

    # A reading copy. The CSV is for scoring; multi-thousand-character passage
    # cells are not something anyone should have to label inside a spreadsheet.
    with READABLE.open("w") as fh:
        fh.write("Step 14 -- groundedness labelling\n")
        fh.write("=" * 70 + "\n\n")
        fh.write("For each claim: does the passage below it actually support "
                 "the claim?\n")
        fh.write("Write y or n against the pair id in the CSV's `supported` "
                 "column.\n\n")
        fh.write("Judge only whether the passage SAYS it. Not whether the "
                 "claim is true, not\nwhether it answers the question, and "
                 "not whether it is the figure you would\nhave picked -- a "
                 "wrong-but-quoted number is still supported.\n\n")
        for p in pairs:
            fh.write("=" * 70 + "\n")
            fh.write(f"{p['pair_id']}   ({p['type']}, claim {p['claim_idx']} "
                     f"of {p['n_claims']})\n")
            fh.write(f"question: {p['question']}\n\n")
            fh.write(f"CLAIM:\n  {p['claim']}\n\n")
            fh.write(f"CITED PASSAGE(S) [{p['cited'] or 'none'}]:\n")
            for line in (p["passages"] or "(no citation given)").splitlines():
                fh.write(f"  {line}\n")
            fh.write("\n  supported?  y / n\n\n")

    print(f"\nwrote {OUT.relative_to(config.ROOT)}")
    print(f"wrote {READABLE.relative_to(config.ROOT)}  (the one to read)")


if __name__ == "__main__":
    main()
