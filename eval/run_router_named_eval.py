"""The two rungs nobody measured: company detection by exact string match.

Every router measurement so far -- Step 11's 45, its held-out 30, Step 20's 43
-- asserts at the top that no question names a company. That assertion is
correct and necessary: a question naming a filer is settled by
`route.named_companies` before any model is consulted, so including one would
measure the string match instead of the router under test.

The consequence went unnoticed for three steps. The string match handles the
first two of the router's four rungs, and it has never been scored. It was
described in three documents as "free and never wrong", on the strength of an
argument rather than a measurement:

    `"stewart" in question.lower()` is free and never wrong -- if the word
    Stewart is in the question, the question is about Stewart.

The argument is wrong, and a substring scan of the corpus says why. The alias
`stc` occurs inside `westcor` -- Westcor Land Title Insurance, a real title
insurer that is not one of the three filers. `faf` occurs inside `faff`. The
match has no word boundaries, so any question containing those strings inside
a longer word is routed to a company it never mentioned, silently, before any
model gets a say.

Three buckets:

  N  names exactly one filer   -> filtered to it
  M  names two or more, or says "all three" / "the companies" -> comparative
  X  contains an alias INSIDE another word, and names no filer at all
     -> must NOT be settled here; the question has to fall through to the
        model. An X failure is the expensive kind, because a wrongly filtered
        search returns passages from one company with no sign that a choice
        was made.

This whole file is offline. `named_companies` is pure string work, so there is
no API call, no cost, and no held-out question spent -- which is why this gap
was cheap to close and there was no excuse for leaving it open.
"""

import collections
import csv
from pathlib import Path

from src import config, route

QUESTIONS = config.ROOT / "eval" / "router_questions_named.csv"
OUT = config.ROOT / "results" / "router_eval_named.csv"

BUCKETS = {
    "N": "names one filer -- must filter to it",
    "M": "names several or all -- must fan out",
    "X": "alias hidden inside another word -- must NOT be settled here",
}


def expected(row) -> tuple[str, list[str]]:
    cos = [c for c in row["gold_companies"].split("|") if c]
    return row["gold_mode"], cos


def observed(question: str) -> tuple[str, list[str]]:
    """What the first two rungs do, before any model is reached.

    `route.plan` is called with no conversation history, so a question that
    names nobody reaches the word list rather than Jev -- and `unsettled` is
    recorded for anything the name matching declined to decide, whatever the
    later rungs then did with it.
    """
    cos, _ = route.named_companies(question)
    if not cos:
        return "unsettled", []
    return ("comparative" if len(cos) >= 2 else "filtered"), cos


def main() -> None:
    rows = list(csv.DictReader(QUESTIONS.open()))
    recs = []
    for r in rows:
        want_mode, want_cos = expected(r)
        got_mode, got_cos = observed(r["question"])
        ok = (got_mode == want_mode) and (got_cos == want_cos)
        recs.append({
            "qid": r["qid"], "bucket": r["bucket"], "question": r["question"],
            "gold_mode": want_mode, "gold_companies": "|".join(want_cos),
            "got_mode": got_mode, "got_companies": "|".join(got_cos),
            "outcome": "correct" if ok else "WRONG",
            "why": r["why"],
        })

    print(f"{len(recs)} questions, scored offline -- no model call, no cost.\n")
    print(f"{'bucket':<56}{'correct':>10}")
    print("-" * 66)
    for b, desc in BUCKETS.items():
        sub = [r for r in recs if r["bucket"] == b]
        n_ok = sum(r["outcome"] == "correct" for r in sub)
        print(f"{b} {desc:<54}{n_ok:>6}/{len(sub)}")
    print("-" * 66)
    n_ok = sum(r["outcome"] == "correct" for r in recs)
    print(f"{'overall':<56}{n_ok / len(recs) * 100:>9.1f}%")

    bad = [r for r in recs if r["outcome"] != "correct"]
    if bad:
        print(f"\n{len(bad)} failure(s) -- each one is a question routed to a "
              f"company it never named:")
        for r in bad:
            print(f"\n  {r['qid']} [{r['bucket']}]  {r['question']}")
            print(f"     expected  {r['gold_mode']:<12}"
                  f"{r['gold_companies'] or '(falls through to the model)'}")
            print(f"     got       {r['got_mode']:<12}{r['got_companies']}")
            print(f"     {r['why']}")
    else:
        print("\nno failures: the name matching decides every question here "
              "the way it should.")

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print(f"\nwrote {OUT.relative_to(config.ROOT)}")

    by_bucket = collections.Counter(
        r["bucket"] for r in recs if r["outcome"] != "correct")
    if by_bucket:
        print(f"failures by bucket: {dict(by_bucket)}")


if __name__ == "__main__":
    main()
