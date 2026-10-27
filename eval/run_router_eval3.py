"""Step 20 — can the router tell "I need one company" from "which company?"

Step 11 asked one question: does answering this need every filer searched, or
will one search do? This asks a prior one. Some questions have no correct
answer at all, because they ask for one company's figure and do not say which
company. Thirteen of the thirty questions a third-party writer produced for
Step 11 had that shape, and all thirteen were cut from that set -- not because
they were bad questions, but because no label could score them. They are
bucket E here, copied without rewording.

Four routers decide the same 43 questions:

  rules       src/route.py            -- COMPARATIVE_PHRASES, a word list
  jev         src/jev_router.py       -- the FROZEN two-label router
  jev-ask     src/jev_router_ask.py   -- three labels, adds `unclear`
  all         below                   -- always comparative, brute force

Three of those four cannot score a single point on bucket E, because asking
back is not in their vocabulary. That is the point of the bucket, not a flaw
in the comparison: it shows that the Step 11 router is not making a hard call
on these questions, it is answering a question that has no answer. Buckets
A through D are where the four actually compete, and `jev` must reproduce its
held-out 30/30 there or something has drifted -- checked below.

The error directions, which cost different things and are therefore never
added together:

  MISS     gold comparative, routed single
           one search lands in one filer's neighbourhood and the answer reads
           as though all three were considered. The Step 7/8 failure, 25 per
           cent retrieval, the worst thing the system did.
  SILENT   gold unclear, routed single
           the same failure with a worse story: the question identified no
           company, and the system produced a confident figure anyway. The
           user has no way to see that a choice was made on their behalf.
  hedged   gold unclear, routed comparative
           did not ask, but returned all three. Wrong path, recoverable
           outcome -- the user can see three companies and pick. Counted apart
           from SILENT because treating them as one error hides that one of
           them is honest about its uncertainty and the other is not.
  waste    gold single, routed comparative
           three searches where one would do. Slower, same answer.
  OVERASK  gold single or comparative, routed clarify
           a clear question bounced back at the user, who then answers
           something they already answered. Cheap in correctness, expensive in
           trust, and the failure mode that makes people abandon a tool. This
           is the error Step 20 adds to the system, so it is the one that
           decides whether Step 20 is worth shipping.

Caveat kept in the open, as in Step 11: buckets C and D were written to be
adversarial to a word list. Bucket E was not written to be adversarial to
anything -- it is what the writer produced unprompted, which is why it is the
most informative bucket here and the one with the least of my fingerprints on
it.
"""

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

from src import config, jev_router, jev_router_ask, route

DEFAULT_QUESTIONS = config.ROOT / "eval" / "router_questions_ask.csv"

USD_PER_MTOK_INPUT = 0.042
PRICE_FROM_PLAN_NOT_VERIFIED = True

BUCKETS = {
    "A": "plain comparison (control)",
    "B": "plain single-fact (control)",
    "C": "comparison, no trigger word",
    "D": "single-fact with a superlative",
    "E": "names no company, needs one -- only jev-ask can score",
}

# The plan's mode for "ask the user" versus the gold label for it. Two names
# for one thing, mapped in one place rather than compared loosely at each use.
MODE_TO_LABEL = {"clarify": "unclear"}

# Which buckets every router can in principle get right. Bucket E is excluded
# from the headline comparison because three of the four routers have no label
# for it, and an accuracy figure that includes it would read as those three
# being wrong when they are in fact mute.
COMPETITIVE = ("A", "B", "C", "D")


def always_comparative(question, prev_companies=None):
    """The brute-force baseline: fan every question out to every filer."""
    return {"mode": "comparative", "companies": list(config.TICKERS),
            "why": "baseline: every question fanned out, no decision made"}


def error_kind(gold: str, got: str) -> str:
    """Name the error direction. Six states, because they cost six things."""
    if gold == got:
        return "correct"
    if got == "unclear":
        return "overask"            # clear question bounced back
    if gold == "unclear":
        return "silent" if got == "single" else "hedged"
    return "miss" if gold == "comparative" else "waste"


def score(rows, planner, name, **kw):
    """Run one router over every question. Returns per-question records."""
    out = []
    for r in rows:
        p = planner(r["question"], None, **kw) if kw else planner(r["question"], None)
        got = MODE_TO_LABEL.get(p["mode"], p["mode"])
        jev = p.get("jev") or {}
        out.append({
            "qid": r["qid"], "bucket": r["bucket"], "router": name,
            "question": r["question"], "gold": r["gold_mode"], "got": got,
            "outcome": error_kind(r["gold_mode"], got),
            "confidence": jev.get("confidence"),
            "input_tokens": jev.get("input_tokens"),
            "latency_ms": jev.get("latency_ms"),
            "cached": jev.get("cached"),
            "jev_error": jev.get("error"),
            "why": p["why"],
        })
    return out


def accuracy(recs) -> float:
    return sum(r["outcome"] == "correct" for r in recs) / len(recs) if recs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true",
                    help="force live Jev calls, for timing a cold run")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = ap.parse_args()

    questions = args.questions.resolve()
    out = config.ROOT / "results" / "router_eval_ask.csv"
    rows = list(csv.DictReader(questions.open()))

    # Every question must reach the decision under test, or the comparison is
    # measuring the exact string matches instead of the routers.
    settled = [r["qid"] for r in rows if route.named_companies(r["question"])[0]]
    if settled:
        raise SystemExit(f"these questions never reach the decision: {settled}")

    runs = {
        "rules": score(rows, route.plan, "rules"),
        "jev": score(rows, jev_router.plan, "jev",
                     use_cache=not args.no_cache),
        "jev-ask": score(rows, jev_router_ask.plan, "jev-ask",
                         use_cache=not args.no_cache),
        "all-comparative": score(rows, always_comparative, "all-comparative"),
    }

    for name in ("jev", "jev-ask"):
        failed = [r for r in runs[name] if r["jev_error"]]
        if failed:
            print(f"!! {len(failed)}/{len(rows)} {name} calls failed and fell "
                  f"back to the word list. That column is NOT a measurement "
                  f"of {name}.")
            print(f"   first error: {failed[0]['jev_error']}\n")

    n_e = sum(r["bucket"] == "E" for r in rows)
    print(f"{len(rows)} questions, none naming a company. {n_e} of them "
          f"(bucket E) name no company and need one.\n")

    hdr = f"{'bucket':<50}" + "".join(f"{n:>17}" for n in runs)
    print(hdr)
    print("-" * len(hdr))
    for b, desc in BUCKETS.items():
        line = f"{b} {desc:<48}"
        for recs in runs.values():
            sub = [r for r in recs if r["bucket"] == b]
            line += f"{sum(r['outcome'] == 'correct' for r in sub):>10}/{len(sub):<6}"
        print(line)
    print("-" * len(hdr))

    line = f"{'A-D only (all four can score these)':<50}"
    for recs in runs.values():
        sub = [r for r in recs if r["bucket"] in COMPETITIVE]
        line += f"{accuracy(sub) * 100:>15.1f}% "
    print(line)
    line = f"{'all 43 (E unscoreable for three of them)':<50}"
    for recs in runs.values():
        line += f"{accuracy(recs) * 100:>15.1f}% "
    print(line)

    # Regression guard. `jev` here is the frozen router replaying the same 30
    # cached decisions it scored 30/30 on. If that number moved, the prompt,
    # the model or the cache changed, and nothing below is comparable to the
    # Step 11 result.
    ad = [r for r in runs["jev"] if r["bucket"] in COMPETITIVE]
    print(f"\nfrozen Step 11 router on its own 30: "
          f"{sum(r['outcome'] == 'correct' for r in ad)}/{len(ad)}"
          f"{'  -- matches the held-out run' if accuracy(ad) == 1.0 else '  !! DRIFTED from 30/30'}")

    print("\nthe same results taken apart by error direction:")
    cols = ["correct", "miss", "silent", "hedged", "waste", "overask"]
    print("  " + f"{'router':<18}" + "".join(f"{c:>9}" for c in cols))
    for name, recs in runs.items():
        c = collections.Counter(r["outcome"] for r in recs)
        print("  " + f"{name:<18}" + "".join(f"{c[k]:>9}" for k in cols))
    print("  MISS/SILENT answer as though a choice had not been made for the "
          "user; OVERASK is friction")

    print("\nbucket E, question by question -- what each router did with a "
          "question that names nobody:")
    by = {n: {r["qid"]: r for r in recs} for n, recs in runs.items()}
    for r in rows:
        if r["bucket"] != "E":
            continue
        got = {n: by[n][r["qid"]]["got"] for n in runs}
        print(f"  {r['qid']}  rules={got['rules']:<11}"
              f"jev={got['jev']:<11}jev-ask={got['jev-ask']:<11}"
              f"{r['question'][:46]}")

    ask = runs["jev-ask"]
    live = [r for r in ask if r["latency_ms"] and not r["jev_error"]]
    toks = [r["input_tokens"] for r in live if r["input_tokens"]]
    if toks:
        n_live = sum(1 for r in ask if r["cached"] is False)
        print(f"\ncost of the jev-ask column: {len(toks)} decisions carry "
              f"usage, {n_live} were live calls this run")
        print(f"  input tokens   median {statistics.median(toks):.0f}  "
              f"min {min(toks)}  max {max(toks)}")
        print(f"  latency ms     median "
              f"{statistics.median(r['latency_ms'] for r in live):.0f}"
              f"  (a cache hit is ~0ms and would flatter this)")
        per_k = statistics.median(toks) * 1000 / 1e6 * USD_PER_MTOK_INPUT
        flag = "  (rate from the plan, not re-verified)" if PRICE_FROM_PLAN_NOT_VERIFIED else ""
        print(f"  ${per_k:.4f} per 1000 decisions at "
              f"${USD_PER_MTOK_INPUT}/Mtok{flag}")
        jt = [r["input_tokens"] for r in runs["jev"] if r["input_tokens"]]
        if jt:
            print(f"  the third label costs "
                  f"{statistics.median(toks) - statistics.median(jt):+.0f} "
                  f"input tokens per decision against the frozen router")

    conf = [r for r in ask if r["confidence"] is not None]
    if conf:
        print("\ncalibration -- does the confidence mean anything yet?")
        print(f"  {'confidence':<16}{'n':>4}{'correct':>9}   "
              f"a threshold is only worth having if accuracy rises with it")
        bins = [(0.0, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 0.999), (0.999, 1.01)]
        for lo, hi in bins:
            sub = [r for r in conf if lo <= r["confidence"] < hi]
            if sub:
                label = "exactly 1.000" if lo == 0.999 else f"{lo:.2f} - {hi:.2f}"
                print(f"  {label:<16}{len(sub):>4}{accuracy(sub) * 100:>8.0f}%")
        wrong = [r for r in conf if r["outcome"] != "correct"]
        if not wrong:
            print("  nothing failed, so this table still cannot show the "
                  "confidence carries information.")
        else:
            print(f"  {len(wrong)} wrong decisions, median confidence "
                  f"{statistics.median(r['confidence'] for r in wrong):.2f} "
                  f"against {statistics.median(r['confidence'] for r in conf if r['outcome'] == 'correct'):.2f} "
                  f"for the correct ones -- the first evidence either way.")

    print("\nwhere the two Jev routers disagree:")
    shown = 0
    for r in rows:
        a, b = by["jev"][r["qid"]], by["jev-ask"][r["qid"]]
        if a["got"] != b["got"]:
            shown += 1
            win = "jev-ask" if b["outcome"] == "correct" else (
                "jev" if a["outcome"] == "correct" else "neither")
            print(f"  {r['qid']}  jev={a['got']:<11}jev-ask={b['got']:<11}"
                  f"gold={r['gold_mode']:<11}{win} right")
    print(f"  {shown} of {len(rows)} questions routed differently")

    out.parent.mkdir(exist_ok=True)
    flat = [r for recs in runs.values() for r in recs]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)

    meta = {
        "questions": str(questions.relative_to(config.ROOT)),
        "n_questions": len(rows),
        "buckets": dict(sorted(collections.Counter(
            r["bucket"] for r in rows).items())),
        "gold": dict(sorted(collections.Counter(
            r["gold_mode"] for r in rows).items())),
        "routers": list(runs),
        "accuracy_all": {n: round(accuracy(r), 4) for n, r in runs.items()},
        "accuracy_ad_only": {
            n: round(accuracy([x for x in r if x["bucket"] in COMPETITIVE]), 4)
            for n, r in runs.items()},
        "errors": {n: dict(collections.Counter(x["outcome"] for x in r))
                   for n, r in runs.items()},
        "jev_model": jev_router_ask.MODEL,
        "confidence_floor": jev_router_ask.CONFIDENCE_FLOOR,
        "jev_failures": {n: sum(1 for x in runs[n] if x["jev_error"])
                         for n in ("jev", "jev-ask")},
        "usd_per_mtok_input": USD_PER_MTOK_INPUT,
        "price_verified": not PRICE_FROM_PLAN_NOT_VERIFIED,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(config.ROOT)} and "
          f"{out.with_suffix('.meta.json').name}")


if __name__ == "__main__":
    main()
