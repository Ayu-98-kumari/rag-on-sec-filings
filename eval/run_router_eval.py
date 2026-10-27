"""Step 11 — which router decides best, measured instead of argued.

Three routers decide the same 45 questions, all of which name no company and
therefore all of which reach the one decision Step 11 is about: does answering
this need every filer searched separately, or will one search do?

  rules   src/route.py        -- COMPARATIVE_PHRASES, a hand-written word list
  jev     src/jev_router.py   -- a typed Choice with a confidence threshold
  all     below               -- always comparative, the brute-force baseline

The brute-force baseline is here because it is the honest thing to beat. It
cannot ever miss a filer, so if the only cost of routing badly were a wasted
search, nobody would need a router at all -- you would fan every question out
to three searches and stop reading. What makes that wrong is latency and the
answer model's context budget, not correctness, so the comparison has to show
the two error directions apart rather than add them up.

The two directions are not equally bad, and this is the finding the table is
built to show:

  a MISS  (gold comparative, routed single)      -- one search lands in one
          filer's neighbourhood and the answer reads as though all three were
          considered. This is the Step 7/8 failure: 25 per cent retrieval on
          comparative questions, the worst category in the system.
  a WASTE (gold single, routed comparative)      -- three searches where one
          would do. Slower, more context spent, same answer.

A router with four misses is worse than a router with ten wastes, and a single
accuracy percentage hides that. So accuracy is reported, and then taken apart.

Caveat kept in the open: buckets C and D were written to be adversarial to a
word list, so `rules` scoring badly overall is partly a property of the
question set. Buckets A and B are the control -- a router that fails those
is worse than the word list no matter what it does on C and D.
"""

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

from src import config, jev_router, route

# The development set: written and labelled by the same party that built the
# router, so a perfect score on it is not evidence the router generalises.
# --questions points this at a held-out set written by someone else, which is
# the number worth quoting. The output file name follows the input's, so a
# held-out run cannot overwrite the development run's results.
DEFAULT_QUESTIONS = config.ROOT / "eval" / "router_questions.csv"


def out_for(questions: Path) -> Path:
    """results/router_eval.csv for the development set, _heldout for the other.

    Derived rather than passed so the two runs can never land in the same file
    by forgetting a flag -- which would leave two different measurements under
    one name and no way to tell afterwards which one was which.
    """
    stem = questions.stem.replace("router_questions", "router_eval")
    return config.ROOT / "results" / f"{stem}.csv"

# $42 per billion input tokens, the rate stated in the Step 11 plan. The SDK
# exposes no pricing, so this is not measured here and has not been re-checked
# against published rates -- tokens are the measurement, dollars are arithmetic
# on top of one number. The plan's own cost estimate was wrong by a factor of
# six, and it was the token count that was wrong, not this rate: it guessed an
# 80-token payload where the measured payload is 521, because the label
# descriptions in jev_router.CRITERIA are prose rather than keywords.
USD_PER_MTOK_INPUT = 0.042
PRICE_FROM_PLAN_NOT_VERIFIED = True

BUCKETS = {
    "A": "plain comparison (control)",
    "B": "plain single-fact (control)",
    "C": "comparison, no trigger word -- word list under-fires",
    "D": "single-fact with a superlative -- word list over-fires",
}


def always_comparative(question, prev_companies=None):
    """The brute-force baseline: fan every question out to every filer."""
    return {"mode": "comparative", "companies": list(config.TICKERS),
            "why": "baseline: every question fanned out, no decision made"}


def error_kind(gold: str, got: str) -> str:
    """Name the error direction, because the two cost different things."""
    if gold == got:
        return "correct"
    return "miss" if gold == "comparative" else "waste"


def score(rows, planner, name, **kw):
    """Run one router over every question. Returns per-question records."""
    out = []
    for r in rows:
        p = planner(r["question"], None, **kw) if kw else planner(r["question"], None)
        got = p["mode"]
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
    return sum(r["outcome"] == "correct" for r in recs) / len(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true",
                    help="force live Jev calls, for timing a cold run")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS,
                    help="question set to score (default: the development set)")
    args = ap.parse_args()

    # Resolved, because the metadata records the path relative to ROOT and a
    # relative argument typed at the shell is not under ROOT until it is.
    questions = args.questions.resolve()
    out = out_for(questions)
    rows = list(csv.DictReader(questions.open()))

    # Every question must actually reach the decision, or the comparison is
    # measuring the deterministic rules instead of the routers. Checked here
    # rather than trusted, because it is the assumption the whole step rests on.
    settled = [r["qid"] for r in rows if route.named_companies(r["question"])[0]]
    if settled:
        raise SystemExit(f"these questions never reach the decision: {settled}")

    runs = {
        "rules": score(rows, route.plan, "rules"),
        "jev": score(rows, jev_router.plan, "jev", use_cache=not args.no_cache),
        "all-comparative": score(rows, always_comparative, "all-comparative"),
    }

    failed = [r for r in runs["jev"] if r["jev_error"]]
    if failed:
        print(f"!! {len(failed)}/{len(rows)} Jev calls failed and fell back to "
              f"the word list. The jev column below is NOT a Jev measurement.")
        print(f"   first error: {failed[0]['jev_error']}\n")

    print(f"{len(rows)} questions, none naming a company, all reaching the "
          f"decision.\n")

    hdr = f"{'bucket':<46}" + "".join(f"{n:>17}" for n in runs)
    print(hdr)
    print("-" * len(hdr))
    for b, desc in BUCKETS.items():
        line = f"{b} {desc:<44}"
        for recs in runs.values():
            sub = [r for r in recs if r["bucket"] == b]
            line += f"{sum(r['outcome'] == 'correct' for r in sub):>10}/{len(sub):<6}"
        print(line)
    print("-" * len(hdr))
    line = f"{'overall':<46}"
    for recs in runs.values():
        line += f"{accuracy(recs) * 100:>15.1f}% "
    print(line)

    print("\nthe same results taken apart by error direction:")
    print(f"  {'router':<18}{'correct':>9}{'MISS':>8}{'waste':>8}   "
          f"a miss answers from one filer as if all three had been considered")
    for name, recs in runs.items():
        c = collections.Counter(r["outcome"] for r in recs)
        print(f"  {name:<18}{c['correct']:>9}{c['miss']:>8}{c['waste']:>8}")

    live = [r for r in runs["jev"] if r["latency_ms"] and not r["jev_error"]]
    toks = [r["input_tokens"] for r in live if r["input_tokens"]]
    if toks:
        n_live = sum(1 for r in runs["jev"] if r["cached"] is False)
        print(f"\ncost of the Jev column: {len(toks)} decisions carry usage, "
              f"{n_live} were live calls this run")
        print(f"  input tokens   median {statistics.median(toks):.0f}  "
              f"min {min(toks)}  max {max(toks)}")
        print(f"  latency ms     median {statistics.median(r['latency_ms'] for r in live):.0f}"
              f"  (cache hits excluded; a hit is ~0ms and would flatter this)")
        per_k = statistics.median(toks) * 1000 / 1e6 * USD_PER_MTOK_INPUT
        flag = "  (rate from the plan, not re-verified)" if PRICE_FROM_PLAN_NOT_VERIFIED else ""
        print(f"  ${per_k:.4f} per 1000 decisions at "
              f"${USD_PER_MTOK_INPUT}/Mtok{flag}")

    conf = [r for r in runs["jev"] if r["confidence"] is not None]
    if conf:
        print("\ncalibration -- does the confidence mean anything?")
        print(f"  {'confidence':<16}{'n':>4}{'correct':>9}   "
              f"a threshold is only worth having if accuracy rises with it")
        bins = [(0.0, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 0.999), (0.999, 1.01)]
        for lo, hi in bins:
            sub = [r for r in conf if lo <= r["confidence"] < hi]
            if sub:
                label = "exactly 1.000" if lo == 0.999 else f"{lo:.2f} - {hi:.2f}"
                print(f"  {label:<16}{len(sub):>4}{accuracy(sub) * 100:>8.0f}%")
        sat = sum(r["confidence"] >= 0.999 for r in conf)
        if sat > len(conf) * 0.8:
            print(f"  NOTE {sat}/{len(conf)} decisions came back at 1.000. The "
                  f"{jev_router.CONFIDENCE_FLOOR} floor never fires on this set,")
            print("       so it is an inert knob here, not a working safeguard.")

    print("\ndisagreements between the word list and Jev:")
    by_qid = {r["qid"]: r for r in runs["rules"]}
    shown = 0
    for jr in runs["jev"]:
        rr = by_qid[jr["qid"]]
        if rr["got"] != jr["got"]:
            shown += 1
            win = "jev" if jr["outcome"] == "correct" else "rules"
            print(f"  {jr['qid']}  rules={rr['got']:<12}jev={jr['got']:<12}"
                  f"gold={jr['gold']:<12}{win} right")
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
        "routers": list(runs),
        "accuracy": {n: round(accuracy(r), 4) for n, r in runs.items()},
        "errors": {n: dict(collections.Counter(x["outcome"] for x in r))
                   for n, r in runs.items()},
        "jev_model": jev_router.MODEL,
        "confidence_floor": jev_router.CONFIDENCE_FLOOR,
        "jev_failures": len(failed),
        "usd_per_mtok_input": USD_PER_MTOK_INPUT,
        "price_verified": not PRICE_FROM_PLAN_NOT_VERIFIED,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(config.ROOT)} and "
          f"{out.with_suffix('.meta.json').name}")


if __name__ == "__main__":
    main()
