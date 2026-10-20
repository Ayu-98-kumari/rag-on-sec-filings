"""Step 13 — measure the numeric verifier against the answers already scored.

    python -m eval.run_verify

Makes NO API calls: it re-retrieves (free) and reads answers already recorded
in results/*.csv. Writes results/verify_<label>.csv.

The question being measured is whether the deterministic check earns its place:
does it flag the answers known to be wrong, and does it leave the answers known
to be correct alone? A guardrail that flags everything is useless, and one that
flags nothing is worse than useless because it looks like assurance.
"""

import argparse
import csv

from src import config, retrieve as retrieve_mod, route as route_mod, verify
from src.config import RESULTS, ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline",
                    help="results/<label>.csv to verify")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--retriever", default=None)
    ap.add_argument("--route", action="store_true")
    args = ap.parse_args()

    src = RESULTS / f"{args.label}.csv"
    if not src.exists():
        raise SystemExit(f"no results at {src}")
    rows = [r for r in csv.DictReader(src.open()) if r.get("answer")]
    if not rows:
        raise SystemExit(f"{src} has no answers (was it a --no-llm run?)")

    # Default to the configuration that produced the file, so the passages
    # checked are the passages the model actually saw.
    import json
    meta_f = RESULTS / f"{args.label}.meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    strategy = args.strategy or meta.get("strategy", "baseline")
    retriever = args.retriever or meta.get("retriever", "dense")
    do_route = args.route or bool(meta.get("route"))

    print(f"verifying {len(rows)} answers from {src.relative_to(ROOT)}")
    print(f"re-retrieving with strategy={strategy} retriever={retriever} "
          f"route={do_route}  (no API calls)\n")

    searcher = retrieve_mod.get_retriever(retriever, strategy)

    out = []
    print(f"{'qid':5s} {'verdict':17s} {'figs':>4s} {'gnd':>4s} {'unsup':>6s}  "
          f"unsupported figures")
    print("-" * 78)
    for r in rows:
        hits, plan = route_mod.retrieve(searcher, r["question"],
                                        k=config.K, route=do_route)
        passages = [h["text"] for h in hits]
        rep = verify.verify(r["answer"], passages, plan.get("companies"))
        out.append({
            "qid": r["qid"], "type": r["type"], "split": r["split"],
            "answer_verdict": r["verdict"],
            "n_figures": rep["n_figures"], "n_grounded": rep["n_grounded"],
            "n_unsupported": rep["n_unsupported"],
            "numeric_verdict": rep["verdict"],
            "unsupported": "; ".join(rep["unsupported"]),
        })
        print(f"{r['qid']:5s} {r['verdict'][:17]:17s} {rep['n_figures']:4d} "
              f"{rep['n_grounded']:4d} {rep['n_unsupported']:6d}  "
              f"{'; '.join(rep['unsupported'])[:34]}")

    dest = RESULTS / f"verify_{args.label}.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    # --- does the check earn its place? ---
    # "Wrong" here means the answer verdict says the figure was wrong. False
    # abstentions carry no figure at all, so they are excluded from both sides:
    # the verifier cannot flag a number that was never stated.
    numeric = [r for r in out if r["n_figures"] > 0]
    wrong = [r for r in numeric if r["answer_verdict"] in
             ("wrong", "correct_unscaled", "partial")]
    right = [r for r in numeric if r["answer_verdict"] == "correct"]
    flagged = lambda g: [r for r in g if r["numeric_verdict"] == "flagged"]

    print("\n" + "=" * 78)
    print("DOES THE CHECK EARN ITS PLACE?")
    print("=" * 78)
    print(f"  answers containing at least one figure : {len(numeric)}")
    print(f"  of those, flagged by the verifier      : {len(flagged(numeric))}")
    print()
    print(f"  known-WRONG answers with figures       : {len(wrong)}")
    print(f"    flagged (caught)                     : {len(flagged(wrong))}"
          f"{'  <- recall ' + f'{100*len(flagged(wrong))/len(wrong):.0f}%' if wrong else ''}")
    print(f"  known-CORRECT answers with figures     : {len(right)}")
    print(f"    flagged (false alarm)                : {len(flagged(right))}"
          f"{'  <- ' + f'{100*len(flagged(right))/len(right):.0f}% false alarm rate' if right else ''}")
    tp, fp = len(flagged(wrong)), len(flagged(right))
    if tp + fp:
        print(f"\n  precision of a flag                    : "
              f"{100*tp/(tp+fp):.0f}%  ({tp} of {tp+fp} flags were on wrong answers)")
    tot_figs = sum(r["n_figures"] for r in numeric)
    tot_uns = sum(r["n_unsupported"] for r in numeric)
    print(f"\n  figures checked                        : {tot_figs}")
    print(f"  unsupported figure rate                : "
          f"{100*tot_uns/max(tot_figs,1):.1f}%  ({tot_uns} of {tot_figs})")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
