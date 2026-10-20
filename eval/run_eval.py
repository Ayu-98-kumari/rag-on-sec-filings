"""Step 7 — run the test set through the system and score it.

    python -m eval.run_eval                     # full run: retrieval + answers
    python -m eval.run_eval --no-llm            # retrieval only, zero API calls
    python -m eval.run_eval --strategy improved --label improved

Writes results/<label>.csv (one row per question) and prints the summary.

Retrieval and generation are scored SEPARATELY and on purpose. A wrong answer
has two very different causes -- the right passage never reached the model, or
it reached the model and was misread -- and they need different fixes. Mixing
them into one "accuracy" number would hide which one is happening.

Retrieval scoring costs nothing: it compares block ids. That is what makes
Step 8 affordable. Re-chunk, re-index, re-score recall in about a minute with
no API calls, and only spend tokens regenerating answers once retrieval has
actually improved.
"""

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone

from src import ask as ask_mod
from src import config, embed, llm, retrieve as retrieve_mod, route
from src.config import CHROMA, RESULTS, ROOT
from src.store import get_store

QUESTIONS = ROOT / "eval" / "questions.csv"
RECALL_AT = (1, 5, 10)
RETRIEVE_K = 40                      # retrieve once, score every cut-off from prefixes

# The context budget, in tokens, used for the strategy-vs-strategy comparison.
# Fixing k instead would not be a controlled experiment: the baseline averages
# 383 tokens per chunk and the structure-aware strategy 230, so five chunks
# each means handing one strategy 1,915 tokens of context and the other 1,150.
# The quantity the model is actually limited by is tokens in the prompt, so
# that is what is held constant.
#
# Raised from 1,915 (baseline k=5) to 3,830 (baseline k=10) after a measured
# finding. Recovering the row labels the filers omitted moved the three wrong
# lookups from outside the top ten into it -- recall@10 went to 1 on all three
# -- but 1,915 tokens admits only five or six passages, and the passage
# carrying the answer ranked ninth. The evidence was being retrieved and then
# discarded before the model saw it, so the answers could not improve.
#
# This is still a deliberately small budget, not the maximum. qwen's context
# is 131,072 tokens; feeding it that would stop measuring retrieval at all,
# since recall only means something if what is retrieved is what gets used.
# Long contexts also measurably degrade fact-finding, and a production answer
# at 3,830 tokens is affordable where one at 131,072 is not.
CTX_TOKENS = 3830

# Phrases that indicate the model declined. Kept as a fallback only -- see
# abstained() for why searching for these anywhere in an answer is wrong.
ABSTAIN_RE = re.compile(
    r"not in the provided filings"
    r"|do(?: not|n't) (?:have|contain|include|know)"
    r"|no(?:t)? (?:available|disclosed|provided|stated|mentioned|found)"
    r"|cannot (?:be )?(?:answer|determin|found)"
    r"|insufficient information", re.I)

# Rule 3 of the system prompt in src/ask.py specifies the exact refusal
# sentence: "If the passages do not contain the answer, reply exactly: 'Not in
# the provided filings.'" So that sentence IS the refusal token, and it is what
# we detect.
CANONICAL_REFUSAL = "not in the provided filings"

# An answer short enough that a phrase of absence cannot be a caveat attached
# to a real answer -- there is no room for one.
SHORT_ANSWER = 120


def abstained(ans: str) -> bool:
    """Did the model decline, as opposed to answering with a caveat?

    The first version of this searched ABSTAIN_RE against the whole answer,
    and that was a real bug in this harness rather than a detail. A good
    comparative answer names the companies it DID find and says plainly which
    one it could not -- "The provided passages do not contain the full-year
    2025 figure for Stewart" -- and the phrase "do not contain" inside that
    sentence made the whole answer score as a refusal. Worse, the verdict code
    tests abstention BEFORE the type-specific grader, so the comparative
    grader never ran at all and three correct-but-partial answers across the
    runs were recorded as false abstentions. The harness was penalising the
    model for being honest about what it was missing, which is exactly the
    behaviour the prompt asks for.

    The fix uses the prompt's own contract. Rule 3 names the sentence to reply
    with, so a refusal is that sentence standing on its own -- at the start, or
    as a closing verdict after the model has explained why it cannot answer,
    which is how it usually arrives. A phrase of absence buried mid-answer is
    a caveat, and the answer gets graded on its merits.

    The short-answer fallback catches a refusal worded differently, where
    there is no room for the phrase to be a caveat. Anything longer that never
    produces the canonical sentence is graded as an answer -- the conservative
    direction, because it risks marking a refusal wrong rather than crediting
    a non-answer as a correct abstention.
    """
    text = ans.strip()
    if not text:
        return True
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if sent.strip().strip(".!?'\" ").lower() == CANONICAL_REFUSAL:
            return True
    return len(text) <= SHORT_ANSWER and bool(ABSTAIN_RE.search(text))

SCALE_WORDS = {"thousand": 1e3, "thousands": 1e3, "million": 1e6,
               "millions": 1e6, "billion": 1e9, "billions": 1e9}

# A number, optionally followed by a scale word within a few characters.
NUM_RE = re.compile(
    r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
    r"(thousands?|millions?|billions?)?", re.I)

JUDGE_SYSTEM = (
    "You grade answers about SEC filings against a reference answer.\n"
    "Reply with exactly one word on the first line: CORRECT, PARTIAL or WRONG.\n"
    "CORRECT: the answer conveys the substance of the reference.\n"
    "PARTIAL: some of it, or correct but materially incomplete.\n"
    "WRONG: contradicts the reference, or does not answer.\n"
    "Extra detail beyond the reference is not an error. Different wording is "
    "not an error. Then give one short sentence of reason."
)


# --------------------------------------------------------------------------
# scoring: numeric
# --------------------------------------------------------------------------

def looks_like_year(raw: str, word: str) -> bool:
    """A bare four-digit integer in calendar range is a year, not a figure.

    Every one of these questions names a fiscal year, so the model's answer
    always contains one. Left in the candidate list it is mostly harmless, but
    the magnitude-shift check multiplies candidates by a thousand, a million
    and a billion -- and 2025 x 1e6 is $2.025 billion, which is inside the
    revenue range of two of the three companies. That would have scored a wrong
    answer correct on the strength of a date.
    """
    if word or "." in raw or "," in raw:
        return False
    return len(raw) == 4 and 1900 <= int(raw) <= 2100


def numbers_in(text: str) -> list[tuple[float, bool]]:
    """Every figure in the answer as (dollars, scale_was_explicit)."""
    out = []
    for raw, word in NUM_RE.findall(text):
        if looks_like_year(raw, word):
            continue
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if word:
            out.append((v * SCALE_WORDS[word.lower()], True))
        else:
            out.append((v, False))
    return out


def score_numeric(answer: str, expected: float, tol: float) -> tuple[str, str]:
    """Compare a figure without asking a model to do arithmetic.

    Returns (verdict, note). Three outcomes rather than two, because the
    interesting failure here is not a wrong number but a right number at the
    wrong magnitude: the filings report in thousands and millions, so a model
    reading '2,921,636' off a table and repeating it without the scale has
    retrieved correctly and answered misleadingly. That is worth seeing as its
    own category instead of being scored simply correct or simply wrong.
    """
    found = numbers_in(answer)
    if not found:
        return "no_figure", ""
    for v, explicit in found:
        if expected and abs(v - expected) <= tol * abs(expected):
            return "correct", "explicit scale" if explicit else ""
    # right digits, wrong magnitude?
    for v, _ in found:
        for mult in (1e3, 1e6, 1e9):
            if expected and abs(v * mult - expected) <= tol * abs(expected):
                return "correct_unscaled", f"stated {v:,.0f}, meant x{mult:,.0f}"
    return "wrong", f"figures seen: {', '.join(f'{v:,.0f}' for v, _ in found[:4])}"


def score_comparative(answer: str, expected: str, tol: float) -> tuple[str, str]:
    """Every company named must appear, and its figure must be right."""
    pairs = re.findall(r"([A-Za-z&.' ]+?)\s+([\d,]{7,})", expected)
    if not pairs:
        return "unscorable", ""
    missing_name, wrong_num = [], []
    found = [v for v, _ in numbers_in(answer)]
    for name, num in pairs:
        key = name.strip().split()[0]            # "First", "Fidelity", "Stewart"
        if key.lower() not in answer.lower():
            missing_name.append(key)
        want = float(num.replace(",", ""))
        if not any(abs(v - want) <= tol * want or
                   any(abs(v * m - want) <= tol * want for m in (1e3, 1e6, 1e9))
                   for v in found):
            wrong_num.append(key)
    if not missing_name and not wrong_num:
        return "correct", f"{len(pairs)} companies matched"
    bits = []
    if missing_name:
        bits.append("not named: " + ",".join(missing_name))
    if wrong_num:
        bits.append("figure missing/wrong: " + ",".join(wrong_num))
    return ("partial" if len(missing_name) + len(wrong_num) < len(pairs) else "wrong"), "; ".join(bits)


# --------------------------------------------------------------------------
# scoring: prose, by judge
# --------------------------------------------------------------------------

def judge(question: str, reference: str, answer: str,
          use_cache: bool = True) -> tuple[str, str]:
    prompt = (f"QUESTION: {question}\n\nREFERENCE ANSWER:\n{reference}\n\n"
              f"ANSWER TO GRADE:\n{answer}\n\nVerdict:")
    out = llm.complete(prompt, system=JUDGE_SYSTEM, max_tokens=120,
                       use_cache=use_cache)
    text = out["text"].strip()
    m = re.search(r"\b(CORRECT|PARTIAL|WRONG)\b", text, re.I)
    verdict = m.group(1).lower() if m else "unparsed"
    reason = " ".join(text.split("\n")[1:])[:160] or text[:160]
    return verdict, reason


# --------------------------------------------------------------------------

def company_coverage(gold: set[str], got: set[str], named: list[str]) -> float:
    """Share of the companies a question names that are actually represented.

    recall@k counts a question as a hit if ANY gold block is retrieved, which
    is the right measure for a single-answer question and the wrong one for a
    comparison. "Which had higher revenue, First American or Stewart" is not
    answerable from Stewart's filing alone, but it scores a full hit under
    recall as soon as one Stewart passage appears -- so recall is blind to
    exactly the failure Step 10 exists to fix. This asks the question routing
    actually changes: of the companies named, how many did we retrieve evidence
    for?
    """
    if not named or not gold:
        return 1.0
    hit = 0
    for tk in named:
        g = {b for b in gold if b.startswith(tk + "_")}
        if g and (g & got):
            hit += 1
    return hit / len(named)


def retrieved_blocks(hits: list[dict]) -> list[str]:
    """Union of source block ids across hits, in rank order."""
    seen, out = set(), []
    for h in hits:
        for bid in h["metadata"].get("source_block_ids") or []:
            if bid not in seen:
                seen.add(bid); out.append(bid)
    return out


# Columns the summary does arithmetic on. A row read back from a CSV is all
# strings, so a resumed row would make sum() concatenate or raise in the
# summary rather than anywhere near the cause.
INT_COLS = ([f"recall@{k}" for k in RECALL_AT]
            + ["recall_ctx", "ctx_chunks", "ctx_tokens", "abstained",
               "cached", "completion_tokens"])


def coerce_row(row: dict) -> dict:
    """Restore the numeric types that a CSV round-trip flattens to strings."""
    r = dict(row)
    for c in INT_COLS:
        if r.get(c) not in (None, ""):
            r[c] = int(float(r[c]))
    if r.get("top1_score") not in (None, ""):
        r["top1_score"] = float(r["top1_score"])
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=config.DEFAULT_STRATEGY)
    ap.add_argument("--label", default=None, help="results/<label>.csv")
    ap.add_argument("--no-llm", action="store_true",
                    help="score retrieval only; makes zero API calls")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip questions already in results/<label>.partial.csv "
                         "and append the rest; for picking a run back up after "
                         "it died on a quota limit")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--retriever", default="dense",
                    choices=["dense", "bm25", "hybrid"],
                    help="Step 9: dense embeddings, BM25 keywords, or both "
                         "fused by reciprocal rank")
    ap.add_argument("--route", action="store_true",
                    help="Step 10: decompose multi-company questions into one "
                         "metadata-filtered retrieval per company")
    args = ap.parse_args()
    label = args.label or args.strategy

    rows = list(csv.DictReader(QUESTIONS.open()))
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    print(f"{len(rows)} questions from {QUESTIONS.relative_to(ROOT)}"
          f" (split={args.split}), strategy={args.strategy}")
    if args.no_llm:
        print("--no-llm: retrieval only, no API calls\n")

    searcher = retrieve_mod.get_retriever(args.retriever, args.strategy)
    store = get_store(CHROMA, config.collection_name(args.strategy))
    print(f"retriever: {args.retriever}  |  store: {store.count()} vectors in "
          f"'{config.collection_name(args.strategy)}'  |  routing: "
          f"{'on' if args.route else 'off'}\n")

    # The destination is resolved before the loop now, because rows are written
    # as they complete. A retrieval-only run writes to its own file: it used to
    # write to <label>.csv like a full run, and a --no-llm pass then silently
    # replaced a results file holding answer verdicts with one whose answer
    # columns were blank. Nothing failed; the measurements were simply gone,
    # and they had cost 30 API calls and eleven minutes.
    RESULTS.mkdir(parents=True, exist_ok=True)
    dest = RESULTS / (f"{label}.retrieval.csv" if args.no_llm else f"{label}.csv")
    partial = RESULTS / f"{dest.stem}.partial.csv"

    # A run that died used to leave nothing at all on disk, because the CSV was
    # written once after the loop. The routed run died on question 27 and all 26
    # completed rows went with it -- recoverable only because every answer was
    # in the LLM cache. Rows are flushed as they complete now, and into the
    # .partial file rather than the real destination, so a re-run that dies
    # early can never truncate a complete results file. The invariant is worth
    # stating: <label>.csv exists only for a run that finished.
    done: dict[str, dict] = {}
    header: list[str] = []
    if args.resume and partial.exists():
        rdr = csv.DictReader(partial.open())
        done = {r["qid"]: coerce_row(r) for r in rdr}
        header = list(rdr.fieldnames or [])
        print(f"--resume: {len(done)} rows already in "
              f"{partial.relative_to(ROOT)}; skipping those questions\n")
    elif args.resume:
        print(f"--resume: no {partial.relative_to(ROOT)}; starting from scratch\n")

    sink: dict = {"fh": None, "w": None}

    def emit(row: dict) -> None:
        """Append one finished row and flush it, so a crash keeps the rest."""
        if sink["w"] is None:
            fresh = not done
            cols = list(row.keys())
            # Appending to a file whose header is in a different order would
            # write every value into the wrong column, and DictWriter would not
            # complain. Refusing is better than a results file that looks fine.
            if not fresh and header and header != cols:
                raise SystemExit(
                    f"{partial.name} has a different header to this run's "
                    f"columns; delete it or drop --resume rather than appending "
                    f"misaligned rows")
            sink["fh"] = partial.open("w" if fresh else "a", newline="")
            sink["w"] = csv.DictWriter(sink["fh"], fieldnames=cols)
            if fresh:
                sink["w"].writeheader()
        sink["w"].writerow(row)
        sink["fh"].flush()

    out, t_start = [], time.time()
    stopped = None
    print(f"{'qid':5s} {'type':12s} {'r@1':>4s} {'r@5':>4s} {'r@10':>5s} "
          f"{'answer':18s} note")
    print("-" * 96)

    # A quota limit, a network drop or a Ctrl-C stops the run where it is
    # instead of discarding the questions that already cost tokens. The rows
    # are already on disk; what this adds is a non-zero exit, a meta file that
    # says the run is partial, and the exact command to pick it back up.
    try:
        for r in rows:
            if r["qid"] in done:
                out.append(done[r["qid"]])
                print(f"{r['qid']:5s} {r['type'][:12]:12s} "
                      f"{'':>4s} {'':>4s} {'':>5s} {'(resumed)':18s}")
                continue

            gold = {b for b in r["gold_block_ids"].split(";") if b}
            hits, rplan = route.retrieve(searcher, r["question"],
                                         k=RETRIEVE_K, route=args.route)

            rec = {}
            for k in RECALL_AT:
                got = set(retrieved_blocks(hits[:k]))
                rec[k] = int(bool(gold & got)) if gold else ""

            # recall at equal context budget: take chunks until the prompt is full
            n_ctx, used = 0, 0
            for h in hits:
                t = h["metadata"].get("total_tokens") or 0
                if used + t > CTX_TOKENS and n_ctx:
                    break
                used += t; n_ctx += 1
            ctx_blocks = set(retrieved_blocks(hits[:n_ctx]))
            rec_ctx = int(bool(gold & ctx_blocks)) if gold else ""
            named, _ = route.detect_companies(r["question"])
            cov = (round(company_coverage(gold, ctx_blocks, named), 3)
                   if gold and len(named) >= 2 else "")

            row = dict(
                qid=r["qid"], type=r["type"], split=r["split"],
                question=r["question"], n_gold=r["n_gold"],
                **{f"recall@{k}": rec[k] for k in RECALL_AT},
                recall_ctx=rec_ctx, coverage=cov, ctx_chunks=n_ctx, ctx_tokens=used,
                top1_score=round(hits[0]["score"], 4) if hits else "",
                retrieved_tickers=",".join(sorted({h["metadata"]["ticker"] for h in hits[:5]})),
                route_mode=rplan["mode"],
            )

            if args.no_llm:
                row.update(answer="", verdict="", note="", abstained="",
                           cached="", completion_tokens="")
            else:
                res = ask_mod.ask(r["question"], k=config.K, strategy=args.strategy,
                                  use_cache=not args.no_cache, route=args.route,
                                  retriever=args.retriever)
                ans = res["answer"]
                is_abstention = abstained(ans)

                if r["type"] == "unanswerable":
                    verdict = "correct" if is_abstention else "wrong"
                    note = ("abstained" if is_abstention
                            else "answered when it should not have")
                elif is_abstention:
                    verdict, note = "false_abstention", "declined but the answer is in the corpus"
                elif r["type"] == "lookup":
                    verdict, note = score_numeric(ans, float(r["expected_answer"]),
                                                  float(r["tolerance"]))
                elif r["type"] == "comparative":
                    verdict, note = score_comparative(ans, r["expected_answer"],
                                                      float(r["tolerance"]))
                else:
                    verdict, note = judge(r["question"], r["expected_answer"], ans,
                                          use_cache=not args.no_cache)

                row.update(answer=ans.replace("\n", " "), verdict=verdict, note=note,
                           abstained=int(is_abstention), cached=int(res["cached"]),
                           completion_tokens=res["completion_tokens"])

            out.append(row)
            emit(row)
            print(f"{row['qid']:5s} {row['type'][:12]:12s} "
                  f"{str(row['recall@1']):>4s} {str(row['recall@5']):>4s} "
                  f"{str(row['recall@10']):>5s} {str(row.get('verdict',''))[:18]:18s} "
                  f"{str(row.get('note',''))[:40]}")

    except (Exception, KeyboardInterrupt) as e:
        stopped = f"{type(e).__name__}: {e}"
        print(f"\n!! stopped after {len(out)} of {len(rows)} questions")
        print(f"   {stopped[:300]}")
        if out:
            print(f"   {len(out)} rows kept in {partial.relative_to(ROOT)}")
            print("   re-run the same command with --resume to finish it")

    elapsed = time.time() - t_start

    if sink["fh"] is not None:
        sink["fh"].close()

    if not out:
        print("\nno questions completed; nothing written")
        raise SystemExit(1)

    if stopped:
        # Deliberately NOT written to dest. A results file means a finished run;
        # a partial one that overwrote the real file would be indistinguishable
        # from a complete run two weeks from now.
        print(f"\npartial run: {len(out)} of {len(rows)} questions in "
              f"{partial.relative_to(ROOT)}")
    else:
        cols = list(out[0].keys())
        with dest.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader(); w.writerows(out)
        partial.unlink(missing_ok=True)
        # The partial meta goes too. Leaving it behind means a finished run has
        # a file beside it announcing that it never finished.
        (RESULTS / f"{partial.stem.replace('.partial', '')}.partial.meta.json"
         ).unlink(missing_ok=True)

    # Configuration is stamped alongside every results file. Free-tier model
    # endpoints get updated without notice, so a number with no record of what
    # produced it cannot be compared against next week's number.
    # config.describe() reports the configured DEFAULT strategy, not the one
    # this run used, so it is spread FIRST and the run's actual arguments are
    # written over it. The other way round -- which is how this started -- made
    # every results file claim strategy "baseline" whatever was actually run,
    # in the one file whose entire purpose is to say what produced a number.
    meta = {**config.describe(),
            "label": label,
            "strategy": args.strategy,
            "retriever": args.retriever,
            "route": args.route,
            "ctx_tokens_budget": CTX_TOKENS,
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "questions": len(out), "split": args.split, "llm": not args.no_llm,
            "elapsed_s": round(elapsed, 1), "recall_at": list(RECALL_AT),
            "complete": stopped is None, "asked": len(rows)}
    if stopped:
        meta["stopped_with"] = stopped[:300]
    stem = f"{label}.retrieval" if args.no_llm else label
    (RESULTS / (f"{stem}.partial.meta.json" if stopped
                else f"{stem}.meta.json")).write_text(json.dumps(meta, indent=2))

    # ---------------- summary ----------------
    def pct(xs):
        xs = [x for x in xs if x != ""]
        return f"{100*sum(xs)/len(xs):5.1f}%" if xs else "    -"

    print("\n" + "=" * 70)
    print(f"RETRIEVAL  (no API calls; {len(out)} questions)")
    print("=" * 70)
    print(f"{'type':14s} {'n':>3s} " +
          " ".join(f"{'r@'+str(k):>7s}" for k in RECALL_AT) +
          f"{'r@ctx':>8s} {'chunks':>7s}")
    for t in ("lookup", "conceptual", "comparative"):
        g = [r for r in out if r["type"] == t]
        if g:
            print(f"{t:14s} {len(g):3d} " +
                  " ".join(f"{pct([r[f'recall@{k}'] for r in g]):>7s}" for k in RECALL_AT) +
                  f"{pct([r['recall_ctx'] for r in g]):>8s} "
                  f"{sum(r['ctx_chunks'] for r in g)/len(g):7.1f}")
    ans_g = [r for r in out if r["type"] != "unanswerable"]
    if ans_g:
        print("-" * 70)
        print(f"{'ALL answerable':14s} {len(ans_g):3d} " +
              " ".join(f"{pct([r[f'recall@{k}'] for r in ans_g]):>7s}" for k in RECALL_AT) +
              f"{pct([r['recall_ctx'] for r in ans_g]):>8s} "
              f"{sum(r['ctx_chunks'] for r in ans_g)/len(ans_g):7.1f}")
    print(f"\nr@ctx = recall at an equal {CTX_TOKENS}-token context budget "
          f"(the honest strategy comparison; r@5 is not, because chunk sizes differ)")

    multi = [r for r in out if r["coverage"] != ""]
    if multi:
        cov = sum(float(r["coverage"]) for r in multi) / len(multi)
        full = sum(1 for r in multi if float(r["coverage"]) == 1.0)
        print(f"\nMULTI-COMPANY COVERAGE ({len(multi)} questions naming 2+ filers, "
              f"at the same context budget)")
        print(f"  companies retrieved        : {100*cov:5.1f}% of those named")
        print(f"  questions with ALL covered : {full}/{len(multi)}")
        print("  (recall@k cannot see this: one company satisfies it)")
        for r in multi:
            print(f"    {r['qid']}  coverage {float(r['coverage']):.2f}  "
                  f"route={r['route_mode']:12s} got={r['retrieved_tickers']}")

    if not args.no_llm:
        print("\n" + "=" * 70)
        print("ANSWERS")
        print("=" * 70)
        for t in ("lookup", "conceptual", "comparative", "unanswerable"):
            g = [r for r in out if r["type"] == t]
            if not g:
                continue
            vs = {}
            for r in g:
                vs[r["verdict"]] = vs.get(r["verdict"], 0) + 1
            ok = sum(v for k, v in vs.items() if k == "correct")
            print(f"{t:14s} {len(g):3d}  correct {ok}/{len(g)}   "
                  f"{', '.join(f'{k}={v}' for k, v in sorted(vs.items()))}")

        print("\n" + "=" * 70)
        print("ABSTENTION")
        print("=" * 70)
        un = [r for r in out if r["type"] == "unanswerable"]
        ansble = [r for r in out if r["type"] != "unanswerable"]
        print(f"  correct abstention (should decline, did) : "
              f"{sum(r['abstained'] for r in un)}/{len(un)}")
        print(f"  false abstention  (answer existed, declined): "
              f"{sum(r['abstained'] for r in ansble)}/{len(ansble)}")

        cached = sum(r["cached"] for r in out)
        print(f"\ncache: {cached}/{len(out)} answers served from disk; "
              f"{len(out)-cached} new API calls")

    # Exit non-zero on a partial run. The summary above prints real numbers over
    # however many questions finished, which reads exactly like a complete run;
    # the status code is what stops a partial from being mistaken for one.
    if stopped:
        print(f"\nINCOMPLETE: {len(out)}/{len(rows)} questions. Finish with:")
        print(f"  python -m eval.run_eval --strategy {args.strategy} "
              f"--retriever {args.retriever}"
              f"{' --route' if args.route else ''} --label {label} "
              f"--split {args.split} --resume")
        raise SystemExit(1)

    print(f"\nwrote {dest.relative_to(ROOT)} and {label}.meta.json  "
          f"({elapsed:.0f}s)")
    print("\nDEV/TEST reminder: tune on dev, report on test. "
          f"dev={sum(1 for r in out if r['split']=='dev')} "
          f"test={sum(1 for r in out if r['split']=='test')}")


if __name__ == "__main__":
    main()
