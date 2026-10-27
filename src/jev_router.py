"""Step 11 — the routing decision that was a word list, made by a typed model.

What this replaces, and nothing more. `route.plan` decides four things in
strict precedence (src/route.py:155). Three of them are not judgement calls:

  * a question naming two or more filers is comparative  -- exact alias match
  * a question naming one filer is filtered to it        -- exact alias match
  * a question naming none, mid-conversation, is a follow-up -- state, not text

Those stay. Jev cannot beat `"first american" in question.lower()`, and a
model that is merely as good as an exact string match is a worse component
than the string match, because it can fail at the network.

The fourth is the brittle one, and it is the only thing this module decides:

    a question that names no company -- does answering it need every filer
    searched separately, or will one search do?

Today that is COMPARATIVE_PHRASES (src/route.py:58), a hand-written list of
superlatives. It is right more often than it deserves to be. "Which company
should I contact on the basis of their least loss in the last year" routed
correctly because `least` happens to be on the list -- not because anything
understood that the question ranks three companies. "Who is doing best?"
carries none of those words and falls straight through to a single unfiltered
search, which lands in one filer's neighbourhood and answers as though the
other two had been considered and lost. That is the failure Step 10 was
written to kill, and the word list is where it survives.

Why a Choice rather than prompting the answer model for a label: a chat model
returns prose that has to be parsed, and "comparative" in that prose carries
no usable degree of certainty. Jev returns the label, a calibrated probability
per label, and a confidence -- which is what a threshold needs. The parsing
step disappears, and with it the class of bug where the router silently
defaults because a word came back in an unexpected form.

Design rules this module keeps, from the Step 11 plan:

  * Only the question goes to Jev. Never a retrieved passage. The router
    decides HOW to search, so it has no use for the corpus. Measured at 518
    to 538 input tokens per decision -- the plan guessed ~80, which was wrong
    because the label descriptions below are prose, not keywords. Still
    small enough that routing costs a rounding error next to the answer
    call, which is the property that actually matters.
  * The signature matches `route.plan` exactly, so the two are swappable and
    the comparison is a swapped argument rather than a rewrite.
  * It degrades instead of breaking. Any API failure falls back to the word
    list and says so in the plan, because a router that raises takes the whole
    pipeline down over a decision it could have guessed.

Nothing here changes an existing measurement: `route.retrieve` still calls
`route.plan` unless a planner is passed explicitly.
"""

import hashlib
import json
import time

from dotenv import load_dotenv

from src import config
from src.config import ROOT
from src.route import comparative_trigger, named_companies

load_dotenv(ROOT / ".env")

# Deferred so importing this module costs nothing and a missing key is an
# error at call time, where it can be reported, not at import time, where it
# would break `import src.route` for everyone.
_CLIENT = None

CACHE = ROOT / ".cache" / "jev"

MODEL = "jev-latest"

# Below this confidence, widen to comparative instead of committing to one
# search. The reasoning was: searching all three filers costs no API call,
# because retrieval is local, while routing a comparison to a single search is
# the failure that scored 25 per cent in Step 8 -- so when unsure, widen.
#
# Set to 0.0, which switches the rule off, because eval/run_router_eval.py
# measured it doing harm and buying nothing. Nine of 45 decisions came in below
# 0.60. Every one of them said "single", and every one of them was right. The
# rule overrode nine correct answers and made them three searches instead of
# one. Sweeping the threshold over the same 45 cached decisions:
#
#     floor 0.0   accuracy 100.0%   missed comparisons 0   wasted searches 0
#     floor 0.5   accuracy  82.2%   missed comparisons 0   wasted searches 8
#     floor 0.6   accuracy  80.0%   missed comparisons 0   wasted searches 9
#     floor 0.8   accuracy  73.3%   missed comparisons 0   wasted searches 12
#     floor 1.0   accuracy  53.3%   missed comparisons 0   wasted searches 21
#
# The column that justified the rule is the third one, and it is zero at every
# setting including off. The rule was guarding against something that never
# happened on this set, so it was cost with no benefit.
#
# Kept as a constant rather than deleted, because the honest reading of n=9 is
# that the rule is unsupported, not that it could never help. If a later
# question set produces a missed comparison from a low-confidence decision,
# this is the knob that answers it, and the sweep above is how to reset it.
CONFIDENCE_FLOOR = 0.0

INSTRUCTIONS = (
    "These questions are put to a question-answering system over the SEC "
    "filings (10-K and 10-Q) of three title insurers: First American "
    "Financial, Fidelity National Financial, and Stewart Information "
    "Services. The system can either run one search over the whole corpus, "
    "or run a separate search per company and merge the results. Running a "
    "separate search per company is what makes a comparison possible, "
    "because one search tends to return passages from whichever company's "
    "wording ranks highest and silently omit the others. Which does this "
    "question need?"
)

CRITERIA = {
    "comparative": (
        "Answering this question correctly requires figures or statements "
        "from more than one of the three companies -- comparing them, "
        "ranking them, choosing the highest or lowest or best or worst, or "
        "asking which company has some property. A question that names no "
        "company but implies a choice between them belongs here."
    ),
    "single": (
        "Answering this question needs only one search. It asks for a single "
        "figure, a definition, a policy, a risk, or a fact about the filings "
        "in general, and does not require setting the companies against each "
        "other."
    ),
}


def _client():
    """The Jev client, created once. Raises if TYPESAFE_API_KEY is unset."""
    global _CLIENT
    if _CLIENT is None:
        from typesafe_sdk import TypeSafeClient
        _CLIENT = TypeSafeClient()
    return _CLIENT


def _cache_key(question: str) -> str:
    """Keyed on the prompt as well as the question.

    The label descriptions are part of the decision, so editing CRITERIA has
    to invalidate the cache. Keying on the question alone would silently serve
    answers produced by wording that no longer exists, and the calibration
    table would be measuring a prompt that was never sent.
    """
    blob = json.dumps({"q": question, "model": MODEL,
                       "instructions": INSTRUCTIONS, "criteria": CRITERIA},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _cache_read(key: str) -> dict | None:
    f = CACHE / f"{key}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return None


def _cache_write(key: str, record: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f"{key}.json.tmp"
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    tmp.rename(CACHE / f"{key}.json")      # atomic: no half-written cache files


def classify(question: str, use_cache: bool = True) -> dict:
    """Ask Jev whether this question needs every filer searched separately.

    Returns the decision with everything the comparison table needs:
    the label, its confidence, the probability of each label, input tokens,
    latency, and whether the answer came from disk. On failure returns
    `error` set and `choice` None -- the caller decides what to do about it,
    because only the caller knows whether a fallback is acceptable.

    Latency is recorded only on a live call. A cache hit reports the latency
    of the call that filled it, which would flatter the router in a timing
    table, so `cached` is returned alongside and the eval reports live calls
    separately.
    """
    key = _cache_key(question)
    if use_cache:
        hit = _cache_read(key)
        if hit is not None:
            return {**hit, "cached": True}

    from typesafe_sdk import Choice

    t0 = time.perf_counter()
    try:
        resp = _client().system_one(
            state=question,
            questions={"route": Choice(instructions=INSTRUCTIONS,
                                       criteria=CRITERIA)},
            model=MODEL,
        )
    except Exception as exc:                       # noqa: BLE001 -- see below
        # Deliberately broad. Every typesafe error subclasses TypeSafeError,
        # but a DNS failure or a proxy returning HTML arrives as something
        # else entirely, and the caller's fallback is correct for all of them.
        # Narrowing this would turn an outage into a traceback.
        return {"choice": None, "confidence": 0.0, "probabilities": {},
                "input_tokens": None, "latency_ms": None, "cached": False,
                "error": f"{type(exc).__name__}: {exc}"}

    answer = resp.choices["route"]
    record = {
        "choice": answer.choice,
        "confidence": answer.confidence,
        "probabilities": dict(answer.probabilities),
        "input_tokens": resp.usage.input_tokens if resp.usage else None,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "model": resp.model,
        "error": None,
    }
    _cache_write(key, record)
    return {**record, "cached": False}


def plan(question: str, prev_companies: list[str] | None = None,
         use_cache: bool = True) -> dict:
    """Same signature, same return shape, as `route.plan`. Swappable with it.

    The precedence below is `route.plan`'s, with one rule replaced. Keeping
    the order identical is what makes the comparison honest: if the two
    routers disagree, it is because Jev and the word list disagree, not
    because the rules around them were rearranged.

    The returned dict carries `jev` with the raw decision, so the plan can be
    shown and scored rather than inferred from the mode it produced.
    """
    cos, why = named_companies(question)
    if len(cos) >= 2:
        return {"mode": "comparative", "companies": cos,
                "why": f"{why}; one filtered search each", "jev": None}
    if len(cos) == 1:
        return {"mode": "filtered", "companies": cos,
                "why": f"{why} ({cos[0]}); search restricted to it",
                "jev": None}

    d = classify(question, use_cache=use_cache)

    if d["error"]:
        # Degrade to the word list rather than raise. Said out loud in `why`,
        # because a plan that silently fell back would make the comparison
        # table a measurement of the rules wearing Jev's name.
        hit = comparative_trigger(question)
        if hit:
            return {"mode": "comparative", "companies": list(config.TICKERS),
                    "why": f"jev unavailable ({d['error']}); fell back to the "
                           f"word list, which matched {hit.strip()!r}",
                    "jev": d}
        if prev_companies:
            carried = [tk for tk in config.TICKERS if tk in set(prev_companies)]
            return {"mode": "followup", "companies": carried,
                    "why": f"jev unavailable ({d['error']}); no company named; "
                           f"{', '.join(carried)} carried from the conversation",
                    "jev": d}
        return {"mode": "single", "companies": [],
                "why": f"jev unavailable ({d['error']}); fell back to the word "
                       f"list, which matched nothing; unfiltered search",
                "jev": d}

    low = d["confidence"] < CONFIDENCE_FLOOR

    if d["choice"] == "comparative" or low:
        reason = (f"jev: comparative, confidence {d['confidence']:.2f}"
                  if not low else
                  f"jev: {d['choice']}, confidence {d['confidence']:.2f} below "
                  f"the {CONFIDENCE_FLOOR:.2f} floor, so the search was "
                  f"widened rather than committed")
        return {"mode": "comparative", "companies": list(config.TICKERS),
                "why": f"{reason}; one filtered search each", "jev": d}

    if prev_companies:
        carried = [tk for tk in config.TICKERS if tk in set(prev_companies)]
        return {"mode": "followup", "companies": carried,
                "why": f"jev: single, confidence {d['confidence']:.2f}; no "
                       f"company named; {', '.join(carried)} carried from the "
                       f"conversation, merged with a corpus-wide search",
                "jev": d}

    return {"mode": "single", "companies": [],
            "why": f"jev: single, confidence {d['confidence']:.2f}; "
                   f"unfiltered search", "jev": d}
