"""Step 20 — the third path: ask back instead of guessing.

Step 11 gave the router two answers: one search, or one search per filer. A
question that names nobody gets one of them. That is the whole decision, and
for most questions it is enough.

It is not enough for a question like this one, which a third-party writer
produced without being prompted to:

    "What were the company's total revenues in 2025?"

There are three companies. The answer is a different number depending on which
filing is opened, so neither path is right. `single` answers from whichever
filer's wording ranks highest and presents it as the answer. `comparative`
returns three revenues to a question that asked for one. Thirteen of the
writer's original thirty questions had this shape (see
eval/router_questions_ask.csv, bucket E) -- they were cut from the Step 11 set
precisely because no label could score them, which is the evidence that a
two-way router is missing a case rather than making a hard call.

The test that separates `unclear` from `single`:

    if a different company's filing were opened, would the answer change?

"What types of claims are covered by a title insurance policy?" -- no. Same
answer from any of the three, so one search is correct. "What were the
company's total revenues?" -- yes. So the question has to go back to the user.

Why a separate module instead of a third label in jev_router.py: adding a
label rewrites CRITERIA, and the Step 11 result (30/30 on a held-out set) was
measured on the two-label prompt. Editing that prompt in place would retire a
measurement to add a feature, and the two would never again be comparable.
Both routers are importable, both are scored by eval/run_router_eval3.py, and
the frozen one keeps its number.

The cost table is not the Step 11 cost table. There, a miss (comparison routed
to one search) was the expensive error and a waste (single routed to three) was
merely slow. Here there is a third error with a cost the others do not have:

  * asks when it should not  -- a clear question bounced back; the user answers
    a question they already answered. Friction, and the kind that makes people
    stop using a system.
  * silent when it should ask -- the confident wrong answer, same cost as a
    Step 11 miss.

So over-firing is cheap in correctness and expensive in trust, which is why
`unclear` is described below in terms of the test above rather than in terms of
vague wording, and why it is scored as its own bucket rather than folded in.
"""

import hashlib
import json
import time

from dotenv import load_dotenv

from src import config
from src.config import ROOT
from src.route import comparative_trigger, named_companies

load_dotenv(ROOT / ".env")

_CLIENT = None

# Shared directory with Step 11. Safe: the key hashes INSTRUCTIONS and CRITERIA
# along with the question, and both differ here, so the two routers can never
# read each other's decisions.
CACHE = ROOT / ".cache" / "jev"

MODEL = "jev-latest"

# Step 11 measured its floor doing harm and set it to 0.0. That finding was
# about widening a `single` to `comparative`, which is not the decision here,
# so it does not transfer. Left off for now and deliberately not swept until
# the first run exists -- picking a threshold before seeing the distribution is
# how the Step 11 floor got justified by an argument instead of a measurement.
CONFIDENCE_FLOOR = 0.0

INSTRUCTIONS = (
    "These questions are put to a question-answering system over the SEC "
    "filings (10-K and 10-Q) of three title insurers: First American "
    "Financial, Fidelity National Financial, and Stewart Information "
    "Services. The system can run one search over the whole corpus, or run a "
    "separate search per company and merge the results, or stop and ask the "
    "user which company they meant. Running a separate search per company is "
    "what makes a comparison possible, because one search tends to return "
    "passages from whichever company's wording ranks highest and silently "
    "omit the others. Stopping to ask is for questions that need one "
    "company's figures but do not say which company. Which does this question "
    "need?"
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
        "One search answers this question, and the answer does not depend on "
        "which of the three companies is examined. It asks for a definition, "
        "a general practice of the title insurance industry, how a figure is "
        "calculated, what a filing section means, or a fact that holds across "
        "the filings. Opening a different company's filing would not change "
        "the answer."
    ),
    "unclear": (
        "This question asks for one company's own figure, policy, segment, "
        "headcount, risk, legal proceeding or year-over-year change, but does "
        "not say which of the three companies it means -- for example by "
        "saying 'the company' when there are three. Opening a different "
        "company's filing WOULD change the answer, so there is no single "
        "correct response and the user has to be asked which company they "
        "meant. A question that does name a company, or that asks about all "
        "of them, does not belong here."
    ),
}

# What the system says back when the route is `clarify`. Fixed text rather than
# generated: the question has already been shown not to identify a company, so
# there is nothing to generate from, and a model-written clarification is one
# more thing that can hallucinate a fourth company into the list.
CLARIFY_PROMPT = (
    "That question needs one company's figures, but it does not say which "
    "one. Did you mean First American Financial (FAF), Fidelity National "
    "Financial (FNF), or Stewart Information Services (STC)? Or all three?"
)


def _client():
    """The Jev client, created once. Raises if TYPESAFE_API_KEY is unset."""
    global _CLIENT
    if _CLIENT is None:
        from typesafe_sdk import TypeSafeClient
        _CLIENT = TypeSafeClient()
    return _CLIENT


def _cache_key(question: str) -> str:
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
    """Ask Jev which of the three paths this question needs.

    Same return shape as jev_router.classify, so one eval can score both.
    Latency is recorded only on a live call; a cache hit reports the latency of
    the call that filled it, hence `cached` alongside.
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
    except Exception as exc:                       # noqa: BLE001
        # Broad on purpose: a DNS failure or an HTML error page is not a
        # TypeSafeError, and the caller's fallback is correct for all of them.
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
    """`route.plan`'s signature and return shape, plus one new mode.

    Precedence is `route.plan`'s, unchanged, with the fourth rule replaced:

      1. two or more filers named  -> comparative   (exact alias match)
      2. one filer named           -> filtered      (exact alias match)
      3. nobody named              -> Jev: comparative / single / unclear
      4. unclear, mid-conversation -> followup, NOT a question back

    Rule 4 is the subtle one. "What were the company's revenues?" is unclear
    cold and perfectly clear three turns into a conversation about Stewart --
    the conversation is what the word "the" is pointing at. Asking anyway
    would be the system failing to listen, so `clarify` fires only when there
    is no history to resolve it.

    Mode `clarify` carries `ask` -- the text to put back to the user -- and no
    companies, because no search is run.
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
        # Degrade to the word list, and never to `clarify`: an outage is not
        # evidence that the question was ambiguous, and asking the user to
        # re-type a clear question because the network failed is the worst of
        # the available behaviours. Said out loud in `why`.
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

    if d["choice"] == "unclear":
        if prev_companies:
            carried = [tk for tk in config.TICKERS if tk in set(prev_companies)]
            return {"mode": "followup", "companies": carried,
                    "why": f"jev: unclear, confidence {d['confidence']:.2f}, "
                           f"but {', '.join(carried)} is already under "
                           f"discussion, so the question resolves against the "
                           f"conversation instead of going back to the user",
                    "jev": d}
        return {"mode": "clarify", "companies": [], "ask": CLARIFY_PROMPT,
                "why": f"jev: unclear, confidence {d['confidence']:.2f}; the "
                       f"question needs one company's figures but names none, "
                       f"and there is no conversation to resolve it; asked "
                       f"back instead of searching",
                "jev": d}

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
