"""Step 10 — route a question to the right retrieval, instead of always one search.

Why this step exists, from the Step 7 and 8 measurements: asked to compare two
of the three filers, a single vector search returned passages from ONE of them,
three times out of three. Asked about all three, it returned two and dropped
the third. Comparative questions scored 0 per cent retrieval in Step 7 and 25
per cent after Step 8, the worst category in the system.

The cause is not corpus imbalance, which was the original hypothesis. It is
that one question becomes one query vector, and one query vector lands in one
neighbourhood. These three companies are competitors in a single industry and
their filings use nearly identical language, so the neighbourhood belongs to
whichever company's wording happens to sit closest -- and the others are not
retrieved at all, however many passages are returned.

So a question naming more than one company is decomposed: one metadata-filtered
retrieval per company, merged round-robin by rank. Merging by SCORE would
rebuild the problem it is meant to solve, because the dominant company's
passages would simply take every slot again.
"""

import re

from src import config

# Matched case-insensitively against the question. Deliberately narrow: a
# false positive silently changes how a question is retrieved.
ALIASES: dict[str, tuple[str, ...]] = {
    "FAF": ("first american", "faf"),
    "FNF": ("fidelity national", "fnf", "f&g"),
    "STC": ("stewart", "stc"),
}

# Matched at word edges, not as bare substrings. The substring form shipped for
# three steps described in three documents as "free and never wrong", and the
# offline scorer in eval/run_router_named_eval.py measured it at 5/7 on
# adversarial questions: "What lines of business does Westcor underwrite?" was
# filtered to Stewart, because "weSTCor" contains "stc". Westcor Land Title is a
# real competitor in this industry, so that is a question a user asks, and the
# plan reported "1 company name(s) found" for a company the question never
# named. An English word list does not catch it -- "westcor" is not in one --
# and nor would a competitor list, since the next competitor is not on it
# either. A word edge catches the whole class at no cost. Punctuation is an
# edge, so "Stewart's", "(FNF)" and "stc," all still match.
ALIAS_RE: dict[str, re.Pattern[str]] = {
    tk: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b")
    for tk, words in ALIASES.items()
}

# Phrases that mean every filer without naming one. "Which of the three title
# insurers reported the lowest revenue" names nobody and still needs all three.
ALL_PHRASES = (
    "all three", "the three", "three title insurer", "three companies",
    "each company", "each of the", "across the companies", "all of the companies",
    # Bare definite plurals. "What are the names of the companies" named two of
    # three, because it fell through to one unfiltered search and answered from
    # whichever ten passages ranked -- the third filer was never searched for.
    # Note the singular "the company" is NOT here: it is a follow-up cue below,
    # meaning the one filer already under discussion.
    "the companies", "the filers", "the insurers",
)

# Questions that are comparative without naming anybody. "Which company has the
# highest revenue?" asks about all three filers and mentions none of them, so
# detection by company name misses it entirely and the question falls through to
# a single unfiltered search -- which lands in one neighbourhood and silently
# answers about whichever two companies happened to rank, as if the third had
# been considered and lost. That is a worse failure than retrieving nothing.
#
# These fire ONLY when no company is named and no ALL_PHRASES matched, so
# "First American's highest revenue year" stays a single-company question.
# Kept conservative on purpose: "most" and "better" are ordinary prose in these
# filings and would fan out questions that are not comparisons at all.
COMPARATIVE_PHRASES = (
    "which company", "which insurer", "which firm", "which filer", "which of the",
    "highest", "lowest", "largest", "smallest", "biggest",
    "compare", "comparison", "versus", " vs ", "rank",
)

YEAR_RE = re.compile(r"\b(?:fiscal\s+|fy\s*)?(20\d{2})\b", re.I)

NAMES = {"FAF": "First American Financial",
         "FNF": "Fidelity National Financial",
         "STC": "Stewart Information Services"}

# Longest first, so "stewart information services" is removed before "stewart".
_STRIP_NAMES = sorted(
    {w for ws in ALIASES.values() for w in ws} | {n.lower() for n in NAMES.values()},
    key=len, reverse=True)

# Framing words that belong to the comparison, not to either sub-question.
_STRIP_FRAMING = re.compile(
    r"\b(which|compare|comparison|between|had|has|have|higher|lower|lowest|"
    r"highest|more|less|greater|or|and|the three|three|title insurers|"
    r"insurers|reported|report|of)\b", re.I)


def subquery(question: str, ticker: str) -> str:
    """Rewrite a comparison into a single-company question.

    Metadata filtering alone was not enough, and the measurement said so. The
    same blended query vector was being sent to each company's filtered search,
    and that vector mentions every company in the question -- so inside First
    American's filing it matched passages about competitors rather than First
    American's own revenue table. Coverage stayed at one company in two even
    with thirty passages retrieved.

    Rewriting the query per company took the first comparison question from no
    coverage to full coverage at five passages. Filtering decides WHERE to
    look; the rewrite decides WHAT to look for, and both are needed.
    """
    out = question
    for w in _STRIP_NAMES:
        out = re.sub(rf"\b{re.escape(w)}\b", " ", out, flags=re.I)
    out = _STRIP_FRAMING.sub(" ", out)
    out = re.sub(r"[\s,]{2,}", " ", out).strip(" ,?.")
    return f"{NAMES[ticker]} {out}".strip()


def comparative_trigger(question: str) -> str | None:
    """The superlative or comparison word that fanned a question out, or None."""
    q = question.lower()
    return next((p for p in COMPARATIVE_PHRASES if p in q), None)


def named_companies(question: str) -> tuple[list[str], str]:
    """Companies the question states for itself, by alias or by "all three".

    This is the question's OWN evidence, as against a company carried over from
    an earlier turn or inferred from comparative wording. Kept separate because
    the two are not interchangeable: explicit evidence always wins.
    """
    q = question.lower()
    if any(p in q for p in ALL_PHRASES):
        return list(config.TICKERS), "question refers to all filers explicitly"
    found = {tk for tk, pat in ALIAS_RE.items() if pat.search(q)}
    named = [tk for tk in config.TICKERS if tk in found]
    if named:
        return named, f"{len(named)} company name(s) found in the question"
    return [], ""


def detect_companies(question: str) -> tuple[list[str], str]:
    """Tickers the question is about, in the project's canonical order.

    Returns (tickers, why) so the reason can be shown in the plan rather than
    inferred from the result.

    Precedence matters. An explicit "all three" wins outright; a named company
    is next; only a question that names nobody falls through to the superlative
    and comparison triggers. Checking the triggers last is what keeps "what was
    First American's highest revenue year" a single-company search.
    """
    named, why = named_companies(question)
    if named:
        return named, why
    hit = comparative_trigger(question)
    if hit:
        return (list(config.TICKERS),
                f"comparative wording ({hit.strip()!r}) with no company named")
    return [], "no company named and no comparative wording"


def detect_years(question: str) -> list[str]:
    """Fiscal years named in the question, restricted to those in the corpus."""
    years = {m for m in YEAR_RE.findall(question)}
    return sorted(years & set(config.FISCAL_YEARS)) if hasattr(
        config, "FISCAL_YEARS") else sorted(years)


def plan(question: str, prev_companies: list[str] | None = None) -> dict:
    """Decide how to retrieve, before retrieving. Returned so it can be shown.

    Four rules, in this order:

      1. companies the question names itself   -> filtered / comparative
      2. comparative wording, nobody named     -> one search per filer
      3. nobody named, but a conversation      -> followup (see retrieve)
      4. nothing to go on                      -> one unfiltered search

    Rule 3 used to try to detect whether the wording "referred back" -- short
    questions, pronouns, phrases like "what about". That was a guess about
    phrasing, and it made the user responsible for wording a question so the
    router would understand it: "and what is the loss ratio?" was six words
    and fell through. The guess is gone. When a question names nobody and a
    conversation exists, BOTH searches run and are merged, because retrieval
    costs no API call and merging two cheap searches beats guessing which one
    was wanted.

    `prev_companies` is what the conversation is already about. With no
    history rules 1, 2 and 4 are all that can fire, which is exactly the
    behaviour every measured number in eval/ was produced with.
    """
    cos, why = named_companies(question)
    if len(cos) >= 2:
        return {"mode": "comparative", "companies": cos,
                "why": f"{why}; one filtered search each"}
    if len(cos) == 1:
        return {"mode": "filtered", "companies": cos,
                "why": f"{why} ({cos[0]}); search restricted to it"}

    hit = comparative_trigger(question)
    if hit:
        return {"mode": "comparative", "companies": list(config.TICKERS),
                "why": f"comparative wording ({hit.strip()!r}) with no company "
                       f"named; one filtered search each"}

    if prev_companies:
        carried = [tk for tk in config.TICKERS if tk in set(prev_companies)]
        return {"mode": "followup", "companies": carried,
                "why": f"no company named; {', '.join(carried)} carried from "
                       f"the conversation, merged with a corpus-wide search"}

    return {"mode": "single", "companies": [],
            "why": "no company named and no comparative wording; "
                   "unfiltered search"}


def retrieve(searcher, question: str, k: int,
             route: bool = True,
             prev_companies: list[str] | None = None,
             planner=None,
             ) -> tuple[list[dict], dict]:
    """Retrieve k passages. Returns (hits, the plan that produced them).

    `searcher` is anything with search(query, k, where) -- Dense, Lexical or
    Hybrid from src/retrieve.py. Routing is about WHICH searches to run, not
    about how a single search works, so the two steps compose instead of one
    being rewritten for the other.

    With route=False this is the Step 5 behaviour: one unfiltered search. That
    is kept as the control, so the change can be measured rather than assumed.

    `planner` is the function that decides the plan, defaulting to `plan`
    above. Step 11 replaces the word list with a model, and that model is a
    different planner rather than an edit to this one -- so the swap is an
    argument, and every number already in results/ stays reproducible by
    leaving the argument out. Anything with plan's signature works:
    (question, prev_companies) -> dict with mode, companies and why.
    """
    decide = planner or plan
    p = (decide(question, prev_companies) if route
         else {"mode": "single", "companies": [], "why": "routing disabled"})

    if p["mode"] == "clarify":
        # Nothing is searched. A question the router could not resolve to a
        # company has no right answer to retrieve, and searching anyway is the
        # failure this mode exists to prevent: an unfiltered search would land
        # in one filer's neighbourhood and answer confidently about whichever
        # company happened to rank. The caller asks the user instead.
        return [], p

    if p["mode"] == "single":
        return searcher.search(question, k), p

    if p["mode"] == "filtered":
        return searcher.search(question, k,
                               {"ticker": p["companies"][0]}), p

    if p["mode"] == "followup":
        # One filtered search per company already under discussion, plus one
        # unfiltered search, merged round-robin by rank. The merge is the whole
        # point: neither lane can take every slot, so a follow-up about the
        # current filer still works and a question that was actually about the
        # whole corpus is not silently narrowed to one company.
        #
        # The corpus lane goes last so the carried company wins rank 1 -- the
        # tie-break favours continuing the conversation, which is what a
        # follow-up usually means, without excluding anything.
        lanes = [(tk, searcher.search(question, k, {"ticker": tk}))
                 for tk in p["companies"]]
        lanes.append(("corpus", searcher.search(question, k)))
        p["lanes"] = [name for name, _ in lanes]

        merged, seen = [], set()
        for rank in range(k):
            for _name, lst in lanes:
                if rank < len(lst) and lst[rank]["chunk_id"] not in seen:
                    seen.add(lst[rank]["chunk_id"])
                    merged.append(lst[rank])
        return merged[:k], p

    # Comparative: one REWRITTEN search per company, then round-robin by rank
    # so every company is represented in the first few slots. Each company is
    # asked for enough passages that the merge can fill k even if one is short.
    per = max(4, -(-k // len(p["companies"])) + 1)
    subs = {tk: subquery(question, tk) for tk in p["companies"]}
    lists = [searcher.search(subs[tk], per, {"ticker": tk})
             for tk in p["companies"]]
    p["subqueries"] = subs

    merged, seen = [], set()
    for rank in range(per):
        for lst in lists:
            if rank < len(lst) and lst[rank]["chunk_id"] not in seen:
                seen.add(lst[rank]["chunk_id"])
                merged.append(lst[rank])
    p["per_company_k"] = per
    return merged[:k], p
