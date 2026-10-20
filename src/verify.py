"""Step 13 — deterministic numeric verification. No model, no API call.

A financial answer is mostly numbers, so the cheapest useful guardrail is
arithmetic rather than judgement. Two checks, in order of cost:

  1. GROUNDING. Every figure in the answer must appear in the passages the
     model was given. A figure that does not is one the model produced itself.
  2. CORROBORATION. A figure may additionally match a value the company tagged
     and filed in XBRL, which is independent of both the model and the parser.

The first check is the one that matters, and Step 7 is why. Three of the four
wrong numeric answers in the baseline were not misread figures -- they were
COMPUTED. Asked for First American's total revenue the model found a table with
five revenue columns, summed three of them, reported $6,810.3m against a true
$7,452.2m, and showed its working, which made it read as more careful than a
correct answer would. Asked for income tax expense it summed a tax rate
reconciliation. Asked for Stewart's operating cash flow it added a $70.1m
increase to an unrelated $23.5m.

A computed figure is absent from the evidence by construction. So "does this
number appear in the passages the model was given?" catches exactly that class
of error, costs nothing, and has no false positives of its own: if the figure
is in the evidence, no flag is raised whatever else may be wrong with the
answer. Measured against the baseline answers, it flags all three fabrications
and does not flag the one wrong answer that was a retrieval failure rather than
a fabrication.

What it deliberately does NOT do is judge whether the right figure was chosen.
That is a different error -- reading a real number from the wrong row -- and it
belongs to retrieval, which recall@k already measures.
"""

import json
import re
from functools import lru_cache

from src.config import TICKERS, XBRL

# A figure, with the scale word that may follow it. Four-digit bare integers in
# calendar range are excluded as years, the same trap the eval scorer hit.
# The trailing boundary is a negative lookahead, not \b. With \b, a figure
# followed by a letter -- "$2,762.7m", the shorthand filings and analysts use --
# made the regex backtrack and capture "2,762" instead of "2,762.7", so a real
# figure was reported as absent from the evidence.
NUM_RE = re.compile(
    r"\$?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{3,})\s*"
    r"(thousands?|millions?|billions?|bn|[mkb])?(?![\d.,])", re.I)

SCALE = {"": 1.0,
         "thousand": 1e3, "thousands": 1e3, "k": 1e3,
         "million": 1e6, "millions": 1e6, "m": 1e6,
         "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9}

# Figures this close in relative terms are treated as the same value, which
# covers a filing printing 2,921,636 and an answer saying $2.92 billion.
TOLERANCE = 0.005


def looks_like_year(raw: str, word: str) -> bool:
    if word or "." in raw or "," in raw:
        return False
    return len(raw) == 4 and 1900 <= int(raw) <= 2100


def figures(text: str) -> list[dict]:
    """Every figure in the text, with its scale resolved to dollars."""
    out = []
    for m in NUM_RE.finditer(text or ""):
        raw, word = m.group(1), (m.group(2) or "").lower()
        if looks_like_year(raw, word):
            continue
        try:
            value = float(raw.replace(",", "")) * SCALE.get(word, 1.0)
        except ValueError:
            continue
        start = max(0, m.start() - 60)
        out.append({
            "raw": m.group(0).strip(),
            "number": raw,
            "scale": word or None,
            "value": value,
            "context": text[start:m.end() + 20].replace("\n", " ").strip(),
        })
    return out


def renderings(number: str) -> set[str]:
    """How this figure might be printed in a filing: as given, without commas,
    and at a few decimal places. The scale is NOT varied here -- a figure
    written '2,921,636' in a passage and '$2.92 billion' in an answer are the
    same value, and the value comparison below handles that; this set exists
    for literal string presence."""
    bare = number.replace(",", "")
    out = {number, bare}
    try:
        v = float(bare)
    except ValueError:
        return out
    for dec in (0, 1, 2):
        out.add(f"{v:,.{dec}f}")
        out.add(f"{v:.{dec}f}")
    return {s for s in out if len(re.sub(r"\D", "", s)) >= 3}


SCALE_DECL_RE = re.compile(r"figures in (thousand|million|billion)s?"
                           r"|in\s*[$]?\s*(thousand|million|billion)s?", re.I)


def passage_scale(text: str) -> float:
    """The scale a passage declares for its own bare figures.

    Step 8 writes "Figures in millions." into every table chunk whose caption
    declared a scale, which is what makes this possible: the magnitude travels
    with the numbers instead of having to be inferred from them.
    """
    m = SCALE_DECL_RE.search(text or "")
    if not m:
        return 1.0
    word = (m.group(1) or m.group(2) or "").lower()
    return SCALE.get(word, 1.0)


def why_grounded(fig: dict, passages: list[str]) -> str | None:
    """Which check grounded this figure, or None. Lets a flag be explained."""
    forms = renderings(fig["number"])
    joined = "\n".join(passages)
    for f in forms:
        if re.search(rf"(?<![\d.,]){re.escape(f)}(?![\d.,])", joined):
            return f"printed as '{f}'"
    target = fig["value"]
    if target:
        for p in passages:
            mult = passage_scale(p)
            for other in figures(p):
                value = other["value"] * (1.0 if other["scale"] else mult)
                if abs(value - target) <= TOLERANCE * abs(target):
                    return (f"equals '{other['raw']}' at the passage scale "
                            f"(x{mult:,.0f})")
    return None


def in_passages(fig: dict, passages: list[str]) -> bool:
    """Is this figure printed in the evidence, or equal to a figure that is?

    Two ways to match, because a filing reports in thousands or millions while
    an answer may restate the same value at another scale. The literal check
    catches '2,921,636' appearing as written; the value check catches an answer
    saying '$2.92 billion' about it.
    """
    forms = renderings(fig["number"])
    joined = "\n".join(passages)
    # Boundaries matter. A plain substring test let a transposed figure pass:
    # "7,542.2" was not flagged because "7,542" occurs inside a longer number
    # somewhere in the evidence. This is the same trap the Step 6 gold-label
    # matcher hit, where 2,490 matched inside 12,490,318.
    if any(re.search(rf"(?<![\d.,]){re.escape(f)}(?![\d.,])", joined)
           for f in forms):
        return True
    target = fig["value"]
    if not target:
        return False
    for p in passages:
        mult = passage_scale(p)
        for other in figures(p):
            # A bare figure in a passage is expressed at that passage's
            # declared scale, so it is resolved once rather than tried at
            # every scale. Sweeping x1, x1e3, x1e6 and x1e9 over every figure
            # in every passage was the first version, and at a 0.5 per cent
            # tolerance it gave so many chances to match that a transposed
            # figure -- 7,542.2 against a true 7,452.2 -- passed as grounded.
            value = other["value"] * (1.0 if other["scale"] else mult)
            if abs(value - target) <= TOLERANCE * abs(target):
                return True
    return False


@lru_cache(maxsize=4)
def tagged_values(ticker: str) -> frozenset:
    """Every USD value this company has filed, as a set.

    REPORTED BUT NOT TRUSTED, and the reason is measured. This started out as a
    second line of evidence: a figure absent from the retrieved passages might
    still be corroborated by matching something the company had tagged and
    filed. It made the verifier useless. Flagging dropped to zero, including on
    the three answers known to contain fabricated figures.

    First American has filed 11,304 distinct USD values (Fidelity 5,257,
    Stewart 8,894). Against a set that dense, a 0.5 per cent tolerance matches
    almost anything: of 2,000 random figures drawn between $1m and $10bn,
    **99 per cent matched some value First American had filed**. "Corroborated
    by XBRL" therefore carries essentially no information, while sounding like
    it carries a great deal -- and it was suppressing every true flag.

    Deciding which concept a sentence is about is a semantic problem, and this
    module is the deterministic layer. So the flag now rests on grounding
    alone, and this is surfaced as a note rather than as a status.
    """
    f = XBRL / f"{ticker}_companyfacts.json"
    if not f.exists():
        return frozenset()
    data = json.loads(f.read_text())
    vals = set()
    for tax in data.get("facts", {}).values():
        for concept in tax.values():
            for unit, facts in concept.get("units", {}).items():
                if unit != "USD":
                    continue
                for fact in facts:
                    v = fact.get("val")
                    if isinstance(v, (int, float)):
                        vals.add(float(v))
    return frozenset(vals)


def near_filed_value(fig: dict, tickers: list[str]) -> str | None:
    """Does this figure match a value one of these companies filed?

    Informational only -- see tagged_values for why this is not evidence.
    """
    target = fig["value"]
    if not target:
        return None
    for tk in tickers or TICKERS:
        for v in tagged_values(tk):
            if abs(v - target) <= TOLERANCE * abs(target):
                return tk
    return None


def verify(answer: str, passages: list[str],
           tickers: list[str] | None = None) -> dict:
    """Check every figure in an answer. Returns a report, never raises.

    `status` per figure:
      grounded    -- printed in the evidence the model was given
      unsupported -- not in the evidence. The model produced this number itself.

    The status is binary on purpose. An earlier version let a match against
    filed XBRL values rescue an ungrounded figure, and that took flagging to
    zero -- 99 per cent of plausible figures match something a filer has
    tagged. `near_filed` is still reported per figure, clearly as a note rather
    than as support.
    """
    figs = figures(answer)
    checked = []
    for f in figs:
        why = why_grounded(f, passages)
        if why:
            f["status"] = "grounded"
            f["source"] = why
        else:
            f["status"] = "unsupported"
            f["source"] = None
            f["near_filed"] = near_filed_value(f, tickers or [])
        checked.append(f)

    unsupported = [f for f in checked if f["status"] == "unsupported"]
    return {
        "figures": checked,
        "n_figures": len(checked),
        "n_grounded": sum(1 for f in checked if f["status"] == "grounded"),
        "n_unsupported": len(unsupported),
        "unsupported": [f["raw"] for f in unsupported],
        "verdict": "flagged" if unsupported else "clean",
        "note": (f"{len(unsupported)} figure(s) do not appear in the passages "
                 f"the model was given, so it produced them itself. A computed "
                 f"total is absent from its own evidence by construction."
                 if unsupported else
                 "Every figure in this answer is printed in the evidence."),
    }
