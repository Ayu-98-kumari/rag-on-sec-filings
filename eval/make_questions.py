"""Step 6 — build the evaluation test set.

    python -m eval.make_questions

Writes eval/questions.csv: 30 questions whose correct answers are known BEFORE
the system is ever run against them. Nothing here calls a language model; this
step is pure data preparation. Step 7 does the asking and the scoring.

Two sources of truth, deliberately kept apart:

  * Numeric lookups are generated from SEC's XBRL `companyfacts` API. The
    expected answer is the value the company itself tagged and filed. Nobody
    reads a filing and types a number in, so the answer key cannot inherit a
    transcription error -- and it cannot be accused of having been written to
    flatter the system.
  * Conceptual, comparative and unanswerable questions are hand-written,
    because no API knows which risks a filing discusses or which question is
    plausible-but-absent. These are listed in full below so they can be
    audited.

Gold passages are labelled by BLOCK id, never chunk id. A chunk id encodes
position within one chunking run, and Step 8 re-chunks the whole corpus, so
chunk-keyed labels would make the baseline and improved scores incomparable --
destroying the one controlled experiment the project rests on. Block ids come
from parse order, and Step 8 does not re-parse.
"""

import csv
import json
import re

from src.config import PARSED, ROOT, TICKERS, XBRL

OUT = ROOT / "eval" / "questions.csv"

def poss(name: str) -> str:
    """Possessive that reads correctly for a name already ending in s."""
    return name + ("'" if name.endswith("s") else "'s")


COMPANY = {"FAF": "First American Financial",
           "FNF": "Fidelity National Financial",
           "STC": "Stewart Information Services"}

# Concepts chosen on two criteria: present as an annual USD fact for all three
# filers (32 qualify; these are the ones a human would actually ask about), and
# reported as a line item in the statements, so a gold passage exists to find.
LOOKUP_CONCEPTS = [
    ("Revenues", "total revenues"),
    ("NetIncomeLoss", "net income attributable to the parent"),
    ("IncomeTaxExpenseBenefit", "income tax expense"),
    ("NetCashProvidedByUsedInOperatingActivities", "net cash provided by operating activities"),
    ("ShareBasedCompensation", "share-based compensation expense"),
]
LOOKUP_YEARS = ["2025"]      # 5 concepts x 3 companies x 1 year = 15 questions

# A lookup is scored as correct within this relative tolerance, because an
# answer of "$2.92 billion" is right and "2921636000" is not the only phrasing.
NUMERIC_TOLERANCE = 0.01


# --------------------------------------------------------------------------
# XBRL: extract one unambiguous annual value per (company, concept, year)
# --------------------------------------------------------------------------

def annual_facts(concept: dict) -> dict:
    """Return {fiscal_year: fact} for full-year USD facts from 10-K filings.

    The same fiscal year is reported more than once in companyfacts: in its own
    10-K, and again in the NEXT year's 10-K as the prior-year comparative, under
    a different `fy` and `accn`. STC's FY2024 revenue of 2,490,419,000 appears
    under both fy 2024 (accn ...-25-000004) and fy 2025 (accn ...-26-000007).
    Reading the list naively therefore yields two "correct" answers to one
    question -- and if the figure was restated between filings, two DIFFERENT
    correct answers.

    So: key on the reporting period (start, end), which is the thing the
    question actually asks about, and keep the earliest accession, i.e. the
    value as originally filed rather than as later restated.
    """
    best: dict[tuple, dict] = {}
    for f in concept.get("units", {}).get("USD", []):
        if f.get("form") != "10-K" or not f.get("start"):
            continue
        start, end = f["start"], f["end"]
        if start[5:] != "01-01" or end[5:] != "12-31" or start[:4] != end[:4]:
            continue                      # quarterly or stub period
        key = (start, end)
        if key not in best or f["accn"] < best[key]["accn"]:
            best[key] = f
    return {end[:4]: f for (start, end), f in best.items()}


# --------------------------------------------------------------------------
# Gold passages: find where in the filing the figure is actually printed
# --------------------------------------------------------------------------

SCALE_DIV = {"thousand": 1e3, "million": 1e6, "billion": 1e9, None: None}
CELL_CLEAN = re.compile(r"[\s$,()%]")


def at_scale(val: float, div: float) -> set[str]:
    """How this figure is printed by a table declaring that scale."""
    x = abs(val) / div
    return {f"{x:,.0f}", f"{x:,.1f}", f"{x:,.2f}"}


def any_scale(val: float, min_digits: int = 5) -> set[str]:
    """Fallback for blocks that declare no scale: try every scale, but only
    accept renderings specific enough to identify a line item. Four digits
    would let 2,490 match an unrelated figure; five is the floor."""
    out = set()
    for div in (1, 1e3, 1e6, 1e9):
        out |= at_scale(val, div)
    return {s for s in out if len(re.sub(r"\D", "", s)) >= min_digits}


# "$950.8 million" in a sentence is as much a statement of the figure as a cell
# in the cash-flow statement, and MD&A states most headline figures this way.
# The five-digit floor above rejected it -- 950.8 has four digits -- so every
# figure written in prose as "$X.X million" was missing from the gold labels,
# and retrieval that found the right sentence was scored as a miss.
PROSE_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def prose_forms(val: float) -> list[tuple[str, str]]:
    """(number, scale word) pairs as a sentence would write this figure."""
    out = []
    for word, div in PROSE_SCALES.items():
        x = abs(val) / div
        if 0.5 <= x < 1_000_000:
            for dec in (0, 1, 2):
                out.append((f"{x:,.{dec}f}", word))
    return out


def prose_states_figure(text: str, val: float) -> bool:
    """True if the sentence writes this figure with its scale word attached."""
    for num, word in prose_forms(val):
        if re.search(rf"(?<![\d.,]){re.escape(num)}(?![\d.,])\s*{word}s?\b",
                     text, re.I):
            return True
    return False


def cell_is(cell: str, forms: set[str]) -> bool:
    """A table cell matches only if the WHOLE cell is the figure.

    Substring matching would make 2,490 match 12,490,318.
    """
    c = CELL_CLEAN.sub("", cell)
    return any(c == CELL_CLEAN.sub("", f) for f in forms)


def find_gold_blocks(doc: dict, val: float, limit: int = 8) -> tuple[list[str], str]:
    """Locate the blocks that print this figure.

    The figure in XBRL is in dollars; the figure in the document is not. FAF
    and STC report their statements in thousands, FNF in millions, so the
    digits never match directly and the scale has to be undone first. Guessing
    the scale was the first attempt and it failed on FNF for all five concepts:
    a rendering filter that demanded enough digits to be unambiguous rejected
    FNF's millions form outright.

    The fix uses what Step 2 already recorded. Every parsed table carries the
    `unit_scale` its own caption declared, so each table is searched at ITS
    declared scale -- exact, no guessing. Only blocks with no declared scale
    fall back to trying every scale, and those must match a rendering of at
    least five digits to count.
    """
    exact, loose = [], []
    for b in doc["blocks"]:
        scale = b.get("unit_scale")
        if b["type"] == "table":
            cells = [c for row in b["rows"] for c in row]
            div = SCALE_DIV.get(scale)
            if div:
                if any(cell_is(c, at_scale(val, div)) for c in cells):
                    exact.append(b["block_id"])
            elif any(cell_is(c, any_scale(val)) for c in cells):
                loose.append(b["block_id"])
        else:
            txt = b.get("text", "")
            if prose_states_figure(txt, val):
                exact.append(b["block_id"])
            elif any(re.search(rf"(?<![\d.,]){re.escape(f)}(?![\d.,])", txt)
                     for f in any_scale(val)):
                loose.append(b["block_id"])

    if exact:
        return (exact + loose)[:limit], "declared_scale"
    if loose:
        return loose[:limit], "inferred_scale"
    return [], "none"


# --------------------------------------------------------------------------
# Hand-written questions
# --------------------------------------------------------------------------
# Gold blocks for these are resolved by keyword search below rather than typed
# in by hand, so they stay correct if the parser changes. `must_contain` is a
# set of phrases a passage must carry to count as a gold passage.

CONCEPTUAL = [
    dict(q="What risks does Stewart Information Services identify relating to interest rates?",
         ticker="STC",
         anchors=[["mortgage interest rates", "real estate"],
                  ["interest rates", "line of credit"]],
         expected="Rising mortgage interest rates depress the volume of residential and "
                  "commercial real estate transactions, so the title industry sees reduced "
                  "revenues and earnings and potentially increased claims experience. Higher "
                  "rates also increase the cost of borrowing on the floating rate line of credit."),
    dict(q="How does First American account for title insurance loss reserves?",
         ticker="FAF",
         anchors=[["loss provision rate", "incurred but not reported"]],
         expected="Losses are provided for by a charge to expense when the related premium "
                  "revenue is recognised, determined by applying a loss provision rate to total "
                  "title insurance premiums and escrow fees. Management sets the rate at the "
                  "start of each year and reassesses it quarterly, informed by an in-house "
                  "actuarial review, so that the known claims and IBNR reserves together reflect "
                  "its best estimate of the total cost to settle all claims; an adjustment is "
                  "recorded if the ending IBNR reserve is not adequate."),
    dict(q="What does Fidelity National Financial describe as its principal business segments?",
         ticker="FNF",
         anchors=[["this segment", "title insurance underwriters"],
                  ["F&G", "annuities"],
                  ["corporate and other"]],
         expected="Three segments. Title: the title insurance underwriters and related "
                  "businesses, including escrow and other title-related services, trust "
                  "activities, trustee sale guarantees, home warranty and transaction services. "
                  "F&G: annuities and life insurance, including deferred and immediate annuities "
                  "and IUL, plus funding agreements and PRTs. Corporate and Other: the parent "
                  "holding company, real estate technology subsidiaries, other smaller non-title "
                  "businesses and unallocated corporate overhead."),
    dict(q="What cybersecurity risk management processes does Stewart describe?",
         ticker="STC",
         anchors=[["cybersecurity", "incident response"],
                  ["board", "cybersecurity"]],
         expected="A risk-based programme covering risk management, data protection, incident "
                  "response, identity and access management, threat and vulnerability management, "
                  "disaster recovery and business continuity, with third-party risk monitored "
                  "continuously. Cybersecurity risk is assessed inside the enterprise risk "
                  "management programme, assigned to the Vice President of IT and overseen by the "
                  "Senior Leadership Team. Governance reaches the board: the Board receives a "
                  "cybersecurity risk report at each regularly scheduled meeting, oversight runs "
                  "through the Audit Committee, and the CISO reports quarterly."),
    dict(q="How does First American describe the role of its title plants?",
         ticker="FAF",
         anchors=[["title plants", "principal assets"],
                  ["title plants", "public records"]],
         expected="Title plants are described as one of the company's principal assets. They hold "
                  "information abstracted from public records but arrange it geographically rather "
                  "than by the names of the parties to a document, which is how public records are "
                  "indexed, so title searches can be run more efficiently. Many title plants also "
                  "index prior title insurance policies, adding further searching efficiency, and "
                  "they support operations beyond title insurance."),
    dict(q="What does Fidelity National Financial say about competition in the title insurance market?",
         ticker="FNF",
         anchors=[["competition in the title insurance industry"]],
         expected="Competition is based primarily on service and price, and the number and size of "
                  "competing companies varies by geographic area. Named competitors include other "
                  "major underwriters - First American, Old Republic, Stewart, Westcor Land Title, "
                  "Title Resources Guaranty and WFG National Title - alongside numerous regional "
                  "title insurers, underwritten title companies and independent agency operations "
                  "at regional and local level."),
    dict(q="What factors does Stewart say drive its commercial title revenue?",
         ticker="STC",
         anchors=[["commercial", "fee per file"],
                  ["premiums", "values of the transactions"]],
         expected="The size of commercial transactions closed, more than their number: average "
                  "domestic commercial fee per file was $27,300 in the fourth quarter of 2025 "
                  "against $3,300 for residential, and the 2025 improvement was driven principally "
                  "by larger transactions in the data centre and energy asset classes. Premiums are "
                  "determined in part by the values of the transactions handled, and commercial "
                  "closings are skewed toward the end of the year although large individual deals "
                  "can close at any time."),
    dict(q="What does First American disclose about its escrow deposits and trust assets?",
         ticker="FAF",
         anchors=[["escrow deposits", "not considered assets"],
                  ["trust assets", "fiduciary"]],
         expected="Escrow deposits are administered as a service to customers of the direct title "
                  "operations and totalled $9.3 billion at 31 December 2025 against $8.9 billion a "
                  "year earlier, of which $3.7 billion was held at the company's own First American "
                  "Trust and the remainder at third-party financial institutions. Deposits held at "
                  "third-party institutions are not considered assets of the Company and are not "
                  "included in the consolidated balance sheets. FA Trust separately administered "
                  "$5.6 billion of trust assets, which are fiduciary client assets and likewise off "
                  "balance sheet, leaving the Company contingently liable."),
]

COMPARATIVE = [
    dict(q="Which had higher total revenues in fiscal 2025, First American or Stewart Information Services?",
         concept="Revenues", year="2025", tickers=["FAF", "STC"], higher=True),
    dict(q="Which had higher net income attributable to the parent in fiscal 2025, First American or Stewart Information Services?",
         concept="NetIncomeLoss", year="2025", tickers=["FAF", "STC"], higher=True),
    dict(q="Compare net cash provided by operating activities in fiscal 2025 for First American and Stewart Information Services.",
         concept="NetCashProvidedByUsedInOperatingActivities", year="2025", tickers=["FAF", "STC"], higher=None),
    dict(q="Which of the three title insurers reported the lowest total revenues in fiscal 2025?",
         concept="Revenues", year="2025", tickers=["FAF", "FNF", "STC"], higher=False),
]

# Plausible, correctly formed, and genuinely absent. Each is absent for a
# stated reason -- outside the corpus years, or a figure companies do not
# disclose in this form. Absurd questions would make abstention look better
# than it is.
#
# Checked against the corpus rather than assumed, and each one turned out to
# have a DISTRACTOR: a block carrying the right year and a plausible number of
# the wrong quantity. That makes these hard abstentions rather than easy ones
# -- retrieval will return something that looks relevant, and the model has to
# notice the quantity is wrong rather than simply find nothing.
UNANSWERABLE = [
    dict(q="What was Stewart Information Services' total revenue in fiscal 2019?",
         why="Outside corpus coverage (FY2024, FY2025, Q2 FY2026 only). DISTRACTOR: "
             "STC's FY2024 10-K stock performance graph has a 2019 column, but it is "
             "indexed total shareholder return (100.00), not revenue."),
    dict(q="What is First American's projected title premium revenue for fiscal 2027?",
         why="Forward-looking figure companies do not disclose; no guidance of this kind "
             "is in the filings. DISTRACTOR: FAF's 10-Q has a maturity schedule listing "
             "'2027 $15.4 million', which is a payment schedule, not premium revenue."),
    dict(q="How many employees did Fidelity National Financial's F&G segment have in fiscal 2023?",
         why="Both outside corpus years and below the segment granularity reported. "
             "DISTRACTOR: generic personnel-cost paragraphs mention employees but give "
             "no headcount for any segment or year."),
]


def find_text_blocks(doc: dict, anchors: list[list[str]], per_anchor: int = 3):
    """Gold passages for a prose question.

    `anchors` is a list of phrase groups. A block matches a group if it carries
    every phrase in that group, and the gold set is the union across groups,
    capped per group.

    One group per question was the first design and it was too blunt. A single
    keyword like "competit" matched Fidelity's passages on competitive margins
    and competitive advantage while missing its actual Competition section, and
    a single "cybersecurity" anchor returned Stewart's programme description but
    stopped before the board-governance passages that the reference answer also
    describes. Several narrow groups say what the answer is actually made of.
    """
    gold: list[str] = []
    for group in anchors:
        n = 0
        for b in doc["blocks"]:
            if b["type"] == "table" or b["block_id"] in gold:
                continue
            txt = b.get("text", "").lower()
            if len(txt) < 200:
                continue                   # too short to be the answer
            if all(p.lower() in txt for p in group):
                gold.append(b["block_id"])
                n += 1
                if n >= per_anchor:
                    break
    return gold


def main() -> None:
    facts, docs = {}, {}
    for tk in TICKERS:
        facts[tk] = json.loads((XBRL / f"{tk}_companyfacts.json").read_text())
        for fy in LOOKUP_YEARS + ["2024"]:
            p = PARSED / f"{tk}_10K_{fy}-12-31.json"
            if p.exists():
                docs[(tk, fy)] = json.loads(p.read_text())

    rows, dropped = [], []

    # ---- 1. numeric lookups, generated from XBRL -------------------------
    n = 0
    for concept, phrase in LOOKUP_CONCEPTS:
        for tk in TICKERS:
            c = facts[tk]["facts"]["us-gaap"].get(concept)
            if not c:
                dropped.append(f"{tk} {concept}: not reported"); continue
            by_year = annual_facts(c)
            for fy in LOOKUP_YEARS:
                if fy not in by_year:
                    dropped.append(f"{tk} {concept} FY{fy}: no annual fact"); continue
                f = by_year[fy]
                gold, tier = find_gold_blocks(docs[(tk, fy)], f["val"])
                n += 1
                rows.append(dict(
                    qid=f"L{n:03d}", type="lookup",
                    question=f"What was {poss(COMPANY[tk])} {phrase} in fiscal {fy}?",
                    ticker=tk, fiscal_year=fy,
                    expected_answer=f["val"],
                    answer_source=f"xbrl:us-gaap:{concept}",
                    accession=f["accn"],
                    gold_block_ids=";".join(gold),
                    gold_match=tier,
                    tolerance=NUMERIC_TOLERANCE,
                    notes=f"period {f['start']}..{f['end']}; as originally filed",
                ))

    # ---- 2. conceptual ---------------------------------------------------
    for i, q in enumerate(CONCEPTUAL, start=1):
        tk = q["ticker"]
        gold = find_text_blocks(docs[(tk, "2025")], q["anchors"])
        rows.append(dict(
            qid=f"C{i:03d}", type="conceptual", question=q["q"],
            ticker=tk, fiscal_year="2025",
            expected_answer=q["expected"], answer_source="hand-written",
            accession="", gold_block_ids=";".join(gold),
            gold_match="keyword" if gold else "none", tolerance="",
            notes="scored by LLM judge in Step 7; retrieval scored on gold blocks",
        ))

    # ---- 3. comparative --------------------------------------------------
    for i, q in enumerate(COMPARATIVE, start=1):
        vals, gold = {}, []
        for tk in q["tickers"]:
            c = facts[tk]["facts"]["us-gaap"].get(q["concept"])
            v = annual_facts(c).get(q["year"]) if c else None
            if not v:
                continue
            vals[tk] = v["val"]
            g, _ = find_gold_blocks(docs[(tk, q["year"])], v["val"], limit=2)
            gold += g
        if len(vals) < 2:
            dropped.append(f"comparative {q['concept']}: fewer than 2 values"); continue
        if q["higher"] is True:
            exp = f"{COMPANY[max(vals, key=vals.get)]} " + \
                  "(" + "; ".join(f"{COMPANY[t]} {vals[t]:,}" for t in q['tickers'] if t in vals) + ")"
        elif q["higher"] is False:
            exp = f"{COMPANY[min(vals, key=vals.get)]} " + \
                  "(" + "; ".join(f"{COMPANY[t]} {vals[t]:,}" for t in q['tickers'] if t in vals) + ")"
        else:
            exp = "; ".join(f"{COMPANY[t]} {vals[t]:,}" for t in q["tickers"] if t in vals)
        rows.append(dict(
            qid=f"M{i:03d}", type="comparative", question=q["q"],
            ticker="+".join(q["tickers"]), fiscal_year=q["year"],
            expected_answer=exp, answer_source=f"xbrl:us-gaap:{q['concept']}",
            accession="", gold_block_ids=";".join(gold),
            gold_match="declared_scale", tolerance=NUMERIC_TOLERANCE,
            notes="needs every named company retrieved; one vector search over an "
                  "FNF-heavy corpus is expected to fail here (motivates Step 10)",
        ))

    # ---- 4. unanswerable -------------------------------------------------
    for i, q in enumerate(UNANSWERABLE, start=1):
        rows.append(dict(
            qid=f"U{i:03d}", type="unanswerable", question=q["q"],
            ticker="", fiscal_year="",
            expected_answer="ABSTAIN", answer_source="by construction",
            accession="", gold_block_ids="", gold_match="n/a", tolerance="",
            notes=q["why"],
        ))

    # ---- 5. dev/test split ----------------------------------------------
    # Stratified and deterministic: within each type, every 5th question in
    # order goes to dev. Tuning happens on dev only; the test rows are not
    # looked at until Step 7 reports, so an improvement cannot be a fit.
    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    for t, group in by_type.items():
        n_dev = round(0.4 * len(group))
        for i, r in enumerate(group):
            r["split"] = "dev" if i < n_dev else "test"

    # n_gold is recorded because recall@k counts a question as a hit if ANY of
    # its gold blocks is retrieved. A question with eight gold blocks is easier
    # to hit than one with a single block, so the count has to be visible
    # alongside the score rather than hidden inside it.
    for r in rows:
        r["n_gold"] = len([x for x in r["gold_block_ids"].split(";") if x])

    cols = ["qid", "type", "split", "question", "ticker", "fiscal_year",
            "expected_answer", "answer_source", "accession", "gold_block_ids",
            "n_gold", "gold_match", "tolerance", "notes"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # ---- verification ----------------------------------------------------
    print(f"wrote {len(rows)} questions -> {OUT.relative_to(ROOT)}\n")
    print(f"{'type':14s} {'n':>3s} {'dev':>4s} {'test':>5s} {'with gold':>10s}")
    print("-" * 42)
    for t in ("lookup", "conceptual", "comparative", "unanswerable"):
        g = by_type.get(t, [])
        if not g:
            continue
        print(f"{t:14s} {len(g):3d} "
              f"{sum(1 for r in g if r['split']=='dev'):4d} "
              f"{sum(1 for r in g if r['split']=='test'):5d} "
              f"{sum(1 for r in g if r['gold_block_ids']):10d}")
    print("-" * 42)
    print(f"{'TOTAL':14s} {len(rows):3d} "
          f"{sum(1 for r in rows if r['split']=='dev'):4d} "
          f"{sum(1 for r in rows if r['split']=='test'):5d}")

    sizes = [r["n_gold"] for r in rows if r["type"] != "unanswerable"]
    print(f"\n  gold blocks per answerable question: min {min(sizes)}, "
          f"max {max(sizes)}, mean {sum(sizes)/len(sizes):.1f}")
    need_gold = [r for r in rows if r["type"] != "unanswerable"]
    missing = [r["qid"] for r in need_gold if not r["gold_block_ids"]]
    print("\n=== verification ===")
    print(f"  duplicate qids                        : "
          f"{len(rows) - len({r['qid'] for r in rows})}  (must be 0)")
    print(f"  duplicate questions                   : "
          f"{len(rows) - len({r['question'] for r in rows})}  (must be 0)")
    print(f"  answerable questions with no gold block: {len(missing)}  "
          f"{missing if missing else '(must be 0)'}")
    tiers: dict[str, int] = {}
    for r in rows:
        tiers[r["gold_match"]] = tiers.get(r["gold_match"], 0) + 1
    print(f"  gold match quality                    : "
          f"{', '.join(f'{k}={v}' for k, v in sorted(tiers.items()))}")
    if dropped:
        print(f"\n  dropped ({len(dropped)}):")
        for d in dropped:
            print(f"    - {d}")


if __name__ == "__main__":
    main()
