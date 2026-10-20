"""Step 5 (half two) — question in, cited answer out. The first end-to-end path.

    python -m src.ask "what drives title insurance loss ratios"
    python -m src.ask "FNF total revenues" --ticker FNF --k 3
    python -m src.ask "..." --show-prompt        # see exactly what the model got

Five moves: embed the question, retrieve k chunks, build a prompt, call the
model, print the answer above the evidence it was given.

This is the BASELINE. The prompt *asks* the model to stay inside the passages
and cite them; nothing here *checks* that it did. That gap is deliberate and it
is the whole subject of Phase 4 — Step 12 verifies the citations resolve, Step
13 re-derives the numbers from XBRL, Step 14 scores groundedness, Step 15
enforces abstention. Every one of those steps is measured against the numbers
this step produces, which is why the honest weak version has to exist first.
"""

import argparse
import json
import time

from src import config, embed, llm, retrieve as retrieve_mod, route as route_mod
from src.config import CHROMA, DEFAULT_STRATEGY, K
from src.store import get_store

SYSTEM = (
    "You are a financial analyst answering questions about SEC filings. "
    "You answer ONLY from the numbered passages you are given.\n"
    "Rules:\n"
    "1. Use only the passages. Do not use outside knowledge about these "
    "companies, and do not estimate or infer figures that are not written.\n"
    "2. Cite the passage number in square brackets after every claim, "
    "like [2].\n"
    "3. If the passages do not contain the answer, reply exactly: "
    "'Not in the provided filings.' Do not guess.\n"
    "4. These companies are competitors with near-identical wording. Always "
    "name which company a figure belongs to, and never merge figures from "
    "different companies or fiscal years.\n"
    "5. Be concise: a few sentences, not an essay."
)

# Appended to SYSTEM only when a conversation history is actually passed, so a
# single-turn question sends the byte-identical system prompt it always has.
# That is not tidiness: llm._cache_key hashes the system prompt, so editing
# SYSTEM in place would invalidate every cached answer and silently make the
# evaluation results irreproducible.
HISTORY_RULE = (
    "\n6. EARLIER CONVERSATION is provided for reference only, to resolve what "
    "words like 'it', 'they' or 'that year' refer to. It is NOT evidence. "
    "Never cite it, and never repeat a figure from it -- if a figure is not in "
    "the numbered passages below, it is not available to you."
)

# How many previous turns to carry. Two is enough to resolve a pronoun and
# cheap: more turns mean more prompt tokens on every question, against a
# 200,000-token daily free-tier budget.
HISTORY_TURNS = 2

# Previous answers are summarised context, not evidence, so a long one is
# truncated rather than allowed to crowd out the passages.
HISTORY_ANSWER_CHARS = 400

# --------------------------------------------------------------------------- #
# structured output
#
# Asked "which one has the highest revenue in 2024", the model produced ~1,000
# tokens of visible deliberation -- "Correction: I must re-evaluate", "Wait, I
# need to check" -- and was cut off mid-sentence by the token cap. The cause was
# not verbosity. The passages genuinely do not contain Fidelity National's 2024
# TOTAL revenue, only the year-over-year change, so the model was right to be
# stuck; it simply had nowhere to put "this figure is not here" except prose.
#
# A length limit would have truncated the same confusion, and would have cost
# real answers: conceptual questions already score four partial out of eight,
# and partial means missing content. So the fix is a shape, not a budget.
# 'unavailable' gives the dead end a field of its own, and constrained decoding
# makes deliberating in prose structurally impossible.
#
# Probed on 2026-10-20: qwen/qwen3.8-27b on Groq accepts strict json_schema,
# so the schema is enforced during generation rather than requested politely.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": (
                "The answer in plain prose, citing passage numbers in square "
                "brackets like [2]. Name the company every figure belongs to. "
                "Decide once: do not revise a statement later in the same "
                "answer. If the passages do not contain the answer, write "
                "exactly: Not in the provided filings."),
        },
        "figures": {
            "type": "array",
            "description": ("Every figure stated in the answer, one entry "
                            "each, taken verbatim from a passage."),
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "enum": list(config.TICKERS)},
                    "metric": {"type": "string",
                               "description": "e.g. total revenues"},
                    "value": {"type": "string",
                              "description": "as written, e.g. 6.1 billion"},
                    "period": {"type": "string",
                               "description": "the fiscal year or quarter"},
                    "passage": {"type": "integer",
                                "description": "the passage number it came from"},
                },
                "required": ["company", "metric", "value", "period", "passage"],
                "additionalProperties": False,
            },
        },
        "unavailable": {
            "type": "array",
            "description": ("Anything the question asks for that the passages "
                            "do not contain. Put it here instead of reasoning "
                            "about it in the answer."),
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "enum": list(config.TICKERS)},
                    "metric": {"type": "string"},
                    "period": {"type": "string"},
                    "reason": {"type": "string",
                               "description": "what the passages give instead"},
                },
                "required": ["company", "metric", "period", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "figures", "unavailable"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "filing_answer", "strict": True,
                    "schema": ANSWER_SCHEMA},
}

# Appended to SYSTEM only on the structured path, for the same cache reason as
# HISTORY_RULE: the unstructured prompt stays byte-identical to the one the
# evaluated numbers were produced with.
STRUCTURED_RULE = (
    "\n6. Reply as JSON matching the schema you are given. Every figure you "
    "state goes in 'figures' with the passage it came from. Anything the "
    "question asks for that the passages do not contain goes in "
    "'unavailable' -- not into 'answer' as deliberation. Rule 3 still holds: "
    "if nothing in the passages answers the question, 'answer' is exactly "
    "'Not in the provided filings.'"
)


def parse_structured(text: str) -> dict | None:
    """The model's JSON, or None if it was not JSON after all.

    Strict mode should make a parse failure impossible, but a provider that
    quietly ignores the schema would otherwise surface as a crash on a public
    page. Returning None lets the caller fall back to treating the output as
    ordinary prose.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or "answer" not in data:
        return None
    data.setdefault("figures", [])
    data.setdefault("unavailable", [])
    return data


def prev_companies(history: list[dict] | None) -> list[str]:
    """Companies the conversation is already about, most recent turn first.

    Taken from the questions rather than from the answers: a question states
    what was asked, whereas an answer may mention a competitor in passing and
    would drag the next follow-up to the wrong filer.
    """
    for t in reversed(history or []):
        q, _ = turn_text(t)
        if not q:
            continue
        named, _why = route_mod.named_companies(q)
        if named:
            return named
    return []


def passage_label(meta: dict) -> str:
    """The identity line the model needs to keep three competitors apart."""
    section = meta.get("section") or "-"
    return (f"{meta.get('company')} ({meta.get('ticker')}) | "
            f"{meta.get('form')} FY{meta.get('fiscal_year')} | Item {section}")


def body_of(text: str) -> str:
    """Chunks carry a header line from Step 3; the label above replaces it."""
    parts = text.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else text


def turn_text(turn: dict) -> tuple[str, str]:
    """One history turn as (question, answer), tolerating either key spelling."""
    q = turn.get("question") or turn.get("q") or ""
    a = turn.get("answer") or turn.get("a") or ""
    return str(q).strip(), str(a).strip()


def history_block(history: list[dict]) -> str:
    """The last HISTORY_TURNS turns, oldest first, or "" if there are none."""
    turns = [t for t in (history or []) if any(turn_text(t))][-HISTORY_TURNS:]
    if not turns:
        return ""
    lines = []
    for t in turns:
        q, a = turn_text(t)
        if len(a) > HISTORY_ANSWER_CHARS:
            a = a[:HISTORY_ANSWER_CHARS].rstrip() + "..."
        lines.append(f"Q: {q}\nA: {a}")
    body = "\n\n".join(lines)
    return f"EARLIER CONVERSATION (reference only, not evidence)\n{'-' * 60}\n{body}\n"


def build_prompt(question: str, hits: list[dict],
                 history: list[dict] | None = None) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] {passage_label(h['metadata'])}\n"
                      f"{body_of(h['text']).strip()}")
    passages = "\n\n".join(blocks)
    prefix = history_block(history)
    if prefix:
        prefix += "\n"
    return (f"{prefix}PASSAGES\n{'=' * 60}\n{passages}\n{'=' * 60}\n\n"
            f"QUESTION: {question}\n\nAnswer, citing passage numbers:")


def ask(question: str, k: int = K, ticker: str | None = None,
        model: str | None = None, use_cache: bool = True,
        strategy: str = DEFAULT_STRATEGY, route: bool = False,
        retriever: str = "dense",
        history: list[dict] | None = None,
        structured: bool = False) -> dict:
    """One question, one cited answer.

    `history` makes this multi-turn: previous turns resolve what a follow-up
    refers to, both for routing (which company "it" is) and for the model
    (which figure "that" was). An empty history reproduces the single-turn
    path exactly -- same route, same prompt, same system prompt, same cache
    key -- which is what keeps the evaluated numbers valid.
    """
    searcher = retrieve_mod.get_retriever(retriever, strategy)
    prev = prev_companies(history)

    t0 = time.time()
    if ticker:                       # an explicit --ticker overrides the router
        hits = searcher.search(question, k, {"ticker": ticker})
        plan = {"mode": "filtered", "companies": [ticker],
                "why": "ticker given on the command line"}
    else:
        hits, plan = route_mod.retrieve(searcher, question, k=k, route=route,
                                        prev_companies=prev)
    t_retrieve = time.time() - t0

    if not hits:
        return {"question": question, "answer": "Not in the provided filings.",
                "hits": [], "plan": plan, "t_retrieve": t_retrieve, "cached": False,
                "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                "model": model or config.MODEL, "prompt": "",
                "structured": None}

    prompt = build_prompt(question, hits, history)
    system = SYSTEM
    if history_block(history):
        system += HISTORY_RULE
    if structured:
        system += STRUCTURED_RULE
    out = llm.complete(prompt, system=system, model=model, use_cache=use_cache,
                       response_format=RESPONSE_FORMAT if structured else None)

    # The prose stays the headline answer whether or not a schema was used, so
    # verify.py, the graders and the UI all keep working on one field.
    data = parse_structured(out["text"]) if structured else None
    answer = data["answer"] if data else out["text"]
    return {"question": question, "answer": answer, "hits": hits,
            "plan": plan, "prompt": prompt, "t_retrieve": t_retrieve,
            "structured": data, **out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--ticker", choices=config.TICKERS,
                    help="restrict retrieval to one filer (metadata filter)")
    ap.add_argument("--model", default=None, choices=list(config.MODELS))
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY)
    ap.add_argument("--retriever", default="dense",
                    choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--route", action="store_true",
                    help="Step 10: one filtered search per company named")
    ap.add_argument("--structured", action="store_true",
                    help="ask for JSON matching ANSWER_SCHEMA")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    r = ask(args.question, k=args.k, ticker=args.ticker, model=args.model,
            use_cache=not args.no_cache, strategy=args.strategy,
            route=args.route, retriever=args.retriever,
            structured=args.structured)

    if args.show_prompt:
        print(f"\n{'-' * 70}\nPROMPT SENT\n{'-' * 70}\n{r['prompt']}")

    print(f"\n{'=' * 70}\nQ: {r['question']}\n{'=' * 70}")
    print(f"\n{r['answer']}\n")

    d = r.get("structured")
    if d:
        for f in d["figures"]:
            print(f"  figure      {f['company']:<4} {f['metric']} "
                  f"{f['period']}: {f['value']}  [{f['passage']}]")
        for u in d["unavailable"]:
            print(f"  unavailable {u['company']:<4} {u['metric']} "
                  f"{u['period']} - {u['reason']}")
        if d["figures"] or d["unavailable"]:
            print()

    print(f"{'-' * 70}\nROUTE: {r['plan']['mode']} - {r['plan']['why']}")
    print(f"{'-' * 70}\nEVIDENCE RETRIEVED ({len(r['hits'])} passages)\n{'-' * 70}")
    for i, h in enumerate(r["hits"], 1):
        snippet = body_of(h["text"])[:110].replace("\n", " ")
        print(f"[{i}] {h['score']:.3f}  {passage_label(h['metadata'])}")
        print(f"     {snippet}...")

    tag = "CACHED" if r["cached"] else f"{r['latency_s']:.2f}s"
    print(f"\n{'-' * 70}")
    print(f"model {r['model']} | retrieval {r['t_retrieve']*1000:.0f}ms | "
          f"generation {tag} | tokens {r['prompt_tokens']} in / "
          f"{r['completion_tokens']} out")
    stats = llm.cache_stats()
    print(f"cache: {stats['entries']} entries, {stats['bytes']/1024:.0f} KB")


if __name__ == "__main__":
    main()
