"""Step 16 — the API layer. Retrieval and generation behind HTTP.

    uvicorn app.api:app --reload --host 127.0.0.1 --port 8000

Bound to localhost by default: this service can spend the Groq key in .env.

The split matters beyond tidiness. The Streamlit version called src/ask.py in
the same process, so the UI and the pipeline could not be deployed, scaled or
rate-limited separately, and nothing else could consume the retrieval. Behind
HTTP, the frontend is one client among possible others and the configuration a
request ran under travels back with the answer -- which is the same discipline
the results files follow, for the same reason.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import ask as ask_mod
from src import config, llm, retrieve as retrieve_mod, route as route_mod, verify
from src.store import get_store

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"

Retriever = Literal["hybrid", "dense", "bm25"]
Strategy = Literal["improved", "baseline"]
Router = Literal["rules", "jev", "jev-ask"]


class Turn(BaseModel):
    """One earlier exchange, sent back by the browser on a follow-up.

    The conversation lives in the page, not on the server: this app is a
    single public process with no sessions, so holding per-user state here
    would mean two visitors sharing one conversation.
    """
    question: str = Field(default="", max_length=500)
    answer: str = Field(default="", max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    retriever: Retriever = "hybrid"
    strategy: Strategy = "improved"
    route: bool = True
    # rules = the local phrase list. jev = the two-label classifier. jev-ask
    # adds `unclear`, which answers with a question instead of a search.
    router: Router = "rules"
    generate: bool = True
    k: int = Field(default=config.K, ge=1, le=20)
    # Capped because this endpoint is public: only the last two turns are
    # used (src.ask.HISTORY_TURNS), so a longer list is a way to inflate the
    # request, not the answer.
    history: list[Turn] = Field(default_factory=list, max_length=12)
    # Structured output is on by default here and off in eval/, because the
    # schema changes the system prompt and so the cached answers the measured
    # numbers came from. Flip it off to reproduce the evaluated behaviour.
    structured: bool = True


_cache: dict[tuple[str, str], object] = {}


def searcher(kind: str, strategy: str):
    """Retrievers are expensive to build -- BM25 tokenises the whole corpus --
    so they are built once per process rather than once per request."""
    key = (kind, strategy)
    if key not in _cache:
        _cache[key] = retrieve_mod.get_retriever(kind, strategy)
    return _cache[key]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build the retrievers before the first request rather than during it.

    Lazily, the first query paid for loading the embedding model and tokenising
    the whole corpus for BM25 -- 5.8 seconds, attributed to retrieval in the
    response and paid by whoever happened to ask first. Warming at startup puts
    that cost at boot, where it belongs, so the latency the API reports is the
    latency of the actual search.

    This is a lifespan handler because the startup-event decorator is
    deprecated in this version of FastAPI and silently did nothing.
    """
    for kind in ("hybrid", "dense"):
        for strat in ("improved", "baseline"):
            t0 = time.time()
            try:
                # A real search, not just construction. Building Dense only
                # opens the Chroma collection; the sentence-transformer model
                # loads on the first embed_query, so constructing the retriever
                # warmed nothing and the first real request still paid 5.4s.
                searcher(kind, strat).search("total revenues", 1)
                print(f"  warm {kind}/{strat}: {time.time() - t0:.1f}s",
                      flush=True)
            except Exception as e:
                print(f"  warm {kind}/{strat}: unavailable ({e})", flush=True)
    yield


app = FastAPI(title="SEC Filings RAG", version="1.0",
              description="Measured retrieval over nine SEC filings.",
              lifespan=lifespan)


def is_table(text: str) -> bool:
    return text.split("\n\n", 1)[-1].lstrip().startswith("[TABLE]")


def shape_hit(h: dict, i: int) -> dict:
    m = h["metadata"]
    body = h["text"].split("\n\n", 1)[-1]
    return {
        "n": i,
        "chunk_id": h["chunk_id"],
        "score": round(float(h["score"]), 4),
        "rrf": h.get("rrf_score"),
        "ticker": m.get("ticker"),
        "company": m.get("company"),
        "form": m.get("form"),
        "fiscal_year": m.get("fiscal_year"),
        "section": m.get("section"),
        "is_table": is_table(h["text"]),
        "tokens": m.get("total_tokens"),
        "n_blocks": len(m.get("source_block_ids") or []),
        "text": body,
    }


@app.get("/api/config")
def get_config() -> dict:
    out = {"config": config.describe(), "collections": {}}
    for s in ("improved", "baseline"):
        try:
            out["collections"][s] = get_store(
                config.CHROMA, config.collection_name(s)).count()
        except Exception:
            out["collections"][s] = None
    out["cache"] = llm.cache_stats()
    return out


def get_planner(router: str):
    """The function that decides the plan, or None for the phrase list.

    A failure to load the model router is reported in the response rather than
    raised: the request can still be served by the phrase list, and a 500 here
    would turn someone else's outage into ours. The caller puts the reason in
    `plan_fallback` so a wrong-looking plan can be explained after the fact.
    """
    if router == "rules":
        return None, None
    try:
        if router == "jev-ask":
            from src import jev_router_ask
            return jev_router_ask.plan, None
        from src import jev_router
        return jev_router.plan, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


@app.post("/api/ask")
def post_ask(req: AskRequest) -> dict:
    t0 = time.time()
    try:
        s = searcher(req.retriever, req.strategy)
    except Exception as e:
        raise HTTPException(503, f"no index for '{req.strategy}': {e}")

    history = [t.model_dump() for t in req.history]
    prev = ask_mod.prev_companies(history)
    planner, plan_fallback = get_planner(req.router)

    # A question the router could not resolve to a company is answered with a
    # question, not a search. Decided before retrieval so it costs neither a
    # search nor an answering call; the router's own cache makes the second
    # lookup inside ask() free.
    if planner is not None and req.route:
        pre = planner(req.question, prev)
        if pre["mode"] == "clarify":
            return {
                "question": req.question,
                "plan": pre,
                "hits": [],
                "retrieve_ms": round((time.time() - t0) * 1000, 1),
                "companies": [],
                "settings": req.model_dump(exclude={"history"}),
                "history_turns": len(history),
                "clarify": pre["ask"],
                "answer": None, "error": None, "cached": None,
                "model": None, "latency_s": None, "structured": None,
                "plan_fallback": plan_fallback,
            }

    hits, plan = route_mod.retrieve(s, req.question, k=req.k, route=req.route,
                                    prev_companies=prev, planner=planner)
    t_retrieve = time.time() - t0

    out = {
        "question": req.question,
        "plan": plan,
        "hits": [shape_hit(h, i) for i, h in enumerate(hits, 1)],
        "retrieve_ms": round(t_retrieve * 1000, 1),
        "companies": sorted({h["metadata"]["ticker"] for h in hits}),
        # echoed back so a screenshot of an answer records what produced it.
        # The history itself is excluded: it is already on the page, and
        # echoing it would double the size of every follow-up response.
        "settings": req.model_dump(exclude={"history"}),
        "history_turns": len(history),
        "answer": None, "error": None, "cached": None,
        "model": None, "latency_s": None, "structured": None,
        # Non-null only when the chosen router could not be loaded and the
        # phrase list answered instead.
        "plan_fallback": plan_fallback,
    }

    if not req.generate:
        return out

    try:
        res = ask_mod.ask(req.question, k=req.k, strategy=req.strategy,
                          route=req.route, retriever=req.retriever,
                          history=history, structured=req.structured,
                          planner=planner)
        out.update(answer=res["answer"], cached=res["cached"],
                   model=res["model"], latency_s=round(res["latency_s"], 2),
                   completion_tokens=res.get("completion_tokens"),
                   structured=res.get("structured"))
        # Step 13 runs on every answer before it is returned. It costs nothing
        # -- no model, no network -- so there is no reason to make it optional.
        out["numeric"] = verify.verify(res["answer"],
                                       [h["text"] for h in hits],
                                       plan.get("companies"))
    except Exception as e:
        msg = str(e)
        daily = "per day" in msg.lower() or "daily token" in msg.lower()
        out["error"] = {
            "kind": "quota_daily" if daily else "generation_failed",
            "message": ("Daily token quota reached on the free tier. Retrieval "
                        "is unaffected -- it makes no API calls. Test-set "
                        "questions answer from cache."
                        if daily else msg[:400]),
        }
    return out


@app.get("/api/questions")
def get_questions() -> list[dict]:
    import csv
    f = ROOT / "eval" / "questions.csv"
    if not f.exists():
        return []
    return [{"qid": r["qid"], "type": r["type"], "split": r["split"],
             "question": r["question"], "expected": r["expected_answer"][:160],
             "n_gold": r["n_gold"]} for r in csv.DictReader(f.open())]


@app.get("/api/results")
def get_results() -> dict:
    """The measured numbers, read from the committed results files."""
    import csv
    import json
    out: dict = {"runs": {}}
    for f in sorted((ROOT / "results").glob("*.csv")):
        label = f.stem.replace(".retrieval", "")
        rows = list(csv.DictReader(f.open()))
        meta_f = f.with_suffix("").with_suffix(".meta.json")
        cand = [p for p in (ROOT / "results").glob(f"{f.stem}*.meta.json")]
        meta = json.loads(cand[0].read_text()) if cand else {}
        out["runs"][f.stem] = {
            "label": label, "meta": meta, "n": len(rows),
            "rows": [{k: r[k] for k in
                      ("qid", "type", "split", "question", "recall_ctx",
                       "coverage", "verdict", "n_gold", "answer")
                      if k in r} for r in rows],
        }
    # The headline table is stated rather than recomputed: these are the
    # numbers in the README and the docs, and a UI that silently recomputed
    # them could disagree with the writeup without anyone noticing.
    out["summary"] = {
        # Measured at an equal 3,830-token context budget with k=10, so the
        # recall reported here is recall over the passages the model is
        # actually sent. See eval/run_eval.py CTX_TOKENS for why the earlier
        # 1,915-token figures described a prompt that was never sent.
        "retrieval": [
            {"config": "baseline", "lookup": 86.7, "conceptual": 87.5,
             "comparative": 25.0, "all": 77.8},
            {"config": "+ table-aware chunking", "lookup": 86.7,
             "conceptual": 87.5, "comparative": 25.0, "all": 77.8},
            {"config": "+ comparative routing", "lookup": 86.7,
             "conceptual": 87.5, "comparative": 100.0, "all": 88.9},
            {"config": "+ hybrid retrieval", "lookup": 100.0,
             "conceptual": 87.5, "comparative": 100.0, "all": 96.3},
        ],
        # "lookups_wrong" counts lookups the eval scored wrong. "ungrounded"
        # counts answers carrying a figure absent from the passages the model
        # was given -- Step 13's flag, and the only one of the two that means
        # the model invented something. They are different numbers and were
        # once conflated here under the name "fabricated": the three wrong
        # lookups in the final run are all fully grounded misselections, the
        # right table read on the wrong row.
        "answers": [
            {"config": "baseline", "correct": 12, "lookup_correct": 6,
             "lookups_wrong": 4, "ungrounded": 3, "correct_abstention": 3,
             "false_abstention": 9},
            {"config": "+ table-aware chunking", "correct": 15,
             "lookup_correct": 9, "lookups_wrong": 1, "ungrounded": 0,
             "correct_abstention": 3, "false_abstention": 9},
            {"config": "+ comparative routing", "correct": 16,
             "lookup_correct": 9, "lookups_wrong": 1, "ungrounded": 1,
             "correct_abstention": 3, "false_abstention": 6},
            {"config": "+ hybrid retrieval", "correct": 16,
             "lookup_correct": 10, "lookups_wrong": 3, "ungrounded": 1,
             "correct_abstention": 3, "false_abstention": 2},
            # Shipped configuration: recovered XBRL row labels, k=10.
            {"config": "+ XBRL labels, k=10", "correct": 20,
             "lookup_correct": 12, "lookups_wrong": 2, "ungrounded": 0,
             "correct_abstention": 3, "false_abstention": 1},
        ],
        "caveats": [
            "+18.5 points is 5 questions out of 27 -- a direction, not a demonstration.",
            "The test split was looked at during iteration, so it is not clean.",
            "Answer scores are not controlled across configurations: only the final one was re-scored after the chunking and context changes. The free tier's 200,000 token/day cap stopped the re-run. Retrieval WAS re-measured for all four.",
            "Structure-aware chunking buys no recall at this context size (77.8% -> 77.8%). It earns its place in answer quality, by keeping labels and unit scales with their tables.",
            "Comparative answers are 1 of 4 fully correct at 100% recall@ctx. Multi-company coverage is 79.2%, which is the honest number: on some of them one named company's gold passage is never retrieved at all.",
            "Conceptual answers barely moved (3/8 -> 4/8) although conceptual retrieval has been 87.5% throughout. That bottleneck is generation, not retrieval.",
            "Citations are not resolved. Step 13 checks every figure against the passages (37 of 37 grounded, 0 flagged); no code yet checks that [n] supports the claim it is attached to.",
            "The numeric verifier now catches nothing, and that is the finding: 3 fabricated figures at baseline, 0 now, including 0 of the 5 wrong answers -- all of which are grounded. Misselection has replaced fabrication, and grounding cannot see it.",
            "Entity detection is a keyword list. A fourth filer means editing a dictionary.",
        ],
    }
    return out


if STATIC.exists():
    app.mount("/assets", StaticFiles(directory=STATIC), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")
