"""Step 17 — the interface. A demo that carries its own evidence.

    streamlit run app/streamlit_app.py

Three pages in a sidebar nav: a chat over the filings, the same question run
through two configurations side by side, and the measured results. The last is
the point: a demo that shows a system working is a toy, and a demo that shows
what the system scores on a held-out test set is a portfolio piece.

Pages rather than st.tabs, for one concrete reason. st.chat_input is pinned to
the bottom of the window only when it is called from the main body of the app;
nested inside a tab, a column or a container it renders inline, wherever the
script happens to reach it. The question box sitting halfway up the page was
that, not styling. One page per view puts the composer back in the main body.

Designed to stay usable with no API quota, because the free tier has a daily
token cap and a demo that dies at the cap is worse than no demo. Retrieval
makes no API calls at all, every answer to a test-set question is already in
the response cache, and a rate limit is reported as a message beside the
retrieved passages rather than a stack trace.
"""

import csv
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Bridge Streamlit's secret store into the environment before src/ is imported.
# src/llm.py reads os.getenv("GROQ_API_KEY") and loads a local .env, which is
# right for development and absent in a deployment -- .env is gitignored and
# never leaves this machine. On Streamlit Community Cloud the key is set as an
# app secret instead, so it is copied across here. Done explicitly rather than
# relying on the platform exporting secrets as environment variables, because
# that behaviour is the platform's to change and a missing key takes out
# generation entirely.
for _k in ("GROQ_API_KEY", "GEMINI_API_KEY", "TYPESAFE_API_KEY"):
    try:
        if _k not in os.environ and _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        # No secrets file at all is the normal local case, not an error.
        pass

from src import ask as ask_mod
from src import config, retrieve as retrieve_mod, route as route_mod
from src.store import get_store

st.set_page_config(page_title="SEC Filings RAG", page_icon="📊", layout="wide")

QUESTIONS = ROOT / "eval" / "questions.csv"
RESULTS = ROOT / "results"


@st.cache_resource(show_spinner="Loading retriever...")
def retriever(kind: str, strategy: str):
    return retrieve_mod.get_retriever(kind, strategy)


@st.cache_data
def test_questions() -> list[dict]:
    if not QUESTIONS.exists():
        return []
    return list(csv.DictReader(QUESTIONS.open()))


@st.cache_data
def results(label: str) -> list[dict]:
    f = RESULTS / f"{label}.csv"
    return list(csv.DictReader(f.open())) if f.exists() else []


def passage_block(h: dict, i: int) -> None:
    m = h["metadata"]
    is_table = h["text"].split("\n\n", 1)[-1].lstrip().startswith("[TABLE]")
    label = (f"**[{i}]** {m.get('ticker')} · {m.get('form')} FY{m.get('fiscal_year')} · "
             f"Item {m.get('section') or '—'} · score {h['score']:.3f}"
             + ("  ·  📋 table" if is_table else ""))
    with st.expander(label):
        st.text(h["text"].split("\n\n", 1)[-1][:2500])
        st.caption(f"`{h['chunk_id']}` · {len(m.get('source_block_ids') or [])} "
                   f"source blocks · {m.get('total_tokens')} tokens")


def slim_history(thread: list[dict]) -> list[dict]:
    """The thread as src/ask.py wants it: question and answer text only.

    Every turn is kept, including ones with no answer -- a quota error, or
    retrieval-only mode. Dropping those was a bug: routing a follow-up reads
    the previous QUESTION to find which company is under discussion, so with
    generation switched off no follow-up ever carried, and "and the loss
    ratio?" went back to searching the whole corpus.

    An absent answer is marked rather than left blank, so the model is told the
    previous turn was not answered instead of being handed an empty A: line it
    would read as a statement about the filings.
    """
    return [{"question": t["question"], "answer": t.get("answer") or "(not answered)"}
            for t in thread if t.get("question")]


def get_planner(router_kind: str):
    """The function that decides the plan, or None for the phrase list.

    Imported inside the branch rather than at module scope: the Jev router
    needs a credential and an SDK, and neither should be able to stop the app
    booting on the deterministic path. If the import fails the app says so and
    carries on with the phrase list, which is the same degradation the router
    itself performs on an API error.
    """
    if router_kind == "rules":
        return None
    try:
        if router_kind == "jev-ask":
            from src import jev_router_ask
            return jev_router_ask.plan
        from src import jev_router
        return jev_router.plan
    except Exception as e:
        st.sidebar.warning(f"Jev router unavailable, using rules: {str(e)[:80]}")
        return None


def turn_result(question: str, kind: str, strategy: str, do_route: bool,
                generate: bool, k: int, history: list[dict],
                planner=None) -> dict:
    """One completed turn, stored in session state and re-rendered on rerun.

    Streamlit re-executes the whole script on every interaction, so nothing
    drawn directly survives. The turn has to be DATA -- answer, plan, hits,
    figures -- and the transcript is redrawn from the list of them each time.
    That is the whole reason earlier answers used to vanish.

    Retrieval failures and quota failures are different things and are kept
    apart: a quota error still returns the passages, because retrieval makes
    no API call and evidence is the part that always works.
    """
    turn: dict = {"question": question, "answer": "", "error": None,
                  "structured": None, "model": "", "cached": False,
                  "latency_s": 0.0, "completion_tokens": 0,
                  "history_turns": len(history)}
    # The plan is decided BEFORE generation, because a question the router
    # cannot resolve has nothing to answer from: asking the user back costs no
    # answering call, and the router's own cache makes the second lookup inside
    # ask() free.
    if planner is not None and do_route:
        pre = planner(question, ask_mod.prev_companies(history))
        if pre["mode"] == "clarify":
            turn.update(answer=pre["ask"], plan=pre, hits=[], clarify=True)
            return turn

    if generate:
        try:
            res = ask_mod.ask(question, k=k, strategy=strategy, route=do_route,
                              retriever=kind, history=history, structured=True,
                              planner=planner)
            turn.update(answer=res["answer"], hits=res["hits"],
                        plan=res["plan"], structured=res["structured"],
                        model=res["model"], cached=res["cached"],
                        latency_s=res["latency_s"],
                        completion_tokens=res["completion_tokens"])
            return turn
        except Exception as e:
            msg = str(e)
            turn["error"] = (
                "**Daily token quota reached on the free tier.** The passages "
                "below are unaffected -- retrieval makes no API call. Test-set "
                "questions answer from cache."
                if "per day" in msg.lower() or "daily token" in msg.lower()
                else f"Generation failed: {msg[:300]}")

    searcher = retriever(kind, strategy)
    hits, plan = route_mod.retrieve(searcher, question, k=k, route=do_route,
                                    prev_companies=ask_mod.prev_companies(history),
                                    planner=planner)
    turn.update(hits=hits, plan=plan)
    return turn


def render_turn(turn: dict) -> None:
    """Draw one assistant turn: the answer, then how it was arrived at."""
    if turn["error"]:
        st.warning(turn["error"])
    elif turn.get("clarify"):
        st.info(f"**The question is ambiguous, so nothing was searched.**\n\n"
                f"{turn['answer']}")
    elif turn["answer"]:
        st.markdown(turn["answer"])
    else:
        st.info(f"Retrieval only -- no API call made. "
                f"{len(turn.get('hits') or [])} passages retrieved.")

    plan = turn.get("plan") or {}
    bits = [f"route `{plan.get('mode', '?')}`"]
    jev = plan.get("jev")
    if jev:
        # `error` set means the router fell back to the phrase list, and that is
        # said plainly: a plan that quietly degraded would credit Jev with a
        # decision the word list actually made.
        bits.append("router `rules` (jev unreachable)" if jev.get("error")
                    else f"router `jev` · {jev.get('choice')} "
                         f"{jev.get('confidence', 0):.2f}")
    if turn["model"]:
        bits.append(f"`{turn['model']}`")
        bits.append("from cache (no API call)" if turn["cached"]
                    else f"{turn['latency_s']:.1f}s · "
                         f"{turn['completion_tokens']} tokens out")
    if turn["history_turns"]:
        bits.append(f"{turn['history_turns']} earlier turn(s) sent as context")
    st.caption(" · ".join(bits))

    data = turn.get("structured") or {}
    figs = data.get("figures") or []
    if figs:
        st.markdown("###### Figures, as written in the filings")
        st.dataframe(
            [{"company": f.get("company"), "metric": f.get("metric"),
              "period": f.get("period"), "value": f.get("value"),
              "passage": f.get("passage")} for f in figs],
            width="stretch", hide_index=True)
    for u in data.get("unavailable") or []:
        st.caption(f"not found: {u.get('company')} · {u.get('metric')} "
                   f"· {u.get('period')} -- {u.get('reason')}")

    hits = turn.get("hits") or []

    # Provenance in two labelled drawers rather than loose text: HOW the
    # passages were chosen, and WHICH passages they were. Both collapsed, so
    # the thread reads as a conversation until you want to check it.
    with st.expander(f"How this was retrieved · route: {plan.get('mode', '?')}"):
        st.markdown(plan.get("why", "—"))
        if jev and not jev.get("error"):
            probs = jev.get("probabilities") or {}
            if probs:
                st.caption("Jev returns a calibrated probability per label, so "
                           "there is no prose to parse. Only the question is "
                           "sent -- never a retrieved passage.")
                st.dataframe(
                    [{"label": lab, "probability": round(pr, 4)}
                     for lab, pr in sorted(probs.items(), key=lambda kv: -kv[1])],
                    width="stretch", hide_index=True)
            cost = [f"`{jev.get('model', '?')}`"]
            if jev.get("cached"):
                cost.append("from cache (no API call)")
            else:
                if jev.get("input_tokens") is not None:
                    cost.append(f"{jev['input_tokens']} input tokens")
                if jev.get("latency_ms") is not None:
                    cost.append(f"{jev['latency_ms']:.0f} ms")
            st.caption(" · ".join(cost))
        if plan.get("lanes"):
            st.caption("Searches merged by rank, earlier lane wins ties: "
                       + ", ".join(f"`{x}`" for x in plan["lanes"]))
        for tk, sq in (plan.get("subqueries") or {}).items():
            st.caption(f"`{tk}` searched for: *{sq}*")
        if plan.get("per_company_k"):
            st.caption(f"{plan['per_company_k']} passages requested per company")

    if hits:
        tickers = sorted({h["metadata"]["ticker"] for h in hits})
        with st.expander(f"Evidence · {len(hits)} passages · "
                         f"{', '.join(tickers)}"):
            for i, h in enumerate(hits, 1):
                passage_block(h, i)


def run(question: str, kind: str, strategy: str, do_route: bool,
        generate: bool, k: int) -> None:
    searcher = retriever(kind, strategy)
    from src import route as route_mod

    hits, plan = route_mod.retrieve(searcher, question, k=k, route=do_route)

    st.caption(f"**Route:** `{plan['mode']}` — {plan['why']}")
    if plan.get("subqueries"):
        for tk, sq in plan["subqueries"].items():
            st.caption(f"  ↳ `{tk}`: *{sq}*")

    if not generate:
        st.info("Retrieval only — no API call made. "
                f"{len(hits)} passages retrieved.")
    else:
        try:
            res = ask_mod.ask(question, k=k, strategy=strategy,
                              route=do_route, retriever=kind)
            st.markdown("#### Answer")
            st.markdown(res["answer"])
            tag = "from cache (no API call)" if res["cached"] else \
                  f"{res['latency_s']:.1f}s · {res['completion_tokens']} tokens out"
            st.caption(f"`{res['model']}` · {tag}")
        except Exception as e:
            msg = str(e)
            if "per day" in msg.lower() or "daily token" in msg.lower():
                st.warning("**Daily token quota reached on the free tier.** "
                           "Retrieval below is unaffected — it makes no API "
                           "calls. Test-set questions answer from cache.")
            else:
                st.error(f"Generation failed: {msg[:300]}")

    st.markdown(f"#### Evidence · {len(hits)} passages")
    tickers = sorted({h["metadata"]["ticker"] for h in hits})
    st.caption("Companies represented: " + ", ".join(tickers))
    for i, h in enumerate(hits, 1):
        passage_block(h, i)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Configuration")
strategy = st.sidebar.selectbox(
    "Chunking", ["improved", "baseline"],
    help="improved = tables kept whole with headers and units; "
         "baseline = fixed 384-token windows, tables flattened")
kind = st.sidebar.selectbox(
    "Retriever", ["hybrid", "dense", "bm25"],
    help="hybrid = dense + BM25 fused by reciprocal rank")
do_route = st.sidebar.toggle(
    "Comparative routing", value=True,
    help="Decompose a question naming 2+ companies into one rewritten, "
         "metadata-filtered search per company")
router_kind = st.sidebar.radio(
    "Router", ["rules", "jev", "jev-ask"], horizontal=True,
    help="rules = the hand-written phrase list, local and cannot fail at the "
         "network. jev = the two-label classifier, measured at 100% on the "
         "held-out comparative/single questions against the phrase list's 40%. "
         "jev-ask adds a third label, `unclear`, for a question that needs one "
         "company's figures but names none: it asks which company instead of "
         "answering about whichever one happened to rank. Both fall back to the "
         "phrase list if the API is unreachable",
    disabled=not do_route)
k = st.sidebar.slider("Passages (k)", 3, 12, config.K)
generate = st.sidebar.toggle(
    "Generate an answer", value=True,
    help="Off = retrieval only, which makes no API calls")

try:
    store = get_store(config.CHROMA, config.collection_name(strategy))
    st.sidebar.caption(f"`{config.collection_name(strategy)}` · "
                       f"{store.count():,} vectors")
except Exception as e:
    st.sidebar.error(f"No index for '{strategy}': {e}")

st.sidebar.divider()
st.sidebar.caption(
    "**Known limitations.** Comparative questions answer correctly 1 time in 4. "
    "Conceptual answers score 3/8 despite 87.5% retrieval — those are "
    "generation-bound, not retrieval-bound. Nothing verifies citations yet."
)

# ---------------------------------------------------------------- main
st.title("Measured RAG over SEC filings")
st.caption("Nine filings · three title insurers · FY2024–Q2 FY2026 · "
           "retrieval 77.8% → 96.3% at an equal context budget")

def page_ask() -> None:
    """A conversation, not a search box.

    The thread lives in st.session_state because the server holds no session of
    its own and the whole script re-runs on every keystroke-sized interaction.
    It is also what gets sent back as context, so what the model is told and
    what you can see on screen are the same thing by construction.
    """
    thread: list[dict] = st.session_state.setdefault("thread", [])

    picked = ""
    qs = test_questions()
    if qs:
        with st.expander("Pick a test-set question "
                         "(answers are cached \u2014 free and instant)"):
            for t in ("lookup", "comparative", "conceptual", "unanswerable"):
                group = [r for r in qs if r["type"] == t]
                if not group:
                    continue
                st.markdown(f"**{t}**")
                cols = st.columns(2)
                for i, r in enumerate(group):
                    if cols[i % 2].button(r["question"], key=f"q{r['qid']}",
                                          width="stretch"):
                        picked = r["question"]

    # Drawn before the transcript on purpose: a click re-runs the script, and
    # clearing here means the loop below has nothing left to draw.
    if thread and st.button("New conversation", icon=":material/restart_alt:"):
        st.session_state["thread"] = thread = []

    for turn in thread:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            render_turn(turn)

    # Main body, not nested in a container -- that is what pins it to the
    # bottom of the window.
    typed = st.chat_input("Ask about the filings. Follow-ups work: "
                          "\u201cand the loss ratio?\u201d")
    question = typed or picked
    if question:
        history = slim_history(thread)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"), st.spinner("Retrieving\u2026"):
            turn = turn_result(question, kind, strategy, do_route,
                               generate, k, history,
                               planner=get_planner(router_kind))
        thread.append(turn)
        # Re-run so the page is drawn from state alone. Without this the new
        # turn is the only thing on screen that was not, and anything declared
        # above the transcript -- the New conversation button -- stays missing
        # until the next interaction, because it was drawn before the turn
        # existed.
        st.rerun()


def page_compare() -> None:
    st.markdown("Same question, two configurations. This is the Step 8 and "
                "Step 10 result as something you can click.")
    cq = st.text_input("Question to compare", key="cmpq",
                       placeholder="Which had higher total revenues in fiscal 2025, "
                                   "First American or Stewart?")
    gen_cmp = st.toggle("Generate answers (2 API calls)", value=False, key="gcmp")
    if cq:
        left, right = st.columns(2)
        with left:
            st.subheader("Baseline")
            st.caption("fixed 384-token chunks · dense · no routing")
            run(cq, "dense", "baseline", False, gen_cmp, k)
        with right:
            st.subheader("Final")
            st.caption("structure-aware chunks · hybrid · routing")
            run(cq, "hybrid", "improved", True, gen_cmp, k)

def page_results() -> None:
    """The measured numbers, from the files in results/ where that is possible.

    The per-question table is read from the shipped configuration's result file
    rather than retyped, so it cannot drift away from what was measured.
    """
    st.markdown("#### Retrieval, at an equal 3,830-token context budget")
    st.caption("Recall at a fixed k would not be a controlled comparison: "
               "baseline chunks average 383 tokens and structure-aware chunks "
               "230, so five chunks each hands one configuration 40% less "
               "context than the other. 27 answerable questions.")
    st.table({
        "configuration": ["baseline \u2014 384-token chunks, dense",
                          "+ structure- and table-aware chunking",
                          "+ comparative routing",
                          "+ hybrid retrieval (shipped)"],
        "r@ctx": ["77.8%", "77.8%", "88.9%", "96.3%"],
        "change": ["\u2014", "0.0", "+11.1", "+7.4"],
    })
    st.caption("The second row is a null result and is reported as one. "
               "Structure-aware chunking buys no recall at this budget; it "
               "earns its place by keeping row labels and unit scales attached "
               "to their tables, which shows up in answers, not in recall.")

    st.markdown("#### By question type, baseline \u2192 shipped")
    st.table({
        "measure": ["Exact lookup (15)", "Conceptual (8)", "Comparative (4)",
                    "Multi-company coverage", "recall@1"],
        "baseline": ["86.7%", "87.5%", "25.0%", "12.5%", "25.9%"],
        "shipped": ["100.0%", "87.5%", "100.0%", "79.2%", "33.3%"],
    })
    st.caption("Coverage, not r@ctx, is the number to read on comparisons. "
               "Recall counts a question as a hit if ANY gold passage is found, "
               "so \u201ccompare First American and Stewart\u201d scores full marks the "
               "moment one Stewart passage appears. Coverage asks instead how "
               "many of the companies named were actually retrieved: 79.2% "
               "where recall says 100%.")

    st.markdown("#### Answers")
    st.caption("The last column is the shipped configuration at k=10. The "
               "first four were measured at k=5 before the chunking and "
               "context changes, and the free tier\u2019s daily token cap stopped "
               "the re-run \u2014 so 12 \u2192 20 mixes several changes and is not a "
               "controlled claim. What is controlled is the retrieval table "
               "above, re-measured for all four.")
    st.table({
        "measure": ["Correct (30)", "Lookups correct (15)",
                    "Lookups scored wrong", "Ungrounded figures",
                    "Correct abstention (3)", "False abstention (27)",
                    "Conceptual correct (8)", "Comparative correct (4)"],
        # Strings, not ints: one row is "3/3" and a mixed column cannot be
        # typed, which Streamlit repairs by guessing and logs as a traceback.
        "baseline k=5": ["12", "6", "4", "3", "3/3", "9", "3", "0"],
        "+ chunking k=5": ["15", "9", "1", "0", "3/3", "9", "3", "0"],
        "+ routing k=5": ["16", "9", "1", "1", "3/3", "6", "3", "1"],
        "+ hybrid k=5": ["16", "10", "3", "1", "3/3", "2", "3", "0"],
        "shipped k=10": ["20", "12", "2", "0", "3/3", "1", "4", "1"],
    })
    st.caption("Ungrounded figures, 3 \u2192 0, is the row the verifier owns. The "
               "baseline\u2019s wrong answers were not blanks but COMPUTED figures: "
               "asked for First American\u2019s total revenue it found a table with "
               "five revenue columns, summed three, and reported "
               "$6,810.3 million with its working shown against a true "
               "$7,452.2 million.")

    rows = results("hybrid_k10_b3830")
    if rows:
        st.markdown("#### Per question, shipped configuration")
        st.caption("Read straight from results/hybrid_k10_b3830.csv. Ground "
                   "truth for lookups comes from SEC\u2019s XBRL companyfacts API.")
        st.dataframe(
            [{"qid": r["qid"], "type": r["type"], "split": r["split"],
              "question": r["question"][:70], "route": r["route_mode"],
              "r@ctx": r["recall_ctx"], "coverage": r["coverage"],
              "verdict": r["verdict"], "gold blocks": r["n_gold"]}
             for r in rows],
            width="stretch", hide_index=True)


# ---------------------------------------------------------------- navigation
# st.navigation renders the page list in the sidebar and returns the chosen
# page; nav.run() executes it. The configuration widgets above are declared
# once, outside the pages, so a setting survives moving between them.
nav = st.navigation([
    st.Page(page_ask, title="Chat", icon=":material/forum:", default=True),
    st.Page(page_compare, title="Compare configurations",
            icon=":material/compare_arrows:"),
    st.Page(page_results, title="Results", icon=":material/insights:"),
])
nav.run()
