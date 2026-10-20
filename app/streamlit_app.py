"""Step 17 — the interface. A demo that carries its own evidence.

    streamlit run app/streamlit_app.py

Three tabs: ask a question, compare two configurations on the same question,
and read the measured results. The third tab is the point: a demo that shows a
system working is a toy, and a demo that shows what the system scores on a
held-out test set is a portfolio piece.

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
for _k in ("GROQ_API_KEY", "GEMINI_API_KEY"):
    try:
        if _k not in os.environ and _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        # No secrets file at all is the normal local case, not an error.
        pass

from src import ask as ask_mod
from src import config, retrieve as retrieve_mod
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
           "retrieval recall 51.9% → 81.5% across three measured changes")

tab_ask, tab_cmp, tab_res = st.tabs(["Ask", "Compare configurations", "Results"])

with tab_ask:
    qs = test_questions()
    picked = ""
    if qs:
        with st.expander("Pick a test-set question (answers are cached — free and instant)"):
            for t in ("lookup", "comparative", "conceptual", "unanswerable"):
                group = [r for r in qs if r["type"] == t]
                if not group:
                    continue
                st.markdown(f"**{t}**")
                cols = st.columns(2)
                for i, r in enumerate(group):
                    if cols[i % 2].button(r["question"], key=f"q{r['qid']}",
                                          use_container_width=True):
                        picked = r["question"]

    question = st.text_input("Question", value=picked,
                             placeholder="What were Stewart's total revenues in fiscal 2025?")
    if question:
        run(question, kind, strategy, do_route, generate, k)

with tab_cmp:
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

with tab_res:
    st.markdown("#### Retrieval, at an equal 1,915-token context budget")
    st.caption("Recall at a fixed k would not be a controlled comparison: "
               "baseline chunks average 383 tokens and structure-aware chunks "
               "230, so five chunks each hands one configuration 40% less "
               "context than the other.")
    st.table({
        "configuration": ["baseline", "+ table-aware chunking",
                          "+ comparative routing", "+ hybrid retrieval"],
        "lookup": ["46.7%", "66.7%", "66.7%", "73.3%"],
        "conceptual": ["87.5%", "87.5%", "87.5%", "87.5%"],
        "comparative": ["0.0%", "25.0%", "100.0%", "100.0%"],
        "all answerable": ["51.9%", "66.7%", "77.8%", "81.5%"],
    })

    st.markdown("#### Answers")
    st.table({
        "configuration": ["baseline", "+ table-aware chunking", "+ routing"],
        "correct / 30": [12, 15, 16],
        "lookups correct / 15": [6, 9, 9],
        "lookups FABRICATED / 15": [4, 1, 1],
        "correct abstention / 3": ["3", "3", "3"],
        "false abstention / 27": [9, 9, 7],
    })
    st.caption("Fabricated answers are the row that matters. The baseline's "
               "wrong answers were not blanks — asked for First American's "
               "revenue it summed three of five revenue columns, showed its "
               "working, and reported \\$6,810.3m against a true \\$7,452.2m.")

    rows = results("routed")
    if rows:
        st.markdown("#### Per question")
        st.caption("`routed` configuration. Ground truth for lookups comes from "
                   "SEC's XBRL companyfacts API.")
        st.dataframe(
            [{"qid": r["qid"], "type": r["type"], "split": r["split"],
              "question": r["question"][:70],
              "r@ctx": r["recall_ctx"], "verdict": r["verdict"],
              "gold blocks": r["n_gold"]} for r in rows],
            use_container_width=True, hide_index=True)

    st.markdown("#### What these numbers do not support")
    st.markdown(
        "- +29.6 points is **8 questions out of 27**. A direction, not a demonstration.\n"
        "- The **test split has been looked at** during iteration, so it is not clean.\n"
        "- **Comparative answers are 1 of 4** even at 100% retrieval — coverage is "
        "capped by the context budget.\n"
        "- **Conceptual answers never moved** (3/8) despite retrieval recovering fully.\n"
        "- Hybrid + routing **answer** scores are missing: the run hit the free tier's "
        "daily token cap."
    )
