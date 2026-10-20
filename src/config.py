"""Single source of truth for paths, model names and tuning knobs.

The rule this file follows, and the reason it is not simply "every constant":

    A value lives here if MORE THAN ONE MODULE needs it, or if Step 7 varies
    it between runs. A value used by exactly one module and never tuned stays
    in that module.

So the parser's regexes, the answer prompt in ask.py, and the SEC endpoint URLs
in fetch.py are deliberately NOT here. Nobody tunes a regex between runs, and
hauling them in would make this file a junk drawer and parse.py unreadable.
What IS here is everything that was previously written down twice.

Why a Python module rather than YAML or TOML: no new dependency, the paths can
be computed, and a typo raises ImportError at startup instead of returning None
at midnight. The trade-off is that changing a knob means editing code — which
is fine here, because every knob change is an experiment that belongs in git
history anyway.

`describe()` returns the whole configuration as a dict. Step 7 stamps it into
every results file, so any measurement can be traced to the settings that
produced it. A strategy comparison you cannot reproduce is not a measurement.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# paths — defined once, derived everywhere
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"                  # fetched filing HTML
XBRL = DATA / "xbrl"                # companyfacts JSON
MANIFEST = DATA / "manifest.csv"    # one row per filing, with sha256
PARSED = DATA / "parsed"            # ordered labelled blocks
CHUNKS = DATA / "chunks"            # one .jsonl per chunking strategy
CHROMA = DATA / "chroma"            # the vector store directory

EVAL = ROOT / "eval"                # test set (Step 6)
RESULTS = ROOT / "results"          # measurements (Step 7)

LLM_CACHE = ROOT / ".cache" / "llm"

# --------------------------------------------------------------------------- #
# the corpus
# --------------------------------------------------------------------------- #

# Three title insurers. Competitors in one industry, which is the point: their
# filings are near-identical in wording, so retrieval has to distinguish them
# by metadata rather than by meaning.
TICKERS = ["FAF", "FNF", "STC"]
N_ANNUAL = 2                        # 10-K filings per company
N_QUARTERLY = 1                     # 10-Q filings per company

# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #

# EMBED_MODEL and QUERY_PREFIX must change together: the prefix is what THIS
# model was trained to expect on the query side. That coupling is why they sit
# adjacent rather than in the module that uses them.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --------------------------------------------------------------------------- #
# chunking — swept by Step 7, compared by Step 8
# --------------------------------------------------------------------------- #

# NOT the same 384 as EMBED_DIM above; the coincidence is unfortunate.
# This is a token budget, that is a vector width. They are unrelated.
CHUNK_TOKENS = 384      # total budget INCLUDING the header; the model truncates at 512
OVERLAP_TOKENS = 48     # ~12%, so a sentence cut at a boundary survives intact

DEFAULT_STRATEGY = "baseline"


def collection_name(strategy: str = DEFAULT_STRATEGY) -> str:
    """One chroma collection per chunking strategy, so both can be compared."""
    return f"filings_{strategy}"


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #

# Passages per answer. Raised from 5 to 10 on measured evidence: recovering the
# row labels the filers omitted moved the three wrong lookups into the top ten
# (recall@10 = 1 on all three), but at k=5 the passage carrying the answer --
# rank 9 for Stewart's total revenues -- was retrieved and then discarded
# before the model saw it.
#
# This also closes a gap between the metric and the system. eval/run_eval.py
# reports recall at an equal TOKEN budget so that two chunking strategies can
# be compared fairly, but generation took a fixed k, so the two could disagree:
# at a 3,830-token budget the metric claimed 96.3% recall for a context the
# model was never given. 10 improved chunks average ~2,700 tokens, which fits
# inside that budget, so the number now describes the prompt that was actually
# sent.
K = 10

# --------------------------------------------------------------------------- #
# answer model
# --------------------------------------------------------------------------- #

# Chat-capable models this Groq key can reach, confirmed with models.list() on
# 2026-10-17. allam-2-7b is excluded: its 4,096-token window cannot hold a
# five-passage prompt. No Llama chat model is available to this key
# (llama-3.3-70b-versatile and llama-3.1-8b-instant 404; llama3-70b-8192
# reports it has been decommissioned).
MODELS = {
    "openai/gpt-oss-120b": {"ctx": 131072, "reasoning": True},
    "openai/gpt-oss-20b":  {"ctx": 131072, "reasoning": True},
    "qwen/qwen3.8-27b":    {"ctx": 131072, "reasoning": False},
}

# Default chosen on MEASURED instruction-following, not model size. Steps 12,
# 14 and 15 parse this model's output, so obeying the output format is the
# property that matters. Over three questions (2026-10-17), asked to cite as
# "[n]": qwen produced 22 correct citations and 0 wrong; gpt-oss-120b produced
# 5 correct and 6 in the CJK-bracket form it picked up from browsing-tool
# training, giving zero parseable citations on 2 of 3 answers. qwen is also
# ~5x faster and has no reasoning-token budget to get wrong.
MODEL = "qwen/qwen3.8-27b"

REASONING_EFFORT = "low"    # only sent to models flagged reasoning=True
MAX_OUTPUT_TOKENS = 1024    # reasoning + answer must both fit inside this
TEMPERATURE = 0.0           # an eval harness needs the same answer twice

GEMINI_MODEL = "gemini-2.0-flash"   # second provider; see llm._gemini_call

# Free tier measured from the rate-limit headers on 2026-10-17: 1,000 requests
# per day and 8,000 tokens per MINUTE. Tokens per minute is the binding limit.
TPM_SAFETY_MARGIN = 200     # never ask for the last token in the bucket
MAX_RETRIES = 4

# --------------------------------------------------------------------------- #

def describe() -> dict:
    """The full configuration, for stamping into results files in Step 7."""
    return {
        "tickers": TICKERS,
        "n_annual": N_ANNUAL,
        "n_quarterly": N_QUARTERLY,
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "chunk_tokens": CHUNK_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "strategy": DEFAULT_STRATEGY,
        "k": K,
        "answer_model": MODEL,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
