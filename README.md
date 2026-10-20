# Measured RAG over messy SEC filings

> Every number here is reproducible from the committed `results/` files, each of
> which carries a `.meta.json` recording the configuration that produced it.

A retrieval-augmented question-answering system over nine SEC filings from three
title-insurance companies, built so that every improvement claim has a number
behind it and every number has a recorded configuration behind it.

**Headline:** retrieval recall rose from **77.8% to 96.3%** and correct answers
from 12 to **20 of 30**, on a test set whose numeric ground truth comes from
SEC's XBRL API rather than from me reading filings and typing numbers in.
Answers carrying a figure that appears nowhere in the retrieved passages fell
from **3 to 0**: fabrication stopped being the failure mode, and what remains is
misselection — the right table read on the wrong row.

The gains came from four changes, and the two that mattered most were not the
ones I expected. Recovering row labels the filers had omitted, and giving the
model ten passages instead of five, each did more than the retrieval work that
preceded them.

---

## Why this corpus

Three competitors in one industry — First American (FAF), Fidelity National
(FNF), Stewart (STC) — two annual reports and one quarterly each, nine documents,
49.5 MB of HTML.

The corpus was chosen because it is **hard in specific, measurable ways**:

| | |
|---|---|
| Three filers, three HTML structures | A parser tuned to one finds nothing in another: a literal `Item 7.` match returns zero hits in Stewart's filing |
| Figures at different scales | FAF and STC report in thousands, FNF in millions. `2,921,636` and `14,445` are both revenue |
| Negatives in brackets | `(0.3)` means minus 0.3 |
| Units declared outside the table | "in millions" is in the sentence *above* the numbers |
| Near-identical language | All three sell the same product, so a vector search cannot tell them apart by wording |
| One company is not comparable | FNF owns an annuities business the others don't, so a naive revenue comparison is silently wrong |
| Invisible machine-readable noise | Inline XBRL puts a hidden header in the HTML. FAF's contains its own CIK **536 times** |

And it is **gradeable**: SEC's `companyfacts` API serves the same figures as
machine-readable data, so numeric answers have ground truth that nobody typed.

---

## The result

Retrieval, measured at an **equal 3,830-token context budget** (see *Measurement*
below for why not a fixed `k`), on 27 answerable questions:

| | r@ctx | change |
|---|---|---|
| Baseline — fixed 384-token chunks, dense retrieval | 77.8% | — |
| **+ structure- and table-aware chunking** (Step 8) | 77.8% | 0.0 |
| **+ comparative routing** (Step 10) | 88.9% | +11.1 |
| **+ hybrid retrieval** (Step 9) | **96.3%** | +7.4 |

That second row is a null result and it is reported as one. At this context
budget, structure-aware chunking buys no recall over the naive chunker. It
earned its place for a different reason — it keeps row labels, captions and
unit scales attached to their tables, which is what makes the *answers*
correct — but the recall claim it was built on does not survive a larger
context. At the original 1,915-token budget the same change was worth +14.8
points, so what it was really fixing was a context too small to tolerate a
badly-cut chunk.

By question type, baseline → final:

| | baseline | final |
|---|---|---|
| Exact lookup (15) | 86.7% | **100.0%** |
| Conceptual (8) | 87.5% | 87.5% |
| Comparative (4) | 25.0% | **100.0%** |
| Multi-company coverage | 12.5% | **79.2%** |
| recall@1 | 25.9% | **33.3%** |

**Coverage is the number to read on comparisons, not `r@ctx`.** Recall counts a
question as a hit if *any* gold passage is found, so "compare First American and
Stewart" scores full marks the moment one Stewart passage appears. Coverage asks
instead: of the companies named, how many did we retrieve evidence for? It says
79.2% where recall says 100%, and it is the honest one. The metric was added
because the standard one was blind to the project's worst bug.

Answer quality. The final column is the shipped configuration; the first three
were measured before the chunking and context changes described below, and the
free tier's daily token cap stopped the re-run:

| | baseline | + chunking | + routing | + hybrid | **final** |
|---|---|---|---|---|---|
| | *k=5* | *k=5* | *k=5* | *k=5* | *k=10* |
| Correct (30) | 12 | 15 | 16 | 16 | **20** |
| Lookups correct (15) | 6 | 9 | 9 | 10 | **12** |
| Lookups scored wrong | 4 | 1 | 1 | 3 | **2** |
| **Ungrounded figures** | **3** | 0 | 1 | 1 | **0** |
| Correct abstention (3) | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| False abstention (27) | 9 | 9 | 6 | 2 | **1** |
| Conceptual correct (8) | 3 | 3 | 3 | 3 | **4** |
| Comparative correct (4) | 0 | 0 | 1 | 0 | **1** |

Comparing 12 against 20 mixes several changes, so it is not a controlled claim.
What *is* controlled: retrieval was re-measured for all four configurations
under the final setup, and that is the 77.8% → 96.3% table above.

Two of those rows are different failures and conflating them would hide the
result.

**`Ungrounded figures`, `3 → 0`, is the one the guardrail owns.** The baseline's
wrong answers were not missing answers — they were *computed* ones. Asked for
First American's total revenue, it found a table with five revenue columns,
summed three, and reported `$6,810.3 million` with its working shown. The true
figure is `$7,452.2 million`; it had skipped net investment income and investment
gains. That table *has* a `Total Revenues` column — the information was in the
corpus and the baseline chunker had destroyed the structure that made it
findable. A computed figure is absent from the evidence by construction, so
Step 13 catches the whole class for free: three flagged at baseline, none after
table-aware chunking.

**`Lookups scored wrong` is a regression the guardrail cannot see, and fixing it
needed the parser.** These answers are fully grounded — the right table, the
right company, the right year, read on the wrong row. Stewart's income statement
labels `Operating revenues 2,858,301` and leaves the true total `2,921,636`
unlabelled in the row beneath, so a cited, grounded answer of $2.86 billion is
wrong and nothing in the pipeline can tell.

The filer had already supplied the missing label. Every figure in these filings
is wrapped in inline XBRL naming the concept it reports:

```html
<ix:nonFraction name="us-gaap:Revenues" scale="3">2,921,636</ix:nonFraction>
```

The parser was reading the cell's text and ignoring its attributes. It now reads
the concept name and fills the row label in **only where the filer left it
blank** — 371 rows across the nine filings, out of 5,354 tagged figures in 929
tables. Existing labels are never overwritten, no figure is altered, and each
table records how many labels it recovered so the edit is auditable.

The effect, in the text the model receives:

```
before:  Operating revenues | 2,858,301 | 2,422,112 | 2,212,240
         | 2,921,636 | 2,490,419 | 2,257,341
after:   Operating revenues | 2,858,301 | 2,422,112 | 2,212,240
         Revenues           | 2,921,636 | 2,490,419 | 2,257,341
```

Measured honestly, this bought **no recall at all** for the shipped
configuration — 81.5% before and after. What it changed was what the retrieved
passage *says*, which recall cannot see: the model now distinguishes the two
rows and cites the total instead of conflating it with the subtotal.

**Five passages was the real limit, and the metric was hiding it.** With the
labels recovered, all three wrong lookups moved into the top ten — `recall@10`
= 1 on every one — but the passage carrying the answer ranked *ninth*, and the
model was being given five. The evidence was retrieved and then discarded before
it was ever seen.

Worse, the harness could not show this. `r@ctx` was computed against a token
budget while generation took a fixed `k`, so the two could disagree: at a
3,830-token budget the metric reported 96.3% recall for a context the model was
never handed. Raising `k` from 5 to 10 closed the gap and made the number
describe the prompt actually sent. Correct answers went 16 → 20.

This is still a deliberately small context. qwen's window is 131,072 tokens;
filling it would stop measuring retrieval altogether, since recall only means
something if what is retrieved is what gets used.

**Verdicts are from the re-scored runs.** The first abstention test matched
phrases of absence anywhere in an answer, so a comparative answer that named the
companies it found and said plainly which figure it could not — "the provided
passages do not contain the full-year figure for Stewart" — was recorded as a
refusal, and because abstention is tested before the type grader, the comparative
grader never ran on it. Three rows across the four runs were affected. Re-scored,
false abstentions read 9 → 9 → 6 → 2; the originals and the corrected runs are
both in `results/`.

---

## The three changes, and what each addresses

**Step 8 — structure- and table-aware chunking.** Each table becomes its own
chunk with column headers, row labels, caption and unit scale preserved; prose
never spans two document sections. Nil markers (`—`) are *kept*, because dropping
them shifts every subsequent column under the wrong heading. Targeted at the
measured finding that figures living only inside tables scored **0 of 3** while
figures also stated in prose scored **3 of 3**.

**Step 10 — comparative retrieval.** A question naming two companies is
decomposed: the query is rewritten per company *and* filtered by metadata, with
results merged round-robin by rank. Targeted at the measured finding that
"First American or Stewart" returned **Stewart only, three times out of three**.
Metadata filtering alone did not work — the same blended query vector was being
sent to each filtered search, so inside FAF's filing it matched passages about
*competitors*. **Filtering decides where to look; the rewrite decides what to
look for.**

**Step 9 — hybrid retrieval.** BM25 alongside the embeddings, fused by reciprocal
rank. BM25 *alone* is a regression — 59.3% against dense's 66.7%, collapsing to
50% on conceptual questions because it cannot tell "revenues declined" from
"sales decreased". But it is twice as good on comparatives, because company names
are exact strings an embedding blurs into generic industry language. Each is
blind where the other isn't:

| r@ctx | dense | BM25 | hybrid |
|---|---|---|---|
| Lookup | 66.7% | 66.7% | **73.3%** |
| Conceptual | 87.5% | **50.0%** | 87.5% |
| Comparative | 25.0% | **50.0%** | 25.0% |
| All | 66.7% | **59.3%** | **70.4%** |

---

## Measurement

The test set is 30 questions — 15 numeric lookups, 8 conceptual, 4 comparative,
3 unanswerable — split 12 dev / 18 test.

**Numeric ground truth is generated, not written.** A script walks SEC's XBRL
`companyfacts` JSON, so the expected answer is the figure the company itself
tagged and filed. The answer key cannot contain a transcription error and cannot
be accused of having been written to flatter the system.

**Gold passages are keyed on `block_id`, not `chunk_id`.** A `chunk_id` encodes
position within one chunking run, and Step 8 re-chunks the whole corpus. Labels
keyed on chunk ids would have made baseline and improved recall incomparable
while still producing two plausible numbers.

**Recall is reported at an equal context budget, not a fixed `k`.** Baseline
chunks average 383 tokens and structure-aware chunks 272, so "ten chunks each"
hands one strategy 30% less context than the other. The model is limited by
tokens in its prompt, so that is what is held constant: **3,830 tokens**, which
is ten baseline chunks.

**And the budget must match what generation actually receives.** It did not, at
first. `r@ctx` was computed against the token budget while generation took a
fixed `k=5`, so at 3,830 tokens the metric reported 96.3% recall for a context
the model was never handed — the right passage ranked ninth and only five were
sent. A retrieval metric that describes a prompt you do not send is not a
retrieval metric. `k` is now 10, and ten improved chunks average ~2,700 tokens,
inside the budget.

**Multi-company coverage exists because recall@k is blind to the thing Step 10
fixes.** Recall counts a hit if *any* gold passage is found, so a comparison
question scores full marks the moment one company appears. Coverage asks: of the
companies named, how many did we retrieve evidence for? It is the harsher and
more honest number — 79.2% where recall says 100%.

**Every results file ships with a `.meta.json`** recording the embedding model,
chunk strategy, retriever, routing flag, `k`, context budget, answer model and a
UTC timestamp. Free-tier endpoints change without notice, so a number with no
record of what produced it cannot be compared to next week's.

---

## Running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.in
cp .env.example .env        # add GROQ_API_KEY and a SEC User-Agent email

python -m src.fetch                        # 9 filings + 3 companyfacts JSON
python -m src.parse                        # -> 9,392 labelled blocks
python -m src.chunk --strategy improved    # -> 3,197 chunks
python -m src.index --strategy improved    # embed + load into Chroma

python -m src.ask "what drives commercial title revenue" --route --retriever hybrid
```

API and interface:

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000     # http://127.0.0.1:8000
```

A FastAPI service (`POST /api/ask`, `GET /api/config|questions|results`) with a
custom frontend served from `app/static/`. Three tabs: **Ask** (shows the query
plan, including the per-company sub-queries a comparative question was rewritten
into, then the answer with clickable citations, then the evidence), **Compare**
(the same question through the first and last configurations, side by side), and
**Evaluation** (the measured tables, the per-question verdicts, and what the
numbers do not support).

Table chunks are rendered **as tables** in the evidence pane, because the whole
argument of the chunking work is that the grid carries meaning.

Bound to localhost on purpose: this service can spend the API key in `.env`. It
stays usable with **no API quota** — retrieval makes no calls, every test-set
question is one click and answers from the response cache, generation can be
switched off entirely, and a daily-limit error is reported beside the retrieved
passages rather than as a traceback.

A Streamlit version also exists (`streamlit run app/streamlit_app.py`) and calls
the pipeline in-process. The API split is the one worth defending: the UI and the
pipeline can be deployed and rate-limited separately, and the configuration a
request ran under travels back with the answer.

Evaluation:

```bash
python -m eval.make_questions                      # build the test set
python -m eval.run_eval --strategy improved --retriever hybrid --route
python -m eval.run_eval --strategy improved --no-llm   # retrieval only, 0 API calls
```

Retrieval scoring makes **no API calls** — it compares block ids — so a chunking
change can be scored in six seconds for free, and tokens are only spent once
retrieval has actually improved. LLM responses are cached on
`(provider, model, system, prompt, params)`, so re-running an unchanged
evaluation costs nothing.

---

## Stack

| | |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5`, 384-dim, local on Apple M4 via PyTorch MPS |
| Vector store | ChromaDB, cosine, behind a 4-method interface so pgvector is a drop-in |
| Lexical | `rank_bm25`, fused by reciprocal rank (k=60) |
| Generation | `qwen/qwen3.8-27b` via Groq — chosen on *measured* instruction-following (22/22 citations in the requested format vs 5/11 for a model four times larger) |
| Parsing | BeautifulSoup + lxml |

---

## Layout

```
src/     fetch · parse · chunk · embed · index · store · retrieve · route · ask · llm · config
eval/    make_questions.py · run_eval.py · questions.csv
results/ <config>.csv + <config>.meta.json per measured run
data/    raw → parsed → chunks → chroma  (derived stages gitignored, rebuildable)
```

