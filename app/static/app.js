"use strict";
const $ = s => document.querySelector(s);
const el = (t, c, h) => { const e = document.createElement(t);
  if (c) e.className = c; if (h !== undefined) e.innerHTML = h; return e; };
const esc = s => String(s ?? "").replace(/[&<>"]/g,
  c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;" }[c]));

const S = { strategy:"improved", retriever:"hybrid", route:true,
            router:"rules", generate:true, k:10, cmpGen:false };   /* k matches config.K:
   the page was asking for five passages while every measured number in the
   Evaluation tab was produced at ten, so the demo was quietly weaker than
   the results it displayed. */

/* The conversation, oldest first. It lives here rather than on the server:
   the API is one public process with no sessions, so server-side state would
   put every visitor in the same conversation. Sent on each request; only the
   last two turns are used (src/ask.py HISTORY_TURNS). */
const THREAD = [];

const HINTS = {
  strategy:{ improved:"Tables kept whole with column headers, caption and unit scale.",
             baseline:"Fixed 384-token windows. Tables flattened to a line of digits." },
  retriever:{ hybrid:"Embeddings and BM25, fused by reciprocal rank.",
              dense:"bge-small-en-v1.5 embeddings. Searches by meaning.",
              bm25:"Keyword matching. Finds exact line-item names, blind to synonyms." },
  router:{ rules:"A hand-written phrase list. Local, and it cannot fail at the network.",
           jev:"A classifier: one search, or one per company. Returns a calibrated probability per label, and only the question is ever sent.",
           "jev-ask":"Adds a third label, `unclear`. A question that needs one company's figures but names none gets a question back instead of a confident answer about whichever company ranked." }
};

/* ---------------- controls ---------------- */
function seg(id, key, onChange){
  $(id).addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    [...$(id).children].forEach(x => x.setAttribute("aria-pressed", x === b));
    S[key] = b.dataset.v;
    const h = $("#hint" + key[0].toUpperCase() + key.slice(1));
    if (h) h.textContent = HINTS[key][S[key]];
    onChange && onChange();
  });
}
function sw(id, key){
  const n = $(id), flip = () => { S[key] = !S[key];
    n.setAttribute("aria-checked", String(S[key])); };
  n.addEventListener("click", flip);
  n.addEventListener("keydown", e => {
    if (e.key === " " || e.key === "Enter") { e.preventDefault(); flip(); } });
}
seg("#segStrategy", "strategy");
seg("#segRetriever", "retriever");
seg("#segRouter", "router");
sw("#swRoute", "route"); sw("#swGenerate", "generate"); sw("#swCmpGen", "cmpGen");
$("#kRange").addEventListener("input", e => {
  S.k = +e.target.value; $("#kOut").textContent = S.k; });

document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(x =>
    x.setAttribute("aria-selected", String(x === t)));
  document.querySelectorAll(".pane").forEach(p =>
    p.classList.toggle("on", p.id === "pane-" + t.dataset.pane));
}));

/* ---------------- rendering ---------------- */
function metric(k, v, d, cls){
  return `<div class="metric"><div class="k">${k}</div>
    <div class="v ${cls||""}">${v}</div>${d?`<div class="d">${d}</div>`:""}</div>`;
}

/* A table chunk is rendered as a table. The whole argument of Step 8 is that
   the grid carries meaning, so showing it as a wall of text in the UI would
   undercut the thing the UI exists to demonstrate. */
function renderChunk(text){
  if (!text.trimStart().startsWith("[TABLE]"))
    return `<pre>${esc(text)}</pre>`;
  const lines = text.split("\n");
  const head = [], rows = [];
  let cols = null;
  for (const ln of lines){
    if (ln.startsWith("Columns: ")) { cols = ln.slice(9).split(" | "); continue; }
    if (cols === null) { head.push(ln); continue; }
    if (ln.trim()) rows.push(ln.split(" | "));
  }
  if (!cols) return `<pre>${esc(text)}</pre>`;
  const width = Math.max(cols.length, ...rows.map(r => r.length));
  const th = Array.from({length:width}, (_,i) => `<th>${esc(cols[i] ?? "")}</th>`).join("");
  const tb = rows.map(r =>
    "<tr>" + Array.from({length:width}, (_,i) =>
      `<td>${esc(r[i] ?? "")}</td>`).join("") + "</tr>").join("");
  const cap = head.filter(h => h.startsWith("Caption:") || h.startsWith("Figures")
                            || h.startsWith("Section:")).join(" · ");
  return (cap ? `<div class="capt">${esc(cap)}</div>` : "") +
         `<table class="chunk"><thead><tr>${th}</tr></thead><tbody>${tb}</tbody></table>`;
}

function evidence(hits){
  if (!hits.length) return `<p class="empty">Nothing retrieved.</p>`;
  return hits.map(h => `
    <details class="ev">
      <summary>
        <div class="n">${h.n}</div>
        <div class="src">${esc(h.ticker)} · ${esc(h.form)} FY${esc(h.fiscal_year)}
          · Item ${esc(h.section ?? "—")}</div>
        ${h.is_table ? `<span class="badge tbl">table</span>` : ""}
        <div class="sc">${h.score.toFixed(3)}</div>
      </summary>
      <div class="body">
        ${renderChunk(h.text)}
        <div class="foot">${esc(h.chunk_id)} · ${h.n_blocks} source blocks ·
          ${h.tokens} tokens</div>
      </div>
    </details>`).join("");
}

/* Earlier turns, shown collapsed so the current answer stays the focus. The
   point of showing them at all is that they are what the follow-up resolved
   against -- a thread you cannot see is a thread you cannot check. */
function threadCard(){
  if (!THREAD.length) return "";
  const turns = THREAD.map((t, i) => `<details class="turn"${
      i === THREAD.length - 1 ? " open" : ""}>
    <summary><span class="tn">${i + 1}</span>${esc(t.question)}</summary>
    <div class="answer">${esc(t.answer)}</div></details>`).join("");
  const used = Math.min(THREAD.length, 2);
  return `<div class="card"><h3>Conversation · ${THREAD.length} turn${
    THREAD.length > 1 ? "s" : ""}</h3>${turns}
    <div class="sub" style="color:var(--fg-3);margin-top:10px">last ${used} turn${
      used > 1 ? "s" : ""} sent as context · figures must still come from the
      passages, never from the thread</div></div>`;
}

function planCard(p, companies, ms){
  const cls = p.mode === "comparative" ? "cmp" : p.mode === "filtered" ? "flt" : "";
  const subs = p.subqueries ? Object.entries(p.subqueries).map(([tk, sq]) =>
    `<div class="sub">↳ <b>${esc(tk)}</b> &nbsp;${esc(sq)}</div>`).join("") : "";
  return `<div class="card"><h3>Query plan</h3>
    <div class="plan">
      <div class="planrow"><span class="pill ${cls}">${esc(p.mode)}</span>
        <span style="color:var(--fg-2)">${esc(p.why)}</span></div>
      ${subs}
      <div class="sub" style="color:var(--fg-3)">retrieved in ${ms} ms ·
        companies represented: ${companies.map(esc).join(", ") || "—"}</div>
    </div></div>`;
}

function numericCard(v){
  if (!v || !v.n_figures) return "";
  const flagged = v.verdict === "flagged";
  const rows = v.figures.map(f => `<tr>
      <td style="text-align:left">${esc(f.raw)}</td>
      <td>${f.value.toLocaleString()}</td>
      <td class="${f.status === "grounded" ? "best" : "zero"}"
          style="text-align:left">${f.status}</td>
      <td style="text-align:left;color:var(--fg-3)">${esc(f.source || "not in the evidence")}</td>
    </tr>`).join("");
  return `<div class="card">
    <h3>Numeric verification · deterministic, no model call</h3>
    ${flagged ? `<div class="note bad" style="margin-bottom:14px">
        <b>${v.n_unsupported} figure(s) not grounded in the evidence.</b><br>
        ${esc(v.note)}</div>`
      : `<div class="note" style="margin-bottom:14px;border-left-color:var(--good)">
        ${esc(v.note)}</div>`}
    <table class="dt"><thead><tr><th style="text-align:left">Figure</th>
      <th>Value (USD)</th><th style="text-align:left">Status</th>
      <th style="text-align:left">Grounded by</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
}

function routerTag(d){
  const j = (d.plan && d.plan.jev) || null;
  if (!j) return "";
  if (j.error) return " · fell back to rules";
  const c = (j.confidence ?? 0).toFixed(2);
  return ` · ${esc(j.choice)} ${c}${j.cached ? " (cached)" : ""}`;
}

function answerCard(d){
  if (d.clarify){
    const j = (d.plan && d.plan.jev) || {};
    return `<div class="card"><h3>The question is ambiguous, so nothing was
      searched</h3>
      <div class="answer">${esc(d.clarify)}</div>
      <div class="meta"><span>router jev-ask</span>
        <span>${esc(j.choice)} ${(j.confidence ?? 0).toFixed(2)}</span>
        <span>no search, no answering call</span></div></div>`;
  }
  if (d.error){
    const k = d.error.kind === "quota_daily" ? "warn" : "bad";
    return `<div class="note ${k}"><b>${d.error.kind === "quota_daily"
      ? "Daily token quota reached" : "Generation failed"}</b><br>
      ${esc(d.error.message)}</div>`;
  }
  if (d.answer === null) return "";
  const withCites = esc(d.answer).replace(/\[(\d+)\]/g,
    (_, n) => `<span class="cite" data-n="${n}">${n}</span>`);
  const tag = d.cached ? "from cache · no API call"
                       : `${d.latency_s}s · ${d.completion_tokens ?? "?"} tokens out`;
  return `<div class="card"><h3>Answer</h3>
    <div class="answer">${withCites}</div>
    <div class="meta"><span>${esc(d.model)}</span><span>${tag}</span>
      <span>k=${d.settings.k}</span><span>${esc(d.settings.retriever)}</span>
      <span>${esc(d.settings.strategy)}</span>
      <span>routing ${d.settings.route ? "on" : "off"}</span>
      <span>router ${esc(d.settings.router || "rules")}${routerTag(d)}</span>
      </div></div>`;
}

/* The model returns its figures as data, not prose, so each one can point at
   the passage it came from. That link is what the inline [n] citations used to
   do, except it cannot drift out of step with the sentence around it. */
function figuresCard(d){
  const st = d.structured;
  if (!st) return "";
  const figs = st.figures || [], gaps = st.unavailable || [];
  if (!figs.length && !gaps.length) return "";

  const rows = figs.map(f => `<tr>
      <td style="text-align:left"><b>${esc(f.company)}</b></td>
      <td style="text-align:left">${esc(f.metric)}</td>
      <td style="text-align:left">${esc(f.period)}</td>
      <td style="text-align:left">${esc(f.value)}</td>
      <td><span class="cite" data-n="${esc(f.passage)}">${esc(f.passage)}</span></td>
    </tr>`).join("");

  const table = figs.length ? `<table class="dt">
      <thead><tr><th style="text-align:left">Company</th>
        <th style="text-align:left">Metric</th>
        <th style="text-align:left">Period</th>
        <th style="text-align:left">Value as written</th>
        <th>Passage</th></tr></thead>
      <tbody>${rows}</tbody></table>` : "";

  /* Shown as plainly as the figures. A question the filings cannot answer is
     a result, not an error -- and saying so beats reasoning about it in the
     answer, which is what the model used to do. */
  const missing = gaps.length ? `<div class="note warn"${
      figs.length ? ' style="margin-top:14px"' : ""}>
      <b>Not in the retrieved passages</b>
      ${gaps.map(u => `<div class="sub">${esc(u.company)} · ${esc(u.metric)}
        · ${esc(u.period)} — ${esc(u.reason)}</div>`).join("")}</div>` : "";

  return `<div class="card"><h3>Figures stated · ${figs.length}${
    gaps.length ? ` · ${gaps.length} unavailable` : ""}</h3>
    ${table}${missing}</div>`;
}

function wireCites(root){
  root.querySelectorAll(".cite").forEach(c => c.addEventListener("click", () => {
    const list = root.querySelectorAll(".ev");
    const t = list[+c.dataset.n - 1];
    if (t){ t.open = true; t.scrollIntoView({behavior:"smooth", block:"center"}); }
  }));
}

/* ---------------- requests ---------------- */
async function ask(body){
  const r = await fetch("/api/ask", { method:"POST",
    headers:{ "Content-Type":"application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return r.json();
}

async function runAsk(){
  const q = $("#q").value.trim(); if (!q) return;
  const out = $("#askOut");
  $("#go").disabled = true;
  out.innerHTML = `<p class="empty"><span class="spin"></span> searching…</p>`;
  try{
    const d = await ask({ question:q, ...S, k:S.k, history:THREAD });
    out.innerHTML = threadCard()
      + `<div class="card"><h3>Question</h3>
           <div class="answer">${esc(q)}</div></div>`
      + planCard(d.plan, d.companies, d.retrieve_ms)
      + answerCard(d)
      + figuresCard(d)
      + numericCard(d.numeric)
      + `<div class="card"><h3>Evidence · ${d.hits.length} passages</h3>
           ${evidence(d.hits)}</div>`;
    wireCites(out);
    out.scrollTop = 0;          /* the newest answer, not the old transcript */
    $("#askMetrics").innerHTML =
      metric("Passages", d.hits.length, `k = ${d.settings.k}`) +
      metric("Tables", d.hits.filter(h => h.is_table).length, "of retrieved") +
      metric("Companies", d.companies.length || "—", d.companies.join(" ")) +
      metric("Retrieval", d.retrieve_ms + " ms", "no API call") +
      metric("Route", d.plan.mode, d.plan.companies.join(" ") || "unfiltered") +
      (d.history_turns
        ? metric("Context", d.history_turns + " turn" +
            (d.history_turns > 1 ? "s" : ""), "carried into this question")
        : "") +
      (d.numeric && d.numeric.n_figures
        ? metric("Figures checked", d.numeric.n_figures,
            d.numeric.n_unsupported ? d.numeric.n_unsupported + " ungrounded" : "all grounded",
            d.numeric.n_unsupported ? "down" : "up")
        : "");
    /* Only a real answer joins the thread. A quota error or a retrieval-only
       run would otherwise leave a turn with an empty answer, and the next
       follow-up would resolve against a question that was never answered. */
    if (d.answer){
      THREAD.push({ question:q, answer:d.answer });
      $("#q").value = "";
      $("#q").placeholder = "Follow up: what about Stewart? · and the loss ratio?";
    }
    $("#newchat").hidden = !THREAD.length;
  } catch(e){
    out.innerHTML = `<div class="note bad">${esc(e.message)}</div>`;
  } finally { $("#go").disabled = false; }
}
$("#go").addEventListener("click", runAsk);
$("#q").addEventListener("keydown", e => { if (e.key === "Enter") runAsk(); });

$("#newchat").addEventListener("click", () => {
  THREAD.length = 0;
  $("#newchat").hidden = true;
  $("#q").value = "";
  $("#q").placeholder = "What were Stewart's total revenues in fiscal 2025?";
  $("#askOut").innerHTML = "";
  $("#askMetrics").innerHTML = "";
  $("#q").focus();
});

async function runCompare(){
  const q = $("#cq").value.trim(); if (!q) return;
  const L = $("#cmpL"), R = $("#cmpR");
  $("#cgo").disabled = true;
  L.innerHTML = R.innerHTML = `<p class="empty"><span class="spin"></span> …</p>`;
  const side = (node, cfg) => ask({ question:q, k:S.k, generate:S.cmpGen, ...cfg })
    .then(d => { node.innerHTML = planCard(d.plan, d.companies, d.retrieve_ms)
        + answerCard(d)
        + numericCard(d.numeric)
      + `<div class="card"><h3>Evidence · ${d.hits.length}</h3>${evidence(d.hits)}</div>`;
      wireCites(node); })
    .catch(e => { node.innerHTML = `<div class="note bad">${esc(e.message)}</div>`; });
  await Promise.all([
    side(L, { strategy:"baseline", retriever:"dense", route:false }),
    side(R, { strategy:"improved", retriever:"hybrid", route:true }),
  ]);
  $("#cgo").disabled = false;
}
$("#cgo").addEventListener("click", runCompare);
$("#cq").addEventListener("keydown", e => { if (e.key === "Enter") runCompare(); });

/* ---------------- boot ---------------- */
(async function boot(){
  try{
    const c = await (await fetch("/api/config")).json();
    const n = c.collections.improved;
    $("#collCount").textContent = n ? n.toLocaleString() + " vectors" : "no index";
    $("#statline").innerHTML = `
      <div><span>embeddings</span><b>${esc(c.config.embed_model.split("/").pop())}</b></div>
      <div><span>dimensions</span><b>${c.config.embed_dim}</b></div>
      <div><span>answer model</span><b>${esc(c.config.answer_model.split("/").pop())}</b></div>
      <div><span>chunk budget</span><b>${c.config.chunk_tokens} tok</b></div>
      <div><span>response cache</span><b>${c.cache.entries} entries</b></div>`;
  } catch(e){ $("#collCount").textContent = "api unreachable"; }

  try{
    const qs = await (await fetch("/api/questions")).json();
    $("#qgrid").innerHTML = qs.map(q =>
      `<button class="qchip" data-q="${esc(q.question)}">
         <span class="tag t-${q.type}">${q.type} · ${q.qid}</span>
         ${esc(q.question)}</button>`).join("");
    $("#qgrid").addEventListener("click", e => {
      const b = e.target.closest(".qchip"); if (!b) return;
      $("#q").value = b.dataset.q;
      document.querySelector("details.samples").open = false;
      runAsk();
    });
  } catch(e){}

  try{
    const r = await (await fetch("/api/results")).json();
    const sum = r.summary;
    const last = sum.retrieval[sum.retrieval.length - 1];
    const lastA = sum.answers[sum.answers.length - 1];
    $("#resMetrics").innerHTML =
      metric("Recall, final", last.all + "%", "from 77.8% at baseline", "up") +
      metric("Comparative", last.comparative + "%", "from 25.0%", "up") +
      metric("Correct answers", lastA.correct + " / 30", "from 12 at baseline", "up") +
      metric("Ungrounded figures", String(lastA.ungrounded),
             "from 3 at baseline", "up") +
      metric("Questions", "30", "12 dev / 18 test") +
      metric("Ground truth", "XBRL", "SEC companyfacts API");

    const rt = sum.retrieval.map((x, i) => `<tr class="${
        i === sum.retrieval.length - 1 ? "final" : ""}">
      <td>${esc(x.config)}</td>
      <td>${x.lookup}%</td><td>${x.conceptual}%</td>
      <td class="${x.comparative === 0 ? "zero" : x.comparative === 100 ? "best" : ""}">${x.comparative}%</td>
      <td class="${i === sum.retrieval.length - 1 ? "best" : ""}">${x.all}%</td></tr>`).join("");
    $("#tblRetrieval").innerHTML = `<table class="dt"><thead><tr>
      <th>Configuration</th><th>Lookup (15)</th><th>Conceptual (8)</th>
      <th>Comparative (4)</th><th>All answerable (27)</th>
      </tr></thead><tbody>${rt}</tbody></table>`;

    const at = sum.answers.map((x, i) => `<tr class="${
        i === sum.answers.length - 1 ? "final" : ""}">
      <td>${esc(x.config)}</td><td>${x.correct}</td><td>${x.lookup_correct}</td>
      <td class="${x.lookups_wrong > 1 ? "zero" : "best"}">${x.lookups_wrong}</td>
      <td class="${x.ungrounded > 0 ? "zero" : "best"}">${x.ungrounded}</td>
      <td>${x.correct_abstention}/3</td><td>${x.false_abstention}</td></tr>`).join("");
    $("#tblAnswers").innerHTML = `<table class="dt"><thead><tr>
      <th>Configuration</th><th>Correct / 30</th><th>Lookups / 15</th>
      <th title="Lookups the eval scored wrong">Lookups wrong</th>
      <th title="Answers carrying a figure absent from the passages given">Ungrounded</th>
      <th>Correct abstention</th><th>False abstention / 27</th>
      </tr></thead><tbody>${at}</tbody></table>`;

    $("#caveats").innerHTML = sum.caveats.map(c => `<li>${esc(c)}</li>`).join("");

    const run = r.runs["hybrid_k10_b3830"] || r.runs["hybrid_routed.regraded"]
                || r.runs["hybrid_routed"] || r.runs["routed"];
    if (run){
      const rows = run.rows.map(x => `<tr>
        <td>${esc(x.qid)}</td><td>${esc(x.type)}</td><td>${esc(x.split)}</td>
        <td style="text-align:left;white-space:normal;max-width:430px">${esc(x.question)}</td>
        <td class="${x.recall_ctx === "1" ? "best" : x.recall_ctx === "0" ? "zero" : ""}">
          ${x.recall_ctx === "" ? "—" : x.recall_ctx}</td>
        <td class="${x.verdict === "correct" ? "best"
          : /wrong|false/.test(x.verdict) ? "zero" : ""}">${esc(x.verdict || "—")}</td>
        <td>${esc(x.n_gold)}</td></tr>`).join("");
      $("#tblPerQ").innerHTML = `<table class="dt"><thead><tr>
        <th>QID</th><th>Type</th><th>Split</th><th>Question</th>
        <th>r@ctx</th><th>Verdict</th><th>Gold</th>
        </tr></thead><tbody>${rows}</tbody></table>`;
    }
  } catch(e){ $("#resMetrics").innerHTML = `<div class="note bad">results unavailable</div>`; }
})();
