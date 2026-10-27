"""Step 14 — render the claim/passage pairs as one labelling page.

Why a page and not the CSV plus the text file:

  Labelling needs the claim and the passage side by side AND a place to record
  the verdict. Splitting those across eval/groundedness_pairs.txt (readable)
  and eval/groundedness_pairs.csv (recordable) meant reading in one window and
  typing a pair id into another, 51 times. That is a transcription error
  waiting to happen as well as slow.

  The second reason is tables. A filing table is serialised into the chunk as
  pipe-separated rows under a "Columns:" line. In a plain text file those rows
  wrap at the terminal width and the columns stop lining up, so a figure cannot
  be traced to its heading -- which is the only thing a numeric claim needs.
  Rendered as a real table the alignment is restored without altering a
  character of the passage.

Nothing is reformatted or summarised. The rows, the order and the wording are
the chunk's own; only the presentation changes. The identity line, Section,
Caption and scale phrase are shown as a header block because that is what they
are in the chunk, and the scale phrase in particular ("Figures in millions.")
decides whether 14,445 means thousands or millions.

What is deliberately absent, as in the CSV: any pre-filled label, any hint of a
second opinion, and Noul's score. The page starts blank. A label shown next to
a claim is a label the reader partly agrees with by anchoring, and the whole
point of the exercise is an independent judgement to measure a scorer against.

Labels are held in the browser's localStorage so the page can be closed and
reopened mid-pass, and exported as CSV text to paste back. Nothing is uploaded;
the file is opened from disk and makes no network request.
"""

import csv
import html
import json
import re

from src import config

PAIRS = config.ROOT / "eval" / "groundedness_pairs.csv"
OUT = config.ROOT / "eval" / "label_groundedness.html"

# A serialised table row: cells joined by " | ". The header lines of a table
# chunk ("Section:", "Caption:", "Columns:") are matched separately because
# they are metadata about the table, not data in it.
META_LINE = re.compile(r"^(\[TABLE\].*|Section:.*|Caption:.*|Columns:.*|Figures in .*)$")
# A cell that should be right-aligned: a number, possibly negative in
# accounting parentheses, possibly a dash standing for zero.
NUMERIC = re.compile(r"^[(\[]?\s*[-–—$]?\s*[\d,.]+\s*\)?%?$|^[-–—]$")


def render_passage(text: str) -> str:
    """One passage as HTML: metadata as a header block, pipe rows as a table.

    The first line is the chunk's identity line -- company, form, item -- and
    it is pipe-separated like a table row, so it is taken off the front before
    anything else looks for rows. Left in, it rendered as a one-row grid that
    looked like data from the filing.
    """
    first, _, rest = text.partition("\n")
    out = [f'<div class="ident">{html.escape(first.strip())}</div>']
    rows: list[list[str]] = []
    text = rest

    def flush_rows() -> None:
        if not rows:
            return
        # Cell counts differ between rows -- a label row may hold one cell and
        # a data row six -- so no row is promoted to a <thead>. Forcing one
        # would silently drop or pad cells, which is exactly the kind of
        # distortion this page exists to avoid.
        width = max(len(r) for r in rows)
        body = []
        for r in rows:
            tds = []
            for c in r:
                cls = "num" if NUMERIC.match(c.strip()) else "lab"
                tds.append(f'<td class="{cls}">{html.escape(c.strip())}</td>')
            # Pad short rows so the browser does not collapse the grid.
            tds += ['<td class="lab"></td>'] * (width - len(r))
            body.append("<tr>" + "".join(tds) + "</tr>")
        out.append('<table class="tbl">' + "".join(body) + "</table>")
        rows.clear()

    for line in text.splitlines():
        s = line.strip()
        if not s:
            flush_rows()
            continue
        if " | " in s and not s.startswith("Columns:"):
            rows.append(s.split(" | "))
            continue
        flush_rows()
        if META_LINE.match(s):
            out.append(f'<div class="meta">{html.escape(s)}</div>')
        else:
            # The identity line and any prose. Prose passages have no pipes at
            # all and pass straight through as paragraphs.
            out.append(f'<p class="prose">{html.escape(s)}</p>')
    flush_rows()
    return "\n".join(out)


def main() -> None:
    pairs = list(csv.DictReader(PAIRS.open()))
    ids = [p["pair_id"] for p in pairs]

    blocks = []
    for i, p in enumerate(pairs):
        passages = "".join(
            f'<div class="psg"><div class="psg-n">passage {n}</div>{render_passage(t)}</div>'
            for n, t in zip((p["cited"] or "?").split(";"),
                            p["passages"].split("\n\n---\n\n")))
        blocks.append(f"""
<section class="card" id="c{i}" data-pid="{html.escape(p['pair_id'])}">
  <div class="hdr">
    <span class="pid">{html.escape(p['pair_id'])}</span>
    <span class="tag">{html.escape(p['type'])}</span>
    <span class="tag">claim {p['claim_idx']} of {p['n_claims']}</span>
    <span class="tag cited">cited: [{html.escape(p['cited'] or 'none')}]</span>
    <span class="n">{i + 1} / {len(pairs)}</span>
  </div>
  <div class="q">{html.escape(p['question'])}</div>
  <div class="claim-l">CLAIM</div>
  <div class="claim">{html.escape(p['claim'])}</div>
  <div class="psg-l">CITED PASSAGE{'S' if p['n_passages'] != '1' else ''}</div>
  {passages or '<div class="psg">(no citation given)</div>'}
  <div class="vote">
    <span class="vq">Does the passage say it?</span>
    <button class="y" data-v="y">supported &nbsp;<kbd>y</kbd></button>
    <button class="n" data-v="n">not supported &nbsp;<kbd>n</kbd></button>
    <button class="s" data-v="">clear</button>
    <span class="mark"></span>
  </div>
  <div class="note-l">NOTE <span class="note-h">&mdash; optional; why, or what was unclear</span></div>
  <textarea class="note" rows="2" placeholder="e.g. the passage never says which column is 2025"></textarea>
</section>""")

    page = TEMPLATE.replace("__BLOCKS__", "".join(blocks))
    page = page.replace("__IDS__", json.dumps(ids))
    page = page.replace("__N__", str(len(pairs)))
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(config.ROOT)}  ({len(pairs)} pairs, "
          f"{OUT.stat().st_size // 1024} KB)")
    print(f"open it with:  open {OUT.relative_to(config.ROOT)}")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Groundedness labelling</title>
<style>
:root{
  --bg:#faf9f7; --fg:#1b1b1a; --dim:#6b6b68; --line:#e2e0db; --card:#fff;
  --claim:#fffbe9; --claim-b:#e8d9a0; --psg:#f4f6f8; --y:#1a7f4b; --n:#b4371f;
  --accent:#2d5f8a;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#17181a; --fg:#e8e6e3; --dim:#9b9a97; --line:#303236; --card:#1e2023;
  --claim:#2a2616; --claim-b:#5c5021; --psg:#22252a; --y:#4fb37a; --n:#e2765c;
  --accent:#7aa8d2;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
#top{position:sticky;top:0;z-index:9;background:var(--card);
  border-bottom:1px solid var(--line);padding:10px 20px;
  display:flex;gap:16px;align-items:center;flex-wrap:wrap}
#top h1{font-size:15px;margin:0;font-weight:650}
#bar{flex:1;min-width:120px;height:7px;background:var(--line);border-radius:4px;overflow:hidden}
#fill{height:100%;width:0;background:var(--accent);transition:width .2s}
#count{font-variant-numeric:tabular-nums;color:var(--dim);font-size:13px}
#top button{font:inherit;font-size:13px;padding:5px 11px;border:1px solid var(--line);
  background:var(--bg);color:var(--fg);border-radius:6px;cursor:pointer}
#top button:hover{border-color:var(--accent)}
#rule{max-width:860px;margin:22px auto 0;padding:0 20px;color:var(--dim);font-size:14px}
#rule b{color:var(--fg)}
main{max-width:860px;margin:0 auto;padding:10px 20px 60vh}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:18px 20px;margin:22px 0;scroll-margin-top:70px}
.card.done{border-color:var(--y)}
.card.done.no{border-color:var(--n)}
.card.here{box-shadow:0 0 0 2px var(--accent)}
.hdr{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.pid{font:600 13px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--accent)}
.tag{font-size:11.5px;color:var(--dim);border:1px solid var(--line);
  border-radius:20px;padding:1px 9px}
.tag.cited{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.n{margin-left:auto;font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums}
.q{color:var(--dim);font-size:13.5px;font-style:italic;margin-bottom:14px}
.claim-l,.psg-l{font:650 11px/1 system-ui;letter-spacing:.09em;color:var(--dim);
  margin:0 0 6px}
.psg-l{margin-top:18px}
.claim{background:var(--claim);border:1px solid var(--claim-b);border-radius:7px;
  padding:11px 13px;font-size:15.5px}
.psg{background:var(--psg);border:1px solid var(--line);border-radius:7px;
  padding:11px 13px;margin-bottom:10px;overflow-x:auto}
.psg-n{font:650 10.5px/1 system-ui;letter-spacing:.08em;color:var(--dim);margin-bottom:8px}
.ident{font:650 12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
  color:var(--fg);margin-bottom:6px}
.meta{font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--dim)}
.prose{margin:7px 0;font-size:14px}
.tbl{border-collapse:collapse;margin:9px 0;font-size:13px;
  font-variant-numeric:tabular-nums;width:auto}
.tbl td{border:1px solid var(--line);padding:3px 9px;vertical-align:top}
.tbl td.num{text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  white-space:nowrap}
.tbl td.lab{max-width:430px}
.tbl tr:nth-child(odd) td{background:color-mix(in srgb,var(--fg) 3%,transparent)}
.vote{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:18px;
  padding-top:14px;border-top:1px solid var(--line)}
.vq{font-size:13.5px;color:var(--dim);margin-right:3px}
.vote button{font:inherit;font-size:14px;padding:7px 15px;border-radius:7px;
  cursor:pointer;border:1.5px solid var(--line);background:var(--bg);color:var(--fg)}
.vote button.y:hover,.vote button.y[aria-pressed=true]{border-color:var(--y);
  background:var(--y);color:#fff}
.vote button.n:hover,.vote button.n[aria-pressed=true]{border-color:var(--n);
  background:var(--n);color:#fff}
.vote button.s{font-size:12.5px;color:var(--dim);padding:7px 11px}
kbd{font:600 11px ui-monospace,monospace;opacity:.75;border:1px solid currentColor;
  border-radius:3px;padding:0 4px}
.mark{font-size:13px;color:var(--dim)}
.note-l{font:650 11px/1 system-ui;letter-spacing:.09em;color:var(--dim);margin:14px 0 6px}
.note-h{font-weight:400;letter-spacing:0;font-size:11.5px;text-transform:none}
textarea.note{width:100%;resize:vertical;font:13.5px/1.5 inherit;padding:8px 10px;
  background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px}
textarea.note:focus{outline:none;border-color:var(--accent)}
.card.noted .note-l{color:var(--accent)}
#out{position:fixed;inset:auto 0 0 0;background:var(--card);
  border-top:1px solid var(--line);padding:14px 20px;display:none;z-index:10}
#out textarea{width:100%;height:150px;font:12px/1.5 ui-monospace,monospace;
  background:var(--bg);color:var(--fg);border:1px solid var(--line);
  border-radius:6px;padding:9px}
#out .row{display:flex;gap:10px;align-items:center;margin-bottom:8px}
#out button{font:inherit;font-size:13px;padding:5px 11px;border:1px solid var(--line);
  background:var(--bg);color:var(--fg);border-radius:6px;cursor:pointer}
</style></head><body>

<div id="top">
  <h1>Groundedness labelling</h1>
  <div id="bar"><div id="fill"></div></div>
  <span id="count">0 / __N__</span>
  <button id="jump">next unlabelled</button>
  <button id="export">export labels</button>
</div>

<div id="rule">
  <p><b>One question per claim: does the passage below it actually say the claim?</b></p>
  <p>Not whether the claim is true. Not whether it answers the question. Not whether
  it is the figure you would have picked &mdash; a wrong-but-quoted number is still
  supported. Only: is it in the passage.</p>
  <p>Keys: <kbd>y</kbd> supported &middot; <kbd>n</kbd> not supported &middot;
  <kbd>j</kbd>/<kbd>k</kbd> or <kbd>&darr;</kbd>/<kbd>&uarr;</kbd> move.
  Labelling with <kbd>y</kbd>/<kbd>n</kbd> advances on its own. Progress is saved in
  this browser, so you can close the page and come back.</p>
  <p>Each card has a <b>note</b> box &mdash; use it for anything the label alone
  cannot carry: a missing column header, a qualifier the claim dropped, a judgement
  you were unsure about. Notes are exported alongside the labels. While you are
  typing in one, <kbd>y</kbd>/<kbd>n</kbd> type instead of voting; press
  <kbd>esc</kbd> to leave the box.</p>
</div>

<main>__BLOCKS__</main>

<div id="out">
  <div class="row">
    <b style="font-size:13px">Paste this back into the chat</b>
    <button id="copy">copy</button>
    <button id="dl">download .csv</button>
    <button id="hide" style="margin-left:auto">close</button>
  </div>
  <textarea id="csv" readonly></textarea>
</div>

<script>
const IDS = __IDS__, KEY = "groundedness.v1";
const cards = [...document.querySelectorAll(".card")];
let labels = {}, notes = {}, here = 0;

// localStorage can throw in a private window or with site data blocked, and
// can come back with nothing. Either way the page must still work, so every
// read and write is guarded and an unreadable store simply means a blank pass.
try {
  const st = JSON.parse(localStorage.getItem(KEY) || "{}") || {};
  // v1 stored the labels at the top level; v2 keeps labels and notes side by
  // side. An older store is read rather than discarded so a pass already made
  // is not lost by opening a rebuilt page.
  labels = st.labels || st || {};
  notes = st.notes || {};
} catch (e) { labels = {}; notes = {}; }
const save = () => {
  try { localStorage.setItem(KEY, JSON.stringify({ labels, notes })); } catch (e) {}
};

function paint(i) {
  const c = cards[i], v = labels[c.dataset.pid] || "";
  c.classList.toggle("done", v === "y" || v === "n");
  c.classList.toggle("no", v === "n");
  c.querySelectorAll(".vote button[data-v]").forEach(b =>
    b.setAttribute("aria-pressed", b.dataset.v === v && v !== ""));
  c.querySelector(".mark").textContent =
    v === "y" ? "supported" : v === "n" ? "not supported" : "";
  c.classList.toggle("noted", !!(notes[c.dataset.pid] || "").trim());
}

function tally() {
  const n = IDS.filter(id => labels[id]).length;
  document.getElementById("count").textContent = n + " / " + IDS.length;
  document.getElementById("fill").style.width = (100 * n / IDS.length) + "%";
}

function focus(i, scroll = true) {
  here = Math.max(0, Math.min(cards.length - 1, i));
  cards.forEach((c, j) => c.classList.toggle("here", j === here));
  if (scroll) cards[here].scrollIntoView({ block: "center", behavior: "smooth" });
}

function set(i, v) {
  const id = cards[i].dataset.pid;
  if (v) labels[id] = v; else delete labels[id];
  save(); paint(i); tally();
}

cards.forEach((c, i) => {
  c.querySelectorAll(".vote button[data-v]").forEach(b => b.onclick = () => {
    set(i, b.dataset.v);
    // Advance only on a verdict, never on "clear" -- clearing a label means
    // reconsidering this claim, so jumping away from it would be wrong.
    if (b.dataset.v) { focus(i + 1); } else { focus(i, false); }
  });
  c.onclick = e => {
    if (!e.target.closest("button") && e.target.tagName !== "TEXTAREA") focus(i, false);
  };
  const ta = c.querySelector(".note");
  ta.value = notes[c.dataset.pid] || "";
  ta.oninput = () => {
    const t = ta.value;
    if (t.trim()) notes[c.dataset.pid] = t; else delete notes[c.dataset.pid];
    save(); paint(i);
  };
  // Escape leaves the box so the y/n keys work again without reaching for the
  // mouse; while the box has focus those keys must type, not vote.
  ta.onkeydown = e => { if (e.key === "Escape") { ta.blur(); focus(i, false); } };
  paint(i);
});

document.onkeydown = e => {
  if (e.target.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey) return;
  const k = e.key.toLowerCase();
  if (k === "y" || k === "n") { set(here, k); focus(here + 1); }
  else if (k === "j" || e.key === "ArrowDown") focus(here + 1);
  else if (k === "k" || e.key === "ArrowUp") focus(here - 1);
  else if (k === "x" || k === "backspace") set(here, "");
  else return;
  e.preventDefault();
};

document.getElementById("jump").onclick = () => {
  const i = cards.findIndex(c => !labels[c.dataset.pid]);
  focus(i < 0 ? cards.length - 1 : i);
};

// A note can hold a comma, a quote or a newline, so every note is written as
// a quoted CSV field with inner quotes doubled. Written raw, one comma in a
// note would shift every later column by one.
const q = t => '"' + String(t).replace(/"/g, '""') + '"';
const text = () => "pair_id,my_label,my_note\n" +
  IDS.map(id => [id, labels[id] || "", q(notes[id] || "")].join(",")).join("\n") + "\n";

document.getElementById("export").onclick = () => {
  const miss = IDS.filter(id => !labels[id]).length;
  document.getElementById("csv").value =
    (miss ? "# " + miss + " pair(s) still unlabelled\n" : "") + text();
  document.getElementById("out").style.display = "block";
};
document.getElementById("hide").onclick = () =>
  document.getElementById("out").style.display = "none";
document.getElementById("copy").onclick = async () => {
  const ta = document.getElementById("csv");
  try { await navigator.clipboard.writeText(ta.value); }
  catch (e) { ta.select(); document.execCommand("copy"); }
  document.getElementById("copy").textContent = "copied";
  setTimeout(() => document.getElementById("copy").textContent = "copy", 1200);
};
document.getElementById("dl").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text()], { type: "text/csv" }));
  a.download = "groundedness_my_labels.csv";
  a.click(); URL.revokeObjectURL(a.href);
};

tally(); focus(0, false);
</script></body></html>
"""

if __name__ == "__main__":
    main()
