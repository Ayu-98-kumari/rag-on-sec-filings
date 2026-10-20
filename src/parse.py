"""Step 2 — turn the raw filing HTML into clean, ordered, labelled blocks.

Reads data/raw/*.html, writes one JSON file per filing into data/parsed/.
No searching, no embedding: this step only extracts readable text and keeps
tables intact.

    python -m src.parse

Four jobs, in order:
  1. Drop the invisible inline-XBRL header and any hidden elements. In FAF's
     10-K this removes the company's own CIK repeated 536 times.
  2. Keep each table as rows and columns instead of flattening it to a string
     of digits, and record whether its figures are in thousands or millions.
  3. Label every block with the 10-K section (Item 1, Item 1A, Item 7, ...) it
     belongs to, so later steps can filter by section.
  4. Drop page furniture: headers and footers that repeat across the document.
"""

import collections
import json
import re
import warnings
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from src.config import MANIFEST, PARSED as OUT, RAW, ROOT

# Tolerant on purpose: STC writes its headings differently from FAF and FNF,
# so a literal "Item 7." match finds nothing in STC's filing. Allows any
# whitespace (including non-breaking), optional bold markup, and . : or -.
ITEM_RE = re.compile(r"^item\s*(\d{1,2}[ABC]?)\s*[.:–—-]?\s*(.{0,200})$", re.I)

SECTION_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures", "5": "Market for Common Equity",
    "6": "Reserved", "7": "Management's Discussion and Analysis (MD&A)",
    "7A": "Market Risk Disclosures", "8": "Financial Statements",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "10": "Directors and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Related Transactions",
    "14": "Accountant Fees", "15": "Exhibits", "16": "Form 10-K Summary",
}

SCALE_RE = re.compile(r"in\s*[$€£]?\s*(thousand|million|billion)s?", re.I)
FURNITURE_MIN_REPEATS = 8   # a short line repeated this often is page furniture

# Many filers satisfy Item 8 with a cross-reference ("the information required by
# this item is included in our consolidated financial statements beginning on
# page F-1") and place the actual statements at the END of the document, after the
# numbered items. STC does exactly this, which left 313 blocks of balance sheets
# and notes labelled "Item 16. Form 10-K Summary". This marker locates where that
# financial-statements region begins so it can be relabelled to Item 8.
FSTMT_START_RE = re.compile(
    r"report of independent registered public accounting firm"
    r"|consolidated balance sheets?"
    r"|consolidated statements? of (income|operations)", re.I)

# Only these labels may be overwritten: trailing summary/exhibit items and
# unlabelled blocks. A filer that already puts its statements under Item 8 is
# left untouched.
REASSIGNABLE = {None, "15", "16", "9B"}

# If Item 8 collected more blocks than this, the filer put its statements
# inline and no reassignment is needed.
FPAGES_ITEM8_MAX = 5


def norm(s: str) -> str:
    """Collapse all whitespace, including non-breaking spaces."""
    return re.sub(r"[\s   ]+", " ", s or "").strip()


def strip_invisible(soup: BeautifulSoup) -> int:
    """Remove the inline-XBRL header, hidden elements, scripts and styles."""
    removed = 0
    for el in soup.find_all(re.compile(r"^ix:(header|hidden)$", re.I)):
        el.decompose(); removed += 1
    for el in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        el.decompose(); removed += 1
    for el in soup(["script", "style"]):
        el.decompose(); removed += 1
    return removed


IX_NONFRACTION = re.compile(r"^ix:nonfraction$", re.I)

# "us-gaap:NetCashProvidedByUsedInOperatingActivities"
#   -> "Net Cash Provided By Used In Operating Activities"
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def humanise_concept(name: str) -> str:
    """Turn an XBRL concept name into something a reader and a model can use.

    The namespace prefix carries no meaning for the reader -- every concept in
    these filings is us-gaap or a company extension -- so it is dropped, and
    the CamelCase is split into words.
    """
    local = name.split(":", 1)[-1]
    return CAMEL_BOUNDARY.sub(" ", local).strip()


def cell_concepts(td) -> list[str]:
    """The XBRL concept names tagged on numbers inside one table cell.

    These come from the inline-XBRL element that wraps the figure:

        <ix:nonFraction name="us-gaap:Revenues" scale="3">2,921,636</...>

    strip_invisible() removes ix:header and ix:hidden but NOT ix:nonFraction,
    so these elements are still in the tree at this point. The reason they were
    invisible to us before is only that get_text() returns a cell's text and
    ignores its elements' attributes.
    """
    out = []
    for el in td.find_all(IX_NONFRACTION):
        name = el.get("name")
        if name and name not in out:
            out.append(name)
    return out


def table_rows(table) -> tuple[list[list[str]], list[str], int]:
    """Extract a table as rows, and recover the row labels the filer omitted.

    This is the fix for the hardest defect measured in Step 7: a figure and its
    meaning living apart. Stewart's income statement labels its subtotal
    "Operating revenues" and leaves the cell beside the true total EMPTY,
    because a human reads the total from its position at the foot of a column.
    A retrieved chunk has no position, so the model read the labelled row above
    and answered $2.86bn against a true $2.92bn -- grounded, cited, and wrong.

    The filer already told us which row is which. Every figure is wrapped in an
    ix:nonFraction element naming the concept it reports, so when a row's label
    cell is blank and its figures carry a concept, the label is filled in from
    that concept. The number is never altered and a label that exists is never
    overwritten; only empty cells are filled.

    Returns (rows, concepts, n_filled) -- the rows, the distinct concepts the
    table reports, and how many labels had to be recovered. The count is
    returned rather than logged because a parser that silently edits its own
    output is a parser you cannot audit.
    """
    rows, concepts, n_filled = [], [], 0
    for tr in table.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        cells = [norm(td.get_text(" ", strip=True)) for td in tds]
        if not any(cells):
            continue

        row_concepts = []
        for td in tds:
            for name in cell_concepts(td):
                if name not in row_concepts:
                    row_concepts.append(name)
        for name in row_concepts:
            if name not in concepts:
                concepts.append(name)

        # Recover the label only when the filer left it out.
        if cells and not cells[0] and row_concepts:
            cells[0] = humanise_concept(row_concepts[0])
            n_filled += 1

        rows.append(cells)
    return rows, concepts, n_filled


def is_toc_entry(el) -> bool:
    """Table-of-contents headings are hyperlinks; body headings are not."""
    return el.find_parent("a") is not None or el.find("a", href=True) is not None


def parse_filing(path: Path, meta: dict) -> dict:
    soup = BeautifulSoup(path.read_text(errors="ignore"), "lxml")
    n_hidden = strip_invisible(soup)

    raw_blocks, section, section_title = [], None, None

    # find_all returns document order, so blocks stay in reading sequence.
    # FAF wraps body text in <p>; FNF and STC use <div> only. So we accept both
    # and keep the *innermost* block element -- one that contains no further
    # block element -- which avoids emitting a container and its children twice.
    for el in soup.find_all(["p", "div", "table"]):
        if el.find_parent("table") is not None:
            continue  # text inside a table belongs to that table, not on its own

        if el.name in ("p", "div") and el.find(["p", "div", "table"]) is not None:
            continue  # a container, not a text block

        if el.name == "table":
            rows, concepts, n_filled = table_rows(el)
            if not rows:
                continue
            flat = " ".join(" ".join(r) for r in rows[:3])
            m = SCALE_RE.search(flat)
            raw_blocks.append({
                "type": "table", "section": section, "section_title": section_title,
                "rows": rows, "n_rows": len(rows),
                "n_cols": max(len(r) for r in rows),
                "unit_scale": m.group(1).lower() if m else None,
                "concepts": concepts,
                "labels_recovered": n_filled,
            })
            continue

        text = norm(el.get_text(" ", strip=True))
        if not text:
            continue

        m = ITEM_RE.match(text)
        if m and not is_toc_entry(el) and m.group(1).upper() in SECTION_TITLES:
            section = m.group(1).upper()
            section_title = SECTION_TITLES[section]
            raw_blocks.append({"type": "heading", "section": section,
                               "section_title": section_title, "text": text})
        else:
            raw_blocks.append({"type": "paragraph", "section": section,
                               "section_title": section_title, "text": text})

    # A table's unit caption ("in $ millions") is frequently the paragraph just
    # before it rather than a cell inside it. Backfill by looking back.
    for i, b in enumerate(raw_blocks):
        if b["type"] != "table" or b["unit_scale"]:
            continue
        for prev in reversed(raw_blocks[max(0, i - 3):i]):
            if prev["type"] != "paragraph":
                continue
            m = SCALE_RE.search(prev["text"])
            if m:
                b["unit_scale"] = m.group(1).lower()
                b["unit_scale_source"] = "preceding_caption"
                break

    # Relabel a trailing financial-statements region as Item 8 (see FSTMT_START_RE).
    # The decision is made AFTER the main pass, on the only evidence that matters:
    # whether Item 8 actually ended up with content. FAF and FNF put their
    # statements inline under Item 8 (hundreds of blocks) and are left alone.
    # STC's Item 8 is a two-block cross-reference, so its statements are elsewhere.
    n_reassigned = 0
    item8_own = sum(1 for b in raw_blocks if b.get("section") == "8")
    if item8_own <= FPAGES_ITEM8_MAX:
        start_idx = next(
            (i for i, b in enumerate(raw_blocks)
             if b["type"] == "paragraph" and FSTMT_START_RE.search(b["text"])),
            None)
        if start_idx is not None:
            for b in raw_blocks[start_idx:]:
                if b.get("section") in REASSIGNABLE:
                    b["section"] = "8"
                    b["section_title"] = SECTION_TITLES["8"]
                    b["section_source"] = "f_pages_reassigned"
                    n_reassigned += 1

    # Job 4: drop page furniture — short lines repeating many times.
    counts = collections.Counter(
        b["text"] for b in raw_blocks
        if b["type"] == "paragraph" and len(b["text"]) < 90
    )
    furniture = {t for t, n in counts.items() if n >= FURNITURE_MIN_REPEATS}
    blocks = [b for b in raw_blocks
              if not (b["type"] == "paragraph" and b["text"] in furniture)]

    for i, b in enumerate(blocks):
        b["order"] = i
        b["block_id"] = f"{meta['ticker']}_{meta['form'].replace('-', '')}_{meta['period']}_{i:05d}"

    return {
        "meta": meta,
        "stats": {
            "hidden_elements_removed": n_hidden,
            "blocks": len(blocks),
            "headings": sum(1 for b in blocks if b["type"] == "heading"),
            "paragraphs": sum(1 for b in blocks if b["type"] == "paragraph"),
            "tables": sum(1 for b in blocks if b["type"] == "table"),
            "tables_with_scale": sum(1 for b in blocks
                                     if b["type"] == "table" and b["unit_scale"]),
            "tables_with_concepts": sum(1 for b in blocks
                                        if b.get("concepts")),
            "tagged_numbers": sum(len(b.get("concepts") or []) for b in blocks),
            "row_labels_recovered": sum(b.get("labels_recovered", 0)
                                        for b in blocks),
            "furniture_lines_dropped": len(furniture),
            "fpages_blocks_reassigned_to_item8": n_reassigned,
            "sections_found": sorted({b["section"] for b in blocks if b["section"]},
                                     key=lambda s: (int(re.sub(r"\D", "", s)), s)),
            "unsectioned_blocks": sum(1 for b in blocks if not b["section"]),
            "text_chars": sum(len(b.get("text", "")) for b in blocks),
        },
        "blocks": blocks,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(MANIFEST)

    print(f"{'filing':30s} {'blocks':>7s} {'head':>5s} {'para':>6s} "
          f"{'tbl':>5s} {'scale':>6s} {'furn':>5s} {'nosec':>6s}")
    print("-" * 80)

    totals = collections.Counter()
    for _, row in man.iterrows():
        path = ROOT / row["local_path"]
        meta = {k: (int(row[k]) if k == "cik" else str(row[k]))
                for k in ("ticker", "cik", "company", "form", "period",
                          "fiscal_year", "accession")}
        doc = parse_filing(path, meta)
        s = doc["stats"]

        dest = OUT / f"{meta['ticker']}_{meta['form'].replace('-', '')}_{meta['period']}.json"
        dest.write_text(json.dumps(doc, indent=1))

        print(f"{path.name[:30]:30s} {s['blocks']:7d} {s['headings']:5d} "
              f"{s['paragraphs']:6d} {s['tables']:5d} {s['tables_with_scale']:6d} "
              f"{s['furniture_lines_dropped']:5d} {s['unsectioned_blocks']:6d}")
        for k in ("blocks", "tables", "paragraphs", "headings"):
            totals[k] += s[k]

    print("-" * 80)
    print(f"{'TOTAL':30s} {totals['blocks']:7d} {totals['headings']:5d} "
          f"{totals['paragraphs']:6d} {totals['tables']:5d}")
    print(f"\nwrote {len(man)} files -> {OUT.relative_to(ROOT)}/")

    # Verification: every 10-K should expose the four sections that matter.
    print("\n=== section coverage check (10-Ks) ===")
    must = ["1", "1A", "7", "8"]
    for f in sorted(OUT.glob("*10K*.json")):
        found = set(json.loads(f.read_text())["stats"]["sections_found"])
        missing = [m for m in must if m not in found]
        status = "OK" if not missing else f"MISSING {missing}"
        print(f"  {f.stem:28s} {len(found):2d} sections   {status}")


if __name__ == "__main__":
    main()
