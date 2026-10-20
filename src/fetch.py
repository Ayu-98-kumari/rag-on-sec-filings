"""Fetch the filing corpus from SEC EDGAR, reproducibly.

Resolves tickers -> CIK -> filing list via EDGAR's JSON APIs, downloads the
primary HTML document for each selected filing, pulls XBRL company facts, and
records everything in data/manifest.csv (committed, so the corpus is verifiable).

    python -m src.fetch

SEC fair-access rules: a descriptive User-Agent is required and requests are
capped near 10/s. See https://www.sec.gov/os/accessing-edgar-data
"""

import hashlib
import json
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import (MANIFEST, N_ANNUAL, N_QUARTERLY, RAW, ROOT, TICKERS,
                        XBRL)

# Explicit path rather than a bare load_dotenv(): that one searches upward from
# the caller's directory, so it depends on where the command was run from.
load_dotenv(ROOT / ".env")

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"

THROTTLE = 0.15  # seconds between requests, well under SEC's ~10/s ceiling


def session() -> requests.Session:
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua or "example.com" in ua:
        sys.exit(
            "Set SEC_USER_AGENT in .env to a real contact string, e.g.\n"
            '  SEC_USER_AGENT="messy-filings-rag yourname@email.com"\n'
            "SEC blocks requests without one."
        )
    s = requests.Session()
    s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
    return s


def get(s: requests.Session, url: str) -> requests.Response:
    time.sleep(THROTTLE)
    r = s.get(url, timeout=60)
    r.raise_for_status()
    return r


def resolve_ciks(s: requests.Session, tickers: list[str]) -> dict[str, dict]:
    """Map ticker -> {cik, title} using SEC's official ticker file."""
    data = get(s, TICKER_MAP_URL).json()
    by_ticker = {row["ticker"].upper(): row for row in data.values()}
    out = {}
    for t in tickers:
        if t.upper() not in by_ticker:
            sys.exit(f"Ticker {t} not found in SEC ticker file.")
        row = by_ticker[t.upper()]
        out[t] = {"cik": int(row["cik_str"]), "company": row["title"]}
    return out


def select_filings(s: requests.Session, cik: int) -> list[dict]:
    """Pick the latest N_ANNUAL 10-Ks and N_QUARTERLY 10-Qs (excluding amendments)."""
    sub = get(s, SUBMISSIONS_URL.format(cik=cik)).json()
    recent = sub["filings"]["recent"]
    rows = [
        {
            "form": recent["form"][i],
            "accession": recent["accessionNumber"][i],
            "filed_date": recent["filingDate"][i],
            "period": recent["reportDate"][i],
            "primary_doc": recent["primaryDocument"][i],
        }
        for i in range(len(recent["form"]))
    ]
    picked = []
    for form, n in (("10-K", N_ANNUAL), ("10-Q", N_QUARTERLY)):
        matches = sorted(
            (r for r in rows if r["form"] == form),
            key=lambda r: r["filed_date"],
            reverse=True,
        )
        if len(matches) < n:
            print(f"  warning: only {len(matches)} {form} filings available")
        picked.extend(matches[:n])
    return picked


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    XBRL.mkdir(parents=True, exist_ok=True)
    s = session()

    print("Resolving tickers -> CIK ...")
    companies = resolve_ciks(s, TICKERS)
    for t, meta in companies.items():
        print(f"  {t}: CIK {meta['cik']} ({meta['company']})")

    records = []
    for ticker, meta in companies.items():
        cik, company = meta["cik"], meta["company"]
        print(f"\n{ticker} ({company})")

        for f in select_filings(s, cik):
            acc_nodash = f["accession"].replace("-", "")
            url = DOC_URL.format(cik=cik, accession=acc_nodash, doc=f["primary_doc"])
            fy = f["period"][:4] if f["period"] else "unknown"
            local = RAW / f"{ticker}_{f['form'].replace('-', '')}_{f['period']}.html"

            if local.exists():
                body = local.read_bytes()
                print(f"  cached  {f['form']:5s} {f['period']}  {local.name}")
            else:
                body = get(s, url).content
                local.write_bytes(body)
                print(f"  fetched {f['form']:5s} {f['period']}  "
                      f"{len(body) / 1e6:.1f} MB  {local.name}")

            records.append({
                "ticker": ticker,
                "cik": cik,
                "company": company,
                "form": f["form"],
                "period": f["period"],
                "fiscal_year": fy,
                "filed_date": f["filed_date"],
                "accession": f["accession"],
                "url": url,
                "local_path": str(local.relative_to(ROOT)),
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            })

        # XBRL company facts: ground truth for numeric questions + the numeric verifier.
        facts_path = XBRL / f"{ticker}_companyfacts.json"
        if facts_path.exists():
            print(f"  cached  XBRL  {facts_path.name}")
        else:
            facts = get(s, FACTS_URL.format(cik=cik)).json()
            facts_path.write_text(json.dumps(facts))
            n_concepts = sum(len(v) for v in facts.get("facts", {}).values())
            print(f"  fetched XBRL  {n_concepts} concepts  {facts_path.name}")

    df = pd.DataFrame(records).sort_values(["ticker", "form", "period"])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)

    print(f"\n{len(df)} filings -> {MANIFEST.relative_to(ROOT)}")
    print(f"corpus size: {df['bytes'].sum() / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
