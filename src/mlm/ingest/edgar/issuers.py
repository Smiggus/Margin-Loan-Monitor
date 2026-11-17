"""Step 4: issuer reference data from SEC company records (data.sec.gov/submissions).

Adds industry (SIC code + description), sector (SIC division), tickers, exchange and state of
incorporation for every issuer that appears in pledges.parquet."""
import json

import pandas as pd

from mlm import config
from mlm.ingest.edgar.client import SecClient
from mlm.storage import get_text, put_text

# SIC divisions, the standard top level of the SIC hierarchy
SIC_DIVISIONS = [
    (100, 999, "Agriculture, Forestry & Fishing"),
    (1000, 1499, "Mining & Energy"),
    (1500, 1799, "Construction"),
    (2000, 3999, "Manufacturing"),
    (4000, 4999, "Transportation, Communications & Utilities"),
    (5000, 5199, "Wholesale Trade"),
    (5200, 5999, "Retail Trade"),
    (6000, 6799, "Finance, Insurance & Real Estate"),
    (7000, 8999, "Services"),
    (9100, 9999, "Public Administration & Other"),
]


def sic_division(sic: str | int | None) -> str:
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return "Unclassified"
    for lo, hi, name in SIC_DIVISIONS:
        if lo <= code <= hi:
            return name
    return "Unclassified"


def fetch_issuer(client: SecClient, cik: str) -> dict:
    key = f"sec/submissions/{int(cik)}.json.gz"
    raw = get_text(key)
    if raw is None:
        r = client.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
        if r is None or r.status_code != 200:
            return dict(issuer_cik=cik)
        raw = r.text
        put_text(key, raw)
    d = json.loads(raw)
    return dict(
        issuer_cik=cik, sec_name=d.get("name"), sic=d.get("sic"), sic_description=d.get("sicDescription"),
        sector=sic_division(d.get("sic")), tickers=",".join(d.get("tickers") or []),
        exchanges=",".join(e for e in (d.get("exchanges") or []) if e),
        state_of_incorporation=d.get("stateOfIncorporation"), category=d.get("category"))


def main() -> None:
    pledges = pd.read_parquet(config.PROCESSED / "pledges.parquet")
    ciks = sorted({c for c in pledges["issuer_cik"] if c})
    cache_path = config.RAW / "issuers.jsonl"
    cache = {}
    if cache_path.exists():
        cache = {j["issuer_cik"]: j for j in map(json.loads, cache_path.open())}
    client = SecClient()
    with cache_path.open("a") as out:
        for cik in ciks:
            if cik not in cache:
                cache[cik] = fetch_issuer(client, cik)
                out.write(json.dumps(cache[cik]) + "\n")
    issuers = pd.DataFrame([cache[c] for c in ciks])
    issuers.to_parquet(config.PROCESSED / "issuers.parquet", index=False)
    print(f"{len(issuers)} issuers; sectors:")
    print(issuers["sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
