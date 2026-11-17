"""Companies House (UK) REST API client: company search, profiles, charges, officers, PSCs, filed documents.

Used for the UK layer of the exposure graph: a charge registered by a holding company over shares
names the lender or security agent (`persons_entitled`).

Auth: Companies House REST keys use HTTP Basic with the key as username and a blank password.
Set COMPANIES_HOUSE_API_KEY; if the environment injects the key at the network layer instead, leave
it unset and requests still authenticate.
Rate limit: 600 requests per 5 minutes per key; the client paces itself to stay under it.
Every response is cached in the raw store under companies_house/..., so nothing is fetched twice.

Spike findings (see docs/DATA_QUALITY.md):
  - the profile's `has_charges` flag is unreliable (Tesco PLC reports False with 9 charges); always
    call /charges
  - random samples of holding companies find charges over *private subsidiary* shares, not listed
    stakes; seed lookups from named holders instead
  - charge detail is often only in the filed PDF ("see image for full details") -> document API"""
import json
import os
import re
import sys
import threading
import time

import requests

from mlm.storage import get_text, put_text

API = "https://api.company-information.service.gov.uk"
DOCS = "https://document-api.company-information.service.gov.uk"
SHARE_LIKE = re.compile(r"\bshares?\b|securities|stock|custody|investment account|\bplc\b", re.I)


class CompaniesHouse:
    def __init__(self, api_key: str | None = os.environ.get("COMPANIES_HOUSE_API_KEY"),
                 max_per_5min: int = 550, use_cache: bool = True):
        self.s = requests.Session()
        if api_key:
            self.s.auth = (api_key, "")
        self._interval = 300.0 / max_per_5min
        self._lock, self._last = threading.Lock(), 0.0
        self.use_cache = use_cache

    def _throttle(self) -> None:
        with self._lock:
            wait = self._interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _get(self, path: str, cache_key: str | None = None, base: str = API, **params) -> dict | None:
        if cache_key and self.use_cache:
            cached = get_text(cache_key)
            if cached is not None:
                return json.loads(cached)
        for attempt in range(4):
            self._throttle()
            r = self.s.get(base + path, params=params or None, timeout=30)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            if r.status_code != 200:
                print(f"companies house {r.status_code} {path}", file=sys.stderr)
                return None
            if cache_key:
                put_text(cache_key, r.text)
            return r.json()
        return None

    # ---------------------------------------------------------------- endpoints
    def search(self, q: str, n: int = 20) -> list[dict]:
        return (self._get("/search/companies", q=q, items_per_page=n) or {}).get("items", [])

    def advanced_search(self, **filters) -> list[dict]:
        """e.g. sic_codes="64209", company_status="active", size=5000, start_index=0"""
        return (self._get("/advanced-search/companies", **filters) or {}).get("items", [])

    def profile(self, number: str) -> dict | None:
        return self._get(f"/company/{number}", f"companies_house/profile/{number}.json.gz")

    def charges(self, number: str) -> list[dict]:
        d = self._get(f"/company/{number}/charges", f"companies_house/charges/{number}.json.gz", items_per_page=100)
        return (d or {}).get("items", [])

    def officers(self, number: str) -> list[dict]:
        d = self._get(f"/company/{number}/officers", f"companies_house/officers/{number}.json.gz", items_per_page=100)
        return (d or {}).get("items", [])

    def pscs(self, number: str) -> list[dict]:
        d = self._get(f"/company/{number}/persons-with-significant-control",
                      f"companies_house/psc/{number}.json.gz", items_per_page=100)
        return (d or {}).get("items", [])

    def document_metadata(self, document_id: str) -> dict | None:
        return self._get(f"/document/{document_id}", f"companies_house/documents/{document_id}.json.gz", base=DOCS)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def charge_rows(number: str, name: str, charges: list[dict]) -> list[dict]:
        rows = []
        for c in charges:
            desc = ((c.get("particulars") or {}).get("description") or "")
            rows.append(dict(company_number=number, company_name=name, charge_code=c.get("charge_code"),
                             status=c.get("status"), created_on=c.get("created_on"),
                             delivered_on=c.get("delivered_on"), satisfied_on=c.get("satisfied_on"),
                             lenders=[p.get("name") for p in c.get("persons_entitled", [])],
                             description=desc[:1000], share_like=bool(SHARE_LIKE.search(desc)),
                             filing_link=(c.get("links") or {}).get("self")))
        return rows

    def find_exact(self, name: str) -> dict | None:
        """Exact (normalised) name match only: a loose match links the wrong company."""
        def norm(s: str) -> str:
            s = s.lower().replace("limited", "ltd").replace("ltd.", "ltd")
            return re.sub(r"[^a-z0-9]", "", s)
        return next((i for i in self.search(name, 10) if norm(i.get("title", "")) == norm(name)), None)


def charges_for_names(names: list[str]) -> list[dict]:
    """Seeded lookup: named holders -> exact company match -> charges."""
    ch, rows = CompaniesHouse(), []
    for name in names:
        hit = ch.find_exact(name)
        if hit:
            rows += ch.charge_rows(hit["company_number"], hit["title"], ch.charges(hit["company_number"]))
    return rows


def main() -> None:
    import pandas as pd
    from mlm import config
    pledges = pd.read_parquet(config.PROCESSED / "pledges.parquet")
    uk = re.compile(r"\b(limited|ltd\.?|plc|llp)\b", re.I)
    names = sorted({h for h in pledges["holder"] if uk.search(h)})
    rows = charges_for_names(names)
    out = pd.DataFrame(rows)
    out.to_parquet(config.PROCESSED / "uk_charges.parquet", index=False)
    print(f"{len(names)} UK-style holder names -> {out['company_number'].nunique() if len(out) else 0} companies, "
          f"{len(out)} charges ({int(out['share_like'].sum()) if len(out) else 0} over shares)")


if __name__ == "__main__":
    main()
