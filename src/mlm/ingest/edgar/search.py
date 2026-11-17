"""Step 1: find filings that mention share pledges, using EDGAR full-text search."""
import json
import sys

from mlm import config
from mlm.ingest.edgar.client import SecClient

URL = "https://efts.sec.gov/LATEST/search-index"


def search(client: SecClient, start: str = config.EDGAR_START, end: str = config.EDGAR_END) -> list[dict]:
    hits: dict[tuple, dict] = {}
    for kind, forms in config.EDGAR_FORMS.items():
        for phrase in config.EDGAR_PHRASES:
            offset, total = 0, 0
            while True:
                r = client.get(URL, params={"q": phrase, "forms": forms, "dateRange": "custom",
                                            "startdt": start, "enddt": end, "from": offset})
                if r is None or r.status_code != 200:
                    break
                block = r.json()["hits"]
                items, total = block["hits"], block["total"]["value"]
                for h in items:
                    src = h["_source"]
                    adsh, fname = h["_id"].split(":")
                    rec = hits.setdefault((adsh, fname), dict(
                        kind=kind, adsh=adsh, file=fname, form=src["form"], date=src["file_date"],
                        ciks=src["ciks"], names=src["display_names"], phrases=[]))
                    rec["phrases"].append(phrase)
                offset += len(items)
                if not items or offset >= min(total, 10_000):  # EDGAR caps a query at 10,000 hits
                    break
            print(f"{kind:5} {phrase:26} {total:>5}", file=sys.stderr)
    return list(hits.values())


def main() -> None:
    config.ensure_dirs()
    hits = search(SecClient())
    out = config.RAW / "edgar_hits.json"
    out.write_text(json.dumps(hits, indent=1))
    print(f"{len(hits)} documents -> {out}")


if __name__ == "__main__":
    main()
