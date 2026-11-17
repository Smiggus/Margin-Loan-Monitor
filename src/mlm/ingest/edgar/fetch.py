"""Step 2: download each filing and keep only pledge-related sentences (resumable).

Raw HTML goes to the raw store (mlm.storage), so a re-run, a new extraction rule or a move to
MinIO never needs another trip to the SEC.

For Schedule 13D filings we also read the filing index to learn which party is the
subject company and which is the filer (the person or vehicle that pledged)."""
import html
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from mlm import config
from mlm.ingest.edgar.client import SecClient
from mlm.ingest.edgar.text import pledge_sentences
from mlm.storage import get_text, put_text

ARCHIVE = "https://www.sec.gov/Archives/edgar/data"
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    return WS.sub(" ", html.unescape(TAG.sub(" ", raw)))


def cached_get(client: SecClient, url: str, key: str) -> str | None:
    """Raw store first; download only on a miss, then keep the raw bytes."""
    text = get_text(key)
    if text is not None:
        return text
    r = client.get(url)
    if r is None or r.status_code != 200:
        return None
    put_text(key, r.text)
    return r.text


def filing_roles(client: SecClient, cik: int, adsh: str) -> dict:
    raw = cached_get(client, f"{ARCHIVE}/{cik}/{adsh.replace('-', '')}/{adsh}-index.htm",
                     f"sec/edgar/index/{cik}/{adsh}.html.gz")
    roles: dict[str, list[str]] = {}
    if raw:
        for m in re.finditer(r'companyName">\s*(.*?)\s*\((Subject|Filed by|Filer|Reporting)\)', raw, re.S):
            roles.setdefault(m.group(2), []).append(WS.sub(" ", TAG.sub("", m.group(1))).strip())
    return roles


def fetch_one(client: SecClient, hit: dict) -> dict:
    rec = dict(hit)
    nodash = hit["adsh"].replace("-", "")
    for cik in hit["ciks"]:
        url = f"{ARCHIVE}/{int(cik)}/{nodash}/{hit['file']}"
        raw = cached_get(client, url, f"sec/edgar/filings/{int(cik)}/{hit['adsh']}/{hit['file']}.gz")
        if raw is not None:
            rec["url"] = url
            rec["passages"] = pledge_sentences(html_to_text(raw))
            if hit["kind"] == "13d":
                rec["roles"] = filing_roles(client, int(cik), hit["adsh"])
            return rec
    rec["error"] = "fetch_failed"
    rec["passages"] = []
    return rec


def main(workers: int = 6) -> None:
    config.ensure_dirs()
    hits = json.loads((config.RAW / "edgar_hits.json").read_text())
    out_path = config.RAW / "edgar_passages.jsonl"
    done = set()
    if out_path.exists():
        # Failed fetches are retried: keep only successful records, then rewrite the file without the failures.
        recs = [j for j in map(json.loads, out_path.open()) if "error" not in j]
        out_path.write_text("".join(json.dumps(j) + "\n" for j in recs))
        done = {(j["adsh"], j["file"]) for j in recs}
    todo = [h for h in hits if (h["adsh"], h["file"]) not in done]
    print(f"{len(todo)} to fetch, {len(done)} already done", file=sys.stderr)
    client, lock, count, failed = SecClient(), threading.Lock(), [0], [0]
    with out_path.open("a") as out:
        def work(h):
            rec = fetch_one(client, h)
            with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
                count[0] += 1
                failed[0] += "error" in rec
                if count[0] % 200 == 0:
                    print(count[0], file=sys.stderr, flush=True)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, todo))
    print(f"fetched {count[0]} ({failed[0]} failed) -> {out_path}")
    if count[0] and failed[0] == count[0]:
        sys.exit("every fetch failed: check SEC_USER_AGENT (SEC requires a contact email) and network access")


if __name__ == "__main__":
    main()
