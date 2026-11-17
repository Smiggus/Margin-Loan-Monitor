"""Step 3: turn pledge sentences into graph edges.

Outputs (data/processed):
  pledges.parquet   one row per (holder, issuer): shares pledged, status, evidence, source URL
  lenders.parquet   one row per (holder, issuer, lender) where the lender is named

Rule-based on purpose: transparent and testable. Measured precision on hand-checked samples is
in docs/DATA_QUALITY.md; anything that fails a rule is dropped rather than guessed."""
import collections
import json
import re

import pandas as pd

from mlm import config

# ---------------------------------------------------------------- lenders
LENDER_CANON = {
    "JPMorgan": ["JPMorgan Chase Bank", "JPMorgan Chase", "J.P. Morgan", "JP Morgan", "JPMorgan"],
    "Citi": ["Citibank", "Citigroup", "Citi"],
    "Bank of America": ["Bank of America", "BofA", "Merrill Lynch"],
    "Morgan Stanley": ["Morgan Stanley"],
    "Goldman Sachs": ["Goldman Sachs", "Goldman"],
    "Wells Fargo": ["Wells Fargo"],
    "UBS": ["UBS"],
    "Credit Suisse": ["Credit Suisse"],
    "Deutsche Bank": ["Deutsche Bank"],
    "Barclays": ["Barclays"],
    "HSBC": ["HSBC"],
    "BNP Paribas": ["BNP Paribas"],
    "Societe Generale": ["Societe Generale", "Société Générale"],
    "RBC": ["Royal Bank of Canada", "RBC"],
    "TD": ["TD Bank", "Toronto-Dominion"],
    "BMO": ["Bank of Montreal", "BMO"],
    "Scotiabank": ["Scotiabank", "Bank of Nova Scotia"],
    "Nomura": ["Nomura"],
    "Mizuho": ["Mizuho"],
    "MUFG": ["MUFG", "Mitsubishi UFJ"],
    "SMBC": ["Sumitomo Mitsui"],
    "Jefferies": ["Jefferies"],
    "Raymond James": ["Raymond James"],
    "Schwab": ["Charles Schwab", "Schwab"],
    "Fidelity": ["Fidelity"],
    "Interactive Brokers": ["Interactive Brokers"],
    "BNY": ["BNY Mellon", "Bank of New York"],
    "Northern Trust": ["Northern Trust"],
    "State Street": ["State Street"],
    "Macquarie": ["Macquarie"],
    "Julius Baer": ["Julius Baer"],
    "Lombard Odier": ["Lombard Odier"],
    "Cantor Fitzgerald": ["Cantor Fitzgerald"],
    "Stifel": ["Stifel"],
    "Axos Bank": ["Axos Bank"],
    "East West Bank": ["East West Bank"],
    "PNC": ["PNC"],
    "Truist": ["Truist"],
    "Comerica": ["Comerica"],
    "Zions": ["Zions"],
    "Santander": ["Santander"],
    "Standard Chartered": ["Standard Chartered"],
    "DBS": ["DBS Bank"],
    "Abu Dhabi Commercial Bank": ["Abu Dhabi Commercial Bank"],
    "Apollo": ["Apollo"],
}
_ALIAS = {a: canon for canon, aliases in LENDER_CANON.items() for a in aliases}
KNOWN_LENDER = re.compile(r"(?<![A-Za-z])(" + "|".join(
    re.escape(a) for a in sorted(_ALIAS, key=len, reverse=True)) + r")(?![A-Za-z])")
GENERIC_BANK = re.compile(r"\b((?:[A-Z][\w&.'\-]*\s){1,4}(?:Bank|Bancorp|Banking Corporation|Trust Company))\b")
NOT_A_LENDER = re.compile(
    r"^(The\s|Federal Reserve|Depository|Investment Bank|Anticipated|Commercial Bank$|Agreement|Pledge|Credit|Loan|"
    r"Security|Such|Each|Any|Our|Its)", re.I)


def find_lenders(sentence: str) -> set[str]:
    found = {_ALIAS[m] for m in KNOWN_LENDER.findall(sentence)}
    masked = KNOWN_LENDER.sub(" ~ ", sentence)  # so "Royal Bank of Canada" is not also read as "Royal Bank"
    for g in GENERIC_BANK.findall(masked):
        g = g.strip()
        if not NOT_A_LENDER.search(g) and not KNOWN_LENDER.search(g):
            found.add(g)
    return found


# ---------------------------------------------------------------- pledge sentences
NUM = r"(\d{1,3}(?:,\d{3})+|\d{4,})"
SHARES_BEFORE = re.compile(NUM + r"\s+(?:[\w\-]+\s){0,4}shares", re.I)
SHARES_AFTER = re.compile(r"pledg\w*[^.;]{0,80}?" + NUM + r"\s+(?:\w+\s){0,3}shares", re.I)
NEGATIVE = re.compile(
    r"\b(none of|no shares|not pledged|have not pledged|has not pledged|no (?:director|executive|officer)|prohibit|"
    r"not permitted|may not|policy|negative pledge)", re.I)
HYPOTHETICAL = re.compile(
    r"pledge 1%|no pledging|there are no contracts|raises potential risks|board believes|anti-pledging|"
    r"hedging and pledging|pledging polic|pledging of (?:company|our)|may be pledged|may pledge|could be pledged|"
    r"from time to time|in such instances|exception may|permitted to pledge|forfeit|may be used as collateral|"
    r"agreed not to|short-selling|hedging or monetization|may be sold by the broker|tsr portion|"
    r"not to \(?\w\)? ?(?:offer|sell|pledge)|lock-up", re.I)
RELEASED = re.compile(r"fully repaid|was repaid|released from the pledge|pledge was released|terminated the pledge", re.I)
CONTEXT_13D = re.compile(r"loan|credit|collateral|secur|margin", re.I)

# holder names inside a sentence, most specific first
HOLDER_PATTERNS = [
    re.compile(r"\b((?:The\s)?(?:[A-Z][\w&.\-]*\s){1,5}(?:Trust|Foundation|LLC|L\.P\.|LP|Partners|Holdings|Holdco))\s+"
               r"(?:has|have)\s+pledged"),
    re.compile(r"\b((?:Mr|Ms|Mrs|Dr)\.\s+[A-Z][A-Za-z'\-]+(?:\s[A-Z][A-Za-z'\-]+)?)\s+(?:has|have)\s+pledged"),
    re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z]\.)?\s[A-Z][a-zA-Z'\-]+)\s+(?:has|have)\s+pledged"),
    re.compile(r"pledged by ((?:[A-Z][\w&.\-]*\s?){1,5})"),
    re.compile(r"owned by ((?:Mr|Ms|Mrs|Dr)\.\s+[A-Z][A-Za-z'\-]+|(?!The\b)[A-Z][a-z]+(?:\s[A-Z]\.)?\s[A-Z][a-zA-Z'\-]+)"),
    re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z]\.)?\s[A-Z][a-zA-Z\-]+)[’']s\s+(?:ownership|shares|holdings|beneficial)"),
    re.compile(r"\b((?:Mr|Ms|Mrs|Dr)\.\s+[A-Z][A-Za-z'\-]+)"),
]
HOLDER_STOP = re.compile(r"^(The Company|Company|Common Stock|Class [AB]|Board|Shares)$", re.I)


def is_pledge_sentence(s: str, kind: str) -> bool:
    if "pledg" not in s.lower() or NEGATIVE.search(s) or HYPOTHETICAL.search(s):
        return False
    if kind == "proxy":
        return bool(SHARES_BEFORE.search(s) or SHARES_AFTER.search(s) or re.search(r"margin (loan|account)", s, re.I))
    return bool(CONTEXT_13D.search(s))


def shares_pledged(s: str) -> int | None:
    i = s.lower().find("pledg")
    before = [m for m in SHARES_BEFORE.finditer(s) if m.end() <= i and i - m.end() < 120]
    if before:
        return int(before[-1].group(1).replace(",", ""))
    m = SHARES_AFTER.search(s)
    return int(m.group(1).replace(",", "")) if m else None


def holder_in(s: str) -> str | None:
    for rx in HOLDER_PATTERNS:
        m = rx.search(s)
        if m:
            name = m.group(1).strip().rstrip(",")
            if not HOLDER_STOP.search(name):
                return name
    return None


# ---------------------------------------------------------------- filing metadata
def parse_display_name(name: str) -> tuple[str, str, str]:
    """'FOX CORP  (FOXA, FOX)  (CIK 0001754301)' -> ('FOX CORP', 'FOXA', '0001754301')"""
    m = re.match(r"(.*?)\s+\(([A-Z0-9.\-, ]+)\)\s+\(CIK (\d+)\)", name)
    if m:
        return m.group(1).strip(), m.group(2).split(",")[0].strip(), m.group(3)
    m = re.match(r"(.*?)\s+\(CIK (\d+)\)", name)
    return (m.group(1).strip(), "", m.group(2)) if m else (name, "", "")


def norm(s: str | None) -> str:
    s = re.sub(r"\b(mr|ms|mrs|dr)\.?\s+", "", (s or "").lower())
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def issuer_and_default_holder(rec: dict) -> tuple[str, str, str, str | None] | None:
    if rec["kind"] == "proxy":
        name, ticker, cik = parse_display_name(rec["names"][0])
        return name, ticker, cik, None
    roles = rec.get("roles") or {}
    subject = (roles.get("Subject") or [None])[0]
    if not subject:
        return None
    filers = roles.get("Filed by") or roles.get("Filer") or []
    ticker, cik = "", ""
    for n in rec["names"]:
        nm, tk, ck = parse_display_name(n)
        if norm(nm)[:15] == norm(subject)[:15]:
            ticker, cik = tk, ck
    return subject, ticker, cik, (filers[0] if filers else None)


# ---------------------------------------------------------------- main
ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
DASHES = re.compile(r"[\u2012\u2013\u2014\u2015\u2212]")  # normalised to "-" in stored evidence
FOOTNOTE = re.compile(r"\s(?=\(\d{1,2}\)\s)")  # "... indebtedness. (2) Includes shares ..."


def pieces(passages: list[str]) -> list[str]:
    """Split stored passages again at footnote markers, so a name in footnote (2) is never read as the
    holder of a pledge in footnote (1)."""
    out = []
    for p in passages:
        p = DASHES.sub("-", ZERO_WIDTH.sub("", p))
        out += [x.strip() for x in FOOTNOTE.split(p) if x.strip()]
    return out


def latest_proxy_snapshot(pledges: pd.DataFrame) -> pd.DataFrame:
    """Proxies restate every pledge each year. Keep only the pledges in each issuer's latest proxy, so a
    2024 count is never added to the 2025 count for the same person. 13D rows are kept (already latest)."""
    proxy = pledges["source"] == "sec_proxy"
    latest = pledges[proxy].groupby("issuer_cik")["last_seen"].transform("max")
    stale = proxy & (pledges["last_seen"] < latest.reindex(pledges.index))
    return pledges[~stale.fillna(False)].reset_index(drop=True)


def drop_shadow_duplicates(pledges: pd.DataFrame) -> pd.DataFrame:
    """The same pledge often appears twice: once in a footnote (holder unresolved) and once in prose that
    names the holder. Drop the unresolved row when a named row at the same issuer has the same share count."""
    named = pledges[pledges["holder_resolved"] & pledges["shares_pledged"].notna()]
    seen = set(zip(named["issuer_cik"], named["shares_pledged"]))
    match = pd.Series([(c, n) in seen for c, n in zip(pledges["issuer_cik"], pledges["shares_pledged"])],
                      index=pledges.index)
    shadow = ~pledges["holder_resolved"].astype(bool) & match
    pledges = pledges[~shadow]
    # same person reported by two sources, e.g. proxy "Mr. Zage" and 13D "ZAGE GEORGE RAYMOND III":
    # same issuer, same share count, and the shorter name's surname appears in the longer one -> keep the longer
    drop = set()
    for _, grp in pledges[pledges["holder_resolved"] & pledges["shares_pledged"].notna()].groupby(
            ["issuer_cik", "shares_pledged"]):
        if len(grp) < 2:
            continue
        rows = sorted(grp.itertuples(), key=lambda r: len(norm(r.holder).split()), reverse=True)
        keep_tokens = set(norm(rows[0].holder).split())
        for r in rows[1:]:
            toks = norm(r.holder).split()
            if toks and toks[-1] in keep_tokens:
                drop.add(r.Index)
    return pledges.drop(index=list(drop)).reset_index(drop=True)


def extract(records) -> tuple[pd.DataFrame, pd.DataFrame, collections.Counter]:
    stats = collections.Counter()
    edges: dict[tuple, dict] = {}
    for rec in records:
        stats[f"{rec['kind']}_docs"] += 1
        good = [s for s in pieces(rec.get("passages", [])) if is_pledge_sentence(s, rec["kind"])]
        if not good:
            continue
        meta = issuer_and_default_holder(rec)
        if meta is None:
            continue
        issuer, ticker, cik, default_holder = meta
        stats[f"{rec['kind']}_docs_with_pledge"] += 1
        for s in good:
            holder = default_holder or holder_in(s)
            n = shares_pledged(s)
            # unnamed footnote pledges stay separate per share count; named holders merge across years
            key = (cik or norm(issuer), norm(holder) or f"unresolved:{n}")
            e = edges.setdefault(key, dict(
                issuer=issuer, ticker=ticker, issuer_cik=cik,
                holder=holder or "(insider, name not resolved)", holder_resolved=bool(holder),
                insider_filing=rec["kind"] == "proxy", source=f"sec_{rec['kind']}",
                shares_pledged=None, status="active", first_seen=rec["date"], last_seen=rec["date"],
                url=rec.get("url"), evidence=s[:400], lenders=set()))
            if n and (e["shares_pledged"] is None or rec["date"] >= e["last_seen"]):
                e.update(shares_pledged=n, evidence=s[:400], url=rec.get("url"))
            if RELEASED.search(s) and rec["date"] >= e["last_seen"]:
                e["status"] = "released"
            e["first_seen"] = min(e["first_seen"], rec["date"])
            e["last_seen"] = max(e["last_seen"], rec["date"])
            e["lenders"] |= find_lenders(s)
    pledges = drop_shadow_duplicates(latest_proxy_snapshot(pd.DataFrame(edges.values())))
    lenders = (pledges[["holder", "issuer", "issuer_cik", "source", "url", "lenders"]]
               .explode("lenders").dropna(subset=["lenders"]).rename(columns={"lenders": "lender"}))
    pledges["lenders"] = pledges["lenders"].map(lambda x: "; ".join(sorted(x)))
    stats.update(
        pledge_edges=len(pledges), pledge_edges_named_holder=int(pledges["holder_resolved"].sum()),
        pledge_edges_with_shares=int(pledges["shares_pledged"].notna().sum()),
        distinct_issuers=pledges["issuer_cik"].replace("", pd.NA).fillna(pledges["issuer"]).nunique(),
        lender_edges=len(lenders), distinct_lenders=lenders["lender"].nunique())
    return pledges, lenders, stats


def main() -> None:
    config.ensure_dirs()
    records = map(json.loads, (config.RAW / "edgar_passages.jsonl").open())
    pledges, lenders, stats = extract(records)
    pledges.to_parquet(config.PROCESSED / "pledges.parquet", index=False)
    lenders.to_parquet(config.PROCESSED / "lenders.parquet", index=False)
    pledges.to_csv(config.PROCESSED / "pledges.csv", index=False)
    lenders.to_csv(config.PROCESSED / "lenders.csv", index=False)
    print(json.dumps(stats, indent=1, default=int))
    print(lenders["lender"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
