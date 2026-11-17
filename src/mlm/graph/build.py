"""Build the exposure graph as two tables (vertices, edges) from the processed filings data.

Vertex types
  holder   person, family vehicle or fund that pledged shares
  issuer   listed company whose shares are pledged
  lender   bank or broker named as the secured party
  industry SIC industry of the issuer
  sector   SIC division of the industry

Edge types (src -> dst)
  pledges      holder -> issuer   shares_pledged, pledged_value_usd, status
  insider_of   holder -> issuer   the holder is a director/officer/affiliate (wrong-way risk)
  lends_to     lender -> holder   secured on that holder's pledge (issuer kept as an attribute)
  in_industry  issuer -> industry
  in_sector    industry -> sector

Every edge carries `provenance` (sec_proxy | sec_13d | sec_company_record | synthetic)."""
import re

import pandas as pd

from mlm import config
from mlm.ingest.edgar.extract import norm

ENTITY = re.compile(r"\b(llc|lp|l p|ltd|limited|inc|corp|corporation|trust|foundation|partners|holdings?|holdco|"
                    r"fund|capital|management|group|sa de cv|plc|gmbh|ag|bv|nv)\b")


def holder_id(holder: str, resolved: bool, issuer_cik: str, shares: float | None = None) -> str:
    """Entities and full names are global ids; surnames and unresolved insiders are scoped to the issuer,
    so two different 'Mr. Smith's at two companies never merge."""
    n = norm(holder)
    if not resolved:  # each unnamed footnote pledge is its own (unknown) holder
        return f"H:insider@{issuer_cik}" + (f"#{int(shares)}" if shares == shares and shares is not None else "")
    if ENTITY.search(n) or (len(n.split()) >= 2 and not re.match(r"^(mr|ms|mrs|dr)\b", holder.lower())):
        return f"H:{n}"
    return f"H:{n}@{issuer_cik}"


def build(pledges: pd.DataFrame, lenders: pd.DataFrame, issuers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pledges.copy()
    p["issuer_cik"] = p["issuer_cik"].fillna("").replace("", None).fillna(p["issuer"].map(norm))
    p["holder_id"] = [holder_id(h, r, c, n) for h, r, c, n in
                      zip(p["holder"], p["holder_resolved"], p["issuer_cik"], p["shares_pledged"])]
    p["issuer_id"] = "S:" + p["issuer_cik"].astype(str)
    iss = issuers.set_index("issuer_cik") if len(issuers) else pd.DataFrame()

    # ---------------- vertices
    v_hold = (p.groupby("holder_id").agg(name=("holder", "first")).reset_index().rename(columns={"holder_id": "id"})
              .assign(type="holder"))
    v_iss = (p.groupby("issuer_id").agg(name=("issuer", "first"), ticker=("ticker", "first"),
                                        cik=("issuer_cik", "first")).reset_index().rename(columns={"issuer_id": "id"})
             .assign(type="issuer"))
    if len(iss):
        v_iss = v_iss.join(iss[["sic", "sic_description", "sector"]], on="cik")
    l = lenders.merge(p[["holder", "issuer", "holder_id", "issuer_id"]], on=["holder", "issuer"], how="inner")
    l["lender_id"] = "L:" + l["lender"].map(norm)
    v_lend = (l.groupby("lender_id").agg(name=("lender", "first")).reset_index().rename(columns={"lender_id": "id"})
              .assign(type="lender"))
    frames = [v_hold, v_iss, v_lend]
    e_ind, e_sec = pd.DataFrame(), pd.DataFrame()
    if "sic" in v_iss:
        known = v_iss.dropna(subset=["sic"])
        v_ind = (known.groupby("sic").agg(name=("sic_description", "first"), sector=("sector", "first")).reset_index()
                 .assign(id=lambda d: "I:" + d["sic"].astype(str), type="industry"))
        v_sec = (pd.DataFrame({"name": sorted(known["sector"].dropna().unique())})
                 .assign(id=lambda d: "X:" + d["name"].map(norm), type="sector"))
        frames += [v_ind[["id", "type", "name"]], v_sec]
        e_ind = known.assign(src=known["id"], dst="I:" + known["sic"].astype(str), relationship="in_industry",
                             provenance="sec_company_record")[["src", "dst", "relationship", "provenance"]]
        e_sec = v_ind.assign(src=v_ind["id"], dst="X:" + v_ind["sector"].map(norm), relationship="in_sector",
                             provenance="sec_company_record")[["src", "dst", "relationship", "provenance"]]
    vertices = pd.concat(frames, ignore_index=True)

    # ---------------- edges
    value_col = "pledged_value_usd" if "pledged_value_usd" in p else None
    e_pledge = p.assign(src=p["holder_id"], dst=p["issuer_id"], relationship="pledges", provenance=p["source"])
    keep = ["src", "dst", "relationship", "provenance", "shares_pledged", "status", "last_seen", "url"]
    if value_col:
        keep.insert(5, value_col)
    e_pledge = e_pledge[keep]
    e_insider = (p[p["insider_filing"] | (p["source"] == "sec_13d")]
                 .assign(src=lambda d: d["holder_id"], dst=lambda d: d["issuer_id"], relationship="insider_of",
                         provenance=lambda d: d["source"])[["src", "dst", "relationship", "provenance"]])
    e_lend = l.assign(src=l["lender_id"], dst=l["holder_id"], relationship="lends_to", provenance=l["source"],
                      secured_on=l["issuer_id"])[["src", "dst", "relationship", "provenance", "secured_on", "url"]]
    edges = (pd.concat([e_pledge, e_insider, e_lend, e_ind, e_sec], ignore_index=True)
             .drop_duplicates(subset=["src", "dst", "relationship", "secured_on"]))
    return vertices, edges


def main() -> None:
    pledges = pd.read_parquet(config.PROCESSED / "pledges.parquet")
    lenders = pd.read_parquet(config.PROCESSED / "lenders.parquet")
    path = config.PROCESSED / "issuers.parquet"
    issuers = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    vertices, edges = build(pledges, lenders, issuers)
    vertices.to_parquet(config.PROCESSED / "graph_vertices.parquet", index=False)
    edges.to_parquet(config.PROCESSED / "graph_edges.parquet", index=False)
    print("vertices:", vertices["type"].value_counts().to_dict())
    print("edges:   ", edges["relationship"].value_counts().to_dict())


if __name__ == "__main__":
    main()
