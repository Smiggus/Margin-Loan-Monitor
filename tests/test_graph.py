"""Graph construction on a tiny hand-made book: two families, three issuers, two lenders."""
import pandas as pd

from mlm.graph.build import build, holder_id

PLEDGES = pd.DataFrame([
    # holder, resolved, issuer, cik, ticker, source, insider, shares, value
    ("LGC Holdco", True, "NEWS CORP", "0001564708", "NWSA", "sec_proxy", True, 30_404_378, 8.0e8),
    ("LGC Holdco", True, "FOX CORP", "0001754301", "FOXA", "sec_proxy", True, 37_002_060, 2.2e9),
    ("Mr. Smith", True, "ACME BANK", "0000000001", "ACME", "sec_proxy", True, 10_000, 1.0e6),
    ("Mr. Smith", True, "OTHER CO", "0000000002", "OTHR", "sec_proxy", True, 5_000, 5.0e5),
    ("(insider, name not resolved)", False, "OTHER CO", "0000000002", "OTHR", "sec_proxy", True, 1_000, 1.0e5),
], columns=["holder", "holder_resolved", "issuer", "issuer_cik", "ticker", "source", "insider_filing",
            "shares_pledged", "pledged_value_usd"]).assign(status="active", last_seen="2025-05-01", url="u")
LENDERS = pd.DataFrame([
    ("LGC Holdco", "NEWS CORP", "0001564708", "sec_proxy", "u", "JPMorgan"),
    ("LGC Holdco", "FOX CORP", "0001754301", "sec_proxy", "u", "JPMorgan"),
    ("Mr. Smith", "ACME BANK", "0000000001", "sec_proxy", "u", "Lakeside Bank"),
], columns=["holder", "issuer", "issuer_cik", "source", "url", "lender"])
ISSUERS = pd.DataFrame([
    ("0001564708", "4833", "Television Broadcasting", "Transportation, Communications & Utilities"),
    ("0001754301", "4833", "Television Broadcasting", "Transportation, Communications & Utilities"),
    ("0000000001", "6022", "State Commercial Banks", "Finance, Insurance & Real Estate"),
], columns=["issuer_cik", "sic", "sic_description", "sector"])


def test_holder_ids_never_merge_surnames_across_companies():
    assert holder_id("LGC Holdco", True, "1") == holder_id("LGC Holdco", True, "2")
    assert holder_id("Mr. Smith", True, "1") != holder_id("Mr. Smith", True, "2")
    assert holder_id("anything", False, "7") == "H:insider@7"
    assert holder_id("anything", False, "7", 100.0) != holder_id("anything", False, "7", 200.0)


def test_graph_shape():
    v, e = build(PLEDGES, LENDERS, ISSUERS)
    counts = v["type"].value_counts().to_dict()
    assert counts["holder"] == 4          # LGC Holdco, Smith@1, Smith@2, insider@2
    assert counts["issuer"] == 4
    assert counts["lender"] == 2
    assert counts["industry"] == 2 and counts["sector"] == 2
    rel = e["relationship"].value_counts().to_dict()
    assert rel["pledges"] == 5 and rel["insider_of"] == 5
    assert rel["lends_to"] == 3           # JPMorgan->LGC secured on News Corp and on Fox, Lakeside->Smith@1
    assert set(e["provenance"]) <= {"sec_proxy", "sec_13d", "sec_company_record"}


def test_every_edge_endpoint_is_a_vertex():
    v, e = build(PLEDGES, LENDERS, ISSUERS)
    ids = set(v["id"])
    assert set(e["src"]) <= ids and set(e["dst"]) <= ids
