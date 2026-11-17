"""Extraction rules, pinned on real sentences from 2024-2025 SEC filings."""
from mlm.ingest.edgar.extract import (find_lenders, holder_in, is_pledge_sentence, parse_display_name,
                                      shares_pledged)
from mlm.ingest.edgar.text import split_sentences

# ---------------------------------------------------------------- sentence splitting

def test_split_keeps_honorific_with_name():
    text = "Shares are held by a trust. Mr. Hancock has pledged 9,500,000 shares. Other text follows."
    sents = split_sentences(text)
    assert "Mr. Hancock has pledged 9,500,000 shares." in sents


def test_split_keeps_corporate_abbreviations():
    text = "Owned by Folco Development Corp. and pledged to UBS. Next sentence."
    assert split_sentences(text)[0] == "Owned by Folco Development Corp. and pledged to UBS."


# ---------------------------------------------------------------- real pledges are kept

REAL = [
    ("proxy", "Hancock entered into a margin loan with UBS, as agent, and certain lenders party thereto, pursuant to "
              "which he has pledged 9,500,000 shares of common stock as collateral", 9_500_000),
    ("proxy", "As of September 10, 2025, 30,404,378 shares of Class B Common Stock are pledged by LGC Holdco as "
              "collateral to loans.", 30_404_378),
    ("proxy", "(5) Includes 107,568 shares held by a family limited partnership that have been pledged as security "
              "for a bank guarantee.", 107_568),
    ("proxy", "Barrett Brown's ownership (a) includes (i) 2,324 shares owned by children living in his household, as "
              "to which he disclaims beneficial ownership, and (ii) 90,000 shares pledged as collateral for a credit "
              "facility", 90_000),
    ("13d", "As of December 17, 2024, there were 331,483,000 B Shares pledged as collateral.", 331_483_000),
]


def test_real_pledges_are_kept_with_correct_share_count():
    for kind, s, n in REAL:
        assert is_pledge_sentence(s, kind), s
        assert shares_pledged(s) == n, s


# ---------------------------------------------------------------- noise is dropped

NOISE = [
    ("13d", "In such instances, the positions held in the margin accounts are pledged as collateral security for the "
            "repayment of debit balances in the account, which may exist from time to time."),
    ("13d", "Common Stock positions held in margin accounts may be pledged as collateral security for the repayment "
            "of debit balances in the margin accounts."),
    ("proxy", "No pledging of common stock or other securities of the Corporation as collateral for margin accounts "
              "or loans."),
    ("proxy", "In 2024, our board of directors approved the issuance and donation of 181,961 shares of our common "
              "stock, the fourth installment of our Pledge 1% commitment, to the Remitly Foundation Fund."),
    ("proxy", "Unless otherwise indicated, none of the shares listed are pledged as security."),
    ("proxy", "Allow associates and Board members to engage in short-selling, margin transactions, trading in "
              "exchange-traded options or hedging or monetization transactions, or pledge Company stock."),
    ("13d", "Baker in his personal capacity agreed, without the prior written consent of the representatives of the "
            "Underwriters, not to (1) offer, pledge, sell, contract to sell any shares."),
]


def test_noise_is_dropped():
    for kind, s in NOISE:
        assert not is_pledge_sentence(s, kind), s


# ---------------------------------------------------------------- holders and lenders

def test_holder_names():
    assert holder_in("Longview Grindr has pledged 26,588,867 shares of common stock to certain lenders") \
        == "Longview Grindr"
    assert holder_in("Mr. Schlosberg has pledged 100,000 common shares.") == "Mr. Schlosberg"
    assert holder_in("The Carlino Family Trust has pledged an aggregate of 1,195,741 shares") \
        == "The Carlino Family Trust"
    assert holder_in("30,404,378 shares of Class B Common Stock are pledged by LGC Holdco as collateral") \
        == "LGC Holdco"


def test_lenders_are_canonicalised():
    s = "SFS has entered into two separate Collateral Assignment agreements with JPMorgan Chase Bank, N.A."
    assert find_lenders(s) == {"JPMorgan"}
    s = "A Reporting Person has pledged 96,019 Common Shares to Royal Bank of Canada to secure certain obligations"
    assert find_lenders(s) == {"RBC"}
    assert find_lenders("pledged in favour of Lakeside Bank under a credit agreement") == {"Lakeside Bank"}
    assert find_lenders("deposited with The Depository Trust Company") == set()


def test_display_name_parsing():
    assert parse_display_name("FOX CORP  (FOXA, FOX)  (CIK 0001754301)") == ("FOX CORP", "FOXA", "0001754301")
    assert parse_display_name("HELU CARLOS SLIM  (CIK 0001010139)") == ("HELU CARLOS SLIM", "", "0001010139")


def test_footnotes_are_split_and_neighbouring_names_ignored():
    from mlm.ingest.edgar.extract import pieces
    passage = ("Includes 235,998,721 shares pledged as collateral to secure certain personal indebtedness. \u200b (2) "
               "Includes shares beneficially owned by The Vanguard Group, of which The Vanguard Group has shared voting")
    first = pieces([passage])[0]
    assert first.endswith("indebtedness.")
    assert shares_pledged(first) == 235_998_721
    assert holder_in(first) is None


def test_shadow_duplicates_dropped():
    import pandas as pd
    from mlm.ingest.edgar.extract import drop_shadow_duplicates
    df = pd.DataFrame([dict(issuer_cik="1341439", holder="Mr. Ellison", holder_resolved=True, shares_pledged=346e6),
                       dict(issuer_cik="1341439", holder="(insider)", holder_resolved=False, shares_pledged=346e6),
                       dict(issuer_cik="1341439", holder="(insider)", holder_resolved=False, shares_pledged=5e3)])
    assert len(drop_shadow_duplicates(df)) == 2


def test_same_person_across_sources_merged():
    import pandas as pd
    from mlm.ingest.edgar.extract import drop_shadow_duplicates
    df = pd.DataFrame([dict(issuer_cik="1820144", holder="Mr. Zage", holder_resolved=True, shares_pledged=85926333.0),
                       dict(issuer_cik="1820144", holder="ZAGE GEORGE RAYMOND III", holder_resolved=True,
                            shares_pledged=85926333.0)])
    out = drop_shadow_duplicates(df)
    assert list(out["holder"]) == ["ZAGE GEORGE RAYMOND III"]


def test_latest_proxy_snapshot_drops_last_years_counts():
    import pandas as pd
    from mlm.ingest.edgar.extract import latest_proxy_snapshot
    df = pd.DataFrame([
        dict(issuer_cik="1318605", source="sec_proxy", last_seen="2024-04-29", shares_pledged=238_441_261),
        dict(issuer_cik="1318605", source="sec_proxy", last_seen="2025-09-17", shares_pledged=235_998_721),
        dict(issuer_cik="1318605", source="sec_13d", last_seen="2024-01-10", shares_pledged=1.0),
    ])
    out = latest_proxy_snapshot(df)
    assert sorted(out["shares_pledged"]) == [1.0, 235_998_721]
