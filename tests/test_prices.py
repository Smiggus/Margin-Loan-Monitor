import pandas as pd

from mlm.ingest.prices import pledge_values


def _as_traded(closes, splits):
    """Mirror of the yfinance-source arithmetic: undo every split after each date."""
    ratio = pd.Series(splits).replace(0, 1.0)
    after = ratio[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
    return list(pd.Series(closes) * after)


def test_reverse_split_restores_traded_price():
    # 1-for-100 reverse split on day 3: adjusted history shows $100, the stock really traded at $1
    assert _as_traded([100.0, 100.0, 100.0, 100.0], [0, 0, 0.01, 0]) == [1.0, 1.0, 100.0, 100.0]


def test_forward_split_restores_traded_price():
    # 4-for-1 split on day 2: adjusted history shows $25, the stock really traded at $100
    assert _as_traded([25.0, 25.0, 25.0], [0, 4.0, 0]) == [100.0, 25.0, 25.0]


def test_pledge_value_uses_close_on_or_before_filing_date():
    pledges = pd.DataFrame([dict(ticker="ORCL", shares_pledged=346e6, last_seen="2025-09-21")])
    prices = pd.DataFrame([
        dict(date="2025-09-18", ticker="ORCL", close=290.0, close_as_traded=290.0),
        dict(date="2025-09-19", ticker="ORCL", close=283.46, close_as_traded=283.46),
        dict(date="2025-09-22", ticker="ORCL", close=300.0, close_as_traded=300.0),
    ])
    out = pledge_values(pledges, prices)
    assert out.loc[0, "pledged_value_usd"] == 346e6 * 283.46
