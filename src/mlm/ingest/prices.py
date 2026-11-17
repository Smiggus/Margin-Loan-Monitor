"""Daily prices for every pledged ticker, and the dollar value of each pledge.

Prices come through a `PriceSource`, so the provider can change without touching the rest of the
pipeline. Today: `YFinanceSource` (cached per ticker in the raw store). Planned: a datalake source
reading the personal financial datalake; implement `daily_bars` and register it in SOURCES.

Select with MLM_PRICE_SOURCE (default "yfinance").

Pledged value = shares pledged x closing price on (or just before) the filing date that reported it.
Caveat: some issuers report pledges in local share classes that differ from the US-listed line
(e.g. America Movil B shares vs ADRs). Those rows are flagged, not converted."""
import io
import os
import warnings
from typing import Protocol

import pandas as pd

from mlm import config
from mlm.storage import RawStore, get_store

warnings.filterwarnings("ignore")
FOREIGN_SHARE_CLASS = {"AMX"}  # pledge reported in local shares, ticker is an ADR
BAR_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume", "stock_splits", "close_as_traded"]


class PriceSource(Protocol):
    name: str

    def daily_bars(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Long table with BAR_COLUMNS, one row per ticker per trading day.

        close            split-adjusted close (comparable across the whole history)
        close_as_traded  the price that day before any later split (pairs with share counts in filings)"""
        ...


class YFinanceSource:
    name = "yfinance"

    def __init__(self, store: RawStore | None = None):
        self.store = store or get_store()

    def _key(self, ticker: str) -> str:
        return f"prices/{self.name}/{ticker}.parquet"

    def _download(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        import yfinance as yf
        d = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False, threads=True,
                        group_by="ticker", actions=True)
        frames = []
        for t in tickers:
            sub = d[t] if isinstance(d.columns, pd.MultiIndex) else d
            if sub.empty:
                continue
            sub = sub.dropna(how="all").reset_index()
            sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
            if "stock_splits" not in sub:
                sub["stock_splits"] = 0.0
            sub = sub.dropna(subset=["close"]).sort_values("date")
            ratio = sub["stock_splits"].replace(0, 1.0).fillna(1.0)
            # yfinance closes are split-adjusted: undo every split that happened after each date
            after = ratio[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
            sub["close_as_traded"] = sub["close"] * after
            frames.append(sub.assign(ticker=t)[BAR_COLUMNS])
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BAR_COLUMNS)

    def daily_bars(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        cached, missing = [], []
        for t in tickers:
            raw = self.store.get(self._key(t))
            if raw is not None:
                df = pd.read_parquet(io.BytesIO(raw))
                if len(df) and "close_as_traded" in df and str(df["date"].min())[:10] <= start and str(df["date"].max())[:10] >= end[:7]:
                    cached.append(df)
                    continue
            missing.append(t)
        for i in range(0, len(missing), 100):
            got = self._download(missing[i:i + 100], start, end)
            for t, df in got.groupby("ticker"):
                buf = io.BytesIO()
                df.to_parquet(buf, index=False)
                self.store.put(self._key(t), buf.getvalue())
                cached.append(df)
        return pd.concat(cached, ignore_index=True) if cached else pd.DataFrame(columns=BAR_COLUMNS)


SOURCES = {"yfinance": YFinanceSource}


def get_source() -> PriceSource:
    return SOURCES[os.environ.get("MLM_PRICE_SOURCE", "yfinance")]()


def pledge_values(pledges: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    p = pledges.drop(columns=[c for c in ("close", "pledged_value_usd", "value_flag") if c in pledges]).copy()
    p["asof"] = pd.to_datetime(p["last_seen"]).astype("datetime64[ns]")
    px = (prices.dropna(subset=["close"]).assign(date=lambda d: pd.to_datetime(d["date"]).dt.tz_localize(None).astype("datetime64[ns]"))
          .sort_values("date"))
    px = px.rename(columns={"close": "close_adjusted"}).rename(columns={"close_as_traded": "close"})
    p = pd.merge_asof(p.sort_values("asof"), px[["date", "ticker", "close"]], left_on="asof", right_on="date",
                      by="ticker", direction="backward").drop(columns=["date", "asof"])
    p["pledged_value_usd"] = p["shares_pledged"] * p["close"]
    foreign = p["ticker"].isin(FOREIGN_SHARE_CLASS)
    p["value_flag"] = None
    p.loc[foreign, "value_flag"] = "foreign_share_class"
    p.loc[foreign, "pledged_value_usd"] = None
    return p


def main(start: str = "2023-12-01", end: str | None = None) -> None:
    # Download up to today: providers adjust history for every split to date, so a window that stops
    # early would hide later splits and leave as-traded prices wrong.
    end = end or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    pledges = pd.read_parquet(config.PROCESSED / "pledges.parquet")
    tickers = sorted({t for t in pledges["ticker"].dropna() if t})
    source = get_source()
    prices = source.daily_bars(tickers, start, end)
    prices.to_parquet(config.PROCESSED / "prices.parquet", index=False)
    valued = pledge_values(pledges, prices)
    valued.to_parquet(config.PROCESSED / "pledges.parquet", index=False)
    valued.to_csv(config.PROCESSED / "pledges.csv", index=False)
    got = valued["pledged_value_usd"].notna()
    print(f"[{source.name}] prices for {prices['ticker'].nunique()}/{len(tickers)} tickers; {got.sum()} pledges "
          f"valued, total ${valued.loc[got, 'pledged_value_usd'].sum() / 1e9:,.1f}bn")


if __name__ == "__main__":
    main()
