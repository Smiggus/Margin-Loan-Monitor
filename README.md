# Margin Loan Monitor

Who has pledged listed shares as collateral, to which banks, and where those exposures cluster.

This project grew out of a Modern Data Architectures II project during my Master in Business Analytics and Data Science course at IE.
It is a streaming monitor that replayed March 2020 and compared fixed vs volatility-aware margin calls on seven fictional share-backed loans.
This repository extends it to **real, public exposures**: a graph of share pledges built from SEC filings, analysed with Spark GraphFrames, with a UK layer from Companies House to follow.

## What it does

```
EDGAR full-text search ──> filings (proxy DEF 14A, Schedule 13D) ──> raw store
        │
        ▼
pledge sentences ──> pledges (holder -> issuer, shares, status)      SEC company records ──> industry / sector
                 └─> lenders (lender -> holder, secured on issuer)   prices (as traded)    ──> pledged $ value
        │
        ▼
exposure graph (holders, issuers, lenders, industries, sectors) ──> GraphFrames
        wrong-way risk · cross-pledgers · lender x sector · connected components · PageRank · $ by sector/lender
```

Current output (2024-2025 filings): **540 pledges, $251bn pledged, 361 issuers, 47 named lenders**.
See [docs/GRAPH_FINDINGS.md](docs/GRAPH_FINDINGS.md) and [docs/DATA_QUALITY.md](docs/DATA_QUALITY.md).

## Quick start

Requires Python 3.10+ and Java 17+ (for Spark).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # add ".[dev,s3]" for MinIO/S3
export SEC_USER_AGENT="Your Name research you@example.com"   # SEC requires a contact

mlm all          # or step by step:
mlm search       # EDGAR full-text search for pledge disclosures
mlm fetch        # download filings into the raw store, keep pledge sentences
mlm extract      # sentences -> pledges.csv, lenders.csv
mlm issuers      # SIC industry and sector per issuer
mlm prices       # daily prices; dollar value of each pledge
mlm graph        # vertex and edge tables
mlm analytics    # GraphFrames analytics -> docs/GRAPH_FINDINGS.md
mlm uk           # optional: Companies House charges for UK-registered holders

pytest           # 22 tests; GraphFrames tests are skipped without Java
```

## Layout

```
src/mlm/
  config.py                 paths, SEC contact, search terms, Spark packages
  storage.py                raw store: local disk or S3/MinIO (every download is cached)
  ingest/edgar/             search -> fetch -> extract -> issuers
  ingest/prices.py          PriceSource interface (yfinance today, datalake next)
  ingest/companies_house.py UK charges register client
  graph/build.py            vertices / edges with provenance
  graph/analytics.py        GraphFrames motifs, components, PageRank, exposure tables
tests/                      extraction rules pinned on real filing sentences; graph and motif tests
data/processed/             pledges.csv, lenders.csv (small, committed); parquet is rebuilt
docs/                       findings and data-quality notes
```

## Storage and data sources

| Setting | Default | Purpose |
|---|---|---|
| `MLM_RAWSTORE` | `local` | `s3://bucket/prefix` to keep raw downloads in MinIO/S3 |
| `MLM_S3_ENDPOINT` | | MinIO endpoint, e.g. `https://minio.example.com` |
| `MLM_PRICE_SOURCE` | `yfinance` | swap in another `PriceSource` (e.g. a personal datalake) |
| `COMPANIES_HOUSE_API_KEY` | | Companies House REST key (Basic auth, key as username) |

## Roadmap

1. UK layer: FCA National Storage Mechanism (director dealings, major holdings) -> Companies House charges.
2. Stress propagation: replay a sell-off through the graph; flag simultaneous calls on connected nodes.
3. Predict the call: Spark ML on volatility, liquidity, LTV and graph features.
4. Live and unattended: intraday prices, `spark-submit`, alerts.
