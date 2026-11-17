# Data quality

What the extracted data can and cannot be trusted for. Every number here comes from hand-checking random
samples of the pipeline output against the source sentence.

## Coverage (2024-2025 filings)

| Item | Count |
|---|---|
| Filings searched (proxy DEF 14A + Schedule 13D) | 2,539 documents |
| Proxies with an actual pledge | ~425 |
| 13Ds with an actual pledge | ~270 |
| Pledge edges (holder -> issuer), latest snapshot | 540 |
| ... with a named holder | 346 (64%) |
| ... with a share count | 381 |
| ... with a dollar value | 329, total $251bn |
| Issuers | 361 across 10 SIC sectors |
| Named lender links | ~110, 47 lenders |

## Precision (hand-checked samples)

| Output | Sample | Real | Notes |
|---|---|---|---|
| Pledge edges | 25 | 21-22 (~86%) | false positives are pledging-policy text and generic margin-account wording |
| Lender links | 15 | ~10 (~65%) | noise from long 13D legal text, e.g. noteholder agreements naming agents |

## Rules that matter

- **Latest snapshot.** Proxies restate pledges every year. Only each issuer's latest proxy is kept, so 2024
  and 2025 counts for the same person are never added together.
- **Shadow duplicates.** The same pledge often appears as an unnamed footnote and as named prose. The unnamed
  copy is dropped when a named pledge at the same issuer has the same share count. The same person reported by
  a proxy and a 13D ("Mr. Zage" / "ZAGE GEORGE RAYMOND III") is merged on issuer + shares + surname.
- **Footnotes.** Passages are split at footnote markers "(n)" so a name in footnote (2) is never read as the
  holder of footnote (1). This is what fixed Tesla being attributed to "The Vanguard".
- **Holder identity.** Entities and full names are global ids; surname-only and unnamed holders are scoped to
  one issuer, so two different "Mr. Smith"s never merge.
- **Prices.** Pledged value = shares x the price *as traded* on the filing date. Provider closes are
  split-adjusted to today, so every later split is undone first. Prices are downloaded up to today, otherwise
  splits after the window are hidden (this produced a $10MM share price for Conduit Pharmaceuticals before
  the fix).

## Known limitations

- 36% of holders are unnamed: proxy footnotes refer to a table row. Linking footnotes to table rows (or an
  LLM extraction pass) is the next improvement.
- Lender precision is the weakest output. Invacare's noteholder group (Cantor Fitzgerald, GLAS Trust, Bank of
  America) is a restructuring, not a share pledge, and still appears.
- Foreign share classes are not converted (America Movil B shares vs ADRs); such rows are flagged
  `foreign_share_class` and left unvalued.
- 37 tickers have no price (delisted or renamed since filing).
- 13D "Item 6" text sometimes describes a pledge that has since been released; `status` catches only
  explicit wording ("fully repaid", "released").
- UK layer: Companies House works but needs seeding from named holders (see `mlm.ingest.companies_house`).
