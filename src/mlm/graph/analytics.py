"""GraphFrames analytics on the exposure graph (Spark 3.5).

1. Wrong-way risk      motif: holder pledges shares of a company it is an insider of
2. Cross-pledgers      motif: one holder pledging two or more issuers
3. Lender x sector     motif: lender -> holder -> issuer -> industry -> sector
4. Clusters            connected components on holders, issuers and lenders
5. Centrality          PageRank and in-degree on the pledge/lending subgraph
6. Exposure            pledged value by sector and by named lender

Results go to data/processed/analytics/*.parquet and docs/GRAPH_FINDINGS.md."""
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, functions as F

from mlm import config


def spark_session(app: str = "exposure-graph") -> SparkSession:
    spark = (SparkSession.builder.appName(app)
             .config("spark.jars.packages", config.GRAPHFRAMES_PACKAGE)
             .config("spark.jars.repositories", config.SPARK_PACKAGES_REPO)
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir(str(config.DATA / "checkpoints" / "graph"))
    return spark


def load_graph(spark: SparkSession):
    from graphframes import GraphFrame
    v = pd.read_parquet(config.PROCESSED / "graph_vertices.parquet")
    e = pd.read_parquet(config.PROCESSED / "graph_edges.parquet")
    for c in ("shares_pledged", "pledged_value_usd"):
        if c in e:
            e[c] = e[c].astype("float64")
    v = v[["id", "type", "name"]].astype(str)
    e = e.astype({c: str for c in e.columns if c not in ("shares_pledged", "pledged_value_usd")})
    edges = spark.createDataFrame(e)
    for c in ("shares_pledged", "pledged_value_usd"):
        if c in edges.columns:  # unpriced pledges are null, not NaN, so sums skip them
            edges = edges.withColumn(c, F.when(F.isnan(c), None).otherwise(F.col(c)))
    return GraphFrame(spark.createDataFrame(v), edges)


def wrong_way(g):
    m = (g.find("(h)-[p]->(s); (h)-[i]->(s)")
         .filter((F.col("p.relationship") == "pledges") & (F.col("i.relationship") == "insider_of")))
    return m.select(F.col("h.name").alias("holder"), F.col("s.name").alias("issuer"),
                    F.col("p.shares_pledged").alias("shares_pledged"),
                    F.col("p.pledged_value_usd").alias("pledged_value_usd"), F.col("p.provenance").alias("source"))


def cross_pledgers(g):
    m = (g.find("(h)-[a]->(s1); (h)-[b]->(s2)")
         .filter((F.col("a.relationship") == "pledges") & (F.col("b.relationship") == "pledges")
                 & (F.col("s1.id") < F.col("s2.id"))))
    pairs = m.select(F.col("h.id").alias("holder_id"), F.col("h.name").alias("holder"),
                     F.col("s1.name").alias("issuer"), F.col("a.pledged_value_usd").alias("v1"),
                     F.col("s2.name").alias("issuer_2"), F.col("b.pledged_value_usd").alias("v2"))
    one = pairs.select("holder_id", "holder", "issuer", F.col("v1").alias("value"))
    two = pairs.select("holder_id", "holder", F.col("issuer_2").alias("issuer"), F.col("v2").alias("value"))
    return (one.union(two).distinct().groupBy("holder_id", "holder")
            .agg(F.count("*").alias("issuers"), F.sort_array(F.collect_set("issuer")).alias("issuer_names"),
                 F.sum("value").alias("pledged_value_usd"))
            .orderBy(F.desc("issuers"), F.desc("pledged_value_usd")))


def lender_sector(g):
    m = (g.find("(l)-[a]->(h); (h)-[b]->(s); (s)-[c]->(i); (i)-[d]->(x)")
         .filter((F.col("a.relationship") == "lends_to") & (F.col("b.relationship") == "pledges")
                 & (F.col("a.secured_on") == F.col("s.id"))
                 & (F.col("c.relationship") == "in_industry") & (F.col("d.relationship") == "in_sector")))
    return (m.groupBy(F.col("l.name").alias("lender"), F.col("x.name").alias("sector"))
            .agg(F.countDistinct("h.id").alias("borrowers"), F.countDistinct("s.id").alias("issuers"),
                 F.sum("b.pledged_value_usd").alias("pledged_value_usd"),
                 F.sort_array(F.collect_set("s.name")).alias("issuer_names"))
            .orderBy(F.desc("borrowers"), F.desc("pledged_value_usd")))


def clusters(g):
    from graphframes import GraphFrame
    core = GraphFrame(g.vertices.filter(F.col("type").isin("holder", "issuer", "lender")),
                      g.edges.filter(F.col("relationship").isin("pledges", "lends_to")))
    cc = core.connectedComponents()
    sizes = (cc.groupBy("component")
             .agg(F.count("*").alias("size"),
                  F.sum(F.when(F.col("type") == "lender", 1).otherwise(0)).alias("lenders"),
                  F.sum(F.when(F.col("type") == "issuer", 1).otherwise(0)).alias("issuers"),
                  F.sum(F.when(F.col("type") == "holder", 1).otherwise(0)).alias("holders"),
                  F.sort_array(F.collect_list(F.concat(
                      F.when(F.col("type") == "lender", "1").when(F.col("type") == "issuer", "2").otherwise("3"),
                      F.lit("|"), F.col("type"), F.lit(": "), F.col("name")))).alias("members"))
             .orderBy(F.desc("size")))
    return cc, sizes


def sector_exposure(g):
    m = (g.find("(h)-[p]->(s); (s)-[a]->(i); (i)-[b]->(x)")
         .filter((F.col("p.relationship") == "pledges") & (F.col("a.relationship") == "in_industry")
                 & (F.col("b.relationship") == "in_sector")))
    return (m.groupBy(F.col("x.name").alias("sector"))
            .agg(F.count("*").alias("pledges"), F.countDistinct("s.id").alias("issuers"),
                 F.sum("p.pledged_value_usd").alias("pledged_value_usd"))
            .orderBy(F.desc("pledged_value_usd")))


def lender_exposure(g):
    m = (g.find("(l)-[a]->(h); (h)-[p]->(s)")
         .filter((F.col("a.relationship") == "lends_to") & (F.col("p.relationship") == "pledges")
                 & (F.col("a.secured_on") == F.col("s.id"))))
    return (m.groupBy(F.col("l.name").alias("lender"))
            .agg(F.countDistinct("h.id").alias("borrowers"), F.countDistinct("s.id").alias("issuers"),
                 F.sum("p.pledged_value_usd").alias("pledged_value_usd"),
                 F.sort_array(F.collect_set("s.name")).alias("issuer_names"))
            .orderBy(F.desc("borrowers"), F.desc("pledged_value_usd")))


def centrality(g):
    from graphframes import GraphFrame
    core = GraphFrame(g.vertices.filter(F.col("type").isin("holder", "issuer", "lender")),
                      g.edges.filter(F.col("relationship").isin("pledges", "lends_to")))
    pr = core.pageRank(resetProbability=0.15, maxIter=20).vertices
    deg = core.inDegrees
    out = core.outDegrees
    return (pr.join(deg, "id", "left").join(out, "id", "left").fillna(0, ["inDegree", "outDegree"])
            .select("id", "type", "name", F.round("pagerank", 4).alias("pagerank"), "inDegree", "outDegree")
            .orderBy(F.desc("pagerank")))


def _fmt_bn(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"${x / 1e9:,.2f}bn" if x >= 1e9 else f"${x / 1e6:,.0f}MM"


def report(out: dict[str, pd.DataFrame], path: Path) -> None:
    ww, cp, ls, cs, ce, se, le = (out[k] for k in ("wrong_way", "cross_pledgers", "lender_sector",
                                                     "cluster_sizes", "centrality", "sector_exposure",
                                                     "lender_exposure"))
    lines = ["# Exposure graph: findings", "",
             "Generated by `mlm.graph.analytics` from public SEC filings (2024-2025). Loans and lenders are as "
             "disclosed; pledged value = shares pledged x close on the filing date.", "",
             "## 1. Wrong-way risk", "",
             f"{len(ww)} pledges are made by insiders or 5%+ holders of the same company, worth "
             f"{_fmt_bn(ww['pledged_value_usd'].sum())} where a price is available. If the stock falls, the "
             "borrower's wealth and the collateral fall together.", "",
             "| Holder | Issuer | Shares pledged | Value |", "|---|---|---|---|"]
    for r in ww.sort_values("pledged_value_usd", ascending=False).head(15).itertuples():
        lines.append(f"| {r.holder} | {r.issuer} | {r.shares_pledged:,.0f} | {_fmt_bn(r.pledged_value_usd)} |"
                     if pd.notna(r.shares_pledged) else f"| {r.holder} | {r.issuer} | n/a | n/a |")
    lines += ["", "## 2. Holders pledging more than one company", "", "| Holder | Issuers | Value |", "|---|---|---|"]
    for r in cp.head(15).itertuples():
        lines.append(f"| {r.holder} | {', '.join(r.issuer_names)} | {_fmt_bn(r.pledged_value_usd)} |")
    lines += ["", "## 3. Exposure by sector", "", "| Sector | Pledges | Issuers | Value |", "|---|---|---|---|"]
    for r in se.itertuples():
        lines.append(f"| {r.sector} | {r.pledges} | {r.issuers} | {_fmt_bn(r.pledged_value_usd)} |")
    lines += ["", "## 4. Named lenders", "", "| Lender | Borrowers | Issuers | Value (where priced) | Collateral |",
              "|---|---|---|---|---|"]
    for r in le.head(20).itertuples():
        lines.append(f"| {r.lender} | {r.borrowers} | {r.issuers} | {_fmt_bn(r.pledged_value_usd)} | "
                     f"{', '.join(list(r.issuer_names)[:4])} |")
    lines += ["", "## 5. Lender concentration by sector", "",
              "| Lender | Sector | Borrowers | Issuers | Value |", "|---|---|---|---|---|"]
    for r in ls.head(20).itertuples():
        lines.append(f"| {r.lender} | {r.sector} | {r.borrowers} | {', '.join(r.issuer_names[:4])} | "
                     f"{_fmt_bn(r.pledged_value_usd)} |")
    lines += ["", "## 6. Largest connected clusters", "",
              "| Size | Holders | Issuers | Lenders | Members (first 6) |", "|---|---|---|---|---|"]
    for r in cs.head(10).itertuples():
        members = [m.split("|", 1)[1] for m in list(r.members)]
        lines.append(f"| {r.size} | {r.holders} | {r.issuers} | {r.lenders} | {'; '.join(members[:6])} |")
    lines += ["", "## 7. Most central lenders and issuers (PageRank, unweighted)", "",
              "Unweighted PageRank rewards many small links, so bank holding companies whose directors each pledge a "
              "little rank high. Read it with the value-weighted tables above.", "", "| Type | Name | PageRank | In | Out |",
              "|---|---|---|---|---|"]
    for r in ce[ce["type"].isin(["lender", "issuer"])].head(15).itertuples():
        lines.append(f"| {r.type} | {r.name} | {r.pagerank} | {r.inDegree} | {r.outDegree} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    spark = spark_session()
    g = load_graph(spark)
    out_dir = config.PROCESSED / "analytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    cc, sizes = clusters(g)
    results = {"wrong_way": wrong_way(g), "cross_pledgers": cross_pledgers(g), "lender_sector": lender_sector(g),
               "cluster_sizes": sizes, "centrality": centrality(g), "sector_exposure": sector_exposure(g),
               "lender_exposure": lender_exposure(g)}
    out = {}
    for name, df in results.items():
        pdf = df.toPandas()
        pdf.to_parquet(out_dir / f"{name}.parquet", index=False)
        out[name] = pdf
        print(f"{name:15} {len(pdf):>5} rows")
    report(out, config.ROOT / "docs" / "GRAPH_FINDINGS.md")
    spark.stop()


if __name__ == "__main__":
    main()
