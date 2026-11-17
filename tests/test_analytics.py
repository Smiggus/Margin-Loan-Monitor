"""GraphFrames motifs on the small hand-made book (needs Java; skipped otherwise)."""
import shutil

import pytest

from tests.test_graph import ISSUERS, LENDERS, PLEDGES

pytestmark = pytest.mark.skipif(shutil.which("java") is None, reason="Spark needs Java")


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    from mlm.graph import analytics
    from mlm.graph.build import build
    spark = analytics.spark_session("tests")
    spark.sparkContext.setCheckpointDir(str(tmp_path_factory.mktemp("ckpt")))
    v, e = build(PLEDGES, LENDERS, ISSUERS)
    from graphframes import GraphFrame
    e = e.astype({c: str for c in e.columns if c not in ("shares_pledged", "pledged_value_usd")})
    g = GraphFrame(spark.createDataFrame(v[["id", "type", "name"]].astype(str)), spark.createDataFrame(e))
    yield g
    spark.stop()


def test_wrong_way_finds_all_insider_pledges(graph):
    from mlm.graph.analytics import wrong_way
    assert wrong_way(graph).count() == 5


def test_cross_pledger_is_the_family_holdco(graph):
    from mlm.graph.analytics import cross_pledgers
    rows = cross_pledgers(graph).collect()
    assert len(rows) == 1
    assert rows[0]["holder"] == "LGC Holdco" and rows[0]["issuers"] == 2
    assert rows[0]["pledged_value_usd"] == pytest.approx(3.0e9)


def test_lender_sector_concentration(graph):
    from mlm.graph.analytics import lender_sector
    rows = {(r["lender"], r["sector"]): r for r in lender_sector(graph).collect()}
    jpm = rows[("JPMorgan", "Transportation, Communications & Utilities")]
    assert jpm["borrowers"] == 1 and jpm["issuers"] == 2


def test_components_join_family_and_lender(graph):
    from mlm.graph.analytics import clusters
    _, sizes = clusters(graph)
    biggest = sizes.first()
    assert biggest["size"] == 4  # LGC Holdco + News Corp + Fox + JPMorgan
