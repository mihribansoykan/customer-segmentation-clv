import numpy as np
import pandas as pd
import pytest

from retail_clv import clustering, cohorts, rfm
from retail_clv.data import invoices

SNAPSHOT = pd.Timestamp("2011-12-10")


@pytest.fixture(scope="module")
def table(tx):
    t = rfm.rfm_table(tx, SNAPSHOT)
    return t[t["monetary"] > 0]


def test_rfm_table_values(tx, table):
    cid = table.index[0]
    p = tx[(tx["customer_id"] == cid) & ~tx["is_return"]]
    assert table.loc[cid, "frequency"] == p["invoice"].nunique()
    assert table.loc[cid, "recency_days"] == (SNAPSHOT - p["invoice_date"].max()).days
    assert (table["return_rate"].between(0, 1)).all()


def test_scores_and_segments(table):
    scored = rfm.rfm_scores(table)
    for c in "RFM":
        assert set(scored[c].unique()) == {1, 2, 3, 4, 5}
    assert set(scored["segment"]) <= set(rfm.SEGMENT_ACTIONS)
    # most recent customers must get the best recency score
    assert scored.loc[scored["recency_days"].idxmin(), "R"] == 5


@pytest.mark.parametrize(
    ("r", "f", "name"),
    [
        (5, 5, "Champions"),
        (1, 1, "Hibernating"),
        (1, 5, "Can't Lose"),
        (5, 1, "New Customers"),
        (2, 3, "At Risk"),
        (3, 3, "Need Attention"),
    ],
)
def test_segment_grid(r, f, name):
    assert rfm.segment_for(r, f) == name


def test_every_rf_cell_has_a_segment():
    for r in range(1, 6):
        for f in range(1, 6):
            assert rfm.segment_for(r, f) in rfm.SEGMENT_ACTIONS


def test_segment_summary_shares(table):
    s = rfm.segment_summary(rfm.rfm_scores(table))
    assert s["customer_share"].sum() == pytest.approx(1.0)
    assert s["revenue_share"].sum() == pytest.approx(1.0)


def test_kmeans_pipeline(table):
    x, _ = clustering.prepare(table)
    assert np.allclose(x.mean(axis=0), 0, atol=1e-9)
    scores = clustering.evaluate_k(x, ks=range(2, 5))
    assert list(scores["k"]) == [2, 3, 4]
    km = clustering.fit(x, 3)
    prof = clustering.profile(table, km.labels_)
    assert prof["customers"].sum() == len(table)
    assert prof["name"].is_unique


def test_name_clusters_tiers():
    prof = pd.DataFrame(
        {
            "median_monetary": [5000, 500, 100],
            "median_recency": [10, 120, 300],
            "median_frequency": [12, 3, 1],
        }
    )
    names = clustering.name_clusters(prof, overall_monetary=600).tolist()
    assert names == ["High-value active", "Mid-value cooling", "Low-value lapsed"]


def test_retention_matrix(tx):
    ret = cohorts.retention_matrix(invoices(tx))
    assert (ret[0] == 1.0).all()
    assert ret.drop(columns="cohort_size").max().max() <= 1.0
