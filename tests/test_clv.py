import numpy as np
import pandas as pd
import pytest

from retail_clv import clv

CUTOFF = pd.Timestamp("2011-06-09")


def test_features_ignore_future_transactions(tx):
    """Leakage guard: changing anything after the cutoff must not change the features."""
    base = clv.customer_features(tx, CUTOFF)
    future = tx["invoice_date"] >= CUTOFF
    perturbed = tx.copy()
    perturbed.loc[future, "quantity"] *= 10
    perturbed.loc[future, "revenue"] *= 10
    pd.testing.assert_frame_equal(base, clv.customer_features(perturbed, CUTOFF))


def test_feature_window_and_population(tx):
    x = clv.customer_features(tx, CUTOFF)
    assert (x["recency_days"] <= clv.LOOKBACK_DAYS).all()
    assert (x["n_invoices"] >= x["n_invoices_90d"]).all()
    assert (x["n_invoices_90d"] >= x["n_invoices_30d"]).all()
    assert x["mean_gap_days"].isna().eq(x["n_invoices"] == 1).all()


def test_target_window(tx):
    y = clv.target(tx, CUTOFF)
    end = CUTOFF + pd.Timedelta(days=clv.HORIZON_DAYS)
    w = tx[(tx["invoice_date"] >= CUTOFF) & (tx["invoice_date"] < end)]
    assert y.sum() == pytest.approx(w.groupby("customer_id")["revenue"].sum().clip(lower=0).sum())
    assert (y >= 0).all()


def test_training_targets_end_before_test_cutoff():
    last = max(clv.TRAIN_CUTOFFS) + pd.Timedelta(days=clv.HORIZON_DAYS)
    assert last <= clv.TEST_CUTOFF
    assert len(clv.TRAIN_CUTOFFS) == 7


def test_q4_share():
    assert clv.q4_share(pd.Timestamp("2011-10-01"), 92) == pytest.approx(1.0)
    assert clv.q4_share(pd.Timestamp("2011-01-01"), 90) == 0.0


def test_stacked_training_set_with_btyd(tx):
    cutoffs = [pd.Timestamp("2010-12-09"), pd.Timestamp("2011-01-09")]
    x, y = clv.make_training_set(tx, cutoffs, with_btyd=True)
    sizes = [len(clv.customer_features(tx, c)) for c in cutoffs]
    assert len(x) == len(y) == sum(sizes)
    assert set(clv.BTYD_FEATURES) <= set(x.columns)
    assert x["p_alive"].between(0, 1).all()
    assert (x["btyd_pred"] >= 0).all()


def test_model_trains_and_predicts(tx):
    x, y = clv.make_dataset(tx, pd.Timestamp("2010-12-09"))
    model = clv.train_model(x, y)
    pred = model.predict(x)
    assert (pred >= 0).all() and len(pred) == len(x)


def test_capture_curve_and_metrics():
    y = np.array([100.0, 0, 50, 0, 10])
    perfect = clv.evaluate(y, y)
    assert perfect["mae"] == 0 and perfect["spearman"] == pytest.approx(1.0)
    assert perfect["top20_capture_vs_oracle"] == pytest.approx(1.0)
    curve = clv.capture_curve(y, y)
    assert curve["revenue_share"].iloc[0] == 0 and curve["revenue_share"].iloc[-1] == pytest.approx(
        1
    )
    assert clv.top_share_capture(y, y[::-1]) < clv.top_share_capture(y, y)
    assert clv.evaluate(y, y * 1.5)["total_bias"] == pytest.approx(0.5)
