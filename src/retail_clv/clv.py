"""Supervised 6-month customer value prediction and evaluation utilities.

Time design (no leakage):

    For every cutoff date c, features use transactions in [c - 180 days, c) and the target is the
    net revenue in [c, c + 182 days). The population is every customer with at least one purchase
    in the look-back window.

    train: 7 monthly cutoffs 2010-06-09 ... 2010-12-09 (stacked) -> all targets end <= 2011-06-09
    test : cutoff 2011-06-09                                      -> target 2011-06-09 .. 2011-12-08

Stacking several cutoffs multiplies the training rows (rolling-origin training). Because target
windows from different cutoffs contain different amounts of the Q4 gift season, the share of the
target window falling in October-December is passed as a (known-in-advance) calendar feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

from retail_clv import btyd
from retail_clv.data import invoices, purchases

HORIZON_DAYS = 182
LOOKBACK_DAYS = 180
TRAIN_CUTOFFS = list(pd.date_range("2010-06-09", "2010-12-09", freq=pd.DateOffset(months=1)))
TRAIN_CUTOFF = TRAIN_CUTOFFS[0]
TEST_CUTOFF = pd.Timestamp("2011-06-09")
BTYD_FEATURES = ["p_alive", "btyd_purchases", "btyd_value", "btyd_pred"]


def customer_features(
    tx: pd.DataFrame, cutoff: pd.Timestamp, lookback_days: int = LOOKBACK_DAYS
) -> pd.DataFrame:
    """Behavioural features from the look-back window ``[cutoff - lookback, cutoff)``."""
    start = cutoff - pd.Timedelta(days=lookback_days)
    window = tx[(tx["invoice_date"] >= start) & (tx["invoice_date"] < cutoff)]
    p = purchases(window)
    inv = (
        p.groupby(["customer_id", "invoice"])
        .agg(date=("invoice_date", "min"), value=("revenue", "sum"))
        .reset_index()
    )
    g = inv.groupby("customer_id")
    days_ago = (cutoff - inv["date"]).dt.days

    feats = pd.DataFrame(
        {
            "recency_days": (cutoff - g["date"].max()).dt.days,
            "first_seen_days": (cutoff - g["date"].min()).dt.days,
            "n_invoices": g["invoice"].size(),
            "n_invoices_90d": inv[days_ago < 90].groupby("customer_id").size(),
            "n_invoices_30d": inv[days_ago < 30].groupby("customer_id").size(),
            "revenue": g["value"].sum(),
            "revenue_90d": inv[days_ago < 90].groupby("customer_id")["value"].sum(),
            "revenue_30d": inv[days_ago < 30].groupby("customer_id")["value"].sum(),
            "avg_basket": g["value"].mean(),
            "max_basket": g["value"].max(),
            "purchase_days": g["date"].apply(lambda d: d.dt.normalize().nunique()),
            "n_products": p.groupby("customer_id")["stock_code"].nunique(),
            "n_lines": p.groupby("customer_id").size(),
        }
    )
    feats = feats.fillna({c: 0 for c in feats.columns if c.endswith(("_90d", "_30d"))})

    returned = -window[window["is_return"]].groupby("customer_id")["revenue"].sum()
    feats["return_rate"] = (returned.reindex(feats.index).fillna(0) / feats["revenue"]).clip(0, 1)
    gaps = (
        inv.sort_values("date")
        .groupby("customer_id")["date"]
        .apply(lambda d: d.diff().dt.days.mean())
    )
    feats["mean_gap_days"] = gaps.reindex(feats.index)  # NaN for single-invoice customers
    country = p.groupby("customer_id")["country"].agg(lambda s: s.mode().iloc[0])
    feats["is_uk"] = (country.reindex(feats.index) == "United Kingdom").astype(int)
    return feats.sort_index()


def target(tx: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int = HORIZON_DAYS) -> pd.Series:
    """Net revenue (purchases minus returns, floored at 0) in ``[cutoff, cutoff + horizon)``."""
    end = cutoff + pd.Timedelta(days=horizon_days)
    window = tx[(tx["invoice_date"] >= cutoff) & (tx["invoice_date"] < end)]
    return window.groupby("customer_id")["revenue"].sum().clip(lower=0).rename("future_revenue")


def q4_share(cutoff: pd.Timestamp, horizon_days: int = HORIZON_DAYS) -> float:
    """Fraction of the target window that falls in October-December (calendar, known upfront)."""
    days = pd.date_range(cutoff, periods=horizon_days, freq="D")
    return float((days.month >= 10).mean())


def btyd_features(tx: pd.DataFrame, cutoff: pd.Timestamp, index: pd.Index) -> pd.DataFrame:
    """BG/NBD + Gamma-Gamma outputs fitted only on history before ``cutoff`` (no leakage)."""
    inv = invoices(tx[tx["invoice_date"] < cutoff])
    s = btyd.summary_data(inv, cutoff - pd.Timedelta(days=1))
    bg = btyd.BetaGeo().fit(s["frequency"], s["recency"], s["T"], penalizer=1e-3)
    gg = btyd.GammaGamma().fit(s["frequency"], s["monetary_value"], penalizer=1e-3)
    out = pd.DataFrame(index=s.index)
    out["p_alive"] = bg.p_alive(s["frequency"], s["recency"], s["T"])
    out["btyd_purchases"] = bg.expected_purchases(
        HORIZON_DAYS, s["frequency"], s["recency"], s["T"]
    )
    out["btyd_value"] = gg.expected_value(s["frequency"], s["monetary_value"])
    out["btyd_pred"] = out["btyd_purchases"] * out["btyd_value"]
    return out.reindex(index)


def make_dataset(
    tx: pd.DataFrame, cutoff: pd.Timestamp, with_btyd: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    x = customer_features(tx, cutoff)
    x["q4_share"] = q4_share(cutoff)
    if with_btyd:
        x = x.join(btyd_features(tx, cutoff, x.index))
    y = target(tx, cutoff).reindex(x.index).fillna(0.0)
    return x, y


def make_training_set(
    tx: pd.DataFrame, cutoffs: list[pd.Timestamp], with_btyd: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Stack datasets from several cutoffs; every target window ends before the test cutoff."""
    parts = [make_dataset(tx, c, with_btyd=with_btyd) for c in cutoffs]
    x = pd.concat([p[0] for p in parts], ignore_index=True)
    y = pd.concat([p[1] for p in parts], ignore_index=True)
    return x, y


def train_model(x: pd.DataFrame, y: pd.Series, seed: int = 42) -> LGBMRegressor:
    """Tweedie objective: the target is zero for many customers and heavily right-skewed."""
    model = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.3,
        n_estimators=600,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        verbosity=-1,
    )
    return model.fit(x, y)


def naive_baseline(x: pd.DataFrame) -> pd.Series:
    """'The next six months look like the last six months.'"""
    return (x["revenue"] * HORIZON_DAYS / LOOKBACK_DAYS).clip(lower=0).rename("naive_pred")


# ------------------------------------------------------------------------- evaluation
def capture_curve(y_true: np.ndarray, y_pred: np.ndarray, points: int = 101) -> pd.DataFrame:
    """Share of actual future revenue captured when targeting the top-x% customers by prediction."""
    y_true = np.asarray(y_true, dtype=float)
    order = np.argsort(-np.asarray(y_pred, dtype=float), kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(y_true[order])]) / y_true.sum()
    share = np.linspace(0, 1, points)
    idx = np.round(share * len(y_true)).astype(int)
    return pd.DataFrame({"customer_share": share, "revenue_share": cum[idx]})


def top_share_capture(y_true: np.ndarray, y_pred: np.ndarray, share: float = 0.2) -> float:
    curve = capture_curve(y_true, y_pred)
    return float(np.interp(share, curve["customer_share"], curve["revenue_share"]))


def evaluate(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    oracle = top_share_capture(y_true, y_true)
    top20 = top_share_capture(y_true, y_pred)
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "top20_capture": top20,
        "top20_capture_vs_oracle": top20 / oracle,
        "total_bias": float(y_pred.sum() / y_true.sum() - 1),
    }
