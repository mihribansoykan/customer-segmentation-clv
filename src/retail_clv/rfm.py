"""RFM (recency, frequency, monetary) features, scores and rule-based segments."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from retail_clv.data import purchases

# Classic R x F segment grid (R and F scored 1-5). Patterns are matched against "<R><F>".
SEGMENT_RULES: list[tuple[str, str]] = [
    (r"[1-2][1-2]", "Hibernating"),
    (r"[1-2][3-4]", "At Risk"),
    (r"[1-2]5", "Can't Lose"),
    (r"3[1-2]", "About to Sleep"),
    (r"33", "Need Attention"),
    (r"[3-4][4-5]", "Loyal"),
    (r"41", "Promising"),
    (r"51", "New Customers"),
    (r"[4-5][2-3]", "Potential Loyalists"),
    (r"5[4-5]", "Champions"),
]

SEGMENT_ACTIONS: dict[str, str] = {
    "Champions": "Reward, early access to new ranges, ask for referrals",
    "Loyal": "Upsell higher-value ranges, loyalty programme",
    "Potential Loyalists": "Membership offer, personalised recommendations",
    "New Customers": "Onboarding journey, second-purchase incentive",
    "Promising": "Brand awareness, free-shipping threshold nudges",
    "Need Attention": "Time-limited offers based on purchase history",
    "About to Sleep": "Reactivation campaign with popular products",
    "At Risk": "Personal win-back outreach before they churn",
    "Can't Lose": "High-touch win-back (account manager call), survey why they left",
    "Hibernating": "Low-cost automated reactivation; suppress from expensive channels",
}


def rfm_table(tx: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Customer-level behavioural summary as of ``snapshot`` (exclusive).

    * recency_days - days since last purchase
    * frequency    - number of purchase invoices
    * monetary     - net revenue (purchases minus returns)
    * avg_basket   - mean purchase invoice value
    * return_rate  - returned value / purchased value
    * tenure_days  - days since first purchase
    """
    tx = tx[tx["invoice_date"] < snapshot]
    p = purchases(tx)
    g = p.groupby("customer_id")
    gross = g["revenue"].sum()
    returns = -tx[tx["is_return"]].groupby("customer_id")["revenue"].sum()

    out = pd.DataFrame(
        {
            "recency_days": (snapshot - g["invoice_date"].max()).dt.days,
            "frequency": g["invoice"].nunique(),
            "monetary": gross - returns.reindex(gross.index).fillna(0.0),
            "gross_revenue": gross,
            "tenure_days": (snapshot - g["invoice_date"].min()).dt.days,
            "n_products": g["stock_code"].nunique(),
            "country": g["country"].agg(lambda s: s.mode().iloc[0]),
        }
    )
    out["avg_basket"] = out["gross_revenue"] / out["frequency"]
    out["return_rate"] = (returns.reindex(out.index).fillna(0.0) / out["gross_revenue"]).clip(0, 1)
    return out


def _quintile(s: pd.Series, ascending: bool = True) -> pd.Series:
    """1-5 score by quintile; ranking first breaks ties so every bin is populated."""
    ranks = s.rank(method="first", ascending=ascending)
    return pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    out = rfm.copy()
    out["R"] = _quintile(out["recency_days"], ascending=False)  # recent -> 5
    out["F"] = _quintile(out["frequency"])
    out["M"] = _quintile(out["monetary"])
    out["RFM"] = out["R"].astype(str) + out["F"].astype(str) + out["M"].astype(str)
    out["segment"] = [segment_for(r, f) for r, f in zip(out["R"], out["F"], strict=True)]
    return out


def segment_for(r: int, f: int) -> str:
    key = f"{r}{f}"
    for pattern, name in SEGMENT_RULES:
        if re.fullmatch(pattern, key):
            return name
    raise ValueError(f"No segment rule for R={r}, F={f}")  # pragma: no cover


def segment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    total_rev = scored["monetary"].clip(lower=0).sum()
    s = scored.groupby("segment").agg(
        customers=("monetary", "size"),
        revenue=("monetary", lambda x: x.clip(lower=0).sum()),
        median_recency=("recency_days", "median"),
        median_frequency=("frequency", "median"),
        median_monetary=("monetary", "median"),
    )
    s["customer_share"] = s["customers"] / s["customers"].sum()
    s["revenue_share"] = s["revenue"] / total_rev
    s["action"] = s.index.map(SEGMENT_ACTIONS)
    return s.sort_values("revenue", ascending=False)


def log_features(rfm: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return np.log1p(rfm[cols].clip(lower=0))
