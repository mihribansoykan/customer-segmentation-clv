"""Data-driven segmentation with K-Means on log-scaled behavioural features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = ["recency_days", "frequency", "monetary", "avg_basket", "n_products"]


def prepare(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """log1p removes the heavy right skew (a few wholesalers spend 1000x the median)."""
    x = np.log1p(rfm[CLUSTER_FEATURES].clip(lower=0).to_numpy(dtype=float))
    scaler = StandardScaler().fit(x)
    return scaler.transform(x), scaler


def evaluate_k(x: np.ndarray, ks: range = range(2, 9), seed: int = 42) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(x), size=min(len(x), 4000), replace=False)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(x[sample], km.labels_[sample]),
                "davies_bouldin": davies_bouldin_score(x, km.labels_),
            }
        )
    return pd.DataFrame(rows)


def fit(x: np.ndarray, k: int, seed: int = 42) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit(x)


def profile(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = rfm.assign(cluster=labels)
    total = df["monetary"].clip(lower=0).sum()
    prof = df.groupby("cluster").agg(
        customers=("monetary", "size"),
        median_recency=("recency_days", "median"),
        median_frequency=("frequency", "median"),
        median_monetary=("monetary", "median"),
        median_basket=("avg_basket", "median"),
        median_products=("n_products", "median"),
        revenue=("monetary", lambda s: s.clip(lower=0).sum()),
    )
    prof["customer_share"] = prof["customers"] / prof["customers"].sum()
    prof["revenue_share"] = prof["revenue"] / total
    prof["name"] = name_clusters(prof, overall_monetary=df["monetary"].median())
    return prof.sort_values("revenue", ascending=False)


def name_clusters(prof: pd.DataFrame, overall_monetary: float) -> pd.Series:
    """Readable, rule-based names: value tier x activity tier."""

    def value_tier(m: float) -> str:
        if m >= 3 * overall_monetary:
            return "High-value"
        if m >= 0.75 * overall_monetary:
            return "Mid-value"
        return "Low-value"

    def activity_tier(r: float, f: float) -> str:
        if r <= 60:
            return "active" if f > 1 else "new / one-off"
        if r <= 180:
            return "cooling"
        return "lapsed"

    names = [
        f"{value_tier(m)} {activity_tier(r, f)}"
        for m, r, f in zip(
            prof["median_monetary"], prof["median_recency"], prof["median_frequency"], strict=True
        )
    ]
    # Disambiguate duplicates so every cluster keeps a unique label.
    seen: dict[str, int] = {}
    unique = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        unique.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return pd.Series(unique, index=prof.index)


def centroid_zscores(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean standardised (log) feature value per cluster - the classic 'snake plot' data."""
    x, _ = prepare(rfm)
    z = pd.DataFrame(x, columns=CLUSTER_FEATURES, index=rfm.index).assign(cluster=labels)
    return z.groupby("cluster").mean()
