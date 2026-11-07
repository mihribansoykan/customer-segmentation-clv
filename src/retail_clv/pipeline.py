"""End-to-end pipeline.

    uv run retail-clv            # download (first run only), analyse, write reports/

Outputs
-------
reports/metrics.json            all headline numbers quoted in the README
reports/segment_summary.csv     RFM segments with size, revenue, P(alive), revenue at risk, action
reports/cluster_profiles.csv    K-Means cluster profiles
reports/clv_test_metrics.csv    model comparison on the 2011 hold-out period
reports/figures/*.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from retail_clv import btyd, clustering, clv, cohorts, data, plots, rfm

log = logging.getLogger("retail_clv")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
SNAPSHOT = pd.Timestamp("2011-12-10")  # day after the last transaction
N_CLUSTERS = 4
SEED = 42


def run_btyd_holdout(tx: pd.DataFrame, population: pd.Index) -> tuple[pd.Series, dict]:
    """Fit BG/NBD + Gamma-Gamma on history before the test cutoff, predict the hold-out window."""
    inv = data.invoices(tx[tx["invoice_date"] < clv.TEST_CUTOFF])
    summary = btyd.summary_data(inv, clv.TEST_CUTOFF - pd.Timedelta(days=1))
    bg = btyd.BetaGeo().fit(summary["frequency"], summary["recency"], summary["T"], penalizer=1e-3)
    gg = btyd.GammaGamma().fit(summary["frequency"], summary["monetary_value"], penalizer=1e-3)
    pred = btyd.predict_revenue(bg, gg, summary, clv.HORIZON_DAYS).reindex(population)
    if pred.isna().any():
        raise ValueError(f"{pred.isna().sum()} customers missing from the BTYD summary")
    return pred, {"bgnbd": bg.params, "gamma_gamma": gg.params}


def run(force_download: bool = False) -> dict:
    FIGS.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}

    # 1 ── data ------------------------------------------------------------------------
    raw = data.load_raw(data.download(force=force_download))
    tx, report = data.clean(raw)
    inv = data.invoices(tx)
    metrics["cleaning"] = report.to_frame().to_dict(orient="records")
    metrics["data"] = {
        "transactions": len(tx),
        "invoices": int(inv["invoice"].nunique()),
        "customers": int(tx["customer_id"].nunique()),
        "countries": int(tx["country"].nunique()),
        "period": [str(tx["invoice_date"].min().date()), str(tx["invoice_date"].max().date())],
        "net_revenue": round(float(tx["revenue"].sum()), 2),
        "return_value_share": round(
            float(-tx.loc[tx["is_return"], "revenue"].sum() / data.purchases(tx)["revenue"].sum()),
            4,
        ),
    }
    plots.monthly_revenue(inv, FIGS / "monthly_revenue.png")
    retention = cohorts.retention_matrix(inv)
    plots.cohort_heatmap(retention, FIGS / "cohort_retention.png")
    metrics["cohorts"] = {
        "mean_month1_retention": round(float(retention[1].mean()), 4),
        "mean_month3_retention": round(float(retention[3].mean()), 4),
    }

    # 2 ── RFM segmentation ------------------------------------------------------------
    table = rfm.rfm_table(tx, SNAPSHOT)
    table = table[table["monetary"] > 0]  # drop customers whose returns exceed purchases
    scored = rfm.rfm_scores(table)

    # P(alive) and expected next-6-month revenue from BTYD models fitted on the full history.
    summary = btyd.summary_data(inv, SNAPSHOT - pd.Timedelta(days=1))
    bg = btyd.BetaGeo().fit(summary["frequency"], summary["recency"], summary["T"], penalizer=1e-3)
    gg = btyd.GammaGamma().fit(summary["frequency"], summary["monetary_value"], penalizer=1e-3)
    scored["p_alive"] = pd.Series(
        bg.p_alive(summary["frequency"], summary["recency"], summary["T"]), index=summary.index
    ).reindex(scored.index)
    scored["pred_6m_revenue"] = btyd.predict_revenue(bg, gg, summary, clv.HORIZON_DAYS).reindex(
        scored.index
    )
    last_year = clv.target(tx, SNAPSHOT - pd.Timedelta(days=365), horizon_days=365)
    scored["revenue_last_12m"] = last_year.reindex(scored.index).fillna(0.0)
    scored["revenue_at_risk"] = scored["revenue_last_12m"] * (1 - scored["p_alive"])

    seg = rfm.segment_summary(scored)
    extra = scored.groupby("segment").agg(
        mean_p_alive=("p_alive", "mean"),
        revenue_last_12m=("revenue_last_12m", "sum"),
        revenue_at_risk=("revenue_at_risk", "sum"),
        pred_6m_revenue=("pred_6m_revenue", "sum"),
    )
    seg = seg.join(extra)
    seg.round(4).to_csv(REPORTS / "segment_summary.csv")
    plots.segment_bars(seg, FIGS / "rfm_segments.png")
    top = seg.reindex(["Champions", "Loyal"]).fillna(0)
    risk = seg.reindex(["At Risk", "Can't Lose"]).fillna(0)
    metrics["rfm"] = {
        "customers_segmented": len(scored),
        "champions_loyal_customer_share": round(float(top["customer_share"].sum()), 4),
        "champions_loyal_revenue_share": round(float(top["revenue_share"].sum()), 4),
        "at_risk_cant_lose_customers": int(risk["customers"].sum()),
        "at_risk_cant_lose_revenue_last_12m": round(float(risk["revenue_last_12m"].sum()), 2),
        "at_risk_cant_lose_revenue_at_risk": round(float(risk["revenue_at_risk"].sum()), 2),
        "btyd_params_full_history": {**bg.params, **gg.params},
    }

    # 3 ── K-Means ---------------------------------------------------------------------
    x, _ = clustering.prepare(scored)
    k_scores = clustering.evaluate_k(x, seed=SEED)
    km = clustering.fit(x, N_CLUSTERS, seed=SEED)
    prof = clustering.profile(scored, km.labels_)
    prof.round(4).to_csv(REPORTS / "cluster_profiles.csv")
    plots.k_selection(k_scores, N_CLUSTERS, FIGS / "kmeans_k_selection.png")
    plots.cluster_heatmap(
        clustering.centroid_zscores(scored, km.labels_), prof["name"], FIGS / "cluster_profiles.png"
    )
    metrics["kmeans"] = {
        "k": N_CLUSTERS,
        "k_scores": k_scores.round(4).to_dict(orient="records"),
        "clusters": prof[["name", "customers", "customer_share", "revenue_share"]]
        .round(4)
        .to_dict(orient="records"),
        "crosstab_rfm_vs_cluster": pd.crosstab(
            scored["segment"], prof["name"].reindex(km.labels_).to_numpy()
        ).to_dict(),
    }

    # 4 ── CLV: model comparison on the 2011 hold-out ------------------------------------
    x_tr, y_tr = clv.make_training_set(tx, clv.TRAIN_CUTOFFS, with_btyd=True)
    x_te, y_te = clv.make_dataset(tx, clv.TEST_CUTOFF, with_btyd=True)
    behavioural = [c for c in x_te.columns if c not in clv.BTYD_FEATURES]
    model_beh = clv.train_model(x_tr[behavioural], y_tr, seed=SEED)
    model_hyb = clv.train_model(x_tr, y_tr, seed=SEED)
    preds = {
        "Naive (last 6 months)": clv.naive_baseline(x_te),
        "BG/NBD + Gamma-Gamma": x_te["btyd_pred"].rename("btyd_pred"),
        "LightGBM (behavioural)": pd.Series(model_beh.predict(x_te[behavioural]), index=x_te.index),
        "LightGBM + BTYD features": pd.Series(model_hyb.predict(x_te), index=x_te.index),
    }
    _, btyd_params = run_btyd_holdout(tx, x_te.index)
    results = pd.DataFrame({name: clv.evaluate(y_te, p) for name, p in preds.items()}).T
    results.round(4).to_csv(REPORTS / "clv_test_metrics.csv")

    curves = {name: clv.capture_curve(y_te, p) for name, p in preds.items()}
    curves["perfect foresight"] = clv.capture_curve(y_te, y_te)
    plots.capture_curves(curves, FIGS / "clv_capture_curves.png")
    plots.decile_plot(
        y_te,
        {k: preds[k] for k in ("BG/NBD + Gamma-Gamma", "LightGBM + BTYD features")},
        FIGS / "clv_deciles.png",
    )
    contrib = model_hyb.booster_.predict(x_te, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=x_te.columns)
    plots.importance(
        imp.sort_values().tail(12), FIGS / "clv_feature_importance.png", "Hybrid CLV model drivers"
    )

    metrics["clv"] = {
        "horizon_days": clv.HORIZON_DAYS,
        "train": {
            "cutoffs": [str(c.date()) for c in clv.TRAIN_CUTOFFS],
            "rows": len(x_tr),
            "share_with_future_purchase": round(float((y_tr > 0).mean()), 4),
        },
        "test": {
            "cutoff": str(clv.TEST_CUTOFF.date()),
            "customers": len(x_te),
            "share_with_future_purchase": round(float((y_te > 0).mean()), 4),
            "actual_revenue": round(float(y_te.sum()), 2),
        },
        "results": results.round(4).to_dict(orient="index"),
        "btyd_params_holdout": btyd_params,
        "top_features": imp.sort_values(ascending=False).head(8).round(2).to_dict(),
    }

    (REPORTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    log.info("CLV hold-out results:\n%s", results.round(3).to_string())
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the segmentation + CLV pipeline")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(force_download=args.force_download)


if __name__ == "__main__":
    main()
