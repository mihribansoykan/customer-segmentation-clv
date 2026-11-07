"""Report figures with one consistent, colour-blind-checked style."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker as mticker

if "ipykernel" not in sys.modules:  # headless backend for CLI/CI; keep inline plots in notebooks
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#b83232", "#e34948", "#f0efec", "#3987e5", "#184f95"]  # red <- gray -> blue
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        }
    )


def save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def monthly_revenue(invoices: pd.DataFrame, path: str | Path) -> Path:
    apply_style()
    m = (
        invoices.set_index("invoice_date")["value"].resample("MS").sum().iloc[:-1]
    )  # drop partial Dec-11
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(m.index, m.to_numpy() / 1e3, color=SERIES[0], marker="o", ms=4)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("£{x:,.0f}k"))
    ax.set(title="Monthly revenue (registered customers)", ylabel="Revenue")
    return save(fig, path)


def cohort_heatmap(retention: pd.DataFrame, path: str | Path) -> Path:
    apply_style()
    data = retention.drop(columns="cohort_size").iloc[:, 1:]  # month 0 is always 100%
    data = data.dropna(how="all")  # cohorts too recent to have any follow-up month
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(data.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(data.shape[1]), data.columns)
    ax.set_yticks(range(data.shape[0]), data.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(data.to_numpy(dtype=float)):
        if not np.isnan(v):
            ax.text(
                j,
                i,
                f"{v:.0%}",
                ha="center",
                va="center",
                fontsize=7,
                color="#ffffff" if v > 0.3 else INK,
            )
    ax.set(
        title="Monthly cohort retention: share of new customers buying again k months later",
        xlabel="Months since first purchase",
        ylabel="Acquisition cohort",
    )
    fig.colorbar(im, ax=ax, fraction=0.025, format=mticker.PercentFormatter(1))
    return save(fig, path)


def segment_bars(summary: pd.DataFrame, path: str | Path) -> Path:
    apply_style()
    s = summary.sort_values("revenue_share")
    y = np.arange(len(s))
    h = 0.38
    bar = {"height": h, "edgecolor": SURFACE, "linewidth": 2}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y + h / 2, s["customer_share"], color=SERIES[0], label="share of customers", **bar)
    ax.barh(y - h / 2, s["revenue_share"], color=SERIES[1], label="share of revenue", **bar)
    ax.set_yticks(y, s.index)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set(title="RFM segments: customers vs. revenue")
    return save(fig, path)


def k_selection(scores: pd.DataFrame, chosen_k: int, path: str | Path) -> Path:
    apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(scores["k"], scores["inertia"], marker="o", ms=5, color=SERIES[0])
    a1.set(title="Elbow: within-cluster SSE", xlabel="k", ylabel="Inertia")
    a2.plot(scores["k"], scores["silhouette"], marker="o", ms=5, color=SERIES[0])
    a2.set(title="Silhouette score (higher = better)", xlabel="k", ylabel="Silhouette")
    for ax in (a1, a2):
        ax.axvline(chosen_k, color=MUTED, ls="--", lw=1)
    a2.annotate(
        f"chosen k = {chosen_k}",
        (chosen_k, a2.get_ylim()[1]),
        xytext=(6, -14),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )
    return save(fig, path)


def cluster_heatmap(z: pd.DataFrame, names: pd.Series, path: str | Path) -> Path:
    apply_style()
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    data = z.loc[names.index]
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(data) + 1.6))
    lim = float(np.nanmax(np.abs(data.to_numpy())))
    im = ax.imshow(data.to_numpy(), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(data.shape[1]), [c.replace("_", " ") for c in data.columns])
    ax.set_yticks(range(data.shape[0]), names.to_numpy())
    ax.grid(False)
    for (i, j), v in np.ndenumerate(data.to_numpy()):
        ax.text(
            j,
            i,
            f"{v:+.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="#ffffff" if abs(v) > 0.6 * lim else INK,
        )
    ax.set_title("K-Means cluster profiles (mean z-score of log features)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    return save(fig, path)


def capture_curves(curves: dict[str, pd.DataFrame], path: str | Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, c) in enumerate(curves.items()):
        style = (
            {"color": MUTED, "ls": "--", "lw": 1.2}
            if name == "perfect foresight"
            else {"color": SERIES[i]}
        )
        ax.plot(c["customer_share"], c["revenue_share"], label=name, **style)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=":", label="random targeting")
    ax.axvline(0.2, color=GRID, lw=1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))
    ax.legend(loc="lower right")
    ax.set(
        title="Revenue captured by targeting top-ranked customers",
        xlabel="Customers targeted (ranked by predicted 6-month value)",
        ylabel="Share of actual future revenue",
    )
    return save(fig, path)


def decile_plot(y_true: pd.Series, preds: dict[str, pd.Series], path: str | Path) -> Path:
    """Mean actual vs. mean predicted revenue by decile of the model's prediction."""
    apply_style()
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4), sharey=True)
    for ax, (name, p) in zip(np.atleast_1d(axes), preds.items(), strict=True):
        dec = pd.qcut(p.rank(method="first"), 10, labels=range(1, 11))
        g = (
            pd.DataFrame({"actual": y_true, "pred": p, "dec": dec})
            .groupby("dec", observed=True)
            .mean()
        )
        x = np.arange(1, 11)
        bar = {"width": 0.38, "edgecolor": SURFACE, "linewidth": 2}
        ax.bar(x - 0.19, g["pred"], color=SERIES[0], label="predicted", **bar)
        ax.bar(x + 0.19, g["actual"], color=SERIES[1], label="actual", **bar)
        ax.set_xticks(x)
        ax.grid(axis="x", visible=False)
        ax.set(title=name, xlabel="Decile of prediction")
    np.atleast_1d(axes)[0].set_ylabel("Mean 6-month revenue (£)")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    return save(fig, path)


def importance(mean_abs: pd.Series, path: str | Path, title: str) -> Path:
    apply_style()
    s = mean_abs.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(s) + 1.2))
    ax.barh(s.index, s.values, color=SERIES[0], height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set(title=title, xlabel="Mean |SHAP value|")
    return save(fig, path)
