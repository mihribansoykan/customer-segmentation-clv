"""Monthly acquisition cohorts and retention."""

from __future__ import annotations

import pandas as pd


def retention_matrix(invoices: pd.DataFrame, max_age: int = 12) -> pd.DataFrame:
    """Share of each monthly acquisition cohort that purchases again k months later.

    Rows: cohort month (first purchase). Columns: months since acquisition (0 = 100%).
    """
    inv = invoices[["customer_id", "invoice_date"]].copy()
    inv["month"] = inv["invoice_date"].dt.to_period("M")
    inv["cohort"] = inv.groupby("customer_id")["month"].transform("min")
    inv["age"] = (inv["month"] - inv["cohort"]).apply(lambda d: d.n)

    counts = (
        inv.drop_duplicates(["customer_id", "age"])
        .pivot_table(index="cohort", columns="age", values="customer_id", aggfunc="count")
        .reindex(columns=range(max_age + 1))
    )
    retention = counts.div(counts[0], axis=0)
    retention.index = retention.index.astype(str)
    retention.insert(0, "cohort_size", counts[0])
    return retention
