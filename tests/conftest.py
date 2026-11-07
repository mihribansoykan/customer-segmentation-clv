"""Synthetic transactions in the cleaned schema - tests never need the real 45 MB download."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_transactions(n_customers: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-09")
    span = (end - start).days
    rows = []
    inv_no = 500000
    for cid in range(12000, 12000 + n_customers):
        first = rng.integers(0, span - 30)
        rate = rng.gamma(2.0, 1 / 60)  # purchases per day while active
        life = rng.exponential(400)
        t = float(first)
        while t < min(span, first + life):
            inv_no += 1
            for _ in range(rng.integers(1, 5)):
                qty = int(rng.integers(1, 24))
                price = float(rng.choice([0.85, 1.25, 2.1, 4.95]))
                rows.append(
                    (
                        str(inv_no),
                        f"{rng.integers(10000, 99999)}",
                        qty,
                        start + pd.Timedelta(days=t),
                        price,
                        cid,
                        "United Kingdom" if cid % 5 else "France",
                    )
                )
            t += rng.exponential(1 / rate)
    df = pd.DataFrame(
        rows,
        columns=[
            "invoice",
            "stock_code",
            "quantity",
            "invoice_date",
            "price",
            "customer_id",
            "country",
        ],
    )
    df["description"] = "ITEM"
    df["is_return"] = False
    # A few returns
    ret = df.sample(frac=0.02, random_state=seed).copy()
    ret["invoice"] = "C" + ret["invoice"]
    ret["quantity"] = -ret["quantity"]
    ret["invoice_date"] = ret["invoice_date"] + pd.Timedelta(days=3)
    ret["is_return"] = True
    df = pd.concat([df, ret], ignore_index=True)
    df["revenue"] = df["quantity"] * df["price"]
    return df.sort_values("invoice_date", ignore_index=True)


@pytest.fixture(scope="session")
def tx() -> pd.DataFrame:
    return make_transactions()
