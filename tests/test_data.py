import pandas as pd

from retail_clv.data import clean, invoices


def _raw() -> pd.DataFrame:
    rows = [
        # invoice, stock, desc, qty, date, price, customer, country
        ("489434", "85048", "LAMP", 12, "2010-01-01 10:00", 6.95, 13085.0, "United Kingdom"),
        ("489434", "85048", "LAMP", 12, "2010-01-01 10:00", 6.95, 13085.0, "United Kingdom"),  # dup
        ("489434", "79323P", "LIGHT", 6, "2010-01-01 10:00", 2.10, 13085.0, "United Kingdom"),
        ("489435", "POST", "POSTAGE", 1, "2010-01-02 11:00", 18.0, 13085.0, "United Kingdom"),
        (
            "489436",
            "22041",
            "CARD",
            10,
            "2010-01-03 12:00",
            0.0,
            13085.0,
            "United Kingdom",
        ),  # price 0
        ("489437", "22041", "CARD", 10, "2010-01-03 12:00", 0.5, None, "United Kingdom"),  # no id
        ("C489438", "85048", "LAMP", -2, "2010-01-05 09:00", 6.95, 13085.0, "United Kingdom"),
        ("489439", "21232", "MUG", -5, "2010-01-06 09:00", 1.25, 13085.0, "United Kingdom"),  # bad
        ("489440", "21232", "MUG", 4, "2010-01-07 09:00", 1.25, 14000.0, "France"),
    ]
    cols = [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "price",
        "customer_id",
        "country",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def test_clean_applies_all_rules():
    tx, report = clean(_raw())
    assert len(tx) == 4  # 2 product lines + 1 return + 1 other customer
    assert tx["customer_id"].dtype.kind == "i"
    assert "POST" not in set(tx["stock_code"])
    assert tx["is_return"].sum() == 1
    assert tx.loc[tx["is_return"], "revenue"].iloc[0] < 0
    steps = report.to_frame()
    assert steps["rows"].iloc[0] == 9
    assert steps["removed"].sum() == 9 - 4


def test_invoices_aggregate_baskets():
    tx, _ = clean(_raw())
    inv = invoices(tx)
    assert set(inv["invoice"]) == {"489434", "489440"}  # returns excluded
    assert inv.set_index("invoice").loc["489434", "value"] == 12 * 6.95 + 6 * 2.10
