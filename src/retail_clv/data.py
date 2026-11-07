"""Download, load and clean the Online Retail II transactions.

Source: Chen, D. (2019). Online Retail II. UCI Machine Learning Repository, id 502. CC BY 4.0.
All transactions of a UK-based, registered non-store online gift retailer between
2009-12-01 and 2011-12-09. Many customers are wholesalers.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DIR = Path("data/raw")
PARQUET = RAW_DIR / "online_retail_ii.parquet"

COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def download(url: str = URL, parquet_path: Path = PARQUET, force: bool = False) -> Path:
    """Download the zipped Excel workbook once and cache both sheets as a single Parquet file."""
    if parquet_path.exists() and not force:
        return parquet_path
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=300) as resp:  # noqa: S310
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        xlsx_name = next(n for n in zf.namelist() if n.endswith(".xlsx"))
        sheets = pd.read_excel(
            io.BytesIO(zf.read(xlsx_name)),
            sheet_name=None,
            engine="calamine",
            dtype={"Invoice": str, "StockCode": str},
        )
    df = pd.concat(sheets.values(), ignore_index=True)
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype("string")
    df.to_parquet(parquet_path, index=False)
    log.info("Cached %d rows to %s", len(df), parquet_path)
    return parquet_path


def load_raw(path: Path = PARQUET) -> pd.DataFrame:
    df = pd.read_parquet(path).rename(columns=COLUMNS)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


@dataclass
class CleaningReport:
    steps: list[tuple[str, int]] = field(default_factory=list)

    def add(self, step: str, rows: int) -> None:
        self.steps.append((step, rows))

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame(self.steps, columns=["step", "rows"])
        out["removed"] = (-out["rows"].diff()).fillna(0).astype(int)
        return out


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Return clean line items (purchases *and* returns) with a ``revenue`` column.

    Rules
    -----
    1. Drop lines without a customer id - they cannot be attributed to a customer.
    2. Drop exact duplicate lines.
    3. Keep only real products: stock codes starting with 5 digits. This removes postage (POST),
       manual adjustments (M, ADJUST), bank charges, Amazon fees, test items, etc.
    4. Drop lines with a non-positive unit price.
    5. Cancellations (invoice starts with "C") are kept as returns with negative revenue so that
       customer value is measured *net* of returns; purchases must have positive quantity.
    """
    rep = CleaningReport()
    df = raw.copy()
    rep.add("raw", len(df))

    df = df.dropna(subset=["customer_id"])
    rep.add("drop missing customer_id", len(df))

    df = df.drop_duplicates()
    rep.add("drop exact duplicates", len(df))

    df = df[df["stock_code"].astype(str).str.match(r"^\d{5}")]
    rep.add("keep product stock codes", len(df))

    df = df[df["price"] > 0]
    rep.add("drop non-positive prices", len(df))

    df = df.assign(is_return=df["invoice"].astype(str).str.startswith("C"))
    df = df[df["is_return"] | (df["quantity"] > 0)]
    rep.add("drop non-return lines with quantity <= 0", len(df))

    df = df.assign(
        customer_id=df["customer_id"].astype(int),
        revenue=df["quantity"] * df["price"],
    ).reset_index(drop=True)
    return df, rep


def purchases(tx: pd.DataFrame) -> pd.DataFrame:
    return tx[~tx["is_return"]]


def invoices(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per purchase invoice (basket) with its total value."""
    p = purchases(tx)
    return (
        p.groupby(["invoice", "customer_id"], as_index=False)
        .agg(
            invoice_date=("invoice_date", "min"),
            value=("revenue", "sum"),
            country=("country", "first"),
        )
        .sort_values("invoice_date", ignore_index=True)
    )
