"""'Buy Till You Die' models implemented from the original papers (no external CLV library).

* BG/NBD  - Fader, Hardie & Lee (2005), "Counting Your Customers the Easy Way".
  While alive, a customer buys following a Poisson process with rate lambda ~ Gamma(r, alpha);
  after each purchase they become inactive with probability p ~ Beta(a, b).
* Gamma-Gamma - Fader, Hardie & Lee (2005), "RFM and CLV: Using Iso-Value Curves".
  Spend per transaction varies around a customer-level mean that is itself Gamma distributed
  across customers.

Input per customer (calibration period, time unit = days):
    frequency  x   - number of *repeat* purchase days
    recency    t_x - age of the customer at their last purchase
    T              - age of the customer at the end of the calibration period
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1


def summary_data(invoices: pd.DataFrame, calibration_end: pd.Timestamp) -> pd.DataFrame:
    """Frequency / recency / T / monetary per customer from invoices up to ``calibration_end``.

    Purchases on the same day are merged into one transaction (as in the BG/NBD literature).
    ``monetary_value`` is the mean daily spend over *repeat* transactions (Gamma-Gamma convention).
    """
    end_day = calibration_end.normalize()
    inv = invoices[invoices["invoice_date"] < end_day + pd.Timedelta(days=1)].copy()  # whole day
    inv["day"] = inv["invoice_date"].dt.normalize()
    daily = inv.groupby(["customer_id", "day"], as_index=False)["value"].sum()
    daily = daily.sort_values(["customer_id", "day"])

    first = daily.groupby("customer_id")["day"].transform("min")
    daily["is_repeat"] = daily["day"] > first

    g = daily.groupby("customer_id")
    out = pd.DataFrame(
        {
            "frequency": g["is_repeat"].sum().astype(float),
            "recency": (g["day"].max() - g["day"].min()).dt.days.astype(float),
            "T": (end_day - g["day"].min()).dt.days.astype(float),
        }
    )
    repeat_value = daily[daily["is_repeat"]].groupby("customer_id")["value"].mean()
    out["monetary_value"] = repeat_value.reindex(out.index).fillna(0.0)
    return out


# --------------------------------------------------------------------------- BG/NBD
@dataclass
class BetaGeo:
    r: float = np.nan
    alpha: float = np.nan
    a: float = np.nan
    b: float = np.nan

    @staticmethod
    def _neg_ll(log_params: np.ndarray, x, t_x, T, penalizer: float) -> float:
        r, alpha, a, b = np.exp(log_params)
        a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        a3 = -(r + x) * np.log(alpha + T)
        with np.errstate(divide="ignore", invalid="ignore"):
            a4 = np.where(
                x > 0,
                np.log(a) - np.log(b + x - 1) - (r + x) * np.log(alpha + t_x),
                -np.inf,
            )
        ll = a1 + a2 + np.logaddexp(a3, a4)
        return -ll.mean() + penalizer * np.sum(np.exp(log_params) ** 2)

    def fit(self, frequency, recency, T, penalizer: float = 0.0) -> BetaGeo:
        x, t_x, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        res = minimize(
            self._neg_ll,
            x0=np.log([1.0, max(T.mean(), 1.0), 1.0, 1.0]),
            args=(x, t_x, T, penalizer),
            method="Nelder-Mead",
            options={"maxiter": 20_000, "xatol": 1e-8, "fatol": 1e-10},
        )
        if not res.success:  # pragma: no cover
            raise RuntimeError(f"BG/NBD fit did not converge: {res.message}")
        self.r, self.alpha, self.a, self.b = np.exp(res.x)
        return self

    def p_alive(self, frequency, recency, T) -> np.ndarray:
        """Probability that the customer is still active at the end of the calibration period."""
        x, t_x, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        with np.errstate(divide="ignore", invalid="ignore"):
            odds = (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x)
        return np.where(x > 0, 1.0 / (1.0 + odds), 1.0)

    def expected_purchases(self, t: float, frequency, recency, T) -> np.ndarray:
        """E[number of purchases in (T, T + t] | x, t_x, T]."""
        x, t_x, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        z = t / (alpha + T + t)
        hyp = hyp2f1(r + x, b + x, a + b + x - 1, z)
        num = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp)
        with np.errstate(divide="ignore", invalid="ignore"):
            den = 1 + np.where(
                x > 0, (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0
            )
        return num / den

    @property
    def params(self) -> dict[str, float]:
        return {"r": self.r, "alpha": self.alpha, "a": self.a, "b": self.b}


# ---------------------------------------------------------------------- Gamma-Gamma
@dataclass
class GammaGamma:
    p: float = np.nan
    q: float = np.nan
    v: float = np.nan
    empirical_mean: float = np.nan  # fallback when the population mean is undefined (q <= 1)

    @staticmethod
    def _neg_ll(log_params: np.ndarray, x, m, penalizer: float) -> float:
        p, q, v = np.exp(log_params)
        ll = (
            gammaln(p * x + q)
            - gammaln(p * x)
            - gammaln(q)
            + q * np.log(v)
            + (p * x - 1) * np.log(m)
            + (p * x) * np.log(x)
            - (p * x + q) * np.log(x * m + v)
        )
        return -ll.mean() + penalizer * np.sum(np.exp(log_params) ** 2)

    def fit(self, frequency, monetary_value, penalizer: float = 0.0) -> GammaGamma:
        x = np.asarray(frequency, dtype=float)
        m = np.asarray(monetary_value, dtype=float)
        mask = (x > 0) & (m > 0)
        res = minimize(
            self._neg_ll,
            x0=np.log([1.0, 2.0, max(np.median(m[mask]), 1.0)]),
            args=(x[mask], m[mask], penalizer),
            method="Nelder-Mead",
            options={"maxiter": 20_000, "xatol": 1e-8, "fatol": 1e-10},
        )
        if not res.success:  # pragma: no cover
            raise RuntimeError(f"Gamma-Gamma fit did not converge: {res.message}")
        self.p, self.q, self.v = np.exp(res.x)
        self.empirical_mean = float(m[mask].mean())
        return self

    @property
    def population_mean(self) -> float:
        """E[spend per transaction] for a customer with no repeat history.

        Equals p*v/(q-1), which only exists for q > 1. Heavy-tailed spend (e.g. wholesalers)
        can yield q <= 1, i.e. an infinite theoretical mean; the empirical mean is used instead.
        """
        if self.q > 1:
            return self.p * self.v / (self.q - 1)
        return self.empirical_mean

    def expected_value(self, frequency, monetary_value) -> np.ndarray:
        """Conditional expected spend per transaction (shrinks noisy averages to the population)."""
        x = np.asarray(frequency, dtype=float)
        m = np.asarray(monetary_value, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            conditional = self.p * (self.v + x * m) / (self.p * x + self.q - 1)
        return np.where(x > 0, conditional, self.population_mean)

    @property
    def params(self) -> dict[str, float]:
        return {"p": self.p, "q": self.q, "v": self.v}


def predict_revenue(
    bg: BetaGeo, gg: GammaGamma, summary: pd.DataFrame, horizon_days: float
) -> pd.Series:
    """Expected revenue in the next ``horizon_days`` = E[purchases] x E[spend per purchase]."""
    n = bg.expected_purchases(horizon_days, summary["frequency"], summary["recency"], summary["T"])
    v = gg.expected_value(summary["frequency"], summary["monetary_value"])
    return pd.Series(n * v, index=summary.index, name="btyd_pred")
