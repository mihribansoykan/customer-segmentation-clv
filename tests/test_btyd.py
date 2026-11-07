"""The hand-written BG/NBD and Gamma-Gamma models must recover parameters from simulated data."""

import numpy as np
import pandas as pd
import pytest

from retail_clv import btyd


def simulate_bgnbd(n, r, alpha, a, b, T, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(r, 1 / alpha, n)
    p = rng.beta(a, b, n)
    x = np.zeros(n)
    t_x = np.zeros(n)
    for i in range(n):
        t = 0.0
        while True:
            t += rng.exponential(1 / lam[i])
            if t > T:
                break
            x[i] += 1
            t_x[i] = t
            if rng.random() < p[i]:
                break
    return x, t_x, np.full(n, T)


@pytest.fixture(scope="module")
def simulated():
    return simulate_bgnbd(4000, r=0.8, alpha=40.0, a=0.6, b=2.5, T=365, seed=1)


def test_bgnbd_recovers_parameters(simulated):
    x, t_x, T = simulated
    bg = btyd.BetaGeo().fit(x, t_x, T)
    assert bg.r == pytest.approx(0.8, rel=0.25)
    assert bg.alpha == pytest.approx(40.0, rel=0.35)
    assert bg.a == pytest.approx(0.6, rel=0.4)
    assert bg.b == pytest.approx(2.5, rel=0.5)


def test_bgnbd_predictions_are_sane(simulated):
    x, t_x, T = simulated
    bg = btyd.BetaGeo().fit(x, t_x, T)
    pa = bg.p_alive(x, t_x, T)
    assert ((pa >= 0) & (pa <= 1)).all()
    assert (pa[x == 0] == 1).all()
    e30 = bg.expected_purchases(30, x, t_x, T)
    e90 = bg.expected_purchases(90, x, t_x, T)
    assert (e90 >= e30).all() and (e30 >= 0).all()
    # A frequent buyer who has just purchased should be more valuable than one silent for a year.
    recent = bg.expected_purchases(90, np.array([10.0]), np.array([360.0]), np.array([365.0]))
    stale = bg.expected_purchases(90, np.array([10.0]), np.array([30.0]), np.array([365.0]))
    assert recent[0] > stale[0]


def test_gamma_gamma_recovers_mean_spend():
    rng = np.random.default_rng(2)
    p, q, v = 6.0, 4.0, 60.0
    n = 3000
    nu = rng.gamma(q, 1 / v, n)  # per-customer rate
    x = rng.integers(1, 15, n).astype(float)
    m = np.array([rng.gamma(p, 1 / nu_i, int(k)).mean() for nu_i, k in zip(nu, x, strict=True)])
    gg = btyd.GammaGamma().fit(x, m)
    assert gg.p * gg.v / (gg.q - 1) == pytest.approx(p * v / (q - 1), rel=0.15)
    ev = gg.expected_value(x, m)
    # shrinkage: conditional expectation lies between the customer's mean and the population mean
    pop = gg.p * gg.v / (gg.q - 1)
    lo, hi = np.minimum(m, pop), np.maximum(m, pop)
    assert ((ev >= lo - 1e-6) & (ev <= hi + 1e-6)).all()


def test_gamma_gamma_heavy_tail_fallback():
    """q <= 1 means an infinite theoretical mean: use the empirical mean, never negative."""
    gg = btyd.GammaGamma(p=10.0, q=0.9, v=10.0, empirical_mean=250.0)
    assert gg.population_mean == 250.0
    ev = gg.expected_value(np.array([0.0, 3.0]), np.array([0.0, 100.0]))
    assert ev[0] == 250.0 and ev[1] > 0


def test_summary_data_and_prediction(tx):
    from retail_clv.data import invoices

    end = pd.Timestamp("2011-06-08")
    inv = invoices(tx)
    s = btyd.summary_data(inv, end)
    # every customer with a purchase on or before the end *day* (any time of day) is included
    expected = inv.loc[inv["invoice_date"] < end + pd.Timedelta(days=1), "customer_id"].unique()
    assert set(s.index) == set(expected)
    assert (s["recency"] <= s["T"]).all()
    assert (s.loc[s["frequency"] == 0, "monetary_value"] == 0).all()
    bg = btyd.BetaGeo().fit(s["frequency"], s["recency"], s["T"], penalizer=1e-3)
    gg = btyd.GammaGamma().fit(s["frequency"], s["monetary_value"], penalizer=1e-3)
    pred = btyd.predict_revenue(bg, gg, s, 182)
    assert (pred >= 0).all() and pred.index.equals(s.index)
