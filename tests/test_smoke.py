"""Smoke tests for dualcoeff. Exercises every public class on synthetic data
to verify nothing is import- or shape-broken end to end.

Run:
    pip install -e .
    pytest tests/

Or:
    python tests/test_smoke.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dualcoeff import (
    BetaSCombinedCI,
    EdgeValidator,
    HDF,
    KalmanGraph,
    NullPanel,
)
from dualcoeff.validator import EdgeSpec


def _synthetic_log_prices(seed: int = 0, n_months: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-31", periods=n_months, freq="ME")
    # Two correlated log-price series + several near-independent ones
    z = rng.standard_normal((n_months, 5))
    # source: random walk
    src = np.cumsum(z[:, 0] * 0.05)
    # target: 0.6 * src + idiosyncratic
    tgt = 0.6 * src + np.cumsum(z[:, 1] * 0.04)
    other = np.cumsum(z[:, 2:] * 0.05, axis=0)
    df = pd.DataFrame(
        np.column_stack([src, tgt, other]),
        index=idx,
        columns=["gas", "urea", "wheat", "corn", "hogs"],
    )
    # Convert to log-prices by adding a positive offset before exp / log
    return np.log(np.exp(df) + 10)


def test_hdf_runs_and_returns_finite_ci():
    log_prices = _synthetic_log_prices()
    hdf = HDF(
        beta_s_low=0.55,
        beta_s_high=0.65,
        target_returns=log_prices["urea"].diff(6).iloc[6::6],
        source_returns=log_prices["gas"].diff(6).iloc[6::6],
    ).fit()
    assert np.isfinite(hdf.point)
    assert np.isfinite(hdf.ci_95[0]) and np.isfinite(hdf.ci_95[1])
    assert hdf.ci_95[0] < hdf.ci_95[1]
    assert 0 <= hdf.var_share_from_prior <= 1


def test_edge_validator_runs():
    log_prices = _synthetic_log_prices()
    edges = [EdgeSpec("gas", "urea", 0.6), EdgeSpec("wheat", "corn", 0.2)]
    results = EdgeValidator(edges, log_prices).run()
    assert len(results) == 2
    for r in results:
        assert r.classification in {"Strong", "Moderate", "Weak"}
        assert r.n > 0


def test_null_panel_runs():
    log_prices = _synthetic_log_prices()
    null_edges = [EdgeSpec("hogs", "wheat"), EdgeSpec("hogs", "corn")]
    out = NullPanel(null_edges, log_prices, denominator_k=10).run()
    assert "passes" in out
    assert out["passes"] >= 0


def test_kalman_graph_runs():
    log_prices = _synthetic_log_prices()
    A = np.array(
        [
            [0.95, 0.0, 0.0, 0.0],
            [0.10, 0.80, 0.20, 0.0],
            [0.00, 0.20, 0.80, 0.0],
            [0.00, 0.65, 0.0, 0.10],
        ]
    )
    sub = log_prices[["gas", "corn", "wheat", "hogs"]]
    graph = KalmanGraph(A, sigma_eps=0.06, sigma_eta=0.04, log_prices=sub).fit_filter()
    out = graph.state_premium_predict("hogs")
    assert np.isfinite(out["r_mr"])
    assert 0 <= out["p"] <= 1


def test_beta_s_combined_ci_runs():
    log_prices = _synthetic_log_prices()
    edges = [
        ("gas", "urea", 0.55, 0.65),
        ("wheat", "corn", 0.10, 0.30),
    ]
    out = BetaSCombinedCI(edges, log_prices, n_draws=50, seed=1).run()
    assert "ci_95_combined" in out
    lo, hi = out["ci_95_combined"]
    assert lo < hi
    assert 0 <= out["var_share_from_prior"] <= 1


if __name__ == "__main__":
    test_hdf_runs_and_returns_finite_ci()
    test_edge_validator_runs()
    test_null_panel_runs()
    test_kalman_graph_runs()
    test_beta_s_combined_ci_runs()
    print("all smoke tests pass")
