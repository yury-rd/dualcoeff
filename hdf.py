"""
HDF — Hedging Dampening Factor.

Definition:
    HDF_{ij} = 1 - beta_E / beta_S

where beta_S is a documented engineering / cost-survey structural coefficient
(uniform prior over [low, high]) and beta_E is the empirical 6-month
passthrough estimated from non-overlapping returns by OLS.

Confidence interval propagates both beta_E sampling variance (HC3) and the
beta_S prior variance via the delta method:
    Var(HDF) ≈ (1/beta_S)^2 * Var(beta_E) + (beta_E/beta_S^2)^2 * Var(beta_S)
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HDF:
    """Hedging Dampening Factor with delta-method 95% CI.

    Parameters
    ----------
    beta_s_low, beta_s_high : float
        Bounds of the uniform prior on the structural coefficient. The mean
        beta_S = (low+high)/2 is used as the central estimate; variance =
        (high-low)^2 / 12 (uniform-distribution variance).
    target_returns, source_returns : pandas.Series
        Non-overlapping 6-month log returns. Indices must align.

    Attributes (after .fit())
    -------------------------
    beta_e : float — OLS slope.
    se_beta_e : float — HC3 standard error.
    point : float — HDF point estimate at central beta_S.
    se_hdf : float — Total HDF standard error (delta method).
    ci_95 : tuple[float, float] — 95% CI for HDF.
    var_share_from_prior : float — Fraction of HDF variance from beta_S prior.
    """

    beta_s_low: float
    beta_s_high: float
    target_returns: pd.Series | None = None
    source_returns: pd.Series | None = None

    def fit(self) -> "HDF":
        if self.target_returns is None or self.source_returns is None:
            raise ValueError("supply target_returns and source_returns")
        df = pd.concat(
            [self.target_returns.rename("y"), self.source_returns.rename("x")],
            axis=1,
        ).dropna()
        n = len(df)
        if n < 10:
            raise ValueError(f"need >= 10 observations, got {n}")
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        x_centered = x - x.mean()
        beta_e = float(np.sum(x_centered * (y - y.mean())) / np.sum(x_centered ** 2))
        intercept = float(y.mean() - beta_e * x.mean())
        resid = y - (intercept + beta_e * x)
        # HC3 standard error
        h = x_centered ** 2 / np.sum(x_centered ** 2)
        adj_resid_sq = (resid / (1.0 - h)) ** 2
        var_beta_e = float(np.sum(x_centered ** 2 * adj_resid_sq) / np.sum(x_centered ** 2) ** 2)
        se_beta_e = float(np.sqrt(var_beta_e))

        beta_s_mean = 0.5 * (self.beta_s_low + self.beta_s_high)
        var_beta_s = (self.beta_s_high - self.beta_s_low) ** 2 / 12.0

        point = 1.0 - beta_e / beta_s_mean
        # delta-method variance
        d_d_beta_e = -1.0 / beta_s_mean
        d_d_beta_s = beta_e / (beta_s_mean ** 2)
        var_hdf = d_d_beta_e ** 2 * var_beta_e + d_d_beta_s ** 2 * var_beta_s
        se_hdf = float(np.sqrt(var_hdf))
        ci = (point - 1.96 * se_hdf, point + 1.96 * se_hdf)
        var_share_from_prior = float(d_d_beta_s ** 2 * var_beta_s / var_hdf)

        self.n = n
        self.beta_e = beta_e
        self.se_beta_e = se_beta_e
        self.point = point
        self.se_hdf = se_hdf
        self.ci_95 = ci
        self.var_share_from_prior = var_share_from_prior
        return self

    def __repr__(self) -> str:
        if not hasattr(self, "point"):
            return f"HDF(beta_s in [{self.beta_s_low}, {self.beta_s_high}], unfitted)"
        return (
            f"HDF(point={self.point:+.3f}, "
            f"95% CI=[{self.ci_95[0]:+.3f}, {self.ci_95[1]:+.3f}], "
            f"prior_var_share={self.var_share_from_prior:.1%}, n={self.n})"
        )
