"""
KalmanGraph — State-Premium Predictor (SPP).

Constrained linear-Gaussian state-space model with structural off-diagonals
encoding the validated commodity graph. The state premium pi_t = y_t - x_hat_t
predicts the next-month return r_{t+1} via mean-reversion to the implied
fundamental.

Identification:
    Persistence diagonals constrained to (-1, 1) for stationarity.
    Structural off-diagonals fixed at beta^S (not estimated).
    Noise covariances Q, R fixed at calibrated values to avoid collapse to
    perfect tracking. Only the ratio sigma_eps / sigma_eta is identified.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KalmanGraph:
    """State-space model with constrained transition matrix and fixed noise.

    Parameters
    ----------
    A : np.ndarray (k, k)
        Transition matrix. Off-diagonals are structural beta^S; diagonals
        are own-dynamics persistence.
    sigma_eps : float — observation-noise std-dev.
    sigma_eta : float — state-innovation std-dev.
    log_prices : pd.DataFrame (T, k)

    Methods
    -------
    fit_filter()             — run Kalman filter, store premia.
    state_premium_predict()  — return mean-reversion correlation r_mr.
    fit_expanding_window()   — refit at each test date for honest OOS.
    """

    A: np.ndarray
    sigma_eps: float
    sigma_eta: float
    log_prices: pd.DataFrame

    def __post_init__(self):
        self.k = self.A.shape[0]
        if self.A.shape != (self.k, self.k):
            raise ValueError("A must be square")
        if self.log_prices.shape[1] != self.k:
            raise ValueError(f"log_prices must have {self.k} columns")
        self.Q = np.eye(self.k) * self.sigma_eta ** 2
        self.R = np.eye(self.k) * self.sigma_eps ** 2

    def fit_filter(self) -> "KalmanGraph":
        T = len(self.log_prices)
        y = self.log_prices.to_numpy()
        x_filt = np.zeros((T, self.k))
        P = np.eye(self.k) * 1.0
        x = y[0].copy()
        for t in range(T):
            # predict
            if t > 0:
                x = self.A @ x
                P = self.A @ P @ self.A.T + self.Q
            # update
            S = P + self.R
            K = P @ np.linalg.inv(S)
            innov = y[t] - x
            x = x + K @ innov
            P = (np.eye(self.k) - K) @ P
            x_filt[t] = x
        self.x_filt = x_filt
        self.premium = pd.DataFrame(
            y - x_filt, index=self.log_prices.index, columns=self.log_prices.columns,
        )
        return self

    def state_premium_predict(self, commodity: str) -> dict:
        if not hasattr(self, "premium"):
            raise RuntimeError("call fit_filter() first")
        pi = self.premium[commodity]
        ret = self.log_prices[commodity].diff().shift(-1)
        df = pd.concat([pi.rename("pi"), ret.rename("ret")], axis=1).dropna()
        # r_mr defined as -corr(pi, ret) so positive = mean-reversion
        r_mr = -float(df["pi"].corr(df["ret"]))
        from scipy.stats import pearsonr
        _, p = pearsonr(df["pi"], df["ret"])
        return {"commodity": commodity, "r_mr": r_mr, "p": float(p), "n": int(len(df))}
