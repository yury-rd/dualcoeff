"""
dualcoeff — Dual-Coefficient Framework for Physical Commodity Spillovers.

Reference: Lazarichev, Golubev, Novikov, Sultimov, Volkov, Maximov (2026).
"The Hedging Dampening Factor: A Dual-Coefficient Framework for Commodity
Cost Chains." Journal of Commodity Markets.

Public API:
    HDF                  — Hedging Dampening Factor with delta-method 95% CI.
    EdgeValidator        — Non-overlapping 6m panel validation w/ Bonferroni.
    NullPanel            — Pre-registered null-edge falsification panel.
    KalmanGraph          — State-space cross-innovation + state-premium predictor.
    BetaSCombinedCI      — Generated-regressor bootstrap correction (Pagan 1984).

Quick start:
    >>> from dualcoeff import HDF
    >>> hdf = HDF(beta_s_low=0.75, beta_s_high=0.85,
    ...           target_returns=urea_6m, source_returns=gas_6m)
    >>> hdf.fit()
    >>> hdf.point, hdf.ci_95
    (0.4979, (0.1691, 0.8266))
"""
from .hdf import HDF
from .validator import EdgeValidator
from .falsification import NullPanel
from .kalman import KalmanGraph
from .bootstrap import BetaSCombinedCI

__all__ = ["HDF", "EdgeValidator", "NullPanel", "KalmanGraph", "BetaSCombinedCI"]
__version__ = "0.1.0"
