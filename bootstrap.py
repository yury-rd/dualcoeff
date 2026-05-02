"""
BetaSCombinedCI — generated-regressor bootstrap correction (Pagan 1984).

Stacked-panel pooled OLS with edge fixed effects treats the structural
coefficient beta^S as known. When beta^S is itself drawn from a documented
prior (engineering uncertainty), cluster-robust SEs understate true
uncertainty.

This bootstrap samples beta^S from per-edge uniform priors, refits the
pooled OLS at each draw, and combines the across-prior variance with the
within-prior cluster sampling variance:

    Var_total = Var(beta_hat across draws) + E[SE_cluster^2]

CI computed from parametric draws of Normal(beta_hat_d, se_cluster_d) at
each prior draw d.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class BetaSCombinedCI:
    """Combined-CI bootstrap for pooled-panel beta with prior on beta^S.

    Parameters
    ----------
    edges : list of (source, target, beta_s_low, beta_s_high) tuples
    log_prices : pd.DataFrame
    horizon : int (default 6 — 6m cumulative log returns)
    n_draws : int (default 2000)
    seed : int

    Returns from .run() : dict with point, ci_95_combined, p_vs_1.
    """

    edges: list
    log_prices: pd.DataFrame
    horizon: int = 6
    n_draws: int = 2000
    seed: int = 42

    def _build_panel(self, beta_s_vec: dict) -> pd.DataFrame:
        rows = []
        for source, target, _lo, _hi in self.edges:
            r_src = self.log_prices[source].diff(self.horizon).iloc[self.horizon::self.horizon]
            r_tgt = self.log_prices[target].diff(self.horizon).iloc[self.horizon::self.horizon]
            df = pd.DataFrame(
                {"y": r_tgt.values, "x": r_src.values}, index=r_tgt.index,
            ).dropna()
            df["xw"] = beta_s_vec[(source, target)] * df["x"]
            df["edge_id"] = f"{source}->{target}"
            rows.append(df)
        panel = pd.concat(rows, axis=0)
        panel["edge_id"] = panel["edge_id"].astype("category")
        return panel

    def _fit_clustered(self, panel: pd.DataFrame) -> tuple[float, float]:
        y = panel["y"].values
        edge_dummies = pd.get_dummies(panel["edge_id"], drop_first=True).astype(float).values
        X = np.column_stack([np.ones(len(panel)), panel["xw"].values, edge_dummies])
        date_codes = panel.index.values.astype("datetime64[ns]").astype("int64")
        fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": date_codes})
        return float(fit.params[1]), float(fit.bse[1])

    def run(self) -> dict:
        rng = np.random.default_rng(self.seed)
        betas = []
        ses = []
        for _ in range(self.n_draws):
            beta_s_vec = {
                (s, t): rng.uniform(lo, hi) for (s, t, lo, hi) in self.edges
            }
            panel = self._build_panel(beta_s_vec)
            beta, se = self._fit_clustered(panel)
            betas.append(beta)
            ses.append(se)
        betas = np.array(betas)
        ses = np.array(ses)
        z = rng.standard_normal(len(betas))
        betas_combined = betas + z * ses
        ci = (
            float(np.percentile(betas_combined, 2.5)),
            float(np.percentile(betas_combined, 97.5)),
        )
        p_vs_1 = float(2 * min(
            (betas_combined < 1).mean(), (betas_combined > 1).mean()
        ))
        return {
            "n_draws": self.n_draws,
            "beta_mean": float(betas.mean()),
            "ci_95_combined": ci,
            "p_vs_1": p_vs_1,
            "var_share_from_prior": float(betas.var(ddof=1) / (betas.var(ddof=1) + np.mean(ses ** 2))),
        }
