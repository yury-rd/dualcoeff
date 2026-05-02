"""
EdgeValidator — non-overlapping 6m panel validation w/ Bonferroni.

Implements the dual-coefficient validation step:
    1. For each edge, compute non-overlapping 6m correlation r and binomial
       directional consistency.
    2. Apply Bonferroni at family-wise alpha / k.
    3. Classify each edge as Strong / Moderate / Weak based on direction +
       Pearson significance.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, binomtest


@dataclass
class EdgeSpec:
    source: str
    target: str
    beta_s: float = 1.0


@dataclass
class EdgeResult:
    source: str
    target: str
    n: int
    r: float
    p_pearson: float
    direction_pct: float
    p_direction: float
    bonferroni_pass: bool
    classification: str


@dataclass
class EdgeValidator:
    edges: Iterable[EdgeSpec]
    log_prices: pd.DataFrame
    horizon: int = 6
    alpha: float = 0.05

    def run(self) -> list[EdgeResult]:
        edges = list(self.edges)
        k = len(edges)
        bonf_thresh = self.alpha / k
        results = []
        for spec in edges:
            r_src = self.log_prices[spec.source].diff(self.horizon).iloc[self.horizon::self.horizon]
            r_tgt = self.log_prices[spec.target].diff(self.horizon).iloc[self.horizon::self.horizon]
            df = pd.concat([r_tgt.rename("y"), r_src.rename("x")], axis=1).dropna()
            n = len(df)
            if n < 5:
                continue
            r, p_r = pearsonr(df["x"], df["y"])
            sign_match = (np.sign(df["x"]) == np.sign(df["y"])).sum()
            direction_pct = sign_match / n
            p_dir = binomtest(int(sign_match), n, 0.5, alternative="two-sided").pvalue
            bonf = p_r < bonf_thresh
            if direction_pct >= 0.75 and p_r < self.alpha and bonf:
                classification = "Strong"
            elif direction_pct >= 0.60 and p_r < self.alpha:
                classification = "Moderate"
            else:
                classification = "Weak"
            results.append(EdgeResult(
                source=spec.source, target=spec.target, n=int(n),
                r=float(r), p_pearson=float(p_r),
                direction_pct=float(direction_pct), p_direction=float(p_dir),
                bonferroni_pass=bool(bonf), classification=classification,
            ))
        return results
