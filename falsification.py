"""
NullPanel — pre-registered null-edge falsification.

Runs the same passthrough test as EdgeValidator on commodity pairs without
a structural prior. The expected pass rate under the null of no effect is
alpha; observed pass rates substantially below the validated-edge rate
demonstrate that framework results are not artefacts of the testing
procedure.
"""
from __future__ import annotations
from dataclasses import dataclass

import pandas as pd

from .validator import EdgeSpec, EdgeValidator


@dataclass
class NullPanel:
    """Pre-registered null edges and the validator they share with the
    validated set. ``denominator_k`` allows specifying the full pre-specified
    edge count even if some edges were dropped pre-test for data reasons
    (mirrors the conservative Bonferroni accounting in the paper)."""

    null_edges: list[EdgeSpec]
    log_prices: pd.DataFrame
    denominator_k: int | None = None
    alpha: float = 0.05
    horizon: int = 6

    def run(self) -> dict:
        k = self.denominator_k or len(self.null_edges)
        v = EdgeValidator(
            edges=self.null_edges,
            log_prices=self.log_prices,
            horizon=self.horizon,
            alpha=self.alpha,
        )
        # use the null-panel-specific Bonferroni denominator
        v.alpha = self.alpha * len(self.null_edges) / k
        results = v.run()
        passes = sum(1 for r in results if r.bonferroni_pass)
        return {
            "edges_tested": len(results),
            "edges_pre_specified": k,
            "passes": passes,
            "pass_rate": passes / max(len(results), 1),
            "expected_under_null": self.alpha,
            "results": results,
        }
