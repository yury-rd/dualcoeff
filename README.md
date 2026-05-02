# dualcoeff

**Dual-Coefficient Framework for Physical Commodity Spillovers**

Companion package for the paper:

> Lazarichev, N., Golubev, A., Novikov, I., Sultimov, R., Volkov, A., Maximov, Yu. (2026).
> "The Hedging Dampening Factor: A Dual-Coefficient Framework for Commodity Cost Chains."
> *Journal of Commodity Markets*.

## What it does

`dualcoeff` implements the framework's five named methodological contributions on
user-supplied commodity-return panels:

| API | Contribution | Purpose |
|-----|--------------|---------|
| `HDF` | Hedging Dampening Factor | Quantify physical-passthrough absorption with delta-method 95% CI |
| `EdgeValidator` | Dual-Coefficient Framework (DCF) | Non-overlapping 6m validation with Bonferroni correction |
| `NullPanel` | Null-Edge Falsification Panel (NEFP) | Pre-registered null-edge falsification |
| `KalmanGraph` | State-Premium Predictor (SPP) | Kalman state-space cross-innovation + premium-return prediction |
| `BetaSCombinedCI` | β^S Combined-CI Bootstrap | Generated-regressor variance correction (Pagan 1984) |

## Quick start

```python
import pandas as pd
from dualcoeff import HDF, EdgeValidator, NullPanel, KalmanGraph, BetaSCombinedCI
from dualcoeff.validator import EdgeSpec

log_prices = pd.read_parquet("monthly_prices_all.parquet")  # your data

# 1. Headline HDF
hdf = HDF(beta_s_low=0.75, beta_s_high=0.85,
          target_returns=log_prices["urea"].diff(6).iloc[6::6],
          source_returns=log_prices["gas"].diff(6).iloc[6::6])
print(hdf.fit())  # HDF(point=+0.50, 95% CI=[+0.17, +0.83], ...)

# 2. Validate a custom edge set
edges = [EdgeSpec("gas", "urea", 0.80), EdgeSpec("urea", "corn", 0.12)]
validator = EdgeValidator(edges, log_prices)
for r in validator.run():
    print(r)

# 3. Falsify against null edges
null_edges = [EdgeSpec("gold", "wheat"), EdgeSpec("coffee", "soy")]
panel = NullPanel(null_edges, log_prices, denominator_k=10)
print(panel.run())

# 4. Kalman state-premium predictor (4-commodity subsystem)
import numpy as np
A = np.array([[0.90, 0, 0, 0],
              [0.10, 0.77, 0.20, 0],
              [0,    0.20, 0.79, 0],
              [0,    0.65, 0,    0.09]])
graph = KalmanGraph(A, sigma_eps=0.06, sigma_eta=0.04,
                    log_prices=log_prices[["gas", "corn", "wheat", "hogs"]])
graph.fit_filter()
print(graph.state_premium_predict("hogs"))

# 5. β^S combined-CI bootstrap
edges_with_priors = [("soy", "soymeal", 0.65, 0.75),
                     ("gas", "corn",    0.07, 0.13)]
boot = BetaSCombinedCI(edges_with_priors, log_prices, n_draws=2000)
print(boot.run())  # {'beta_mean': +0.77, 'ci_95_combined': [+0.37, +1.17], ...}
```

## Installation

```bash
pip install dualcoeff
```

(For now: `pip install -e .` from this directory.)

## Replication archive

The replication archive accompanying the paper contains the frozen parquet
data files used to produce every result reported in the manuscript, along
with figure-generation scripts that exercise every entry point in
`dualcoeff`. See the parent repository for archive structure.

## Citation

```bibtex
@article{lazarichev2026hdf,
  title   = {The Hedging Dampening Factor:
             A Dual-Coefficient Framework for Commodity Cost Chains},
  author  = {Lazarichev, Nikita and Golubev, Alexey and Novikov, Ivan and
             Sultimov, Roman and Volkov, Aleksandr and Maximov, Yury},
  journal = {Journal of Commodity Markets},
  year    = {2026}
}
```

## License

MIT.
