"""Binormal scores with a known population AUC.

A positive row scores `d + e`, a negative row `e`, with `e ~ N(0, 1)`, so the population AUC is
`Phi(d / sqrt(2))`. Two models on the same rows share the label and have errors with correlation `rho`. Scores can
be rounded to a few levels to create ties; the population AUC (with ties counted as one half) is then computed
exactly from the bin probabilities.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def auc_to_d(auc: float) -> float:
    return float(np.sqrt(2) * norm.ppf(auc))


def true_auc(d: float, levels: int | None = None) -> float:
    """Population AUC of the binormal model; with `levels`, of the scores rounded to that many bins."""
    if levels is None:
        return float(norm.cdf(d / np.sqrt(2)))
    edges = _edges(d, levels)
    p0, p1 = np.diff(norm.cdf(edges)), np.diff(norm.cdf(edges - d))
    below0 = np.concatenate([[0.0], np.cumsum(p0)[:-1]])
    return float(np.sum(p1 * (below0 + 0.5 * p0)))


def _edges(d: float, levels: int) -> np.ndarray:
    """Bin edges at equally spaced quantiles of the pooled (50/50) score distribution."""
    grid = np.linspace(-6, d + 6, 4001)
    mix = 0.5 * (norm.cdf(grid) + norm.cdf(grid - d))
    inner = np.interp(np.arange(1, levels) / levels, mix, grid)
    return np.concatenate([[-np.inf], inner, [np.inf]])


def sample(rng: np.random.Generator, n: int, prevalence: float, d: float | np.ndarray, rho: float = 0.0,
           levels: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores for one model (scalar `d`, scores shape (n,)) or several (vector `d`, shape (n, k))."""
    d = np.atleast_1d(np.asarray(d, dtype=float))
    k = len(d)
    n1 = max(2, min(n - 2, rng.binomial(n, prevalence)))  # at least two rows per class, so the DeLong variance exists
    y = np.zeros(n, dtype=np.int8)
    y[:n1] = 1
    cov = np.full((k, k), rho) + (1 - rho) * np.eye(k)
    e = rng.multivariate_normal(np.zeros(k), cov, size=n) if k > 1 else rng.standard_normal((n, 1))
    s = e + np.outer(y, d)
    if levels is not None:
        s = np.column_stack([np.searchsorted(_edges(dj, levels), s[:, j]) for j, dj in enumerate(d)]).astype(float)
    return y, (s[:, 0] if k == 1 else s)
