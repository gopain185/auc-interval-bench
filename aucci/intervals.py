"""AUC, its DeLong covariance, and confidence intervals for one AUC or for the difference of two.

All intervals are two-sided at level `1 - alpha`. Each function returns `(low, high)`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as one half."""
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def delong(y: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUCs of the columns of `S` and their DeLong covariance matrix (Sun & Xu 2014 fast algorithm)."""
    y = np.asarray(y).astype(bool)
    S = np.asarray(S, dtype=float).reshape(len(y), -1)
    pos, neg = S[y], S[~y]
    m, n = len(pos), len(neg)
    tx = np.column_stack([rankdata(c) for c in pos.T])
    ty = np.column_stack([rankdata(c) for c in neg.T])
    tz = np.column_stack([rankdata(c) for c in np.vstack([pos, neg]).T])
    aucs = (tz[:m].sum(0) / m - (m + 1) / 2) / n
    v01 = (tz[:m] - tx) / n  # placement of each positive among the negatives
    v10 = 1 - (tz[m:] - ty) / m  # placement of each negative among the positives
    cov = np.atleast_2d(np.cov(v01, rowvar=False)) / m + np.atleast_2d(np.cov(v10, rowvar=False)) / n
    return aucs, cov


def _z(alpha: float) -> float:
    return float(norm.ppf(1 - alpha / 2))


def ci_wald_delong(y, s, alpha=0.05):
    a, cov = delong(y, s)
    h = _z(alpha) * np.sqrt(cov[0, 0])
    return float(a[0] - h), float(a[0] + h)


def ci_logit_delong(y, s, alpha=0.05):
    """DeLong variance carried to the logit scale by the delta method, so the interval stays inside (0, 1)."""
    a, cov = delong(y, s)
    a = float(np.clip(a[0], 1e-9, 1 - 1e-9))
    se = np.sqrt(cov[0, 0]) / (a * (1 - a))
    lo, hi = np.log(a / (1 - a)) + np.array([-1, 1]) * _z(alpha) * se
    return float(1 / (1 + np.exp(-lo))), float(1 / (1 + np.exp(-hi)))


def ci_hanley_mcneil(y, s, alpha=0.05):
    """Hanley & McNeil (1982): variance from the AUC and the class counts alone (exponential approximation)."""
    y = np.asarray(y).astype(bool)
    m, n, a = y.sum(), (~y).sum(), auc(y, s)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var = (a * (1 - a) + (m - 1) * (q1 - a * a) + (n - 1) * (q2 - a * a)) / (m * n)
    h = _z(alpha) * np.sqrt(max(var, 0.0))
    return float(a - h), float(a + h)


def _bootstrap_aucs(rng, y, S, B):
    """AUCs of the columns of `S` on `B` resamples, stratified by class so both classes are always present."""
    y = np.asarray(y).astype(bool)
    S = np.asarray(S, dtype=float).reshape(len(y), -1)
    ip, ineg = np.flatnonzero(y), np.flatnonzero(~y)
    out = np.empty((B, S.shape[1]))
    for b in range(B):
        idx = np.concatenate([rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))])
        yb = y[idx]
        for j in range(S.shape[1]):
            out[b, j] = auc(yb, S[idx, j])
    return out


def ci_bootstrap(y, s, alpha=0.05, B=400, rng=None):
    """Stratified percentile bootstrap."""
    boot = _bootstrap_aucs(rng or np.random.default_rng(0), y, s, B)[:, 0]
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def diff_paired_delong(y, S, alpha=0.05):
    """Interval for AUC(S[:, 0]) - AUC(S[:, 1]) on the same rows, with the DeLong covariance of the pair."""
    a, cov = delong(y, S)
    h = _z(alpha) * np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0))
    return float(a[0] - a[1] - h), float(a[0] - a[1] + h)


def diff_unpaired(y, S, alpha=0.05):
    """The common mistake: add the two variances as if the models were scored on independent rows."""
    a, cov = delong(y, S)
    h = _z(alpha) * np.sqrt(cov[0, 0] + cov[1, 1])
    return float(a[0] - a[1] - h), float(a[0] - a[1] + h)


def diff_paired_bootstrap(y, S, alpha=0.05, B=400, rng=None):
    boot = _bootstrap_aucs(rng or np.random.default_rng(0), y, S, B)
    diff = boot[:, 0] - boot[:, 1]
    return float(np.quantile(diff, alpha / 2)), float(np.quantile(diff, 1 - alpha / 2))


SINGLE = {"DeLong (Wald)": ci_wald_delong, "DeLong (logit)": ci_logit_delong,
          "Hanley-McNeil": ci_hanley_mcneil, "bootstrap": ci_bootstrap}
PAIRED = {"paired DeLong": diff_paired_delong, "unpaired": diff_unpaired, "paired bootstrap": diff_paired_bootstrap}
