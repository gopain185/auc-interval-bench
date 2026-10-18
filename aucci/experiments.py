"""The three experiments of the benchmark. Each returns a tidy table; `summarise_*` turns it into coverage."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm

from .intervals import PAIRED, SINGLE, delong
from .sim import auc_to_d, sample, true_auc

SEED = 20261019
ALPHA = 0.05


def single_coverage(ns=(200, 1000, 10_000), prevalences=(0.5, 0.05), aucs=(0.75, 0.9, 0.98), levels=(None, 5),
                    reps=600, boot_reps=150, B=200) -> pd.DataFrame:
    """One interval per method and replicate: does it contain the population AUC?"""
    rows = []
    for (n, prev, a, lv), i in zip(itertools.product(ns, prevalences, aucs, levels), itertools.count()):
        rng = np.random.default_rng([SEED, 1, i])
        d = auc_to_d(a)
        truth = true_auc(d, lv)
        for rep in range(reps):
            y, s = sample(rng, n, prev, d, levels=lv)
            for name, fn in SINGLE.items():
                if name == "bootstrap":
                    if rep >= boot_reps:
                        continue
                    lo, hi = fn(y, s, ALPHA, B=B, rng=rng)
                else:
                    lo, hi = fn(y, s, ALPHA)
                rows.append((n, prev, a, "continuous" if lv is None else f"{lv} levels", int(y.sum()), rep, name,
                             truth, lo, hi))
    return pd.DataFrame(rows, columns=["n", "prevalence", "auc", "scores", "positives", "rep", "method", "truth",
                                       "low", "high"])


def paired_difference(n=60_000, prevalence=0.43, auc=0.96, deltas=(0.0, 0.0003, 0.001), rhos=(0.9, 0.97, 0.99),
                      reps=400, boot_reps=60, B=200, boot_rho=0.97) -> pd.DataFrame:
    """Two models on the same rows, AUCs `auc` and `auc - delta`, errors correlated at `rho`."""
    rows = []
    for (delta, rho), i in zip(itertools.product(deltas, rhos), itertools.count()):
        rng = np.random.default_rng([SEED, 2, i])
        d = [auc_to_d(auc), auc_to_d(auc - delta)]
        truth = true_auc(d[0]) - true_auc(d[1])
        for rep in range(reps):
            y, S = sample(rng, n, prevalence, d, rho=rho)
            for name, fn in PAIRED.items():
                if name == "paired bootstrap":
                    if rho != boot_rho or rep >= boot_reps:
                        continue
                    lo, hi = fn(y, S, ALPHA, B=B, rng=rng)
                else:
                    lo, hi = fn(y, S, ALPHA)
                rows.append((n, prevalence, auc, delta, rho, rep, name, truth, lo, hi))
    return pd.DataFrame(rows, columns=["n", "prevalence", "auc", "delta", "rho", "rep", "method", "truth", "low",
                                       "high"])


def detectable_difference(ns=(10_000, 60_000, 200_000, 700_000), rhos=(0.9, 0.97, 0.99, 0.995), auc=0.96,
                          prevalence=0.43, big=2_000_000, power=0.8) -> pd.DataFrame:
    """Smallest AUC difference a paired DeLong test at level ALPHA detects with the given power.

    The variance of the difference scales as 1/n, so it is estimated once on a very large sample and rescaled.
    """
    rows = []
    d = auc_to_d(auc)
    for i, rho in enumerate(rhos):
        rng = np.random.default_rng([SEED, 3, i])
        y, S = sample(rng, big, prevalence, [d, d], rho=rho)
        _, cov = delong(y, S)
        v = (cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]) * big
        corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        for n in ns:
            se = np.sqrt(v / n)
            rows.append((rho, round(float(corr), 4), n, float(np.sqrt(cov[0, 0] * big / n)), float(se),
                         float((norm.ppf(1 - ALPHA / 2) + norm.ppf(power)) * se)))
    return pd.DataFrame(rows, columns=["rho", "auc_corr", "n", "se_single", "se_diff", "detectable"])


def summarise_single(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(miss_low=df.truth < df.low, miss_high=df.truth > df.high, width=df.high - df.low)
    g = df.groupby(["n", "prevalence", "auc", "scores", "method"], sort=False)
    out = g.agg(reps=("rep", "size"), positives=("positives", "mean"), miss_low=("miss_low", "mean"),
                miss_high=("miss_high", "mean"), width=("width", "mean")).reset_index()
    out["coverage"] = 1 - out.miss_low - out.miss_high
    return out


def summarise_paired(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(cover=(df.low <= df.truth) & (df.truth <= df.high), detect=df.low > 0, wrong_sign=df.high < 0,
                   width=df.high - df.low)
    g = df.groupby(["delta", "rho", "method"], sort=False)
    return g.agg(reps=("rep", "size"), coverage=("cover", "mean"), detect=("detect", "mean"),
                 wrong_sign=("wrong_sign", "mean"), width=("width", "mean")).reset_index()
