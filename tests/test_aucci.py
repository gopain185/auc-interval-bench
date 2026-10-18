import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from aucci import experiments as ex
from aucci.intervals import PAIRED, SINGLE, auc, delong, diff_paired_delong, diff_unpaired
from aucci.sim import auc_to_d, sample, true_auc


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y, s = sample(rng, 3000, 0.3, auc_to_d(0.85), levels=4)
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


@pytest.mark.parametrize("levels", [None, 3, 6])
def test_true_auc_matches_large_sample(levels):
    rng = np.random.default_rng(1)
    d = auc_to_d(0.9)
    y, s = sample(rng, 400_000, 0.5, d, levels=levels)
    assert auc(y, s) == pytest.approx(true_auc(d, levels), abs=0.002)


def test_ties_lower_the_population_auc():
    d = auc_to_d(0.9)
    assert true_auc(d, 3) < true_auc(d, 10) < true_auc(d)


def test_delong_matches_direct_placement_formula():
    rng = np.random.default_rng(2)
    y, s = sample(rng, 300, 0.4, auc_to_d(0.8), levels=7)
    pos, neg = s[y == 1], s[y == 0]
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    v10, v01 = psi.mean(1), psi.mean(0)
    var = v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg)
    a, cov = delong(y, s)
    assert a[0] == pytest.approx(psi.mean(), abs=1e-12)
    assert cov[0, 0] == pytest.approx(var, rel=1e-10)


def test_delong_variance_matches_sampling_variance():
    rng = np.random.default_rng(3)
    d = auc_to_d(0.9)
    est, var = [], []
    for _ in range(1500):
        y, s = sample(rng, 400, 0.25, d)
        a, cov = delong(y, s)
        est.append(a[0])
        var.append(cov[0, 0])
    assert np.mean(var) == pytest.approx(np.var(est), rel=0.12)


def test_pairing_shrinks_the_difference_interval():
    rng = np.random.default_rng(4)
    y, S = sample(rng, 5000, 0.4, [auc_to_d(0.9), auc_to_d(0.89)], rho=0.95)
    lo_p, hi_p = diff_paired_delong(y, S)
    lo_u, hi_u = diff_unpaired(y, S)
    assert hi_p - lo_p < 0.5 * (hi_u - lo_u)


def test_every_interval_contains_its_estimate():
    rng = np.random.default_rng(5)
    y, S = sample(rng, 800, 0.3, [auc_to_d(0.85), auc_to_d(0.83)], rho=0.8)
    a = auc(y, S[:, 0])
    for fn in SINGLE.values():
        lo, hi = fn(y, S[:, 0])
        assert lo <= a <= hi
    diff = a - auc(y, S[:, 1])
    for fn in PAIRED.values():
        lo, hi = fn(y, S)
        assert lo <= diff <= hi


def test_delong_coverage_is_nominal_on_easy_setting():
    df = ex.summarise_single(ex.single_coverage(ns=(2000,), prevalences=(0.5,), aucs=(0.8,), levels=(None,),
                                                reps=400, boot_reps=0))
    cov = df.set_index("method").coverage
    assert cov["DeLong (Wald)"] == pytest.approx(0.95, abs=0.03)


def test_paired_test_holds_its_level_and_detects_large_differences():
    df = ex.summarise_paired(ex.paired_difference(n=20_000, deltas=(0.0, 0.003), rhos=(0.97,), reps=150,
                                                  boot_reps=0))
    t = df[df.method == "paired DeLong"].set_index("delta")
    assert t.loc[0.0, "detect"] + t.loc[0.0, "wrong_sign"] < 0.1
    assert t.loc[0.003, "detect"] > 0.9


def test_detectable_difference_shrinks_with_n_and_rho():
    df = ex.detectable_difference(ns=(10_000, 100_000), rhos=(0.9, 0.99), big=300_000)
    t = df.set_index(["rho", "n"]).detectable
    assert t[(0.9, 100_000)] < t[(0.9, 10_000)]
    assert t[(0.99, 10_000)] < t[(0.9, 10_000)]
    assert t[(0.9, 10_000)] / t[(0.9, 100_000)] == pytest.approx(np.sqrt(10), rel=1e-9)
