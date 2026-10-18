"""Builds the Kaggle dataset and notebook for this project.

    python kaggle/build_kaggle.py [--public] [--notebook-only] [--reuse-results]

writes kaggle/dataset/ (one paired sample with a known AUC difference, the result tables of the three experiments
and the datasheet) and kaggle/notebook/ (a CPU notebook that embeds the package source and reruns the experiments).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aucci.experiments import SEED, detectable_difference, paired_difference, single_coverage  # noqa: E402
from aucci.sim import auc_to_d, sample, true_auc  # noqa: E402

OUT = ROOT / "kaggle"
DATASET_ID = "locdunhat/auc-confidence-interval-simulations"
NOTEBOOK_ID = "locdunhat/is-that-auc-gain-real-confidence-intervals"
TITLE = "Is That AUC Gain Real? Confidence Intervals"
N, PREVALENCE, AUC, DELTA, RHO = 60_000, 0.43, 0.96, 0.0003, 0.97


def build_dataset(public: bool, rerun: bool = True) -> Path:
    out = OUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([SEED, 0])
    d = [auc_to_d(AUC), auc_to_d(AUC - DELTA)]
    y, S = sample(rng, N, PREVALENCE, d, rho=RHO)
    pd.DataFrame({"row_id": np.arange(N), "y": y.astype(int), "model_a": S[:, 0].round(6),
                  "model_b": S[:, 1].round(6)}).to_csv(out / "paired_scores.csv", index=False)
    pd.DataFrame([{"model": "model_a", "signal": d[0], "true_auc": true_auc(d[0])},
                  {"model": "model_b", "signal": d[1], "true_auc": true_auc(d[1])}]).assign(
        error_correlation=RHO).to_csv(out / "paired_truth.csv", index=False)
    if rerun:
        for name, fn in [("exp1_single_coverage", single_coverage), ("exp2_paired_difference", paired_difference),
                         ("exp3_detectable_difference", detectable_difference)]:
            fn().to_csv(out / f"{name}.csv", index=False)
    else:  # reuse the tables written by `python -m aucci results`
        for f in sorted((ROOT / "results").glob("exp*.csv")):
            (out / f.name).write_bytes(f.read_bytes())
    datasheet = (OUT / "DATASHEET.md").read_text(encoding="utf-8")
    (out / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "AUC Confidence Interval Simulations",
        "id": DATASET_ID,
        "subtitle": "Coverage of ROC AUC confidence intervals on scores with a known true AUC",
        "description": datasheet,
        "isPrivate": not public,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["tabular", "benchmark", "classification"],
    }, indent=2), encoding="utf-8")
    return out


def _sources() -> dict[str, str]:
    files = sorted((ROOT / "aucci").glob("*.py"))
    return {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in files}


SETUP = """

import os, sys
ROOT = "/kaggle/working/auc-interval-bench" if os.path.isdir("/kaggle/working") else os.path.abspath("auc-interval-bench")
for path, text in SOURCES.items():
    os.makedirs(os.path.dirname(os.path.join(ROOT, path)), exist_ok=True)
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        f.write(text)
sys.path.insert(0, ROOT)
"""

IMPORTS = """import time, warnings
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from aucci import experiments as ex
pd.set_option("display.precision", 4); pd.set_option("display.width", 160)
"""

EXP1 = """t = time.time(); e1 = ex.summarise_single(ex.single_coverage()); print(round(time.time() - t), "s")
e1.pivot_table(index=["scores", "n", "prevalence", "auc"], columns="method", values="coverage", sort=False)
"""

PLOT1 = """fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, scores in zip(axes, ["continuous", "5 levels"]):
    g = e1[e1.scores == scores].copy()
    g["setting"] = g.n.astype(str) + " / " + g.prevalence.astype(str) + " / " + g.auc.astype(str)
    for method, h in g.groupby("method", sort=False):
        ax.plot(h.setting, h.coverage, marker="o", label=method)
    ax.axhline(0.95, color="k", lw=0.8); ax.set_title(f"{scores} scores"); ax.tick_params(axis="x", rotation=90)
    ax.set_xlabel("n / prevalence / AUC")
axes[0].set_ylabel("coverage of a 95% interval"); axes[0].legend(); plt.tight_layout(); plt.show()
"""

EXP2 = """t = time.time(); e2 = ex.summarise_paired(ex.paired_difference()); print(round(time.time() - t), "s"); e2
"""

EXP3 = """e3 = ex.detectable_difference(); e3.pivot(index="rho", columns="n", values="detectable")
"""

SAMPLE = """import glob
from aucci.intervals import SINGLE, PAIRED
hits = glob.glob("/kaggle/input/**/paired_scores.csv", recursive=True)
if hits:
    base = os.path.dirname(hits[0])
    df = pd.read_csv(hits[0]); truth = pd.read_csv(os.path.join(base, "paired_truth.csv"))
    y, S = df.y.to_numpy(), df[["model_a", "model_b"]].to_numpy()
    print(truth.to_string(index=False))
    for name, fn in SINGLE.items():
        print(f"{name:<16} model_a {np.round(fn(y, S[:, 0]), 5)}   model_b {np.round(fn(y, S[:, 1]), 5)}")
    print(f"true difference {truth.true_auc[0] - truth.true_auc[1]:.5f}")
    for name, fn in PAIRED.items():
        print(f"{name:<16} {np.round(fn(y, S), 5)}")
else:
    print("dataset not attached")
"""


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", (OUT / "notebook_intro.md").read_text(encoding="utf-8")),
        ("code", "SOURCES = " + json.dumps(_sources(), indent=0) + SETUP),
        ("code", IMPORTS),
        ("markdown", "## 1. One AUC: do the intervals cover?\n\n600 replicates per setting (150 for the bootstrap). "
                     "`coverage` should be 0.95. `miss_low` is how often the interval sits entirely above the true "
                     "AUC, `miss_high` how often it sits below it, and `positives` is the mean number of positive "
                     "rows. Scores are either continuous or rounded to 5 levels (heavy ties)."),
        ("code", EXP1),
        ("code", PLOT1),
        ("markdown", "## 2. Two models on the same rows\n\n60,000 rows (about the size of a Playground public "
                     "split), AUC 0.96 against 0.96 minus `delta`, errors correlated at `rho`. `detect` is how "
                     "often the interval for the difference excludes zero on the right side (a correct win), "
                     "`wrong_sign` how often it excludes zero on the wrong side."),
        ("code", EXP2),
        ("markdown", "## 3. The smallest difference you can see\n\nThe gain a paired DeLong test detects 80% of the "
                     "time at the 5% level, by number of rows and by how correlated the two models' errors are. "
                     "`auc_corr` is the resulting correlation of the two AUC estimates; `se_single` is the standard "
                     "error of one AUC, for comparison."),
        ("code", EXP3),
        ("markdown", "## 4. The sample in the dataset\n\nThe companion dataset holds one paired sample: 60,000 rows, "
                     "true AUCs 0.96 and 0.9597, errors correlated at 0.97. Here are the intervals for each AUC "
                     "and for the difference."),
        ("code", SAMPLE),
        ("markdown", (OUT / "notebook_findings.md").read_text(encoding="utf-8")),
    ]


def build_notebook(public: bool) -> Path:
    out = OUT / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for i, (kind, src) in enumerate(_cells()):
        c = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "auc-intervals.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": NOTEBOOK_ID, "title": TITLE, "code_file": "auc-intervals.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": not public, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [DATASET_ID], "competition_sources": [], "kernel_sources": [],
        "keywords": ["evaluation", "benchmark"],
    }, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    public = "--public" in sys.argv
    if "--notebook-only" not in sys.argv:
        print("wrote", build_dataset(public, rerun="--reuse-results" not in sys.argv))
    print("wrote", build_notebook(public))
