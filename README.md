# auc-interval-bench

How often does a 95% confidence interval for ROC AUC contain the true AUC? And when is a gain between two models
real? This repository samples classifier scores whose **population AUC is known exactly** and measures the
coverage of the usual intervals for one AUC and for the difference of two AUCs on the same rows.

- **Kaggle notebook:** [Is That AUC Gain Real? Confidence Intervals](https://www.kaggle.com/code/locdunhat/is-that-auc-gain-real-confidence-intervals)
- **Kaggle dataset:** [AUC Confidence Interval Simulations](https://www.kaggle.com/datasets/locdunhat/auc-confidence-interval-simulations) (CC0)

## The model

Binormal scores: a negative row scores `e`, a positive row `d + e`, with `e ~ N(0, 1)`. The population AUC is
`Phi(d / sqrt(2))`. Two models on the same rows share the labels and have errors correlated at `rho`. Ties are
made by rounding the scores to a few bins, and the population AUC of the rounded scores is computed exactly
from the bin probabilities.

Intervals compared (`aucci/intervals.py`):

| one AUC | difference of two AUCs on the same rows |
| --- | --- |
| DeLong, Wald interval on the AUC | paired DeLong (with the covariance of the two AUCs) |
| DeLong on the logit scale | paired stratified bootstrap |
| Hanley-McNeil (1982) | "unpaired": the two variances added, as if the rows were independent |
| stratified percentile bootstrap | |

`delong` is the fast algorithm of Sun & Xu (2014). It is tested against the direct pairwise placement formula
and against the sampling variance of the AUC.

## Results

**1. One AUC** (600 replicates per setting, 150 for the bootstrap). Coverage of nominal 95% intervals:

| rows | prevalence | AUC | positives | DeLong (Wald) | DeLong (logit) | Hanley-McNeil | bootstrap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10,000 | 0.5 | 0.90 | 5,000 | 0.953 | 0.955 | 0.967 | 0.940 |
| 10,000 | 0.05 | 0.98 | 500 | 0.933 | 0.938 | 0.997 | 0.920 |
| 1,000 | 0.05 | 0.98 | 50 | 0.883 | 0.918 | 0.993 | 0.880 |
| 200 | 0.05 | 0.90 | 10 | 0.792 | 0.863 | 0.942 | 0.787 |
| 200 | 0.05 | 0.98 | 10 | 0.695 | 0.862 | 0.925 | 0.733 |

All 36 settings (3 sizes × 2 prevalences × 3 AUCs × continuous or tied scores) are in `results/` and in the
notebook. In short:

- With hundreds of positives, DeLong and the bootstrap are fine.
- Hanley-McNeil is too wide at high AUC: up to 75% wider than DeLong.
- With a few dozen positives or fewer, everything except Hanley-McNeil undercovers, and the interval sits above
  the truth. The logit DeLong interval holds up best.

**2. Two models on 60,000 rows** (about a Playground public split), AUC 0.96 against 0.96 − `delta`, 400
replicates per setting:

| delta | error correlation | paired DeLong: coverage | detects the gain | unpaired: width / paired width |
| --- | --- | --- | --- | --- |
| 0 | 0.97 | 0.953 | 1.8% (+3.0% wrong sign) | 4.6× |
| 0.0003 | 0.90 | 0.955 | 8.5% | 2.5× |
| 0.0003 | 0.97 | 0.948 | 28% | 4.6× |
| 0.0003 | 0.99 | 0.943 | 63% | 7.9× |
| 0.001 | 0.97 | 0.940 | 99% | 4.6× |

The unpaired interval never detected a gain, even 0.001.

**3. Smallest gain a paired test detects** (80% power, 5% level, AUC 0.96):

| error correlation | 10k rows | 60k rows | 200k rows | 700k rows |
| --- | --- | --- | --- | --- |
| 0.90 | 0.0027 | 0.0011 | 0.00060 | 0.00032 |
| 0.97 | 0.0015 | 0.00061 | 0.00034 | 0.00018 |
| 0.99 | 0.00087 | 0.00036 | 0.00019 | 0.00010 |
| 0.995 | 0.00062 | 0.00025 | 0.00014 | 0.00007 |

**4. A check on real OOF predictions** (`results/s6e10_lb_noise_check.csv`). For 8 pairs of the author's models in
Kaggle Playground S6E10 (700k OOF rows), paired DeLong rescaled to 60k rows predicts the spread of the AUC gap over
200 random 60k-row slices of the OOF within about 10%. Example: RealMLP v6 against v7 has a predicted sd of
0.000082 and a measured sd of 0.000078.

## Usage

```bash
pip install -r requirements.txt
python -m pytest -q                         # 12 tests, about 15 s
python -m aucci results                     # all experiments, about 45 minutes on CPU
python kaggle/build_kaggle.py --reuse-results   # Kaggle dataset and notebook from results/
```

```python
import numpy as np
from aucci.intervals import delong, ci_logit_delong, diff_paired_delong

aucs, cov = delong(y, np.column_stack([pred_a, pred_b]))   # AUCs and their covariance
low, high = ci_logit_delong(y, pred_a)                      # 95% interval for one AUC
low, high = diff_paired_delong(y, np.column_stack([pred_a, pred_b]))  # for AUC(a) - AUC(b)
```

## Layout

```
aucci/sim.py           binormal sampling and the exact population AUC (also with ties)
aucci/intervals.py     AUC, fast DeLong covariance, all intervals
aucci/experiments.py   the three experiments and their summaries
tests/                 checks of the AUC, the DeLong variance, coverage and power
kaggle/                builder, datasheet and text of the Kaggle notebook
results/               raw tables of the last full run
```

## Limitations

- Scores are binormal and rows independent. Skewed scores or grouped rows behave differently.
- Bootstrap results use fewer replicates (150 per setting for one AUC, 60 for the difference), so their coverage
  is only good to about ±0.02 for one AUC and ±0.04 for the difference.

## License

MIT. Author: Loc Nguyen.
