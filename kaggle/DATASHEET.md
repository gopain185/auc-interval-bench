# AUC Confidence Interval Simulations

Simulated classifier scores whose **population ROC AUC is known exactly**, and the coverage of the usual
confidence intervals for one AUC (DeLong, DeLong on the logit scale, Hanley-McNeil, stratified bootstrap) and
for the difference of two AUCs measured on the same rows (paired DeLong, paired bootstrap, and the unpaired
mistake). Use it to check an interval or a significance test against ground truth, or to see how large an AUC
gain has to be before a leaderboard of a given size can resolve it.

Companion notebook: [Is That AUC Gain Real? Confidence Intervals](https://www.kaggle.com/code/locdunhat/is-that-auc-gain-real-confidence-intervals)

## How the data is generated

Binormal scores: a negative row scores `e`, a positive row `d + e`, with `e ~ N(0, 1)`, so the population AUC is
`Phi(d / sqrt(2))`. Two models scored on the same rows share the labels and have errors with correlation `rho`.
For tied scores, the continuous scores are rounded to a few bins at equally spaced quantiles of the pooled score
distribution, and the population AUC (ties counting one half) is computed exactly from the bin probabilities.
Every sample keeps at least two rows of each class.

## Files

| file | rows | content |
| --- | --- | --- |
| `paired_scores.csv` | 60,000 | `row_id`, `y`, `model_a`, `model_b`: one paired sample, prevalence 0.43, errors correlated at 0.97 |
| `paired_truth.csv` | 2 | `model`, `signal` (`d`), `true_auc` (0.96 and 0.9597), `error_correlation` |
| `exp1_single_coverage.csv` | 70,200 | one interval per setting, replicate and method for a single AUC |
| `exp2_paired_difference.csv` | 7,380 | one interval per setting, replicate and method for the difference of two AUCs on 60,000 rows |
| `exp3_detectable_difference.csv` | 16 | the AUC gain a paired DeLong test detects with 80% power, by rows and error correlation |

Columns of the experiment tables:

- Settings: `n` (rows), `prevalence`, `auc` (true AUC of the first model), `scores` (`continuous` or `5 levels`),
  `delta` (true AUC gap), `rho` (error correlation), `rep` (replicate), `positives` (positive rows in the sample).
- Each interval: `method`, `truth` (the population AUC or AUC difference), `low`, `high`.
- `exp3`: `auc_corr` (correlation of the two AUC estimates), `se_single` (standard error of one AUC), `se_diff`
  (standard error of the difference) and `detectable` (`(z_0.975 + z_0.8) * se_diff`).

## Provenance

Everything is simulated by the `aucci` package (repository `auc-interval-bench` by Loc Nguyen, MIT), seed
`20261019`. The notebook embeds the full source and regenerates every table. No real people, competitions or
submissions are involved.

## Intended use

Checking confidence intervals and significance tests for ROC AUC against a known truth; teaching why two models
compared on the same rows must be compared with a paired method; sizing how many evaluation rows are needed to
resolve a given AUC gain.

## Limitations

- Scores are Gaussian given the label (the binormal model). Real scores can be skewed or have heavier tails,
  which mostly matters for the Hanley-McNeil formula, since it assumes a different (exponential) shape.
- Rows are independent. Grouped or time-ordered evaluation data needs a grouped bootstrap or a cluster-aware
  variance.
- One metric (ROC AUC) and two-sided 95% intervals only.

## License

CC0 1.0.
