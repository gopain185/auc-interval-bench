# Is That AUC Gain Real? Confidence Intervals

Your new model scores 0.9612 on the public leaderboard and the old one 0.9609. Is the new one better? A
confidence interval should answer that, but which interval? Several are in common use, and they can disagree:

- **DeLong**: the variance of the Mann-Whitney statistic, from the placement of every positive among the
  negatives and of every negative among the positives. Used as a Wald interval on the AUC itself, or on the
  logit scale so that it stays inside (0, 1).
- **Hanley-McNeil (1982)**: a variance formula that needs only the AUC and the two class counts.
- **Bootstrap**: resample rows within each class and take percentiles.

For a difference between two models scored on the same rows, the pairing matters. The two AUC estimates are
strongly correlated, and an interval that ignores this ("unpaired") is far too wide.

The notebook samples binormal scores whose **population AUC is known exactly** (also with ties) and measures:

1. how often each interval for one AUC contains the true value, from 200 to 10,000 rows, at prevalence 0.5 and
   0.05, and AUC 0.75 to 0.98;
2. coverage, false alarms and power for the difference of two correlated models on 60,000 rows;
3. the smallest AUC gain a paired test can detect, from 10k to 700k rows.

Everything runs on CPU in about 45 minutes, and the package source is embedded below.
