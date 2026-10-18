## What the simulation shows

Coverage of nominal 95% intervals, from the tables above (600 replicates per setting, 150 for the bootstrap, so a
coverage is good to about ±0.01, ±0.02 for the bootstrap).

**One AUC**

- **With hundreds of positives, every method is fine.** At 10,000 rows, DeLong (Wald or logit) covers 0.93-0.98
  everywhere and the bootstrap 0.90-0.96.
- **Hanley-McNeil is too wide at high AUC.** From 1,000 rows on, it covers 0.985-1.00 at AUC 0.98, and its interval is up to 75% wider
  than DeLong's (0.0174 against 0.0099 at 10,000 rows, prevalence 0.05). Its formula assumes exponential
  rather than binormal scores, and with ties (5 levels) it is wider still.
- **Few positives break the others.** With about 10 positives (200 rows at prevalence 0.05), DeLong-Wald covers
  only 0.70-0.89, the bootstrap 0.73-0.94 and DeLong on the logit scale 0.86-0.91. Almost all misses fall on the
  same side: the interval sits above the true AUC. Even with 50 positives at AUC 0.98, Wald covers 0.88 and the
  bootstrap 0.88. Hanley-McNeil looks better there (0.93-0.98) only because it is wide everywhere.
- **The logit DeLong interval is the safest default.** It is never more than 0.005 below the Wald one and is much
  better near AUC 1, at the same cost.

**Two models on the same rows** (60,000 rows, AUC 0.96)

- **Paired DeLong is calibrated.** It covers the true difference 0.94-0.96 of the time, and with equal AUCs it
  declares a winner 4-5% of the time (both signs together), as a 5% test should.
- **The unpaired interval is useless.** Adding the two variances as if the models had been scored on different
  rows gives intervals 2.5-8 times too wide (0.0039 against 0.0015 to 0.0005). It never detected even a true gain
  of 0.001.
- **A true gain of 0.0003 is hard to see on 60k rows.** The paired test detects it 9% of the time when the two
  models' errors correlate at 0.9, 28% at 0.97 and 63% at 0.99.

**How big a gain has to be** (80% power, 5% level, AUC 0.96)

| error correlation | AUC estimates' correlation | 60k rows | 700k rows |
| --- | --- | --- | --- |
| 0.90 | 0.85 | 0.0011 | 0.00032 |
| 0.97 | 0.95 | 0.00061 | 0.00018 |
| 0.99 | 0.98 | 0.00036 | 0.00010 |
| 0.995 | 0.99 | 0.00025 | 0.00007 |

What to do with it:

- Compare two models with a paired interval or test (paired DeLong or a paired bootstrap), never by checking
  whether two single-model intervals overlap.
- Report AUC intervals on the logit scale, and do not trust any of them with only a few dozen positives.
- On a Playground public split (about 60k rows), a gain below about 0.0003 between two closely related models
  is mostly noise. The 700k OOF rows resolve gains 3-4 times smaller.
