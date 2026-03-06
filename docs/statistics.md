# Statistical Tests Used in This Project

This document lists statistical tests currently used by active experiments.

## Scope

Current test usage is concentrated in:

- `ring-asymmetry`
- `ring-burnin-stability`

Main code location:

- `circuit_model/ring/cli.py`
- `circuit_model/ring/plotting.py` (for asymmetry plot annotations)

## 1) One-sample t-test

- Function: `scipy.stats.ttest_1samp`
- Where: `ring-asymmetry`
- Data: per-condition asymmetry values (pre-cue and delay)
- Null: condition mean asymmetry is 0
- Alternative: two-sided

Interpretation:

- Significant p-value implies a systematic directional bias relative to zero.

## 2) Wilcoxon signed-rank test (one-sample vs zero)

- Function: `scipy.stats.wilcoxon`
- Where: `ring-asymmetry`
- Data: same asymmetry values as above
- Null: median asymmetry is 0
- Alternative: two-sided

Interpretation:

- Significant p-value implies the distribution is shifted away from zero.

## 3) Mann-Whitney U test (pairwise between conditions)

- Function: `scipy.stats.mannwhitneyu`
- Where: `ring-asymmetry`
- Data: trial-wise `|asymmetry|` values across condition pairs
- Null: the two distributions are equal
- Alternative: two-sided

Notes:

- This tests asymmetry magnitude differences, not direction.
- Multiple-comparison correction is not currently applied.

## 4) Pearson Correlation Test

- Function: `scipy.stats.pearsonr`
- Where: `ring-asymmetry` correlation plot annotation
- Data: trial pairs `(pre_cue_asym, delay_asym)` per condition
- Null: correlation coefficient is 0

Interpretation:

- Significant result indicates linear association between pre-cue and delay asymmetry.

## 5) Kruskal-Wallis Test

- Function: `scipy.stats.kruskal`
- Where: `ring-burnin-stability`
- Data: per-window distributions of amplitude and `|asymmetry|`
- Null: all burn-in windows come from the same distribution

Interpretation:

- Non-significant p-value supports stationarity across burn-in windows.

## 6) Adjacent-Window Mann-Whitney U

- Function: `scipy.stats.mannwhitneyu`
- Where: `ring-burnin-stability`
- Data: adjacent burn-in windows for amplitude and `|asymmetry|`
- Null: adjacent windows have equal distributions

Interpretation:

- Used as a local follow-up check to identify which transitions differ.

## Significance Labels

- `*`: `p < 0.05`
- `**`: `p < 0.01`
- `***`: `p < 0.001`
- Otherwise: `n.s.`
