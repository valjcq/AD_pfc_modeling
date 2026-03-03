# Statistical Tests Used in This Project

This document lists **all statistical tests currently performed in the codebase** (as of March 2026), with where they are used and how to interpret them.

## Scope

Only tests that compute a test statistic and p-value are included here. In this repository, those are currently used by the ring-attractor asymmetry analysis pipeline.

Main code locations:

- `circuit_model/ring/cli.py` (asymmetry statistical report)
- `circuit_model/ring/plotting.py` (correlation annotation on scatter plots)

---

## 1) One-sample t-test

- **Function**: `scipy.stats.ttest_1samp`
- **Where**: `ring-asymmetry` command in `circuit_model/ring/cli.py`
- **Data**: per-condition asymmetry values (pre-cue and delay), one value per trial
- **Null hypothesis**: the condition mean asymmetry is 0
- **Alternative**: two-sided (mean asymmetry different from 0)
- **Design type**: one-sample (not between-condition)
- **Why used**: tests whether a condition has systematic directional bias (left/right) relative to zero asymmetry

Interpretation:

- Significant p-value means the average asymmetry in that condition is non-zero.
- It does **not** compare two conditions to each other.

---

## 2) Wilcoxon signed-rank test (one-sample vs zero)

- **Function**: `scipy.stats.wilcoxon(vals, alternative='two-sided')`
- **Where**: `ring-asymmetry` command in `circuit_model/ring/cli.py`
- **Data**: same per-condition asymmetry values as above
- **Applied when**: `n >= 10` in current implementation
- **Null hypothesis**: median asymmetry is 0 (symmetry around zero)
- **Alternative**: two-sided
- **Design type**: one-sample signed test (not between-condition)
- **Why used**: robust non-parametric counterpart to one-sample t-test when normality may be questionable

Interpretation:

- Significant p-value means the distribution is shifted away from zero.
- As above, this is **not** a WT vs WT_APP type comparison.

---

## 3) Mann-Whitney U test (pairwise between conditions)

- **Function**: `scipy.stats.mannwhitneyu(abs_a, abs_b, alternative='two-sided')`
- **Where**: `ring-asymmetry` command in `circuit_model/ring/cli.py`
- **Data**: absolute asymmetry magnitudes `|asymmetry|` for each trial in two different conditions
- **Null hypothesis**: both condition distributions are equal (equivalently, no stochastic dominance)
- **Alternative**: two-sided
- **Design type**: unpaired two-sample test
- **Why used**: compares asymmetry **magnitude** across conditions without assuming Gaussian distributions

Important notes:

- This is currently done for all pairwise condition combinations.
- No multiple-comparison correction is applied in the current implementation.
- Using `|asymmetry|` removes sign information (tests magnitude differences, not direction differences).

---

## 4) Pearson correlation test

- **Function**: `scipy.stats.pearsonr(pre, delay)`
- **Where**: asymmetry correlation plot in `circuit_model/ring/plotting.py`
- **Data**: per-trial pairs `(pre_cue_asym, delay_asym)` within each condition
- **Null hypothesis**: correlation coefficient $r = 0$
- **Alternative**: two-sided (SciPy default)
- **Design type**: association test
- **Why used**: quantifies whether trials with pre-cue asymmetry tend to preserve/invert that asymmetry at delay

Implementation note:

- This test is used for plot annotation (`r` and significance stars), not for the text report generated in `asymmetry_stats.txt`.

---

## Significance Labels Used

Across these analyses, p-values are converted to stars with:

- `*` for $p < 0.05$
- `**` for $p < 0.01$
- `***` for $p < 0.001$
- otherwise `n.s.` (or `ns` in some plot labels)

---

## What Is *Not* Currently Implemented

The current code does **not** implement:

- global 3+ group omnibus tests (e.g., one-way ANOVA or Kruskal-Wallis)
- repeated-measures / blocked-by-seed group tests (e.g., Friedman)
- pairwise multiple-comparison correction (Holm, Bonferroni, FDR)

If you need formal multi-condition inference for groups such as `WT`, `WT_APP`, `A7_KO_APP`, add an omnibus test first and then corrected post-hoc comparisons.