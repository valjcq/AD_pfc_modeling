# Fitting Roadmap: AD Data → Circuit Model

## Overview

The `optimize` command fits the circuit model to target firing rates per cell type. This roadmap documents how to use in vivo spike rate data from `AD_data/AD_spikes/` to set `optimize` arguments, and how many fits to run.

Model output rates are in **Hz** (spikes/s), confirmed by:
```
PYR  3.013 Hz   SOM  3.225 Hz   PV   1.423 Hz   VIP  2.476 Hz
```
(from `best_param_optim.json` + `run` command)

---

## Step 1 — Data Source: Direct Spike Rates

Firing rates come from **`AD_data/AD_spikes/datafiles/firing_rate_data.csv`** — a cleaned summary table with one row per genotype × timepoint (30 rows). These are direct spike rates in **Hz** extracted from the raw two-photon recordings (see `AD_data/AD_spikes/README.md` for the full processing pipeline).

### Columns used for fitting

| Column | Description |
|---|---|
| `per_neuron_mean` | Mean firing rate across all neurons pooled from all mice (Hz) — **use as optimization target** |
| `per_neuron_n` | Number of neurons in the pool |
| `sampled_mean` | Mean of 5000 bootstrap group-means (groups of ~10 neurons) — used as fallback when `per_neuron_*` is unavailable |

**Target selection rule:** use `per_neuron_mean`; fall back to `sampled_mean` for `b2_KO_control` at 1mo and `b2_KO_APP` at 3mo, whose per-neuron arrays were truncated in the source CSV (see README for truncation details).

### How rates were derived

Each `per_neuron_mean` is the arithmetic mean of all individual neuron firing rates pooled across mice and recording sessions for that genotype × timepoint. Rates are in Hz (spikes/s), as confirmed by the original analysis scripts and figure comparisons (see `firing_rate_figures/significance_tests.txt`).

---

## Step 2 — Genotype → Model Condition Mapping

| AD_data folder | Cell type recorded | Model condition / argument |
|---|---|---|
| `WT` | PYR (pyramidal, unlabelled) | `--target_pyr` (WT baseline) |
| `PV_control` | PV interneurons (Cre+) | `--target_pv` |
| `SST_control` | SOM interneurons (Cre+) | `--target_som` |
| `VIP_control` | VIP interneurons (Cre+) | `--target_vip` |
| `a7KO_control` | PYR in α7-KO background | `--target_alpha7_ko_pyr` |
| `b2KO_control` | PYR in β2-KO background | `--target_beta2_ko_pyr` |
| `a5KO_control` | PYR in α5-KO background | `--target_alpha5_ko_pyr` |

The APP genotypes (`WT_APP`, `a7KO_APP`, …) represent Alzheimer's disease background
data. In the current workflow, APP is handled by a separate fitted parameter family
(WT_APP), and KO_APP is obtained by applying the same KO rule on that family.

---

## Step 3 — Data Availability per Timepoint

| Genotype | 1mo | 3mo | 6mo |
|---|:---:|:---:|:---:|
| WT | ✓ (6 mice) | ✓ (4 mice) | ✓ (3 mice) |
| PV_control | ✓ (5 mice) | — | — |
| SST_control | ✓ (5 mice) | — | — |
| VIP_control | ✓ (4 mice) | — | — |
| a7KO_control | ✓ (4 mice) | ✓ (4 mice) | ✓ (3 mice) |
| b2KO_control | ✓ (4 mice) | ✓ (3 mice) | — |
| a5KO_control | ✓ (6 mice) | — | — |

→ **3 fits total**, one per timepoint, with decreasing constraint:

| Fit | Timepoint | Free targets | KO constraints |
|---|---|---|---|
| **Fit 1** | 1mo | PYR + PV + SOM + VIP | α7-KO + β2-KO + α5-KO |
| **Fit 2** | 3mo | PYR only (freeze PV/SOM/VIP from Fit 1) | α7-KO + β2-KO |
| **Fit 3** | 6mo | PYR only (freeze from Fit 1) | α7-KO only |

---

## Step 4 — Target Rates

From `AD_data/AD_spikes/datafiles/firing_rate_data.csv`, column `per_neuron_mean` (mean firing rate across all pooled neurons). Fallback to `sampled_mean` where per-neuron data is truncated (†).

### All fit targets

| Variable | Genotype (CSV label) | Timepoint | mean (Hz) | n neurons |
|---|---|:---:|:---:|:---:|
| `WT_PYR_1mo` | WT | 1mo | 8.214 | 563 |
| `PV_ctrl_1mo` | PV-Cre_control | 1mo | 4.073 | 207 |
| `SST_ctrl_1mo` | SST-Cre | 1mo | 4.295 | 249 |
| `VIP_ctrl_1mo` | VIP-Cre_control | 1mo | 6.051 | 291 |
| `a7KO_PYR_1mo` | a7_KO_control | 1mo | 17.539 | 409 |
| `b2KO_PYR_1mo` | b2_KO_control | 1mo | 17.965† | — |
| `a5KO_PYR_1mo` | a5_KO | 1mo | 9.285 | 925 |
| `WT_PYR_3mo` | WT_ctr | 3mo | 13.325 | 411 |
| `a7KO_PYR_3mo` | a7_KO | 3mo | 14.105 | 501 |
| `b2KO_PYR_3mo` | b2_KO | 3mo | 16.529 | 775 |
| `WT_PYR_6mo` | WT_control | 6mo | 10.745 | 501 |
| `a7KO_PYR_6mo` | a7_KO_control | 6mo | 16.221 | 451 |

† `b2_KO_control` at 1mo: per-neuron array truncated in source CSV; value is `sampled_mean`.

## Step 5 — CLI Commands (with actual values)

### Fit 1 — 1-month post-injection (full constraint)

```bash
python -m circuit_model optimize \
  --target_pyr   8.214 \
  --target_som   4.295 \
  --target_pv    4.073 \
  --target_vip   6.051 \
  --target_alpha7_ko_pyr 17.539 \
  --target_beta2_ko_pyr  17.965 \
  --target_alpha5_ko_pyr 9.285 \
  --optimizer chaining \
  --n_samples 50000 \
  --save_best_json params/new/WT_1mo.json \
  --log_file figs/optim/1mo/log.jsonl
```

### Fit 2 — 3-month post-injection (PYR + 2 KOs, freeze interneurons)

```bash
python -m circuit_model optimize \
  --target_pyr   13.325 \
  --target_som   4.295 \
  --target_pv    4.073 \
  --target_vip   6.051 \
  --target_alpha7_ko_pyr 14.105 \
  --target_beta2_ko_pyr  16.529 \
  --freeze Theta_pv,Theta_som,Theta_vip,alpha_pv,alpha_som,alpha_vip \
  --params_json params/new/WT_1mo.json \
  --n_samples 5000 \
  --save_best_json params/new/WT_3mo.json \
  --log_file figs/optim/3mo/log.jsonl
```

### Fit 3 — 6-month post-injection (PYR + 1 KO only)

```bash
python -m circuit_model optimize \
  --target_pyr   10.745 \
  --target_som   4.295 \
  --target_pv    4.073 \
  --target_vip   6.051 \
  --target_alpha7_ko_pyr 16.221 \
  --freeze Theta_pv,Theta_som,Theta_vip,alpha_pv,alpha_som,alpha_vip,I_beta2_som,I_alpha5_vip \
  --params_json params/new/WT_1mo.json \
  --n_samples 5000 \
  --save_best_json params/new/WT_6mo.json \
  --log_file figs/optim/6mo/log.jsonl
```

---

## Step 6 — APP Conditions (Alzheimer Disease Model)

### What APP is

APP (amyloid precursor protein) transgenic mice model early Alzheimer's disease. In
the current circuit workflow, APP is represented by fitting a separate WT_APP local
circuit parameter family (instead of sampling receptor desensitization multipliers).

Condition rules in `circuit_model/study.py` are:
- WT family: `WT`, `a7_KO`, `b2_KO`, `a5_KO`
- WT_APP family: `WT_APP`, `a7_KO_APP`, `b2_KO_APP`, `a5_KO_APP`
- KO effect only: set targeted receptor activation to 0 (`act_alpha7`, `act_beta2`, `act_alpha5`; plus `g_alpha7=0` for α7 KO)

### APP rates from `firing_rate_data.csv` (per_neuron_mean, Hz)

| Genotype (CSV label) | 1mo | 3mo | 6mo |
|---|:---:|:---:|:---:|
| WT-APP                   | 12.466 (n=679) | 16.340 (n=410) | 14.189 (n=882) |
| PV-Cre_APP               | 4.241 (n=333)  | —              | —              |
| SST-Cre_APP              | 4.814 (n=299)  | —              | —              |
| VIP-Cre_APP              | 5.551 (n=434)  | —              | —              |
| a7_KO_APP                | 13.599 (n=507) | 12.272 (n=536) | —              |
| b2_KO_APP                | 19.109 (n=711) | 16.567†        | —              |
| a5_KO_APP                | 3.113 (n=247)  | —              | —              |
| WT-APP_reexp             | 10.949 (n=565) | —              | —              |
| a7_KO_APP_re-expression  | 19.077 (n=354) | —              | —              |
| b2_KO_APP_re-expression  | 17.416 (n=589) | —              | —              |
| a7b2_KO_APP              | 4.940 (n=241)  | —              | —              |

† `b2_KO_APP` at 3mo: per-neuron array truncated; value is `sampled_mean`.

**reexp** = re-expression of the knocked-out receptor in the APP background (rescue experiment: restores the receptor to test whether it normalises activity).
**a7b2KO_APP** = double knockout (α7 + β2) in APP background.

### Re-expression rescue effect (1mo PYR rates)

| Genotype | Control | APP | APP + reexp | APP→reexp | reexp vs control |
|---|:---:|:---:|:---:|:---:|:---:|
| WT   | 8.214 | 12.466 | 10.949 | −12% | +33% |
| a7KO | 17.539 | 13.599 | 19.077 | +40% | +9% |
| b2KO | 17.965† | 19.109 | 17.416 | −9% | −3% |

† b2KO control: `sampled_mean` (per-neuron truncated).

### Control vs APP at 1mo (disease effect)

| Cell type | Control | APP | Change |
|---|:---:|:---:|:---:|
| PYR (WT)  | 8.214  | 12.466 | **+52%** |
| PV        | 4.073  | 4.241  | +4%      |
| SOM       | 4.295  | 4.814  | +12%     |
| VIP       | 6.051  | 5.551  | −8%      |

The dominant APP signature at 1mo is a large PYR rate increase (+52%). Interneuron
changes are modest: PV and SOM slightly elevated, VIP slightly reduced.

### Fitting strategies

**Strategy A — Separate WT and WT_APP parameter families (recommended)**

Fit WT and WT_APP independently on their own 4-population targets (optionally with
KO PYR targets). This is the current default strategy and is consistent with
`circuit_model/study.py`.

Example WT_APP fit:

```bash
python -m circuit_model optimize \
  --target_pyr   12.466 \
  --target_som   4.814 \
  --target_pv    4.241 \
  --target_vip   5.551 \
  --target_alpha7_ko_pyr 13.599 \
  --target_beta2_ko_pyr  19.109 \
  --target_alpha5_ko_pyr 3.113 \
  --n_samples 5000 \
  --save_best_json params/new/WT_APP_1mo_article_ko.json \
  --log_file figs/optim/1mo_APP/log.jsonl
```

**Strategy B — Optional receptor-sensitivity sweep (non-default analysis)**

If needed for mechanistic analysis, sweep `act_alpha7/act_alpha5/act_beta2` around
fitted families to probe sensitivity thresholds. This is an analysis layer, not the
baseline condition definition.

> **Recommended order**: run Strategy A as the baseline. Use Strategy B only as a
> supplementary mechanistic probe.

---

---

## Step 7 — Optimizer Choice

The `--optimizer` flag controls the Nevergrad algorithm used. Choose based on the stage of fitting:

| Flag | Algorithm | When to use |
|---|---|---|
| `de` | TwoPointsDE (default) | First run from scratch — broad global exploration |
| `cma` | CMA-ES | Warm-start from a known good `best_params.json`; fast local convergence; learns parameter correlations |
| `chaining` | TwoPointsDE → CMA-ES | Single-pass best-of-both-worlds: explores globally for first 50% of budget, then refines with CMA-ES |
| `auto` | NGOpt | Nevergrad selects the algorithm automatically based on problem dimension and budget |

**Recommended workflow for Fit 1:**

```bash
# Pass 1 — global exploration with DE
python -m circuit_model optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --target_alpha7_ko_pyr 17.539 --target_beta2_ko_pyr 17.965 --target_alpha5_ko_pyr 9.285 \
  --optimizer chaining \
  --n_samples 50000 \
  --save_best_json params/new/WT_1mo.json \
  --log_file figs/optim/1mo/log.jsonl

# Pass 2 — local refinement with CMA-ES from best found so far
python -m circuit_model optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --target_alpha7_ko_pyr 17.539 --target_beta2_ko_pyr 17.965 --target_alpha5_ko_pyr 9.285 \
  --optimizer cma \
  --params_json params/new/WT_1mo.json \
  --n_samples 20000 \
  --save_best_json params/new/WT_1mo.json \
  --log_file figs/optim/1mo/log.jsonl --resume
```

**Notes on weight bounds:** All synaptic weights are constrained to `[0.1, 5×default]`
in log space, preventing the optimizer from silencing any connection (a degenerate
solution the circuit model is otherwise prone to).

---

## What to do before running fits

1. The `near_zero_threshold = 0.5` in `circuit_model/loss.py` penalises any population
   firing below 0.5 Hz (targets now range from ~4–18 Hz, so this only catches degenerate states)
2. All synaptic weights have `lo = 0.1` — no connection can be silenced
3. Run Fit 1, inspect result, use its params as `--params_json` starting point for Fits 2 & 3
4. **Loss function**: MSPE (squared percentage error) is now the default. It penalises large
   per-population errors quadratically — a 37% error on PYR contributes ~14× more than a 10% error,
   preventing the optimizer from tolerating a large miss on one population while the others are exact.
   Combine with the Jacobian upper-bound penalty (`max_gain=5`) to avoid biologically implausible
   solutions with runaway effective gains. Pass `--no_squared_loss` to revert to MAPE.
