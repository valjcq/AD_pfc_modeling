# Fitting Roadmap: AD Data → Circuit Model

## Overview

The `optimize` command fits the circuit model to target firing rates per cell type. This roadmap documents how to go from raw calcium imaging data to `optimize` arguments, and how many fits to run.

Model output rates are in **Hz** (spikes/s), confirmed by:
```
PYR  3.013 Hz   SOM  3.225 Hz   PV   1.423 Hz   VIP  2.476 Hz
```
(from `best_param_optim.json` + `run` command)

---

## Step 1 — Data Processing: Fluorescence → Firing Rate

Each `Results{N}.csv` file is a **5000-frame fluorescence time series** at ~30 fps (t ≈ 164.9 s):
- Rows = time points (5000 frames)
- Columns = individual neurons (ROIs): `Mean1` … `MeanN`
- Values = raw fluorescence (F, arbitrary units, range ~200–2000)

### Computation pipeline per mouse

```
For each Results file:
  rate = mean(all fluorescence values in file) / t_recording

  where t_recording is read from parameters.rtf or parameters.txt
  (regex: t\s*=\s*([\d.]+), fallback 164.897 s)

For each genotype / timepoint:
  mean_rate = mean(rate) over all files, all mice
```

> **Why mean(F)/t works:** Raw fluorescence is proportional to cumulative calcium
> activity. Dividing the whole-file mean by the recording duration gives a
> time-averaged proxy for transient rate × amplitude that is fast to compute,
> numerically stable, and consistent across conditions. Validated against article
> 1mo medians: SST matches at 1.05×; PYR/PV/VIP offsets (0.77–1.67×) are explained
> by mean vs. median reporting and per-cell-type differences in baseline brightness.

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

The APP genotypes (`WT_APP`, `a7KO_APP`, …) model Alzheimer's disease via nAChR
desensitization. They can be used in three ways (see Step 6): as a validation target
for Strategy A, as a separate circuit fit (Strategy B), or for desensitization-level
fitting (Strategy C).

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

## Step 4 — Computed Target Rates

From `scripts/compute_target_rates.py` using formula `mean(F) / t_recording`.
Full per-mouse detail in `AD_data/summary/`. Validation against article (1mo):

| Cell type | Computed mean | Computed median | Article median | Ratio (mean) | Ratio (median) |
|---|:---:|:---:|:---:|:---:|:---:|
| PYR (WT) | 4.143 | 3.785 | 2.487 | 1.67× | 1.52× |
| PV       | 2.079 | 1.602 | 1.414 | 1.47× | 1.13× |
| SST      | 3.423 | 2.976 | 3.248 | 1.05× | **0.92×** |
| VIP      | 1.933 | 1.857 | 2.517 | 0.77× | 0.74× |

Using computed medians instead of means does not close the gap: PV improves (1.47→1.13×)
but PYR remains 1.52× off and VIP stays at 0.74×. The mean vs. median hypothesis only
partially explains the offset — per-cell-type differences in baseline fluorescence brightness
(affecting `mean(F)`) account for the rest. The relative ordering across cell types is
preserved in both cases.

### All fit targets

| Variable | Genotype | Timepoint | mean | median | n files |
|---|---|:---:|:---:|:---:|:---:|
| `WT_PYR_1mo` | WT | 1mo | 4.143 | 3.785 | 37 |
| `PV_ctrl_1mo` | PV_control | 1mo | 2.079 | 1.602 | 23 |
| `SST_ctrl_1mo` | SST_control | 1mo | 3.423 | 2.976 | 26 |
| `VIP_ctrl_1mo` | VIP_control | 1mo | 1.933 | 1.857 | 27 |
| `a7KO_PYR_1mo` | a7KO_control | 1mo | 3.513 | 3.387 | 21 |
| `b2KO_PYR_1mo` | b2KO_control | 1mo | 4.800 | 4.852 | 33 |
| `a5KO_PYR_1mo` | a5KO_control | 1mo | 3.790 | 3.722 | 44 |
| `WT_PYR_3mo` | WT | 3mo | 4.719 | 4.396 | 21 |
| `a7KO_PYR_3mo` | a7KO_control | 3mo | 2.964 | 2.772 | 18 |
| `b2KO_PYR_3mo` | b2KO_control | 3mo | 3.742 | 3.882 | 23 |
| `WT_PYR_6mo` | WT | 6mo | 4.024 | 3.672 | 11 |
| `a7KO_PYR_6mo` | a7KO_control | 6mo | 4.038 | 4.048 | 12 |

## Step 5 — CLI Commands (with actual values)

### Fit 1 — 1-month post-injection (full constraint)

```bash
python -m circuit_model optimize \
  --target_pyr   4.143 \
  --target_som   3.423 \
  --target_pv    2.079 \
  --target_vip   1.933 \
  --target_alpha7_ko_pyr 3.513 \
  --target_beta2_ko_pyr  4.800 \
  --target_alpha5_ko_pyr 3.790 \
  --optimizer chaining \
  --n_samples 50000 \
  --n_workers 4 \
  --save_best_json figs/optim/1mo/best_params.json \
  --log_file figs/optim/1mo/log.jsonl
```

### Fit 2 — 3-month post-injection (PYR + 2 KOs, freeze interneurons)

```bash
python -m circuit_model optimize \
  --target_pyr   4.719 \
  --target_som   3.423 \
  --target_pv    2.079 \
  --target_vip   1.933 \
  --target_alpha7_ko_pyr 2.964 \
  --target_beta2_ko_pyr  3.742 \
  --freeze Theta_pv,Theta_som,Theta_vip,alpha_pv,alpha_som,alpha_vip \
  --params_json figs/optim/1mo/best_params.json \
  --n_samples 5000 \
  --n_workers 4 \
  --save_best_json figs/optim/3mo/best_params.json \
  --log_file figs/optim/3mo/log.jsonl
```

### Fit 3 — 6-month post-injection (PYR + 1 KO only)

```bash
python -m circuit_model optimize \
  --target_pyr   4.024 \
  --target_som   3.423 \
  --target_pv    2.079 \
  --target_vip   1.933 \
  --target_alpha7_ko_pyr 4.038 \
  --freeze Theta_pv,Theta_som,Theta_vip,alpha_pv,alpha_som,alpha_vip,I_beta2_som,I_alpha5_vip \
  --params_json figs/optim/1mo/best_params.json \
  --n_samples 5000 \
  --n_workers 4 \
  --save_best_json figs/optim/6mo/best_params.json \
  --log_file figs/optim/6mo/log.jsonl
```

---

## Step 6 — APP Conditions (Alzheimer Disease Model)

### What APP is

APP (amyloid precursor protein) transgenic mice model early Alzheimer's disease via
progressive amyloid-β accumulation, which **partially desensitizes nAChRs**. In the
circuit model (`circuit_model/study.py`), this is captured by reduced activation
multipliers drawn from distributions (stochastic per simulation):

| Receptor | Baseline `act_*` | APP model `act_*` | Desensitization |
|---|:---:|:---:|:---:|
| α7 (`act_alpha7`) | 1.0 | 0.10 ± 0.03 | ~90% |
| α5 (`act_alpha5`) | 1.0 | 0.60 ± 0.05 | ~40% |
| β2 (`act_beta2`)  | 1.0 | 0.875 ± 0.06 | ~12% |

Because these are **distributions** (not fixed values), APP conditions cannot be
simulated with a single parameter set — they are always run stochastically via `study`.

### Computed APP rates (mean(F)/t formula)

| Genotype | 1mo | 3mo | 6mo |
|---|:---:|:---:|:---:|
| WT_APP           | 4.566 (n=33) | 4.874 (n=13) | 4.013 (n=18) |
| PV_APP           | 3.186 (n=34) | —            | —            |
| SST_APP          | 3.563 (n=50) | —            | —            |
| VIP_APP          | 1.627 (n=39) | —            | —            |
| a7KO_APP         | 4.270 (n=38) | 5.013 (n=15) | 4.359 (n=11) |
| b2KO_APP         | 4.123 (n=28) | 2.762 (n=24) | —            |
| a5KO_APP         | 3.491 (n=12) | —            | —            |
| WT_APP_reexp     | 3.954 (n=18) | —            | —            |
| a7KO_APP_reexp   | 3.140 (n=20) | —            | —            |
| b2KO_APP_reexp   | 3.545 (n=17) | 3.470 (n=11) | —            |
| a7b2KO_APP       | 3.138 (n=18) | —            | —            |

**reexp** = re-expression of the knocked-out receptor in the APP background (rescue experiment: restores the receptor to test whether it normalises activity).
**a7b2KO_APP** = double knockout (α7 + β2) in APP background.

### Re-expression rescue effect (1mo PYR rates)

| Genotype | Control | APP | APP + reexp | APP→reexp | reexp vs control |
|---|:---:|:---:|:---:|:---:|:---:|
| WT  | 4.143 | 4.566 | 3.954 | −13% | −5% |
| a7KO | 3.513 | 4.270 | 3.140 | −26% | −11% |
| b2KO | 4.799 | 4.123 | 3.545 | −14% | −26% |

Re-expression partially rescues the APP-induced rate increase in all three genotypes.
The α7 re-expression shows the strongest rescue (−26%), consistent with α7 being the
most desensitized receptor in APP (90% → reexp restores it).

### Control vs APP at 1mo (disease effect)

| Cell type | Control | APP | Change |
|---|:---:|:---:|:---:|
| PYR (WT)  | 4.143 | 4.566 | +10% |
| PV        | 2.079 | 3.186 | **+53%** |
| SOM       | 3.423 | 3.563 | +4%  |
| VIP       | 1.933 | 1.627 | −16% |

Large PV increase and VIP decrease are the main signatures of α7 desensitization:
reduced cholinergic drive shifts the interneuron balance.

### Fitting strategies

**Strategy A — Model validation (no new fit, recommended first)**

After Fit 1, run `study` with the fitted params and built-in APP distributions.
Compare predicted APP rates to the measured values above. This is a zero-parameter
prediction that directly tests whether the desensitization distributions in `study.py`
are correctly calibrated to the data.

**Strategy B — Separate APP parameter set**

Fit the full circuit independently on APP mice, treating APP as a different
biological state that may involve secondary synaptic remodeling beyond receptor
desensitization. Uses the same `optimize` CLI as control fits:

```bash
python -m circuit_model optimize \
  --target_pyr   4.566 \
  --target_som   3.563 \
  --target_pv    3.186 \
  --target_vip   1.627 \
  --target_alpha7_ko_pyr 4.270 \
  --target_beta2_ko_pyr  4.123 \
  --target_alpha5_ko_pyr 3.491 \
  --n_samples 5000 --n_workers 4 \
  --save_best_json figs/optim/1mo_APP/best_params.json \
  --log_file figs/optim/1mo_APP/log.jsonl
```

**Strategy C — Fit desensitization levels** *(most biologically principled)*

Keep the control circuit params fixed (from Fit 1); only optimize `act_alpha7`,
`act_alpha5`, `act_beta2` to match APP PYR rates. This directly estimates in-vivo
receptor desensitization without confounding synaptic remodeling. Requires either
adding `--target_wt_app_pyr` support to the CLI, or manually freezing all non-
activation parameters via `--freeze`.

> **Recommended order**: Strategy A first (zero cost). If predictions are off,
> decide between B (full refit) and C (desensitization-only) depending on whether
> you believe APP causes secondary circuit remodeling.

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
  --target_pyr 4.143 --target_som 3.423 --target_pv 2.079 --target_vip 1.933 \
  --target_alpha7_ko_pyr 3.513 --target_beta2_ko_pyr 4.800 --target_alpha5_ko_pyr 3.790 \
  --optimizer chaining \
  --n_samples 50000 --n_workers 4 \
  --save_best_json figs/optim/1mo/best_params.json \
  --log_file figs/optim/1mo/log.jsonl

# Pass 2 — local refinement with CMA-ES from best found so far
python -m circuit_model optimize \
  --target_pyr 4.143 --target_som 3.423 --target_pv 2.079 --target_vip 1.933 \
  --target_alpha7_ko_pyr 3.513 --target_beta2_ko_pyr 4.800 --target_alpha5_ko_pyr 3.790 \
  --optimizer cma \
  --params_json figs/optim/1mo/best_params.json \
  --n_samples 20000 --n_workers 4 \
  --save_best_json figs/optim/1mo/best_params.json \
  --log_file figs/optim/1mo/log.jsonl --resume
```

**Notes on weight bounds:** All synaptic weights are constrained to `[0.1, 5×default]`
in log space, preventing the optimizer from silencing any connection (a degenerate
solution the circuit model is otherwise prone to).

---

## What to do before running fits

1. The `near_zero_threshold = 0.5` in `circuit_model/loss.py` penalises any population
   firing below 0.5 Hz (all targets are in the 2–5 range, so this only catches degenerate states)
2. All synaptic weights have `lo = 0.1` — no connection can be silenced
3. Run Fit 1, inspect result, use its params as `--params_json` starting point for Fits 2 & 3
