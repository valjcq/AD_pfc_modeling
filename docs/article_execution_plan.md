# Article Execution Plan — Detailed & Actionable

**Working title**: *Interneuron-class-specific control of persistent activity in a prefrontal ring attractor: implications for Alzheimer's disease*

**Purpose of this document**: Exhaustive, agent-ready task list. Each block specifies *what* to run, *what* outputs to expect, *what* metrics to extract, *why* it matters, and *how* to interpret results. Organized by article section.

---

## Current State of the Project (as of 2026-03-19)

### What exists and is valid

| Resource | Status | Location | Notes |
|---|---|---|---|
| Raw calcium imaging data | Available | `AD_data/1mo_post_injection/`, `3mo_post_injection/`, `6mo_post_injection/` | 19 genotypes at 1mo, fewer at 3mo/6mo |
| Target rate computation script | Written | `scripts/compute_target_rates.py` | Formula: `mean(F) / t_recording` |
| Computed target rate JSONs | Generated | `AD_data/summary/targets_{1mo,3mo,6mo}.json` | Validated |
| Data processing documentation | Written | `docs/data/fitting_roadmap.md` | Documents formula, genotype mapping, CLI commands |
| **WT 1mo article params** | **CONVERGED** | `params/new/WT_1mo_article.json` | loss=1e-4; PYR=2.487, SOM=3.248, PV=1.414, VIP=2.517 — targets are article median values; boxplot matches article figure |
| **WT_APP 1mo article params** | **CONVERGED** | `params/new/WT_APP_1mo_article.json` | loss=9.8e-5; PYR=3.279, SOM=3.167, PV=3.166, VIP=2.501 |

> **Model simplification (2026-03-19)**: Two biologically unsupported connections have been removed from the model (see `docs/changes_removed_connections.md`): `w_vv` (VIP→VIP self-inhibition) and `w_ps` (PV→SOM cross-inhibition). Both were absent from the reference circuit schematic. The new article param files were fitted with this simplified model. Old JSON files with `w_vv`/`w_ps` keys are silently ignored at load time.

### Parameter file convention going forward

All ring-level analyses use:
- **WT**: `params/new/WT_1mo_article.json`
- **WT_APP**: `params/new/WT_APP_1mo_article.json`

KO conditions (a7_KO, b2_KO, a5_KO) are simulated either via zero-activation on the WT params (Option A) or via dedicated fits (Option B — not yet done). See Section 2.5 of `roadmap_article.md` for strategy discussion.

> **Note on targets**: The article params were fitted to the published article's median values (PYR=2.487 Hz, SOM=3.248 Hz, PV=1.414 Hz, VIP=2.517 Hz) rather than the `mean(F)/t` computed rates. This was motivated by the systematic offsets between the formula's output and the reported medians, and by the desire to match the paper's reference point. The boxplot of simulated vs. experimental rates matches the article figure qualitatively, which is a positive validation.

### What is missing (ordered by priority)

1. **Ring calibration** — Must be run with the new article params for WT and WT_APP
2. **All oscillation, asymmetry, diffusion, distractor analyses** — Blocked by calibration
3. **KO conditions** — Simulated via zero-activation on WT params (Option A); no separate fit needed or planned
4. **Parametric desensitization sweep**
5. **3mo/6mo fits**

---

## PHASE -1 — Data Validation (BLOCKING — must complete first)

> **Goal**: Confirm that the target firing rates derived from calcium imaging are correct. The user has expressed uncertainty about the data processing pipeline. If the targets are wrong, all optimization is wasted.

### Task -1.1 — Validate data organization and file structure

**What**: Manually inspect the AD_data directory structure to confirm genotype-to-folder mapping is correct.

**Current structure** (1mo_post_injection):
```
AD_data/1mo_post_injection/
├── WT/              (PYR — unlabelled pyramidal cells)
├── PV_control/      (PV interneurons, Cre+ line)
├── SST_control/     (SOM interneurons, Cre+ line)
├── VIP_control/     (VIP interneurons, Cre+ line)
├── a7KO_control/    (PYR in α7-KO background)
├── b2KO_control/    (PYR in β2-KO background)
├── a5KO_control/    (PYR in α5-KO background)
├── WT_APP/          (PYR in APP transgenic)
├── PV_APP/          (PV in APP transgenic)
├── SST_APP/         (SOM in APP transgenic)
├── VIP_APP/         (VIP in APP transgenic)
├── a7KO_APP/        (PYR in α7-KO × APP)
├── b2KO_APP/        (PYR in β2-KO × APP)
├── a5KO_APP/        (PYR in α5-KO × APP)
├── WT_APP_reexp/    (PYR in APP + receptor re-expression rescue)
├── a7KO_APP_reexp/  (PYR in α7-KO × APP + re-expression)
├── b2KO_APP_reexp/  (PYR in β2-KO × APP + re-expression)
├── a7b2KO_control/  (PYR in α7+β2 double KO)
└── a7b2KO_APP/      (PYR in α7+β2 double KO × APP)
```

Each genotype folder contains `mouse_N/` subdirectories, each with `Results1.csv` ... `ResultsK.csv` + `parameters.rtf` or `parameters.txt`.

**Checks to perform**:
1. Open 2-3 sample `ResultsN.csv` files and verify: rows = timepoints (~5000), columns = individual neurons (`Mean1`...`MeanN`), values = raw fluorescence (range ~200-2000) or something else ? maybe already processed, like spiking during that time bin ? Need to double check with Koukouli.
2. Open 2-3 `parameters.rtf`/`.txt` files and verify: recording duration `t` is parsed correctly (expect ~164.9s for most, some may differ)
3. Cross-check mouse count per genotype against `targets_1mo.json`: e.g., WT should have 5 mice (mouse_1..6 minus one missing), PV_control should have 5, etc.

**DECISION POINT**: Does the current formula `mean(F) / t_recording` make biological sense?
- This gives a rate-like quantity in "fluorescence units per second" — NOT actual spikes/s or maybe it does ? we still need to confirm with Koukouli what the raw values represent and whether this formula is justified
- It's a *proxy* for firing rate that preserves relative ordering across conditions
- The fitting_roadmap.md shows validation against published article data: SST matches well (1.05×), PV is 1.47×, PYR is 1.67×, VIP is 0.77×
- **Context**: Raw fluorescence values were tried first but were off-scale for model fitting. The `mean(F) / t_recording` formula was chosen empirically because it brings the target values into a plausible Hz-like range (1–6 Hz). This is the current working approach but still needs validation.
- **Remaining check**: Verify that the *relative* rates between populations (PYR > SOM ≈ PV > VIP) and between conditions (e.g., APP vs control direction) are consistent with published data. The absolute values matter less since the model will be fitted to these targets — what matters is that the ordering and ratios are biologically plausible.

---

### Task -1.2 — Cross-validate computed rates against published data

**What**: Compare the computed target rates (from `AD_data/summary/targets_1mo.json`) against the original article's reported values.

**Computed 1mo target rates** (from `targets_1mo.json`):
| Cell type | Genotype folder | Computed mean | Computed median | n_files | n_mice |
|---|---|---|---|---|---|
| PYR | WT | 4.143 | 3.785 | 37 | 5 |
| PV | PV_control | 2.079 | 1.602 | 23 | 5 |
| SOM | SST_control | 3.423 | 2.976 | 26 | 5 |
| VIP | VIP_control | 1.933 | 1.857 | 27 | 4 |
| PYR (a7KO) | a7KO_control | 3.513 | 3.387 | 21 | 4 |
| PYR (b2KO) | b2KO_control | 4.799 | 4.851 | 33 | 4 |
| PYR (a5KO) | a5KO_control | 3.790 | 3.722 | 44 | 6 |

**Sanity checks**:
What's the impact of the nicotinic KOs on PYR rates? Does this match the published article's directionality?

---

### Task -1.3 — Verify recording duration parsing

**What**: The `parse_duration()` function in `compute_target_rates.py` uses a regex `t\s*=\s*([\d.]+)` to extract duration from `parameters.rtf`. Some `.rtf` files have rich-text formatting that could cause misparses.

**Check**: Run the script and print per-mouse durations:
```bash
cd /home/val/Depot_Github/internship_ENS
python -c "
from scripts.compute_target_rates import parse_duration
from pathlib import Path
root = Path('AD_data/1mo_post_injection')
for geno in sorted(root.iterdir()):
    if not geno.is_dir(): continue
    for mouse in sorted(geno.iterdir()):
        if not mouse.is_dir(): continue
        t = parse_duration(mouse)
        print(f'{geno.name}/{mouse.name}: t={t:.1f}s')
" 2>&1 | head -50
```

**What to look for**:
- All durations should be ~164.9s (the default). If some are very different (e.g., 330s), those mice have longer recordings and the rate computation is still correct (mean(F)/t normalizes for duration)
- If any durations are the fallback 164.897, check that the parameters file actually doesn't exist or can't be parsed
You can check in a different level of hierarchy if needed, it might be at a higher level to define the recording duration for all mice in that genotype.
---

### Task -1.4 — Decision: use mean or median for fitting targets

The `fitting_roadmap.md` uses **means** as default targets. The per-mouse variance is high (e.g., PV_control: mouse_2=0.91, mouse_5=5.95). Using means vs medians affects the targets:

| Cell type | Mean | Median | Difference |
|---|---|---|---|
| PYR | 4.143 | 3.785 | +9% |
| PV | 2.079 | 1.602 | +30% |
| SOM | 3.423 | 2.976 | +15% |
| VIP | 1.933 | 1.857 | +4% |

**Recommendation**: Use **means** for consistency (the optimization already uses `--target_pyr 4.143` etc.), but note the sensitivity to outlier mice. The PV mean is heavily influenced by one high-firing mouse (mouse_5: 5.95). Consider reporting both in the paper.

**DECISION POINT for user**: Mean or median? This is a judgment call. Mean is standard but outlier-sensitive. Median is robust but ignores the tails.

---

## PHASE 0 — WT Parameter Optimization — COMPLETE for WT and WT_APP at 1mo

> **Goal**: Achieve a converged WT parameter fit where all 4 populations match target rates and all 3 KO conditions produce correct PYR rate changes. This is the foundation for everything downstream.
>
> **Status**: `params/new/WT_1mo_article.json` (loss=1e-4) and `params/new/WT_APP_1mo_article.json` (loss=9.8e-5) are converged. Boxplot matches article figure. Model was simplified by removing `w_vv` and `w_ps` (see `docs/changes_removed_connections.md`). Tasks 0.1–0.3 below are kept for reference but are superseded by the article params.

---

### Task 0.2 — WT and WT_APP 1mo optimization (DONE)

**Targets**: Taken from `data_1mo_article.md` — the published article's median activity values (transients/min), used directly as fitting targets. No KO PYR constraints were included because `data_1mo_article.md` only covers WT and WT_APP per-population rates.

| Population | WT target | WT_APP target |
|---|---|---|
| PYR | 2.487 | 3.279 |
| SOM | 3.248 | 3.167 |
| PV  | 1.414 | 3.166 |
| VIP | 2.517 | 2.501 |

**Commands used**:
```bash
# WT fit
python -m circuit_model optimize \
  --target_pyr 2.487 \
  --target_som 3.248 \
  --target_pv  1.414 \
  --target_vip 2.517 \
  --optimizer chaining \
  --n_samples 50000 \
  --n_workers 8 \
  --save_best_json params/new/WT_1mo_article.json \
  --log_file figs/optim/1mo/log.jsonl

# WT_APP fit
python -m circuit_model optimize \
  --target_pyr 3.279 \
  --target_som 3.167 \
  --target_pv  3.166 \
  --target_vip 2.501 \
  --optimizer chaining \
  --n_samples 50000 \
  --n_workers 8 \
  --save_best_json params/new/WT_APP_1mo_article.json \
  --log_file figs/optim/1mo_APP/log.jsonl
```

**Results**:
| | Loss | PYR | SOM | PV | VIP |
|---|---|---|---|---|---|
| WT | 1e-4 | 2.487 | 3.248 | 1.414 | 2.517 |
| WT_APP | 9.8e-5 | 3.279 | 3.167 | 3.166 | 2.501 |

Both fits converged. Boxplot of simulated rates matches the article figure.

---

### Task 0.3 — Validate converged WT fit

**What**: Once optimization converges, validate the fitted parameters.

**Validation steps**:
1. **Single run visualization**:
   ```bash
   python -m circuit_model run \
     --params_json params/new/WT_1mo_article.json \
     --noise_type white --T_ms 5000 --burn_in_ms 2000 \
     --save_plot figs/data/fit_validation_WT_run.png
   ```
   Verify: all 4 populations fire at reasonable rates, no population is silent or saturating.

2. **8-condition study** (zero-parameter prediction for KO and APP):
   ```bash
   python -m circuit_model study \
     --params_json params/new/WT_1mo_article.json \
     --n_runs 50 --noise_type white --n_workers 8 \
     --save_plot figs/data/study_WT_params_all8.png
   ```
   Verify: KO PYR rates match targets within ~20%. APP rates are a free prediction.

3. **Biological plausibility check**:
   - PYR adaptation timescale should be ~100-500 ms (biological range)
   - SOM adaptation timescale should be ~1000-3000 ms (slow, biological)
   - Transfer function thresholds should be positive
   - Synaptic weights should be positive
   - All receptor currents should be positive

**Metrics to extract and report** (for Fig 2):
- Per-population firing rate comparison: simulated vs experimental (box plot or bar chart)
- Per-condition (8 conditions): PYR rate comparison
- Relative error per population and per condition

**Figure destination**: Fig 2 (fit validation panel)

---

### Task 0.4 — Prepare KO parameter sets (Option A: zero-activation)

**What**: Once WT is fitted, prepare KO conditions by zeroing the relevant receptor activation. No new optimization needed — the `--condition` flag handles this internally.

| Condition | Parameter changes from WT (applied internally) |
|---|---|
| a7_KO | `act_alpha7 = 0`, `g_alpha7 = 0` |
| b2_KO | `act_beta2 = 0` |
| a5_KO | `act_alpha5 = 0` |
| WT_APP | `act_alpha7 = 0.10`, `act_beta2 = 0.875`, `act_alpha5 = 0.60` |

**Verification**: Already done in Task 0.3 step 2 (the `study` command runs all 8 conditions).

---

### Task 0.6 — Ring calibration (WT and WT_APP)

**What**: Run calibration separately for each parameter set to find the (amplitude, w_pyr_pyr_inter, w_pv_global) triplet that produces a stable working memory bump. Because WT and WT_APP have different local circuit dynamics, they will have different operating regimes and must be calibrated independently.

**Commands**:
```bash
# --- WT calibration ---
python -m circuit_model ring-noise-floor   
  --conditions WT   
  --w_inter_values 3 4 5 6 7 7.5 8 8.1 8.25 8.3 8.4 8.5 9  
  --params_json params/new/WT_1mo_article.json 
  --n_baseline 100 --sigma_pyr_deg 15 --n_workers 8

python -m circuit_model ring-calibrate \
  --conditions WT \
  --amplitudes 2 4 6 8 10 12 15 18 22 \
  --w_inter_values 3 4 5 6 7 8 9 \
  --params_json params/new/WT_1mo_article.json \
  --n_trials 200 --sigma_pyr_deg 15 --n_workers 8

# --- WT_APP calibration ---
python -m circuit_model ring-noise-floor \
  --conditions WT \
  --params_json params/new/WT_APP_1mo_article.json \
  --w_inter_values 3 4 5 6 7 8 9 \
  --n_baseline 200 --sigma_pyr_deg 15 --n_workers 8

python -m circuit_model ring-calibrate \
  --conditions WT \
  --amplitudes 2 4 6 8 10 12 15 18 22 27 33 40 \
  --w_inter_values 1 2 3 4 5 6 7 8 9 10 11 12 \
  --params_json params/new/WT_APP_1mo_article.json \
  --n_trials 200 --sigma_pyr_deg 15 --n_workers 8
```

**Expected outputs** (one set per param file): Heatmaps of success rate, mean Â, peak PYR rate across the 2D grid. A `calibration_recommended.json` with the best (amplitude, w_inter) pair.

**Selection criteria** (from existing calibration logic):
- Success rate = 1.0 (bump always forms)
- Peak PYR rate in [17.0, 18.5] Hz (physiological WM range)
- Among candidates: highest mean Â

**Result notation**: After this task, the following placeholders are defined and used in all subsequent commands:
- `AMP_WT`, `W_INTER_WT`, `W_PV_WT` — from WT calibration (`params/new/WT_1mo_article.json`)
- `AMP_APP`, `W_INTER_APP`, `W_PV_APP` — from WT_APP calibration (`params/new/WT_APP_1mo_article.json`)
- `SIGMA` = 15 (shared, fixed)

---

### Task 0.7 — Decision point: parameter strategy for KO ring simulations

KO conditions (a7_KO, b2_KO, a5_KO) are simulated using Option A only: the WT parameter set (`params/new/WT_1mo_article.json`) with the relevant receptor activation zeroed, and the WT calibration values (AMP_WT, W_INTER_WT, W_PV_WT). This is the only viable approach because KO calcium imaging data provides PYR rates only — a full re-optimization would be underdetermined.

WT_APP uses its own fitted parameter set (`params/new/WT_APP_1mo_article.json`) with its own calibration values (AMP_APP, W_INTER_APP, W_PV_APP). The disease effect is captured entirely in the different local circuit parameters, not in the receptor activation levels (which are kept at 1.0 for ring runs). All ring commands for WT_APP use `--condition WT` with the APP params file.

---

## PHASE 1 — WT and WT_APP Baseline Characterization (Article Section 3)

> **Goal**: Fully characterize both WT and WT_APP bumps side by side. This is the primary comparison of the paper — every analysis is run with both parameter sets to directly compare healthy vs disease circuit.
>
> **PREREQUISITE**: Task 0.6 complete — both calibrations done.
>
> **Ring parameter convention**:
> - WT commands: `--params_json params/new/WT_1mo_article.json --amplitude AMP_WT --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15`
> - WT_APP commands: `--params_json params/new/WT_APP_1mo_article.json --amplitude AMP_APP --w_pyr_pyr_inter W_INTER_APP --w_pv_global W_PV_APP --sigma_pyr_deg 15`
> - KO commands (Phase 2): use WT params + WT calibration values + `--condition a7_KO` etc.

### Task 1.1 — Bump baseline ring-run (WT and WT_APP)

**What**: Generate the canonical bump visualization for both conditions for the paper.

**Commands**:
```bash
# WT
python -m circuit_model ring-run \
  --condition WT \
  --params_json params/new/WT_1mo_article.json \
  --amplitude AMP_WT --delay_ms 8000 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --seed 42

# WT_APP
python -m circuit_model ring-run \
  --condition WT \
  --params_json params/new/WT_APP_1mo_article.json \
  --amplitude AMP_APP --delay_ms 8000 \
  --w_pyr_pyr_inter W_INTER_APP --w_pv_global W_PV_APP --sigma_pyr_deg 15 \
  --seed 42
```

**Expected outputs** (per condition): `dashboard.png`, `bump_metrics.png`, `connectome.png`, `snapshot_evolution.mp4`

**Metrics to report**:
- Mean PYR rate at bump center during delay
- Mean bump amplitude (Â) and width (σ°) during delay
- Qualitative: stability, oscillation, persistence

**Comparison**: Does the WT_APP bump differ in shape, width, or stability from WT at matched operating points?

**Figure destination**: Fig 3 panel A (WT) and corresponding WT_APP panel

---

### Task 1.2 — Oscillation characterization (WT and WT_APP)

**What**: Extract dominant oscillation frequency and power during the delay period for both conditions.

**Commands**:
```bash
# WT
python -m circuit_model ring-oscillation-study \
  --conditions WT \
  --amplitudes AMP_WT \
  --params_json params/new/WT_1mo_article.json \
  --n_trials 500 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --min_freq_hz 1.0 --max_freq_hz 15.0 \
  --tf_window_s 1.0 --tf_overlap 0.8 \
  --n_workers 8

# WT_APP
python -m circuit_model ring-oscillation-study \
  --conditions WT \
  --amplitudes AMP_APP \
  --params_json params/new/WT_APP_1mo_article.json \
  --n_trials 500 \
  --w_pyr_pyr_inter W_INTER_APP --w_pv_global W_PV_APP --sigma_pyr_deg 15 \
  --min_freq_hz 1.0 --max_freq_hz 15.0 \
  --tf_window_s 1.0 --tf_overlap 0.8 \
  --n_workers 8
```

**Metrics to report**:
- **Dominant frequency** (Hz): median ± IQR across trials per condition
- **Power stability**: variance of instantaneous frequency over delay
- **Power magnitude**: median delay-averaged power
- **WT vs WT_APP comparison**: frequency shift, power change, stability change

**Interpretation**:
- Stable oscillation in WT → PV-mediated fast inhibitory feedback loop
- If WT_APP shows higher or lower frequency → different PV engagement in APP circuit
- If WT_APP power is less stable → disease degrades oscillatory coherence

**Figure destination**: Fig 3 panel B/C (WT), same panels or adjacent for WT_APP

---

### Task 1.3 — Amplitude sweep: oscillation vs drive strength (WT and WT_APP)

**What**: Sweep stimulus amplitudes to test whether oscillation frequency/power depends on drive strength, for both conditions.

**Commands**:
```bash
# WT
python -m circuit_model ring-oscillation-study \
  --conditions WT \
  --amplitudes 2 4 6 8 10 12 15 18 22 27 33 40 \
  --params_json params/new/WT_1mo_article.json \
  --n_trials 200 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --min_freq_hz 1.0 --max_freq_hz 15.0 \
  --n_workers 8

# WT_APP
python -m circuit_model ring-oscillation-study \
  --conditions WT \
  --amplitudes 2 4 6 8 10 12 15 18 22 27 33 40 \
  --params_json params/new/WT_APP_1mo_article.json \
  --n_trials 200 \
  --w_pyr_pyr_inter W_INTER_APP --w_pv_global W_PV_APP --sigma_pyr_deg 15 \
  --min_freq_hz 1.0 --max_freq_hz 15.0 \
  --n_workers 8
```

**Metrics**: Frequency vs amplitude curve for each condition, power vs amplitude, Pearson r.

**Key comparison**: Does the WT_APP frequency-amplitude relationship differ from WT? A steeper or shifted curve would indicate different feedback gain in the APP circuit.

**Figure destination**: Fig 3 supplementary

---

### Task 1.4 — Asymmetry analysis (WT and WT_APP)

**What**: Measure corrected bump asymmetry for both conditions across a range of amplitudes.

**Commands**:
```bash
# WT — amplitude sweep
python -m circuit_model ring-asymmetry \
  --conditions WT \
  --params_json params/new/WT_1mo_article.json \
  --amplitudes 2 4 6 8 10 12 15 18 22 27 33 40 \
  --n_trials 500 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --correct_asymmetry \
  --n_workers 8

# WT_APP — amplitude sweep
python -m circuit_model ring-asymmetry \
  --conditions WT \
  --params_json params/new/WT_APP_1mo_article.json \
  --amplitudes 2 4 6 8 10 12 15 18 22 27 33 40 \
  --n_trials 500 \
  --w_pyr_pyr_inter W_INTER_APP --w_pv_global W_PV_APP --sigma_pyr_deg 15 \
  --correct_asymmetry \
  --n_workers 8
```

**Metrics to report**:
- **mean|A(t)|** per amplitude per condition
- **mean|A(t)| vs amplitude slope** (OLS): steeper in WT_APP → disease amplifies noise sensitivity
- **std(A)**: Variability over delay
- **Pre-cue → delay correlation**: Pearson r — expect stronger in WT_APP (reduced inhibitory reset)

**Interpretation**: The asymmetry amplitude-sweep is a mechanistic signature. If WT_APP steepens the slope, the disease network converts stimulus drive into spatial instability more efficiently. The asymmetry is also a direct metric of noise sensitivity: higher asymmetry → more noise-sensitive bump.

**Figure destination**: Fig 3 panel D + WT vs WT_APP slope comparison panel

---

### Task 1.5 — MSD / Diffusion analysis (WT and WT_APP)

**What**: Compute mean squared displacement of bump center during delay for both conditions.

**Commands**:
```bash
# WT
python -m circuit_model ring-diffusion \
  --conditions WT \
  --params_json params/new/WT_1mo_article.json \
  --amplitude AMP_WT \
  --n_trials 500 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --n_workers 8

# WT_APP
python -m circuit_model ring-diffusion \
  --conditions WT \
  --params_json params/new/WT_APP_1mo_article.json \
  --amplitude AMP_APP \
  --n_trials 500 \
  --w_pyr_pyr_inter W_INTER_APP --w_pv_global W_PV_APP --sigma_pyr_deg 15 \
  --n_workers 8
```

**Metrics to report**:
- **B̂ (rad²/s)** per condition: oscillation-corrected diffusion coefficient
- **MSD curve overlay**: WT vs WT_APP
- **Oscillatory component frequency**: should match Task 1.2 result per condition

**Interpretation**: Higher B̂ in WT_APP → greater memory drift → worse WM precision in disease.

**Figure destination**: Fig 3 panel E

---

### Task 1.6 — Bump decay study (WT and WT_APP)

**What**: Determine whether the bump is self-sustained or decaying for both conditions, and compare persistence timescales.

**Commands**:
```bash
# WT
python -m circuit_model ring-bump-decay-study \
  --conditions WT \
  --amplitudes 2 4 6 8 10 12 15 18 22 27 33 40 \
  --params_json params/new/WT_1mo_article.json \
  --delay_ms 15000 \
  --n_trials 200 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --n_workers 8

# WT_APP
python -m circuit_model ring-bump-decay-study \
  --conditions WT \
  --amplitudes 2 4 6 8 10 12 15 18 22 27 33 40 \
  --params_json params/new/WT_APP_1mo_article.json \
  --delay_ms 15000 \
  --n_trials 200 \
  --w_pyr_pyr_inter W_INTER_APP --w_pv_global W_PV_APP --sigma_pyr_deg 15 \
  --n_workers 8
```

**Metrics to report**:
- **Normalized amplitude at delay end** per condition vs amplitude
- **Critical amplitude**: Stimulus strength above which bump is self-sustained — compare WT vs WT_APP
- **Decay timescale** τ_decay if decaying: is it faster in WT_APP?

**Interpretation**: If WT_APP requires a stronger stimulus to sustain the bump, the disease circuit is less efficient at maintaining persistent activity.

**Figure destination**: Fig 3 supplementary

---

### Task 1.7 — Distractor baseline (WT and WT_APP)

**What**: Characterize how both conditions respond to distractors at varying angular offsets. This is one of the key novel results of the paper.

**Commands**:
```bash
# WT
python -m circuit_model ring-osc-distractor-study \
  --conditions WT \
  --amplitudes AMP_WT \
  --params_json params/new/WT_1mo_article.json \
  --distractor_factors 0.5 1.0 1.5 \
  --offsets_deg 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 \
  --delay1_ms 1500 --distractor_duration_ms 200 --delay2_ms 4000 \
  --n_trials 100 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --min_freq_hz 1 --max_freq_hz 15 \
  --n_workers 8

# WT_APP
python -m circuit_model ring-osc-distractor-study \
  --conditions WT \
  --amplitudes AMP_APP \
  --params_json params/new/WT_APP_1mo_article.json \
  --distractor_factors 0.5 1.0 1.5 \
  --offsets_deg 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 \
  --delay1_ms 1500 --distractor_duration_ms 200 --delay2_ms 4000 \
  --n_trials 100 \
  --w_pyr_pyr_inter W_INTER_APP --w_pv_global W_PV_APP --sigma_pyr_deg 15 \
  --min_freq_hz 1 --max_freq_hz 15 \
  --n_workers 8
```

**Metrics to report**:
- **Merge threshold**: Smallest offset with alternation per condition — does WT_APP shift this threshold?
- **PLV vs offset curve**: Overlay WT and WT_APP on same plot
- **Power change at each offset**: cue power suppression by distractor per condition
- **Regime map**: merge / intermediate / alternate per offset per condition
- **Distractor factor sweep**: how does distractor strength modulate merge threshold in each condition?

**Key comparison**: WT_APP should show reduced phase alignment capacity — the alternation regime may collapse or shift, reflecting reduced oscillatory coherence in disease.

**Clinical interpretation**: Reduced distractor suppression in WT_APP → corresponds to attention/WM deficits in early Alzheimer's.

**Figure destination**: Fig 4 (merge/alternate map + PLV curves, WT vs WT_APP side by side)

---

### Task 1.8 — Phase-dependent distractor timing (WT and WT_APP)

**What**: At a fixed far offset (in the alternation regime), sweep the distractor onset time within one oscillation period. Test whether timing relative to oscillation phase determines outcome, and whether this gating is preserved in WT_APP.

**Commands**:
```bash
# WT
python -m circuit_model ring-osc-phase-distractor \
  --conditions WT \
  --amplitudes AMP_WT \
  --params_json params/new/WT_1mo_article.json \
  --distractor_factors 1.0 \
  --offsets_deg 150 170 \
  --delay1_base_ms 500 \
  --distractor_duration_ms 200 --delay2_ms 4000 \
  --n_phase_sweep 32 \
  --n_trials 50 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --n_workers 8

# WT_APP
python -m circuit_model ring-osc-phase-distractor \
  --conditions WT \
  --amplitudes AMP_APP \
  --params_json params/new/WT_APP_1mo_article.json \
  --distractor_factors 1.0 \
  --offsets_deg 150 170 \
  --delay1_base_ms 500 \
  --distractor_duration_ms 200 --delay2_ms 4000 \
  --n_phase_sweep 32 \
  --n_trials 50 \
  --w_pyr_pyr_inter W_INTER_APP --w_pv_global W_PV_APP --sigma_pyr_deg 15 \
  --n_workers 8
```

**Metrics to report**:
- **Phase modulation depth**: max(PLV) - min(PLV) across phases per condition
- **Polar plot comparison**: WT vs WT_APP phase tuning curves
- **Key question**: Is phase-gating weaker in WT_APP? If so → disease degrades temporal control of distractor suppression

**Interpretation**: If WT shows strong phase modulation but WT_APP does not → the oscillation in the APP circuit is present but has lost its ability to gate information entry. This is a mechanistically rich disease signature distinct from simple amplitude reduction.

**Figure destination**: Fig 4 panel D (WT), Fig 4 panel E (WT_APP), or side-by-side polar comparison

---

## PHASE 2 — KO Interneuron Dissection (Article Section 4)

> **Goal**: Understand the specific contribution of each interneuron class to bump formation, oscillation, and distractor resistance. All KO conditions use WT params (`params/new/WT_1mo_article.json`) + WT calibration values. WT_APP from Phase 1 serves as the disease reference throughout.

### Task 2.0 — Firing rates and synaptic drive during delay (WT and WT_APP)

**What**: Characterize the circuit during active WM: per-population firing rates at bump vs background nodes for both conditions.

**Approach**: Extract from `ring-run` results (Task 1.1):
1. PYR, PV, SOM, VIP rates at bump center vs background (> 60° away)
2. Ratio of bump/background rate per population per condition
3. Cholinergic contribution: fraction of interneuron drive from nAChR currents

**New code needed**: Script loading a `ring-run` result and producing:
- Panel A: Spatial profile of each population at a delay timepoint
- Panel B: Time course of bump-node rates per population during delay
- Panel C: Bump/background ratio bar chart per population, WT vs WT_APP

**Metrics**: PYR rate at bump (~17 Hz from calibration); PV, SOM, VIP engagement at bump; how these change in WT_APP.

**Figure destination**: Fig 5

---

### Task 2.1 — Ring study for all KO conditions

**What**: Run the full ring-study across WT + 3 KOs, using WT params and calibration.

**Command**:
```bash
python -m circuit_model ring-study \
  --conditions WT a7_KO b2_KO a5_KO \
  --amplitudes AMP_WT \
  --params_json params/new/WT_1mo_article.json \
  --n_trials 1000 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --n_workers 8 --error_band sem
```

**Metrics to extract per condition**:
| Metric | What it measures | Expected direction per KO |
|---|---|---|
| Amplitude (Â) | Bump strength/confidence | ↓ in a5_KO (less disinhibition), ↑ or ~ in a7_KO |
| Width (σ°) | Spatial precision | ↑ in a7_KO (less lateral inhibition) |
| Error from cue (°) | Systematic drift | ↑ in a7_KO |
| PYR rate at bump (Hz) | Excitability | ↑ in a7_KO and b2_KO, ↓ in a5_KO |

**Figure destination**: Fig 6 (KO bump metrics comparison)

---

### Task 2.2 — KO oscillation analysis

**What**: Compare oscillation frequency and power across WT and all 3 KOs. Run a dense amplitude sweep to map frequency-amplitude relationships per KO.

**Commands**:
```bash
# Single calibrated amplitude
python -m circuit_model ring-oscillation-study \
  --conditions WT a7_KO b2_KO a5_KO \
  --amplitudes AMP_WT \
  --params_json params/new/WT_1mo_article.json \
  --n_trials 500 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --min_freq_hz 1.0 --max_freq_hz 15.0 \
  --n_workers 8

# Amplitude sweep per KO
python -m circuit_model ring-oscillation-study \
  --conditions WT a7_KO b2_KO a5_KO \
  --amplitudes 2 4 6 8 10 12 15 18 22 27 33 40 \
  --params_json params/new/WT_1mo_article.json \
  --n_trials 200 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --min_freq_hz 1.0 --max_freq_hz 15.0 \
  --n_workers 8
```

**Metrics to report**:
| Condition | Expected frequency change | Expected power change | Mechanism |
|---|---|---|---|
| WT | Baseline | Baseline | Reference |
| a7_KO | Shift (PV feedback altered) | Change | PV loses α7 drive → feedback loop weakened |
| b2_KO | Minimal acute change | Late-delay damping reduced | SOM adaptation timescale → effect grows over delay |
| a5_KO | Preserved frequency | Lower power | VIP loss → SOM hyperactive → PYR reduced → weaker oscillation |

**Statistical tests**: Mann-Whitney U between conditions on `freq_mean_hz` and `power_mean`.

**Figure destination**: Fig 7

---

### Task 2.3 — KO asymmetry analysis

**What**: Compare corrected asymmetry across WT and KOs, with amplitude sweep.

**Commands**:
```bash
# Amplitude sweep across all KOs
python -m circuit_model ring-asymmetry \
  --conditions WT a7_KO b2_KO a5_KO \
  --params_json params/new/WT_1mo_article.json \
  --amplitudes 2 4 6 8 10 12 15 18 22 27 33 40 \
  --n_trials 500 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --correct_asymmetry \
  --n_workers 8
```

**Metrics**:
| Condition | Expected mean|A(t)| | Expected pre-cue→delay r | Mechanism |
|---|---|---|---|
| WT | Low (baseline) | Weak | Inhibitory reset works |
| a7_KO | Higher | Possibly stronger | Reduced PV/SOM feedback → less spatial stabilization |
| b2_KO | Higher at late delay | Weak | SOM adaptation loss → late-delay spatial instability |
| a5_KO | Higher (amplitude-normalized) | Weak | Lower amplitude → more noise-vulnerable |

**Special analysis for b2_KO**: Compute mean|A(t)| in early delay (0–2s) vs late delay (3–5s) separately.

**Figure destination**: Fig 6 (asymmetry panel)

---

### Task 2.4 — KO diffusion analysis

**What**: Compare MSD / diffusion coefficient across WT and KOs.

**Command**:
```bash
python -m circuit_model ring-diffusion \
  --conditions WT a7_KO b2_KO a5_KO \
  --params_json params/new/WT_1mo_article.json \
  --amplitude AMP_WT \
  --n_trials 500 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --n_workers 8
```

**Metrics**:
- **B̂ per condition**: Expect a7_KO > b2_KO > a5_KO relative to WT (to validate after running)
- **MSD curve overlay**: All 4 conditions on same plot

**Figure destination**: Fig 6 (MSD panel)

---

### Task 2.5 — KO distractor experiments

**What**: Test distractor merge/alternate behavior per KO at a dense offset grid. Compare merge threshold and PLV curves across conditions.

**Commands**:
```bash
# Full offset sweep, multiple distractor strengths
python -m circuit_model ring-osc-distractor-study \
  --conditions WT a7_KO b2_KO a5_KO \
  --amplitudes AMP_WT \
  --params_json params/new/WT_1mo_article.json \
  --distractor_factors 0.5 1.0 1.5 \
  --offsets_deg 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 \
  --delay1_ms 1500 --distractor_duration_ms 200 --delay2_ms 4000 \
  --n_trials 100 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --n_workers 8
```

**Per-KO predictions**:
| KO | Merge threshold | PLV stability | Distractor dominance |
|---|---|---|---|
| a7_KO | Larger (less PV inhibition) | Less stable | Higher |
| b2_KO | Similar or slightly larger | Late-delay degradation | Moderate |
| a5_KO | Similar | Less stable (lower amplitude) | Higher |

**Figure destination**: Fig 8

---

### Task 2.6 — KO comparison summary

**What**: Compile all metrics into a single comparison table and figure.

**Analysis** (post-processing of Tasks 2.1–2.5):

**Summary table to create**:
| Metric | WT | a7_KO | b2_KO | a5_KO |
|---|---|---|---|---|
| Amplitude (Â) | — | ↑/↓/~ | ↑/↓/~ | ↑/↓/~ |
| Width (σ°) | — | ↑/↓/~ | ↑/↓/~ | ↑/↓/~ |
| mean\|A(t)\| | — | ↑/↓/~ | ↑/↓/~ | ↑/↓/~ |
| B̂ (diffusion) | — | ↑/↓/~ | ↑/↓/~ | ↑/↓/~ |
| Osc. frequency (Hz) | — | ↑/↓/~ | ↑/↓/~ | ↑/↓/~ |
| Osc. power | — | ↑/↓/~ | ↑/↓/~ | ↑/↓/~ |
| Merge threshold (°) | — | ↑/↓/~ | ↑/↓/~ | ↑/↓/~ |
| PLV at 170° | — | ↑/↓/~ | ↑/↓/~ | ↑/↓/~ |

**Key message**: Each interneuron class has a *distinct fingerprint*:
- **PV/SOM (via α7)**: precision + oscillation frequency + distractor resistance
- **SOM (via β2)**: temporal stability + late-delay damping
- **VIP (via α5)**: amplitude + oscillation power + distractor vulnerability

**New code needed**: Script to aggregate CSV outputs from Tasks 2.1–2.5 and produce the summary table + bar plot figure.

**Figure destination**: Fig 9

---

## PHASE 3 — KO×APP Interaction Study (Article Section 5)

> **Goal**: Characterize the additional effect of APP on top of each KO. The WT vs WT_APP baseline comparison is already established in Phase 1. This phase focuses on the 8-condition interaction matrix and on the desensitization landscape.

### Task 3.1 — Full 8-condition ring study

**What**: Run ring-study for all 8 conditions (WT, WT_APP-via-desensitization, 3×KO, 3×KO_APP) using WT params + APP desensitization levels. Note: this is distinct from the Phase 1 WT_APP which uses the fitted APP params. Here we use the nAChR desensitization formulation to characterize the KO×APP interaction.

**Command**:
```bash
python -m circuit_model ring-study \
  --conditions WT WT_APP a7_KO a7_KO_APP b2_KO b2_KO_APP a5_KO a5_KO_APP \
  --amplitudes AMP_WT \
  --params_json params/new/WT_1mo_article.json \
  --n_trials 1000 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --n_workers 8 --error_band sem
```

**Key analysis**: For each KO×APP pair, compute the additional effect of APP beyond the KO:

| Pair | APP additional effect | Expected |
|---|---|---|
| a7_KO → a7_KO_APP | α7 already gone → APP hits remaining α5 (60%) and β2 (87.5%) | Small (saturation) |
| b2_KO → b2_KO_APP | β2 already gone → APP hits α7 (10%) and α5 (60%) | Moderate |
| a5_KO → a5_KO_APP | α5 already gone → APP hits α7 (10%) and β2 (87.5%) | Moderate |

**Key question**: Does a7_KO_APP ≈ a7_KO? If yes → α7 is the dominant driver of APP WM degradation.

**Figure destination**: Fig 10

---

### Task 3.2 — Rate-matched comparison (supplementary)

**What**: Find ring connectivity parameters that produce matched peak PYR rates for WT and WT_APP (from their respective param files). Compare bump quality at matched operating points.

**Approach**:
1. From Task 0.6 calibration heatmaps, identify (amp, w_inter) yielding ~18 Hz peak PYR for each
2. Re-run ring-study at these matched parameters
3. Compare bump metrics: differences that persist after rate-matching are attributable to circuit structure, not operating regime

**Metrics**: Parameter shift required per condition; bump quality comparison at equal PYR rate.

**Why**: Without rate-matching, differences could simply reflect operating regime shifts. This is the most principled comparison.

**Figure destination**: Supplementary Figure

---

## PHASE 4 — Parametric Desensitization Study (Article Section 5.3)

> **Goal**: Map the continuous landscape of receptor desensitization effects on WM quality.

### Task 4.1 — α7 desensitization sweep

**What**: Vary act_alpha7 from 0 to 1 in 10 steps, keeping β2 and α5 at WT levels. For each, run ring-study and extract bump metrics.

**Approach**: This requires running simulations with custom activation levels. The `--condition` flag uses preset values, so we need a different approach:

**Command** (new script — see Code 5.2):
```bash
python scripts/parametric_desensitization_sweep.py \
  --params_json params/new/WT_1mo_article.json \
  --sweep_param act_alpha7 \
  --sweep_values 0.0 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.0 \
  --n_trials 500 \
  --amplitude AMP_WT \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --n_workers 8
```

**Metrics per activation level**:
- mean|A(t)| (corrected asymmetry)
- Oscillation frequency and power
- Diffusion coefficient B̂
- Amplitude

**Output**: Line plots of each metric vs act_alpha7, showing how WM quality degrades with increasing desensitization.

**Key question**: Is there a threshold in α7 desensitization below which oscillatory stability collapses? The APP operating point (act_alpha7 = 0.10) should be near or past this threshold.

**Figure destination**: Fig 13 (x-axis)

---

### Task 4.2 — α5 desensitization sweep

**What**: Same as Task 4.1 but for act_alpha5.

**Expected findings**: α5 primarily affects amplitude → smooth, monotonic decrease in bump quality without a sharp threshold.

**Figure destination**: Fig 13 (y-axis)

---

### Task 4.3 — 2D desensitization heatmap

**What**: Sweep both act_alpha7 and act_alpha5 simultaneously (21×21 grid = 441 combinations). For each, extract a single WM quality metric.

**Metrics for heatmap**:
- Primary: corrected asymmetry mean|A(t)| (lower = better)
- Secondary: oscillation frequency stability (variance of instantaneous frequency)

**Output**: 2D heatmap with act_alpha7 on x-axis, act_alpha5 on y-axis, color = WM quality. Mark the WT operating point (1.0, 1.0) and the APP operating point (0.10, 0.60).

**Why**: This is the most visually compelling result for the disease story. It shows the full landscape of how dual receptor desensitization degrades WM, and places the APP condition on that landscape.

**Interpretation**:
- If there's a sharp boundary (phase transition) → below this, WM is impossible regardless of network compensation
- If α7 drives a steeper gradient than α5 → confirms α7 is the dominant receptor for WM maintenance
- If there's an interaction (non-additive effects) → α7 and α5 pathways interact, which is biologically interesting

**Figure destination**: Fig 13 (main panel)

---

## PHASE 5 — New Code Requirements

> These are scripts/modifications needed that don't exist yet.

### Code 5.1 — Synaptic drive decomposition at bump nodes

**Purpose**: For Task 2.0 (WT baseline synaptic analysis)

**What to implement**:
- Function that takes a ring simulation result and extracts:
  - Per-population firing rate at bump center ± 10° vs background (> 60° from bump)
  - Ratio of nAChR current to total excitatory current per interneuron at bump node
  - Stacked bar chart of excitatory/inhibitory drives at bump node

**Location**: New function in `circuit_model/ring/analysis.py` + plotting in `ring/plotting.py`

**Inputs**: RingSimulationResult, cue_angle_deg
**Outputs**: Dictionary with per-population bump/background rates + drive decomposition

---

### Code 5.2 — Parametric activation sweep (if --set not in ring-study)

**Purpose**: For Tasks 4.1–4.3

**What to implement**:
- Script `scripts/parametric_desensitization_sweep.py` that:
  1. Accepts ranges for act_alpha7, act_alpha5, act_beta2
  2. For each combination: modifies params, runs N trial ring simulations, extracts metrics
  3. Saves results to a single CSV
  4. Generates 1D line plots and 2D heatmaps

**Key design**: Re-use `simulate_ring_batch` from `circuit_model/ring/simulation.py` and `compute_bump_metrics` + `compute_oscillation_band_timecourse` from `circuit_model/ring/analysis.py`.

---

### Code 5.3 — KO summary aggregation script

**Purpose**: For Task 2.6

**What to implement**:
- Script `scripts/aggregate_ko_summary.py` that:
  1. Reads `study_metrics.csv` from ring-study output
  2. Reads `asymmetry_trials.csv` from ring-asymmetry output
  3. Reads `oscillation_trial_summary.csv` from ring-oscillation-study output
  4. Reads `diffusion_summary.csv` from ring-diffusion output
  5. Reads `osc_distractor_trials.csv` from ring-osc-distractor-study output
  6. Compiles a single summary table with all metrics per condition
  7. Generates the comparison bar plot figure (Fig 9)

---

### Code 5.4 — Early vs late delay analysis for b2_KO

**Purpose**: For Task 2.3 (testing SOM adaptation timescale hypothesis)

**What to implement**:
- Modify `ring-asymmetry` or add post-processing script that:
  1. Splits the delay into early (0–2s) and late (3–5s) windows
  2. Computes mean|A(t)| and asym_std separately for each window
  3. Tests whether the late-delay asymmetry is significantly higher than early-delay in b2_KO (but not in WT)
  4. Statistical test: paired Wilcoxon signed-rank test (early vs late within each trial)

---

## PHASE 6 — Figures Roadmap (Complete)

| Figure | Content | Data source (Tasks) | Status |
|---|---|---|---|
| **Fig 1** | Circuit diagram (4 populations + nAChR) + Ring schematic | Manual illustration | To draw |
| **Fig 2** | Fit validation: simulated vs experimental rates, WT and WT_APP | Task 0.2 | To do |
| **Fig 3** | WT vs WT_APP bump baseline: dashboard + oscillation + asymmetry slope + MSD | Tasks 1.1–1.5 | To do |
| **Fig 4** | Distractor: merge/alternate map + PLV curves + phase gating, WT vs WT_APP | Tasks 1.7–1.8 | To do |
| **Fig 5** | Delay-period firing rates per population at bump/background, WT vs WT_APP | Task 2.0 | To do (needs Code 5.1) |
| **Fig 6** | KO bump metrics: amplitude, asymmetry, MSD, width per KO | Tasks 2.1, 2.3, 2.4 | To do |
| **Fig 7** | KO oscillation: spectrogram per KO + frequency/power summary | Task 2.2 | To do |
| **Fig 8** | KO distractor: PLV curves per KO + merge threshold comparison | Task 2.5 | To do |
| **Fig 9** | KO comparison summary: directional matrix + bar plots | Task 2.6 | To do (needs Code 5.3) |
| **Fig 10** | 8-condition ring study: KO×APP interaction matrix | Task 3.1 | To do |
| **Fig 11** | Parametric desensitization: 2D heatmap (α7 × α5 → WM quality) | Tasks 4.1–4.3 | To do (needs Code 5.2) |
| **Suppl. S1** | Calibration heatmaps: WT and WT_APP parameter spaces | Task 0.6 | To do |
| **Suppl. S2** | Rate-matched comparison | Task 3.2 | To do |
| **Suppl. S3** | Bump decay study: WT vs WT_APP | Task 1.6 | To do |
| **Suppl. S4** | WT vs WT_APP Jacobian + parameter comparison | Task 0.3 | To do |
| **Suppl. S5** | KO oscillation amplitude sweep | Task 2.2 | To do |

---

## PHASE 7 — Statistical Tests Checklist

| Comparison | Test | Where |
|---|---|---|
| KO vs WT (per metric) | Mann-Whitney U (non-parametric, unequal variance) | Figs 6, 7, 8, 9 |
| Pre-cue vs delay asymmetry (same trials) | Wilcoxon signed-rank (paired) | Fig 6 |
| Pre-cue → delay correlation | Pearson r with p-value | Fig 6 |
| Asymmetry slope (WT vs APP) | Bootstrap CI on OLS slope difference | Fig 11 |
| Early vs late delay asymmetry (b2_KO) | Wilcoxon signed-rank (paired) | Fig 6 (Code 5.4) |
| PLV across conditions | Mann-Whitney U per offset | Fig 8, 12 |
| Oscillation frequency across conditions | Mann-Whitney U | Fig 7 |
| Phase modulation significance | Rayleigh test for circular uniformity | Fig 4/8 |

**Error bands**: Use SEM for mean comparison plots (shows precision of mean); use SD for distribution plots (shows variability).

**Multiple comparisons**: When comparing 3 KOs to WT simultaneously, apply Bonferroni correction (p_adjusted = p × 3) or use Kruskal-Wallis as omnibus test first.

---

## PHASE 8 — Open Questions to Resolve During Analysis

These are investigative tasks that don't have a predetermined answer. The analysis may reveal unexpected findings that require adjusting the plan.

### Q1. Disentangling PV vs SOM in α7 effects
**Experiment**: Run two custom conditions: (1) α7 drive to PV only = 0 (SOM keeps α7), (2) α7 drive to SOM only = 0 (PV keeps α7). Compare bump metrics.
**How**: Requires modifying params to selectively zero I_alpha7_pv or I_alpha7_som. Use `--set I_alpha7_pv=0` or `--set I_alpha7_som=0`.
**If possible**: Run ring-study for these two selective conditions alongside full a7_KO.

### Q2. Is the ~7 Hz oscillation theta-like?
**Analysis**: Compare oscillation frequency to known theta-band range (4-8 Hz in rodent PFC). If it falls within this range, cite relevant literature (Compte et al., Wimmer et al.).
**Caveat**: Rate model frequencies are approximate — the exact correspondence requires a spiking model.

### Q3. Compensation mechanism
**Analysis**: When PYR rate rises in KO conditions, does bump width systematically increase? Plot width vs PYR rate across all conditions. If they correlate → no width compensation (wider = worse). If width is stable despite rate changes → some compensation exists.

### Q4. Bump persistence in APP
**Analysis**: Run bump-decay-study for WT and WT_APP. If APP bump decays faster → memory trace lifetime is shorter in disease. This has direct clinical implications.

---

## Execution Priority Order

### DONE — Phases -1 and 0 (Data validation + WT/WT_APP parameter fits)
- Converged WT 1mo params: `params/new/WT_1mo_article.json` (loss=1e-4)
- Converged WT_APP 1mo params: `params/new/WT_APP_1mo_article.json` (loss=9.8e-5)
- Boxplot validates against article figure
- Biologically unsupported connections (`w_vv`, `w_ps`) removed

### Tier 0 — CURRENT BLOCKER
1. **Task 0.6** → Ring calibration for both WT and WT_APP — **this is the #1 bottleneck**
2. **Task 0.3** → Validate both fits on all conditions (ring-study)

### Tier 1 — Critical path (after Tier 0)
3. Tasks 1.1–1.5 → WT and WT_APP baseline characterization (Figs 3)
4. Tasks 1.7–1.8 → Distractor experiments, WT and WT_APP (Fig 4)
5. Tasks 2.1–2.5 → KO dissection (Figs 5–8)
6. Task 2.6 → KO summary (Fig 9)
7. Task 3.1 → 8-condition ring study (Fig 10)

### Tier 2 — High impact, parallelizable (after Tier 0)
- Task 1.8 → Phase-distractor experiment — novel contribution
- Tasks 4.1–4.3 → Parametric sweep (Fig 11)

### Tier 3 — Supplementary / nice-to-have
- Task 1.6 → Bump decay study
- Task 3.2 → Rate-matched comparison
- Task 1.3 → Oscillation amplitude sweep
- Q1 → PV vs SOM disentangling in a7_KO

---

## Computational Resource Estimation

| Task | Trials × Conditions | Total estimate |
|---|---|---|
| ring-calibrate (WT, 12×12 grid, 200 trials) | 28800 | ~2-3 hours |
| ring-calibrate (WT_APP, same) | 28800 | ~2-3 hours |
| ring-study (4 KO conditions, 1000 trials) | 4000 | ~30 min |
| ring-study (8 conditions, 1000 trials) | 8000 | ~1 hour |
| ring-oscillation-study (4 cond, 500 trials, 12 amp) | 24000 | ~1-2 hours |
| ring-asymmetry (4 cond, 500 trials, 12 amp) | 24000 | ~1-2 hours |
| ring-diffusion (4 cond, 500 trials) | 2000 | ~20 min |
| ring-osc-distractor (4 cond × 17 offsets × 3 factors, 100 trials) | 20400 | ~1-2 hours |
| ring-osc-phase-distractor (32 phases × 2 offsets, 50 trials, WT+APP) | 6400 | ~30 min |
| Parametric sweep (21×21 grid × 500 trials) | 220500 | ~5-8 hours |

**Total**: roughly 15-25 hours of compute with 4 workers. Previously would have been ~85-90 hours.
