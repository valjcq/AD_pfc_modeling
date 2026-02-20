# CLI Reference

The unified CLI is invoked via `python -m circuit_model <command>`.

```
python -m circuit_model {run,optimize,study,ring-run,ring-study,ring-diffusion,ring-drift-field,ring-distractor-sweep,ring-calibrate} [options]
```

---

## Table of Contents

1. [run](#run) -- Single-circuit simulation with plotting
2. [optimize](#optimize) -- Nevergrad parameter optimization
3. [study](#study) -- Batch study across 8 experimental conditions
4. [ring-run](#ring-run) -- Ring attractor single-condition simulation
5. [ring-study](#ring-study) -- Ring attractor multi-condition comparison
6. [ring-diffusion](#ring-diffusion) -- MSD diffusion analysis (Seeholzer et al. 2019)
7. [ring-drift-field](#ring-drift-field) -- Distractor drift field analysis (Seeholzer et al. 2019)
8. [ring-distractor-sweep](#ring-distractor-sweep) -- 2D distractor sweep (Δφ × amplitude)
9. [ring-calibrate](#ring-calibrate) -- 2D parameter calibration (amplitude x w_inter)

---

## `run`

Run a single 4-population circuit simulation and visualize the results.

```bash
python -m circuit_model run [options]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--params_json` | str | `""` | Load parameters from JSON file |
| `--T_ms` | float | `2500.0` | Simulation duration (ms) |
| `--dt_ms` | float | `0.1` | Integration time step (ms) |
| `--noise_type` | str | `"none"` | Noise type: `none`, `white`, or `ou` (Ornstein-Uhlenbeck) |
| `--tau_noise_ms` | float | `5.0` | OU noise time constant (ms) |
| `--seed` | int | `None` | Random seed for reproducibility |
| `--burn_in_ms` | float | `500.0` | Burn-in period for statistics (ms) |
| `--time_range` | str | `""` | Time range to plot: `start,end` in ms (e.g. `1000,2000`) |
| `--save_plot` | str | `""` | Save plot to file path (e.g. `output.png`) |
| `--no_show` | flag | `False` | Don't display the plot (useful for batch processing) |
| `--unit` | str | `"transients/min"` | Rate unit for display: `transients/min` or `Hz` |

#### Transient Current Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--enable_transient` | flag | `False` | Enable time-dependent transient current |
| `--trans_start_ms` | float | `1000.0` | Transient onset time (ms) |
| `--trans_duration_ms` | float | `500.0` | Transient duration (ms) |
| `--trans_factor` | float | `0.2` | Transient as fraction of each population's I0 |

### Examples

```bash
# Default simulation
python -m circuit_model run

# Custom parameters with noise
python -m circuit_model run --params_json my_params.json --T_ms 5000 --noise_type ou

# With transient current
python -m circuit_model run --enable_transient --trans_start_ms 1000 --trans_duration_ms 500
```

---

## `optimize`

Run Nevergrad (TwoPointsDE) optimization to find parameters matching target firing rates.

```bash
python -m circuit_model optimize --target_pyr 5 --target_som 10 --target_pv 15 --target_vip 8 [options]
```

### Target Rates (required)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--target_pyr` | float | **required** | Target mean firing rate for PYR |
| `--target_som` | float | **required** | Target mean firing rate for SOM |
| `--target_pv` | float | **required** | Target mean firing rate for PV |
| `--target_vip` | float | **required** | Target mean firing rate for VIP |

### Optional Knockout Targets

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--target_alpha7_ko_pyr` | float | `None` | Target PYR rate under alpha7 knockout |
| `--target_alpha5_ko_pyr` | float | `None` | Target PYR rate under alpha5 knockout |
| `--target_beta2_ko_pyr` | float | `None` | Target PYR rate under beta2 knockout |

### Optimization Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n_samples` | int | `5000` | Number of optimization samples |
| `--top_k` | int | `10` | Keep top K candidates |
| `--early_stop_loss` | float | `1e-4` | Stop if loss falls below this value |

### Simulation Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--params_json` | str | `""` | Load base parameters from JSON file |
| `--T_ms` | float | `2500.0` | Simulation duration (ms) |
| `--dt_ms` | float | `0.1` | Integration time step (ms) |
| `--noise_type` | str | `"none"` | Noise type: `none`, `white`, or `ou` |
| `--tau_noise_ms` | float | `5.0` | OU noise time constant (ms) |
| `--seed` | int | `None` | Random seed |
| `--burn_in_ms` | float | `1800.0` | Burn-in period (ms) |
| `--window_ms` | float | `500.0` | Averaging window (ms) |
| `--n_trials` | int | `8` | Trials per parameter set |
| `--init_rate_scale` | float | `0.2` | Scale for random initial conditions |
| `--max_rate` | float | `200.0` | Maximum allowed rate (stability check) |

### KO Penalty Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--ko_min_effect_penalty` | float | `5.0` | Penalty weight for weak KO effect |
| `--ko_wrong_direction_penalty` | float | `10.0` | Penalty weight for wrong-direction KO effect |

### Parameter Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--freeze` | str | `""` | Comma-separated parameter names to freeze |
| `--set` | str | `""` | Override values: `name=val,name=val` (e.g. `--set w_vv=0,w_sp=0`) |
| `--show_params` | flag | `False` | Show which parameters are free vs frozen |

### I/O Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save_best_json` | str | `"best_params.json"` | Save best parameters to JSON file |
| `--log_file` | str | `"results_log.jsonl"` | Log results to JSONL file |
| `--log_interval` | int | `50` | Log every N steps |
| `--n_workers` | int | `None` | Parallel workers (auto if None) |
| `--unit` | str | `"transients/min"` | Rate unit for display: `transients/min` or `Hz` |

### Examples

```bash
# Basic optimization
python -m circuit_model optimize \
    --target_pyr 5 --target_som 10 --target_pv 15 --target_vip 8

# With knockout targets and frozen parameters
python -m circuit_model optimize \
    --target_pyr 5 --target_som 10 --target_pv 15 --target_vip 8 \
    --target_alpha7_ko_pyr 7 --target_beta2_ko_pyr 6 \
    --freeze "tau_s,g_gaba_base" --show_params --n_samples 10000

# Override specific parameters
python -m circuit_model optimize \
    --target_pyr 5 --target_som 10 --target_pv 15 --target_vip 8 \
    --set "w_vv=0,w_sp=0" --n_samples 5000
```

---

## `study`

Run simulations across 8 experimental conditions (WT, APP, KO variants) and generate box plots of firing rate distributions.

```bash
python -m circuit_model study [options]
```

### Conditions

The 8 conditions are:

| Key | Name | Description |
|-----|------|-------------|
| `WT` | Wild Type | All receptors active (act = 1.0) |
| `WT_APP` | Wild Type + APP | Receptor desensitization (sampled from distributions) |
| `a7_KO` | alpha7 KO | alpha7 = 0, g_alpha7 = 0 |
| `a7_KO_APP` | alpha7 KO + APP | alpha7 = 0 + APP desensitization on alpha5, beta2 |
| `b2_KO` | beta2 KO | beta2 = 0 |
| `b2_KO_APP` | beta2 KO + APP | beta2 = 0 + APP desensitization on alpha7, alpha5 |
| `a5_KO` | alpha5 KO | alpha5 = 0 |
| `a5_KO_APP` | alpha5 KO + APP | alpha5 = 0 + APP desensitization on alpha7, beta2 |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--params_json` | str | `""` | Load base parameters from JSON file |
| `--n_runs` | int | `50` | Number of simulations per condition |
| `--T_ms` | float | `2500.0` | Simulation duration (ms) |
| `--dt_ms` | float | `0.1` | Integration time step (ms) |
| `--noise_type` | str | `"white"` | Noise type (default: white) |
| `--tau_noise_ms` | float | `5.0` | OU noise time constant (ms) |
| `--seed` | int | `None` | Random seed |
| `--burn_in_ms` | float | `1800.0` | Burn-in period for statistics (ms) |
| `--window_ms` | float | `500.0` | Averaging window (ms) |
| `--fixed_receptor_values` | flag | `False` | Use fixed mean receptor values instead of sampling |
| `--n_workers` | int | `None` | Parallel workers (auto if None) |
| `--save_plot` | str | `""` | Save box plot to file path |
| `--no_show` | flag | `False` | Don't display the plot |
| `--unit` | str | `"transients/min"` | Rate unit for display |

### Examples

```bash
# Default study (50 runs per condition, white noise)
python -m circuit_model study

# Quick test with fixed receptor values
python -m circuit_model study --n_runs 10 --fixed_receptor_values --no_show

# Custom parameters with OU noise
python -m circuit_model study --params_json my_params.json --noise_type ou --tau_noise_ms 10
```

---

## `ring-run`

Run a ring attractor simulation for a single experimental condition with visualization.

```bash
python -m circuit_model ring-run [options]
```

### Condition Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--condition` | str | `"WT"` | Experimental condition. Valid: `WT`, `WT_APP`, `a5_KO`, `a5_KO_APP`, `a7_KO`, `a7_KO_APP`, `b2_KO`, `b2_KO_APP` |

### Common Ring Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n_nodes` | int | `128` | Number of nodes on the ring |
| `--params_json` | str | `""` | Load local circuit parameters from JSON file |
| `--amplitude` | float | `15.0` | Stimulus amplitude as factor of I_ext_pyr baseline (20 = 20× baseline current) |
| `--delay_ms` | float | `3000.0` | Delay period duration (ms) |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--no_show` | flag | `False` | Don't display plots |
| `--total_time_ms` | float | `None` | Total simulation time (overrides automatic timing) |
| `--record_dt_ms` | float | `1.0` | Recording time step (ms). Only every record_dt_ms the state is stored. Lower values use more memory |

#### Connectivity Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pyr_profile` | str | `"gaussian"` | PYR→PYR connectivity profile: `gaussian` or `compte` |
| `--J_plus` | float | `1.6` | Compte profile J+ parameter (local excitation). Only with `--pyr_profile compte` |
| `--sigma_pyr_deg` | float | `30.0` | PYR→PYR connectivity width (degrees) |
| `--w_pyr_pyr_inter` | float | `4.0` | Total PYR→PYR coupling for Gaussian profile. Not used with Compte |
| `--w_pv_global` | float | `2.0` | Total PV→PYR global inhibition strength |

**Gaussian profile** (default): Row-sum normalized Gaussian. `w_pyr_pyr_inter` controls total coupling strength.

**Compte et al. (2000) profile**: Local excitation + surround inhibition (Mexican hat). `J_plus` controls excitation peak; `J_-` is computed from normalization. Matrix scaled by 1/N for network-size invariance.

#### Response Transient Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--response_onset_ms` | float | `0.0` | Response transient onset after delay end (ms). 0 = disabled |
| `--response_duration_ms` | float | `500.0` | Duration of response transient (ms) |
| `--response_factor` | float | `0.5` | Response transient amplitude as fraction of I0 (+50% of baseline) |
| `--post_response_ms` | float | `3000.0` | Simulation time after response transient ends (ms) |

### Outputs

Generates in `figs/ring/<n_nodes>/<params_stem>/<conn_label>/amp<N>/<condition>/`:
- `dashboard.png` -- Activity heatmap, snapshots, firing rate traces
- `bump_metrics.png` -- Bump center, width, amplitude over time
- `connectome.png` -- PYR-PYR and PV-PYR connectivity matrices

The `<conn_label>` encodes both excitatory and inhibitory connectivity:
- Gaussian: `gauss_w<w>_s<sigma>-pv_unif_<w_pv>` (e.g. `gauss_w4_s30-pv_unif_2`)
- Compte: `compte_J<J+>_s<sigma>-pv_unif_<w_pv>` (e.g. `compte_J1.6_s30-pv_unif_2`)

### Examples

```bash
# Wild type, default amplitude (128 nodes)
python -m circuit_model ring-run --condition WT

# Smaller network
python -m circuit_model ring-run --condition WT --n_nodes 64

# Alpha7 KO with higher amplitude (30× baseline)
python -m circuit_model ring-run --condition a7_KO --amplitude 30 --delay_ms 5000

# With response transient
python -m circuit_model ring-run --condition WT --response_onset_ms 500 --response_factor 0.3
```

---

## `ring-study`

Run ring attractor simulations across multiple conditions and generate comparison plots.

```bash
python -m circuit_model ring-study [options]
```

### Study-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--conditions` | str (list) | all 8 | Conditions to simulate (space-separated) |
| `--amplitudes` | float (list) | `[15]` | Multiple stimulus amplitude factors (× I_ext_pyr) to compare |
| `--n_trials` | int | `100` | Number of trials per condition x amplitude |
| `--n_workers` | int | `None` | Number of parallel workers (default: min(4, cpu_count)) |
| `--delay_step_ms` | float | `200` | Delay evaluation step size (ms) |
| `--no_cache` | flag | `False` | Ignore existing CSV cache and recompute |
| `--amp_eval_step_ms` | float | `500` | Step (ms) for timed metrics-vs-amplitude plots. 0 = disabled |

Plus all [common ring parameters](#common-ring-parameters) from `ring-run`.

### Outputs

Generates in `figs/ring/<n_nodes>/<params_stem>/<conn_label>/`:
- `amp<N>/metrics_vs_delay.png` -- Bump metrics at multiple delay timepoints per condition
- `amp<N>/bump_metrics_comparison.png` -- Side-by-side activity and bump metrics
- `metrics_vs_amplitude.png` -- Cross-amplitude comparison at full delay (if multiple amplitudes)
- `metrics_vs_amplitude_at_<T>s.png` -- Cross-amplitude comparison at delay=T (every `amp_eval_step_ms`)
- `connectome.png` -- Connectivity visualization
- `study_metrics.csv` -- Cached metrics for all jobs (condition, amplitude, trial)

The `<conn_label>` ensures different connectivity configurations produce separate output directories (see [ring-run outputs](#outputs) for format).

### Examples

```bash
# All conditions, default amplitude (128 nodes)
python -m circuit_model ring-study

# Smaller network
python -m circuit_model ring-study --n_nodes 64

# Subset of conditions
python -m circuit_model ring-study --conditions WT WT_APP a7_KO

# Multi-amplitude study with trials
python -m circuit_model ring-study \
    --amplitudes 8 10 15 20 \
    --conditions WT WT_APP \
    --n_trials 10 --n_workers 4

# Custom delay evaluation
python -m circuit_model ring-study --delay_step_ms 500 --delay_ms 5000

# Force recompute (ignore cache)
python -m circuit_model ring-study --no_cache
```

---

## `ring-diffusion`

Compute the mean squared displacement (MSD) of the bump center during delay periods across conditions, and extract the diffusion strength $\hat{B}$ (slope of MSD vs time, in $\text{rad}^2/\text{s}$). Based on the drift-diffusion framework of Seeholzer, Deger & Gerstner (2019).

```bash
python -m circuit_model ring-diffusion [options]
```

### Diffusion-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--conditions` | str (list) | all 8 | Conditions to simulate (space-separated) |
| `--n_trials` | int | `50` | Number of trials per condition |
| `--n_workers` | int | `None` | Number of parallel workers (default: min(4, cpu_count)) |
| `--error_band` | str | `"sem"` | Error band type for plots: `sem` or `sd` |
| `--filter_cutoff_hz` | float | auto | Low-pass cutoff (Hz) for bump center trajectory. Auto-detected from oscillation spectrum. Set to `0` to disable. |

Plus all [common ring parameters](#common-ring-parameters) from `ring-run`.

### Method

For each condition:
1. Run `n_trials` clean delay trials (no distractor), each with a different noise seed
2. Decode the bump center $\varphi(t)$ via population vector during the delay period
3. **Detect oscillations**: compute the FFT of the per-trial bump amplitude; identify the dominant oscillation frequency (see [Bump Amplitude Oscillations](#bump-amplitude-oscillations-in-docs-ring_attractormd))
4. **Filter**: apply a zero-phase low-pass Butterworth filter to each $\varphi(t)$ trajectory at 0.4 × $f_\text{osc}$ (auto-detected) or at the value given by `--filter_cutoff_hz`
5. Compute MSD: $\langle[\varphi(t+\tau) - \varphi(t)]^2\rangle$ averaged over time pairs and trials
6. **Oscillation-corrected fit**: if an oscillation frequency $f_\text{osc}$ was detected, fit the model $\text{MSD}(\tau) = B\tau + C(1-\cos(2\pi f_\text{osc}\tau)) + \text{offset}$ to extract $\hat{B}$; otherwise fall back to a standard linear fit

### Outputs

Generates in `figs/diffusion/<n_nodes>/<params_stem>/<conn_label>/`:
- `diffusion_msd_<band>.png` -- Three-panel figure: MSD vs lag (left, oscillatory-regime shaded), $\hat{B}$ bar chart (centre), amplitude timecourse (right).
- `diffusion_oscillation_spectrum.png` -- Power spectrum of bump amplitude per condition with detected frequency annotated, plus bar chart of dominant period (ms).
- `diffusion_summary.csv` -- `condition_key`, `B_hat_rad2_per_s`, `r_squared`, `n_trials`, `delay_ms`, `amplitude_factor`
- `diffusion_msd_curves.csv` -- MSD curve data: `condition_key`, `lag_s`, `msd_mean`, `msd_sem`, `msd_sd`, `fit_line`
- `diffusion_amplitude.csv` -- `condition_key`, `t_s`, `amp_mean`, `amp_sem`, `survival_frac`, `noise_threshold`
- `diffusion_oscillation.csv` -- `condition_key`, `dominant_freq_hz`, `dominant_period_ms`, `filter_cutoff_hz`

### Examples

```bash
# All conditions, 50 trials each (oscillation auto-detected and corrected)
python -m circuit_model ring-diffusion --no_show

# Compare WT vs alpha7 KO with 20 trials
python -m circuit_model ring-diffusion --conditions WT a7_KO --n_trials 20

# Longer delay for better MSD estimation
python -m circuit_model ring-diffusion --conditions WT a7_KO --delay_ms 5000 --n_trials 30

# Disable oscillation filtering (raw MSD, for comparison)
python -m circuit_model ring-diffusion --conditions WT --filter_cutoff_hz 0
```

---

## `ring-drift-field`

Sweep distractor angular offsets $\Delta\varphi \in [0°, 180°]$ and measure bump displacement to estimate the empirical drift field $\hat{A}(\Delta\varphi)$ (rad/s). Based on the distractor analysis of Seeholzer, Deger & Gerstner (2019, Fig. 7).

```bash
python -m circuit_model ring-drift-field [options]
```

### Drift-Field-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--conditions` | str (list) | all 8 | Conditions to simulate (space-separated) |
| `--n_trials` | int | `50` | Number of trials per condition per offset |
| `--distractor_steps` | float | `10.0` | Angular step size for distractor sweep (degrees) |
| `--distractor_amplitude` | float | `15.0` | Distractor stimulus amplitude as factor of I_ext_pyr baseline (15.0 = 15× baseline) |
| `--distractor_duration_ms` | float | `200.0` | Distractor duration (ms) |
| `--distractor_onset_ms` | float | `1500.0` | Distractor onset after stimulus offset (ms) |
| `--n_workers` | int | `None` | Number of parallel workers (default: min(4, cpu_count)) |
| `--error_band` | str | `"sem"` | Error band type for plots: `sem` or `sd` |

Plus all [common ring parameters](#common-ring-parameters) from `ring-run`.

### Method

For each condition and each distractor offset $\Delta\varphi$:
1. Run `n_trials` trials with the cue at 180° and a distractor at $180° + \Delta\varphi$
2. Measure bump position before and after the distractor via population vector
3. Compute signed displacement $\hat{\varphi}_{\text{post}} - \varphi_0$
4. Normalize by distractor duration $T_D$ to estimate drift velocity: $\hat{A}(\Delta\varphi) = \langle\hat{\varphi}_{\text{post}} - \varphi_0\rangle / T_D$

The resulting curve $\hat{A}(\Delta\varphi)$ is the **distractor susceptibility fingerprint** of the network.

### Outputs

Generates in `figs/drift_field/<n_nodes>/<params_stem>/<conn_label>/`:
- `drift_field_<band>.png` -- $\hat{A}(\Delta\varphi)$ vs $\Delta\varphi$, one colored line per condition with error shading. `<band>` is `sem` or `sd`.
- `drift_field_trials.csv` -- Per-trial raw data: `condition_key`, `offset_deg`, `trial_idx`, `seed`, `displacement_rad`, `pre_amp`, `post_amp`
- `drift_field_summary.csv` -- Aggregated: `condition_key`, `offset_deg`, `A_hat_rad_per_s`, `A_hat_sem`, `A_hat_sd`, `n_trials`, `distractor_amplitude_factor`, `distractor_duration_ms`, `distractor_onset_ms`

### Examples

```bash
# All conditions, 50 trials, 10° steps
python -m circuit_model ring-drift-field --no_show

# Quick test with coarser sweep
python -m circuit_model ring-drift-field --conditions WT a7_KO --n_trials 10 --distractor_steps 30

# Custom distractor parameters
python -m circuit_model ring-drift-field \
    --conditions WT WT_APP a7_KO b2_KO \
    --distractor_amplitude 15.0 --distractor_duration_ms 250 --error_band sd \
    --n_trials 50 --distractor_steps 10
```

---

## `ring-distractor-sweep`

Sweep a 2D grid of distractor angular offset $\Delta\varphi \times$ distractor amplitude (relative to cue), measuring bump drift and collapse probability for a single condition. Protocol: `cue (250 ms) → delay₁ (1000 ms) → distractor (250 ms) → delay₂ (1000 ms)`.

```bash
python -m circuit_model ring-distractor-sweep [options]
```

### Sweep-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--condition` | str | `"WT"` | Experimental condition to simulate |
| `--offsets_deg` | float (list) | `0 5 10 15 20 30 40 60 80 100 130 150 180` | Distractor angular offsets from cue (degrees) |
| `--amp_factors` | float (list) | `0.5 0.75 1.0 1.25 1.5` | Distractor amplitude factors relative to cue |
| `--n_trials` | int | `10` | Number of trials per grid cell |
| `--delay1_ms` | float | `1000.0` | Delay period before distractor (ms) |
| `--delay2_ms` | float | `1000.0` | Delay period after distractor (ms) |
| `--distractor_duration_ms` | float | `250.0` | Distractor duration (ms) |
| `--collapse_threshold` | float | auto | Population-vector amplitude $\hat{A}$ below which the bump is declared collapsed. Auto-detected from `calibration_summary.csv` (run `ring-calibrate` first); falls back to 0.2 with a warning if not found |
| `--n_workers` | int | `None` | Number of parallel workers (default: min(4, cpu_count)) |

Plus all [common ring parameters](#common-ring-parameters) from `ring-run` (network size, amplitude, connectivity, etc.).

### Method

For each grid cell $(\Delta\varphi, \alpha)$ where $\alpha$ is the distractor amplitude factor:

1. Run `n_trials` trials with the cue at 180° and the distractor at $180° + \Delta\varphi$ with amplitude $\alpha \times A_{\text{cue}}$
2. Measure bump position $\hat{\theta}_{\text{before}}$ (50 ms before distractor onset) and $\hat{\theta}_{\text{after}}$ (100 ms after distractor offset) via population vector decoding
3. Compute signed bump shift: $\Delta\hat{\theta} = \hat{\theta}_{\text{after}} - \hat{\theta}_{\text{before}}$ (wrapped to $[-\pi, \pi]$)
4. Declare collapse if $\hat{A}_{\text{after}} < $ `collapse_threshold`

Aggregate across trials: mean drift ± SEM and collapse probability per cell.

For Figure 3 (timecourses), up to 6 representative cells are re-run with a single seed to record the full bump trajectory $\hat{\theta}(t)$.

### Outputs

Generates in `figs/distractor_sweep/<n_nodes>/<params_stem>/<conn_label>/`:

**Figures:**
- `distractor_sweep_drift.png` — 2D heatmap of mean bump shift (degrees). Axes: $\Delta\varphi$ (x) × distractor amplitude (y). Diverging colormap (`RdBu_r`) centred at 0
- `distractor_sweep_collapse.png` — 2D heatmap of collapse probability [0–1]. Sequential colormap (`YlOrRd`). Cells annotated with percentage
- `distractor_sweep_timecourses.png` — Multi-panel figure showing $\hat{\theta}(t)$ for 6 representative (Δφ, amplitude) conditions. Shaded regions mark the cue and distractor windows; dashed horizontal line marks the cue location

**Data:**
- `distractor_sweep_trials.csv` — Per-trial raw data: `offset_deg`, `amp_factor`, `trial_idx`, `displacement_deg`, `pre_amp`, `post_amp`
- `distractor_sweep_summary.csv` — Aggregated per cell: `offset_deg`, `amp_factor`, `n_trials`, `drift_mean_deg`, `drift_sd_deg`, `drift_sem_deg`, `collapse_prob`, `pre_amp_mean`, `post_amp_mean`

### Examples

```bash
# Full 5×5 grid, 50 trials, WT condition (default)
python -m circuit_model ring-distractor-sweep --no_show

# Quick smoke test (3×3 grid, 5 trials)
python -m circuit_model ring-distractor-sweep \
    --n_trials 5 --offsets_deg 0 90 180 --amp_factors 0.5 1.0 1.5 --no_show

# Custom grid with parallel workers
python -m circuit_model ring-distractor-sweep \
    --condition WT \
    --offsets_deg 0 30 60 90 120 150 180 \
    --amp_factors 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
    --n_trials 50 --n_workers 8 --no_show

# Shorter delays (to speed things up for testing)
python -m circuit_model ring-distractor-sweep \
    --delay1_ms 500 --delay2_ms 500 --n_trials 10 --no_show
```

---

## `ring-calibrate`

Sweep a 2D grid of (stimulus_amplitude, w_pyr_pyr_inter) to find parameter combinations that produce a stable memory bump. Estimates a noise floor from no-stimulus baseline trials and outputs diagnostic figures plus a JSON summary with recommended parameters.

```bash
python -m circuit_model ring-calibrate [options]
```

### Calibrate-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--conditions` | str (list) | `WT` | Conditions to calibrate (default: WT only) |
| `--amplitudes` | float (list) | `5 10 15 20 25 30` | Stimulus amplitude factors to sweep |
| `--w_inter_values` | float (list) | `2.0 3.0 4 5.0 6.0` | w_pyr_pyr_inter values to sweep |
| `--n_trials` | int | `50` | Number of trials per grid point |
| `--n_baseline` | int | `100` | Number of no-stimulus baseline trials per w_inter |
| `--noise_percentile` | float | `95` | Percentile of baseline A_hat for noise floor threshold |
| `--n_workers` | int | `None` | Number of parallel workers (default: min(4, cpu_count)) |
| `--error_band` | str | `"sem"` | Error band type for time course plots: `sem` or `sd` |

Plus all [common ring parameters](#common-ring-parameters) from `ring-run`.

### Method

1. **Noise floor estimation**: For each w_inter value, run `n_baseline` trials without stimulus. Decode population vector amplitude (A_hat) and take the specified percentile as the noise floor threshold.
2. **Grid exploration**: For each (amplitude, w_inter) combination, run `n_trials` with the standard WM protocol. Measure A_hat at end of delay, peak PYR rate, angular error.
3. **Success criterion**: A trial is "successful" if A_hat at delay end exceeds the noise floor threshold for that w_inter.
4. **Recommendation**: Select the (amplitude, w_inter) with highest success rate; ties broken by highest mean A_hat. Warning if peak PYR rate > 100 Hz.

### Outputs

Generates in `figs/calibration/<n_nodes>/<params_stem>/<base_conn_label>/`:

**Figures:**
- `noise_floor.png` -- Histogram of baseline A_hat per w_inter, with noise floor threshold
- `heatmap_success_rate.png` -- 2D heatmap: success rate across the grid
- `heatmap_A_hat.png` -- 2D heatmap: mean A_hat across the grid
- `heatmap_peak_pyr.png` -- 2D heatmap: peak PYR firing rate
- `timecourses_<band>.png` -- A_hat time courses for representative grid points
- `scatter_summary.png` -- Mean A_hat vs success rate, colored by peak PYR rate

**Data:**
- `calibration_results.csv` -- Per-trial data: `condition_key`, `amplitude`, `w_inter`, `trial_idx`, `seed`, `A_hat_final`, `peak_pyr_rate`, `center_final_deg`, `error_from_cue_deg`
- `calibration_summary.csv` -- Aggregated per grid point: `condition_key`, `amplitude`, `w_inter`, `success_rate`, `mean_A_hat`, `peak_pyr_rate`, `mean_error_deg`, `noise_threshold`, `n_trials`
- `calibration_recommended.json` -- Best parameters with metadata

### Examples

```bash
# Default calibration (WT, 6x5 grid, 50 trials/point, 100 baseline)
python -m circuit_model ring-calibrate --no_show

# Quick test with small grid
python -m circuit_model ring-calibrate --amplitudes 10 20 --w_inter_values 3.0 4.0 --n_trials 5 --n_baseline 10

# Custom grid with more resolution
python -m circuit_model ring-calibrate \
    --amplitudes 5 8 10 12 15 18 20 25 30 \
    --w_inter_values 2.0 2.5 3.0 3.5 4.0 4.5 5.0 \
    --n_trials 50 --n_baseline 100 --no_show

# Calibrate multiple conditions
python -m circuit_model ring-calibrate --conditions WT a7_KO --amplitudes 10 20 30 --n_trials 20
```

---

## Fixed Protocol Parameters (Ring)

The ring attractor uses a fixed stimulus protocol (constants in `circuit_model/ring/cli.py`):

| Constant | Value | Description |
|----------|-------|-------------|
| `BURN_IN_MS` | 10000 ms | Burn-in period before stimulus |
| `STIM_ONSET_MS` | 10500 ms | Stimulus onset (burn-in + 500 ms) |
| `STIM_DURATION_MS` | 250 ms | Stimulus duration |
| `STIM_CENTER_DEG` | 180 deg | Stimulus angular location |
| `STIM_SIGMA_DEG` | 20 deg | Stimulus spatial width (Gaussian sigma) |

The total simulation time is computed as: `STIM_ONSET_MS + STIM_DURATION_MS + delay_ms` (unless `--total_time_ms` or `--response_onset_ms` override it).
