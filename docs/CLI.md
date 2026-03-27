# CLI Reference

The unified CLI is invoked via `python -m circuit_model <command>`.

```
python -m circuit_model {plot-transfer,run,optimize,study,ring-run,ring-study,ring-oscillation-study,ring-osc-distractor-study,ring-osc-phase-distractor,ring-diffusion,ring-noise-floor,ring-calibrate,ring-asymmetry,ring-burnin-stability,ring-bump-decay-study,ring-optimize} [options]
```

---

## Table of Contents

1. [plot-transfer](#plot-transfer) -- Plot transfer functions for all 4 populations
2. [run](#run) -- Single-circuit simulation with plotting
3. [optimize](#optimize) -- Nevergrad parameter optimization
4. [study](#study) -- Batch study across 8 experimental conditions
5. [ring-run](#ring-run) -- Ring attractor single-condition simulation
6. [ring-study](#ring-study) -- Ring attractor multi-condition comparison
7. [ring-oscillation-study](#ring-oscillation-study) -- Cue-only oscillation analysis in a selected frequency band
8. [ring-osc-distractor-study](#ring-osc-distractor-study) -- Oscillation + distractor study (STFT at cue/distractor nodes + PLV)
9. [ring-osc-phase-distractor](#ring-osc-phase-distractor) -- Phase-dependent distractor study (vary distractor timing relative to oscillation cycle)
10. [ring-diffusion](#ring-diffusion) -- MSD diffusion analysis (Seeholzer et al. 2019)
11. [ring-noise-floor](#ring-noise-floor) -- Noise floor estimation from no-stimulus baseline trials
12. [ring-calibrate](#ring-calibrate) -- 2D parameter calibration (amplitude x w_inter)
13. [ring-asymmetry](#ring-asymmetry) -- Left/right bump asymmetry analysis across conditions and trials
14. [ring-burnin-stability](#ring-burnin-stability) -- Burn-in stationarity analysis via window comparison
15. [ring-bump-decay-study](#ring-bump-decay-study) -- Assess whether a bump is a self-sustained attractor or a decaying transient
16. [ring-optimize](#ring-optimize) -- Joint optimization of CircuitParams + RingParams against ring-level firing rate targets

---

## `plot-transfer`

Plot the Wong-Wang transfer function Φ(I) for all 4 populations on a single axis, allowing direct comparison of their input-output curves.

```bash
python -m circuit_model plot-transfer [options]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--params_json` | str | `""` | Load parameters from JSON file (default: built-in defaults) |
| `--condition` | str | `""` | Apply condition preset (`WT`, `WT_APP`, `a7_KO`, `a7_KO_APP`, `b2_KO`, `b2_KO_APP`, `a5_KO`, `a5_KO_APP`). If `--params_json` is omitted, default project WT/WT_APP fit files are used when available. |
| `--set` | str | `""` | Override parameter values: `name=val,name=val` |
| `--I_min` | float | `-5.0` | Minimum input current to plot |
| `--I_max` | float | `80.0` | Maximum input current to plot |
| `--save_plot` | str | `""` | Explicit output path (overrides auto path) |
| `--no_show` | flag | `False` | Don't display the plot |

### Output path

If `--save_plot` is not given, the figure is saved automatically:
- With `--params_json params/new/WT_1mo_article.json` → `figs/optim/transfer_functions_WT_1mo_article.png`
- With `--condition WT_APP` → `figs/optim/transfer_functions_WT_APP.png`
- Without `--params_json` / `--condition` → `figs/optim/transfer_functions.png`

### Examples

```bash
# Default parameters
python -m circuit_model plot-transfer

# From a fitted parameter file (auto-saved with filename suffix)
python -m circuit_model plot-transfer --params_json params/new/WT_1mo_article.json

# Switch directly by condition preset (uses project default fitted files)
python -m circuit_model plot-transfer --condition WT_APP

# Custom current range
python -m circuit_model plot-transfer --I_min -2 --I_max 40

# Override specific parameters
python -m circuit_model plot-transfer --set "g=0.5,A_pyr=3"

# Save to explicit path without displaying
python -m circuit_model plot-transfer --save_plot figs/my_transfer.png --no_show
```

---

## `run`

Run a single 4-population circuit simulation and visualize the results.

```bash
python -m circuit_model run [options]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--params_json` | str | `""` | Load parameters from JSON file. If omitted, `run` first tries `params/new/ring_firing_rate/WT_1mo_article_ko.json` and falls back to built-in defaults only if that file is missing. |
| `--condition` | str | `""` | Apply condition preset (`WT`, `WT_APP`, `a7_KO`, `a7_KO_APP`, `b2_KO`, `b2_KO_APP`, `a5_KO`, `a5_KO_APP`). If `--params_json` is omitted, default project WT/WT_APP fit files are used when available. |
| `--T_ms` | float | `2500.0` | Simulation duration (ms) |
| `--dt_ms` | float | `0.1` | Integration time step (ms) |
| `--noise_type` | str | `"none"` | Noise type: `none`, `white`, or `ou` (Ornstein-Uhlenbeck) |
| `--tau_noise_ms` | float | `5.0` | OU noise time constant (ms) |
| `--seed` | int | `None` | Random seed for reproducibility |
| `--burn_in_ms` | float | `500.0` | Burn-in period for statistics (ms) |
| `--time_range` | str | `""` | Time range to plot: `start,end` in ms (e.g. `1000,2000`) |
| `--save_plot` | str | `""` | Save plot to file path (e.g. `output.png`) |
| `--no_show` | flag | `False` | Don't display the plot (useful for batch processing) |
| `--unit` | str | `"Hz"` | Rate unit for display: `Hz` |

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

# Switch directly by condition preset (no explicit params_json needed)
python -m circuit_model run --condition WT_APP

# Custom duration with OU noise
python -m circuit_model run --T_ms 5000 --noise_type ou

# With transient current
python -m circuit_model run --enable_transient --trans_start_ms 1000 --trans_duration_ms 500
```

---

## `optimize`

Run Nevergrad optimization to find parameters matching target firing rates. The optimizer can be selected via `--optimizer`; the recommended default is `chaining` (global DE search followed by CMA-ES refinement).

```bash
python -m circuit_model optimize --target_pyr 5 --target_som 10 --target_pv 15 --target_vip 8 [options]
```

### Target Rates (required, unless using `--resume`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--target_pyr` | float | **required** | Target mean firing rate for PYR (in the selected `--unit`, default: Hz) |
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
| `--n_samples` | int | `5000` | Total optimization budget (number of candidate evaluations) |
| `--optimizer` | str | `"de"` | Algorithm — see table below |
| `--top_k` | int | `10` | Keep top K candidates |
| `--early_stop_loss` | float | `1e-4` | Stop early if loss falls below this value |
| `--squared_loss` | flag | `True` | Use MSPE (squared percentage error) instead of MAPE. Default on; pass `--no_squared_loss` to revert to MAPE. MSPE penalises large per-population errors quadratically, preventing the optimizer from tolerating a 30–40% error on one population if the others are exact. |

#### Optimizer choices

| Value | Algorithm | When to use |
|-------|-----------|-------------|
| `de` | TwoPointsDE | Robust global search; good default for a first run |
| `cma` | CMA-ES | Fast local convergence; warm-start from `--resume` after DE has found a good basin |
| `chaining` | DE → CMA-ES | **Recommended.** DE explores globally for `min(n_samples//5, 10000)` steps, then CMA-ES refines for the rest |
| `auto` | NGOpt | Nevergrad auto-selects the best algorithm for the problem size and budget |

With `--optimizer chaining` the DE budget is set automatically to `min(n_samples // 5, 10 000)` — empirically DE converges within ~3 000–5 000 steps on this problem, so this avoids wasting budget on a plateaued DE phase.

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
| `--set` | str | `""` | Override values: `name=val,name=val` (e.g. `--set w_sp=0`) |
| `--show_params` | flag | `False` | Show which parameters are free vs frozen |
| `--no_adapt` | flag | `False` | Disable spike-frequency adaptation: sets `J_adapt_pyr=0` and `J_adapt_som=0` and freezes them. |
| `--turing_weight` | float | `0.0` | Weight of two-sided Turing bistability penalty (0 = disabled). Penalises rest-state gain above `1 − margin` AND cue-state gain below `1 + margin`. |
| `--turing_margin` | float | `0.05` | Safety margin around the Turing threshold. |
| `--turing_w_inter_ref` | float | `10.0` | Reference inter-node weight (proxy for `w_pyr_pyr_inter` in single-node mode). |
| `--turing_cue_scale` | float | `5.0` | Multiplier on `I0_pyr` used to approximate the cue operating point. |

### I/O Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save_best_json` | str | `"best_params.json"` | Save best parameters to JSON file |
| `--log_file` | str | `"results_log.jsonl"` | Log results to JSONL file |
| `--log_interval` | int | `500` | Log every N steps |
| `--resume` | flag | `False` | Resume from `--save_best_json`, loading targets from the last log entry and appending to the log. The optimizer is warm-started from the saved parameters. |
| `--n_workers` | int | `None` | Parallel workers (auto if None) |
| `--unit` | str | `"Hz"` | Rate unit for display: `Hz` |

### Weight bounds

All synaptic weights have a **relative floor** of `max(0.1, 5% × default)`. This prevents high-default connections (e.g. `w_ep = 42.5`, `w_pp = 105`) from collapsing to near-zero — a flat absolute floor of 0.1 would be functionally zero for those weights and allows the optimizer to silently decouple populations.

### Examples

```bash
# Recommended: chaining optimizer with KO targets
python -m circuit_model optimize \
    --target_pyr 4.143 --target_som 3.423 --target_pv 2.079 --target_vip 1.933 \
    --target_alpha7_ko_pyr 3.513 --target_beta2_ko_pyr 4.8 --target_alpha5_ko_pyr 3.79 \
    --optimizer chaining \
    --n_samples 50000 --n_workers 4 \
    --save_best_json params/new/WT_1mo.json \
    --log_file figs/optim/1mo/log.jsonl

# Resume with CMA-ES to refine from a previous run's best params
python -m circuit_model optimize \
    --optimizer cma --n_samples 30000 --n_workers 4 \
    --save_best_json params/new/WT_1mo.json \
    --log_file figs/optim/1mo/log.jsonl \
    --resume

# With frozen parameters and verbose param listing
python -m circuit_model optimize \
    --target_pyr 5 --target_som 10 --target_pv 15 --target_vip 8 \
    --target_alpha7_ko_pyr 7 --target_beta2_ko_pyr 6 \
    --freeze "tau_s,g_gaba_base" --show_params --n_samples 10000
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
| `WT_APP` | Wild Type + APP | Uses WT_APP fitted parameter family; receptors stay active (act = 1.0) |
| `a7_KO` | alpha7 KO | alpha7 = 0, g_alpha7 = 0 |
| `a7_KO_APP` | alpha7 KO + APP background | WT_APP family + alpha7 = 0, g_alpha7 = 0 |
| `b2_KO` | beta2 KO | beta2 = 0 |
| `b2_KO_APP` | beta2 KO + APP background | WT_APP family + beta2 = 0 |
| `a5_KO` | alpha5 KO | alpha5 = 0 |
| `a5_KO_APP` | alpha5 KO + APP background | WT_APP family + alpha5 = 0 |

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
| `--unit` | str | `"Hz"` | Rate unit for display |

### Examples

```bash
# Default study (50 runs per condition, white noise)
python -m circuit_model study

# Quick test with fixed receptor values
python -m circuit_model study --n_runs 10 --fixed_receptor_values --no_show

# OU noise study
python -m circuit_model study --noise_type ou --tau_noise_ms 10
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
| `--n_nodes` | int | from ring params JSON or `128` | Number of nodes on the ring |
| `--params_json` | str | `""` | Load local circuit parameters from JSON file. If omitted, defaults to `params/new/ring_firing_rate/WT_1mo_article.json` (WT) and `WT_APP_1mo_article.json` (APP). |
| `--amplitude` | float | `10.0` | Stimulus amplitude as factor of I_ext_pyr baseline (0.1 = 10% of baseline current) |
| `--delay_ms` | float | `5000.0` | Delay period duration (ms) |
| `--seed` | int | `442` | Random seed for reproducibility |
| `--no_show` | flag | `False` | Don't display plots |
| `--total_time_ms` | float | `None` | Total simulation time (overrides automatic timing) |
| `--record_dt_ms` | float | `5.0` | Recording time step (ms). Only every record_dt_ms the state is stored. Lower values use more memory |
| `--snapshot_anim_fps` | int | `30` | FPS for snapshot evolution animation |
| `--snapshot_anim_step_ms` | float | `2.0` | Time step between animation frames (ms) |
| `--quality_high` | flag | `False` | Use moderately higher-quality animation rendering (higher DPI + AV1 quality; up to ~2× slower encoding) |

#### Noise Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sigma_noise` | float | from params (default `0.3`) | Relative noise amplitude. The std of the current noise injected into each PYR node equals `sigma_noise × I_ext_pyr` (nA). Noise enters before the transfer function, so its effect on firing rate is naturally gated by the transfer function slope. Set to `0` to disable noise. |

#### Connectivity Parameters

Ring connectivity parameters are loaded by default from `params/new/ring_firing_rate/WT_1mo_article_ko_ring.json`. Explicit CLI values always override the file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sigma_pyr_deg` | float | from ring params JSON or `30.0` | PYR→PYR connectivity width (degrees) |
| `--w_pyr_pyr_inter` | float (one or more) | from ring params JSON or `8.0` | Total PYR→PYR coupling strength. Multi-condition commands accept one value per condition (e.g. `--w_pyr_pyr_inter 8.0 7.5` for WT and WT_APP). A single value is broadcast to all conditions. |
| `--w_pv_global` | float | from ring params JSON or `10.0` | Total PV→PYR global inhibition strength (uniform) |

**PYR→PYR**: Row-sum normalized Gaussian. `w_pyr_pyr_inter` controls total coupling strength.

#### Response Transient Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--response_onset_ms` | float | `0.0` | Response transient onset after delay end (ms). 0 = disabled |
| `--response_duration_ms` | float | `500.0` | Duration of response transient (ms) |
| `--response_factor` | float | `0.5` | Response transient amplitude as fraction of I0 (+50% of baseline) |
| `--post_response_ms` | float | `3000.0` | Simulation time after response transient ends (ms) |

### Outputs

Generates in `figs/ring/run/cue/amp<N>/<condition>/`:
- `dashboard.png` -- Activity heatmap, snapshots, firing rate traces
- `snapshot_evolution.mp4` -- Ring snapshot animation (requires ffmpeg)
- `bump_metrics_over_time.png` -- Bump center, width, amplitude over time
- `connectivity_matrices.png` -- PYR-PYR and PV-PYR connectivity matrices
- `experiment_config.txt` -- Full parameter summary for reproducibility

`amp<N>` encodes the stimulus amplitude factor (e.g. `amp0.1` for `--amplitude 0.1`).

### Examples

```bash
# Wild type, default amplitude
python -m circuit_model ring-run --condition WT

# Smaller network, longer delay
python -m circuit_model ring-run --condition WT --n_nodes 64 --delay_ms 8000

# Alpha7 KO, amplitude 0.3×
python -m circuit_model ring-run --condition a7_KO --amplitude 0.3 --delay_ms 5000

# Disable noise
python -m circuit_model ring-run --condition WT --sigma_noise 0

# Stronger noise
python -m circuit_model ring-run --condition WT --sigma_noise 0.5

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
| `--amplitudes` | float (list) | `[30]` | Multiple stimulus amplitude factors (× I_ext_pyr) to compare |
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

## `ring-oscillation-study`

Run cue-only ring simulations across conditions and amplitudes, extract dominant oscillation trajectories in a selected frequency band (default `2-12 Hz`), and generate distribution/heatmap outputs.

```bash
python -m circuit_model ring-oscillation-study [options]
```

### Oscillation-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--conditions` | str (list) | `WT WT_APP` | Conditions to simulate (space-separated) |
| `--amplitudes` | float (list) | uses `--amplitude` | Cue amplitude factors (x `I_ext_pyr`) |
| `--n_trials` | int | `50` | Trials per condition x amplitude |
| `--n_workers` | int | `None` | Parallel workers (auto if `None`) |
| `--osc_skip_ms` | float | `200.0` | Initial delay segment skipped before oscillation analysis |
| `--min_freq_hz` | float | `2.0` | Lower bound of frequency band used for dominant-frequency search |
| `--max_freq_hz` | float | `12.0` | Upper bound of frequency band used for dominant-frequency search |
| `--tf_window_s` | float | `1.0` | STFT window length (seconds) |
| `--tf_overlap` | float | `0.8` | STFT overlap fraction in `[0,1)` |
| `--sample_time_frac` | float | `0.75` | Delay fraction used for single-timepoint summary metrics |

Plus all [common ring parameters](#common-ring-parameters) from `ring-run`.

### Outputs

Generates in `figs/ring/oscillation/<params_stem>/<conn_label>/`:
- `oscillation_trial_summary.csv` -- Trial-level summary metrics (`freq_mean_hz`, `power_mean`, `freq_sample_hz`, `power_sample`)
- `oscillation_dominant_timecourse.csv` -- Dominant trajectory over time (`dominant_freq_hz`, `dominant_power`, `cue_rate_hz`) per trial
- `oscillation_stats.csv` -- Pairwise two-sided Mann-Whitney U tests across conditions (per amplitude)
- `amp<N>/violin_power_mean.png` -- Violin plot of delay-averaged dominant power by condition
- `amp<N>/violin_power_sample.png` -- Violin plot of single-timepoint dominant power by condition
- `amp<N>/heatmap_<COND>.png` -- High-quality smooth spectrogram-style heatmap (`frequency x time`) per condition
- `amp<N>/oscillation_vs_time.png` -- Oscillation metrics over analyzed delay time (power + picked frequency + cue-node PYR rate, conditions compared)

For full computation details (detrending, STFT, dominant-band metrics), see `docs/oscillation_analysis.md`.

### Examples

```bash
# Default WT vs WT_APP analysis in the 2-12 Hz band
python -m circuit_model ring-oscillation-study --no_show

# Multi-amplitude comparison with more trials
python -m circuit_model ring-oscillation-study \
    --conditions WT WT_APP a7_KO_APP \
    --amplitudes 10 15 20 25 30 \
    --n_trials 100 --n_workers 8 --no_show

# Narrow the band to theta-like oscillations
python -m circuit_model ring-oscillation-study \
    --min_freq_hz 4 --max_freq_hz 10 \
    --tf_window_s 0.6 --tf_overlap 0.85 --no_show
```

---

## `ring-osc-distractor-study`

Run cue + distractor ring simulations sweeping cue amplitude, distractor angular offset, and distractor amplitude factor. Measures oscillatory power at the cue and distractor nodes (via STFT) and their phase synchrony (PLV — Phase Locking Value) across the full delay period.

```bash
python -m circuit_model ring-osc-distractor-study [options]
```

### Distractor-Oscillation-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--conditions` | str (list) | `WT` | Conditions to simulate (space-separated) |
| `--amplitudes` | float (list) | `--amplitude` | Cue amplitude factors (× I\_ext\_pyr). If omitted, uses `--amplitude`. |
| `--distractor_factors` | float (list) | `0.75 1.0` | Distractor amplitude as fraction of cue amplitude |
| `--offsets_deg` | float (list) | `30 70 90 120 170` | Distractor angular offsets from cue (degrees) |
| `--delay1_ms` | float | `1500.0` | Post-cue, pre-distractor delay (ms) |
| `--distractor_duration_ms` | float | `200.0` | Duration of distractor stimulus (ms) |
| `--delay2_ms` | float | `3000.0` | Post-distractor delay until trial end (ms) |
| `--n_trials` | int | `10` | Trials per (condition × amplitude × factor × offset) |
| `--n_workers` | int | `None` | Parallel workers (default: auto) |
| `--min_freq_hz` | float | `2.0` | Lower frequency bound for STFT and PLV bandpass (Hz) |
| `--max_freq_hz` | float | `12.0` | Upper frequency bound for STFT and PLV bandpass (Hz) |
| `--tf_window_s` | float | `1.0` | STFT / PLV sliding window length (s) |
| `--tf_overlap` | float | `0.8` | STFT / PLV window overlap fraction [0, 1) |
| `--no_cache` | flag | off | Force re-simulation even if cached results exist |

Plus all [common ring parameters](#common-ring-parameters) from `ring-run`.

### Protocol Timeline

```
[burn-in 10 s] → [pre-cue 0.5 s] → [cue 0.25 s] → [delay1] → [distractor] → [delay2]
                                                       ↑              ↑             ↑
                                                   1500 ms        200 ms        3000 ms  (defaults)
```

- **Cue**: Gaussian spatial profile (sigma = 18°) at 180°, amplitude swept via `--amplitudes`.
- **Distractor**: Same sigma, amplitude = `distractor_factor × cue_amplitude`, centered at `180° + offset_deg`.
- **Control**: Each amplitude also runs without a distractor (offset = None) for baseline comparison.

### Metrics

#### Per-node STFT
Reuses `compute_oscillation_band_timecourse` applied independently to the PYR firing rate at:
- **Cue node**: node closest to 180°
- **Distractor node**: node closest to `180° + offset_deg`

Returns dominant frequency and power per STFT time bin, over the full post-cue window (delay1 + distractor + delay2 in one contiguous STFT to avoid boundary artefacts).

#### Phase Locking Value (PLV)
New metric computed by `compute_plv_timecourse`:
1. Bandpass-filter both node signals in [min\_freq\_hz, max\_freq\_hz] (zero-phase Butterworth order 4)
2. Extract instantaneous phases φ₁(t), φ₂(t) via Hilbert transform
3. Sliding window: `PLV(t) = |mean(exp(i·(φ₁ − φ₂)))|` over windows matching the STFT bin grid

PLV = 0 means no phase coupling; PLV = 1 means perfect phase locking.

**Time axis**: all timecourses are returned in "seconds since cue offset". Distractor onset = `delay1_ms / 1000.0`. Plots align x-axis to distractor onset (t = 0).

### Output Directory

```
figs/ring/osc_distractor/{network_label}/{condition_key}/factor{F}/
```

### Outputs

| File | Description |
|------|-------------|
| `osc_distractor_trials.csv` | Trial-level summary (cue/dist freq median, power median, PLV median in delay2) |
| `.osc_dist_cache_{key}.pkl` | Cached raw simulation results (pickle) |
| `factor{F}/osc_distractor_timecourses_amp{X}.png` | 3-row timecourse figure per amplitude: cue node power, distractor node power, PLV — one colored line per offset angle; dashed black = no-distractor control |
| `factor{F}/osc_distractor_spectrograms_amp{X}_offset{Y}.png` | 2-column STFT heatmap (cue node \| distractor node) with dominant frequency overlay and distractor epoch markers |
| `factor{F}/osc_distractor_amp_sweep.png` | Connected-dot amplitude sweep: x = cue amplitude, y = mean post-distractor PLV, one line per offset angle |

### Examples

```bash
# Default run: WT, all offsets, both distractor factors
python -m circuit_model ring-osc-distractor-study --no_show

# Single amplitude and offset for a quick look
python -m circuit_model ring-osc-distractor-study \
    --amplitudes 4 --offsets_deg 90 --distractor_factors 1.0 \
    --n_trials 20 --conditions WT --no_show

# Full amplitude sweep with more trials
python -m circuit_model ring-osc-distractor-study \
    --amplitudes 1 2 4 6 8 10 \
    --n_trials 30 --n_workers 8 \
    --conditions WT WT_APP --no_show
```

---

## `ring-osc-phase-distractor`

Phase-dependent distractor experiment. Runs the **same burn-in and cue simulation** (fixed seed, deterministic trajectory) for every trial, then injects a distractor at different points in the ongoing oscillation cycle. The timing offset is expressed in units of **π radians** of the oscillation, so a sweep from 0 to 2π covers exactly one full oscillation cycle.

```bash
python -m circuit_model ring-osc-phase-distractor [options]
```

### Phase-Distractor-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--conditions` | str (list) | `WT` | Conditions to simulate |
| `--amplitudes` | float (list) | `--amplitude` | Cue amplitude factors (× I\_ext\_pyr) |
| `--distractor_factors` | float (list) | `1.0` | Distractor amplitude as fraction of cue amplitude |
| `--offsets_deg` | float (list) | `90.0` | Distractor angular offsets from cue (degrees) |
| `--delay1_base_ms` | float | `500.0` | Base delay between cue offset and distractor onset (ms). Actual delay1 = `delay1_base + phase_pi × T_osc / 2` |
| `--distractor_duration_ms` | float | `200.0` | Duration of distractor stimulus (ms) |
| `--delay2_ms` | float | `2000.0` | Post-distractor delay (ms) |
| `--n_phase_sweep` | int | `16` | Number of equally-spaced phase values in [0, 2π) for the continuous sweep |
| `--osc_freq_hz` | float | `5.0` | Fallback oscillation frequency if auto-detection fails (Hz) |
| `--n_trials` | int | `10` | Distractor trials per (condition × amplitude × factor × offset × phase) |
| `--n_workers` | int | `None` | Parallel workers (default: auto) |
| `--min_freq_hz` | float | `2.0` | Lower frequency bound for STFT and PLV bandpass (Hz) |
| `--max_freq_hz` | float | `12.0` | Upper frequency bound for STFT and PLV bandpass (Hz) |
| `--tf_window_s` | float | `1.0` | STFT / PLV sliding window length (s) |
| `--tf_overlap` | float | `0.8` | STFT / PLV window overlap fraction [0, 1) |
| `--no_cache` | flag | off | Force re-simulation even if cached results exist |

Plus all [common ring parameters](#common-ring-parameters) from `ring-run`.

### Protocol Timeline

```
[burn-in 10 s] → [pre-cue 0.5 s] → [cue 0.25 s] → [delay1(φ)] → [distractor] → [delay2]
                                                          ↑               ↑            ↑
                                               delay1_base + φ·T/2      200 ms      2000 ms
                                               (same deterministic trajectory up to this point)
```

The oscillation frequency `T` (and thus period `T_osc = 1000/f` ms) is **auto-detected** from a reference no-distractor simulation before the phase grid is built. For each phase value `φ` (in units of π):

```
delay1(φ) = delay1_base_ms + φ × T_osc / 2
```

So `φ = 0` → distractor at `delay1_base`; `φ = 1` (π) → distractor half a period later; `φ = 2` (2π) → same as `φ = 0`.

### Identical Pre-Distractor State Guarantee

For a given `phase_pi` value, the network state at distractor onset is **deterministic** — all `n_trials` distractor simulations start from the exact same state, because:

1. The burn-in uses the same fixed seed for all phase values.
2. Each pre-distractor simulation (burn-in → cue → delay1) uses the same fixed seed.
3. Since the simulation is deterministic given the seed and initial state, the trajectory is identical up to the respective `delay1` end point.

Only the post-distractor noise realisation differs across trials.

### Metrics

Identical to `ring-osc-distractor-study`: STFT-based dominant power at the cue node and distractor node, and PLV between them. Averaged over the post-distractor delay2 window to produce scalar summary metrics as a function of phase.

### Output Directory

```
figs/ring/osc_phase_distractor/{network_label}/{condition_key}/factor{F}/amp{X}/offset{Y}/
```

### Outputs

| File | Description |
|------|-------------|
| `osc_phase_trials.csv` | Trial-level CSV: condition, amplitude, factor, offset, phase\_pi, PLV / cue power / dist power mean over delay₂ |
| `.osc_phase_cache_{key}.pkl` | Pickle cache of raw trial results |
| `phase_plv_4panel.png` | 2×2 grid of PLV timecourses for phases 0, π/2, π, 3π/2 — each panel shows mean ± SD; black dashed = no-distractor control |
| `phase_cue_power_4panel.png` | Same layout for cue node dominant power |
| `phase_dist_power_4panel.png` | Same layout for distractor node dominant power |
| `phase_sweep.png` | 3-row summary: PLV / cue power / dist power (mean over delay₂) vs. continuous phase (0 to 2π), with SEM bands and no-distractor baseline |
| `phase_polar.png` | Polar rose version of `phase_sweep.png`; one subplot per metric |
| `phase_heatmap_plv.png` | Phase × time heatmap of mean PLV (phase on y-axis, time relative to distractor on x-axis) |
| `phase_heatmap_cue_power.png` | Same for cue node power |
| `phase_heatmap_dist_power.png` | Same for distractor node power |

### Examples

```bash
# Default: WT, 90° offset, factor 1.0, 16-phase sweep
python -m circuit_model ring-osc-phase-distractor --no_show

# Higher-resolution phase sweep, 20 trials
python -m circuit_model ring-osc-phase-distractor \
    --amplitude 35 --offsets_deg 90 \
    --n_phase_sweep 24 --n_trials 20 --no_show

# Compare two conditions and two distractor offsets
python -m circuit_model ring-osc-phase-distractor \
    --conditions WT WT_APP \
    --offsets_deg 90 170 \
    --distractor_factors 0.75 1.0 \
    --n_phase_sweep 16 --n_trials 15 \
    --n_workers 8 --no_show
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

## `ring-noise-floor`

Run no-stimulus baseline trials and compute a noise floor threshold as the Nth percentile of bump amplitude under spontaneous noise. Saves `baseline_A_hat.csv` which is consumed automatically by `ring-calibrate`. Run this command first when you want custom baseline parameters (trial count, percentile, etc.).

```bash
python -m circuit_model ring-noise-floor [options]
```

### Noise Floor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--conditions` | str (list) | `WT` | Conditions to run (default: WT only) |
| `--w_inter_values` | float (list) | `2.0 3.0 4.0 5.0 6.0` | w_pyr_pyr_inter values for baseline sweep |
| `--n_baseline` | int | `100` | Number of no-stimulus trials per w_inter |
| `--noise_percentile` | float | `95` | Percentile of baseline A_hat used as threshold |
| `--n_workers` | int | `None` | Number of parallel workers (default: min(4, cpu_count)) |
| `--batch_chunk_size` | int | `50` | Max trials per simulation batch chunk |
| `--no_cache` | flag | `False` | Ignore existing baseline cache and recompute from scratch |
| `--replot_only` | flag | `False` | Regenerate noise floor plots from cached `baseline_A_hat.csv` without re-simulating |

Plus all [common ring parameters](#common-ring-parameters) from `ring-run`.

### Method

For each (condition, w_inter) combination, run `n_baseline` trials without any stimulus. Decode population-vector amplitude (A_hat) at every recorded time step and at the end of the delay. The threshold is set at the specified percentile of all A_hat values across trials. By default, existing cache is reused per condition and per `w_inter`, now accounting for cached trial count: if you request more trials, only the missing trials are simulated, appended to cache, and thresholds are recomputed from the merged old+new data (equivalent to a weighted update by sample count). Use `--no_cache` to force full recompute. Results are saved to `baseline_A_hat.csv` and used by `ring-calibrate` as the success criterion.

### Outputs

Generates in `figs/calibration/<n_nodes>/<params_stem>/<base_conn_label>/`:

**Figures:**
- `<cond_key>/noise_floor.png` -- Histogram of baseline A_hat per w_inter with threshold line
- `noise_summary.png` -- Cross-condition noise threshold summary

**Data:**
- `<cond_key>/baseline_A_hat.csv` -- Raw A_hat values: `condition_key`, `w_inter`, `a_hat_value`

### Examples

```bash
# Default noise floor (WT only, default w_inter values)
python -m circuit_model ring-noise-floor --no_show

# Custom baseline for multiple conditions
python -m circuit_model ring-noise-floor \
    --conditions WT a7_KO --w_inter_values 2.0 3.0 4.0 5.0 \
    --n_baseline 200 --noise_percentile 99 --no_show

# Replot from existing cache without re-running
python -m circuit_model ring-noise-floor --replot_only --no_show
```

---

## `ring-calibrate`

Sweep a 2D grid of (stimulus_amplitude, w_pyr_pyr_inter) to find parameter combinations that produce a stable memory bump. Uses a pre-computed noise floor as the success criterion — if `baseline_A_hat.csv` is not found, `ring-noise-floor` is automatically run first with default parameters (n_baseline=100, noise_percentile=95). Run `ring-noise-floor` explicitly beforehand to customise the baseline.

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
| `--noise_percentile` | float | `95` | Percentile applied when reading cached baseline A_hat data |
| `--n_workers` | int | `None` | Number of parallel workers (default: min(4, cpu_count)) |
| `--error_band` | str | `"sem"` | Error band type for time course plots: `sem` or `sd` |
| `--no_cache` | flag | `False` | Ignore existing grid CSV cache and recompute from scratch |
| `--batch_chunk_size` | int | `50` | Max trials per simulation batch chunk to limit peak RAM |

Plus all [common ring parameters](#common-ring-parameters) from `ring-run`.

### Method

1. **Noise floor** (prerequisite): Loaded from `baseline_A_hat.csv` produced by `ring-noise-floor`. If missing, auto-runs noise floor with default parameters before continuing.
2. **Grid exploration**: For each (amplitude, w_inter) combination, run `n_trials` with the standard WM protocol. Measure A_hat at end of delay, peak PYR rate, angular error.
3. **Success criterion**: A trial is "successful" if A_hat at delay end exceeds the noise floor threshold for that w_inter.
4. **Recommendation**: Select the (amplitude, w_inter) with highest success rate; ties broken by highest mean A_hat. Warning if peak PYR rate > 100 Hz.

### Outputs

Generates in `figs/calibration/<n_nodes>/<params_stem>/<base_conn_label>/`:

**Figures:**
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
# Default calibration (WT, 10x5 grid, 50 trials/point — runs noise floor first if needed)
python -m circuit_model ring-calibrate --no_show

# Run noise floor explicitly first, then calibrate
python -m circuit_model ring-noise-floor --no_show
python -m circuit_model ring-calibrate --no_show

# Quick test with small grid
python -m circuit_model ring-calibrate --amplitudes 10 20 --w_inter_values 3.0 4.0 --n_trials 5

# Custom grid with more resolution
python -m circuit_model ring-calibrate \
    --amplitudes 5 8 10 12 15 18 20 25 30 \
    --w_inter_values 2.0 2.5 3.0 3.5 4.0 4.5 5.0 \
    --n_trials 50 --no_show

# Calibrate multiple conditions
python -m circuit_model ring-calibrate --conditions WT a7_KO --amplitudes 10 20 30 --n_trials 20
```

---

## `ring-asymmetry`

Analyse the left/right asymmetry of the activity bump across multiple trials and conditions. Each trial receives a unique noisy settling period before the cue so that the pre-cue spontaneous state varies across trials. The experiment tests whether asymmetry is balanced (zero mean) and whether pre-cue asymmetry predicts delay asymmetry, and produces full visualisations for the worst-case trial per condition.

```bash
python -m circuit_model ring-asymmetry [options]
```

### Asymmetry-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--conditions` | str (list) | `WT WT_APP a7_KO_APP` | Conditions to analyse (space-separated) |
| `--n_trials` | int | `100` | Number of trials per condition |
| `--n_workers` | int | `None` | Number of parallel workers (default: auto) |
| `--random_cue_location` | flag | off | Draw a uniformly random cue angle in [0°, 360°) per trial (inherently balanced, skips balance correction) |
| `--no_cue_balance` | flag | off | Disable the automatic balance correction (even N → between nodes; odd N → on nearest node). Leaves the cue at raw 180°, which for even N creates a structural bias of −1/(N−1). |
| `--correct_asymmetry` | flag | on | Use amplitude-weighted normalized asymmetry in each window: $\sum A(t)\,\mathrm{Amp}(t) / \sum \mathrm{Amp}(t)$ |
| `--no_correct_asymmetry` | flag | off | Disable amplitude-based asymmetry correction and use raw asymmetry index |

Plus all [common ring parameters](#common-ring-parameters) from `ring-run`.

### Method

The asymmetry index is defined as:

$$\text{asymmetry} = \frac{\sum_{\text{right}} r_i - \sum_{\text{left}} r_i}{\sum_{\text{right}} r_i + \sum_{\text{left}} r_i} \in [-1, 1]$$

where "left" and "right" are nodes with signed angular offset < 0 or > 0 relative to the cue location. A value of −1 means all activity is on the left; +1 means all activity is on the right; 0 means perfectly symmetric.

With the default correction enabled, pre-cue and delay asymmetry are computed as amplitude-weighted normalized means:

$$a_{\text{window,corr}} = \frac{\sum_{t \in \mathcal{T}_{\text{window}}} A(t)\,\mathrm{Amp}(t)}{\sum_{t \in \mathcal{T}_{\text{window}}} \mathrm{Amp}(t)}$$

This down-weights time points where the bump is weak and normalizes by total bump strength, making values more comparable across conditions.

**Trial design** — each trial is fully independent:
1. **Per-trial burn-in**: each trial starts from zero initial conditions and runs `ASYM_SETTLING_MS` (6000 ms) of noisy spontaneous activity with its own unique seed, producing fully uncorrelated pre-cue states across trials
2. **Pre-cue window**: asymmetry measured over the last `ASYM_PRE_CUE_WINDOW_MS` (500 ms) of the burn-in period
3. **Cue + delay**: standard working-memory protocol with the specified `--delay_ms`
4. **Delay asymmetry**: asymmetry measured over the delay period (after the initial transient)

All trials are run in parallel using `ProcessPoolExecutor`.

#### Balance correction and structural pre-cue bias

The asymmetry index excludes the node at offset = 0 (cue position) and counts offset = −180° (antipodal) as "left". For even N with the cue on a node, left has one more node than right → bias = −1/(N−1).

The **balance correction** (on by default) fixes this:
- **Even N**: cue placed at `nearest_node + step/2` (halfway between two nodes) → left = right = N/2.
- **Odd N**: cue snapped to nearest node (antipodal never on a node → always balanced).

A diagnostic is printed whenever N is even. Use `--no_cue_balance` to revert to raw 180° (e.g. for comparison with old results).

**`--random_cue_location`**: continuous random angle per trial → inherently balanced (left = right = N/2), balance correction skipped. Plot titles show `cue@random` vs `cue@181.41° (balanced)` for the default.

### Outputs

Generates in `figs/asymmetry/<n_nodes>/<params_stem>/<conn_label>/amp<N>_<mode>/` where `<mode>` is `corrected` or `uncorrected`:

**Summary figures** (title includes cue mode and asymmetry mode, e.g. `cue@180°`, `cue@random`, `asymmetry corrected`):
- `asymmetry_distribution.png` -- Violin + jittered strip plots of pre-cue and delay asymmetry per condition
- `asymmetry_correlation.png` -- Scatter plot of pre-cue vs delay asymmetry per condition with Pearson *r* annotated
- `asymmetry_summary.png` -- Three-panel bar chart: mean delay asymmetry ± SEM, fraction of rightward trials, mean |asymmetry| ± SEM

**Worst-case per condition** (trial with highest |delay asymmetry|), in `worst_case/<cond>/`:
- `dashboard.png` -- Full activity heatmap and firing rate traces
- `bump_metrics.png` -- Bump center, width, amplitude, and asymmetry over time
- `animation.mp4` -- Ring snapshot animation (if ffmpeg is available)

**Data:**
- `asymmetry_trials.csv` -- Per-trial raw data: `condition`, `trial_idx`, `seed`, `cue_deg`, `pre_cue_asym`, `delay_asym`, `delay_ms`, `amplitude`, `random_cue` (0/1), `balance_cue` (0/1), `correct_asymmetry` (0/1)

### Examples

```bash
# Default: WT vs WT_APP vs a7_KO_APP, 100 trials each, balanced cue (on by default)
python -m circuit_model ring-asymmetry --no_show

# Disable balance correction (raw 180°, reintroduces structural bias for even N)
python -m circuit_model ring-asymmetry --no_cue_balance --no_show

# Disable amplitude correction (legacy raw asymmetry index)
python -m circuit_model ring-asymmetry --no_correct_asymmetry --no_show

# Random cue location (inherently balanced, useful for comparison)
python -m circuit_model ring-asymmetry --random_cue_location --no_show

# Subset of conditions with fewer trials
python -m circuit_model ring-asymmetry --conditions WT WT_APP --n_trials 50 --no_show

# Longer delay to assess asymmetry stability
python -m circuit_model ring-asymmetry --delay_ms 8000 --n_trials 100 --n_workers 8 --no_show

# All conditions with custom connectivity
python -m circuit_model ring-asymmetry \
    --conditions WT WT_APP a7_KO a7_KO_APP b2_KO b2_KO_APP \
    --n_trials 100 --n_workers 8 \
    --w_pyr_pyr_inter 7 --sigma_pyr_deg 30 --w_pv_global 10 --no_show
```

---

## `ring-burnin-stability`

Test whether a noisy spontaneous burn-in period has reached stationarity. Runs `n_trials` independent simulations from zero initial conditions, divides each into windows of `period_ms`, and compares window distributions with Kruskal-Wallis and pairwise Mann-Whitney U tests. See [§20 of ring_experiments.md](ring_experiments.md#20-burn-in-stationarity-analysis) for full details.

```
python -m circuit_model ring-burnin-stability [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_trials` | 100 | Number of independent noisy trials |
| `--burnin_ms` | 10000.0 | Total burn-in duration in ms |
| `--period_ms` | 1000.0 | Duration of each comparison window in ms |
| `--ref_deg` | 0.0 | Fixed reference angle (degrees) for asymmetry |
| `--conditions` | `WT` | Conditions to test (space-separated) |
| `--n_workers` | auto | Number of parallel worker processes |
| `--seed` | 42 | Base random seed |
| `--no_show` | — | Suppress interactive plot display |
| `--n_nodes` | from ring params JSON or `128` | Number of ring nodes |
| `--w_pyr_pyr_inter` | from ring params JSON or `8.0` | PYR→PYR inter-node coupling |
| `--sigma_pyr_deg` | from ring params JSON or `30.0` | PYR→PYR connectivity width (degrees) |
| `--w_pv_global` | from ring params JSON or `10.0` | PV→PYR global inhibition strength |
| `--params_json` | — | Load local circuit parameters from JSON |

### Outputs

```
figs/burnin_stability/{n_nodes}/{connectivity_label}/
├── burnin_stability_trials.csv      # per-trial, per-window metrics
├── burnin_stability_summary.csv     # Kruskal-Wallis H and p per condition/metric
└── burnin_stability_{cond}.png      # box plots + adjacent-window MWU brackets
```

### Examples

```bash
# Default: WT, 100 trials, 10 s burn-in split into 10 × 1000 ms windows
python -m circuit_model ring-burnin-stability --no_show

# Quick smoke test: 10 trials, 3 s burn-in
python -m circuit_model ring-burnin-stability --n_trials 10 --burnin_ms 3000 --no_show

# Multiple conditions, custom window size
python -m circuit_model ring-burnin-stability \
    --conditions WT WT_APP a7_KO_APP \
    --burnin_ms 10000 --period_ms 500 \
    --n_trials 100 --n_workers 8 --no_show

# Different asymmetry reference angle
python -m circuit_model ring-burnin-stability --ref_deg 180 --no_show
```

---

## `ring-bump-decay-study`

Test whether a post-cue bump is a self-sustained attractor state or a decaying transient. Runs multiple trials per condition and amplitude, records the bump amplitude timecourse over a long delay, and computes whether the amplitude grows, stays flat, or decays relative to a reference window shortly after cue offset.

All oscillatory noise is removed by averaging the amplitude within non-overlapping `--window_ms` bins (default 500 ms). Each trial is normalised by the mean amplitude in the reference bin (the bin whose centre is closest to `cue_offset + ref_offset_ms`).

```
python -m circuit_model ring-bump-decay-study [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--conditions` | `WT WT_APP` | Conditions to simulate (space-separated) |
| `--amplitudes` | `5 10 15 20 25` | Cue amplitude factors (× I_ext_pyr) |
| `--delay_ms` | `10000.0` | Delay duration after cue offset (ms) |
| `--window_ms` | `500.0` | Time-bin width for oscillation averaging and normalisation (ms) |
| `--ref_offset_ms` | `400.0` | Time after cue offset (ms) used as normalization reference |
| `--n_trials` | `50` | Trials per condition × amplitude |
| `--n_workers` | auto | Number of parallel worker processes |
| `--w_inter_values` | — | Additional w_pyr_pyr_inter values for 2D heatmap sweep |
| `--no_cache` | — | Ignore existing pickle cache and recompute |
| `--seed` | 42 | Base random seed |
| `--no_show` | — | Suppress interactive plot display |
| `--n_nodes` | from ring params JSON or `128` | Number of ring nodes |
| `--w_pyr_pyr_inter` | from ring params JSON or `8.0` | PYR→PYR inter-node coupling |
| `--sigma_pyr_deg` | from ring params JSON or `30.0` | PYR→PYR connectivity width (degrees) |
| `--w_pv_global` | from ring params JSON or `10.0` | PV→PYR global inhibition strength |
| `--params_json` | — | Load local circuit parameters from JSON |

### How normalisation works

1. Each trial's bump amplitude timecourse is binned into non-overlapping `window_ms` windows (starting at cue onset).
2. The reference bin is the one whose centre is closest to `STIM_DURATION_MS + ref_offset_ms` (default 750 ms from cue onset).
3. Every bin value is divided by the reference bin value → normalised timecourse where the reference bin = 1.0.
4. A ratio > 1 at late times indicates a growing or sustained attractor; < 1 indicates decay.

### Outputs

```
figs/ring/bump_decay/{params}/{connectivity_label}/
├── bump_decay_trials.csv              # per-trial: ref_amplitude, end_val_normalized
├── .bump_decay_cache_<hash>.pkl       # simulation cache (auto-reused)
├── bump_decay_amp_sweep.png           # mean norm. Â at last time bin vs amplitude (all conditions)
├── amp{X}/
│   ├── bump_decay_timecourse.png      # mean ± SEM errorbar timecourse per condition
│   └── bump_decay_boxplot.png         # violin + jitter distributions over time bins per condition
└── {cond_key}/
    └── bump_decay_heatmap.png         # 2D heatmap (amplitude × w_inter) — only if w_inter sweep
```

The `amp{X}/` subdirectory contains `w{W}/` sub-levels only when `--w_inter_values` sweeps more than one value.

### Examples

```bash
# Standard run: WT vs WT_APP, 10 s delay, 50 trials, 7 amplitude levels
python -m circuit_model ring-bump-decay-study \
  --n_nodes 128 --w_pv_global 10 --w_pyr_pyr_inter 8 --sigma_pyr_deg 30 \
  --conditions WT WT_APP --amplitudes 10 15 20 25 30 35 40 --delay_ms 10000 --n_trials 50 --no_show

# Quick smoke test: 2 amplitudes, 5 trials, short delay
python -m circuit_model ring-bump-decay-study \
  --amplitudes 10 20 --delay_ms 3000 --n_trials 5 --no_show --no_cache

# 2D heatmap sweep: vary w_inter alongside amplitude
python -m circuit_model ring-bump-decay-study \
  --amplitudes 10 20 30 --w_inter_values 6 7 8 9 --n_trials 50 --no_show

# Finer time bins (250 ms) with a longer reference offset
python -m circuit_model ring-bump-decay-study \
  --window_ms 250 --ref_offset_ms 500 --delay_ms 10000 --no_show
```

---

## `ring-optimize`

Joint gradient-free optimization of `CircuitParams` (local circuit, ~60 parameters) and `RingParams` (`w_pyr_pyr_inter`, `w_pv_global`, `sigma_pyr_deg`) in a single run. The ring is simulated at rest (no stimulus) and the node-averaged firing rates are matched to quiet-wakefulness target rates. Knockout (KO) conditions are run on single-node by default (cheap) or on the ring (`--ko_on_ring`).

Two modes:
- **Mode 1** (default): match ring resting firing rates to `TargetRates`. Loss = rate loss + KO loss + Jacobian penalty.
- **Mode 2** (`--bump_mode`): same as Mode 1, plus a soft bump quality constraint — the ring must form a localized bump after a test stimulus. Bump and rate targets are independent (bump targets are biophysical constraints, not from experimental data).

```bash
python -m circuit_model ring-optimize [options]
```

### Parameters

#### Target firing rates (required)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--target_pyr` | float | — | Target mean PYR firing rate (Hz) |
| `--target_som` | float | — | Target mean SOM firing rate (Hz) |
| `--target_pv` | float | — | Target mean PV firing rate (Hz) |
| `--target_vip` | float | — | Target mean VIP firing rate (Hz) |
| `--target_alpha7_ko_pyr` | float | `None` | Target PYR rate under alpha7 knockout (Hz) |
| `--target_alpha5_ko_pyr` | float | `None` | Target PYR rate under alpha5 knockout (Hz) |
| `--target_beta2_ko_pyr` | float | `None` | Target PYR rate under beta2 knockout (Hz) |

#### Starting point

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--params_json` | str | `""` | Load initial `CircuitParams` from JSON (default: project WT default) |
| `--n_nodes` | int | `64` | Number of ring nodes — fixed during optimization, not optimized |
| `--w_pyr_pyr_inter_init` | float | `8.0` | Initial inter-node PYR→PYR weight |
| `--w_pv_global_init` | float | `10.0` | Initial global PV→PYR inhibition weight |
| `--sigma_pyr_deg_init` | float | `15.0` | Initial PYR connectivity Gaussian width (degrees) |

#### Ring parameter search bounds

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--w_pyr_pyr_inter_lo/hi` | float | `1.0 / 30.0` | Search bounds for `w_pyr_pyr_inter` |
| `--w_pv_global_lo/hi` | float | `0.5 / 20.0` | Search bounds for `w_pv_global` |
| `--sigma_pyr_deg_lo/hi` | float | `10.0 / 60.0` | Search bounds for `sigma_pyr_deg` |

#### Optimization settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n_samples` | int | `5000` | Number of optimization steps |
| `--top_k` | int | `10` | Keep top K candidates |
| `--optimizer` | str | `de` | `de` = TwoPointsDE (recommended), `cma` = CMA-ES, `chaining` = DE→Nelder-Mead, `auto` = NGOpt |
| `--early_stop_loss` | float | `1e-4` | Stop early if loss falls below this value |
| `--plateau_patience` | int | `1000` | Stop if no improvement for this many steps (0 = disable) |
| `--seed` | int | `0` | Random seed |
| `--freeze` | str | `""` | Comma-separated `CircuitParams` field names to freeze |
| `--set` | str | `""` | Override `CircuitParams` values before optimizing: `name=val,name=val` (e.g. `--set tau_s=20,g_exc=0.16,g_inh=0.087`). Combine with `--freeze` to pin biophysical constants. |

#### Ring simulation settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n_trials_ring` | int | `3` | Ring simulations per candidate (fewer than single-node due to cost) |
| `--ko_on_ring` | flag | `False` | Run KO conditions on ring (consistent but slower). Default: single-node. |
| `--T_ms` | float | `2500.0` | Ring simulation duration (ms) |
| `--burn_in_ms` | float | `1800.0` | Burn-in period to discard transients (ms) |
| `--window_ms` | float | `500.0` | Rate averaging window (ms) |
| `--noise_type` | str | `none` | `none`, `white`, or `ou` |
| `--ko_min_effect_penalty` | float | `5.0` | Penalty weight for weak KO effect |
| `--ko_wrong_direction_penalty` | float | `10.0` | Penalty weight for wrong-direction KO |

#### Adaptation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--no_adapt` | flag | `False` | Disable spike-frequency adaptation: sets `J_adapt_pyr=0` and `J_adapt_som=0` and freezes them. |

#### Turing instability penalty (optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--turing_weight` | float | `0.0` | Weight of two-sided Turing bistability penalty (0 = disabled). Penalises rest-state gain above `1 − margin` AND cue-state gain below `1 + margin`. |
| `--turing_margin` | float | `0.05` | Safety margin around the Turing threshold. |
| `--turing_cue_scale` | float | `5.0` | Multiplier on `I0_pyr` used to approximate the cue operating point (matches the bump stimulus amplitude). |

#### Mode 2 — bump quality (optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--bump_mode` | flag | `False` | Enable Mode 2: add bump quality constraint |
| `--min_bump_amplitude` | float | `0.3` | Minimum bump amplitude `[0, 1]`. Penalised if not reached. |
| `--bump_loss_weight` | float | `2.0` | Weight of bump loss relative to rate loss |
| `--bump_stim_amplitude` | float | `5.0` | Peak current of test stimulus (applied to PYR) |
| `--bump_stim_sigma_deg` | float | `20.0` | Gaussian width of test stimulus (degrees) |
| `--bump_stim_duration_ms` | float | `250.0` | Test stimulus duration (ms) |
| `--bump_eval_window_ms` | float | `500.0` | Post-stimulus window for bump amplitude evaluation (ms) |

#### I/O

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output_dir` | str | `ring_optim_output` | Directory to save `best_circuit_params.json` and `best_ring_params.json`. Overridden by the explicit path flags below. |
| `--save_best_circuit_json` | str | `""` | Explicit output path for the circuit params JSON (overrides `output_dir` for that file) |
| `--save_best_ring_json` | str | `""` | Explicit output path for the ring params JSON (overrides `output_dir` for that file) |
| `--log_file` | str | `ring_optim_log.jsonl` | JSONL log file (one entry per improvement) |
| `--log_interval` | int | `50` | Also log every N steps regardless of improvement |

### Output

```
ring_optim_output/
├── best_circuit_params.json   # Best CircuitParams (same format as optimize command)
└── best_ring_params.json      # Best RingParams (w_pyr_pyr_inter, w_pv_global, sigma_pyr_deg, n_nodes)
```

### Loss structure

**Mode 1:**
```
loss = ring_rate_loss + ko_loss / n_ko + jacobian_penalty
     [+ turing_weight × max(0, 1 + turing_margin − Φ'(I*_PYR)·w_pyr_pyr_inter)²  if turing_weight > 0]
```
where `ring_rate_loss` = MSPE between node-averaged ring rates and `TargetRates`.

The Turing penalty has two terms: (1) zero when `Φ'(I*_rest) × w_pyr_pyr_inter ≤ 1 − turing_margin` (no spontaneous bump at rest); (2) zero when `Φ'(I*_cue) × w_pyr_pyr_inter ≥ 1 + turing_margin` (bump can form under cue). `I*_cue` is evaluated with `I0_pyr` scaled by `turing_cue_scale`.

**Mode 2:**
```
loss = ring_rate_loss + ko_loss / n_ko + jacobian_penalty
     [+ turing_weight × turing_loss  if turing_weight > 0]
     + bump_loss_weight × max(0, min_amplitude − mean_amplitude)²
```
Bump loss is zero once the bump amplitude exceeds `min_amplitude`.

### Examples

```bash
# Mode 1: match resting firing rates on a 64-node ring
python -m circuit_model ring-optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --n_nodes 64 --n_samples 5000 --output_dir ring_optim_output

# Mode 1 with KO targets
python -m circuit_model ring-optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --target_alpha7_ko_pyr 17.539 --target_alpha5_ko_pyr 9.285 --target_beta2_ko_pyr 17.965 \
  --n_nodes 64 --n_samples 5000 --output_dir ring_optim_output

# Mode 2: also require bump formation (independent soft constraint)
python -m circuit_model ring-optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --n_nodes 64 --n_samples 5000 \
  --bump_mode --min_bump_amplitude 0.3 --bump_loss_weight 2.0 \
  --output_dir ring_optim_output

# Start from a previously fitted circuit, only optimize ring params
python -m circuit_model ring-optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --params_json params/new/ring_firing_rate/WT_1mo_article_ko.json \
  --freeze "tau_s,g_gaba_base,w_ee,w_pe,w_ep,w_pp,w_es,w_se,w_vs,w_sp,w_vp,w_ev" \
  --n_nodes 64 --n_samples 3000 --output_dir ring_optim_ring_only

# Quick smoke test (5 steps, small ring)
python -m circuit_model ring-optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --n_nodes 32 --n_samples 5 --n_trials_ring 1 --output_dir /tmp/ring_test
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
| `ASYM_SETTLING_MS` | 3000 ms | Per-trial independent noisy burn-in before cue (ring-asymmetry only) |
| `ASYM_PRE_CUE_WINDOW_MS` | 500 ms | Pre-cue window for asymmetry measurement (ring-asymmetry only) |

The total simulation time is computed as: `STIM_ONSET_MS + STIM_DURATION_MS + delay_ms` (unless `--total_time_ms` or `--response_onset_ms` override it).
