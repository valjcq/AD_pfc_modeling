# CLI Reference

The unified CLI is invoked via `python -m circuit_model <command>`.

```
python -m circuit_model {plot-transfer,diagnostic,run,optimize,study,random-bistable-search,ring-run,ring-calibrate,ring-bump-decay-study,ring-optimize} [options]
```

---

## Table of Contents

1. [plot-transfer](#plot-transfer) -- Plot transfer functions for all 4 populations
2. [diagnostic](#diagnostic) -- Analytical diagnostic plots (Turing gain + transfer functions)
3. [run](#run) -- Single-circuit simulation with plotting
4. [optimize](#optimize) -- Nevergrad parameter optimization
5. [study](#study) -- Batch study across 8 experimental conditions
6. [ring-run](#ring-run) -- Ring attractor single-condition simulation
7. [ring-calibrate](#ring-calibrate) -- 3D parameter sweep (w_pv_global × w_pyr_pyr_inter × amplitude)
8. [ring-bump-decay-study](#ring-bump-decay-study) -- Assess whether a bump is a self-sustained attractor or a decaying transient
9. [ring-optimize](#ring-optimize) -- Joint optimization of CircuitParams + RingParams against ring-level firing rate targets

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
- With `--params_json params/new/ring_firing_rate/WT_1mo_article_ko.json` → `figs/optim/transfer_functions_WT_1mo_article_ko.png`
- With `--condition WT_APP` → `figs/optim/transfer_functions_WT_APP.png`
- Without `--params_json` / `--condition` → `figs/optim/transfer_functions.png`

### Examples

```bash
# Default parameters
python -m circuit_model plot-transfer

# From a fitted parameter file (auto-saved with filename suffix)
python -m circuit_model plot-transfer --params_json params/new/ring_firing_rate/WT_1mo_article_ko.json

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

## `diagnostic`

Generate analytical (no-simulation) diagnostic plots for a given parameter set: **(1)** Turing gain product vs PYR firing rate with marked operating points, and **(2)** transfer functions for all four populations with operating point overlays.

```bash
python -m circuit_model diagnostic [options]
```

### Purpose

This command provides fast analytical diagnostics without running simulations. It uses the same default parameters as `ring-run` for consistency across the CLI. Features:
- Computes the Turing gain product G_eff × w_pyr_inter across a fine grid of PYR firing rates (0–80 Hz)
- Overlays three operating points (rest, bump, cue) to visualize the Turing threshold crossing
- Plots transfer functions for all four populations with operating point markers
- Outputs gain product values at the three operating points to the console

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--params_json` | str | `params/new/ring_firing_rate/WT_1mo_article_ko.json` | Path to circuit parameters JSON file (auto-loaded if available) |
| `--ring_params_json` | str | `params/new/ring_firing_rate/WT_1mo_article_ko_ring.json` | Path to ring parameters JSON file (auto-loaded if available) |
| `--target_pyr` | float | `8.0` | Rest PYR firing rate for operating point marker (Hz) |
| `--turing_bump_hz` | float | `40.0` | Target PYR firing rate (Hz) for the bump operating point marker |
| `--turing_cue_scale` | float | `2.0` | Multiplier for I0_pyr to compute cue operating point |
| `--out_dir` | str | `figs/diagnostic` | Output directory for PNG files |
| `--no_show` | flag | `False` | Don't display plots (useful for batch processing) |

### Output

Two PNG files are saved to the output directory:

1. **turing_gain_product.png**
   - X-axis: PYR firing rate (0–80 Hz, ~500 points)
   - Y-axis: Turing gain product (G_eff × w_pyr_inter)
   - Horizontal dashed line at gain = 1 (Turing instability threshold)
   - Shaded regions: green for gain > 1 (network-driven), red for gain < 1 (stable)
   - Three marked operating points:
     - **Blue**: Rest (target_pyr, default 8 Hz)
     - **Green**: Bump (fixed at turing_bump_hz, default 40 Hz)
     - **Orange**: Cue (turing_cue_scale × I0_pyr)
   - Annotated gain values at each operating point

2. **transfer_functions.png**
   - 2×2 grid of subplots (PYR, SOM, PV, VIP)
   - Each shows Φ(I) curve over input current range [0, 1.5 nA]
   - Population-specific amplitude factors A_x displayed in titles
   - Operating point markers (blue dashed line at rest I_star, green/orange markers for bump/cue)
   - Annotated with current and firing rate at each operating point

### Console Output

A table of gain product values at the three operating points:

```
[TURING GAIN PRODUCT ANALYSIS]

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

Run Nevergrad optimization to find parameters matching target firing rates. The optimizer can be selected via `--optimizer`; the recommended choice is `chaining` (global DE search followed by Nelder-Mead refinement, matching the reference paper pipeline).

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
| `chaining` | DE → Nelder-Mead | **Recommended.** DE explores globally for `min(n_samples//5, 10000)` steps, then Nelder-Mead refines for the rest (matches the reference paper pipeline) |
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
| `--turing_weight` | float | `2.0` | Weight of two-sided Turing bistability penalty (0 = disabled). Penalises rest-state gain above `1 − margin` AND cue-state gain below `1 + margin`. |
| `--turing_margin` | float | `0.05` | Safety margin around the Turing threshold. |
| `--turing_w_inter_ref` | float | `10.0` | Reference inter-node weight used only in the single-node analytical Turing approximation (diagnostic only; ring simulations derive this from `J_NMDA`). |
| `--turing_cue_scale` | float | `0.4` | Multiplier on `I0_pyr` used to approximate the cue operating point. |
| `--ach_ratio_weight` | float | `2.0` | Weight of β2/α7 ACh current ratio penalty (0 = disabled). Penalises `I_beta2_som / I_alpha7_som` deviating from 35 (Koukouli et al. 2025). |

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
| `--w_inter_values` | — | (Deprecated — no longer has effect) |
| `--no_cache` | — | Ignore existing pickle cache and recompute |
| `--seed` | 42 | Base random seed |
| `--no_show` | — | Suppress interactive plot display |
| `--n_nodes` | from ring params JSON or `64` | Number of ring nodes |
| `--sigma_pyr_deg` | from ring params JSON or `15.0` | PYR→PYR Gaussian width (degrees) |
| `--sigma_som_deg` | from ring params JSON or `15.0` | SOM→PYR lateral Gaussian width (degrees) |
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
  --conditions WT WT_APP --amplitudes 10 15 20 25 30 35 40 --delay_ms 10000 --n_trials 50 --no_show

# Quick smoke test: 2 amplitudes, 5 trials, short delay
python -m circuit_model ring-bump-decay-study \
  --amplitudes 10 20 --delay_ms 3000 --n_trials 5 --no_show --no_cache

# Custom ring structure (widths only — strengths come from the circuit params JSON)
python -m circuit_model ring-bump-decay-study \
  --sigma_pyr_deg 20 --sigma_som_deg 10 --amplitudes 10 20 30 --n_trials 50 --no_show

# Finer time bins (250 ms) with a longer reference offset
python -m circuit_model ring-bump-decay-study \
  --window_ms 250 --ref_offset_ms 500 --delay_ms 10000 --no_show
```

---

## `ring-optimize`

Joint gradient-free optimization of `CircuitParams` (local circuit, ~60 parameters) and `RingParams` (`sigma_pyr_deg`, `sigma_som_deg`) in a single run. The ring is simulated at rest (no stimulus) and the node-averaged firing rates are matched to quiet-wakefulness target rates. Connection strengths are **not free parameters** — they are derived from the fitted `CircuitParams` (see [ring_attractor.md §2](ring_attractor.md#row-sum-normalisation-principle)). Knockout (KO) conditions are always run on the ring.

Primary mode:
- Match ring resting firing rates to `TargetRates` with KO and structural penalties.
- Apply a trace-based Turing bistability loss (optional via `--turing_weight`) using a deterministic cue simulation.

Legacy mode:
- `--bump_mode` is deprecated and ignored. Bump constraints are now integrated into the trace-based Turing loss.

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
| `--ring_params_json` | str | `""` | Load initial `RingParams` from JSON (same format as `--save_best_ring_json` output) — **required** |
| `--n_nodes` | int | `64` | Number of ring nodes — fixed during optimization, not optimized |

#### Ring parameter search bounds

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sigma_pyr_deg_lo/hi` | float | `5.0 / 40.0` | Search bounds for `sigma_pyr_deg` (°) |
| `--sigma_som_deg_lo/hi` | float | `5.0 / 40.0` | Search bounds for `sigma_som_deg` (°) |

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
| `--n_trials_ring` | int | `5` | Ring simulations per candidate (for stochastic averaging) |
| `--T_ms` | float | `2500.0` | Ring simulation duration (ms) |
| `--burn_in_ms` | float | `1200.0` | Burn-in period to discard transients (ms) |
| `--window_ms` | float | `500.0` | Rate averaging window (ms) |
| `--noise_type` | str | `white` | `none`, `white`, or `ou` |
| `--ko_min_effect_penalty` | float | `5.0` | Penalty weight for weak KO effect |
| `--ko_wrong_direction_penalty` | float | `10.0` | Penalty weight for wrong-direction KO |

#### Adaptation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--no_adapt` | flag | `False` | Disable spike-frequency adaptation: sets `J_adapt_pyr=0` and `J_adapt_som=0` and freezes them. |

#### Trace-based Turing bistability penalty (optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--turing_weight` | float | `2.0` | Weight of trace-based Turing loss (0 = disabled): rest stability + bump sustain + anti-runaway. |
| `--turing_margin` | float | `0.05` | Safety margin `m` around the Turing threshold. |
| `--turing_cue_amplitude` | float | `0.4` | Additive cue amplitude as factor of `I0_pyr` (PYR-only). |
| `--turing_cue_duration_ms` | float | `250.0` | Cue duration (ms) for deterministic Turing pass. |
| `--turing_cue_sigma_deg` | float | `20.0` | Cue spatial width (deg) for deterministic Turing pass. |
| `--turing_late_delay_ms` | float | `500.0` | Late-delay window length (ms) used for sustain checks. |
| `--turing_bump_min_hz` | float | `35.0` | Minimum bump-node PYR rate in late delay. |
| `--turing_bump_max_hz` | float | `45.0` | Maximum bump-node PYR rate in late delay. |
| `--turing_topk_nodes` | int | `5` | Number of top PYR nodes defining the bump support set. |
| `--turing_activate_below_ring_rate_loss` | float | `1.0` | Gate for Turing loss: trace-based Turing term is applied only when ring firing-rate loss is `<=` this threshold. |

#### ACh receptor ratio penalty (optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--ach_ratio_weight` | float | `2.0` | Weight of β2/α7 ACh current ratio penalty (0 = disabled). Penalises `I_beta2_som / I_alpha7_som` deviating from 35 (Koukouli et al. 2025). |

#### Legacy bump options (deprecated)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--bump_mode` | flag | `False` | Deprecated and ignored (kept for backward compatibility). |
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
└── best_ring_params.json      # Best RingParams (sigma_pyr_deg, sigma_som_deg, n_nodes)
```

### Loss structure

**Current objective:**
```
loss = ring_rate_loss + ko_loss / n_ko + jacobian_penalty
  [+ turing_weight × L_turing_trace if turing_weight > 0 and ring_rate_loss <= turing_activate_below_ring_rate_loss]
     [+ ach_ratio_weight × L_ach_ratio  if ach_ratio_weight > 0]
```
where `ring_rate_loss` = MSPE between node-averaged ring rates and `TargetRates`.

`L_turing_trace` is computed from a deterministic cue simulation and includes:
- rest-rate + rest-gain constraints (no spontaneous bump),
- late-delay bump-rate band around 40 Hz,
- late-delay gain floor for sustain,
- gain/rate ceilings to prevent runaway.

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

# Legacy bump flag is accepted but ignored (trace-based Turing already includes bump constraints)
python -m circuit_model ring-optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --n_nodes 64 --n_samples 5000 \
  --bump_mode \
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
| `--n_nodes` | 128 | Number of ring nodes |
| `--w_pyr_pyr_inter` | 4.0 | PYR→PYR inter-node coupling |
| `--sigma_pyr_deg` | 30.0 | PYR→PYR connectivity width (degrees) |
| `--w_pv_global` | 4.0 | PV→PYR global inhibition strength |
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
