# CLI Reference

The unified CLI is invoked via `python -m circuit_model <command>`.

```
python -m circuit_model {run,optimize,study,ring-run,ring-study} [options]
```

---

## Table of Contents

1. [run](#run) -- Single-circuit simulation with plotting
2. [optimize](#optimize) -- Nevergrad parameter optimization
3. [study](#study) -- Batch study across 8 experimental conditions
4. [ring-run](#ring-run) -- Ring attractor single-condition simulation
5. [ring-study](#ring-study) -- Ring attractor multi-condition comparison

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
| `--amplitude` | float | `150.0` | Stimulus peak current amplitude |
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
| `--sigma_pyr_deg` | float | `10.0` | PYR→PYR connectivity width (degrees) |
| `--w_pyr_pyr_inter` | float | `3.96` | Total PYR→PYR coupling for Gaussian profile. Not used with Compte |
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
- Gaussian: `gauss_w<w>_s<sigma>-pv_unif_<w_pv>` (e.g. `gauss_w3.96_s10-pv_unif_2`)
- Compte: `compte_J<J+>_s<sigma>-pv_unif_<w_pv>` (e.g. `compte_J1.6_s30-pv_unif_2`)

### Examples

```bash
# Wild type, default amplitude (128 nodes)
python -m circuit_model ring-run --condition WT

# Smaller network
python -m circuit_model ring-run --condition WT --n_nodes 64

# Alpha7 KO with higher amplitude
python -m circuit_model ring-run --condition a7_KO --amplitude 200 --delay_ms 5000

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
| `--amplitudes` | float (list) | `[150]` | Multiple stimulus amplitudes to compare |
| `--n_trials` | int | `1` | Number of trials per condition x amplitude |
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
    --amplitudes 50 100 150 200 \
    --conditions WT WT_APP \
    --n_trials 10 --n_workers 4

# Custom delay evaluation
python -m circuit_model ring-study --delay_step_ms 500 --delay_ms 5000

# Force recompute (ignore cache)
python -m circuit_model ring-study --no_cache
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
