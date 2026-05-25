# CLI Reference

All commands are invoked as:

```bash
python -m circuit_model <command> [options]
```

Available commands:
- [`plot-transfer`](#plot-transfer)
- [`run`](#run)
- [`optimize`](#optimize)
- [`study`](#study)

---

## `plot-transfer`

Plot the Wong-Wang transfer functions Φ(I) for all 5 populations (PYR, SOM, PV, VIP, NDNF).

### Common parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--params_json PATH` | — | Load `CircuitParams` from JSON |
| `--condition KEY`    | — | Apply a study condition preset (e.g. `WT`, `a7_KO`, `b2_KO`, `a5_KO`) |
| `--set NAME=VAL,...` | — | Override individual parameter values |
| `--I_min FLOAT`      | 0.0 | Lower bound of input-current sweep |
| `--I_max FLOAT`      | 1.0 | Upper bound of input-current sweep |
| `--save_plot PATH`   | auto | Explicit save path |
| `--no_show`          | False | Don't open the figure window |

### Examples

```bash
python -m circuit_model plot-transfer
python -m circuit_model plot-transfer --condition a7_KO
python -m circuit_model plot-transfer --params_json best_params.json
```

---

## `run`

Run a single deterministic or noisy simulation of the 5-population circuit and produce a dashboard plot.

### Selected parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--T_ms FLOAT`        | 3000.0 | Simulation duration (ms) |
| `--dt_ms FLOAT`       | 0.1   | Integration step (ms) |
| `--burn_in_ms FLOAT`  | 500.0 | Discarded transient for summary stats |
| `--window_ms FLOAT`   | 500.0 | Trailing averaging window |
| `--noise_type {none,white,ou}` | `none` | Noise model |
| `--tau_noise_ms FLOAT` | 5.0 | OU noise time constant |
| `--params_json PATH`  | — | Load circuit parameters from JSON |
| `--condition KEY`     | — | Apply a condition preset |
| `--set NAME=VAL,...`  | — | Override parameters |
| `--use_transient`     | False | Enable PYR-only square transient(s) |
| `--seed INT`          | None | RNG seed (for noise) |
| `--no_show`           | False | Do not open the figure window |

### Examples

```bash
python -m circuit_model run
python -m circuit_model run --noise_type ou --T_ms 5000
python -m circuit_model run --condition a7_KO
```

---

## `optimize`

Nevergrad parameter optimization to match baseline firing rates for the 5 populations and a set of knockout conditions.

### Targets (required unless `--resume`)

| Flag | Description |
|------|-------------|
| `--target_pyr FLOAT`  | Target mean firing rate for PYR (Hz) |
| `--target_som FLOAT`  | Target mean firing rate for SOM (Hz) |
| `--target_pv FLOAT`   | Target mean firing rate for PV (Hz) |
| `--target_vip FLOAT`  | Target mean firing rate for VIP (Hz) |
| `--target_ndnf FLOAT` | Target mean firing rate for NDNF (Hz) |

### Optional KO targets

The optimizer always simulates each global KO condition (so they appear in the report), but they only enter the loss if the corresponding target flag is set.

| Flag | KO condition |
|------|--------------|
| `--target_alpha7_ko_pyr FLOAT` | Global α7-KO (all per-cell α7 zeroed) — PYR target |
| `--target_alpha5_ko_pyr FLOAT` | α5-KO — PYR target |
| `--target_beta2_ko_pyr FLOAT`  | β2-KO — PYR target |

### Optimizer settings

| Flag | Default | Description |
|------|---------|-------------|
| `--n_samples INT`     | 5000 | Total Nevergrad budget |
| `--top_k INT`         | 10 | Number of top candidates to retain |
| `--optimizer {de,twopointde,cma,chaining,auto}` | `de` | Nevergrad optimizer choice. `de` is an alias for `twopointde`. `chaining` = TwoPointsDE → Nelder-Mead. |
| `--squared_loss / --no-squared_loss` | True | MSPE vs MAPE for the base firing-rate loss |
| `--seed INT`          | None | RNG seed |

### Simulation settings

| Flag | Default | Description |
|------|---------|-------------|
| `--T_ms FLOAT`        | 2500 | Simulation duration |
| `--dt_ms FLOAT`       | 0.1 | Integration step |
| `--burn_in_ms FLOAT`  | 1200 | Burn-in discarded before averaging |
| `--window_ms FLOAT`   | 500 | Trailing averaging window |
| `--n_trials INT`      | 8 | Trials per parameter set (noise averaging) |
| `--noise_type {none,white,ou}` | white | Noise model |
| `--tau_noise_ms FLOAT`| 5.0 | OU time constant |
| `--init_rate_scale FLOAT` | 0.2 | Scale for random initial rates |
| `--max_rate FLOAT`    | 200 | Reject candidates with any rate above this (Hz) |

### Loss weights / penalties

| Flag | Default | Description |
|------|---------|-------------|
| `--ko_min_effect_penalty FLOAT`      | 5.0  | Penalty when a KO produces a weak effect |
| `--ko_wrong_direction_penalty FLOAT` | 10.0 | Penalty when a KO moves PYR the wrong way |
| `--skip-jacobian`     | False | Disable the Jacobian connectivity penalty |
| `--jacobian_weight FLOAT` | 1.0 | Weight on the Jacobian connectivity penalty |
| `--ach_ratio_weight FLOAT` | 2.0 | Weight on β2/α7 ratio penalty on SOM (Koukouli 2025) |
| `--w_hi FLOAT` | 0.01 | Upper bound for synaptic weights (nA/Hz) |

### Parameter control

| Flag | Description |
|------|-------------|
| `--params_json PATH` | Use these parameters as the initial point and base for non-fitted fields |
| `--condition KEY` | Apply a condition preset before optimizing |
| `--set NAME=VAL,...` | Override specific parameters before optimizing |
| `--freeze NAME,...` | Comma-separated parameter names to freeze (not optimized) |
| `--show_params` | Print which parameters are free vs frozen |
| `--no_adapt` | Set and freeze `J_adapt_pyr=0` and `J_adapt_som=0` |

### I/O

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir PATH` | — | Directory for all run outputs (`best_params.json/.txt`, `log.jsonl`, loss-evolution plots). Recommended when running many experiments. |
| `--save_best_json PATH` | `best_params.json` | Save best parameter set to this JSON. If `--output_dir` is set and this is at the default, the file goes inside `--output_dir`. |
| `--log_file PATH` | `{output_dir}/log.jsonl` | JSONL log of best-so-far per `--log_interval` steps |
| `--log_interval INT` | 50 | Logging period |
| `--resume` | False | Resume from a previous run's log + best JSON |

### Example

```bash
python -m circuit_model optimize \
    --target_pyr 1.7328 --target_som 1.3564 --target_pv 1.5281 --target_vip 2.9791 \
    --target_ndnf 2.5309 \
    --target_alpha7_ko_pyr 2.1928 --target_beta2_ko_pyr 1.0825 --target_alpha5_ko_pyr 0.4762 \
    --optimizer twopointde --n_samples 20000 \
    --output_dir fits/WT_NDNF_5pop
```

The optimizer prints a Jacobian sanity check and a `Condition × Population` comparison table when finished.

---

## `study`

Batch-simulate the 5-population model across receptor-knockout conditions and produce box plots of firing-rate distributions.

### Selected parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--n_runs INT` | 50 | Trials per condition |
| `--T_ms FLOAT` | 2500 | Per-trial duration |
| `--burn_in_ms FLOAT` | 1800 | Burn-in discarded before averaging |
| `--window_ms FLOAT` | 500 | Trailing averaging window |
| `--noise_type {none,white,ou}` | white | Noise model |
| `--tau_noise_ms FLOAT` | 5.0 | OU time constant |
| `--n_workers INT` | auto | Parallel worker count |
| `--params_json PATH` | — | Initial circuit parameters |
| `--unit Hz` | Hz | Display unit (Hz only, kept for backward compatibility) |

### Conditions

The study sweeps over the entries in `circuit_model.study.CONDITION_ORDER` (e.g. `WT`, `WT_APP`, `a7_KO`, `a7_KO_APP`, `b2_KO`, `b2_KO_APP`, `a5_KO`, `a5_KO_APP`, `APP_sim`). Global α7-KO presets zero all per-cell α7 activations (`act_alpha7_pv`, `act_alpha7_som`, `act_alpha7_ndnf`).

### Example

```bash
python -m circuit_model study --n_runs 100 --noise_type white
```
