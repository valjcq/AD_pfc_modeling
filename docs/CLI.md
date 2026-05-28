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

Nevergrad parameter optimization with two stages.

### `--stage` (default `weights`)

| Stage | Free params | Targets |
|-------|-------------|---------|
| `weights` | weights + currents + adaptation + `g_alpha7`. Receptor activations (`act_alpha7_*`, `act_beta2`, `act_alpha5`) are frozen at 1.0. | baseline + global KOs + selective α7 KOs |
| `receptors` | only `act_alpha7_pv`, `act_alpha7_som`, `act_alpha7_ndnf`, `act_beta2`, `act_alpha5` (bounded `[0, 5]`). Everything else frozen. Requires `--params_json` (Stage-1 fit). | per-drug NDNF/PV targets, fit independently per drug |

### Stage-1 targets (required unless `--resume`)

| Flag | Description |
|------|-------------|
| `--target_pyr FLOAT`  | Target mean firing rate for PYR (Hz) |
| `--target_som FLOAT`  | Target mean firing rate for SOM (Hz) |
| `--target_pv FLOAT`   | Target mean firing rate for PV (Hz) |
| `--target_vip FLOAT`  | Target mean firing rate for VIP (Hz) |
| `--target_ndnf FLOAT` | Target mean firing rate for NDNF (Hz) |

### Optional KO targets

The optimizer always simulates every KO condition (so they appear in the report), but each only enters the loss if its target flag is set. Global KOs are measured on PYR; cell-type-selective α7 KOs are measured on the deleted cell type itself (matching the flx/flx baseline data).

| Flag | KO condition | Measured on |
|------|--------------|-------------|
| `--target_alpha7_ko_pyr FLOAT`        | Global α7-KO (all per-cell α7 = 0) | PYR  |
| `--target_alpha5_ko_pyr FLOAT`        | Global α5-KO  | PYR  |
| `--target_beta2_ko_pyr FLOAT`         | Global β2-KO  | PYR  |
| `--target_alpha7_ndnf_ko_ndnf FLOAT`  | NDNF-selective α7-KO (only `act_alpha7_ndnf = 0`) | NDNF |
| `--target_alpha7_pv_ko_pv FLOAT`      | PV-selective α7-KO (only `act_alpha7_pv = 0`)     | PV   |

### Stage-2 drug targets (used when `--stage receptors`)

| Flag | Description |
|------|-------------|
| `--drugs MLA,PNU[,nicotine]` | Comma-separated list of drugs to fit (default: `MLA,PNU,nicotine`) |
| `--target_mla_ndnf FLOAT` / `--target_mla_pv FLOAT` | MLA targets (NDNF and PV) |
| `--target_pnu_ndnf FLOAT` / `--target_pnu_pv FLOAT` | PNU targets (NDNF and PV) |
| `--target_nicotine_ndnf FLOAT` / `--target_nicotine_pv FLOAT` | Nicotine targets (NDNF and PV) |

> **Ill-posed warning.** Each drug has 2 measurements (or 1 for nicotine) vs 5 free activations.
> Multiple activation tuples can produce the same NDNF + PV rates. Bounds `[0, 5]` keep solutions physiological but the fit is not unique.

### Optimizer settings

| Flag | Default | Description |
|------|---------|-------------|
| `--n_samples INT`     | 5000 | Total Nevergrad budget |
| `--top_k INT`         | 10 | Number of top candidates to retain |
| `--optimizer {de,twopointde,cma,chaining,auto}` | `de` | Nevergrad optimizer choice. `de` is an alias for `twopointde`. `chaining` = TwoPointsDE → Nelder-Mead. |
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

### Loss weights

The per-term loss is a squared log-fold-change:

```
L_term = ( log( max(sim, ε) / target ) )²        ε = 0.01 Hz
```

— target-normalised and symmetric in over/under-shoot, diverges as `sim → 0`
so no population can be silenced to dodge the penalty. The total loss is the
sum over all measurements; per-bucket weights let you tune the balance:

| Flag | Default | Description |
|------|---------|-------------|
| `--weight_base FLOAT`         | 1.0 | Weight on the 5 baseline firing-rate targets |
| `--weight_global_ko FLOAT`    | 1.0 | Weight on global α7/α5/β2 KO PYR targets |
| `--weight_selective_ko FLOAT` | 1.0 | Weight on NDNF/PV-selective α7 KO targets |
| `--weight_drug FLOAT`         | 1.0 | Weight on Stage-2 drug targets |
| `--w_hi FLOAT`                | 0.01 | Upper bound for synaptic weights (nA/Hz) |

There are no Jacobian or ACh-ratio penalties anymore — disabled per project decision.

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

### Examples

**Stage 1** — fit weights + currents:

```bash
python -m circuit_model optimize \
    --target_pyr 1.7328 --target_som 1.3564 --target_pv 1.5281 --target_vip 2.9791 \
    --target_ndnf 2.5309 \
    --target_alpha7_ko_pyr 2.1928 --target_beta2_ko_pyr 1.0825 --target_alpha5_ko_pyr 0.4762 \
    --target_alpha7_ndnf_ko_ndnf 3.0767 --target_alpha7_pv_ko_pv 1.3966 \
    --optimizer twopointde --n_samples 20000 \
    --output_dir fits/WT_NDNF_5pop
```

**Stage 2** — re-fit receptor activations under drugs (uses Stage-1 best params):

```bash
python -m circuit_model optimize --stage receptors \
    --params_json fits/WT_NDNF_5pop/best_params.json \
    --drugs MLA,PNU \
    --target_mla_ndnf 2.6578 --target_mla_pv 1.5106 \
    --target_pnu_ndnf 2.7538 --target_pnu_pv 1.5239 \
    --optimizer twopointde --n_samples 5000 \
    --output_dir fits/WT_NDNF_5pop_stage2
```

Stage 1 prints a Jacobian sanity-check and a `Condition × Population` comparison table. Stage 2 writes one `stage2_results.json` per drug with the fitted activations and predicted NDNF/PV rates.

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
