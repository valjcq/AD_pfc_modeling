# Roadmap — WT vs WT_APP full analysis (sigma_15 network)

## Context
Run all ring-attractor analyses comparing WT and WT_APP, using **per-condition calibrated
parameters** from `figs/ring/calibration/default/128_inhib_10_sigma_15/best_settings.txt`:

| Condition | amplitude | w_inter |
|-----------|-----------|---------|
| WT        | 10        | 8.0     |
| WT_APP    | 10        | 7.5     |

Per-condition weights are passed **in condition order** to `--w_pyr_pyr_inter`:
`WT` first → `8.0`, `WT_APP` second → `7.5`.

---

## Command 1 — ring-study (bump metrics sweep)

```bash
python -m circuit_model ring-study \
  --conditions WT WT_APP \
  --n_nodes 128 \
  --w_pv_global 10 \
  --sigma_pyr_deg 15 \
  --w_pyr_pyr_inter 8.0 7.5 \
  --amplitudes 5 10 15 20 25 \
  --delay_ms 5000 \
  --delay_step_ms 1000 \
  --record_dt_ms 5 \
  --response_onset_ms 0 \
  --n_trials 100 \
  --n_workers 8 \
  --amp_eval_step_ms 500 \
  --error_band sem \
  --seed 442 \
  --no_show
```

**Output:** `figs/ring/study/default/128_inhib_10_sigma_15/`

---

## Command 2 — ring-bump-decay-study (bump persistence)

```bash
python -m circuit_model ring-bump-decay-study \
  --conditions WT WT_APP \
  --n_nodes 128 \
  --w_pv_global 10 \
  --sigma_pyr_deg 15 \
  --w_pyr_pyr_inter 8.0 7.5 \
  --amplitudes 5 10 15 20 25 \
  --delay_ms 10000 \
  --record_dt_ms 5 \
  --response_onset_ms 0 \
  --ref_offset_ms 400 \
  --window_ms 500 \
  --n_trials 100 \
  --n_workers 8 \
  --seed 442 \
  --no_show
```

**Output:** `figs/ring/bump_decay/default/128_inhib_10_sigma_15/`

---

## Command 3 — ring-diffusion (bump drift / MSD)

```bash
python -m circuit_model ring-diffusion \
  --conditions WT WT_APP \
  --n_nodes 128 \
  --w_pv_global 10 \
  --sigma_pyr_deg 15 \
  --w_pyr_pyr_inter 8.0 7.5 \
  --amplitude 10 \
  --delay_ms 5000 \
  --record_dt_ms 5 \
  --response_onset_ms 0 \
  --n_trials 100 \
  --n_workers 8 \
  --error_band sem \
  --seed 442 \
  --no_show
```

> `--filter_cutoff_hz` is intentionally omitted → auto-detected from oscillation spectrum
> (0.4 × dominant oscillation frequency).

**Output:** `figs/ring/diffusion/default/128_inhib_10_sigma_15/`

---

## Command 4 — ring-asymmetry (L/R asymmetry)

```bash
python -m circuit_model ring-asymmetry \
  --conditions WT WT_APP \
  --n_nodes 128 \
  --w_pv_global 10 \
  --sigma_pyr_deg 15 \
  --w_pyr_pyr_inter 8.0 7.5 \
  --amplitude 10 \
  --delay_ms 5000 \
  --record_dt_ms 5 \
  --response_onset_ms 0 \
  --n_trials 100 \
  --n_workers 8 \
  --correct_asymmetry \
  --seed 442 \
  --no_show
```

**Output:** `figs/ring/asymmetry/default/128_inhib_10_sigma_15/`

---

## Command 5 — ring-oscillation-study (2–12 Hz band power)

```bash
python -m circuit_model ring-oscillation-study \
  --conditions WT WT_APP \
  --n_nodes 128 \
  --w_pv_global 10 \
  --sigma_pyr_deg 15 \
  --w_pyr_pyr_inter 8.0 7.5 \
  --amplitudes 5 10 15 20 25 \
  --delay_ms 5000 \
  --record_dt_ms 5 \
  --response_onset_ms 0 \
  --osc_skip_ms 200 \
  --min_freq_hz 2 \
  --max_freq_hz 12 \
  --tf_window_s 1.0 \
  --tf_overlap 0.8 \
  --sample_time_frac 0.75 \
  --n_trials 100 \
  --n_workers 8 \
  --seed 442 \
  --no_show
```

**Output:** `figs/ring/oscillation/default/128_inhib_10_sigma_15/`

---

## Notes

- Commands can be run in any order; each is independent.
- To force re-computation (ignore CSV/pickle cache), add `--no_cache` to any command.
- Legend labels will auto-annotate per-condition weights, e.g. `WT (e=8)` vs `WT APP (e=7.5)`.
- `--amplitude` (single shared value) is used for commands without an amplitude sweep
  (diffusion, asymmetry). `--amplitudes` (list) is used for study-type commands.
