# Ring Network 3D Sweep Plan: w_pv_global × w_pyr_pyr_inter × amplitude

## Overview

The `ring-calibrate` command now runs a 3D parameter sweep that replaces the old 2D
calibration. Instead of measuring bump amplitude against a noise floor, it classifies
each simulated delay period into three mutually exclusive states and reports the
**fraction of delay time** spent in each.

---

## State classification (per trial, per delay timestep)

```
max_PYR(t) = max firing rate across all PYR nodes at time t

resting_threshold  = max(resting_hz × 2.5,  resting_hz + 5,  10 Hz)
                      where resting_hz = mean PYR rate at end of burn-in

  RESTING    : max_PYR(t) < resting_threshold
  BUMP       : resting_threshold ≤ max_PYR(t) < 90 Hz      ← genuine bump
  SATURATED  : max_PYR(t) ≥ 90 Hz                          ← network maxed out
```

| Metric | Definition |
|---|---|
| `delay_bump_frac` | fraction of delay timesteps in BUMP state (target: high) |
| `delay_sat_frac` | fraction in SATURATED state (target: low) |
| `delay_rest_frac` | fraction in RESTING state |
| `quality_score` | `bump_frac × (1 − sat_frac)` |
| `cue_saturated` | 1 if peak PYR during cue ≥ 190 Hz |

**A good working point**: `delay_bump_frac > 0.7`, `delay_sat_frac < 0.1`, `resting_hz < 5 Hz`.

---

## Worker scaling

Benchmarked on this machine (same delay_ms=5000, 3×4×4 grid):

| Workers | Grid time | Per-trial | Total |
|---|---|---|---|
| 6 | 49.4 s | 0.13 s | 66 s |
| **10** | **40.8 s** | **0.11 s** | **59 s** |

**Use `--n_workers 10`** — ~20% faster grid phase at negligible extra CPU overhead.
Burn-in is sequential (cannot be parallelised across (w_pv, w_pyr) combos) and takes
~1.2 s/sim regardless of worker count.

Effective throughput at 10 workers: **≈9.4 trials/sec**.

---

## Sweep 1 — Big map (run first)

**Goal**: Cover the full feasible parameter space in one pass, locate all interesting
regimes before zooming in. Includes both sub-threshold and saturated regions to bracket
the bistable boundary.

```bash
python3 -m circuit_model ring-calibrate \
  --params_json figs/optim/bistable_high_fr/best_params.json \
  --sigma_pyr_deg 15 \
  --w_pv_values   0.03 0.04 0.05 0.06 0.07 0.08 0.10 \
  --w_inter_values \
    0.001 0.001259 0.001585 0.001996 0.002514 0.003165 \
    0.003985 0.005018 0.006319 0.007956 0.010018 0.012615 0.015884 0.02 \
  --amplitudes    0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 \
  --n_trials 20 \
  --n_workers 10 \
  --no_show
```

| Dimension | Values | Count |
|---|---|---|
| w_pv_global | 0.03 → 0.10 (+ 0.10) | 7 |
| w_pyr_pyr_inter | log-spaced 0.001 → 0.020 | 14 |
| amplitude | 0.30 → 0.80 linear | 11 |
| n_trials | — | 20 |
| **Total trials** | 7 × 14 × 11 × 20 | **21 560** |

**Estimated runtime**: burn-in ~2 min + grid ~38 min = **~40 min total**.

Output in `figs/ring/calibration/128_sigma_15/`:
- Per-w_pyr slice heatmaps (rows=w_pv, cols=amplitude): resting rate, cue peak,
  cue-sat fraction, delay-bump fraction, delay-sat fraction
- Summary heatmap: best bump_frac / quality_score over all w_pyr slices

### What to look for in the results

1. **Bistable boundary**: the diagonal contour in amplitude × w_pv space where
   `cue_saturated` transitions from 0 → 1. Below this line, cue is too weak.
2. **Localization transition**: the w_pyr threshold where `delay_bump_frac` jumps
   (known to occur near wpyr ≈ 0.006 at sigma=15°).
3. **Pre-cue saturation cliff**: the w_pyr value where `resting_hz` diverges
   (known at wpyr ≈ 0.008 for wpv=0.05).
4. **Quality score peak**: best `quality_score = bump_frac × (1 − sat_frac)` across
   the full (w_pv, w_pyr, amp) cube.

---

## Sweep 2A — Localisation regime with quiet baseline

**Motivation**: Phase 5 found a genuine localised bump at `wpyr=0.00592, wpv=0.05,
amp=0.55`, but the pre-cue baseline was 22 Hz (too high). Increasing w_pv should
suppress spontaneous activity, but requires higher amplitude to cross the bistable
threshold. This sweep zooms into that trade-off.

Run **after** Sweep 1 confirms the localization transition is in this region.

```bash
python3 -m circuit_model ring-calibrate \
  --params_json figs/optim/bistable_high_fr/best_params.json \
  --sigma_pyr_deg 15 \
  --w_pv_values   0.06 0.07 0.08 0.09 0.10 0.12 \
  --w_inter_values 0.004 0.0048 0.0055 0.006 0.0065 0.007 0.0075 0.008 0.009 \
  --amplitudes    0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.90 1.00 \
  --n_trials 40 \
  --n_workers 10 \
  --no_show
```

| Dimension | Values | Count |
|---|---|---|
| w_pv_global | 0.06 → 0.12 | 6 |
| w_pyr_pyr_inter | fine grid 0.004 → 0.009 | 9 |
| amplitude | 0.50 → 1.00 | 9 |
| n_trials | — | 40 |
| **Total trials** | 6 × 9 × 9 × 40 | **19 440** |

**Estimated runtime**: ~36 min total.

**Target region**: `resting_hz < 5 Hz` AND `delay_bump_frac > 0.6` AND
`delay_sat_frac < 0.15`. If both conditions can coexist, this is the working point.

---

## Sweep 2B — Fine transition zone (quiet-but-diffuse → localised)

**Motivation**: At lower w_pyr (0.003–0.006) and moderate w_pv (0.04–0.07), there
may be a narrow band where the bump is genuinely localised while the pre-cue rate
is still acceptably quiet (< 5–10 Hz). The big sweep will have 14 log-spaced points
across 0.001–0.020, which gives only ~3 points in this range. This sweep fills in
the transition with 10 fine-grained w_pyr values.

Run **after** Sweep 1, **only if** the quality-score peak in the summary heatmap
lies in the 0.003–0.007 w_pyr range at moderate w_pv.

```bash
python3 -m circuit_model ring-calibrate \
  --params_json figs/optim/bistable_high_fr/best_params.json \
  --sigma_pyr_deg 15 \
  --w_pv_values   0.04 0.05 0.055 0.06 0.065 0.07 \
  --w_inter_values 0.003 0.0035 0.004 0.0045 0.005 0.0055 0.006 0.0065 0.007 0.0075 \
  --amplitudes    0.45 0.50 0.55 0.60 0.65 0.70 0.75 \
  --n_trials 50 \
  --n_workers 10 \
  --no_show
```

| Dimension | Values | Count |
|---|---|---|
| w_pv_global | 0.04 → 0.07 (fine) | 6 |
| w_pyr_pyr_inter | 0.003 → 0.0075 (10 steps) | 10 |
| amplitude | 0.45 → 0.75 (bistable window) | 7 |
| n_trials | — | 50 |
| **Total trials** | 6 × 10 × 7 × 50 | **21 000** |

**Estimated runtime**: ~39 min total.

---

## Decision tree after Sweep 1

```
Sweep 1 result
│
├── Quality peak at wpyr > 0.007, high w_pv
│   └──▶ Run Sweep 2A (localization + low baseline)
│
├── Quality peak at wpyr 0.003–0.007, moderate w_pv (0.04–0.07)
│   └──▶ Run Sweep 2B (fine transition zone)
│
├── Both regions look promising
│   └──▶ Run both 2A and 2B (total ~75 min; can stagger or run overnight)
│
└── Quality is uniformly low across all w_pyr (no localization visible)
    └──▶ Check sigma_pyr_deg=10 or reduce delay_ms to 2000 for faster iteration
```

---

## Prior context (what Phase 5 established)

| Observation | Value |
|---|---|
| Bistable threshold (wpv=0.05) | amp ≈ 0.45–0.50 |
| Best delay rate, no localization | wpyr=0.002, wpv=0.05, amp=0.55 → 14 Hz, std=83° |
| Localization transition (sigma=15°) | wpyr ≈ 0.006 → std drops 111° → 36° |
| Best localised bump | wpyr=0.00592, wpv=0.05, amp=0.55 → 129 Hz, std=36°, **baseline=22 Hz** |
| Pre-cue saturation cliff | wpyr ≈ 0.008 at wpv=0.05 |

The 3D sweep will answer whether there exists a (w_pv, w_pyr, amp) triple with:
- Pre-cue baseline < 5 Hz
- delay_bump_frac > 0.6
- center_std < 45°
