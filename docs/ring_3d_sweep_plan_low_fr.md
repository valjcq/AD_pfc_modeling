# Ring Network Calibration (Low FR): Complete Guide to Parameter Sweeps and Results Assessment

> **Note**: This document is the companion to `ring_3d_sweep_plan.md`, adapted for the 
> `bistable_low_fr/best_params.json` parameter set. It follows the same systematic 
> approach but with adjusted ranges to account for lower baseline firing rates and faster adaptation.

## Problem Statement

With the bistable circuit params from `figs/optim/bistable_low_fr/best_params.json`, the
ring network presents different dynamics compared to `bistable_high_fr`:

**Key differences**:
- `I0_pyr = 0.507 nA` (vs 1.07 in high_fr) — weaker external drive
- `g_gaba_base = 4.465` (vs 1.19 in high_fr) — much stronger baseline inhibition
- `tau_adapt_pyr = 307 ms` (vs 1119 ms in high_fr) — faster adaptation

**Consequence**: Network naturally rests near LOW fixed point (~0 Hz) without strong w_pv tuning.

**Challenge**: Find cue amplitude and ring connectivity that produces a reliable, localized bump
without excessive saturation during the cue.

**Target behaviour**: Network rests near ~1–3 Hz baseline; cue triggers sustained HIGH state
(50–100 Hz) with tight localization (center_std < 50°).

---

## Overview: 3D Parameter Space and State Classification

State classification follows the same definitions as `ring_3d_sweep_plan.md`:

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
| `quality_score` | `bump_frac × (1 − sat_frac)` |

**A good working point**: `delay_bump_frac > 0.7`, `delay_sat_frac < 0.2`, `resting_hz < 5 Hz`.

---

## Systematic Phase-by-Phase Approach

### Circuit parameters

| JSON path | I0_pyr | g_gaba_base | Low FP | High FP | tau_adapt_pyr | Notes |
|---|---|---|---|---|---|---|
| `figs/optim/bistable_low_fr/best_params.json` | 0.507 | 4.465 | ~0 Hz | ~75 Hz | 307 ms | Weak drive, strong GABA, fast adaptation |
| `figs/optim/bistable_high_fr/best_params.json` | 1.070 | 1.191 | ~0 Hz | ~78 Hz | 1119 ms | Strong drive, moderate GABA, slow adaptation |

### Phases 0–3: Establishing the w_pv and amplitude ranges

**Goal**: Confirm baseline is naturally quiet, then identify the bistable threshold 
(minimum amplitude needed to trigger HIGH state).

**CRITICAL FINDING**: w_pv_global=0.01 is **too strong** — the network remains silent 
throughout the entire trial, even with amplitude=0.5. The cue fails to trigger the bistable transition.

**Verified result**:

```bash
python3 -m circuit_model ring-run \
  --params_json figs/optim/bistable_low_fr/best_params.json \
  --sigma_pyr_deg 15 \
  --w_pv_global 0.01 \
  --w_pyr_pyr_inter 0.002 \
  --amplitude 0.5 \
  --delay_ms 3000 \
  --output_dir figs/ring/calibration/low_fr_phases/phase2_wpv_0.01
```

**Result**: 
- Pre-cue: ~0 Hz ✓
- Cue response: ~0 Hz ✗ (no bump triggered)
- Delay: ~0 Hz ✗ (no sustained activity)

**Interpretation**: The strong GABA inhibition (g_gaba_base=4.465) in low_fr, combined with 
w_pv=0.01, completely suppresses network excitability. The external drive is too weak 
(I0_pyr=0.507 vs high_fr's 1.070) to overcome the inhibitory tone.

**Implication**: **Lower w_pv values will not work.** We need to reduce w_pv significantly below 0.01, 
or the network needs even higher amplitudes to overcome the suppression.

**Strategy change**: Phase 4 should test w_pv **below 0.01** (e.g., 0.001, 0.003, 0.005, 0.008) 
with higher amplitude ranges to find where bump formation begins.

---

## Phase 4: 2D Sweep — w_pv_global × amplitude

**Goal**: Find the bistable boundary by testing very low w_pv values with increasing amplitudes.

**REVISED Strategy**: Since w_pv=0.01 completely suppresses the network even with amp=0.5:
- Test **much lower w_pv values** (0.0001, 0.0003, 0.001, 0.003, 0.005)
- Use **higher amplitudes** (0.5, 1.0, 1.5, 2.0, 2.5) to overcome stronger inhibition
- Goal: Find the w_pv/amplitude pair that produces _first detectable bump_

**Hypothesis**: Low_fr's strong GABA means the bistable threshold is shifted to require either:
- Much weaker w_pv (allowing spontaneous activity) OR
- Much stronger amplitude (overcoming inhibition)

```bash
python3 -m circuit_model ring-calibrate \
  --params_json figs/optim/bistable_low_fr/best_params.json \
  --sigma_pyr_deg 15 \
  --w_pv_values   0.0001 0.0003 0.001 0.003 0.005 0.008 \
  --w_inter_values 0.002 \
  --amplitudes    0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 \
  --n_trials 15 \
  --n_workers 10 \
  --output_dir figs/ring/calibration/128_sigma_15_low_fr_phase4 \
  --no_show
```

**Output**: `figs/ring/calibration/128_sigma_15_low_fr_phase4/`

### Phase 4 Results: Low_fr **REQUIRES EXTREME AMPLITUDES**

**Initial Phase 4** (amp = 0.5–2.5×): NO activity detected.

**Extended testing** (amp = 1.5–10×): **Sharp bifurcation discovered at amplitude ~7.0×**

| Amplitude | w_pv=0.0001 | w_pv=0.001 | w_pv=0.005 | Status |
|---|---|---|---|---|
| 5.0 | ~1e-22 | ~1e-22 | ~1e-22 | SILENT |
| 6.0 | ~1e-11 | ~1e-11 | ~1e-12 | Silent |
| 6.5 | ~1e-6 | ~1e-6 | ~1e-7 | Threshold region |
| **7.0** | **0.031** | **0.026** | **0.011** | **← BIFURCATION** |
| 7.5 | **0.82** | **0.80** | **0.74** | Active |
| 10.0 | **0.99** | **0.99** | **0.99** | Near-saturated |

**Visual evidence — bifurcation heatmap**:

![Phase 4: Amplitude sweep showing bifurcation at 6.5-7.0×](../figs/ring/calibration/128_sigma_15_low_fr_threshold/w_pv_0.0001/bump_decay_amp_sweep.png)

**Key finding**: The network **IS responsive** — but the bistable threshold is shifted to **amplitude ~7.0×** 
instead of the ~0.5–1.0× seen in high_fr.

**Root cause of the extreme threshold shift**:

| Parameter | low_fr | high_fr | Effect |
|---|---|---|---|
| I0_pyr | 0.507 | 1.070 | Low_fr drive is **2.1× weaker** |
| g_gaba_base | 4.465 | 1.191 | Low_fr GABA is **3.7× stronger** |
| Combined effect | — | — | **~7-14× amplification of bistable threshold** |

**Conclusion**: Low_fr **CAN form ring bumps**, but only with unphysically high stimulus amplitudes 
(7–10× I_ext_pyr vs. 0.5–1.0× for high_fr). This is a fundamental trade-off:
- ✅ Achieves quiet baseline (<1 Hz pre-cue)  
- ❌ Requires extreme input (7–10× normal) to activate

**Practical implications**:
- Low_fr is **technically viable** for ring networks  
- Low_fr is **impractical** for realistic stimulus amplitudes  
- **Use high_fr for ring networks**; reserve low_fr for **single-node only**

---

## Phase 5: w_pyr_pyr_inter × amplitude sweep — COMPLETED (negative result)

**Status**: Executed. Hypothesis **falsified**.

**Hypothesis tested**: Higher `w_pyr_pyr_inter` reduces the bifurcation threshold from 7.0× 
toward biologically plausible amplitudes (1–3×).

**Result**: `w_pyr_pyr_inter` has **zero effect** on the bifurcation threshold.

### Key data

At amplitude 7.0×, `ref_amplitude` is **identical across all w_inter values** (0.001 → 0.040):

| w_pyr_pyr_inter | ref_amplitude (amp=7.0×) | Active? |
|---|---|---|
| 0.001 | 3.105e-02 | No |
| 0.002 | 3.105e-02 | No |
| 0.004 | 3.105e-02 | No |
| ... | 3.105e-02 | No |
| 0.040 | 3.105e-02 | No |

The `ref_amplitude` across all amplitudes at fixed w_inter=0.002 shows the same exponential 
ramp as before — completely insensitive to w_inter:

| Amplitude | ref_amplitude |
|---|---|
| 1.0× | ~1e-58 |
| 3.0× | ~2e-42 |
| 5.0× | ~1e-21 |
| 6.0× | ~2e-11 |
| 7.0× | ~3e-02 |

(The jump from 6→7 is 9 orders of magnitude — a single-node bifurcation signature.)

### Why the hypothesis was wrong

The bifurcation threshold is a **single-node phenomenon**:
- It depends on whether the external current can push an isolated node from the LOW to the HIGH 
  fixed point against its GABA inhibition
- `w_pyr_pyr_inter` carries signal *between* nodes, but until a node is already active there 
  is no signal to propagate — it provides zero help crossing the energy barrier
- The exponential ramp (`1e-58 → 1e-2`) is the signature of a single-node barrier: w_inter 
  cannot add to a signal that is essentially zero

**Consequence**: To lower the bifurcation threshold, one must modify the **single-node 
parameters** (I0_pyr, g_gaba_base), not the ring connectivity.

### Scripts (archived, not needed again)

```bash
# Sweep (already run)
.venv/bin/python3 scripts/run_low_fr_wInter_sweep.py

# Analysis (already run)
.venv/bin/python3 scripts/analyze_low_fr_wInter_sweep.py \
    --sweep_dir figs/ring/calibration/128_sigma_15_low_fr_wInter_amp_sweep
```

**Outputs**: `summary.json`, `threshold_curve.png`, `heatmap_WT_wpv_*.png`  
in `figs/ring/calibration/128_sigma_15_low_fr_wInter_amp_sweep/`

---

## 3D Sweep 1 — Comprehensive coverage [CANCELLED]

**Status**: Permanently cancelled. Phase 5 shows that varying ring connectivity cannot lower 
the bistable threshold; the 3D sweep would find the same result everywhere.

---

## Phase 6: Comprehensive w_pv × w_inter sweep at fixed amplitude 7.0× — ACTIVE

**Context**: The threshold is fixed at ~7.0× by single-node circuit params (Phase 5 confirmed
w_inter has zero effect on the bifurcation threshold). We now fix amplitude = 7.0× and map
the full **w_pv_global × w_pyr_pyr_inter** parameter space in one calibration run.

**Rationale for amplitude 7.0×**: This is just above the bifurcation — the regime where
connectivity choices matter most for whether the bump is sustained or decays.

### Sweep design

| Axis | Values | Rationale |
|---|---|---|
| `amplitude` | **7.0** | Fixed at bifurcation threshold |
| `w_pyr_pyr_inter` | 0.002, 0.004, 0.006, 0.008, 0.012, 0.016, 0.020, 0.024 | Sub- to supra-Turing range, extended beyond Phase 6 |
| `w_pv_global` | 0.0001, 0.0003, 0.001, 0.003, 0.005 | Wide inhibitory sweep |

Total: 5 × 8 × 1 × 20 trials = **800 simulations** (~10–15 min on 10 workers).

### Running the sweep (single command)

```bash
python3 -m circuit_model ring-calibrate \
  --params_json figs/optim/bistable_low_fr/best_params.json \
  --sigma_pyr_deg 15 \
  --w_pv_values   0.0001 0.0003 0.001 0.003 0.005 \
  --w_inter_values 0.002 0.004 0.006 0.008 0.012 0.016 0.020 0.024 \
  --amplitudes    7.0 \
  --n_trials 20 \
  --n_workers 10 \
  --output_dir figs/ring/calibration/128_sigma_15_low_fr_phase6 \
  --no_show
```

**Output structure**:
```
figs/ring/calibration/128_sigma_15_low_fr_phase6/
  calibration_heatmap_WT_amp7.png     ← combined 2×2 heatmap (main result)
  w_pv_0.0001/
    bump_decay_trials.csv
    bump_decay_amp_sweep.png
    WT/bump_decay_heatmap.png
  w_pv_0.0003/  ...
  ...
```

The **combined heatmap** (`calibration_heatmap_WT_amp7.png`) is generated automatically
at the end of the sweep. It shows 4 panels in a 2×2 grid (axes: X = w_inter, Y = w_pv):

| Panel | Metric | Interpretation |
|---|---|---|
| Top-left | `ref_amplitude` (log scale) | Initial bump strength right after cue |
| Top-right | `frac_valid` | Fraction of trials where bump was detected |
| Bottom-left | `end_val` (normalised) | Sustained bump at end of delay |
| Bottom-right | `end_val_std` | Trial-to-trial variability |

### Results

_(to be filled after sweep runs)_

**Best working point**: TBD — inspect `calibration_heatmap_WT_amp7.png`.

### Visual inspection at best working point

```bash
python3 -m circuit_model ring-run \
  --params_json figs/optim/bistable_low_fr/best_params.json \
  --sigma_pyr_deg 15 \
  --w_pyr_pyr_inter <best_w_inter> \
  --w_pv_global <best_w_pv> \
  --amplitude 7.0 \
  --delay_ms 5000 \
  --output_dir figs/ring/run/low_fr_best_phase6
```

---

## Diagnostic checklist — How to assess sweep results

### Visual inspection in the output figures

| File | What to look at | Good sign | Bad sign |
|---|---|---|---|
| `dashboard.png` | Heatmap visualization | Dark pre-cue, bright localized stripe at cue angle during delay | All nodes bright pre-cue, or opposite node bright post-cue |
| `population_activity.png` | Per-node firing trace | Pre-cue: ~0–3 Hz. Post-cue: sharp spike to ~50–150 Hz at single location. Delay: maintained | Pre-cue: high baseline; Post-cue: global saturation; Delay: rapid decay |
| `bump_metrics_over_time.png` | Amplitude and center_std over time | `amplitude` rises at cue onset, plateaus during delay; `center_std` stays low (<50°) | Amplitude decays during delay, or center_std increases |

### Metric-based interpretation

| Metric | Target | Too low | Too high |
|---|---|---|---|
| `baseline_pyr` (pre-cue) | 0–3 Hz | — | >10 Hz indicates excessive spontaneous activity |
| `delay_bump_frac` | >0.7 | <0.3: bump unreliable | >0.95: check quality |
| `delay_sat_frac` | <0.1 | — | >0.2: frequent saturation |
| `quality_score` | >0.6 | <0.4: poor regime | — |
| `center_std` | <45° | <15°: may be artifactual if baseline is zero | >80°: non-localized |
| `cue_peak_hz` | 80–150 Hz | <50: cue too weak | >180: approach saturation |

### Pattern-based diagnosis

| Pattern observed | Meaning | Fix |
|---|---|---|
| All nodes bright before cue | w_pv_global too small | Increase w_pv_global |
| Bump absent despite good pre-cue | w_pv_global too large OR w_pyr too small | Reduce w_pv_global or increase w_pyr_inter |
| peak_pyr_cue > 160 Hz | Amplitude too large | Lower amplitude (this is faster-adapting variant) |
| Bump decays within 1–2 s | w_pyr_pyr_inter too small | Increase w_pyr_pyr_inter |
| Bump spreads to entire ring | w_pyr_pyr_inter too large | Reduce w_pyr_pyr_inter, or increase w_pv_global |
| High baseline (>5 Hz) + good delay bump | w_pyr_pyr_inter allows spontaneous activity | Reduce w_pyr_pyr_inter slightly, or adjust w_pv |

---

## Transient Robustness Testing: Duration × Amplitude Sweep [OPTIONAL]

Once a good working point is identified, validate it with transient perturbations.

**Experimental design**: Same as `ring_3d_sweep_plan.md`, adapted to best working point for low_fr.

```bash
# Run 2D sweep (output in figs/ring/run/transient_sweep_low_fr/)
.venv/bin/python3 scripts/run_transient_sweep.py \
  --params_json figs/optim/bistable_low_fr/best_params.json \
  --sigma_pyr_deg 15 \
  --w_pyr_pyr_inter [best_value] \
  --w_pv_global [best_value] \
  --amplitude [best_value] \
  --workers 10 \
  --output_dir figs/ring/run/transient_sweep_low_fr

# Analyze and generate heatmaps
.venv/bin/python3 scripts/analyze_transient_sweep.py \
  --output_dir figs/ring/run/transient_sweep_low_fr \
  --save figs/ring/run/transient_sweep_low_fr/heatmap_2d.png --no-show
```

**Expected outcome**: 
- Negative transients with low amplitude (~0.1) should cleanly suppress the bump
- Positive transients with duration >300 ms may suppress via exhaustion
- Network should not spiral into chaotic bumps during perturbations

---

## Summary: Current Understanding and Next Steps

| Experiment | Result | Implication |
|---|---|---|
| **Phase 4a**: amp=0.5–2.5× | No activity | Bistable threshold too high |
| **Phase 4b**: amp=1.5–10× | **Bifurcation at 6.5–7.0×** | Network IS responsive |
| **Phase 5**: w_inter 0.001→0.040 at amp=7× | **Zero effect on threshold** | Threshold is single-node, not connectivity |
| **Phase 6**: w_inter × w_pv × amp 7–8× | **w_inter=0.016, amp=8×, w_pv=0.0001** → end_val=0.867 | Best working point found |

**Key insight**: The 7–8× amplitude threshold is locked in by the single-node circuit (weak 
I0_pyr, strong g_gaba_base). Ring connectivity shapes bump persistence once active. Sustained 
bumps require amplitude ≥ 8× and strong recurrent coupling (w_inter ≥ 0.016).

### Key parameter differences driving the problem

| Parameter | low_fr | high_fr | Effect on bump formation |
|---|---|---|---|
| `I0_pyr` | 0.507 | 1.070 | 2.1× weaker drive in low_fr |
| `g_gaba_base` | 4.465 | 1.191 | **3.7× stronger GABA in low_fr** ← critical |
| `tau_adapt_pyr` | 307 ms | 1119 ms | Faster decay (not enough to compensate) |

**The problem**: The 3.7× stronger GABA combined with 2.1× weaker external drive creates a 
"cliff" — the network is either completely suppressed (w_pv > ~0.002) or has spontaneous 
activity (w_pv < ~0.001). The bistable threshold may not exist in a usable regime.

### Expected outcomes from Phase 4

**Scenario A (optimistic)**: Find viable bump at low w_pv + high amplitude
- Would require accepting pre-cue baseline ~5–10 Hz (spontaneous activity)
- Amplitude needs 1.5–2.5× higher than high_fr

**Scenario B (pessimistic)**: No viable bump across tested range
- Would suggest low_fr parameters are fundamentally incompatible with ring attractor
- Root cause: optimization for low baseline firing suppressed network excitability too much

---

## Quick reference: When to Use Low_fr vs High_fr

### Current understanding

Low_fr **CAN** form ring bumps, but the bistable threshold is shifted from ~0.5–1.0× (high_fr) 
to **~7.0× amplitude** (at default w_pyr_pyr_inter = 0.002). Phase 5 tests whether increasing 
`w_pyr_pyr_inter` can lower this threshold to biologically plausible range.

### When to use each parameter set

| Parameter Set | Single-node use | Ring network use | Notes |
|---|---|---|---|
| `bistable_low_fr` | ✅ **Excellent** (baseline ~0 Hz) | ⚠️ **Conditional** | Threshold at 7× unless Phase 5 finds viable regime |
| `bistable_high_fr` | ✅ Good (baseline ~3–5 Hz) | ✅ **Recommended** (amp=0.5–1.0×) | Realistic stimulus range |

**Current recommendation** (pending Phase 5): 
- **For ring networks**: Use **high_fr** until Phase 5 results are available
- **For single-node quiet operation**: Use **low_fr**

### Key Experiments Executed

```bash
# Phase 4a: Initial sweep (amp = 0.5–2.5×)
python3 -m circuit_model ring-calibrate \
  --params_json figs/optim/bistable_low_fr/best_params.json \
  --sigma_pyr_deg 15 \
  --w_pv_values 0.0001 0.0003 0.001 0.003 0.005 0.008 \
  --w_inter_values 0.002 \
  --amplitudes 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 \
  --n_trials 15 --n_workers 10 \
  --output_dir figs/ring/calibration/128_sigma_15_low_fr_phase4 --no_show
```

**Result**: No activity detected.

```bash
# Extended testing: High amplitudes (amp = 1.5–10×)
python3 -m circuit_model ring-calibrate \
  --params_json figs/optim/bistable_low_fr/best_params.json \
  --sigma_pyr_deg 15 \
  --w_pv_values 0.0001 0.001 0.005 \
  --w_inter_values 0.002 \
  --amplitudes 1.5 2.0 2.5 3.0 4.0 5.0 7.5 10.0 \
  --n_trials 10 --n_workers 10 \
  --output_dir figs/ring/calibration/128_sigma_15_low_fr_high_amp --no_show
```

**Result**: Bifurcation found at amplitude 6.5–7.0×.

```bash
# Threshold refinement (amp = 5.0–7.5×)
python3 -m circuit_model ring-calibrate \
  --params_json figs/optim/bistable_low_fr/best_params.json \
  --sigma_pyr_deg 15 \
  --w_pv_values 0.0001 0.001 0.005 \
  --w_inter_values 0.002 \
  --amplitudes 5.0 5.5 6.0 6.5 7.0 7.5 \
  --n_trials 10 --n_workers 10 \
  --output_dir figs/ring/calibration/128_sigma_15_low_fr_threshold --no_show
```

**Result**: Clear threshold at amplitude 7.0×. Phases 5 and 3D Sweep 1 not needed (parameters identified).

---

## Analysis: The Extreme Amplitude Trade-off in Low_fr

### Why low_fr requires 7–14× higher amplitudes

Low_fr was optimized to **minimize baseline firing rates** via:
- **Reduced external drive**: `I0_pyr = 0.507` (vs high_fr: 1.070) = **2.1× weaker**
- **Massively increased GABA**: `g_gaba_base = 4.465` (vs high_fr: 1.191) = **3.7× stronger**

**Trade-off achieved**:
- ✅ Pre-cue baseline: ~0 Hz (extremely quiet)
- ✅ Single-node bistability: Works well for that purpose

**Trade-off cost**:
- ❌ Bistable threshold: Shifted from 0.5–1.0× to **7.0× amplitude**
- ❌ Ring network feasibility: Still technically viable, but unrealistic

### Why the 7–14× amplification?

1. **Direct effect**: 2.1× weaker I0_pyr means direct drive is halved
2. **Inhibitory effect**: 3.7× stronger g_gaba means network-wide inhibition is overwhelming
3. **Recurrent suppression**: Even PYR→PYR coupling can't sustain activity against that much GABA
4. **Combined effect**: These multiply, requiring 7–14× stronger external input to overcome

### What Phase 4 revealed

**Phase 4a** (amp = 0.5–2.5×): No activity — network completely suppressed

**Phase 4b** (amp = 7.0–10×): **Bifurcation at 7.0×** — network suddenly responsive

This is **NOT** incompatibility (like we initially thought). It's **threshold amplification**.

### Can this be fixed?

**For single-node use**: No fix needed; low_fr is **perfect** for quiet baseline operation.

**For ring networks**: Would require reoptimizing I0_pyr and g_gaba_base to lower values, 
which would **destroy the single-node quiet baseline**. The parameters are optimized for 
mutually exclusive goals.

**Practical choice**: Use the appropriate parameter set for the task:
- Ring networks → **high_fr** (amp = 0.5–1.0×)
- Single-node quiet → **low_fr** (amplitude irrelevant for bistability)

---

## Document status

- [x] Structure and command outline created
- [x] **Phase 0–2**: w_pv=0.01 completely suppresses network (as expected)
- [x] **Phase 4a**: Initial sweep (amp=0.5–2.5×) shows no activity
- [x] **Phase 4b (KEY BREAKTHROUGH)**: Extended sweep finds bifurcation at amp=7.0×!
- [x] **Phase 4c**: Threshold refinement confirms 6.5–7.0× as bistable crossing point
- [x] **CONCLUSIVE ANSWER**: Low_fr CAN form ring bumps, requires unrealistically high amplitudes (7×)
- [x] **Phase 5**: w_pyr_pyr_inter × amplitude sweep — **NEGATIVE RESULT**
  - w_pyr_pyr_inter (0.001 → 0.040) has **zero effect** on bifurcation threshold
  - Root cause: threshold is a single-node saddle-node bifurcation, not connectivity-dependent
- [ ] **Phase 6**: w_pv × w_inter sweep at fixed amplitude 7.0× — single `ring-calibrate` command
  - Output: `figs/ring/calibration/128_sigma_15_low_fr_phase6/calibration_heatmap_WT_amp7.png`
  - Figures: 2×2 combined heatmap (ref_amplitude, frac_valid, end_val, end_val_std)
- [ ] **Visual inspection**: `ring-run` at best working point to check localization and dynamics
