# Article Execution Plan — Detailed & Actionable

**Working title**: *Interneuron-class-specific control of persistent activity in a prefrontal ring attractor: implications for Alzheimer's disease*

**Purpose of this document**: Exhaustive, agent-ready task list. Each block specifies *what* to run, *what* outputs to expect, *what* metrics to extract, *why* it matters, and *how* to interpret results. Organized by article section.

---

## PHASE 0 — WT and WT_APP Parameter Optimization

> **Goal**: Achieve converged parameter fits where all 4 populations match spike rate targets. This is the foundation for everything downstream.

---

### Task 0.2 — WT and WT_APP 1mo optimization

**Data source**: `AD_data/AD_spikes/datafiles/firing_rate_data.csv`, column `per_neuron_mean`.

**Targets**:

| Population | WT (1mo) | WT_APP (1mo) | Source genotype (CSV) |
|---|---|---|---|
| PYR | 8.214 | 12.466 | WT / WT-APP |
| SOM | 4.295 | 4.814 | SST-Cre / SST-Cre_APP |
| PV  | 4.073 | 4.241 | PV-Cre_control / PV-Cre_APP |
| VIP | 6.051 | 5.551 | VIP-Cre_control / VIP-Cre_APP |

**Optional KO PYR constraints** (add `--target_*_ko_pyr` flags to constrain KO populations simultaneously):

| KO | WT background | WT_APP background | Source (CSV) |
|---|---|---|---|
| α7-KO | 17.539 | 13.599 | a7_KO_control / a7_KO_APP |
| β2-KO | 17.965† | 19.109 | b2_KO_control† / b2_KO_APP |
| α5-KO | 9.285  | 3.113  | a5_KO / a5_KO_APP |

† b2_KO_control: per-neuron data truncated in source CSV; value is `sampled_mean`.

---

### Transfer function parameter convention

All transfer function shape parameters are fixed to the W&W 2006 values and **frozen** in every optimization run. Only the output-scaling factors `A_x` remain free.

| Parameter | Code field | Fixed value | Source |
|-----------|-----------|-------------|--------|
| PYR gain | `alpha_pyr` | 310 Hz/nA | W&W 2006 |
| PV/SOM/VIP gain | `alpha_pv`, `alpha_som`, `alpha_vip` | 615 Hz/nA | W&W 2006 |
| PYR threshold | `Theta_pyr` | 0.40323 nA (= 125/310) | W&W 2006 |
| PV/SOM/VIP threshold | `Theta_pv`, `Theta_som`, `Theta_vip` | 0.28780 nA (= 177/615) | W&W 2006 |
| PYR curvature | `g_exc` | 0.16 s | W&W 2006 |
| SOM/PV/VIP curvature | `g_inh` | 0.087 s | W&W 2006 |
| Synaptic time constant | `tau_s` | 20 ms | Beierlein 2003 |
| Ring connectivity width | `sigma_pyr_deg` | 15° | fixed protocol |
| PYR adaptation time constant | `tau_adapt_pyr` | 600 ms | Storm 1989; frozen to prevent optimizer from using slow adaptation as a substitute for genuine bistability |

---

The optimization workflow follows a **two-stage approach** (decided 2026-03-31):

- **Stage A** — Single-node optimization (`circuit_model optimize`): fits CircuitParams only, no ring overhead.
- **Stage B** — Ring calibration sweep (`circuit_model ring-calibrate`): sweeps RingParams only, with CircuitParams frozen from Stage A.

**Rationale**:
- Single-node fitting is faster, cleaner, and decoupled from ring effects.
- Ring calibration sweeps are more interpretable than joint optimization for finding bump formation/sustainment thresholds.
- The Turing loss, while useful as a guide, cannot guarantee bump persistence — simulation-based calibration is the ground truth.

---

### Stage A — Single-node optimization

#### WT — single-node fit (base rates only)

```bash
python -m circuit_model optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --optimizer chaining --n_samples 50000 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr" \
  --save_best_json params/new/single_node/WT_1mo_article.json \
  --log_file figs/optim/1mo/single_node/log.jsonl
```

Optional (right after single-node): joint ring optimization with the same targets

```bash
python -m circuit_model ring-optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --params_json params/new/single_node/WT_1mo_article.json \
  --optimizer chaining --n_samples 50000 \
  --turing_weight 1.0 --turing_margin 0.05 --turing_cue_scale 1.4 \
  --spatial_uniformity_weight 1.0 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr" \
  --save_best_circuit_json params/new/ring_optimize/WT_1mo_article_ring_opt_circuit.json \
  --save_best_ring_json params/new/ring_optimize/WT_1mo_article_ring_opt_ring.json \
  --log_file figs/optim/1mo/ring_optimize/log.jsonl
```

#### WT — single-node fit + KO constraints

```bash
python -m circuit_model optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --target_alpha7_ko_pyr 17.539 --target_beta2_ko_pyr 17.965 --target_alpha5_ko_pyr 9.285 \
  --optimizer chaining --n_samples 50000 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr" \
  --save_best_json params/new/single_node/WT_1mo_article_ko.json \
  --log_file figs/optim/1mo_ko/single_node/log.jsonl
```

Optional (right after single-node): joint ring optimization with KO targets

```bash
python -m circuit_model ring-optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --target_alpha7_ko_pyr 17.539 --target_beta2_ko_pyr 17.965 --target_alpha5_ko_pyr 9.285 \
  --params_json params/new/single_node/WT_1mo_article_ko.json \
  --optimizer chaining --n_samples 50000 \
  --turing_weight 1.0 --turing_margin 0.05 --turing_cue_scale 1.4 \
  --spatial_uniformity_weight 1.0 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr" \
  --save_best_circuit_json params/new/ring_optimize/WT_1mo_article_ko_ring_opt_circuit.json \
  --save_best_ring_json params/new/ring_optimize/WT_1mo_article_ko_ring_opt_ring.json \
  --log_file figs/optim/1mo_ko/ring_optimize/log.jsonl
```

#### WT_APP — single-node fit (base rates only)

```bash
python -m circuit_model optimize \
  --target_pyr 12.466 --target_som 4.814 --target_pv 4.241 --target_vip 5.551 \
  --optimizer chaining --n_samples 50000 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr" \
  --save_best_json params/new/single_node/WT_APP_1mo_article.json \
  --log_file figs/optim/1mo_APP/single_node/log.jsonl
```

Optional (right after single-node): joint ring optimization with the same targets

```bash
python -m circuit_model ring-optimize \
  --target_pyr 12.466 --target_som 4.814 --target_pv 4.241 --target_vip 5.551 \
  --params_json params/new/single_node/WT_APP_1mo_article.json \
  --optimizer chaining --n_samples 50000 \
  --turing_weight 1.0 --turing_margin 0.05 --turing_cue_scale 1.4 \
  --spatial_uniformity_weight 1.0 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr" \
  --save_best_circuit_json params/new/ring_optimize/WT_APP_1mo_article_ring_opt_circuit.json \
  --save_best_ring_json params/new/ring_optimize/WT_APP_1mo_article_ring_opt_ring.json \
  --log_file figs/optim/1mo_APP/ring_optimize/log.jsonl
```

#### WT_APP — single-node fit + KO constraints

```bash
python -m circuit_model optimize \
  --target_pyr 12.466 --target_som 4.814 --target_pv 4.241 --target_vip 5.551 \
  --target_alpha7_ko_pyr 13.599 --target_beta2_ko_pyr 19.109 --target_alpha5_ko_pyr 3.113 \
  --optimizer chaining --n_samples 50000 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr" \
  --save_best_json params/new/single_node/WT_APP_1mo_article_ko.json \
  --log_file figs/optim/1mo_APP_ko/single_node/log.jsonl
```

Optional (right after single-node): joint ring optimization with KO targets

```bash
python -m circuit_model ring-optimize \
  --target_pyr 12.466 --target_som 4.814 --target_pv 4.241 --target_vip 5.551 \
  --target_alpha7_ko_pyr 13.599 --target_beta2_ko_pyr 19.109 --target_alpha5_ko_pyr 3.113 \
  --params_json params/new/single_node/WT_APP_1mo_article_ko.json \
  --optimizer chaining --n_samples 50000 \
  --turing_weight 1.0 --turing_margin 0.05 --turing_cue_scale 1.4 \
  --spatial_uniformity_weight 1.0 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr" \
  --save_best_circuit_json params/new/ring_optimize/WT_APP_1mo_article_ko_ring_opt_circuit.json \
  --save_best_ring_json params/new/ring_optimize/WT_APP_1mo_article_ko_ring_opt_ring.json \
  --log_file figs/optim/1mo_APP_ko/ring_optimize/log.jsonl
```

**Stage A acceptance criteria (check before proceeding to Stage B):**
- [ ] PYR, SOM, PV, VIP base rates within 10% of targets
- [ ] alpha7_ko / alpha5_ko / beta2_ko PYR within 15% of targets (KO fits only)
- [ ] No Jacobian entries above 5
- [ ] w_pe >= 0.05 nA/Hz  (PYR→PV not collapsed)
- [ ] w_se >= 0.003 nA/Hz (PYR→SOM not collapsed)
- [ ] Neither w_pe nor w_se saturated at lower bound in [BOUNDS-FINAL]

---

### Stage B — Ring Calibration

#### Step B1 — w_pyr_pyr_inter sweep (WT, base fit)

`ring-calibrate` sweeps a 2D grid of (amplitudes × w_inter_values). Fix amplitude
and provide an explicit list of w_inter values to sweep:

```bash
python -m circuit_model ring-calibrate \
  --params_json params/new/single_node/WT_1mo_article.json \
  --w_pv_global 0.017 \
  --amplitudes 2.0 \
  --w_inter_min 0.001 --w_inter_max 0.05 --n_inter 20 \
  --n_nodes 64 \
  --n_trials 50
```

Note: amplitudes=2.0 = +100% of baseline I0_pyr, consistent with Turing cue
scale. w_pv_global=0.017 is the reference value from previous ring-optimize.

#### Step B2 — w_pv_global sweep (WT, at bump-forming w_inter)

`ring-calibrate` does not support sweeping w_pv_global directly. Run it
repeatedly with different `--w_pv_global` values using a shell loop.
Replace <W_INTER> with the minimum w_pyr_pyr_inter that produced a
self-sustained bump in Step B1:

```bash
for wpv in 0.005 0.008 0.011 0.014 0.017 0.020 0.023 0.026 0.030 0.035 0.040 0.045 0.050; do
  python -m circuit_model ring-calibrate \
    --params_json params/new/single_node/WT_1mo_article.json \
    --w_pv_global $wpv \
    --w_inter_values <W_INTER> \
    --amplitudes 2.0 \
    --n_nodes 64 \
    --n_trials 50
done
```

Repeat Steps B1 and B2 for WT_APP using
  --params_json params/new/single_node/WT_APP_1mo_article.json

**Stage B acceptance criteria:**
- [ ] Bump forms at the selected w_pyr_pyr_inter (non-zero amplitude post-cue)
- [ ] Bump self-sustains for >= 2000 ms after cue offset
- [ ] No spontaneous bump forms during burn-in
- [ ] Node-averaged ring firing rates remain within 15% of single-node targets
- [ ] No runaway to 200 Hz cap

---

### Task 0.3 — Validate fit

**What**: After optimization converges, validate the fitted parameters on the ring.

**Ring sanity check** (128 nodes, 2000 ms delay) — run for both KO and KO_APP conditions:

```bash
python -m circuit_model ring-run \
  --condition WT --n_nodes 128 --delay_ms 2000 --amplitude 1.1

python -m circuit_model ring-run \
  --condition WT_APP --n_nodes 128 --delay_ms 2000 --amplitude 1.1
```

Verify: bump forms and is stable, all 4 populations fire at reasonable rates, no population is silent or saturating.

**Biological plausibility check**:
- Transfer function thresholds should be positive
- Synaptic weights should be positive
- All receptor currents should be positive

**Metrics to extract and report** (for Fig 2):
- Per-population firing rate comparison: simulated vs experimental (box plot or bar chart)
- Per-condition: PYR rate comparison
- Relative error per population and per condition

**Figure destination**: Fig 2 (fit validation panel)

---

## PHASE 1 — WT and WT_APP Baseline Characterization (Article Section 3)

> **Goal**: Fully characterize both WT and WT_APP bumps side by side. This is the primary comparison of the paper — every analysis is run with both parameter sets to directly compare healthy vs disease circuit.
>
> **PREREQUISITE**: Task 0.3 complete — fits validated.
>
> **Ring parameter convention**:
> - WT commands: `--amplitude AMP_WT --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15`
> - WT_APP commands: `--amplitude AMP_APP --w_pyr_pyr_inter W_INTER_APP --w_pv_global W_PV_APP --sigma_pyr_deg 15`
> - KO commands (Phase 2): WT family + WT calibration values + `--condition a7_KO` etc.
> - KO_APP commands (Phase 3): WT_APP family + WT_APP calibration values + `--condition a7_KO_APP` etc.

### Task 1.1 — Bump baseline ring-run (WT and WT_APP)

**What**: Generate the canonical bump visualization for both conditions for the paper.

**Commands**:
```bash
# WT with 128 nodes
python -m circuit_model ring-run \
  --condition WT \
  --amplitude 0.1 --delay_ms 8000 \
  --n_nodes 128

# WT with 64 nodes
python -m circuit_model ring-run \
  --condition WT \
  --amplitude 1.1 --delay_ms 8000 \
  --n_nodes 64

# WT with 128 nodes and higher amplitude
python -m circuit_model ring-run \
  --condition WT \
  --amplitude 1.2 --delay_ms 8000 \
  --n_nodes 128

# WT with 128 nodes and lower amplitude
python -m circuit_model ring-run \
  --condition WT \
  --amplitude 1.3 --delay_ms 8000 \
  --n_nodes 128

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
  --n_trials 500 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --min_freq_hz 1.0 --max_freq_hz 15.0 \
  --tf_window_s 1.0 --tf_overlap 0.8 \
  --n_workers 8

# WT_APP
python -m circuit_model ring-oscillation-study \
  --conditions WT \
  --amplitudes AMP_APP \
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
  --n_trials 200 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --min_freq_hz 1.0 --max_freq_hz 15.0 \
  --n_workers 8

# WT_APP
python -m circuit_model ring-oscillation-study \
  --conditions WT \
  --amplitudes 2 4 6 8 10 12 15 18 22 27 33 40 \
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
  --amplitudes 2 4 6 8 10 12 15 18 22 27 33 40 \
  --n_trials 500 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --correct_asymmetry \
  --n_workers 8

# WT_APP — amplitude sweep
python -m circuit_model ring-asymmetry \
  --conditions WT \
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
  --amplitude AMP_WT \
  --n_trials 500 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --n_workers 8

# WT_APP
python -m circuit_model ring-diffusion \
  --conditions WT \
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
  --delay_ms 15000 \
  --n_trials 200 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --n_workers 8

# WT_APP
python -m circuit_model ring-bump-decay-study \
  --conditions WT \
  --amplitudes 2 4 6 8 10 12 15 18 22 27 33 40 \
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
  --n_trials 500 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --min_freq_hz 1.0 --max_freq_hz 15.0 \
  --n_workers 8

# Amplitude sweep per KO
python -m circuit_model ring-oscillation-study \
  --conditions WT a7_KO b2_KO a5_KO \
  --amplitudes 2 4 6 8 10 12 15 18 22 27 33 40 \
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

> **Goal**: Characterize the additional effect of the WT_APP parameter family on top of each KO. The WT vs WT_APP baseline comparison is already established in Phase 1. This phase focuses on the 8-condition interaction matrix.

### Task 3.1 — Full 8-condition ring study

**What**: Run ring-study for all 8 conditions (WT, WT_APP, 3×KO, 3×KO_APP), with WT and WT_APP each using their own fitted parameter family. KO and KO_APP are obtained only by receptor-activation zeroing within each family.

**Command**:
```bash
python -m circuit_model ring-study \
  --conditions WT WT_APP a7_KO a7_KO_APP b2_KO b2_KO_APP a5_KO a5_KO_APP \
  --amplitudes AMP_WT \
  --n_trials 1000 \
  --w_pyr_pyr_inter W_INTER_WT --w_pv_global W_PV_WT --sigma_pyr_deg 15 \
  --n_workers 8 --error_band sem
```

**Key analysis**: For each KO×APP pair, compute the additional effect of APP beyond the KO:

| Pair | APP additional effect | Expected |
|---|---|---|
| a7_KO → a7_KO_APP | Add WT_APP family shift under fixed α7 KO | Data-driven (from fitted families) |
| b2_KO → b2_KO_APP | Add WT_APP family shift under fixed β2 KO | Data-driven (from fitted families) |
| a5_KO → a5_KO_APP | Add WT_APP family shift under fixed α5 KO | Data-driven (from fitted families) |

**Key question**: Which KO background amplifies or attenuates the WT→WT_APP shift in bump quality metrics?

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

## PHASE 4 — Receptor-Sensitivity Sweep Study (Article Section 5.3, optional)

> **Goal**: Map how continuous receptor-activation changes affect WM metrics as a mechanistic stress test.

### Task 4.1 — α7 activation sweep

**What**: Vary act_alpha7 from 0 to 1 in 10 steps, keeping β2 and α5 at WT levels. For each, run ring-study and extract bump metrics.

**Approach**: This requires running simulations with custom activation levels. The `--condition` flag uses preset values, so we need a different approach:

**Command** (new script — see Code 5.2):
```bash
python scripts/receptor_sensitivity_sweep.py \
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

**Output**: Line plots of each metric vs act_alpha7.

**Key question**: Is there an α7 activation threshold below which oscillatory stability collapses?

**Figure destination**: Fig 13 (x-axis)

---

### Task 4.2 — α5 activation sweep

**What**: Same as Task 4.1 but for act_alpha5.

**Expected findings**: α5 primarily affects amplitude → smooth, monotonic decrease in bump quality without a sharp threshold.

**Figure destination**: Fig 13 (y-axis)

---

### Task 4.3 — 2D activation heatmap

**What**: Sweep both act_alpha7 and act_alpha5 simultaneously (21×21 grid = 441 combinations). For each, extract a single WM quality metric.

**Metrics for heatmap**:
- Primary: corrected asymmetry mean|A(t)| (lower = better)
- Secondary: oscillation frequency stability (variance of instantaneous frequency)

**Output**: 2D heatmap with act_alpha7 on x-axis, act_alpha5 on y-axis, color = WM quality.

**Why**: This characterizes model sensitivity to receptor activation strength and can reveal nonlinear operating boundaries.

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

### Code 5.2 — Receptor-activation sweep (if --set not in ring-study)

**Purpose**: For Tasks 4.1–4.3

**What to implement**:
- Script `scripts/receptor_sensitivity_sweep.py` that:
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
| **Fig 2** | Fit validation: simulated vs experimental rates, WT and WT_APP | Tasks 0.2–0.3 | To do |
| **Fig 3** | WT vs WT_APP bump baseline: dashboard + oscillation + asymmetry slope + MSD | Tasks 1.1–1.5 | To do |
| **Fig 4** | Distractor: merge/alternate map + PLV curves + phase gating, WT vs WT_APP | Tasks 1.7–1.8 | To do |
| **Fig 5** | Delay-period firing rates per population at bump/background, WT vs WT_APP | Task 2.0 | To do (needs Code 5.1) |
| **Fig 6** | KO bump metrics: amplitude, asymmetry, MSD, width per KO | Tasks 2.1, 2.3, 2.4 | To do |
| **Fig 7** | KO oscillation: spectrogram per KO + frequency/power summary | Task 2.2 | To do |
| **Fig 8** | KO distractor: PLV curves per KO + merge threshold comparison | Task 2.5 | To do |
| **Fig 9** | KO comparison summary: directional matrix + bar plots | Task 2.6 | To do (needs Code 5.3) |
| **Fig 10** | 8-condition ring study: KO×APP interaction matrix | Task 3.1 | To do |
| **Fig 11** | Receptor-activation sweep: 2D heatmap (α7 × α5 → WM quality) | Tasks 4.1–4.3 | To do (needs Code 5.2) |
| **Suppl. S1** | Calibration heatmaps: WT and WT_APP parameter spaces | Task 0.6 (= Stage B of Task 0.2 — see Stage B commands in Task 0.2) | To do |
| **Suppl. S2** | Rate-matched comparison | Task 3.2 | To do |
| **Suppl. S3** | Bump decay study: WT vs WT_APP | Task 1.6 | To do |
| **Suppl. S4** | WT vs WT_APP parameter comparison (post-fit) | Task 0.3 | To do |
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

### Tier 0 — CURRENT BLOCKER (updated 2026-03-31)

1. **Stage A** → Single-node optimization for WT and WT_APP (base + KO fits)
   - Output: `params/new/single_node/WT_1mo_article*.json`
   - Acceptance: rates within 10%, w_pe >= 0.05, w_se >= 0.003

2. **Stage B** → Ring calibration sweeps for WT and WT_APP
   - Output: w_inter and w_pv_global values for bump formation
   - Acceptance: self-sustained bump >= 2000 ms, no spontaneous activity

3. **Task 0.3** → Validate both fits on all conditions (ring-study) — after Stage B

Note: ring-optimize with Turing penalty is no longer the primary workflow.
It may still be used for exploratory runs but Stage A + Stage B is the
canonical path to production parameter sets.

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
