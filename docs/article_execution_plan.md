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

The optimization workflow follows a **ring-first approach** with staged constraints:

- **Stage A1** — Joint ring optimization (`circuit_model ring-optimize`) with **rates + KO only** (`--turing_weight 0.0`). Optimizes both circuit and ring parameters.
- **Stage A2** — Restart from A1 with **trace-based Turing loss** enabled. Free all circuit parameters + ring parameters for bistability optimization.
- **Stage B** — Ring calibration sweep (`circuit_model ring-calibrate`): optional post-fit refinement of bump formation/sustainment thresholds.

**Rationale**:
- A1 prevents early Turing domination, establishes good firing-rate/KO basin first.
- A2 adds bistability constraint, allowing circuit parameters to co-evolve with ring connectivity to achieve self-sustained bumps.
- Keeps all fitting constraints in the same ring simulation context used downstream.

---

### Stage A — Joint Ring Optimization (`ring-optimize`, two-pass)

Note: `--params_json` expects a **CircuitParams** JSON. `--ring_params_json` expects a **RingParams** JSON
(same format as `--save_best_ring_json` output).

| Condition | `--params_json` | `--ring_params_json` |
|-----------|----------------|----------------------|
| Base rates (WT / WT_APP) | `params/init/single_ring_init.json` | `params/init/ring_init.json` |
| KO (WT background) | `params/init/single_ring_init_WT.json` | `params/init/ring_init_WT.json` |
| KO (WT_APP background) | `params/init/single_ring_init_APP.json` | `params/init/ring_init_APP.json` |

`*_WT` and `*_APP` init files are extracted from the corresponding A1 base-rates result; they seed
the KO optimizations from a known-good basin for that genotype.

#### WT — base rates

Pass A1 (rates only, no Turing, no adaptation):

```bash
python -m circuit_model ring-optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --params_json params/init/single_ring_init.json \
  --ring_params_json params/init/ring_init.json \
  --optimizer chaining --n_samples 50000 \
  --noise_type none --n_trials_ring 1 \
  --turing_weight 0.0 \
  --spatial_uniformity_weight 1.0 \
  --ach_ratio_weight 1.0 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600,tau_adapt_som=200,sigma_noise=0.1" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr,tau_adapt_som,sigma_noise" \
  --no_adapt \
  --save_best_circuit_json params/new/ring_optimize/A1/WT_1mo_article_A1_no_adapt_circuit.json \
  --save_best_ring_json params/new/ring_optimize/A1/WT_1mo_article_A1_no_adapt_ring.json \
  --log_file figs/optim/1mo/ring_optimize_A1/log.jsonl
```

Pass A2 (restart from A1 and add Turing, free all circuit params, no adaptation):

```bash
python -m circuit_model ring-optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --params_json params/new/ring_optimize/A1/WT_1mo_article_A1_no_adapt_circuit.json \
  --ring_params_json params/new/ring_optimize/A1/WT_1mo_article_A1_no_adapt_ring.json \
  --optimizer cma --n_samples 50000 \
  --turing_weight 2.0 --turing_margin 0.05 \
  --turing_cue_amplitude 0.4 --turing_cue_duration_ms 250 --turing_cue_sigma_deg 20 \
  --turing_late_delay_ms 500 --turing_bump_min_hz 35 --turing_bump_max_hz 45 --turing_topk_nodes 5 \
  --turing_activate_below_ring_rate_loss 1.0 \
  --spatial_uniformity_weight 0.0 \
  --ach_ratio_weight 0.0 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600,sigma_noise=0.1" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr,sigma_noise" \
  --no_adapt \
  --save_best_circuit_json params/new/ring_optimize/A2/WT_1mo_article_A2_no_adapt_circuit.json \
  --save_best_ring_json params/new/ring_optimize/A2/WT_1mo_article_A2_no_adapt_ring.json \
  --log_file figs/optim/1mo/ring_optimize_A2/log.jsonl
```

#### WT_APP — base rates

Pass A1 (rates only, no Turing, no adaptation):

```bash
python -m circuit_model ring-optimize \
  --target_pyr 12.466 --target_som 4.814 --target_pv 4.241 --target_vip 5.551 \
  --params_json params/init/single_ring_init.json \
  --ring_params_json params/init/ring_init.json \
  --optimizer chaining --n_samples 50000 \
  --noise_type none --n_trials_ring 1 \
  --turing_weight 0.0 \
  --spatial_uniformity_weight 1.0 \
  --ach_ratio_weight 1.0 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600,sigma_noise=0.1" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr,sigma_noise" \
  --no_adapt \
  --save_best_circuit_json params/new/ring_optimize/A1/WT_APP_1mo_article_A1_no_adapt_circuit.json \
  --save_best_ring_json params/new/ring_optimize/A1/WT_APP_1mo_article_A1_no_adapt_ring.json \
  --log_file figs/optim/1mo/ring_optimize_A1/log_app.jsonl
```

Pass A2 (restart from A1 and add Turing, free all circuit params, no adaptation):

```bash
python -m circuit_model ring-optimize \
  --target_pyr 12.466 --target_som 4.814 --target_pv 4.241 --target_vip 5.551 \
  --params_json params/new/ring_optimize/A1/WT_APP_1mo_article_A1_no_adapt_circuit.json \
  --ring_params_json params/new/ring_optimize/A1/WT_APP_1mo_article_A1_no_adapt_ring.json \
  --optimizer cma --n_samples 50000 \
  --turing_weight 2.0 --turing_margin 0.05 \
  --turing_cue_amplitude 0.4 --turing_cue_duration_ms 250 --turing_cue_sigma_deg 20 \
  --turing_late_delay_ms 500 --turing_bump_min_hz 35 --turing_bump_max_hz 45 --turing_topk_nodes 5 \
  --turing_activate_below_ring_rate_loss 1.0 \
  --spatial_uniformity_weight 0.0 \
  --ach_ratio_weight 0.0 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600,sigma_noise=0.1" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr,sigma_noise" \
  --no_adapt \
  --save_best_circuit_json params/new/ring_optimize/A2/WT_APP_1mo_article_A2_no_adapt_circuit.json \
  --save_best_ring_json params/new/ring_optimize/A2/WT_APP_1mo_article_A2_no_adapt_ring.json \
  --log_file figs/optim/1mo/ring_optimize_A2/log_app.jsonl
```

---

### Task 0.3 — Compare A1 vs A2 fits via ring visualization (WT and WT_APP)

**What**: After A1 and A2 are complete for both WT and WT_APP, run both sets of parameters through ring-run to compare A1 (rates-only) vs A2 (Turing-constrained) solutions visually. All runs use `--no_adapt` to match the optimization context. An additional `with-adapt` variant is included for each to assess the effect of re-enabling adaptation post-hoc.

**Commands** (run at calibrated ring parameters from each stage):

For WT — A1 (no-adapt, matches optimization):
```bash
python -m circuit_model ring-run \
  --condition WT \
  --params_json params/new/ring_optimize/A1/WT_1mo_article_A1_no_adapt_circuit.json \
  --ring_params_json params/new/ring_optimize/A1/WT_1mo_article_A1_no_adapt_ring.json \
  --amplitude 0.4 --delay_ms 5000 \
  --n_nodes 128 --no_adapt \
  --output_dir figs/ring/run/A1_vs_A2/WT_A1
```

For WT — A2 (no-adapt, matches optimization):
```bash
python -m circuit_model ring-run \
  --condition WT \
  --params_json params/new/ring_optimize/A2/WT_1mo_article_A2_no_adapt_circuit.json \
  --ring_params_json params/new/ring_optimize/A2/WT_1mo_article_A2_no_adapt_ring.json \
  --amplitude 0.4 --delay_ms 5000 \
  --n_nodes 128 --no_adapt \
  --output_dir figs/ring/run/A1_vs_A2/WT_A2
```

For WT_APP — A1 (no-adapt, matches optimization):
```bash
python -m circuit_model ring-run \
  --condition WT_APP \
  --params_json params/new/ring_optimize/A1/WT_APP_1mo_article_A1_no_adapt_circuit.json \
  --ring_params_json params/new/ring_optimize/A1/WT_APP_1mo_article_A1_no_adapt_ring.json \
  --amplitude 0.4 --delay_ms 5000 \
  --n_nodes 128 --no_adapt \
  --output_dir figs/ring/run/A1_vs_A2/WT_APP_A1
```

For WT_APP — A2 (no-adapt, matches optimization):
```bash
python -m circuit_model ring-run \
  --condition WT_APP \
  --params_json params/new/ring_optimize/A2/WT_APP_1mo_article_A2_no_adapt_circuit.json \
  --ring_params_json params/new/ring_optimize/A2/WT_APP_1mo_article_A2_no_adapt_ring.json \
  --amplitude 0.4 --delay_ms 5000 \
  --n_nodes 128 --no_adapt \
  --output_dir figs/ring/run/A1_vs_A2/WT_APP_A2
```

**With-adaptation variants** (re-enable adaptation on the no-adapt optimized params):

For WT — A1 with-adapt:
```bash
python -m circuit_model ring-run \
  --condition WT \
  --params_json params/new/ring_optimize/A1/WT_1mo_article_A1_circuit.json \
  --ring_params_json params/new/ring_optimize/A1/WT_1mo_article_A1_ring.json \
  --amplitude 0.4 --delay_ms 5000 \
  --n_nodes 128 \
  --output_dir figs/ring/run/A1_vs_A2/with_adapt/WT_A1
```

For WT — A2 with-adapt:
```bash
python -m circuit_model ring-run \
  --condition WT \
  --params_json params/new/ring_optimize/A2/WT_1mo_article_ring_opt_circuit.json \
  --ring_params_json params/new/ring_optimize/A2/WT_1mo_article_ring_opt_ring.json \
  --amplitude 0.4 --delay_ms 5000 \
  --n_nodes 128 \
  --output_dir figs/ring/run/A1_vs_A2/with_adapt/WT_A2
```

For WT_APP — A1 with-adapt:
```bash
python -m circuit_model ring-run \
  --condition WT_APP \
  --params_json params/new/ring_optimize/A1/WT_APP_1mo_article_A1_circuit.json \
  --ring_params_json params/new/ring_optimize/A1/WT_APP_1mo_article_A1_ring.json \
  --amplitude 0.4 --delay_ms 5000 \
  --n_nodes 128 \
  --output_dir figs/ring/run/A1_vs_A2/with_adapt/WT_APP_A1
```

For WT_APP — A2 with-adapt:
```bash
python -m circuit_model ring-run \
  --condition WT_APP \
  --params_json params/new/ring_optimize/A2/WT_APP_1mo_article_A2_circuit.json \
  --ring_params_json params/new/ring_optimize/A2/WT_APP_1mo_article_A2_ring.json \
  --amplitude 0.4 --delay_ms 5000 \
  --n_nodes 128 \
  --output_dir figs/ring/run/A1_vs_A2/with_adapt/WT_APP_A2
```

**Expected outputs** (per condition, per stage): `dashboard.png`, `connectivity_matrices.png`, `population_activity.png`

**Acceptance criteria**: 
- A1: All 4 populations fire at target rates (±10%), but bump may be unstable or decay
- A2: Firing rates maintained (±10%), bump is visibly more stable during delay period
- Turing loss should improve without degrading rate matching

**Visual comparison**: Side-by-side dashboards of A1 vs A2 for WT and WT_APP show whether Turing constraints successfully improve bistability while maintaining firing rates.

---

## PHASE 0 — Step 2: KO Extensions (WT and WT_APP with KO constraints)

> **Goal**: Starting from the Step 1 no-adapt fits, add KO constraints (α7, β2, α5) to pin knockout PYR rates simultaneously. Run A1 and A2 passes for both WT and WT_APP backgrounds.

---

### Task 0.4 — KO-constrained ring optimization

#### WT — with KO constraints

Pass A1 (rates + KO only, no Turing):

```bash
python -m circuit_model ring-optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --target_alpha7_ko_pyr 17.539 --target_beta2_ko_pyr 17.965 --target_alpha5_ko_pyr 9.285 \
  --params_json params/new/ring_optimize/A1/WT_1mo_article_A1_no_adapt_circuit.json \
  --ring_params_json params/new/ring_optimize/A1/WT_1mo_article_A1_no_adapt_ring.json \
  --optimizer chaining --n_samples 50000 \
  --noise_type none --n_trials_ring 1 \
  --turing_weight 0.0 \
  --spatial_uniformity_weight 1.0 \
  --ach_ratio_weight 1.0 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600,tau_adapt_som=200,sigma_noise=0.1" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr,tau_adapt_som,sigma_noise" \
  --no_adapt \
  --save_best_circuit_json params/new/ring_optimize/A1/WT_1mo_article_ko_A1_no_adapt_circuit.json \
  --save_best_ring_json params/new/ring_optimize/A1/WT_1mo_article_ko_A1_no_adapt_ring.json \
  --log_file figs/optim/1mo_ko/ring_optimize_A1/log.jsonl
```

Pass A2 (restart from A1 and add Turing, free all circuit params):

```bash
python -m circuit_model ring-optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --target_alpha7_ko_pyr 17.539 --target_beta2_ko_pyr 17.965 --target_alpha5_ko_pyr 9.285 \
  --params_json params/new/ring_optimize/A1/WT_1mo_article_ko_A1_no_adapt_circuit.json \
  --ring_params_json params/new/ring_optimize/A1/WT_1mo_article_ko_A1_no_adapt_ring.json \
  --optimizer cma --n_samples 50000 \
  --turing_weight 2.0 --turing_margin 0.05 \
  --turing_cue_amplitude 0.4 --turing_cue_duration_ms 250 --turing_cue_sigma_deg 20 \
  --turing_late_delay_ms 500 --turing_bump_min_hz 35 --turing_bump_max_hz 45 --turing_topk_nodes 5 \
  --turing_activate_below_ring_rate_loss 1.0 \
  --spatial_uniformity_weight 0.0 \
  --ach_ratio_weight 0.0 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600,tau_adapt_som=200,sigma_noise=0.1" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr,tau_adapt_som,sigma_noise" \
  --no_adapt \
  --save_best_circuit_json params/new/ring_optimize/A2/WT_1mo_article_ko_A2_no_adapt_circuit.json \
  --save_best_ring_json params/new/ring_optimize/A2/WT_1mo_article_ko_A2_no_adapt_ring.json \
  --log_file figs/optim/1mo_ko/ring_optimize_A2/log.jsonl
```

#### WT_APP — with KO constraints

Pass A1 (rates + KO only, no Turing):

```bash
python -m circuit_model ring-optimize \
  --target_pyr 12.466 --target_som 4.814 --target_pv 4.241 --target_vip 5.551 \
  --target_alpha7_ko_pyr 13.599 --target_beta2_ko_pyr 19.109 --target_alpha5_ko_pyr 3.113 \
  --params_json params/new/ring_optimize/A1/WT_APP_1mo_article_A1_no_adapt_circuit.json \
  --ring_params_json params/new/ring_optimize/A1/WT_APP_1mo_article_A1_no_adapt_ring.json \
  --optimizer chaining --n_samples 50000 \
  --noise_type none --n_trials_ring 1 \
  --turing_weight 0.0 \
  --spatial_uniformity_weight 1.0 \
  --ach_ratio_weight 1.0 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600,tau_adapt_som=200,sigma_noise=0.1" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr,tau_adapt_som,sigma_noise" \
  --no_adapt \
  --save_best_circuit_json params/new/ring_optimize/A1/WT_APP_1mo_article_ko_A1_no_adapt_circuit.json \
  --save_best_ring_json params/new/ring_optimize/A1/WT_APP_1mo_article_ko_A1_no_adapt_ring.json \
  --log_file figs/optim/1mo_ko/ring_optimize_A1/log_app.jsonl
```

Pass A2 (restart from A1 and add Turing, free all circuit params):

```bash
python -m circuit_model ring-optimize \
  --target_pyr 12.466 --target_som 4.814 --target_pv 4.241 --target_vip 5.551 \
  --target_alpha7_ko_pyr 13.599 --target_beta2_ko_pyr 19.109 --target_alpha5_ko_pyr 3.113 \
  --params_json params/new/ring_optimize/A1/WT_APP_1mo_article_ko_A1_no_adapt_circuit.json \
  --ring_params_json params/new/ring_optimize/A1/WT_APP_1mo_article_ko_A1_no_adapt_ring.json \
  --optimizer cma --n_samples 50000 \
  --turing_weight 2.0 --turing_margin 0.05 \
  --turing_cue_amplitude 0.4 --turing_cue_duration_ms 250 --turing_cue_sigma_deg 20 \
  --turing_late_delay_ms 500 --turing_bump_min_hz 35 --turing_bump_max_hz 45 --turing_topk_nodes 5 \
  --turing_activate_below_ring_rate_loss 1.0 \
  --spatial_uniformity_weight 0.0 \
  --ach_ratio_weight 0.0 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600,tau_adapt_som=200,sigma_noise=0.1" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr,tau_adapt_som,sigma_noise" \
  --no_adapt \
  --save_best_circuit_json params/new/ring_optimize/A2/WT_APP_1mo_article_ko_A2_no_adapt_circuit.json \
  --save_best_ring_json params/new/ring_optimize/A2/WT_APP_1mo_article_ko_A2_no_adapt_ring.json \
  --log_file figs/optim/1mo_ko/ring_optimize_A2/log_app.jsonl
```

---

## PHASE 5 — WT_APP Fitting Diagnostics: Silent Synapses & Constraint Analysis

> **Goal**: Diagnose why WT_APP fitting with free parameters yields poor firing-rate matches. Test hypothesis that some synaptic connections become silent in APP condition and are artificially prevented by jacobian loss.

### Background

Full free-parameter optimization on WT_APP shows large firing-rate discrepancies (especially PYR), even without KO constraints. Initial hypothesis: in APP condition, certain connections may become functionally silent (very low effective coupling), but the jacobian regularization prevents the optimizer from exploring these regimes because it penalizes low/zero derivatives.

Proposed diagnostic is to incrementally relax constraints and compare achieved firing rates:

---

### Task 5.1 — Baseline: WT_APP fit with full jacobian loss (reference)

**What**: Re-run WT_APP optimization with all parameters free, jacobian loss enabled (current approach).

**Command**:
```bash
python -m circuit_model ring-optimize \
  --target_pyr 12.466 --target_som 4.814 --target_pv 4.241 --target_vip 5.551 \
  --params_json params/init/single_ring_init.json \
  --ring_params_json params/init/ring_init.json \
  --optimizer chaining --n_samples 50000 \
  --noise_type none --n_trials_ring 1 \
  --turing_weight 0.0 \
  --jacobian_weight 0.5 \
  --set "tau_s=20,sigma_pyr_deg=15,alpha_pyr=310,alpha_pv=615,alpha_som=615,alpha_vip=615,Theta_pyr=0.40323,Theta_pv=0.28780,Theta_som=0.28780,Theta_vip=0.28780,g_exc=0.16,g_inh=0.087,tau_adapt_pyr=600,tau_adapt_som=200,sigma_noise=0.1" \
  --freeze "tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr,tau_adapt_som,sigma_noise" \
  --save_best_circuit_json params/new/diagnostics/WT_APP_1mo_jacobian_baseline_circuit.json \
  --save_best_ring_json params/new/diagnostics/WT_APP_1mo_jacobian_baseline_ring.json
```

**Record**: Final loss, per-population percentage errors (PYR, SOM, PV, VIP), max Jacobian coupling.

---

### Task 5.2 — Approach 1: nAChR modulation only (no jacobian loss)

**What**: Start from WT-fitted parameters. Free **only** the nicotinic receptor activation parameters. Disable jacobian loss entirely. All synaptic weights, currents, and transfer function parameters remain frozen.

**Rationale**: Tests whether acetylcholine signaling differences alone explain WT→APP divergence without penalty for silent pathways.

**Command**:
```bash
python -m circuit_model ring-optimize \
  --target_pyr 12.466 --target_som 4.814 --target_pv 4.241 --target_vip 5.551 \
  --params_json params/new/ring_optimize/A1/WT_1mo_article_A1_no_adapt_circuit.json \
  --ring_params_json params/init/ring_init.json \
  --optimizer chaining --n_samples 30000 \
  --noise_type none --n_trials_ring 1 \
  --turing_weight 0.0 \
  --jacobian_weight 0.0 \
  --freeze "w_pyr_pyr,w_pyr_som,w_pyr_vip,w_som_pyr,w_som_som,w_som_vip,w_vip_pyr,w_vip_som,w_vip_vip,I0_pyr,I0_som,I0_pv,I0_vip,g_gaba_pv,g_gaba_som,g_gaba_vip,J_adapt_pyr,A_pyr,A_som,A_pv,A_vip,tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr,tau_adapt_som,sigma_noise" \
  --save_best_circuit_json params/new/diagnostics/WT_APP_1mo_nAChR_only_circuit.json \
  --save_best_ring_json params/new/diagnostics/WT_APP_1mo_nAChR_only_ring.json
```

**Record**: Final loss, per-population percentage errors, achieved receptor activations (compare to WT).

**Expected outcome**: If this approach achieves good fit (errors <10%), APP effects are primarily acetylcholine-mediated. If it fails, synaptic weight changes are necessary.

---

### Task 5.3 — Approach 2: Free weights + nAChR (no jacobian loss)

**What**: Start from WT-fitted parameters. Allow all synaptic weights and nAChR activations to vary freely. Keep baseline currents (I0) fixed and all transfer function parameters frozen. Disable jacobian loss.

**Rationale**: Tests whether allowing synaptic weight changes (without hard constraints) enables better fitting. More permissive than Task 5.2, but reveals which connections actually need to change. Post-hoc analysis of achieved weights compared to WT will identify which changed significantly.

**Command**:
```bash
python -m circuit_model ring-optimize \
  --target_pyr 12.466 --target_som 4.814 --target_pv 4.241 --target_vip 5.551 \
  --params_json params/new/ring_optimize/A1/WT_1mo_article_A1_no_adapt_circuit.json \
  --ring_params_json params/init/ring_init.json \
  --optimizer chaining --n_samples 50000 \
  --noise_type none --n_trials_ring 1 \
  --turing_weight 0.0 \
  --jacobian_weight 0.0 \
  --freeze "I0_pyr,I0_som,I0_pv,I0_vip,g_gaba_pv,g_gaba_som,g_gaba_vip,J_adapt_pyr,A_pyr,A_som,A_pv,A_vip,tau_s,sigma_pyr_deg,alpha_pyr,alpha_pv,alpha_som,alpha_vip,Theta_pyr,Theta_pv,Theta_som,Theta_vip,g_exc,g_inh,tau_adapt_pyr,tau_adapt_som,sigma_noise" \
  --save_best_circuit_json params/new/diagnostics/WT_APP_1mo_free_weights_circuit.json \
  --save_best_ring_json params/new/diagnostics/WT_APP_1mo_free_weights_ring.json
```

**Record**: Final loss, per-population errors, weight changes from WT (identify which connections change and by how much), receptor activations.

**Post-processing**: After optimization, manually compare achieved weights to WT values using:
```python
import json
with open('params/new/ring_optimize/A1/WT_1mo_article_A1_no_adapt_circuit.json') as f:
    wt_params = json.load(f)
with open('params/new/diagnostics/WT_APP_1mo_free_weights_circuit.json') as f:
    app_params = json.load(f)

# Compare weight changes
weights = ['w_pyr_pyr', 'w_pyr_som', 'w_pyr_vip', 'w_som_pyr', 'w_som_som', 'w_som_vip', 'w_vip_pyr', 'w_vip_som', 'w_vip_vip']
for w in weights:
    wt_val = wt_params[w]
    app_val = app_params[w]
    pct_change = 100 * (app_val - wt_val) / wt_val if wt_val != 0 else 0
    print(f"{w}: {wt_val:.6f} → {app_val:.6f} ({pct_change:+.1f}%)")
```

Identify:
- Which connections changed more than ±30% (beyond typical plasticity bounds)
- Which changed minimally but were essential for fit
- Receptor activation changes vs weight changes (correlation)

---

### Task 5.4 — Comparison & Interpretation

**Metrics to compare across 5.1, 5.2, 5.3**:

| Approach | Jacobian loss | Free parameters | Expected outcome |
|---|---|---|---|
| 5.1 (baseline) | Yes | All circuit + ring | Poor fit (reference failure mode) |
| 5.2 (nAChR only) | No | 8 receptor activations | Good fit → APP is nAChR-driven |
| 5.3 (constrained weights) | No (soft) | Synaptic weights ± 30% + nAChR | Good fit → modest plasticity sufficient |

**Interpretation logic**:
- If 5.2 fits well: APP condition is purely acetylcholine-driven; synaptic weights should not change significantly.
- If 5.2 fails but 5.3 fits well: synaptic weights must adapt in APP; quantify which connections change and by how much.
- If both 5.2 and 5.3 fail: missing mechanism (possibly NMDA time constant, adaptation properties, or other structural changes).

**Figure destination**: Supplementary table or figure (method validation, not main narrative).

---

### Implementation Notes

**Current CLI support**:
- `--freeze` is implemented and works as documented
- `--jacobian_weight` controls jacobian regularization strength (set to 0.0 to disable)
- `--save_best_circuit_json` and `--save_best_ring_json` save optimized parameters

**Features used in Tasks 5.2–5.3 that may require validation/extension**:
- Setting `--jacobian_weight 0.0` should disable jacobian loss entirely (verify it doesn't cause errors)
- Task 5.3 relies on post-hoc weight analysis rather than built-in constraints — implement a utility script to compare achieved weights to WT baseline and compute relative changes per connection
