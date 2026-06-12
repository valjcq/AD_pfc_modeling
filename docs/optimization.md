# Parameter Optimization

How the 5-population NDNF circuit model is fit to data: the loss that is
minimised, how each candidate is scored, the search algorithm, and the
two-stage flow. Population order everywhere is `[PYR, SOM, PV, VIP, NDNF]`
(indices 0–4).

Source files:
[`circuit_model/loss.py`](../circuit_model/loss.py),
[`circuit_model/optimization.py`](../circuit_model/optimization.py),
[`circuit_model/params.py`](../circuit_model/params.py),
[`circuit_model/cli.py`](../circuit_model/cli.py).

---

## 1. What is being optimised

We have a rate model whose free parameters are synaptic weights, external
tonic drives, and GABA scaling. We want a single parameter set
whose **mean firing rates** — at baseline *and* under a panel of receptor
knockouts — match values measured experimentally.

The optimiser proposes a `CircuitParams` set, the model is simulated under
every condition, the resulting mean rates are compared to the targets through
the loss below, and a derivative-free search drives that loss down.

This is a black-box fit: the simulator is treated as an opaque
`params → mean rates` map. No gradients are computed; the loss landscape is
stochastic (finite trials, random initial conditions).

---

## 2. The targets

Targets are mean firing rates in Hz. They are produced from calcium-imaging
fluorescence by [`scripts/compute_target_rates.py`](../scripts/compute_target_rates.py)
using `rate = mean(F) / t_recording` per recording, then averaged per genotype.

A `TargetRates` ([`loss.py`](../circuit_model/loss.py)) carries up to 11
scalar targets across three families:

| Family | Targets | Measured on |
|--------|---------|-------------|
| **Baseline** | `mean_r_pyr/som/pv/vip/ndnf` | each of the 5 populations |
| **Global KO** | `alpha7_ko_pyr`, `alpha5_ko_pyr`, `beta2_ko_pyr`, `alpha7_beta2_ko_pyr` | PYR (idx 0) |
| **Selective α7 KO** | `alpha7_ndnf_ko_ndnf`, `alpha7_pv_ko_pv` | the deleted cell type itself (NDNF idx 4, PV idx 2) |

A fourth family, **drug** targets (`DrugTarget`), is used only in Stage 2
(§7). Any target left `None` is simply skipped — its condition is neither
simulated nor scored.

The knockouts are realised by zeroing the relevant receptor-activation
multiplier(s) on a copy of the parameters (`dataclasses.replace`,
[`_build_conditions`](../circuit_model/optimization.py)):

| Condition | Parameter override |
|-----------|--------------------|
| `alpha7_ko` (global) | `act_alpha7_pv = act_alpha7_som = act_alpha7_ndnf = 0` |
| `alpha5_ko` | `act_alpha5 = 0` |
| `beta2_ko` | `act_beta2 = 0` |
| `alpha7_beta2_ko` | all α7 off **and** `act_beta2 = 0` |
| `alpha7_ndnf_ko` | `act_alpha7_ndnf = 0` |
| `alpha7_pv_ko` | `act_alpha7_pv = 0` |

---

## 3. The loss function

### Per-measurement term

Every individual measurement contributes a **squared log fold-change**
([`_log_sq`](../circuit_model/loss.py)):

```
L_term = ( log( max(actual, ε) / max(target, ε) ) )²        ε = 0.01 Hz
```

Properties, and why this form was chosen:

- **Target-normalised.** It is a fold-change, so a target of 1 Hz and a target
  of 50 Hz are weighted comparably — large-rate populations do not dominate.
- **Symmetric.** A 2× overshoot and a 2× undershoot give the same loss
  (`log 2`)² — unlike a raw relative error `(actual−target)/target`, which is
  bounded below by −1 for undershoot but unbounded above.
- **Diverges at zero.** As `actual → 0` the loss → ∞ (no saturation), so the
  optimiser cannot "park" a population at silence to cheaply satisfy other
  terms. `ε = 0.01 Hz` is only a numerical floor keeping the log finite.
- **Zero at a perfect match** (`actual == target`).

There are deliberately **no** ad-hoc penalties (near-zero bonus, wrong-direction
penalty, minimum-effect terms) — the log form already supplies the desired
asymmetry-free, scale-free shape.

### Buckets and weights

Terms are grouped into four buckets, each scaled by a CLI weight
([`_loss_from_results`](../circuit_model/optimization.py)):

| Bucket | Content | Weight flag |
|--------|---------|-------------|
| `base` | Σ over the 5 baseline rates | `--weight_base` |
| `global_ko` | Σ over the 4 global-KO PYR targets | `--weight_global_ko` |
| `selective_ko` | NDNF + PV selective-α7-KO targets | `--weight_selective_ko` |
| `drug` | Σ over drug measurements (Stage 2) | `--weight_drug` |

The total minimised is simply the weighted sum:

```
L_total = w_base·base + w_gko·global_ko + w_sko·selective_ko + w_drug·drug
```

All weights default to **1.0**, i.e. every measurement enters with equal
footing. A `LossBreakdown` carrying the four bucket values plus the total is
returned alongside the rates and logged.

### Invalid candidates

If any condition's simulation produces a non-finite rate or a rate exceeding
`max_rate` (default **200 Hz**, a runaway/instability guard), the whole
candidate is assigned `L = 1e9` and discarded
([`run_trials`](../circuit_model/optimization.py) → `ok=False`).

---

## 4. Scoring one candidate (simulation → loss)

`evaluate_params` / the inner `_evaluate` closure does, for one parameter set:

1. **Build conditions** — `base` plus every KO whose target is set (§2), each
   as a `(name, params, cfg, seed)` tuple.
2. **Simulate each condition** with `run_trials`:
   - Run `n_trials` (default **8**) independent simulations.
   - Each trial starts from a *random* initial rate
     `r0 = init_rate_scale · lognormal(0, 0.6)` over 5 pops
     (`init_rate_scale = 0.2`), with its own seed.
   - Integrate the 5-pop ODE (Euler, `dt = 0.1 ms`, `T = 2500 ms`;
     [`simulation.py`](../circuit_model/simulation.py)).
   - Take the mean rate over the last `window_ms` (default **500 ms**) after a
     `burn_in_ms` transient (default **1800 ms** for `optimize`).
   - **Average the per-trial means** → one `(5,)` mean-rate vector per condition.
3. **Compute the loss** from those mean vectors via §3.

The multiple trials average over the random initial condition, so the score
reflects the rate the circuit *settles to* rather than one lucky basin. Note
that `optimize` defaults to `--noise_type none`: trials then differ only by
their starting point, which probes multistability/basin sensitivity rather
than dynamic noise. (Pass `--noise_type white|ou` to fit under noise.)

> The model itself (transfer functions, NMDA gating, divisive PV inhibition,
> subtractive SOM/NDNF, soft ceilings `R_MAX_*`) is documented in
> [`README.md`](../README.md) and the docstrings of `simulation.py` /
> `transfer.py`. For the fit, it is just the function that turns parameters
> into the mean rates the loss consumes.

---

## 5. The search algorithm

Optimisation is derivative-free via **Nevergrad**
([`nevergrad_optimize`](../circuit_model/optimization.py)). The search space is
built by `build_nevergrad_parametrization`: each free parameter becomes a
bounded `ng.p.Log` (log-mode bounds) or `ng.p.Scalar` (linear), initialised at
the (clamped) base value so `--resume` warm-starts from the loaded best.

### `--optimizer` choices

| Value | Algorithm | Use |
|-------|-----------|-----|
| `de` *(default)* | `TwoPointsDE` | robust global differential evolution; good on discontinuous landscapes |
| `cma` | CMA-ES | fast local refinement; learns parameter correlations |
| `chaining` | `TwoPointsDE → Nelder-Mead` | global then local; matches the reference-paper pipeline |
| `auto` | `NGOpt` | Nevergrad picks the algorithm from dimension/budget |

### Loop and phases

The ask/tell loop ([`_run_phase`](../circuit_model/optimization.py)) repeats
for `--n_samples` steps:

1. `ask()` a candidate → convert to `CircuitParams`.
2. `_evaluate` it (§4) → loss `L`.
3. `tell(x, L)`.
4. Maintain the top-`k` (`--top_k`, default 10) by loss; checkpoint
   `best_params.json` whenever the best improves; log + redraw loss-evolution
   plots every `--log_interval` steps.

Two optional stages reduce the variance of the *reported* best (the loss is a
noisy, finite-trial estimate):

- **Polish** (`--polish_samples N` > 0): after the global search, run CMA-ES for
  `N` more steps **warm-started from the current best** — fast convergence
  once inside a good basin.
- **Final re-evaluation** (`--final_eval_trials M` > `n_trials`): re-score the
  top-`k` candidates with `M` trials (a lower-variance estimate) and re-rank,
  so the winner is not just a lucky low-noise draw.

`Ctrl-C` is caught: the best-so-far is finalised and saved.

---

## 6. Free vs frozen parameters and bounds

Bounds come from [`default_bounds`](../circuit_model/params.py). Highlights:

| Group | Bound | Mode |
|-------|-------|------|
| Synaptic weights `w_*` | `[0.001, 0.01]` nA/Hz (`--w_hi` raises the cap) | log |
| `w_sn` (SOM→NDNF) | `[0.001, 0.05]` — only brake on NDNF, extra headroom | log |
| `J_NMDA` | `[0.05, 2.0]` | log |
| `I0_pyr` | `[0.01, 1.5]` nA | lin |
| `I0_pv/som/vip` | `[0.01, 0.6]` nA | lin |
| `I0_ndnf` | `[0.01, 0.25]` — no PYR drive, kept low to reach ~2.5 Hz | lin |
| `I_alpha7/beta2/alpha5_*` | `[0.01, 0.5]` nA | lin |
| `g_gaba_base`, `g_alpha7` | `[0.1, 5.0]` | lin |

The `0.001` weight floor and the tightened `I0_pv`/`I0_ndnf` caps came out of a
search-space viability study (only ~4% of the naive box was dynamically viable
before tightening; ~47% after).

A parameter is **free** if it is in `bounds` and not frozen. Frozen set =
`--freeze` list ∪ (params with no bound) ∪
**all receptor-activation fields** (`act_*` are always frozen in Stage 1; they
are Stage 2's job). `--show_params` prints the free/frozen breakdown by group.

---

## 7. Two-stage flow

**Stage 1 — `--stage weights` (default).** Fit synaptic weights, external
currents, and GABA scaling to the baseline + KO targets (§2–§5).
Receptor activations are held at `1.0`. Output: a `best_params.json`.

**Stage 2 — `--stage receptors`.** Load a Stage-1 fit (`--params_json`) and,
**independently per drug** (`--drugs MLA,PNU,nicotine`), fit only the five
receptor-activation multipliers (`act_alpha7_pv/_som/_ndnf`, `act_beta2`,
`act_alpha5`), bounded `[0, 5]`, with everything else frozen
([`optimize_drug_activations`](../circuit_model/optimization.py)). Each drug
has an NDNF and a PV target; the per-drug loss is the same squared-log-fold
form (`drug_loss`). With 2 measurements and 5 free activations this fit is
under-constrained — the `[0, 5]` bounds keep solutions physiological. Results
are written to `stage2_results.json`.

---

## 8. Outputs

For `optimize`, everything lands in `--output_dir` (e.g.
[`fits/WT_NDNF_5pop/`](../fits/WT_NDNF_5pop)):

| File | Content |
|------|---------|
| `best_params.json` | best `CircuitParams` (+ fit metadata) |
| `best_params.txt` | human-readable fit summary: actual-vs-target table, transfer-function params, Jacobian, connection gains |
| `log.jsonl` | per-`log_interval` records (step, loss, breakdown, means, KO means, params, targets) |
| `commands.log` | exact CLI invocation, for reproducibility |
| `loss_evolution*.png` | loss-vs-step curves (total and per-bucket ratios) |

At the end, `cmd_optimize` also prints the top-k, runs a Jacobian sanity check
at the fitted steady state, and prints the actual-vs-target comparison table.

---

## 9. Worked example — `WT_NDNF_5pop`

The committed fit was produced by ([`commands.log`](../fits/WT_NDNF_5pop/commands.log)):

```bash
python -m circuit_model optimize \
  --target_pyr 1.7328 --target_som 1.3564 --target_pv 1.5281 \
  --target_vip 2.9791 --target_ndnf 2.5309 \
  --target_alpha7_ko_pyr 2.1928 --target_beta2_ko_pyr 1.0825 \
  --target_alpha5_ko_pyr 0.4762 --target_alpha7_beta2_ko_pyr 1.3465 \
  --target_alpha7_ndnf_ko_ndnf 3.0767 --target_alpha7_pv_ko_pv 1.3966 \
  --optimizer twopointde --n_samples 50000 \
  --polish_samples 10000 --final_eval_trials 32 \
  --output_dir fits/WT_NDNF_5pop
```

50 000 DE steps, a 10 000-step CMA polish, and a 32-trial final re-evaluation,
all weights at their default 1.0. **Final loss ≈ 0.715.** Baseline rates and
the selective α7 KOs land within ~10%; the global α5/β2/α7β2 KOs are the
poorly-fit terms (β2-KO PYR +72%), the known residuals of this fit.
See [`best_params.txt`](../fits/WT_NDNF_5pop/best_params.txt) for the full
comparison table.

---

## 10. Quick reference

```bash
# Stage 1: fit weights + currents to baseline + KO targets
python -m circuit_model optimize \
  --target_pyr 1.73 --target_som 1.36 --target_pv 1.53 \
  --target_vip 2.98 --target_ndnf 2.53 \
  --target_alpha7_ko_pyr 2.19 --target_alpha5_ko_pyr 0.48 \
  --target_beta2_ko_pyr 1.08 --target_alpha7_beta2_ko_pyr 1.35 \
  --target_alpha7_ndnf_ko_ndnf 3.08 --target_alpha7_pv_ko_pv 1.40 \
  --optimizer de --n_samples 50000 \
  --polish_samples 10000 --final_eval_trials 32 \
  --output_dir fits/my_run --show_params

# Resume a run (reuses targets + best params from the log)
python -m circuit_model optimize --resume \
  --save_best_json fits/my_run/best_params.json \
  --log_file fits/my_run/log.jsonl --n_samples 20000

# Stage 2: per-drug receptor-activation fit on top of a Stage-1 result
python -m circuit_model optimize --stage receptors \
  --params_json fits/my_run/best_params.json \
  --drugs MLA,PNU \
  --target_mla_ndnf 1.2 --target_mla_pv 2.0 \
  --target_pnu_ndnf 4.0 --target_pnu_pv 3.5 \
  --output_dir fits/my_run/stage2
```

Key knobs: `--n_samples` (budget), `--optimizer`, `--n_trials` /
`--final_eval_trials` (loss-estimate variance), `--weight_*` (bucket weights),
`--freeze` / `--set` (parameter control), `--w_hi` (weight cap).
Full flag list: [`docs/CLI.md`](CLI.md).
