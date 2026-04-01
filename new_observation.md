### Loss Function & Fitting Pipeline Overhaul (2026-03-23)

#### Summary of all changes made

1. **MAPE → MSPE** (`squared=True` default): base loss now squares the per-population percentage errors. Prevents the optimizer from tolerating a large miss on one population (e.g. PYR −37%) while the others are exact, which MAPE allows because it averages linearly.

2. **Jacobian upper-bound penalty** (`max_gain=5`, quadratic above): MSPE alone pushed weights to extreme values (PV→PV w≈337, J=−72). A per-entry cap on the full 4×4 Jacobian rules out biologically implausible solutions. Threshold meaning: `J[i,j]=5` means a 1 Hz increase in population j causes a 5 Hz change in population i — already very strong coupling.

3. **KO conditions always simulated** (even without KO targets): the three KO conditions (α7, α5, β2) are now run in all fits. When no KO target was specified they appear as `(info)` rows in the summary with the reference value shown for comparison but not contributing to the loss.

4. **KO loss normalisation** (`ko_loss / n_ko`): previously the total loss had 1 base term vs up to 3×4=12 KO sub-terms, making the optimizer ~12× more focused on KO conditions. Now the total KO contribution is divided by the number of active KOs so it always has equal weight to the base loss.

5. **MSPE in `loss_from_ko_pyr`**: the KO per-condition loss was still using MAPE; switched to MSPE for consistency.

6. **Plateau early stopping** (`--plateau_patience`, default 5000 steps): stops if no improvement for N steps. Counter only starts after the DE phase in chaining mode (first 10 000 steps) to avoid triggering during global exploration.

7. **On-demand summary regeneration**: existing `.json` params can be re-evaluated and `.txt` summaries regenerated without relaunching a full optimisation.

---

#### Base fitting — current results (`WT_1mo_article`, `WT_APP_1mo_article`)

These are the canonical fits used for ring model runs. KO rows shown as info only.

| Fit | Loss | PYR err | SOM err | PV err | VIP err | Max \|J\| |
|-----|------|---------|---------|--------|---------|-----------|
| `WT_1mo_article` | ~0 | -0.0% | +0.0% | +0.0% | +0.0% | 3.7 |
| `WT_APP_1mo_article` | ~0 | +0.0% | -0.0% | -0.0% | +0.0% | 4.4 |

Both fits are essentially perfect on base rates with all Jacobian gains ≤5. KO predictions (info, not fitted):

| | α7KO PYR (ref) | α5KO PYR (ref) | β2KO PYR (ref) |
|--|--|--|--|
| `WT` predicted | 13.97 (17.54) | 4.37 (9.29) | 13.15 (17.97) |
| `WT_APP` predicted | 13.43 (13.60) | 4.29 (3.11) | 12.38 (19.11) |

WT_APP α7KO is close to target; α5KO and β2KO are far — the base-only fit cannot simultaneously reproduce all KO effects without being constrained on them.

---

#### KO-constrained fitting — current results (`WT_1mo_article_ko`, `WT_APP_1mo_article_ko`)

These fits include α7/α5/β2 KO targets in the loss. **Need to be relaunched** with the rebalanced loss (KO normalisation + MSPE in KO term) — current params were optimised under the old unbalanced weighting.

| Fit | Loss (new) | PYR base err | SOM base err | PV base err | VIP base err |
|-----|------------|-------------|-------------|------------|-------------|
| `WT_ko` | 0.1215 | +45.9% | -26.8% | -23.3% | +26.6% |
| `WT_APP_ko` | 0.6692 | -43.4% | -23.4% | **-69.8%** | +4.6% |

KO predictions under these params:

| | α7KO PYR | ref | α5KO PYR | ref | β2KO PYR | ref |
|--|--|--|--|--|--|--|
| `WT_ko` | 18.41 | 17.54 | 8.11 | 9.29 | 21.59 | 17.97 |
| `WT_APP_ko` | 17.31 | 13.60 | 3.12 | 3.11 | 13.23 | 19.11 |

Base rates are poorly fitted in both cases — the fundamental tension is that KO conditions push PYR up strongly (α7KO: 8→18 Hz for WT) which requires very different network dynamics from the baseline. The rebalanced loss should improve base rate accuracy at some cost to KO precision. **Relaunch needed.**

---

### Firing-Rate Optimization Attempts: Single Node → Ring (2026-03-24)

**Step 1 (single-node fit, article transfer shape):**
- First approach was to fit only at the single-node level, using the same transfer-function shape parameters as in the article.
- In this setup, fixed vs free parameters followed `parameter_free_set.md`:
	- Fixed: transfer thresholds/gains (`Theta_*`, `alpha_*`), shared curvature `g=1`, and fixed timescales (`tau_s`, `tau_adapt_pyr`).
	- Free: output scales (`A_*`), local synaptic weights, external currents/nAChR currents, GABA modulation terms, adaptation strength, and noise amplitude.
- A Jacobian loss/regularization was added to avoid parameter regions where one link/pathway is effectively unused.
- Goal was to keep the fitted circuit biologically plausible (all key nodes/connections functionally engaged).
- Result: optimization could reproduce firing rates while keeping non-omitted effective couplings.

**Step 2 (extend fit toward network-level ring parameters):**
- Then we tried to include ring-level parameters (`w_pyr_pyr_inter`, `sigma_pyr_deg`, `w_pv_global`) in the fitting logic.
- We could not identify a parameter region where the network robustly formed a bump.
- First hypothesis was that this came from too small effective weights, so the Jacobian upper cap was removed.
- Result: no real improvement; bump-forming region was still not found.

**Step 3 (full ring-in-the-loop optimization):**
- Next, optimization was run with the full ring simulation in the objective, fitting ring activity toward targets.
- With free `sigma_pyr_deg`, fitting could still work at firing-rate level.
- But when constraining `sigma_pyr_deg` to ~15°, optimization failed to recover states reproducing target data.
- This failure appeared across conditions, not only in one genotype.
- In parallel, the network still does not reliably sustain a bump in delay.

**Current interpretation:**
- A constrained narrow PYR-PYR spread (15°) seems incompatible with obtaining both realistic firing rates and stable bump dynamics in the current parameterization.
- More generally, matching single-node rates is not sufficient to guarantee ring-level attractor stability.

**Immediate next tests:**
1. Re-fit without constraining `sigma_pyr_deg` and compare fitted solutions across conditions.
2. Re-test bump sustainability from these unconstrained fits.
3. If bump is still not sustained, investigate missing stabilizing mechanisms (priority hypothesis: absent SOM adaptation current).

---

### Ring Fitting Update: SOM Adaptation Added (2026-03-25)

**What was changed:**
- Added SOM adaptation current in the fit, with fixed `tau_adapt_som = 150 ms`.
- Removed `sigma_pyr_deg` as a free parameter during tests, then checked behavior with/without constraining it.

**Observed behavior:**
- The optimizer still does not find an interesting/stable working-memory solution.
- Time-course inspection shows a mostly silent network, with a small bump moving around even without stimulus.
- This creates an important mismatch: averaging over all nodes can still give an acceptable firing-rate loss, but the spatial variance across nodes in quiet state is too high.
- In the intended baseline regime, nodes should remain close to homogeneous activity (similar rates across the ring before cue).

**Interpretation on `sigma_pyr_deg`:**
- In these runs, `sigma_pyr_deg` does not appear to be the primary limiting factor.
- It does influence behavior when the network is already in a high-variance regime, but that regime is itself non-physiological for the intended baseline.
- Practical decision: keep `sigma_pyr_deg` fixed at 15° for now.

**Next hypothesis to test:**
- Free transfer-function parameters in fitting (instead of keeping article table values fixed).
- Rationale: transfer-function parameters from the article may have been tuned for transient/min-scale dynamics, while our objective is firing-rate-scale behavior. The mismatch in scale/operating regime may prevent valid ring-level solutions.

---

### Ring Fitting with W&W-Grounded Transfer Function & Direct Ring Optimization (2026-03-30)

#### Transfer function rescaling

The transfer function was rescaled to be closer to the Wang (2002) / Wong & Wang (2006) parameterisation (see `docs/transfer_function.md`). This means the operating point of each population is now in the correct nA-scale regime, rather than the dimensionless-threshold regime used previously.

#### Noise process change

The noise is no longer an additive current. Instead, the baseline PYR current `I0_pyr` is multiplied by a noise factor:

$$I_\text{pyr}^\text{noisy}(t) = I_0^\text{PYR} \times (1 + \sigma_\text{noise} \cdot \xi(t))$$

This makes the noise amplitude proportional to the network's own drive, giving a network-space-relative noise current rather than an absolute additive one. The noise factor is also included in the optimization process, so the fitting takes into account the variance induced by this noise when evaluating loss against firing rate targets.

#### Fitting directly on the ring

Fitting at the single-node level and then searching for ring parameters that allow bump formation proved almost intractable: the single-node steady state provides no guarantee that the ring will support a bump, and the parameter space exploration was extremely inefficient. The approach was changed to fit directly on the ring, so that the ring's steady state (homogeneous activity across nodes, no bump) matches the experimentally observed firing rates. This is also more biologically realistic: the observed firing rates are measured in the intact network, not in an isolated single node.

#### Current results

- The ring reaches a steady state consistent with the experimental observations (correct population firing rates, fixed transfer function). This is a positive result.
- Upon cue presentation, a bump forms at the expected location.
- However, the bump does not sustain during the delay period — it decays rapidly after cue offset.
- Deactivating the PYR adaptation current does not rescue bump persistence.

#### Open questions

The lack of sustained bump raises three hypotheses, not yet disambiguated:

1. **Parameter set** — the fitted synaptic weights (single-node level, ring-level parameters, or both) do not place the network in the attractor basin required for self-sustained activity. The bump forms but the network is not truly bistable at those parameters.

2. **Network/ring structure** — unlikely, given that the ring connectivity per se (Gaussian PYR→PYR with global PV inhibition) is well established in the literature to support bumps. But cannot be fully excluded.

3. **Missing equations / mechanisms** — the model currently uses a single synaptic time constant and does not distinguish NMDA from AMPA recurrent excitation. NMDA-mediated slow recurrent excitation (τ ~ 100 ms) is widely considered the key mechanism for bump self-sustainment in working memory models (Wang 2002, Compte et al. 2000). Its absence may prevent the network from maintaining the attractor state across the delay.

---

### New Observation: Turing Targeting During Cue/Offset Transition (2026-03-30)

We made multiple changes to the Turing function, mostly to account for the inhibitory population effect at higher current during cue presentation.

With the new optimization process, we now mostly reach two regimes after cue offset:

- the activity decays rapidly (typically within ~500 ms after cue offset), or
- the activity keeps increasing until it reaches the 200 Hz cap.

There is no intermediate regime where activity remains self-sustained around ~80 Hz (or lower) after cue presentation.

Current interpretation:

- we are likely pushing the Turing criterion too far above 1 during cue (high PYR current),
- while what seems needed for stable post-cue persistence is a value close to 1 from below ($\approx 0.999$) specifically at cue offset timing.

#### Turing Function Shape Problem: Logarithmic Behavior and Runaway Risk

With the actual parameter set, the Turing function shows **logarithmic behavior with a ceiling slightly above 1**. 

The core problem: even if we could achieve an extremely precise cue presentation that settles the network into the perfect firing rate around 0.99999 Turing value, **noise will make the firing rate fluctuate**. If this fluctuation pushes the network above the Turing value of 1, the activity will runaway exponentially. Since noise is unavoidable, we cannot rely on operating right at the edge.

**Proposed solution**: reshape the Turing function to behave as a **square function with a hard maximum at 1**. This would:
- Create a stable "ceiling" at the Turing value
- Allow the network to reach a robust fixed point that is noise-resistant
- Prevent runaway even when firing rates fluctuate above the current logarithmic trajectory

This geometric change in the Turing criterion would decouple network stability from requiring perfect (and impossible) precision in cue presentation.

