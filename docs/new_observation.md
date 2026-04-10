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

---

### Turing Loss: Fundamental Validity Problem of the Analytical Approximation (2026-04-08)

#### Problem statement

The current Turing penalty computes $G_\text{eff}$ at three analytically defined operating points (rest, bump, cue). Both the validity of those operating points and the way the cue current is defined are questionable.

**First problem — population co-evolution.**

All populations evolve together. PV in particular is tightly coupled to PYR via $w_{ep}$: when PYR fires at 40 Hz instead of 8 Hz, PV does not stay at its rest firing rate — it receives a much larger drive and its own firing rate increases substantially. Holding SOM and PV at rest values while evaluating $I^*_\text{PV,bump}$ therefore underestimates the actual PV input current at the bump, and consequently misjudges $\Phi'_\text{PV}$ and $G_\text{eff}$. The same applies to SOM, which is driven by PYR via $w_{es}$ and which in turn inhibits PV via $w_{sp}$.

**Second problem — adaptation current shifts the operating point.**

The bump operating point is found by inverting the bare PYR transfer function at 40 Hz. But during sustained bump activity, the adaptation current $I_\text{adapt}(t) = J_\text{adapt} \cdot r_\text{PYR}(t)$ has built up and reduces the effective PYR drive. The actual PYR input current at a sustained 40 Hz is therefore lower than the bisection gives — the network sits at a different point on the transfer function than assumed. Similarly, the adaptation state at the cue operating point depends on how long the cue has been presented, not just on the cue amplitude.

The same argument applies to the cue operating point: the current at cue presentation depends on both the cue amplitude and the instantaneous state of the adaptation variable, which depends on recent history.

**Consequence.**

The three analytical operating points are approximations that ignore coupled population dynamics and adaptation history. The resulting $G_\text{eff}$ values may be systematically off, so the loss is penalising the wrong region of parameter space.

#### Proposed approach: simulation-based Turing loss (ring only)

This loss replaces the analytical Turing block inside the **ring optimizer** and subsumes bump-support constraints previously delegated to `--bump_mode`. It is **only applicable in the ring optimizer**, not the local single-node optimizer: the cue stimulus and the inter-node weight $w^\text{inter}$ have no meaning outside the ring context.

Rather than approximating the operating points analytically, compute $G_\text{eff}(t)$ directly from the actual instantaneous state at each timestep of a forward ring simulation:

$$G_\text{eff}(t) = \frac{\Phi'_\text{PYR}(I_\text{PYR}(t))}{1 + g_\text{GABA}\, w_{pe}\, \Phi'_\text{PV}(I_\text{PV}(t))\, w_{ep}\, \Phi'_\text{PYR}(I_\text{PYR}(t))}$$

where $I_\text{PYR}(t)$ and $I_\text{PV}(t)$ are the actual per-node input currents at each timestep — already computed during the Euler step, requiring no additional circuit inversions. Adaptation and co-evolution of all populations are naturally included.

The loss is computed over two temporal windows:

- **Rest window** $[t_\text{burn-in}, t_\text{cue})$: the gain should be safely below 1 at the resting operating point.

$$\mathcal{L}_\text{rest} = \frac{1}{|W_\text{rest}|}\sum_{t \in W_\text{rest}} \max\!\left(0,\; G_\text{eff}(t) \cdot w^\text{inter} - (1 - m)\right)^2$$

- **Delay window** $[t_\text{cue}, t_\text{end}]$: the gain should be slightly above 1 to sustain the bump, then naturally fall below 1 as the PYR firing rate rises above the bump target (preventing runaway). The target is therefore $1 + \varepsilon$ (default $\varepsilon = 0.05$, i.e. $G_\text{eff} \cdot w^\text{inter} \approx 1.05$):

$$\mathcal{L}_\text{delay} = \frac{1}{|W_\text{delay}|}\sum_{t \in W_\text{delay}} \left(G_\text{eff}(t) \cdot w^\text{inter} - (1 + \varepsilon)\right)^2$$

The delay loss is unconditional over the delay window — no rate-band masking. The gain trajectory is expected to start high (cue), cross $1 + \varepsilon$, and stay near it during self-sustained activity. The quadratic form penalises deviations in both directions: too low means no attractor, too high means runaway risk.

The total loss is the sum of the two windows, weighted by `--turing_weight`:

$$\mathcal{L}_\text{Turing} = w_T \cdot \left(\mathcal{L}_\text{rest} + \mathcal{L}_\text{delay}\right)$$

**Scope.** The new loss lives in the ring optimizer (`circuit_model/ring/optimization.py`). The single-node optimizer keeps its existing analytical proxy for backward compatibility. In ring CLI, `--bump_mode` is deprecated and ignored because bump constraints are integrated in the new Turing trace loss.

**Noise.** The Turing simulation is run with noise disabled (`sigma_noise = 0`). The gain is a property of the deterministic operating point; stochasticity is handled by the rate loss, not the structural loss.

**Cue protocol.** The cue is an additive current on PYR only, in the range $[0.1,\, 0.5] \times I_0^\text{PYR}$, applied as a square pulse. The default is controlled by `--turing_cue_amplitude` (default 0.4, i.e. $I_0^\text{PYR} \times 1.4$ total). This replaces `--turing_cue_scale`.

#### Diagnostic plots to develop

---

### Firing Rate Loss: Relative Error Weighting (2026-04-09)

The ring rate loss uses MSPE (mean squared percentage error), i.e. each population contributes `((actual - target) / target)²`. This means the loss is **relative to the target**, not absolute.

A practical consequence: a 1 Hz miss on a low-firing population (e.g. PYR at 3 Hz) is penalised much more than the same 1 Hz miss on a high-firing population (e.g. PYR at 15 Hz):

- 1 Hz miss at 3 Hz target → `(1/3)² ≈ 0.11`
- 1 Hz miss at 15 Hz target → `(1/15)² ≈ 0.004`

**Is this the right weighting?** Probably yes biologically — a 1 Hz deviation from a 3 Hz baseline represents a ~33% change in activity, which is more meaningful than the same absolute deviation from a 15 Hz baseline (~7%). However, it is worth keeping in mind when interpreting loss values across conditions with different baseline firing rates (e.g. APP vs WT): a fit with the same absolute Hz errors will have a higher loss in the lower-firing condition.

#### Diagnostic plots to develop

1. **Main trace:** $G_\text{eff}(t) \cdot w^\text{inter}$ and $r_\text{PYR}(t)$ vs time on the same axis, with rest/delay windows shaded and horizontal lines at 1 and $1 + \varepsilon$.
2. **Phase portrait:** $G_\text{eff}(t) \cdot w^\text{inter}$ vs $r_\text{PYR}(t)$, scatter coloured by time phase — shows whether gain is a monotone function of rate and where it crosses the target.
3. **Population co-evolution:** $r_\text{PYR}(t)$, $r_\text{PV}(t)$, $r_\text{SOM}(t)$ — validates the co-movement assumption.
4. **Adaptation current:** $J_\text{adapt} \cdot r_\text{PYR}(t)$ — quantifies the operating-point shift relative to the bare transfer function.
5. **Transfer function slopes:** $\Phi'_\text{PYR}(t)$ and $\Phi'_\text{PV}(t)$ — which factor drives the change in $G_\text{eff}$.

---

### Ring-Level Optimization Insufficient for Bump Self-Sustainment (2026-04-09)

**Observation**: When optimizing **ring-level parameters only** (`w_pyr_pyr_inter`, `w_pv_global`, `sigma_pyr_deg`) while freezing all circuit parameters to A1 values, the optimizer was unable to achieve a self-sustained bump, even with Turing loss enabled.

**What was tried:**
- A2_ring variant: Locked A1 circuit (good resting rates) and optimized only ring connectivity for Turing bistability
- Expected outcome: Ring structure tuning alone might find the right balance between recurrent gain and inhibition
- Actual outcome: No self-sustained activity achieved; bump collapsed post-cue as in A1

**Interpretation:**
- **Ring structure alone is insufficient** — tuning connectivity weights cannot compensate for missing or insufficient mechanisms at the single-node level
- **Circuit parameters must co-evolve** — achieving bistability requires simultaneous adjustment of:
  - Single-node gain/transfer function properties (slope, operating point)
  - Synaptic efficacy (especially recurrent PYR→PYR and inhibitory balance)
  - Adaptation dynamics (strength and timescale)
  - External drive levels
- Recurrent connectivity alone cannot create an attractor if the single nodes don't have sufficient gain or the synaptic time constants don't support slow integration needed for persistent activity

**Implication for A2:**
- A2 (free all circuit parameters + ring parameters) is necessary, not optional
- Locking circuit to A1 and only varying ring params creates an infeasible optimization landscape
- The bistability constraint forces exploration of regions of parameter space far from the rate-matching A1 solution

---

### Spontaneous Bump at Rest: Non-Uniform Resting State After Optimization (2026-04-09)

**Observation**: Recent fits produce states where the average firing rate is acceptable but the ring is not at a homogeneous resting state. Depending on the noise seed, a bump appears spontaneously at a random location, or a wandering bump is present during the rest period. This is physiologically incorrect: before cue presentation, all nodes should fire at approximately the same rate.

**Root cause**: The ring-rate loss averages PYR rates over nodes before computing the error — `ring_means = mean(r_nodes)` — so a spatially non-uniform state (spontaneous bump) can match the target mean while having large node-to-node variance. The optimizer is therefore not penalised for finding this wrong regime.

**Existing mechanism**: A `spatial_uniformity` penalty already exists in the code (`spatial_uniformity_weight`, default `0.0`). It penalises the coefficient of variation of PYR rates across nodes at rest:

$$\mathcal{L}_\text{sp\_unif} = w_\text{sp} \cdot \text{CV}^2_\text{PYR,rest}$$

where $\text{CV} = \sigma(\mathbf{r}_\text{nodes}) / \mu(\mathbf{r}_\text{nodes})$, averaged over trials.

**Proposed action**: Enable this penalty by setting `--spatial_uniformity_weight` to a non-zero value. The weight should be large enough to suppress spontaneous bumps but not so large that it prevents legitimate bump formation after cue. A reasonable starting point is `spatial_uniformity_weight = 1.0`, then sweep if needed.

---

### WT_APP Fitting Failure and Silent Synapses Hypothesis (2026-04-09)

**Observation**: When fitting WT_APP with all circuit parameters free (including jacobian loss), the optimization yields poor firing-rate matches, particularly:
- Large discrepancy in PYR firing rate even without KO constraints
- Jacobian regularization appears to prevent the network from finding valid solutions

**Hypothesis — Silent Connections in APP Condition**:
In the APP condition, some synaptic connections may become functionally silent (very low effective coupling) compared to WT. The jacobian loss, which penalizes having low/zero derivatives (unused pathways), prevents the optimizer from exploring parameter regions where certain connections have minimal contribution. This creates an artificial constraint that does not reflect biological reality: in the real APP network, some connections may genuinely have reduced efficacy.

**Proposed Experimental Approaches**:

1. **Approach 1: Nicotinic receptor modulation only (no jacobian loss)**
   - Start from WT-fitted parameters (established good baseline)
   - Free **only** the nicotinic acetylcholine receptor (nAChR) activation to vary between conditions
   - Remove jacobian loss entirely for this comparison
   - Rationale: Tests whether acetylcholine signaling differences alone explain WT→APP divergence
   - Limitation: Assumes network weights cannot change, which is biologically unrealistic

2. **Approach 2: Constrained weight variation with nAChR freedom**
   - Start from WT-fitted parameters
   - Allow synaptic weights to vary within a constrained window (~±30% bounds)
   - Fix the I0 (baseline input current) - assume no change in driving force
   - Free **only** the nAChR activation parameter across WT vs APP
   - Rationale: Permits modest adaptive changes in connectivity while keeping exploration space bounded
   - More biologically plausible: allows for some plasticity/compensation while preventing extreme parameter drift

**Next Steps**:
1. Implement Approach 1 as a diagnostic: run WT_APP fit with jacobian_weight=0 and free only nAChR parameter
2. Compare firing rates achieved in Approach 1 vs full free-parameter optimization
3. If Approach 1 explains the difference well, conclude that APP effects are primarily acetylcholine-mediated
4. If not, proceed to Approach 2 with constrained weight variation

---

### Post-Optimization Bump Pathology: Spiky Activity and Adaptation Dependency (2026-04-10)

**Observation**: The parameter set produced by the ring optimizer displays a pathological bump structure. Within the bump, individual nodes do not fire at a smooth, slowly-varying rate — instead, neighbouring nodes alternate between near-silent and high-firing states, producing a spiky, non-uniform profile across the bump width (see `figs/ring/spiky_bump`).

**Removing adaptation worsens the situation**: when the PYR adaptation current is disabled, the bump cannot sustain itself even during cue presentation — it collapses while the cue is still active (see `figs/ring/no_adapt_bump`). This is counterintuitive: adaptation is classically expected to limit persistent activity, not to be necessary for it.

**Interpretation**: The optimizer may have found a regime where strong inhibitory drive onto PYR silences most nodes very rapidly. In this regime, the adaptation current on PYR acts as a slow inhibition-release mechanism — it partially counteracts the tonic inhibitory overdrive, allowing a subset of nodes to fire. Removing adaptation therefore removes the only mechanism that keeps any nodes above threshold, explaining the collapse. The spiky intra-bump profile is consistent with this picture: only nodes where local excitation transiently overcomes inhibition manage to fire, giving an irregular alternating pattern rather than a smooth Gaussian envelope.

**Hypothesis — slow inhibitory adaptation as a sustaining mechanism**: The pathological regime suggests that what is functionally needed is a slow process that progressively disinhibits PYR on a timescale longer than the synaptic transient. One candidate biological mechanism is a long-timescale adaptation current on inhibitory interneurons (SOM or PV), which would reduce their firing over the delay period and thereby release PYR from tonic suppression. This is conceptually distinct from adaptation on PYR: rather than adapting the excitatory population, one adapts the inhibitory population to create a slow "disinhibitory ramp" that stabilises the bump.

**Possible biological substrate**: Cholinergic modulation via acetylcholine (ACh) acting on muscarinic receptors on SOM or PV interneurons could mediate such long-timescale disinhibition. ACh-induced suppression of interneuron firing (known as "ACh-induced disinhibition") is well documented in cortex and could provide the slow timescale needed. This would also connect naturally to the nAChR framework already in the model. Needs literature review to ground the timescale and the specific interneuron subtype.

**Immediate actions**:
1. Confirm the inhibitory overdrive hypothesis by inspecting PV and SOM firing rates in the spiky-bump regime.
2. Try adding a slow adaptation current on PV or SOM (long `tau_adapt_inh`, e.g. 500–2000 ms) and test whether it regularises the bump.
3. Search for biological references on ACh-induced interneuron suppression and its timescale.

### Bistable Single-Node Optimization: Conceptual Issue with Resting State Targets (2026-04-10)

**Context**: After implementing a bistable loss (`--mode bistable`) that
enforces a two-fixed-point nullcline structure at the single-node level,
the optimizer consistently fails to simultaneously satisfy the bistability
constraint and match the experimental firing rates.

**The conceptual problem**:
The bistable optimization uses Koukouli WT firing rates as the target for
the *low* fixed point of the bistable system. But these experimental rates
may not represent the true low-state firing rate. If the network operates
in a bistable regime where it spontaneously switches between a near-silent
low state (~0–2 Hz) and an active high state (~30–50 Hz), then the
time-averaged firing rate observed experimentally is a mixture:

    r_observed = p_low * r_low + p_high * r_high

where p_low and p_high are the fractions of time spent in each state.
The Koukouli calcium transient data (transients/min) captures this average,
not the instantaneous rate in either state.

**The transfer function constraint**:
The Wong-Wang transfer function parameters are shape-constrained (frozen
Theta, alpha, g values). With PYR resting at ~6–8 Hz, the operating point
is already above threshold — the neuron is in the approximately linear
regime of the transfer function, not near the silent low state that
classical bistable WM models use (~0–2 Hz low state, ~40–60 Hz high state).
This leaves very little geometric room for a fold in the nullcline at the
correct rate: fitting the low fixed point at 8 Hz while also requiring a
fold to exist there forces the transfer function slope and the recurrent
gain to be simultaneously in a narrow and possibly infeasible regime.

**Implication**:
The resting state firing rates from Koukouli may not be usable as direct
targets for the low fixed point in a bistable optimization. The correct
approach may require either:
1. Treating Koukouli rates as mixture averages and inferring the true
   low-state rate as a free parameter (with mixture fraction p_low as
   an additional unknown)
2. Accepting that this circuit operates in a different bistable regime
   than classical WM models (higher low state, lower high state, smaller
   separation)
3. Reconsidering whether single-node bistability is the right attractor
   mechanism for this model, given the transfer function constraints

**Optimizer result (WT, budget 10k)**:
- Regime found: BISTABLE (2 crossings)
- Low FP: r_PYR = 6.73 Hz (target 8.21, error −18%)
- High FP: r_PYR = 26.07 Hz
- SOM at low FP: 0.00 Hz (target 4.29, error −100%)
- PV at low FP: 0.11 Hz (target 4.07, error −97.4%)
- VIP at low FP: 8.19 Hz (target 6.05, error +35%)
- L_bistab = 1.292 (sign pattern not fully satisfied)
- L_rate = 2.106 (interneuron rates very far from targets)
- Regime classification: BISTABLE but biologically implausible
  (interneurons nearly silent at low FP)


### Single-Node Bistable Optimization: Failure Analysis and Decision to Move On (2026-04-10)

**Context**: Following the diagnosis that bump collapse is due to absence of
genuine single-node bistability, a dedicated bistable optimizer was implemented
(`--mode bistable`) with a loss function enforcing a two-fixed-point nullcline
structure (L_bistab + L_rate + L_margin + L_ceiling + L_jac). Multiple
optimization runs were attempted across different target configurations.

---

**Attempts and results**:

*Run 1 — WT targets, standard rates (r_low = 8.21 Hz, r_high = 30 Hz)*:
- Regime reported by optimizer: BISTABLE (classifier bug — later fixed)
- Regime confirmed by nullcline script: MONOSTABLE
- Low FP: r_PYR = 6.73 Hz; SOM = 0.00 Hz; PV = 0.11 Hz; VIP = 8.19 Hz
- L_bistab = 1.292, L_rate = 2.106 — both nonzero, no valid solution found
- Interneurons nearly silent at low FP — degenerate solution

*Run 2 — Lower rate targets (r_low = 1.75 Hz, r_high = 60.2 Hz)*:
- Regime: MONOSTABLE
- Low FP: r_PYR = 0.16 Hz; SOM = 0.00 Hz; PV = 0.71 Hz
- L_bistab = 2.611, L_margin = 30 — bistability not achieved
- Interneurons still near-silent

*Run 3 — WT targets, rebalanced loss after classifier fix*:
- Regime: MONOSTABLE
- L_bistab = 9.176, L_margin = 30 — complete failure
- VIP = 12.35 Hz (104% above target) while SOM, PV = 0

Across all runs the optimizer converged to the same degenerate pattern:
VIP elevated, SOM and PV silenced, PYR near zero. This is the only way the
optimizer can reduce inhibitory tone enough to create any fold at all —
but even then the fold doesn't appear within the physiological rate range.

---

**Root cause analysis**:

The failure is structural, not a matter of optimizer budget or weight tuning.
Two independent issues compound each other:

*Issue 1 — Transfer function has no saturation above threshold.*
The Wong-Wang transfer function (Abbott & Chance 2005, fitted to LIF neurons):

    Phi(I) = (c·I - I0) / (1 - exp(-g·(c·I - I0)))

is approximately linear above threshold (for large g, it reduces to a
linear-threshold function). It does not saturate. For a fold in the PYR
nullcline to exist, the effective recurrent input I_net(r_PYR) must decrease
over some range of r_PYR — i.e. inhibitory feedback must overcome excitation
in that range. But without transfer function saturation, Phi(I_net) grows
roughly as fast as r_PYR indefinitely, so the nullcline never bends back
below the identity line. The only way to force the fold is to make I_net
itself strongly non-monotone, which requires silencing the inhibitory
populations — the degenerate solution the optimizer keeps finding.

*Issue 2 — Fast-synapse approximation removes the NMDA saturation mechanism.*
In the original Wong-Wang (2006) model, bistability does not come from the
transfer function saturating. It comes from the NMDA gating variable S:

    dS/dt = -S/tau_NMDA + (1 - S) · gamma · r

The (1-S) term causes S to saturate in [0,1] regardless of how high r goes.
Since recurrent excitation enters as J_NMDA · S (not J_NMDA · r), the
effective excitatory drive saturates even though Phi itself does not. This
is what creates the fold.

In our model we use the fast-synapse approximation S ≈ tau_s · r, which is
linear in r — valid for AMPA/GABA timescales but it discards the saturation
that NMDA gating provides. The approximation is appropriate for the synaptic
timescale dynamics but structurally eliminates the bistability mechanism.

*Issue 3 — Experimental rate targets may be mixture averages.*
The Koukouli calcium transient data represents time-averaged firing rates.
If the circuit operates in a bistable regime, the observed ~8 Hz PYR rate
is a mixture: r_obs = p_low · r_low + p_high · r_high. The true low-state
rate could be near-silent (~1-2 Hz) and the high-state rate ~40-50 Hz, with
most time spent in the low state. Using 8 Hz directly as the low fixed point
target is therefore conceptually incorrect and makes the optimization harder:
the optimizer must place the fold at an operating point that may not
correspond to any true fixed point of the bistable system.

---

**Near-bifurcation attempt with standard optimizer (2026-04-10)**:

Before committing to architectural changes, the standard optimizer
(`--mode standard`) was run to search for a near-bifurcation regime —
parameters where single-node gain is close to but below 1 at rest, with
bump persistence provided by the ring architecture rather than single-node
bistability (consistent with Wimmer et al. 2014).

Result: no satisfactory solution found. The optimizer consistently failed
to simultaneously match the Koukouli firing rate targets and produce
parameters in a near-bifurcation regime where the ring could sustain a bump.
This approach was not exhaustively explored — a more systematic search with
larger budget, tighter gain constraints in the loss, or explicit near-
bifurcation penalty terms may yet succeed. This remains a possible fallback.

---

**Why we are not pursuing bistable optimization further (for now)**:

Fixing the transfer function saturation problem requires one of:

1. Adding the NMDA gating variable S explicitly to PYR→PYR recurrence,
   replacing w_ee · r with J_NMDA · S. This restores the fold mechanism
   with biological justification (PFC recurrent excitation is NMDA-dominant).
   tau_NMDA = 100 ms and gamma = 0.641 are fixed from Wong-Wang, so J_NMDA
   replaces w_ee as the single free parameter for local recurrent excitation.

2. Returning to the near-bifurcation approach with a more systematic search
   (larger budget, explicit gain-proximity penalty).

**Decision: attempt Option 1 first** — add NMDA gating to PYR local
recurrence. The biological justification is strong (PFC WM is NMDA-dependent,
consistent with the literature and with Koukouli's own nAChR framework which
modulates NMDA-mediated activity). The architectural change is minimal:
one new dynamical variable S_PYR per node, two fixed parameters (tau_NMDA,
gamma), one free parameter (J_NMDA replacing w_ee). If NMDA gating restores
bistability and allows bump persistence, the near-bifurcation fallback
(Option 2) is shelved. If NMDA gating introduces new fitting problems or
breaks the Koukouli rate targets, Option 2 is revisited with a more
exhaustive search.

---

**Immediate next step**:
Implement NMDA gating on PYR local recurrence:
- Replace w_ee · r_PYR with J_NMDA · S_PYR in the PYR input equation
- Add ODE: tau_NMDA · dS/dt = -S + (1-S) · gamma · r_PYR
  with tau_NMDA = 100 ms, gamma = 0.641 (fixed, from Wong-Wang 2006)
- J_NMDA becomes the free parameter replacing w_ee
- Update nullcline_analysis.py to use J_NMDA · S*(r) in I_net computation
- Update bistable_loss.py accordingly
- Rerun nullcline diagnostic to confirm fold appears before full optimization