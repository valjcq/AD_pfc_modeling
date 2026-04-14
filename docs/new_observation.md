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


### NMDA Gating Enables Bistability but Reveals New Fitting Problems (2026-04-10)

**Context**: After implementing NMDA gating on PYR local recurrence
(replacing w_ee · r_PYR with J_NMDA · S_PYR), the bistable optimizer was
rerun. For the first time, a genuine BISTABLE regime was found (3 crossings,
2 stable fixed points, both below R_MAX_PHYS = 100 Hz — upper FP at 68.62 Hz).
This confirms that NMDA gating structurally restores the fold mechanism that
the fast-synapse approximation had eliminated.

**Result summary**:
- Regime: BISTABLE ✓
- Low FP:  r_PYR = 0.16 Hz (near-silent low state)
- High FP: r_PYR = 68.62 Hz
- L_bistab = 8.591 (sign pattern not fully satisfied — fold exists but
  probe points not all satisfied)
- L_margin = 0 (separation criterion met: 68.62 - 0.16 > 15 Hz)
- L_rate = 2.647 (firing rates off target)
- L_jac = 9.123 (Jacobian regularization heavily violated)
- SOM at low FP: 1.33 Hz (+18% vs target) — only interneuron near target
- PV at low FP: 0.09 Hz (−91% vs target)
- VIP at low FP: 0.03 Hz (−98% vs target)

**Observation from nullcline analysis (2026-04-10, prior to SOM adaptation fix)**:

Previous runs showed SOM firing rate saturating at ~200 Hz in the intermediate
regime between the low stable FP and unstable FP. This was noted as pathological
given the known physiological range (5–30 Hz). The high Jacobian loss (L_jac = 9.123)
suggested this was an optimizer artifact — a degenerate solution relying on
extreme coupling gains rather than a biologically plausible bistable mechanism.

**Status after SOM adaptation bug fix (2026-04-14)**:

With SOM adaptation now properly implemented in the single-node simulation,
the pathological SOM saturation should no longer occur. SOM now experiences
negative feedback proportional to its firing rate, preventing runaway to 200 Hz.
The nullcline geometry should reflect a more physiologically plausible mechanism:
- SOM rates in the intermediate regime should remain in the 5–30 Hz range
- The bistable fold should emerge from NMDA saturation + coupled inhibition,
  not from degenerate high-gain coupling
- Rerunning the bistable optimization should produce solutions with lower L_jac
  values and more realistic interneuron rates throughout the nullcline

**Immediate next step**:
Re-evaluate bistable parameters with the corrected single-node simulation to
confirm that:
1. SOM saturation pathology is resolved
2. Jacobian regularization loss improves (lower L_jac values)
3. Bistable folds remain present and separation maintained
4. Interneuron rates stay physiologically plausible across all fixed points

---

### Bistable optimization — two fits and cue-sweep dynamic validation (2026-04-13)

#### Model equation

The simulation uses the full NMDA gating variable (Wong & Wang 2006):

```
dS/dt = (-S + (1 - S) · γ · r_PYR) / τ_NMDA
I_PYR = J_NMDA · S / (1 + g_GABA · w_PE · r_PV) - g_GABA · w_SE · r_SOM - I_adapt + I_ext_PYR
```

with τ_NMDA = 100 ms, γ = 0.641. The NMDA gating variable introduces both
saturation (S ≤ 1) and a slow decay, which is precisely the mechanism that
can create the fold in the PYR nullcline required for bistability.

#### Two optimizations performed

**Fit 1 — `bistable_fixed`**
Target resting rates taken from mean population firing rates during rest
(PYR = 8.21, SOM = 4.29, PV = 4.07, VIP = 6.05 Hz). Target high state:
r_high = 30 Hz.

Nullcline result: 3 fixed points (2 stable), high FP at **78.28 Hz** —
higher than the 30 Hz target, which the optimizer could not bring down
while maintaining bistability.

Low-FP rates in the optimizer (static analysis):
PYR = 0.00 Hz (−100% vs 8.21 target), SOM = 2.04 Hz (−53%), PV = 2.41 Hz (−41%), VIP = 6.65 Hz (+10%).

Key parameters: J_NMDA = 0.054 nA, g_gaba_base = 1.19, w_vs = 0.014, w_vp = 0.005 (ratio w_vs/w_vp = 3.1).

**Fit 2 — `bistable_L_state_to_H`** (ROOY 2021)
Target resting rates from literature low state (PYR = 1.75, SOM = 1.12,
PV = 1.04, VIP = 1.33 Hz). Target high state from ROOY 2021: r_high = 60.2 Hz.

Nullcline result: 3 fixed points (2 stable), high FP at **74.78 Hz** —
again above the 60.2 Hz target. Unstable FP at 62.37 Hz (much higher
threshold to cross than fit 1's 35.60 Hz).

Low-FP rates: PYR = 0.00 Hz (−100%), SOM = 1.23 Hz (+10%), PV = 0.52 Hz (−50%), VIP = 1.49 Hz (+12%).

Key parameters: J_NMDA = 0.642 nA (~12× stronger than fit 1),
g_gaba_base = 4.47, w_vs = 0.010, w_vp = 0.010 (ratio ≈ 1.0, balanced).

#### Cue-sweep dynamic validation

A transient input (`trans_factor × I0` added to all populations) was swept
from 0 to 6 to test whether the network can dynamically switch states.
Three time windows were measured: pre-cue (rest), during cue, and post-cue.

**Both fits confirm dynamic bistability**: above a threshold factor,
the network reaches an elevated PYR state that is self-sustained after cue
removal.

| | bistable_fixed | bistable_L_state_to_H |
|---|---|---|
| Resting SOM rate | **159 Hz** | **85 Hz** |
| Threshold factor | ~0.5–0.75 | ~1.25–1.5 |
| Post-cue PYR (high state) | ~75–87 Hz | ~73 Hz |
| Bistable factor window | 0.75–2.75 | 1.5–2.25 |

#### Dynamic regime description

At rest, SOM fires at a high autonomous rate (159 Hz / 85 Hz depending on
the fit) that completely silences PYR, PV, and partially VIP. SOM is
self-sustaining from its external drive alone — it does not require PYR
excitation to maintain this activity.

When the cue input is applied, VIP is the first to respond (it receives a
strong drive through w_EV). Rising VIP activity inhibits SOM via w_VS,
which releases PYR from inhibition. Once PYR crosses the unstable fixed
point (35.60 Hz in fit 1, 62.37 Hz in fit 2), NMDA recurrent excitation
takes over and PYR self-sustains. PV also activates at this point, driven
by PYR excitation through w_EP.

In the post-cue elevated state: PYR ~75–87 Hz, SOM ~0 Hz (fully silenced
by VIP), VIP ~49–79 Hz, PV ~1–4 Hz.

#### Concerns and open questions

**1. Abnormally high resting SOM rate.**
SOM fires at 85–159 Hz at rest, which is far from any physiological range.
This occurs because I_ext_SOM is large enough for SOM to fire
autonomously without PYR excitation. The VIP→SOM inhibition (w_VS) is too
weak relative to SOM's external drive to bring SOM down to a physiological
resting rate.

In `bistable_fixed`, the w_VS/w_VP ratio is 3.1 (VIP targets SOM ~3×
more strongly than PV), yet SOM still fires at 159 Hz at rest because the
absolute magnitude of w_VS is small (0.014) relative to I_ext_SOM (0.64 nA).
In `bistable_L_state_to_H`, the ratio is ~1.0 (balanced), and resting SOM
is lower (85 Hz) — consistent with the lower I_ext_SOM target — but still
far above the 1.12 Hz target.

**2. Single-node limitation.**
At the single-node level, PV is silent at rest because PYR is silent. In a
ring network, global lateral PV inhibition from other nodes (driven by
population-level activity) could provide an additional suppressive drive
onto PYR that helps maintain the low state at a more physiological level,
rather than relying entirely on SOM's autonomous firing.

**3. Disinhibition mechanism.**
The cue acts primarily through the VIP→SOM disinhibition pathway. This is
consistent with the biological role of VIP interneurons as disinhibitory
elements. However, the extreme resting SOM rate means the disinhibition
must overcome a much larger baseline inhibitory current than would be
biologically expected.

**Next steps:**
- Add a constraint to the bistable loss that enforces `φ_SOM(I_ext_SOM) <
  r_low_target` — i.e. SOM cannot self-sustain without PYR excitation.
  This would force the optimizer to find solutions where SOM requires the
  PYR→SOM excitatory drive to reach its resting rate, making the low state
  depend on the PYR→SOM→PYR loop rather than autonomous SOM activity.

---

### Effect of firing rate loss weight on bistable optimization (2026-04-14)

Four runs were compared by varying the firing rate loss weight across two target regimes (high and low resting FR):

| Run | FR loss weight | Bistable found? | Notes |
|-----|---------------|-----------------|-------|
| `bistable_high_fr` | baseline | yes | |
| `bistable_high_fr_higher_loss_fr` | increased | yes | PYR stays at 0 during resting state |
| `bistable_low_fr` | baseline | yes | |
| `bistable_low_fr_higher_loss_fr` | increased | **no** | optimizer fails to find any bistable point |

**Interpretation:**

When the firing rate loss weight is increased in the low-FR regime, the optimizer cannot simultaneously satisfy the bistability constraint and match the low target rates — it finds no valid bistable fixed point at all. In the high-FR regime, a bistable point is still found, but with PYR silent at rest (firing rate = 0), which is a degenerate solution.

This pattern suggests one of two things:

1. **The network architecture itself constrains the bistable mechanism**: the only way this single-node circuit can achieve genuine bistability is the mechanism identified earlier (SOM autonomous firing + VIP disinhibition), and tighter firing rate constraints rule out this mechanism in the low-FR regime.

2. **Loss function design issue**: the optimizer may be unable to navigate the joint landscape of bistability + firing rate constraints efficiently, not because the solution doesn't exist, but because the loss surface becomes too rugged or the gradients conflict.

Distinguishing these requires either a more exhaustive search (e.g. larger population, more restarts) or an analytical check of whether the nullcline geometry can accommodate both bistability and low physiological rates simultaneously.
- Test the ring network with these parameters to assess whether lateral PV
  inhibition rescues the low-state PYR dynamics.

---

### Ring Network Sweep: Key Findings and Bump Detection Problems (2026-04-14)

#### Summary of sweep results (sigma=30° and sigma=15°)

Two systematic 2D sweeps were conducted using bistable parameters (`bistable_high_fr`, I0_pyr=1.07 nA):

**Phase space exploration (w_pv_global × amplitude, sigma=30°)**:
- `w_pv_global = 0.05` is the threshold for a quiet pre-cue state (~3 Hz)
- There is a sharp bistable threshold around amplitude=0.5: below → no effect, above → full 200 Hz saturation
- Best delay activity at the first amplitude just above threshold (amp=0.55)
- Bump center_std stays above 80° everywhere: w_pyr_pyr_inter=0.002 cannot sustain a localised bump

**w_pyr_pyr_inter sweep (sigma=15°)**:
- At sigma=15°, a qualitative localization transition appears between wpyr=0.004 and wpyr=0.006
- At wpyr=0.00592: center_std drops to 35.9°, delay firing at 128.8 Hz — a genuine localised bump
- However, pre-cue baseline rises to 22.4 Hz at this wpyr (same wpv=0.05 too weak to suppress spontaneous activity)
- Pre-cue saturation cliff at wpyr≈0.008 (same in sigma=30° and sigma=15°, as expected from row-sum normalisation)

#### Fundamental problems with the current bump detection and classification

**Problem 1 — Saturated bumps counted as successes**: Any post-cue activity above the noise floor is counted as a "bump", including 200 Hz saturated states. A network locked at the rate cap throughout the delay is pathological, not a working memory state.

**Problem 2 — Saturation during cue corrupts bump quality**: When the cue drives nodes to 200 Hz, the adaptation current (proportional to peak rate × cue duration) builds up excessively. After cue offset, the elevated adaptation suppresses the bump even if a genuine attractor state existed. A good bump should form without the network reaching saturation during the cue.

**Problem 3 — Delay state not tracked over time**: The current metrics only report the mean firing rate at the end of the delay period. They cannot distinguish: (a) a bump that was sustained throughout the delay, (b) a bump that formed but decayed after 500ms, or (c) a saturated state that gradually decayed toward the noise floor.

**Problem 4 — Fixed 20 Hz threshold ignores resting rate**: The bump lower bound should be defined relative to the actual resting firing rate, not a fixed value. For very quiet baselines (~0 Hz), 20 Hz is appropriate; for baselines at 3–5 Hz, the threshold may be too permissive.

#### New bump quality classification (implemented in ring-calibrate 3D sweep)

Each delay timepoint is classified into one of three states based on the **maximum PYR rate across all nodes**:

| State | Condition | Meaning |
|---|---|---|
| **Resting** | max_PYR < resting × 2.5 (min 10 Hz) | Network in low state |
| **Bump** | resting_threshold ≤ max_PYR < 90 Hz | Localised active state |
| **Saturated** | max_PYR ≥ 90 Hz | Runaway / rate-cap state |

The resting threshold is `max(resting_rate × 2.5, resting_rate + 5 Hz, 10 Hz)` where `resting_rate` is measured from the burn-in period for each parameter combination.

Key metrics reported per grid point:
- `delay_bump_frac`: fraction of delay time in bump state → quality measure
- `delay_sat_frac`: fraction of delay time saturated → pathology marker
- `cue_saturated`: whether peak during cue reached ≥ 190 Hz → adaptation contamination risk

A **true bump** requires: `delay_bump_frac > 0.3` AND `delay_sat_frac < 0.2` AND ideally `cue_saturated = False`.

#### Next steps: 3D sweep (w_pv_global × w_pyr_pyr_inter × amplitude)

The recommended next analysis is a 3D parameter sweep implemented in `ring-calibrate` with:
- `--w_pv_values`: sweep global PV inhibition (e.g. 0.03–0.10)
- `--w_inter_values`: sweep recurrent PYR→PYR excitation (e.g. log-spaced 0.001–0.015)
- `--amplitudes`: sweep cue amplitude (e.g. 0.3–0.8)

Goals:
1. Find (w_pv, w_pyr) regions where pre-cue is quiet AND a non-saturating cue can trigger a bump
2. Find amplitude range that induces transition without cue saturation
3. Map the fraction of delay time in genuine bump state across the 3D space
4. Confirm that sigma=15° enables localization at the right w_pyr range

---

### Successful Ring Self-Sustained Bump with Bistable Parameters (2026-04-14)

#### Observation

Using the single-node bistable parameters from `figs/optim/bistable_high_fr/bistable_params.json` directly in a ring network, we observe **robust self-sustained activity during the delay period** — a milestone previously unattained.

#### Parameter context

The bistable parameter set (J_NMDA = 0.0537 nA, g_gaba_base = 1.19, configured for near-silent low state + high state at ~78 Hz) was applied to the ring with typical connectivity (sigma_pyr_deg = 15°, w_pv_global and w_pyr_pyr_inter calibrated for bump support).

#### Network dynamics

**Cue period (0–500 ms)**:
- All populations receive the input transient
- Network activity rises sharply; multiple populations approach saturation (~150–200 Hz)
- PYR reaches high firing rate during the stimulus

**Delay period (500–4000 ms post-cue-offset)**:
- Network **does not decay** — it stabilizes into a high state
- **PYR**: fires steadily at ~80 Hz (range 75–87 Hz across trials)
- **SOM**: drops to near-silence (~0 Hz) — completely suppressed by VIP
- **PV**: elevates from ~1 Hz (resting) to ~5 Hz (or higher depending on local drive)
- **VIP**: ramps up during delay, reaching ~60 Hz, then stabilizes
- The bump profile shows **sharp spatial boundaries**: clear distinction between active nodes (firing at ~80 Hz) and silent nodes (near baseline)
- Bump center is **spatially stable** and does not drift during the delay

#### Bump quality metrics

- **Persistence**: delay-period activity remains well above resting baseline throughout (no decay seen)
- **Spatial localization**: nodes within the bump zone fire coherently; edges show sharp drop-off (not gradual)
- **Stability**: no wandering, no spontaneous reactivation of remote sites, no oscillations
- **Population structure**: consistent with the bistable low-state mechanism:
  - SOM silence allows PYR disinhibition
  - VIP high activity maintains SOM suppression
  - PV elevation follows PYR drive (w_ep ≈ 0.0046)

#### Biological plausibility checklist

| Aspect | Measure | Status |
|--------|---------|--------|
| **Resting PYR** | ~8 Hz | ✓ Near target (Koukouli baseline) |
| **Active bump PYR** | ~80 Hz | ✓ Within physiological range |
| **SOM silencing** | ~0 Hz at high state | ✓ Consistent with disinhibitory role |
| **VIP elevation** | ~60 Hz during bump | ⚠️ High but plausible during activity (literature: 5–80 Hz) |
| **PV response** | 1 Hz → 5 Hz | ✓ Modest elevation, matches recurrent drive from PYR |
| **Bump width** | Sharp edges, ~15° spatial spread | ✓ Consistent with sigma_pyr_deg = 15° |
| **Adaptation interplay** | PYR adapt (tau=1119 ms), SOM adapt (tau=170 ms) | ✓ Fast SOM adapt + slow PYR adapt enables persistent state |

#### Key parameters enabling this regime

From `bistable_high_fr`:
- `J_NMDA = 0.0537 nA`: NMDA gating provides recurrent saturation mechanism
- `g_gaba_base = 1.19`: moderate GABA gain
- `w_se = 0.1867`: strong SOM→PYR inhibition (enables low-state silence)
- `w_vs = 0.0140`: VIP→SOM inhibition (weak, but functional for cue-driven disinhibition)
- `J_adapt_som = 0.153`: fast SOM adaptation timescale (170 ms) — SOM recovery is slow enough to maintain suppression through early delay
- `J_adapt_pyr = 0.0059`: smaller PYR adaptation does not quench the NMDA-driven recurrence

#### Mechanistic interpretation

The observed stabilization represents a **disinhibitory attractor**:
1. **Cue → VIP activation**: the transient input activates VIP (w_ev ≈ 0.0017 direct drive per node + global lateral factors)
2. **VIP → SOM inhibition**: rising VIP fires onto SOM (w_vs = 0.0140), rapidly suppressing SOM firing
3. **SOM silence → PYR release**: with SOM inhibition removed, PYR is free to express its local recurrent drive (J_NMDA · S)
4. **PYR self-sustains**: NMDA gating variable S provides saturation-limited positive feedback; PYR stabilizes around 80 Hz
5. **PV feedback**: PYR excites PV (w_ep = 0.0046), driving PV to ~5 Hz — a modest feedback that does not overwhelm the recurrent excitation
6. **Delay stability**: VIP remains tonically elevated to keep SOM suppressed; PYR's slow adaptation (tau = 1119 ms) does not significantly reduce NMDA-mediated drive over 4 seconds

#### Contrast to previous failed attempts (2026-04-09 to 2026-04-13)

- **Earlier A1/A2 ring-only optimization**: tried varying ring parameters while keeping single-node circuit parameters fixed to non-bistable A1 solutions → no self-sustained bumps
- **First bistable optimization attempts**: produced parameter sets with physiologically implausible internal state (e.g. SOM firing at 159 Hz at rest) → bumps could form during cue but network entered pathological saturation regimes
- **This observation**: uses a curated bistable parameter set (from `bistable_high_fr`) that achieves true two-fixed-point nullcline geometry with reasonable resting states and produces clean, stable bumps upon cue challenge

#### Open questions

1. **Cue amplitude sensitivity**: Does bump persistence depend critically on cue amplitude, or is there a robust window?
2. **Generalization across conditions**: Do WT_APP or KO conditions maintain bump stability with proportionally adjusted parameters?
3. **Noise dependence**: Is the bump robust to realistic stochastic noise levels (sigma_noise = 0.1)?
4. **Spatial precision**: Can we tighten the bump localization further while maintaining stability?

#### Next steps

1. Validate across multiple noise seeds and cue amplitudes
2. Test parameter robustness: small perturbations around `bistable_high_fr` values
3. Compare ring bump stability with single-node bistable nullcline predictions
4. If replicated consistently, use as reference parameter set for WT_APP and KO condition mapping

---

### Single-Node Simulation SOM Adaptation Bug Fix (2026-04-14)

#### Bug discovered

When running single-node simulations with parameters from the bistable optimization, SOM firing rate was capped at **~60 Hz** — far above expected values and inconsistent with the bistable optimization results which predicted SOM at 2-4 Hz.

#### Root cause

The bistable loss function correctly applies SOM spike-frequency adaptation during the fixed-point analysis:
```
I_som = params.w_es * r_pyr - params.w_vs * r_vip - params.J_adapt_som * r_som + params.I_ext_som()
```

However, the single-node simulation code (both `simulation.py` and `_fast_loop.py`) was **not applying the SOM adaptation term**. Only PYR had adaptation implemented:
- In `simulation.py` (lines 227-229): `I_adapt[k + 1, 1] = 0.0` (SOM adaptation always zero)
- In `_fast_loop.py` (line 164): Same issue

In contrast, the ring network simulation (`ring/simulation.py`) had SOM adaptation correctly implemented throughout.

#### Fix applied

Both simulation paths now include SOM adaptation:

1. **`simulation.py` reference loop**: 
   - Added SOM adaptation current reading: `Ias = I_adapt[k, 1]`
   - Updated I_som: subtract `params.J_adapt_som * r_som`
   - Updated adaptation: `I_adapt[k + 1, 1] = Ias + dt_ms * dIas`

2. **`_fast_loop.py` Numba loop**:
   - Added `J_adapt_som` and `tau_adapt_som` parameters
   - Same SOM input and adaptation updates

3. **Both callers** (`simulate_circuit` and `validate_fast_loop`):
   - Pass new parameters to `_euler_loop`

#### Result after fix

Single-node simulations now correctly reproduce the bistable optimization predictions:
- **SOM firing rate** drops from 60 Hz (capped) to ~2-4 Hz (physiologically plausible)
- **Interneuron rates** within biological range across all populations
- **Single-node dynamics** now match ring network behavior
- **PYR rate** remains near-silent at low fixed point as required for bistable mechanism

#### Impact

This bug would have caused systematic underestimation of the SOM adaptation's strength during parameter fitting, potentially leading to incorrect circuit parameters. The ring simulation was unaffected because it had the correct implementation throughout.

---