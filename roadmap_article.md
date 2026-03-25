# Article Roadmap — Ring Attractor with Multiple Interneuron Classes in PFC Working Memory

**Working title**: *Interneuron-class-specific control of persistent activity in a prefrontal ring attractor: implications for Alzheimer's disease*

---

## Overview

This paper makes two core contributions:

1. **Methodological**: First ring attractor model to incorporate three distinct interneuron classes (PV, SOM, VIP) with realistic cholinergic modulation via nicotinic acetylcholine receptors (nAChRs).
2. **Biological**: Systematic dissection of each interneuron class's contribution to bump formation, and how disease-state circuit changes (WT_APP fitted family) degrade working memory.

The paper moves from model construction → validation → mechanistic dissection (KO studies) → disease case study (APP).

---

## Paper Structure

### Section 1 — Introduction

**Goal**: Motivate the model and establish novelty.

**Key points to make**:
- Working memory relies on sustained "bump" activity in PFC; ring attractor models are the canonical framework.
- Existing ring attractor models use single or generic inhibitory populations — they miss interneuron diversity.
- PFC contains at least three functionally distinct interneuron classes: PV (perisomatic, fast feedback), SOM (dendritic, slow adaptation), VIP (disinhibitory, top-down modulation).
- Each class expresses distinct nAChR subtypes (α7 on PV/SOM, β2 on SOM, α5 on VIP) → cholinergic tone differentially modulates each population.
- Alzheimer's disease (early stage: APP/Aβ accumulation) shifts the operating circuit state (captured by a dedicated WT_APP fitted parameter family).
- **Gap**: No ring attractor study has used a multi-class interneuron circuit to study how class-specific cholinergic dysfunction impacts bump quality.

**End of intro**: State the three goals — (1) characterize each interneuron's role via KO, (2) define bump quality metrics for each, (3) predict how the WT_APP circuit state degrades WM.

---

### Section 2 — Model Description

**Goal**: Present the circuit and justify design choices. Keep formal and concise; detailed math in Methods/Supplement.

#### 2.1 Local circuit

- Four-population rate model: PYR, PV, SOM, VIP.
- Equations: τ dr/dt = -r + Φ(I_det) + noise.
- Transfer function: Wong-Wang sigmoid (mean-field from spiking).
- Adaptation currents on PYR and SOM (with very different timescales: ~187 ms vs ~2300 ms).
- Connectivity motifs: feedback inhibition (PYR↔PV), dendritic inhibition (SOM→PYR), disinhibition (VIP→SOM→PYR).

Brief literature grounding for each class's expected role (cited, not derived from our simulations):
- PV: fast perisomatic feedback inhibition; classically associated with gamma oscillations and gain control (cite e.g. Cardin et al., Sohal et al.).
- SOM: slow dendritic inhibition; rate adaptation; suppresses sustained activity over long timescales (cite e.g. Silberberg & Markram, Yavorska & Wehr).
- VIP: disinhibitory interneuron; inhibits SOM → releases PYR from dendritic inhibition; modulated by top-down and cholinergic inputs (cite e.g. Pi et al., Lee et al.).

#### 2.2 Cholinergic modulation

- Parameterize three nAChR subtypes via activation multipliers (act_alpha7, act_beta2, act_alpha5).
- α7: drives PV and SOM; also scales GABA transmission strength.
- β2: drives SOM.
- α5: drives VIP.
- Table: receptor subtype → target population → effect on PYR (direct/indirect).

#### 2.3 Ring attractor

- N=128 nodes on a circle, each node = full local circuit.
- Connectivity: Gaussian PYR→PYR (local excitation, σ~15°) + global uniform PV→PYR inhibition.
- Working memory protocol: 500 ms baseline → 500 ms cue → 5000 ms delay.
- Optional distractor during delay (at varying angular offsets).

#### 2.4 Parameter fitting — local circuit

- Optimization target: in vivo spike rate data from WT and WT_APP mice (1-month timepoint; `AD_data/AD_spikes/datafiles/firing_rate_data.csv`).
- Loss function: relative MSE on per-population mean firing rates across conditions.
- Fitted parameters include synaptic weights, time constants, external currents, receptor currents.
- **Figure**: Simulated vs. experimental rates (WT and WT_APP); show good fit.

#### 2.5 Parameter strategy for KO and disease conditions

There are two distinct levels of fitting that must be treated separately.

---

**Level 1 — Local circuit (single-node) parameters**

These are the parameters of the 4-population local circuit: synaptic weights within a node, time constants, external currents, receptor currents. They are fitted to in vivo spike rate data (population mean firing rates from `AD_data/AD_spikes/`).

*Option A — Zero activation*: Use the WT-optimized local circuit parameter set and set the relevant receptor activation to zero (e.g. act_alpha7=0 for A7ko). This implicitly assumes no local synaptic compensation.

*Option B — Re-optimize on KO data*: Fit a new local circuit parameter set to experimental calcium imaging data from KO mice. This is biologically ground-truth but introduces other parameter differences beyond the receptor change.

**Plan**: Option B is the most biologically rigorous and should be the main approach — we have KO calcium imaging data. Option A is used as a sanity check and to isolate the pure receptor effect.

---

**Level 2 — Ring attractor (network) parameters**

These are the connectivity parameters specific to the ring: inter-node PYR→PYR excitation weight, global PV→PYR inhibition weight, spatial spread (sigma_pyr_deg). These are not constrained by the calcium imaging data and need a separate choice.

*Option A — WT network, all conditions*: Use the ring connectivity fitted on WT for all conditions. This assumes no network-level adaptation — the ring's spatial connectivity is the same regardless of condition.

*Option B — Condition-specific network fitting*: Fit the ring connectivity separately per condition, using a physiological constraint. The natural anchor is the mean PYR firing rate during the delay period, which is observed to be ~18 Hz in rodent PFC during WM tasks (Compte et al. 2003). For each condition, find the ring connectivity parameters that achieve this rate. This accounts for the possibility that in vivo networks adjust their spatial connectivity to maintain WM function despite disruptions.

**Plan**: Use Option A as a first pass. Use Option B (rate-matched at ~18Hz PYR during delay) as the main comparison reported in the paper — it grounds each condition in an experimentally-observed WM operating point and enables fair between-condition comparison of bump quality. Present both if they lead to qualitatively different conclusions. The fact that the operating regime shifts between WT and WT_APP is itself an interesting finding that suggests in vivo compensation may be necessary.

---

### Section 3 — Bump Metrics and Baseline Characterization

**Goal**: Define the readout, establish baseline (WT) behavior, and build intuition before perturbations.

#### 3.1 Metrics

Define and justify each metric:

| Metric | What it captures |
|--------|-----------------|
| Bump center (population vector decode) | Where the memory is stored |
| Decoding amplitude | Bump sharpness / confidence |
| Bump width (circular SD) | Precision of spatial representation |
| MSD / drift | Memory diffusion over delay |
| Corrected asymmetry — mean\|A(t)\| | Spatial instability independent of direction; amplitude-normalized to avoid decay-induced bias |
| Distractor resistance/interference | Behavior under distractor presentation |
| Oscillation frequency (2–12 Hz) | Temporal dynamics; spatiotemporal structure of the bump |
| Oscillation power stability (freq. variance over delay) | Consistency of oscillatory regime during WM maintenance |

**Important methodological note on MSD**: because the bump oscillates spatially in some cases, the MSD has a transient oscillatory bias. A way to mitigate this is to compute the MSD at a lag much longer than the oscillation period (e.g. 500 ms lag for a ~7 Hz oscillation). This captures the long-term diffusion of the bump center while averaging out the short-term oscillatory fluctuations. We can also take the minimum MSD across lags to find the "best-case" diffusion, which is less biased by oscillations. But the fact that the bump oscillates at all is itself a key feature of the model and should be characterized separately (Section 3.2).

**Important note on width**: bump width is biased by amplitude — a stronger bump appears narrower via population vector decode. Always report width alongside amplitude, and use corrected asymmetry rather than raw width as the primary precision metric.

**Analysis to run**: For WT condition, characterize all metrics. Show: bump forms, oscillates over delay, has stable corrected asymmetry over delay.

**Figure**: Summary panel of WT bump — raster over delay, population vector trace, oscillation spectrogram, corrected asymmetry time course, MSD curve.

#### 3.2 Oscillatory behavior of the bump

The bump exhibits spatiotemporal oscillations during the delay period. The exact frequency depends on the inter-node weights and stimulus amplitude -> To be verified if this is really the case. This is a core feature of the model and must be characterized as a baseline before any between-condition comparison.

**Key observations to document**:
- The bump does not oscillate as a pure standing wave: neurons on one side fire slightly before the other. This is driven by noise sensitivity — a small noise-induced asymmetry in the bump creates a directional bias, which then gets amplified by the feedback dynamics. The corrected asymmetry metric is therefore not only a readout of spatial instability, but also a proxy for the network's noise sensitivity.
- Oscillation power is measurable via spectrogram of the population-vector amplitude signal.

**Analyses**:
- Asymmetry vs amplitude: Show assymetry, that is equal in both directions with multiple trials, and study the relationship between asymmetry and amplitude (slope of mean|A| vs amplitude) as a measure of noise sensitivity.
- Spectrogram of bump amplitude during delay: dominant frequency, bandwidth, entropy.
- Frequency stability metric: variance of instantaneous frequency over the delay window.
- Test effect of stimulus amplitude on oscillation frequency and power.

#### 3.3 Distractor and oscillatory alternation — WT baseline

This section establishes the baseline behavior when a distractor is presented during the delay. This is developed further in Section 4 (per KO) and Section 5 (APP). Because the merge/alternate dynamics are not well-documented in the ring attractor literature, this constitutes a novel observation that warrants a standalone characterization.

**Protocol**: Present a distractor at varying angular offsets from the cue (e.g. 30°, 90°, 120°, 170°) at a fixed time during the delay.

**Phenomena to document in WT**:
- **Merging regime** (close offsets): the distractor bump and the cue bump merge into a single wider, less peaked bump. The resulting bump's center is pulled toward the distractor location.
- **Alternation regime** (far offsets, ~170°): two separate bumps coexist and alternate in time — each fires during the oscillatory trough of the other. This is only possible because the bump has a sustained oscillatory character.
- **Transition** (intermediate offsets): merging takes longer; intermediate behavior.

**Phase alignment**:
- When the distractor is close to the cue, the oscillations of the two bumps synchronize (phase-lock).
- When the distractor is far (~180°), the oscillation power drops — destructive interference. But they'll still alternate in anti-phase, to avoid complete destructive interference. It is a non-trivial prediction that the network can maintain two separate bumps in anti-phase rather than merging or one dominating the other.
- Metric: cross-correlation of the amplitude timeseries at cue and distractor locations; extract phase lag as function of offset.

**Open question** (to investigate): Does the timing of the distractor's onset relative to the oscillation phase of the first bump have an influence on the differents metric (oscillation power, time to merge/alternate)? This requires sweeping the distractor onset time within one oscillation period at a fixed offset.

**Figure**: Raster at each angular offset; phase alignment curve; merge threshold characterization.

---

### Section 4 — Interneuron Dissection via Knockout Studies and firing rate/synaptic drive analysis

**Goal**: Understand the specific contribution of each interneuron class to bump formation by simulating receptor knockouts. This is the main mechanistic section.

> Rationale: KO = ablating a specific cholinergic drive to an interneuron class. This is biologically grounded (transgenic mice exist for each KO) and provides clean, interpretable perturbations.

#### 4.0 Baseline firing rates and synaptic drive — WT ring during working memory task

**Goal**: Before any KO, characterize the WT circuit during an active delay period: what are the firing rates of each population at bump nodes vs background nodes, and what is the synaptic drive balance during memory maintenance.

**Simulations to run**:
- Ring WT simulation (fitted params from Fit 1), standard cue + delay protocol.
- Record per-population firing rates and synaptic inputs, how they evolve over the delay, how they differ at bump nodes vs background, how they create the oscillatory regime. 

**Analyses**:
- Firing rates per population at bump vs background: how much does each interneuron class activate differentially at the bump location?
- Synaptic drive decomposition at bump nodes: excitatory (recurrent PYR→PYR) vs inhibitory (PV, SOM) vs disinhibitory (VIP→SOM) contributions over the delay.
- Temporal evolution: how do rates and drives evolve from cue offset to end of delay? Identify early vs late-delay regime.
- Cholinergic contribution: how much of each interneuron's firing at bump nodes comes from nAChR currents vs recurrent drive from the bump?

**Figure**: Rate profiles across the ring (bump shape) per population; time courses of bump-node rates during delay; stacked synaptic drive at bump node over time.

**Why first**: This establishes the working memory regime in WT and makes explicit which interneuron is doing what during the delay — grounding all subsequent KO comparisons.

---

#### 4.1 α7-KO (A7ko) — PV and SOM lose cholinergic excitation

**Biological prediction**: Reduced drive to PV and SOM → less feedback inhibition and less dendritic inhibition on PYR → PYR hyperactivation.

**Simulations to run**:
- A7ko vs WT: compare all 4 population firing rates.
- Ring: does the bump still form? How is width/drift affected?
- Does PYR rate increase compensate for reduced inhibitory drive?

**Analyses**:
- Firing rate changes (WT → A7ko): direction and magnitude per population.
- Bump amplitude and corrected asymmetry: expect higher spatial instability (less lateral inhibition).
- MSD / final-position distribution: expect higher drift.
- Oscillation frequency and power: PV drives fast inhibitory feedback → expect oscillation frequency change.
- Distractor resistance: expect lower (bump less robust).
- GABA scaling term g_alpha7 is also knocked out — quantify its contribution separately.

**Interpretation**: α7 nAChRs on PV/SOM are critical for bump precision and oscillatory stability. Loss → diffuse, spatially unstable representation with altered oscillatory dynamics.

#### 4.2 β2-KO (Beta2ko) — SOM loses cholinergic excitation

**Biological prediction**: Reduced drive to SOM only → less dendritic inhibition on PYR → PYR gain increases; VIP no longer inhibited as strongly → disinhibitory loop weakened.

**Simulations to run**:
- Beta2ko vs WT: population rates.
- Ring: bump metrics (width, drift, amplitude).

**Analyses**:
- Firing rate: SOM drops, PYR likely up.
- Bump dynamics: SOM adaptation timescale is ~2300 ms — comparable to the delay period. Loss of β2 drive to SOM may change late-delay stability more than early-delay.
- Oscillation: SOM provides slow adaptation; loss may affect oscillation damping over time rather than instantaneous frequency.
- Corrected asymmetry over time: check if asymmetry grows late in the delay (when SOM adaptation would normally be most active).
- Compare effects of Beta2ko vs A7ko: is the effect smaller and more delayed?

**Interpretation**: β2 nAChRs on SOM shape bump temporal stability through slow adaptation. Loss → gradual drift growth and late-delay spatial instability.

#### 4.3 α5-KO (Alpha5ko) — VIP loses cholinergic excitation

**Biological prediction**: Reduced drive to VIP → less disinhibition of PYR → SOM more active → PYR more inhibited → reduced amplitude.

**Simulations to run**:
- Alpha5ko vs WT: population rates.
- Ring: bump metrics, especially amplitude/decoding confidence.

**Analyses**:
- Firing rate: VIP drops, SOM increases (less inhibited by VIP), PYR decreases.
- Bump amplitude: expect lower (less disinhibition).
- Oscillation: lower amplitude → does oscillation power decrease? Is frequency preserved?
- Corrected asymmetry: a lower-amplitude bump is more vulnerable to noise-induced asymmetry, even if inhibitory damping is intact. Disentangle amplitude effect from inhibitory effect.
- Distractor resistance: weaker bump → may merge earlier or at closer distractor distances.

**Interpretation**: α5/VIP axis controls bump amplitude through disinhibitory gating. Loss → weakly expressed bump (correct location but faint, noisy oscillatory signal).

#### 4.4 Oscillatory analysis per KO

The oscillatory behavior of the bump is a direct readout of the inhibitory feedback structure of the circuit. Each KO affects a different part of that structure — the oscillation frequency and stability should therefore be differentially affected.

**Analyses**:
- Spectrogram per condition: dominant frequency, power, bandwidth.
- Frequency stability over delay: variance of instantaneous oscillation frequency.
- Comparison table: WT vs A7ko vs Beta2ko vs Alpha5ko for oscillation frequency and power stability.

**Expected findings**:
- A7ko: PV feedback reduced → expect shift in oscillation frequency (PV/PYR loop drives fast oscillations).
- Beta2ko: SOM adaptation reduced → expect less damping of oscillation over delay (late-delay oscillation stays strong rather than decaying).
- Alpha5ko: lower amplitude → oscillation power lower, but frequency possibly preserved.

**Figure**: Spectrogram panel per KO + oscillation frequency summary bar plot.

#### 4.5 Distractor and oscillatory alternation — per KO

Building on the WT baseline (Section 3.3), we now test how each KO modifies the merge/alternate regime and the phase alignment dynamics.

The merging/alternation behavior depends on three factors, each modulated by a different KO:
1. **Bump amplitude** — sets the inhibitory "territory" of a bump (Alpha5ko primarily affects this).
2. **Oscillation frequency and phase stability** — determines whether two bumps can lock in anti-phase (A7ko primarily affects this via PV feedback).
3. **Inter-node inhibition strength** — determines the merge threshold distance (A7ko and in part Beta2ko).

**Per-KO predictions**:
- A7ko: reduced PV-driven inter-node inhibition → merge threshold shifts to larger angular distances; oscillatory alternation regime narrowed or absent; phase alignment less stable.
- Beta2ko: effect on merge threshold expected to be mild and delayed (SOM adaptation is slow); may affect whether alternation persists late in the delay.
- Alpha5ko: weaker bump amplitude → the distractor bump dominates more easily (in WT, the distractor bump already has higher amplitude than the cue bump due to recency; in Alpha5ko this imbalance is amplified).

**Analyses per KO**:
- Raster at each angular offset: classify as merge vs alternate.
- Phase alignment curve: cross-correlation between cue and distractor amplitude timeseries; compare to WT baseline.
- Merge threshold: smallest offset at which alternation is observed rather than merging.

**Phase timing experiment** (novel, to develop):
- For a fixed offset (e.g. 170°, alternation regime in WT), sweep the distractor onset time within one oscillation period.
- Hypothesis: if the distractor arrives at the peak of the cue bump's oscillation (maximum inhibition), the distractor bump is suppressed and alternation is more stable. If it arrives at the trough (minimum inhibition), the distractor may merge or destabilize the cue bump.
- This experiment is not in the literature and would be a distinct contribution.

**Figure**: Raster panels per KO at each offset; phase alignment curves (WT vs each KO); phase timing experiment result for one representative KO.

#### 4.6 Comparison across KOs

- Side-by-side figure: all bump metrics for WT, A7ko, Beta2ko, Alpha5ko.
- Key message: each interneuron class contributes differently — PV/SOM (α7) control precision and oscillation frequency; SOM (β2) controls temporal stability and late-delay adaptation; VIP (α5) controls amplitude and oscillation power.
- Table: directional prediction matrix (↑/↓/~) for each metric × each KO, including oscillation frequency and distractor merge threshold.

---

### Section 5 — APP / β-Amyloid Condition: Disease Case Study

**Goal**: Translate the mechanistic understanding from KO studies to a graded, disease-relevant perturbation.

**Rationale**: In APP mice (early Alzheimer's model), the disease effect is represented by a dedicated WT_APP fitted local-circuit parameter family. In this workflow, APP is not applied by receptor desensitization sampling during ring simulations.

#### 5.0 Methodological concern: parameter space mismatch

> **Critical caveat**: WT and WT_APP do not operate in the same parameter space. The working regime (range of weights supporting bump formation) is shifted between WT and WT_APP. In vivo, the network would likely adjust its synaptic weights to the disease state. A naive comparison using the same parameters conflates disease-state circuit differences with the shift in operating regime.

**Approach**:
- Map the 2D working regime (excitatory × inhibitory weight) for both conditions to visualize the shift.
- Use the rate-matched comparison (Option C from Section 2.5) as the main comparison: each condition is run at the weight set that produces ~18Hz mean PYR during delay. This is the principled version.
- Report the naive same-parameter comparison as supplementary, to show what changes with and without the rate-matching correction.

#### 5.1 WT_APP vs WT

- **Already fitted**: model is optimized on these conditions.
- Show population rates match experimental data (validation figure).
- Characterize all bump metrics in WT_APP vs WT (raw comparison, same parameter set).
- Run oscillatory analysis: expect higher oscillation frequency and power in WT_APP (due to higher PYR rate).
- Run corrected asymmetry: expect higher mean|A(t)| and steeper growth with stimulus amplitude in WT_APP.
- Run distractor analysis (Section 3.3 protocol): characterize whether the merge/alternate regime and phase alignment change in WT_APP vs WT. The impairment of phase alignment in APP would suggest reduced oscillatory coordination capacity as a disease marker.

**Prediction from KO analysis**: compare KO effects within each parameter family first, then quantify the additional WT→WT_APP shift under each KO background (KO vs KO_APP).

#### 5.2 APP on KO backgrounds (8 conditions)

Run all 8 conditions (WT, WT_APP, a7KO, a7KO_APP, b2KO, b2KO_APP, a5KO, a5KO_APP) and extract bump metrics, oscillation metrics, and distractor metrics for each.

**Key question**: Does the WT_APP family further degrade WM in KO animals? Is there an additive or buffering interaction?
- a7KO_APP vs a7KO: additional disease-family shift with α7 already removed.
- b2KO_APP vs b2KO: additional disease-family shift with β2 removed.
- a5KO_APP vs a5KO: additional disease-family shift with α5 removed.

**Oscillatory prediction**: quantify which KO background most amplifies WT→WT_APP oscillation changes (frequency, power, stability).

**Interpretation**: Tests subtype-specific robustness of disease-state network changes and identifies which KO backgrounds are most sensitive to WT→WT_APP shifts.

#### 5.3 Rate changes and noise sensitivity as a function of receptor-sensitivity sweeps (optional)

- Parametric sweep: vary act_alpha7 from 0 → 1 (keeping others at WT). Then same for act_beta2 and act_alpha5.
- For each point: extract bump metrics, oscillation frequency, corrected asymmetry, and noise floor ratio.
- **Figure**: Phase diagram of WM quality (corrected asymmetry, oscillation stability) as function of α7 and α5 activation level (2D heatmap).
- Key insight: are there receptor-sensitivity thresholds where bump oscillatory stability collapses, and do these thresholds align with observed WT vs WT_APP differences?

---

### Section 6 — Discussion

**Goal**: Synthesize results and contextualize.

**Points to address**:

1. **Novelty of multi-interneuron ring attractor**:
   - Prior models: single inhibitory population → cannot distinguish PV vs SOM vs VIP contributions.
   - Our model: each interneuron class has a distinct mechanistic role in bump formation (precision, temporal stability, amplitude).

2. **Relationship to experimental data**:
   - KO predictions: do they match existing experimental observations (from literature, and from the calcium imaging data)?
   - WT_APP prediction: model correctly predicts increased PYR rate in APP condition.

3. **Clinical relevance**:
   - Early Alzheimer's: disease-state local-circuit changes primarily degrade oscillatory stability and bump precision.
   - KO-background comparisons (KO vs KO_APP) identify which receptor pathways are most sensitive to the disease-state shift.
   - The relative robustness of β2/SOM pathways can be tested directly via b2_KO vs b2_KO_APP metrics.
   - Implication: working memory errors in early AD may be primarily errors of spatial imprecision and oscillatory instability (the memory trace exists but is spatially noisy), not complete trace loss.
   - The pre-cue network state influences the delay-period trace (A7ko_APP pre-cue → delay correlation): disease may impair the network's ability to reset upon stimulus onset.

4. **Oscillatory dynamics as a functional readout**:
   - The bump's spatial oscillation is not a spurious artifact — it reflects the PV-PYR feedback loop and may correspond to theta/gamma rhythms in PFC.
   - Distractor-induced phase alignment (synchronization of two bumps) is a novel computational phenomenon that may have implications for attentional gating and interference.
   - APP condition impairs phase alignment → may correspond to reduced attentional suppression of distractors in early AD.

5. **Limitations**:
   - Rate model (not spiking) — oscillation frequencies are approximate; spike timing and gamma-band dynamics require spiking models.
   - 1D ring (spatial WM) — may not generalize to non-spatial WM.
   - Spike rate data derived from calcium imaging via event detection; conversion assumptions.
   - KO is an extreme perturbation; in reality, receptor dysfunction is graded.
   - WT and WT_APP do not operate in the same parameter space — some observed differences may reflect operating regime shifts rather than direct nAChR effects.

6. **Future directions**:
   - Spiking network implementation for precise oscillation frequency and spike timing analysis.
   - 3-month and 6-month APP timepoints (progressive disease-state trajectories).
   - Therapeutic predictions: α7-positive allosteric modulators to rescue PV/SOM drive and restore oscillatory stability.
   - Timing of distractor presentation relative to oscillation phase as a predictor of merge vs alternate outcome.

---

## Methods Section Checklist

- [ ] Full parameter table (CircuitParams with all defaults and search bounds)
- [ ] Transfer function mathematical derivation
- [ ] Ring connectivity equations (Gaussian weight profile, N=128 nodes)
- [ ] Noise model (white noise; note: OU noise does not reproduce cross-condition differences)
- [ ] Working memory protocol (timing diagram: 500ms baseline → 500ms cue → 3000ms delay → 500ms post)
- [ ] Bump metric definitions: population vector decode, corrected asymmetry (amplitude-normalized), MSD (τ >> oscillation period), final-position distribution
- [ ] Oscillation metrics: dominant frequency, instantaneous frequency variance, spectral power
- [ ] Distractor protocol: offset angles, phase alignment metric (cross-correlation of amplitude timeseries)
- [ ] Optimization procedure (Nevergrad, loss function, number of iterations)
- [ ] Experimental data: spike rates from calcium imaging (event detection → Hz; `AD_data/AD_spikes/`) → optimization targets (`per_neuron_mean`)
- [ ] Statistical tests for between-condition comparisons
- [ ] Parameter strategy — Level 1 (local circuit): re-optimize on KO data vs. zero activation in WT params
- [ ] Parameter strategy — Level 2 (ring connectivity): WT network for all conditions vs. per-condition rate-matched fit (~18Hz PYR during delay)
- [ ] Distractor protocol: offset angles, phase alignment metric, phase timing sweep
- [ ] Note on parameter space mismatch (WT vs WT_APP) and justification for rate-matched approach

---

## Figures Roadmap

| Figure | Content | Status |
|--------|---------|--------|
| Fig 1 | Circuit diagram (4 populations + nAChR labels) + Ring attractor schematic | To do |
| Fig 2 | Parameter fit validation: simulated vs experimental rates (WT, WT_APP, all 8 conditions) | Partial (figs/data/) |
| Fig 3 | Baseline WT bump: raster, decode trace, corrected asymmetry, oscillation spectrogram, MSD | Partial (figs/ring/run/) |
| Fig 4 | WT distractor baseline: raster at 4 angular offsets, phase alignment curve, merge/alternate regimes | To do |
| Fig 5 | KO effects on population firing rates (all 4 populations × 4 conditions) | To do |
| Fig 6 | KO bump metrics: amplitude, corrected asymmetry, MSD, final-position distribution per KO | To do |
| Fig 7 | KO oscillatory analysis: spectrogram per KO, frequency summary, power stability | To do |
| Fig 8 | KO distractor: raster at 4 offsets per KO, phase alignment curves vs WT, merge threshold; phase timing experiment | To do |
| Fig 9 | KO comparison summary: directional matrix + key metric bar plots (WT vs 3 KOs) | To do |
| Fig 10 | APP condition — all 8 conditions: population rates + bump quality metrics (rate-matched) | Partial (figs/optim/) |
| Fig 11 | APP oscillatory analysis: spectrogram comparison WT vs WT_APP; pre-cue → delay asymmetry correlation | To do |
| Fig 12 | APP distractor: phase alignment WT vs WT_APP; impairment of oscillatory coordination | To do |
| Fig 13 | Parametric receptor-sensitivity sweep: 2D heatmap (act_alpha7 × act_alpha5) → corrected asymmetry / oscillation stability | To do |
| Suppl. | Parameter space working regime (2D weight sweep); rate-matched vs same-parameter comparison | To do |

---

## Analysis Pipeline (Step-by-Step)

### Step 1 — Finalize parameter optimization
- [x] Fit model to 1mo WT and WT_APP data.
- [ ] Validate fit on all 8 KO×APP conditions (forward prediction, not refit).
- [ ] Run `study` command and compare simulated rates to experimental targets for all 8 conditions.

### Step 2 — Characterize WT bump and oscillation baseline
- [ ] Run `ring-run` for WT; generate all baseline metrics.
- [ ] Run `ring-oscillation-study` for WT: dominant frequency, power, instantaneous frequency variance over delay.
- [ ] Run `ring-noise-floor` to characterize spontaneous drift.
- [ ] Run `ring-asymmetry` for WT: corrected asymmetry (mean|A|, std).
- [ ] Run `ring-asymmetry-amp-sweep` for WT: asymmetry vs cue amplitude (establish slope baseline).

### Step 3 — KO ring simulations
- [ ] Run `ring-study` for WT, A7ko, Beta2ko, Alpha5ko.
- [ ] Extract per condition: amplitude, corrected asymmetry (mean|A|, std), MSD, final-position distribution.
- [ ] Run `ring-oscillation-study` per KO: spectrogram, frequency, power stability.
- [ ] Run `ring-distractor` per KO at 4 angular offsets: merge threshold, phase alignment metric.

### Step 4 — APP ring simulations (all 8 conditions)
- [ ] Run `ring-study` for all 8 conditions (WT, WT_APP, 3×KO, 3×KO_APP).
- [ ] Extract bump metrics, oscillation metrics, distractor metrics per condition.
- [ ] Run `ring-asymmetry-amp-sweep` for WT_APP: compare slope to WT.
- [ ] Run `ring-oscillation-study` for WT_APP: check higher power and frequency variance vs WT.
- [ ] Run `ring-distractor` for WT_APP: verify reduced phase alignment vs WT (preliminary result).

### Step 5 — Parameter space analysis
- [ ] Map 2D working regime (excitatory × inhibitory weight) for WT and WT_APP.
- [ ] Quantify size difference of bump-formation region between conditions.
- [ ] Run rate-matched comparison: find weight sets for WT and WT_APP with equal mean PYR delay-period rate; compare bump metrics.

### Step 6 — Parametric receptor-sensitivity study
- [ ] Sweep act_alpha7 (0→1, 10 steps) and act_alpha5 (0→1, 10 steps) independently.
- [ ] For each: extract corrected asymmetry, oscillation stability, noise floor ratio.
- [ ] Plot 2D heatmap: (act_alpha7, act_alpha5) → key WM quality metric.
- [ ] Identify threshold(s) below which oscillatory stability collapses.

### Step 7 — Figures & statistics
- [ ] Finalize all figures (see table above).
- [ ] Statistical comparison: bootstrap CI or permutation test across noise seeds.
- [ ] Write figure captions.

---

## Key Open Questions (to answer during analysis)

**Bump & precision**
1. Does A7ko produce a qualitatively different bump (e.g., bimodal? collapsed?) or just a wider/noisier one?
2. What is the relative contribution of PV vs SOM to α7 effects? (Can we disentangle by selectively turning off α7 drive to only PV or only SOM, keeping the other intact?)
3. Is there a compensation mechanism? When PYR rate rises (KO), does bump amplitude or width compensate, or does it simply scale?

**Oscillations**
4. Does oscillation frequency shift under A7ko? Does it match the prediction from reduced PV feedback (faster or slower)?
5. Does Beta2ko affect late-delay oscillation damping more than early-delay? (SOM adaptation timescale ~2300ms)
6. Is the oscillation in our model related to theta or gamma rhythms in the PFC literature? Can we interpret the ~7 Hz oscillation as a theta-like rhythm driven by the interneuron feedback loop?
7. Is there a critical α7 sensitivity level below which oscillatory stability collapses irreversibly?

**Distractor / phase dynamics**
8. Does the timing of the distractor presentation (relative to the oscillation phase of the first bump) predict whether they merge or alternate? This requires a controlled experiment sweeping distractor onset within one oscillation period.
9. Is the phase-alignment between close bumps a functionally useful mechanism (attentional binding) or a pathological feature (loss of selectivity)?
10. Does the loss of phase alignment in APP condition correspond to a measurable behavioral prediction (reduced ability to suppress distractors)?

**Persistence and rate-matched comparison**
11. Is the bump self-sustained (persistent) or decaying in WT and WT_APP? Does the rate-matched approach (18Hz PYR constraint) change this qualitative difference between conditions?
12. When comparing conditions with rate-matched parameters (each condition tuned to ~18Hz PYR during delay), do the bump quality differences persist or disappear? This is the key test of whether the inhibitory structure change (nAChR loss) matters independently of overall excitability.

**Parameter space**
13. Does the shift in the working regime between WT and WT_APP argue that in vivo compensation (weight adjustment) is necessary for the network to function in the disease state? Can we estimate the magnitude of this compensation from the rate-matching procedure?

---

## Timeline Estimate

| Phase | Content |
|-------|---------|
| Phase 1 | Finalize optimization + validate on all conditions |
| Phase 2 | Baseline WT bump characterization (Section 3) |
| Phase 3 | KO ring simulations (Section 4) |
| Phase 4 | APP + parametric study (Section 5) |
| Phase 5 | All figures + writing |
