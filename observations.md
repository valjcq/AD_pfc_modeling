
## Observations & Debugging Sessions

### Bump Attractor Behavior - Initial Issues

**Problem**: Bump decreasing in amplitude over delay rather than drifting; distractors creating double-bumps instead of shifting.

**Root causes identified**:
- Amplitude decay → recurrent excitation too weak; increased `w_pyr_pyr_inter`  (Increased PYR→PYR weights to strengthen recurrent excitation but issue persists)
- Double bump with distractor → uniform inhibition insufficient for winner-take-all ? (increased the inhibition weights, but issue persists)
- No spontaneous drift → bump pinned to grid despite 128-512 node tests (added noise, but issue persists)
- Gaussian connectivity spread (10°) too narrow; increased to 30°

**Current status**: Bump still decreases in amplitude with distractors rather than shifting. The multiple bump issue is also still present, suggesting further tuning of inhibition and connectivity is needed.

---

### Noise Type & Parameter Sensitivity

- OU noise doesn't reproduce article's cross-condition differences; white noise performs better
- Code params weaker than article; increased PYR current raises all population firing rates
- Fitted parameters don't reproduce article box plots, even without VIP→VIP and PV→SOM connections
- Bump stability improves with higher weights and stimulus amplitude

---

### Rate Model vs State Model

**Key insight**: Most literature uses firing rate models, not state-switching models. Our network operates near bifurcation (UP/DOWN oscillations), suggesting it should be fit on frequency data rather than transient dynamics.

**Implication**: Current fitting approach may not suit working memory tasks if stimuli require monostable (not oscillatory) response.

---

### Connectivity Matrix (Compte et al.)

Their definition allows negative PYR→PYR input (biologically implausible). We use Gaussian matrix instead; verified this matches their 30° spread parameter.

---

### Oscillation of the Bump

**Observation**: Bump oscillates at ~10 Hz in our model, which is not reported in the article. This might be a feature due to our models' inhibitory feedback, creating oscillations rather than a bug. However, it affects MSD analysis, as oscillation creates bias (bump present, absent, then present again).

**Mitigation strategies**:
- Fit MSD only for τ >> oscillation period to remove bias
- Use alternative metrics less affected by oscillation (final position distribution)
- Account for amplitude: low at troughs, high at peaks, for accurate wandering distance
- Try faded stimulus onset/offset to reduce oscillation

---

### Stimuli Presentation

**Hypothesis**: Square function stimuli may contribute to oscillation. Gradual onset/offset (Gaussian or ramped) could stabilize bump dynamics.

**Observation & Results**: Gaussian onset/offset actually reduces bump stability/formation, suggesting the network requires sudden input changes to initiate the bump and adapts too quickly to gradual changes. We'll use square pulse (default) for now.

---

### Proposed Metrics for Working Memory Precision

Instead of B_hat alone:
- **A**: MSD plateau level → direct wandering distance
- **B**: Final position distribution (Δφ at t=T_delay) → WM readout accuracy
- **C**: Cumulative path length → restlessness independent of direction
- **D**: Mean first exit time → stability threshold
- **E**: MSD fit restricted to τ >> oscillation period → removes transient bias

**Recommended approach**: Use B + A (final positions + plateau) for behavioral relevance.

---

### Decaying vs Bistable Bump

What actually characterizes a bump attractor in working memory? Is it a clearly defined two-state system (UP/DOWN) where the bump stays UP during delay then switches DOWN? Or is it a decaying bump with amplitude and stability affected by initial stimulus and noise?

This distinction matters for model fitting and evaluation:
- **Two-state system**: Focus on UP/DOWN bistability, transitions between states, and how conditions affect these transitions
- **Decaying bump**: Focus on amplitude and stability metrics, and how conditions affect them

Our model currently exhibits decaying bump behavior with oscillations, suggesting the latter interpretation is more appropriate. But it's important to explore the litterature to see if i should be aiming for a more bistable model instead.

---

### Noise floor without noise

Due to a miss in the burn-in of the noise, the noise floor experiment without noise was not correctly implemented. There was no noise during the burn-in, and probably the delay of the noise floor ratio ? Not sure
But what we observe is that the weight at which the network tend to have bigger noise floor value as been shifted to higher weight value. Does that mean that the noise participate to the network stability ? That a noise will help the network to be more stable without cue ? That can be interesting to explore, and it can be related to the fact that the network is in the edge of the change of state, and that the noise can help the network to be more stable in this state. It's probably du to the adaptative current that might be low in the absence of noise, and that can make the network more excitable and less stable.

---

### Asymmetry × Amplitude Sweep (`ring-asymmetry-amp-sweep`)

**Motivation**: The single-amplitude asymmetry experiment (at amp=45×) shows non-zero mean|A(t)| and std(A) in both WT and WT_APP. The question is whether these metrics grow with cue amplitude and whether the growth rate (slope) differs between conditions — if WT_APP has a steeper slope, it suggests the disease condition is more sensitive to stimulus drive in terms of spatial instability during the delay.

**Design**:
- Sweep amplitudes (e.g. 20–60× I_ext_pyr) while keeping all other parameters fixed.
- One 6000 ms shared burn-in per condition (from zero state, computed once per sweep run), then a 1000 ms per-trial secondary burn-in with a unique seed. This efficiently samples diverse pre-cue states without repeating the expensive long burn-in for each amplitude.
- Same simulation noise seed for a given trial index across all amplitudes → amplitude is the only variable that changes along that axis.
- Cache is shared with `ring-asymmetry`: per-amplitude `asymmetry_trials.csv` directories are identical in format, so the two commands are interoperable.

**Metrics**:
- `mean_abs_asym` = mean|A(t)| over the delay (after 400 ms transient skip) — captures spatial instability regardless of direction.
- `asym_std` = std(A(t)) over the delay — captures the amplitude of asymmetry fluctuations.

**Expected findings**:
- If both metrics increase with amplitude in a roughly linear fashion, the slope per condition can be extracted (OLS fit with R²) and compared.
- WT_APP should in principle show stronger asymmetry at a given amplitude if the loss of α7 nAChR reduces the inhibitory damping that keeps the bump symmetric.
- A steeper slope in WT_APP vs WT would indicate that the disease condition amplifies how strongly the stimulus drive converts into spatial instability — a possible mechanistic link between nAChR dysfunction and WM precision loss.

**Statistical outputs** (printed to console):
- OLS slope, intercept, R² per condition.
- Mann-Whitney U at each amplitude comparing conditions (to see where the difference becomes significant along the amplitude axis).


### Behavior with distractor

We analyze three angular positions of the distractor (30°, 90°, 120°, 170°) to see how the bump responds to distractors at different distances from the cue.

The closer bump (30°) merge in oscillations with the main bump, the resulting bump is wider and less "peaked".

When the distractor is at 90°, it does the same but the merging of oscillations takes more time, and the resulting bump is less strong but still less "peaked" than the main bump.

In both case, the distractor bump has higher amplitude than the main bump, which can be related to the fact that the distractor is more recent than the cue. But we would think that the active inhibition from the main bump should make the distractor bump less strong, but it is not the case. It can be related to the oscillation of the bump, the actual inhibition over the network is varying over time. If its the case, the distractor relative strength should be related to the phase of the oscillation at which it is presented, and that can be interesting to explore.

With 120°, we have the same observations.

With 170°, the distractor is far enough to not merge with the main bump, it does create a second bump, and the oscillations interfer in the way that they get wider, and they alternate. The first bump fire during the trough of the second bump, and the second bump fire during the trough of the first bump.

We need to analyze more in depth the impact of the distractor on the bump stability, and how it differ accross conditions. We need to developp metrics and visualization to analyze the impact on oscillations, and also experiment to test the impact of the timing of the distractor presentation, and the phase of the oscillation at which it is presented.

---

### Noise Floor with Stronger Inhibition (preliminary — sigma_pyr_deg=30, same params WT/WT_APP)

> **Context**: These results were produced with sigma_pyr_deg=30, using the same parameter set for WT and WT_APP. APP was not fitted to biological data — it approximates the desensitization but is not ground-truth. Methods and qualitative trends are informative; specific thresholds and magnitudes may shift with a proper fit.

**Observation (inhibition weight 4 vs 10)**:
- With lower inhibition weight (~4), disease/KO conditions shift the noise floor (maximum amplitude of a noise-induced bump) — the network enters a noise-sensitive state.
- With stronger inhibition weight (~10), the noise floor remains around the same value. The previously observed shift requires a higher weight to appear.
- Summary: the network shifts to a noise-sensitive state with lower inter-node inhibition weight in disease conditions. This makes the network less resistant to noise. In vivo, the network likely adjusts its weights to counterbalance this effect.

---

### 2D Parameter Space: WT vs WT_APP (preliminary)

**Heatmap observation**:
- The working state (region of parameter space allowing bump formation) is substantially shifted in WT_APP relative to WT.
- This raises an important methodological concern: the two conditions do not operate in the same parameter space. Differences in bump metrics could reflect this shift rather than a direct effect of nAChR desensitization.
- **Biological interpretation**: in vivo, the network would adjust its excitatory weights downward to compensate for the hyperactivity in APP condition.

**Proposed approaches to address this**:
1. Estimate the size of the bump-attractor parameter space for each condition (region that supports bump formation) — a smaller region in APP would itself be a meaningful finding.
2. Compare conditions matched on a shared physiological constraint (e.g. same average PYR firing rate during delay), using different weights per condition. This provides a fairer comparison of bump quality independent of the parameter shift.

---

### Bump Metrics: WT vs WT_APP (preliminary — same params)

**Key observations (weight 8.0–8.5 range)**:
- Bump metrics are broadly similar across conditions at matched weight, but WT_APP shows larger error from cue location.
- WT condition shows higher bump amplitude even when it sits in the lower part of the 2D parameter space (less suitable regime).
- PYR firing rate during delay: ~18 Hz (WT_APP, amp=20×) vs ~11 Hz (WT) — APP drives stronger pyramidal activity.
- Bump width is not trivially comparable when firing rates differ between conditions (width metric is biased by amplitude; see corrected asymmetry section).

---

### Oscillatory Behavior of the Bump (preliminary)

**General**:
- The bump exhibits clear oscillatory spatial displacement around ~7 Hz in both conditions (earlier runs suggested ~10 Hz; the exact frequency depends on weight and stimulus amplitude).
- In WT_APP: bump shifts spatially around the cue location during the delay. Oscillation power is higher, probably due to higher PYR firing rate and bump amplitude.
- The oscillation is not a clean spatial standing wave: neurons on one side of the bump fire before the other side — there is a spatial traveling wave component.
- At higher weight and stimulus amplitude, the bump reaches a near self-sustained state (no longer clearly decaying).

**Effect of APP on oscillation**:
- APP condition shows stronger oscillation power (higher frequency and amplitude with these settings).
- Oscillation in APP condition is also more variable across delay (higher variance of instantaneous frequency), but with higher concentrated spectral power.
- At higher inhibition weights, WT shows stricter variance but sharper frequency concentration.

**Phase alignment with distractor**:
- When the distractor is close to the cue location, the two oscillating bumps phase-align: their oscillations synchronize.
- When the two stimuli are at opposing locations (~180°), the oscillation power drops markedly — the distractor is maximally disruptive.
- In APP condition, this phase-alignment behavior is less well-defined, suggesting reduced capacity for oscillatory coordination between competing stimuli.

---

### Asymmetry Analysis: Corrected Metric and Pre-Cue Correlation (preliminary)

**Problem with raw asymmetry**:
- A decaying bump in WT tends to produce asymmetry at the end of the delay period, inflating the asymmetry metric. This asymmetry is weaker in activity amplitude than the "real" asymmetry in WT_APP.
- **Corrected asymmetry**: weight the asymmetry by the bump amplitude → normalizes by sum of amplitudes to keep the metric comparable. Use mean|A(t)| (not mean A(t)) to avoid cancellation when the direction of asymmetry switches during the delay. Also track std(A) for a sense of side-switching variability.

**Findings with corrected metric**:
- Asymmetry is unbiased with respect to left/right side (confirmed after cue placement correction).
- Pre-cue variance is larger in disease conditions — suggests the network is more sensitive to noise before the cue, since the bump activity during delay normally suppresses noise-induced asymmetry.
- WT_APP shows higher amplitude and variance of asymmetry than WT: consistent with the overall higher noise sensitivity.

**Pre-cue → delay correlation**:
- In α7-KO APP specifically: strong correlation between pre-cue asymmetry and delay-period asymmetry. This does not appear clearly in other conditions.
- Interpretation: in α7-KO APP, the network's pre-existing spatial noise state propagates into the memory trace — the network lacks the inhibitory stabilization to reset upon cue presentation. Further investigation needed.

**Asymmetry vs amplitude sweep**:
- Both mean|A(t)| and std(A) increase with cue amplitude. The rate of increase (slope) is faster in WT_APP — the disease condition converts stimulus drive into spatial instability more efficiently.
- Mechanistic link: loss of α7 nAChR reduces inhibitory damping that normally keeps the bump symmetric, so stronger drive amplifies spatial instability more in APP condition.