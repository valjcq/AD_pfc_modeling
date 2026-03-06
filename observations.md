
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