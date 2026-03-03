
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