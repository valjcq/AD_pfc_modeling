# Observations

## 2026-04-17 — V4 bistable optimization: SOM silenced at high fixed point

**Result**: V4 found a genuinely bistable solution (3 crossings, 2 stable FPs) with the high FP at 57 Hz (target 60 Hz, −5%). All other interneurons (PV, VIP) match their targets well at both states. However, SOM is completely silent at the high fixed point (0 Hz actual vs. 35 Hz target, −100% error).

**Interpretation**: The network achieves bistability by silencing SOM. This is mechanistically plausible — SOM provides feedback inhibition that could return the network to its low state under small perturbations; when the network shifts to the high state, SOM gets suppressed, which removes this stabilizing inhibition and allows the high state to persist. In other words, SOM silence may be what *enables* bistability in this circuit configuration.

**Why this is a problem**: Experimental data shows SOM is *not* silent at the high state (target ≈ 35 Hz). The optimized solution is bistable but biologically inconsistent.

**Possible explanations / hypotheses**:
1. **Multi-node / lateral inhibition effect**: SOM may be important for suppressing *other* (non-activated) nodes when one node switches to the high state. At the single-node level, this role disappears and SOM gets silenced. Going back to the full ring network may restore SOM activity at the high FP.
2. **Missing population**: The circuit may be missing a regulatory population that controls switching between states and also keeps SOM active. Without it, the optimizer can only achieve bistability by silencing SOM.
3. **Parameter bounds too tight**: If the current parameter space is over-constrained, the optimizer cannot reach regions where SOM remains active at the high state. Widening bounds (especially for weights onto/from SOM) could reveal such solutions.
4. **ROOY et al. parameter set**: The ROOY article reportedly found bistable parameters reproducing the targets (different ODE formulation). Their parameter values for interneuron activity at the high state are not reported. A direct test of their parameters on the current model was inconclusive — worth revisiting with the correct model formulation.

**Suggested next steps**: see chat discussion.
