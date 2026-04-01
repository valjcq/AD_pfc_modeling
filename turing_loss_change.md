# Claude Code Task: Fix Turing Instability Penalty

## Goal

Replace the current Turing instability penalty in the ring optimizer with a corrected version that accounts for the full 4-population local circuit, while remaining fully analytical (no additional simulation cost).

---

## Background — Why the Current Criterion Is Wrong

The current penalty uses a 1-population Turing criterion:

$$\Phi'_\text{PYR}(I^*_\text{PYR}) \cdot w^\text{inter}_\text{pyr} \gtrless 1$$

This ignores two things that shape the effective PYR gain in the network:

1. The **divisive PV inhibition** in the PYR input equation, which reduces effective PYR gain.
2. The **global PV→PYR inter-node inhibition**, which opposes the PYR→PYR spatial excitation.

Both are analytically tractable and must be included.

SOM and VIP are local-only populations — they do not add new terms to the spatial Jacobian. However they do influence the operating point $I^*_\text{PYR}$ at the homogeneous fixed point, and therefore $\Phi'_\text{PYR}$ evaluated there. They are already correctly accounted for through the fixed-point computation, which uses the full 4-population circuit equations. No additional terms are needed for them.

---

## The Corrected Criterion

### Step 1 — W&W Transfer Function Derivative

The W&W transfer function for population $x$ is:

$$\Phi^x(I) = A_x \cdot \frac{u}{1 - e^{-g_x u}}, \qquad u = c_x I - I_{0,x}$$

Its derivative with respect to $I$ is:

$${\Phi^x}'(I) = A_x \cdot c_x \cdot \frac{1 - e^{-g_x u}(1 + g_x u)}{(1 - e^{-g_x u})^2}$$

This must be implemented as a helper function `compute_transfer_derivative(I_star, population_class)` where `population_class` is `'E'` for PYR (uses $c_e, I_{0,e}, g_e$) or `'I'` for PV/SOM/VIP (uses $c_i, I_{0,i}, g_i$). The fixed W&W shape parameters must be read from wherever they are currently defined in the codebase — **do not hardcode new values**.

### Step 2 — Effective PYR Gain

Define the effective PYR gain at operating point $I^*$, accounting for the divisive PV feedback loop:

$$G_\text{eff}(I^*) = \frac{{\Phi'}_\text{PYR}(I^*_\text{PYR})}{1 + g_\text{GABA} \cdot w_{pe} \cdot {\Phi'}_\text{PV}(I^*_\text{PV}) \cdot w_{ep} \cdot {\Phi'}_\text{PYR}(I^*_\text{PYR})}$$

where:
- ${\Phi'}_X(I^*_X)$ is the W&W derivative for population $X$ evaluated at its fixed-point input current
- $w_{pe}$ is the PV→PYR divisive weight
- $w_{ep}$ is the PYR→PV weight
- $g_\text{GABA}$ is the GABA scaling factor for the current condition

### Step 3 — Corrected Turing Gain Product

The corrected Turing condition for the bump mode ($k=1$) is:

$$G_\text{eff}(I^*) \cdot \underbrace{\left(w^\text{inter}_\text{pyr} - w^\text{global}_\text{PV} \cdot {\Phi'}_\text{PV}(I^*_\text{PV})\right)}_{\text{net spatial drive}} \gtrless 1$$

The net spatial drive is the effective Mexican hat: PYR→PYR spatial excitation minus the global PV→PYR spatial inhibition weighted by PV's gain at the operating point.

---

## Implementation Instructions

### 1. Locate the Turing penalty computation

Find the function that computes the Turing penalty in the loss/optimization code. Identify where `Phi_prime * w_pyr_inter` is currently computed.

### 2. Add the derivative helper

Implement `compute_transfer_derivative(I_star, population_class)` using the formula in Step 1 above. Read W&W shape parameters from their existing definition in the codebase.

### 3. Replace the gain product

At **both** operating points (rest and cue), compute:
- ${\Phi'}_\text{PYR}$ and ${\Phi'}_\text{PV}$ using the helper
- $G_\text{eff}$ using the Step 2 formula
- The net spatial drive $w^\text{inter}_\text{pyr} - w^\text{global}_\text{PV} \cdot {\Phi'}_\text{PV}$
- The corrected gain product $G_\text{eff} \cdot \text{net\_spatial\_drive}$

Replace the existing gain product in the penalty formula with this corrected quantity.

### 4. Operating point recovery

- The **rest** operating point $I^*_\text{PYR}$ and $I^*_\text{PV}$ must be recovered from the full 4-population circuit equations at the fitted firing rates — exactly as the current implementation does for $I^*_\text{PYR}$. Verify that $I^*_\text{PV}$ is also being computed (not approximated).
- For the **cue** operating point, apply the `turing_cue_scale` factor to $I_0^\text{PYR}$ only, then re-evaluate the fixed-point currents — consistent with the current implementation.

### 5. Keep everything else unchanged

The penalty structure (two-sided soft hinge with margin $m$) stays identical — only the quantity being compared to 1 changes. Do not modify penalty weights, margin parameter, or CLI arguments.

---

## What NOT to Modify

- The W&W shape parameter values ($c_e, c_i, I_{0,e}, I_{0,i}, g_e, g_i$) — fixed biological constants
- The penalty structure, hinge form, weight flags, or CLI arguments
- Any other loss terms (rate loss, KO loss, Jacobian regularization)

---

## Documentation Update

After implementing, update `ring_attractor.md` in the **Turing instability penalty** section (§10.3) as follows:

1. Replace the 1-population criterion formula with the corrected criterion (Steps 1–3 above).
2. Add the $G_\text{eff}$ definition and the W&W derivative formula.
3. Add a short paragraph explaining why SOM and VIP do not add spatial Jacobian terms but do influence the operating point through the full 4-population fixed-point computation.
4. Update the closing note (currently: *"Note that satisfying this penalty guarantees the correct geometry..."*) to reflect that the corrected criterion now properly accounts for PV feedback in both the divisive inhibition and the global spatial inhibition, making it a more accurate proxy for the true bistability condition of the 4-population ring.