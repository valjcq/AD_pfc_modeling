# Loss Function: Mathematical Deep Dive

> **Scope.** This document traces every mathematical step inside `bistable_loss()`,
> from the circuit equations through the nullcline sweep to each penalty term.
> It is meant to be read sequentially; each section builds on the previous one.

---

## Table of Contents

1. [The Circuit Model](#1-the-circuit-model)
2. [The Transfer Function $\Phi$](#2-the-transfer-function-phi)
3. [Steady-State Reduction](#3-steady-state-reduction)
4. [NMDA Gating Variable](#4-nmda-gating-variable)
5. [Self-Consistent Interneuron Solve](#5-self-consistent-interneuron-solve)
6. [The PYR Nullcline $F(r)$](#6-the-pyr-nullcline-fr)
7. [Fixed-Point Detection and Stability](#7-fixed-point-detection-and-stability)
8. [Loss Term A — $L_\text{bistab}$](#8-loss-term-a--l_bistab)
9. [Loss Term B — $L_\text{rate}$](#9-loss-term-b--l_rate)
10. [Loss Term C — $L_\text{rate,high}$](#10-loss-term-c--l_ratehigh)
11. [Loss Term D — $L_\text{margin}$](#11-loss-term-d--l_margin)
12. [Loss Term E — $L_\text{jac}$](#12-loss-term-e--l_jac)
13. [Loss Term F — $L_\text{peak}$](#13-loss-term-f--l_peak)
14. [Total Loss and Gradient Flow](#14-total-loss-and-gradient-flow)

---

## 1. The Circuit Model

The model has four neural populations:

| Symbol | Population | Type |
|---|---|---|
| PYR | Pyramidal cells | Excitatory |
| PV | Parvalbumin interneurons | Inhibitory (perisomatic, fast) |
| SOM | Somatostatin interneurons | Inhibitory (dendritic, slow) |
| VIP | VIP interneurons | Inhibitory (disinhibitory) |

Each population $i$ has a membrane time constant $\tau_s$ and obeys:

$$\tau_s \, \dot{r}_i = -r_i + \Phi_i(I_i^\text{net})$$

where $r_i$ is the firing rate (Hz) and $I_i^\text{net}$ is the total synaptic input current (nA).

The synaptic inputs are (using $g_\text{GABA} = g_\text{GABA,base} + g_{\alpha7}$ as a GABA gain factor):

$$I_\text{PYR}^\text{net} = \frac{J_\text{NMDA} \cdot S^*}{1 + g_\text{GABA} \cdot w_{pe} \cdot r_\text{PV}} - g_\text{GABA} \cdot w_{se} \cdot r_\text{SOM} - J_\text{adapt,PYR} \cdot r_\text{PYR} + I_0^\text{PYR}$$

$$I_\text{SOM}^\text{net} = w_{es} \cdot r_\text{PYR} - w_{vs} \cdot r_\text{VIP} - J_\text{adapt,SOM} \cdot r_\text{SOM} + I_0^\text{SOM}$$

$$I_\text{PV}^\text{net} = w_{ep} \cdot r_\text{PYR} - g_\text{GABA} \cdot w_{pp} \cdot r_\text{PV} - g_\text{GABA} \cdot w_{sp} \cdot r_\text{SOM} - w_{vp} \cdot r_\text{VIP} + I_0^\text{PV}$$

$$I_\text{VIP}^\text{net} = w_{ev} \cdot r_\text{PYR} + I_0^\text{VIP}$$

Three features deserve attention:

**PV inhibition onto PYR is divisive (shunting), not subtractive.** The PV term appears in the *denominator* of the PYR input, not additively. This models perisomatic inhibition that scales the gain of all excitatory inputs, rather than simply subtracting a fixed amount. Mathematically, when $r_\text{PV}$ doubles, it reduces $I_\text{PYR}^\text{net}$ proportionally, which compresses the PYR gain. This is physiologically distinct from SOM inhibition, which is additive (dendritic).

**SOM inhibition onto PYR is subtractive.** It enters $I_\text{PYR}^\text{net}$ with a straight minus sign, shifting the PYR response curve downward uniformly.

**Adaptation currents** $J_\text{adapt} \cdot r_i$ act as negative feedback: the faster a population fires, the more current it subtracts from its own input, providing slow self-regulation.

Weight notation: $w_{XY}$ means the connection **from** population $Y$ **to** population $X$ (e.g., $w_{se}$ = SOM→PYR).

---

## 2. The Transfer Function $\Phi$

Each population converts its net input current $I$ into a firing rate via the Wong-Wang (2006) transfer function:

$$\Phi(I) = \max\!\left(0,\; \frac{u}{1 - e^{-g \cdot u}}\right), \qquad u = c \cdot (I - \theta)$$

Parameters:

| Parameter | PYR | PV / SOM / VIP |
|---|---|---|
| $c$ (gain, Hz/nA) | 310 | 615 |
| $\theta$ (threshold, nA) | $125/310 \approx 0.403$ | $177/615 \approx 0.288$ |
| $g$ (curvature, s) | 0.16 | 0.087 |

These six constants are **fixed from Wong & Wang (2006)** — they are not optimised.

**What does this function do?**

- Below threshold ($I < \theta$): $u < 0$, output clamped to 0 by $\max(\cdot, 0)$ — silent population.
- Near threshold ($u \approx 0$): Taylor expansion gives $\Phi \approx 1/g + u/2$, approximately linear with slope $c/2$.
- Far above threshold ($u \gg 0$): $e^{-gu} \to 0$, so $\Phi \approx u$, growing linearly with $I$.
- The function is monotonically increasing and continuously differentiable everywhere.

**Derivative** (needed for the Jacobian in Section 12):

$$\Phi'(I) = \frac{d\Phi}{dI} = c \cdot \frac{1 - e^{-z}(1 + z)}{(1 - e^{-z})^2}, \qquad z = g \cdot u$$

At $z = 0$: $\Phi'(I) = c/2$ (Taylor limit). The derivative is always non-negative (monotone function).

---

## 3. Steady-State Reduction

The dynamics $\tau_s \dot{r}_i = -r_i + \Phi_i(I_i^\text{net})$ have fixed points where all $\dot{r}_i = 0$, i.e.:

$$r_i^* = \Phi_i(I_i^\text{net}(r_1^*, r_2^*, r_3^*, r_4^*))$$

This is a **4-dimensional nonlinear system**. Analysing its fixed points directly would require solving four coupled nonlinear equations simultaneously, which is expensive and gives no geometric intuition.

**Key simplification:** because VIP has no recurrent connections (it receives no feedback from SOM or PV), and SOM/PV are fast enough to track PYR on the timescale of interest, we can **reduce to a 1D problem** by treating $r_\text{PYR}$ as a free parameter and solving the other three populations self-consistently at each value of $r_\text{PYR}$.

This works because at any $r_\text{PYR}$:
- $r_\text{VIP}$ is determined by $r_\text{PYR}$ alone (no feedback loops into VIP)
- $(r_\text{SOM}, r_\text{PV})$ are determined by $r_\text{PYR}$ and $r_\text{VIP}$ via a 2D self-consistent solve

The result is a function $r_i^*(r_\text{PYR})$ for each interneuron — the value each population would take at steady state if PYR were held fixed. Plugging these back into the PYR equation gives the **1D PYR nullcline**, described in Section 6.

---

## 4. NMDA Gating Variable

PYR recurrent excitation is mediated by NMDA receptors with a gating variable $S$ that integrates spiking activity:

$$\tau_\text{NMDA} \, \dot{S} = -S + \gamma \cdot (1 - S) \cdot r_\text{PYR}$$

At steady state ($\dot{S} = 0$), solving for $S$:

$$S^* = \frac{\gamma \tau_\text{NMDA} \cdot r_\text{PYR}}{1 + \gamma \tau_\text{NMDA} \cdot r_\text{PYR}}$$

This is a saturable, monotonically increasing function of $r_\text{PYR}$ (a Michaelis-Menten-like formula). Parameters: $\gamma = 0.641$, $\tau_\text{NMDA} = 100$ ms.

**Why this matters for bistability:**  
At low $r_\text{PYR}$, $S^* \approx \gamma \tau_\text{NMDA} \cdot r_\text{PYR}$ (linear — NMDA current grows with rate).  
At high $r_\text{PYR}$, $S^* \to 1$ (saturates — NMDA current plateaus regardless of how hard PYR fires).

This saturation is what creates the **S-shaped nullcline** and allows two stable operating points. Without NMDA saturation, $\Phi_\text{PYR}(I_\text{net})$ would grow without bound and there would be at most one intersection with the identity line.

---

## 5. Self-Consistent Interneuron Solve

For each value of $r_\text{PYR}$ in the sweep:

**Step 1 — VIP (direct):**

$$r_\text{VIP}^* = \Phi_\text{VIP}(w_{ev} \cdot r_\text{PYR} + I_0^\text{VIP})$$

No iteration needed. VIP has no inputs from SOM or PV.

**Step 2 — SOM and PV (coupled system):**

Define the residuals:

$$g_1(r_\text{SOM}, r_\text{PV}) = \Phi_\text{SOM}(w_{es} \cdot r_\text{PYR} - w_{vs} \cdot r_\text{VIP}^* - J_\text{adapt,SOM} \cdot r_\text{SOM} + I_0^\text{SOM}) - r_\text{SOM}$$

$$g_2(r_\text{SOM}, r_\text{PV}) = \Phi_\text{PV}(w_{ep} \cdot r_\text{PYR} - g_\text{GABA} \cdot w_{pp} \cdot r_\text{PV} - g_\text{GABA} \cdot w_{sp} \cdot r_\text{SOM} - w_{vp} \cdot r_\text{VIP}^* + I_0^\text{PV}) - r_\text{PV}$$

We want $(r_\text{SOM}^*, r_\text{PV}^*)$ such that $g_1 = g_2 = 0$.

Note: $g_1$ depends on $r_\text{SOM}$ (adaptation self-term) but not $r_\text{PV}$.  
$g_2$ depends on both $r_\text{SOM}$ and $r_\text{PV}$.  
So the coupling is one-directional in the input equations, but because $\Phi$ is nonlinear, the system cannot generally be solved sequentially.

**Newton's method (`fsolve`):**

Starting from an initial guess $\mathbf{x}^{(0)}$, iterate:

$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \left[\mathbf{J}_g(\mathbf{x}^{(k)})\right]^{-1} \mathbf{g}(\mathbf{x}^{(k)})$$

where $\mathbf{J}_g$ is the $2 \times 2$ Jacobian of the residual system:

$$\mathbf{J}_g = \begin{pmatrix} \partial g_1 / \partial r_\text{SOM} & \partial g_1 / \partial r_\text{PV} \\ \partial g_2 / \partial r_\text{SOM} & \partial g_2 / \partial r_\text{PV} \end{pmatrix}$$

Each entry is $\Phi'_i \cdot \partial I_i / \partial r_j - \delta_{ij}$ (the $-\delta_{ij}$ from the $-r_i$ residual term). Because $\Phi'$ is bounded and strictly positive above threshold, $\mathbf{J}_g$ is well-conditioned at any biologically plausible operating point.

The code tries two initial guesses — $(0, 0)$ and $(30, 30)$ Hz — and keeps whichever produces the smaller $|\mathbf{g}|$. This guards against Newton's method converging to a spurious local solution when the residual landscape is non-convex (possible when $\Phi_\text{SOM}$ or $\Phi_\text{PV}$ operates near threshold).

---

## 6. The PYR Nullcline $F(r)$

Having solved for the interneuron rates at each $r_\text{PYR}$, the PYR input current is:

$$I_\text{PYR}^\text{net}(r) = \frac{J_\text{NMDA} \cdot S^*(r)}{1 + g_\text{GABA} \cdot w_{pe} \cdot r_\text{PV}^*(r)} - g_\text{GABA} \cdot w_{se} \cdot r_\text{SOM}^*(r) - J_\text{adapt,PYR} \cdot r + I_0^\text{PYR}$$

where $r \equiv r_\text{PYR}$ and all starred quantities are the self-consistent solutions from Section 5.

The **PYR nullcline function** is:

$$F(r) = \Phi_\text{PYR}(I_\text{PYR}^\text{net}(r)) - r$$

$F(r) = 0$ is the condition that PYR is at its own steady state given that all interneurons are at *their* steady state for that value of $r$. Therefore:

> **Zeros of $F$ are full 4-population fixed points of the circuit.**

The nullcline is swept over $r \in [0, 80]$ Hz at 1000 equally-spaced points. This resolution (~0.08 Hz per step) is fine enough to detect crossings reliably, while remaining fast enough for repeated evaluation during optimisation.

**Why can $F$ have three zeros (bistability)?**

$F(r) = \Phi_\text{PYR}(\cdot) - r$ is the difference between the PYR transfer function output and the identity $r$. Graphically, it is the signed distance between the nullcline $\Phi_\text{PYR}(I_\text{net}(r))$ and the diagonal. If the nullcline has an S-shape (which the NMDA saturation produces), it can cross the diagonal three times: twice stably ($F' < 0$) and once unstably ($F' > 0$), giving the bistable configuration.

The shape of $F$:

```
 F(r)
  ▲
  │  ++++++.                           .+++++++
  │        `.                       .+'         `.
 ─┼──────────`────────────────────.'─────────────`────→ r
  │           ↑                  ↑               ↑
  │         low FP          unstable FP        high FP
  │         (stable)        (repeller)         (stable)
  │         F'(r*) < 0      F'(r*) > 0         F'(r*) < 0
```

---

## 7. Fixed-Point Detection and Stability

**Detecting crossings:**

Zero crossings of $F$ are found by locating where `sign(F)` changes between consecutive sweep points:

```python
sign_changes = np.where(np.diff(np.sign(F)))[0]
```

Each crossing index $k$ means $F[k]$ and $F[k+1]$ have opposite signs, so a zero lies in $(r[k], r[k+1])$.

**Brentq refinement:**

The discrete sweep has ~0.08 Hz resolution. For rate-matching losses, we need the fixed-point location to be accurate to better than that. The code refines each crossing via the Brentq algorithm: a guaranteed-converging bracketed root-finding method that combines bisection (safe) with secant-method acceleration (fast). It requires only that $F$ changes sign across the bracket, which the crossing detection already guarantees.

**Stability classification:**

The numerical derivative $dF/dr$ is computed via `np.gradient(F, r_sweep)` (central differences). At each refined crossing $r^*$:

$$F'(r^*) < 0 \implies \text{stable fixed point (attractor)}$$
$$F'(r^*) > 0 \implies \text{unstable fixed point (repeller / saddle)}$$

Intuitively: if $F'(r^*) < 0$, then a small perturbation $\delta r > 0$ gives $F(r^* + \delta r) < 0$, meaning PYR dynamics push $r$ back down — the system returns to $r^*$. The opposite holds for $F' > 0$.

**Crossings above `R_MAX_PHYS` are discarded** — they arise from numerical artefacts of clamping $\Phi$ at extreme firing rates and do not correspond to biologically meaningful states.

---

## 8. Loss Term A — $L_\text{bistab}$

This term enforces the bistable sign pattern $F(r): (+, -, +, -)$ without requiring fixed probe locations. It has four sub-penalties, all using $\text{relu}(x) = \max(0, x)$.

### 8.1 Adaptive Low Basin

First, detect the **actual** position of the low fixed point — the first downward zero crossing (where $F$ transitions from $+$ to $-$):

```python
down_cross_idx = np.where(np.diff(np.sign(F)) < 0)[0]
r_\text{low,actual} = r_\text{sweep}[down_cross_idx[0]]
```

Then enforce $F > 0$ throughout $[0, r_\text{low,actual}]$:

$$L_\text{low basin} = \text{relu}\!\left(-\max_{r \leq r_\text{low,actual}} F(r)\right) \times \underbrace{\left(1 + \frac{|r_\text{low,actual} - r_\text{low,target}|}{r_\text{low,target}}\right)}_{\text{displacement scale}}$$

The displacement scale is critical. If no low fixed point has been found yet (optimizer is far from bistability), $r_\text{low,actual}$ may be 40 or 70 Hz. The penalty then applies to the entire region $[0, 70]$ Hz, and the scale factor is $\approx 40\times$ larger than when the FP sits at its target. This means both `L_bistab` and `L_rate` simultaneously push the optimizer toward a valid low-FP position, rather than allowing it to find a "bistable" solution with an absurdly displaced low state.

### 8.2 Valley (unstable FP must exist)

After the low FP, $F$ must go negative somewhere — this creates the valley that implies an unstable fixed point separating the two basins:

$$L_\text{valley} = \text{relu}\!\left(\min_{r > r_\text{low,actual}} F(r)\right)$$

- If $F$ goes negative anywhere to the right of the low FP: $L_\text{valley} = 0$
- If $F$ stays positive everywhere: $L_\text{valley} > 0$ (monostable, no unstable separator)

### 8.3 High Basin (windowed)

$F$ must be *positively above* a margin $f_\text{margin} = 1$ Hz within the window $[0.7 \times r_\text{PYR}^\text{H}, 1.0 \times r_\text{PYR}^\text{H}]$ = $[42, 60]$ Hz (defaults `r_high_basin_lo_frac = 0.7`, `r_high_basin_hi_frac = 1.0`):

$$L_\text{high basin} = \text{relu}\!\left(f_\text{margin} - \max_{r \in [42, 60]} F(r)\right)$$

The margin $f_\text{margin} > 0$ prevents the optimizer from satisfying this condition by making $F$ barely non-negative (effectively at zero), which would place a very weak or numerically spurious crossing. The window constraint prevents a spurious bump at low rates (e.g., 17 Hz) from being counted as the high basin.

### 8.4 Tail (high FP must be stable)

At the far end of the sweep ($r \geq 0.85 \times r_\text{max} = 68$ Hz), $F$ must be negative — the high fixed point is stable, not a transient bump:

$$L_\text{tail} = \text{relu}\!\left(\max_{r \geq 68} F(r)\right)$$

If the nullcline keeps rising past 68 Hz and never crosses back down, the "high FP" is not actually a stable attractor — the network would keep accelerating beyond any reasonable rate. `L_tail > 0` penalises this.

### 8.5 Total

$$\boxed{L_\text{bistab} = L_\text{low basin} + L_\text{valley} + L_\text{high basin} + L_\text{tail}}$$

All four sub-terms are zero if and only if the full $(+, -, +, -)$ sign pattern holds with the low FP near target and the high FP within the physiological window.

---

## 9. Loss Term B — $L_\text{rate}$

Once the lowest stable fixed point $r_\text{PYR}^\text{low}$ is identified, the interneuron rates at that point are obtained by calling `_solve_interneurons(r_low_fp)` (the same Newton solver from Section 5).

The loss is the **Mean Squared Proportional Error** (MSPE) across all four populations:

$$L_\text{rate} = \left(\frac{r_\text{PYR}^\text{low} - r_\text{PYR}^\text{L,target}}{r_\text{PYR}^\text{L,target}}\right)^2 + \left(\frac{r_\text{SOM}^\text{low} - r_\text{SOM}^\text{L,target}}{r_\text{SOM}^\text{L,target}}\right)^2 + \left(\frac{r_\text{PV}^\text{low} - r_\text{PV}^\text{L,target}}{r_\text{PV}^\text{L,target}}\right)^2 + \left(\frac{r_\text{VIP}^\text{low} - r_\text{VIP}^\text{L,target}}{r_\text{VIP}^\text{L,target}}\right)^2$$

Default targets (project quiet-wakefulness fit, `BistableConfig`): PYR = 8.0, SOM = 5.0, PV = 3.0, VIP = 2.0 Hz.

**Why proportional (relative) errors?**  
The populations span an order of magnitude in the low state (PYR $\approx$ 8 Hz, VIP $\approx$ 2 Hz). Absolute squared errors would make a 1 Hz miss at VIP (50% error) equivalent to a 1 Hz miss at a hypothetical 50 Hz target (2% error), which is wrong biologically. Dividing by the target normalises each term so all populations contribute equally regardless of their absolute rate.

**Fallback when monostable:** If no stable crossing is found, the code falls back to the position of minimum $|F|$ in the $[0, 15]$ Hz window. This gives a sensible gradient even when the network has not yet found bistability.

---

## 10. Loss Term C — $L_\text{rate,high}$

Symmetric to $L_\text{rate}$, evaluated at the highest stable fixed point:

$$L_\text{rate,high} = \left(\frac{r_\text{PYR}^\text{high} - r_\text{PYR}^\text{H,target}}{r_\text{PYR}^\text{H,target}}\right)^2 + \left(\frac{r_\text{SOM}^\text{high} - r_\text{SOM}^\text{H,target}}{r_\text{SOM}^\text{H,target}}\right)^2 + \left(\frac{r_\text{PV}^\text{high} - r_\text{PV}^\text{H,target}}{r_\text{PV}^\text{H,target}}\right)^2 + \left(\frac{r_\text{VIP}^\text{high} - r_\text{VIP}^\text{H,target}}{r_\text{VIP}^\text{H,target}}\right)^2$$

Default targets (Rooy 2021 active state): PYR = 60.2, SOM = 35.2, PV = 35.3, VIP = 68.8 Hz.

**When monostable: $L_\text{rate,high} = 0$.**  
When no second stable FP exists, adding a penalty against phantom high-FP rates would produce a misleading gradient pointing toward an incoherent target. The bistability terms already supply a strong gradient in that regime.

**What this term prevents:** Without it, the optimizer finds a degenerate "VIP-disinhibitory" bistable solution where VIP fully silences SOM in the high state (SOM $\approx$ 0 Hz, PV $\approx$ 4 Hz), which is bistable but biologically wrong. Adding explicit pressure toward SOM $\approx$ 35 Hz and PV $\approx$ 35 Hz eliminates this mode.

---

## 11. Loss Term D — $L_\text{margin}$

The two stable fixed points must be well-separated to constitute functionally distinct memory states:

$$L_\text{margin} = \text{relu}(\Delta r_\text{min} - (r_\text{high}^\text{stable} - r_\text{low}^\text{stable})), \qquad \Delta r_\text{min} = 15 \text{ Hz}$$

**When monostable** (fewer than two stable FPs): a fixed penalty of $2 \times \Delta r_\text{min} = 30$ is applied. This is strictly larger than any achievable `relu` value when bistable (since the max `relu` value in the bistable case approaches 0 as separation grows), ensuring a clear gradient discontinuity that motivates finding the second FP.

**Why 15 Hz?** With $r_\text{low} \approx 8$ Hz and $r_\text{high} \approx 60$ Hz in the target solution, the actual separation is ~52 Hz — far above the 15 Hz threshold. The threshold is loose: it only penalises degenerate near-monostable solutions where two crossings exist but are so close together they would merge under any perturbation. It is not meant to constrain the actual separation tightly.

---

## 12. Loss Term E — $L_\text{jac}$

The 4×4 Jacobian is computed at the low fixed point $\mathbf{r}^* = (r_\text{PYR}^\text{low}, r_\text{SOM}^\text{low}, r_\text{PV}^\text{low}, r_\text{VIP}^\text{low})$.

**What the Jacobian measures:**

$$J_{ij} = \frac{\partial r_i}{\partial r_j}\bigg|_{\mathbf{r}^*}$$

This is the *effective gain*: "if population $j$ fires 1 Hz faster, how much does population $i$ change at the new steady state?" It is a linearisation of the full nonlinear circuit around the operating point.

**How it is computed:**

By the chain rule, treating adaptation as fixed at its steady-state value:

$$J_{ij} = \Phi'_i(I_i^\text{net}) \cdot \frac{\partial I_i^\text{net}}{\partial r_j}$$

where $\Phi'_i$ is the transfer function derivative from Section 2, and $\partial I_i^\text{net}/\partial r_j$ is the explicit synaptic weight (or NMDA gating derivative for $i = j = \text{PYR}$). Cross-population feedback loops (e.g., PYR → PV → PYR) are **not** iterated here — this is a first-order linearisation, not a full resolvent.

For example:

$$J_{\text{PYR,PYR}} = \Phi'_\text{PYR} \cdot \frac{J_\text{NMDA} \cdot dS^*/dr}{1 + g_\text{GABA} w_{pe} r_\text{PV}^*}$$

$$J_{\text{PYR,SOM}} = \Phi'_\text{PYR} \cdot (-g_\text{GABA} \cdot w_{se})$$

$$J_{\text{PYR,PV}} = \Phi'_\text{PYR} \cdot \left(\frac{-J_\text{NMDA} S^* \cdot g_\text{GABA} w_{pe}}{(1 + g_\text{GABA} w_{pe} r_\text{PV}^*)^2}\right)$$

Note the last entry: PV enters both as a divisive denominator (the $-J_\text{NMDA} S^* / \text{denom}^2$ term from differentiating $1/\text{denom}$ with respect to $r_\text{PV}$) and not additively — this is the mathematical signature of shunting inhibition.

**The penalty:**

$$L_\text{jac} = \text{relu}\!\left(\max_{i,j} |J_{ij}| - 5.0\right)^2$$

The threshold of 5 means: no single direct connection from population $j$ to population $i$ should have an effective gain above 5 Hz/Hz. Values above this indicate that one pathway overwhelmingly dominates the circuit (e.g., an enormous PV→PYR shunting gain), which tends to produce bistability through a degenerate single-pathway mechanism rather than the intended multi-pathway disinhibitory architecture.

The square on the relu makes the penalty grow quadratically with excess gain, creating a stronger restoring force far from the threshold.

---

## 13. Loss Term F — $L_\text{peak}$

The nullcline value at each $r$ is recovered as:

$$\Phi_\text{PYR}(I_\text{net}(r)) = F(r) + r$$

The peak of this curve:

$$L_\text{peak} = \text{relu}\!\left(\max_r \left[F(r) + r\right] - r_\text{peak,max}\right)^2, \qquad r_\text{peak,max} = 80 \text{ Hz (default)}$$

The default weight is $w_\text{peak} = 0$, so the term is inactive in the standard fit (the raw value is computed but not added to the total loss). It is included for experimental use: setting `--w_peak 1.0 --nullcline_peak_max 95` constrains the nullcline peak to $\leq 95$ Hz, which prevents the network from transiently overshooting far past the high FP during the L → H transition triggered by a cue stimulus.

---

## 14. Total Loss and Gradient Flow

$$\boxed{L_\text{total} = w_\text{bistab} \cdot L_\text{bistab} + w_\text{rate} \cdot L_\text{rate} + w_\text{rate,high} \cdot L_\text{rate,high} + w_\text{margin} \cdot L_\text{margin} + w_\text{jac} \cdot L_\text{jac} + w_\text{peak} \cdot L_\text{peak}}$$

Default weights:

| Term | Weight | Rationale |
|---|---|---|
| $L_\text{bistab}$ | 5.0 | Must dominate when monostable to drive the optimizer toward bistability first |
| $L_\text{rate}$ | 1.0 | Secondary: rate matching once bistable |
| $L_\text{rate,high}$ | 1.5 | Slightly higher than low: high-state targets are harder to satisfy |
| $L_\text{margin}$ | 2.0 | Increased from 0.5: monostable solutions need a strong enough penalty to lose to valid bistable ones |
| $L_\text{jac}$ | 0.1 | Regulariser: active only when pathological gains develop |
| $L_\text{peak}$ | 0.0 | Off by default (threshold 80 Hz, but zero-weighted) |

**How the optimizer sees this:**

The optimizer (CMA-ES or similar gradient-free method) receives $L_\text{total}$ as a scalar for each parameter candidate. The loss landscape has a natural hierarchy:

1. When monostable: $L_\text{bistab} \gg 0$ and $L_\text{margin} = 30$ dominate, creating a strong signal to find the second stable FP regardless of rate matching.
2. Once bistable: $L_\text{bistab} \to 0$, $L_\text{margin} \to 0$, and the rate-matching terms $L_\text{rate}$ and $L_\text{rate,high}$ take over, pulling both FPs toward their biological targets.
3. At a good bistable solution: $L_\text{jac}$ acts as a tiebreaker between solutions with correct rates, preferring those where no single pathway has an implausibly dominant gain.

This hierarchy means the optimizer never gets confused by conflicting gradients at an early stage: shape first, rates second, circuit plausibility third.

---

*Last updated: 2026-04-20*
*Source: [`circuit_model/bistable_loss.py`](../circuit_model/bistable_loss.py), [`circuit_model/jacobian.py`](../circuit_model/jacobian.py), [`circuit_model/params.py`](../circuit_model/params.py)*
