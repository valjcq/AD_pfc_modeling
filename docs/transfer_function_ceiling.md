# Transfer Function Saturation: PYR (NMDA) and Interneurons (Soft Ceiling)

---

## 1. The Problem: Unbounded Transfer Functions Produce Pathological Dynamics

The Wong-Wang (2006) transfer function used throughout this model is:

$$\Phi(I) = \frac{\alpha(I - \theta)}{1 - e^{-g \cdot \alpha(I - \theta)}}$$

This function is monotonically increasing and **does not saturate**. For large inputs, it grows approximately linearly without bound. In practice this creates two classes of pathological behavior during optimization and simulation:

- **Runaway excitation**: PYR nodes saturate at the 200 Hz numerical clamp, contaminating adaptation state and preventing clean bump formation.
- **Interneuron over-activation**: SOM neurons were observed firing at 85–159 Hz at rest in bistable parameter regimes — far above any physiological range and inconsistent with the biological role of these cells.

Both problems require introducing a saturation mechanism into the transfer function. However, the **biological justification and implementation differ** between PYR and the three interneuron populations (PV, SOM, VIP).

---

## 2. PYR Saturation: Biologically Grounded via NMDA Gating

### Mechanism

For PYR neurons, saturation is implemented through the **NMDA gating variable** $S_i^{\text{NMDA}} \in [0, 1]$, which replaces the instantaneous linear recurrent drive $w_{ee} \cdot r_i^{\text{PYR}}$:

$$\tau_{\text{NMDA}} \frac{dS_i}{dt} = -S_i + (1 - S_i) \cdot \gamma_{\text{NMDA}} \cdot r_i^{\text{PYR}}$$

The $(1 - S_i)$ term is the saturation mechanism: as $r_{\text{PYR}}$ increases, $S_i$ approaches 1 and the incremental drive from each additional spike decreases. At steady state:

$$S^* = \frac{\gamma_{\text{NMDA}} \cdot \tau_{\text{NMDA}} \cdot r^*}{1 + \gamma_{\text{NMDA}} \cdot \tau_{\text{NMDA}} \cdot r^*}$$

This saturating nonlinearity is what creates the **fold in the PYR nullcline** required for bistability. Without it — using the fast-synapse linear approximation $S \approx \tau_s r$ — the nullcline has no fold and single-node bistability is structurally impossible regardless of parameter choice.

### Biological justification

The NMDA gating approach is fully grounded in the biophysics of NMDA receptor kinetics (Wong & Wang 2006; Jahr & Stevens 1990). The constants $\tau_{\text{NMDA}} = 100$ ms and $\gamma_{\text{NMDA}} = 0.641$ are fixed from the original model and represent measured receptor properties. The saturation reflects the fact that NMDA receptor channels are **voltage- and ligand-gated** with a finite open-channel fraction. This is not a phenomenological correction — it is the correct biophysical description of PFC recurrent excitation, where NMDA receptors dominate slow recurrent currents (Wang 1999; Compte et al. 2000).

The NMDA mechanism also directly connects to the nAChR framework: nicotinic receptors modulate NMDA-mediated transmission in PFC Layer 2/3 (Couey et al. 2007; Gulledge et al. 2009), making this architecture consistent with the Koukouli et al. 2025 experimental context.

### Parameters

| Parameter | Value | Source |
|---|---|---|
| $\tau_{\text{NMDA}}$ | 100 ms | Wong & Wang 2006 (fixed) |
| $\gamma_{\text{NMDA}}$ | 0.641 | Wong & Wang 2006 (fixed) |
| $J_{\text{NMDA}}$ | fitted | Free parameter replacing $w_{ee}$ |

---

## 3. Interneuron Saturation: Why the Same Approach Cannot Be Applied

### Why NMDA gating does not apply to interneurons

The NMDA gating mechanism is specific to **recurrent PYR→PYR excitation** mediated by NMDA receptors. It cannot be applied to PV, SOM, or VIP for three interconnected reasons:

**1. Interneuron synapses are GABA-mediated, not NMDA-mediated.**
The saturation in the NMDA formulation reflects the fraction of open NMDA channels on PYR dendrites. Interneuron outputs are GABAergic, and their *inputs* from PYR are dominated by fast AMPA receptors, not NMDA receptors. There is no equivalent saturable gating variable with a well-characterized kinetic form that would apply here.

**2. There is no analogous bistability requirement for interneurons.**
The fold in the PYR nullcline serves a specific functional purpose: creating two stable fixed points for working memory. Interneurons do not need to be bistable. The saturation constraint on interneurons is purely a physiological upper bound, not a mechanistic requirement of the model architecture.

**3. Short-term synaptic depression is the closest biophysical equivalent — but it is too invasive.**
The most principled biological mechanism that would limit interneuron firing rates at high input levels is **short-term synaptic depression (STD)** at the PYR→interneuron synapses, combined with postsynaptic receptor desensitization. STD (Tsodyks & Markram 1997) is well-characterized for PV inputs and would naturally implement a saturation-like effect. However, introducing STD requires additional dynamical variables per synapse type, substantially increases model complexity, and introduces new free parameters that are not constrained by the Koukouli et al. dataset. This architectural cost is not justified for what is, functionally, a ceiling constraint.

### The soft ceiling approach

Instead, a **hyperbolic soft ceiling** is applied post-hoc to the Wong-Wang transfer function for all three interneuron populations:

$$\Phi_{\text{capped}}^X(I) = \frac{r_{\text{max}}^X \cdot \Phi^X(I)}{r_{\text{max}}^X + \Phi^X(I)}$$

This form has the following properties:

- For $\Phi \ll r_{\text{max}}$: $\Phi_{\text{capped}} \approx \Phi$ — the low-rate operating regime is unaffected.
- As $\Phi \to \infty$: $\Phi_{\text{capped}} \to r_{\text{max}}$ — firing rate is bounded.
- The gain is compressed smoothly: $\Phi'_{\text{capped}} = \Phi' \cdot \dfrac{r_{\text{max}}^2}{(r_{\text{max}} + \Phi)^2}$

The ceiling is a single-parameter modification that does not add dynamical variables, preserves the transfer function shape at physiological operating points, and has closed-form derivatives compatible with the existing nullcline analysis.

---

## 4. Setting the Ceiling: 1.5 × High-State Target Rate

### Rationale

The ceiling values are set to **1.5 times the high-state firing rate target** for each interneuron population, as defined by the Rooy (2021) active-state values used in `L_rate_high`:

$$r_{\text{max}}^X = 1.5 \times r_{\text{high}}^X$$

| Population | High-state target (Rooy 2021, Hz) | Ceiling $r_{\text{max}}$ (Hz) |
|---|---|---|
| **PV** | 35.3 | **53 Hz** |
| **SOM** | 35.2 | **53 Hz** |
| **VIP** | 68.8 | **103 Hz** |

### Justification

This approach is **internally consistent**: the same dataset that defines the target operating points also defines the ceiling, without requiring additional literature sources for interneurons specifically. The 1.5× multiplier guarantees that:

1. The optimizer is unconstrained across the full physiologically observed range (from resting ~1–8 Hz to high-state ~35–70 Hz).
2. The ceiling activates only above the highest observed operating point, preventing pathological runaway without interfering with target-matching.
3. The gain compression at the high-state operating point is bounded: at $r = r_{\text{high}}$, $\Phi'_{\text{capped}} / \Phi' = (1.5)^2 / (1.5 + 1)^2 = 0.36$ — a known, fixed, and uniform compression factor across all interneuron populations.

### Independent corroboration

The SOM ceiling of ~53 Hz is independently supported by Huang et al. (2016, *Scientific Reports*), who report that the maximal firing rate of SOM interneurons in the anterior cingulate cortex of mice stabilizes at approximately 50 Hz after postnatal day 12–14 — the same cortical area targeted by this model. The 53 Hz ceiling is therefore consistent with the directly measured physiological maximum for SOM in this region.

For PV and VIP, no single paper reports a clean numerical maximum firing rate specifically from mouse prefrontal or anterior cingulate cortex under the same conditions. The 1.5× derivation from Rooy 2021 is therefore the most internally consistent and tractable approach for these populations.

---

## 5. Summary Table

| Population | Saturation mechanism | Biological basis | Parameters |
|---|---|---|---|
| **PYR** | NMDA gating variable $S^{\text{NMDA}} \in [0,1]$ | NMDA receptor kinetics (Wong & Wang 2006); required for nullcline fold and bistability | $\tau_{\text{NMDA}} = 100$ ms, $\gamma = 0.641$ (fixed); $J_{\text{NMDA}}$ (free) |
| **PV** | Hyperbolic soft ceiling | Phenomenological; 1.5 × Rooy 2021 high-state target | $r_{\text{max}} = 53$ Hz |
| **SOM** | Hyperbolic soft ceiling | Phenomenological; 1.5 × Rooy 2021; corroborated by Huang et al. 2016 (50 Hz max in ACC) | $r_{\text{max}} = 53$ Hz |
| **VIP** | Hyperbolic soft ceiling | Phenomenological; 1.5 × Rooy 2021 high-state target | $r_{\text{max}} = 103$ Hz |

The asymmetry between PYR and interneurons reflects a fundamental difference in the model architecture: NMDA saturation in PYR is a **mechanistic requirement** for attractor dynamics, whereas the soft ceiling on interneurons is a **physiological constraint** that prevents numerical pathologies without altering the computational function of the inhibitory populations.

---

## References

- Wong, K.-F. & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314–1328.
- Huang, Y. et al. (2016). Postnatal development of the electrophysiological properties of somatostatin interneurons in the anterior cingulate cortex of mice. *Scientific Reports*, 6, 28137.
- Rooy, M. (2021). *Four-population mean-field model of prefrontal cortex working memory*. [Thesis/internal reference]
- Tian, M.-K. et al. (2016). Firing frequency maxima of fast-spiking neurons in human, monkey, and mouse neocortex. *Frontiers in Cellular Neuroscience*, 10, 239.
- Tsodyks, M.V. & Markram, H. (1997). The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. *PNAS*, 94(2), 719–723.