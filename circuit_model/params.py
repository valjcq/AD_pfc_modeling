"""
Circuit model parameters and bounds definitions.

This module contains:
- CircuitParams: All ~60 parameters for the 4-population PFC circuit model
- ParamBound: Search bounds for optimization
- default_bounds: Default parameter search ranges
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal


@dataclass(frozen=True)
class CircuitParams:
    """
    All parameters for the 4-population PFC circuit model.

    This dataclass holds ~60 parameters organized into categories:
    - Time constants: membrane/synaptic dynamics and adaptation
    - Adaptation: spike-frequency adaptation strengths
    - Noise: stochastic input amplitude
    - GABA scaling: inhibitory gain modulation
    - Synaptic weights: connection strengths between populations
    - External currents: tonic and receptor-mediated inputs
    - Transfer function: threshold, gain, and curvature per population

    Naming conventions for weights:
        w_XY means connection FROM population Y TO population X
        e = excitatory (PYR), p = PV, s = SOM, v = VIP
        Example: w_ep = weight from PYR (e) to PV (p)
    """

    # =========================================================================
    # TIME CONSTANTS (ms)
    # =========================================================================
    tau_s: float = 20.0            # Synaptic time constant (all populations)
    tau_adapt_pyr: float = 600.0   # PYR adaptation time constant (~600ms)
    tau_adapt_som: float = 150.0   # SOM adaptation time constant (ms)

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION
    # =========================================================================
    # Adaptation provides negative feedback: high firing -> builds up I_adapt -> reduces firing
    # J_adapt in nA/Hz: at r_pyr ~ 8.5 Hz, J_adapt_pyr=0.002 gives I_adapt ~ 0.017 nA.
    J_adapt_pyr: float = 0.002   # PYR adaptation strength (nA/Hz)
    J_adapt_som: float = 0.0     # SOM adaptation strength (off by default)

    # =========================================================================
    # NOISE
    # =========================================================================
    # Relative noise amplitude: std of noise current injected into each population
    # is sigma_noise * I_ext_pop. Noise enters in current-space, so it is naturally
    # scaled by the drive strength and filtered through the transfer function slope.
    sigma_noise: float = 0.3

    # =========================================================================
    # GABA SCALING (Inhibitory gain modulation)
    # =========================================================================
    # Total GABA scaling = g_gaba_base + g_alpha7
    # This multiplies inhibitory weights, implementing gain control
    g_gaba_base: float = 1.0   # Baseline GABA scaling (dimensionless)
    g_alpha7: float = 0.0      # alpha7 nAChR-dependent GABA enhancement

    # =========================================================================
    # SYNAPTIC WEIGHTS
    # =========================================================================
    # Notation: w_XY = weight from Y to X (e=PYR, p=PV, s=SOM, v=VIP)

    # All weights in nA/Hz.  At r ~ 10 Hz, weight × rate → nA of synaptic input.
    # Default: small uniform starting point for the W&W operating regime.

    # --- Connections FROM PYR (excitatory) ---
    J_NMDA: float = 0.3   # PYR -> PYR: NMDA recurrent coupling (nA); replaces w_ee
    w_ep: float = 0.002   # PYR -> PV:  Drives fast feedback inhibition
    w_es: float = 0.002   # PYR -> SOM: Recruits dendritic inhibition
    w_ev: float = 0.002   # PYR -> VIP: Disinhibitory drive

    # --- Connections FROM PV (inhibitory, perisomatic / DIVISIVE) ---
    w_pe: float = 0.002   # PV -> PYR: Perisomatic shunting inhibition
    w_pp: float = 0.002   # PV -> PV:  Self-inhibition

    # --- Connections FROM SOM (inhibitory, dendritic / subtractive) ---
    w_se: float = 0.002   # SOM -> PYR: Dendritic inhibition
    w_sp: float = 0.002   # SOM -> PV:  Cross-inhibition

    # --- Connections FROM VIP (inhibitory, disinhibitory) ---
<<<<<<< HEAD
    w_vp: float = 0.002   # VIP -> PV:  Weak disinhibition of PV
    w_vs: float = 0.002   # VIP -> SOM: Core disinhibition pathway (VIP→SOM→PYR)
=======
    w_vp: float = 0.0105234  # VIP -> PV:  Weak disinhibition of PV
    w_vs: float = 1.27414    # VIP -> SOM: Core disinhibition pathway (VIP->SOM->PYR)
>>>>>>> origin/main

    # =========================================================================
    # EXTERNAL CURRENTS
    # =========================================================================
    # Each population receives baseline + receptor-mediated currents

    # --- PYR external input ---
    I0_pyr: float = 0.44   # Baseline tonic drive (nA)

    # --- PV external input ---
    I0_pv: float = 0.35            # Baseline tonic drive (nA)
    I_alpha7_pv: float = 0.20      # alpha7 nAChR current (nA)

    # --- SOM external input ---
    I0_som: float = 0.35           # Baseline tonic drive (nA)
    I_alpha7_som: float = 0.20     # alpha7 nAChR current (nA)
    I_beta2_som: float = 0.20      # beta2 nAChR current (nA)

    # --- VIP external input ---
    I0_vip: float = 0.35           # Baseline tonic drive (nA)
    I_alpha5_vip: float = 0.20     # alpha5 nAChR current (nA)

    # =========================================================================
    # RECEPTOR ACTIVATION MULTIPLIERS (for knockout experiments)
    # =========================================================================
    # Set to 0 to simulate receptor knockout; set to 1 for normal condition; use intermediate values for partial blockade/desensitization
    act_alpha7: float = 1.0  # alpha7 nAChR activation (affects PV, SOM, GABA scaling)
    act_beta2: float = 1.0   # beta2 nAChR activation (affects SOM)
    act_alpha5: float = 1.0  # alpha5 nAChR activation (affects VIP)

    # =========================================================================
    # TRANSIENT CURRENT TIMING (for time-varying stimulation)
    # =========================================================================
    # When trans_enabled=True, a transient current = trans_factor * I0_pyr is applied
    # to PYR ONLY during [trans_start_ms, trans_start_ms + trans_duration_ms)
    # trans_factor is a multiplier (e.g., 0.2 means +20% of PYR's baseline I0)
    # A second independent transient (trans2_*) can be enabled separately.
    trans_factor: float = 0.2          # Transient as fraction of PYR's I0
    trans_start_ms: float = 1000.0     # When transient starts (ms)
    trans_duration_ms: float = 500.0   # Duration of transient pulse (ms)
    trans_enabled: bool = False        # Whether to use time-dependent transient

    trans2_factor: float = -0.3        # Second transient factor (negative = inhibitory)
    trans2_start_ms: float = 3000.0    # When second transient starts (ms)
    trans2_duration_ms: float = 500.0  # Duration of second transient pulse (ms)
    trans2_enabled: bool = False       # Whether to use second transient

    # =========================================================================
    # TRANSFER FUNCTION PARAMETERS (Wong-Wang 2006, exact values)
    # =========================================================================
    # Form: Phi(I) = alpha * (I - Theta) / (1 - exp(-g * alpha * (I - Theta)))
    # which is equivalent to the W&W form  (c*I - I0) / (1 - exp(-g*(c*I - I0)))
    # with  alpha = c_x (Hz/nA),  Theta = I0_x / c_x (nA),  g = g_x (s).
    #
    # These six constants are FIXED from W&W 2006 and are NOT optimised.
    # Derived thresholds: Theta_e = 125/310 ≈ 0.403 nA  (PYR begins to fire)
    #                     Theta_i = 177/615 ≈ 0.288 nA  (PV/SST/VIP begin to fire)

    # Excitatory (PYR)
    alpha_pyr: float = 310.0          # c_e  (Hz/nA) — W&W 2006
    Theta_pyr: float = 125.0 / 310.0  # I0_e / c_e  (nA)
    g_exc:     float = 0.16           # g_e  (s)     — W&W 2006

    # Inhibitory (PV, SST, VIP) — same W&W class
    alpha_pv:  float = 615.0          # c_i  (Hz/nA) — W&W 2006
    Theta_pv:  float = 177.0 / 615.0  # I0_i / c_i  (nA)

    alpha_som: float = 615.0
    Theta_som: float = 177.0 / 615.0

    alpha_vip: float = 615.0
    Theta_vip: float = 177.0 / 615.0

    g_inh: float = 0.087              # g_i  (s)     — W&W 2006

    def g_gaba(self) -> float:
        """Total GABA scaling factor."""
        return self.g_gaba_base + self.g_alpha7

    def _transient_delta(self, t_ms: float) -> float:
        """Sum of transient factors (from both transients) active at t_ms."""
        delta = 0.0
        if self.trans_enabled:
            if self.trans_start_ms <= t_ms < self.trans_start_ms + self.trans_duration_ms:
                delta += self.trans_factor
        if self.trans2_enabled:
            if self.trans2_start_ms <= t_ms < self.trans2_start_ms + self.trans2_duration_ms:
                delta += self.trans2_factor
        return delta

    def I_ext_pyr(self) -> float:
        """Total external current to PYR (static, no transient)."""
        return self.I0_pyr

    def I_ext_pyr_at_time(self, t_ms: float) -> float:
        """Total external current to PYR at time t_ms (with transients if enabled)."""
        delta = self._transient_delta(t_ms)
        return self.I0_pyr + delta * self.I0_pyr

    def I_ext_pv(self) -> float:
        """Total external current to PV (with alpha7 modulation, no transient)."""
        return self.I0_pv + self.act_alpha7 * self.I_alpha7_pv

    def I_ext_pv_at_time(self, t_ms: float) -> float:
        """Total external current to PV (transient is PYR-only, so always static)."""
        return self.I_ext_pv()

    def I_ext_som(self) -> float:
        """Total external current to SOM (with alpha7 and beta2 modulation, no transient)."""
        return (
            self.I0_som
            + self.act_alpha7 * self.I_alpha7_som
            + self.act_beta2 * self.I_beta2_som
        )

    def I_ext_som_at_time(self, t_ms: float) -> float:
        """Total external current to SOM (transient is PYR-only, so always static)."""
        return self.I_ext_som()

    def I_ext_vip(self) -> float:
        """Total external current to VIP (with alpha5 modulation, no transient)."""
        return self.I0_vip + self.act_alpha5 * self.I_alpha5_vip

    def I_ext_vip_at_time(self, t_ms: float) -> float:
        """Total external current to VIP (transient is PYR-only, so always static)."""
        return self.I_ext_vip()


@dataclass(frozen=True)
class ParamBound:
    """Search bounds for a single parameter."""
    lo: float
    hi: float
    mode: Literal["lin", "log"] = "log"


def default_bounds(base: CircuitParams, w_hi: float | None = None) -> dict[str, ParamBound]:
    """
    Define search bounds for each optimizable parameter.

    Units (W&W physical convention):
    - Synaptic weights:        nA/Hz  (weight × rate → nA input current)
    - External / nAChR drives: nA     (enter I_syn directly)
    - Adaptation strengths:    nA/Hz  (J_adapt × rate → nA adaptation current)
    - GABA modulation:         dimensionless
    Note: sigma_noise is fixed (not optimized); noise enters optimization via n_trials averaging.

    Transfer function shape parameters (alpha_x, Theta_x, g_exc, g_inh) are FIXED
    from W&W 2006 and are NOT included here.

    Threshold references:
    - Theta_e = 125/310 ≈ 0.403 nA  (PYR threshold)
    - Theta_i = 177/615 ≈ 0.288 nA  (PV/SST/VIP threshold)

    """
    b: dict[str, ParamBound] = {}

    # --- Time constants (ms) — tau_s fixed at 20 ms, not optimised ---
    # We would set it to : tau_adapt_pyr=600, tau_adapt_som=150.
    b["tau_adapt_pyr"] = ParamBound(200.0, 1200.0, mode="log")
    b["tau_adapt_som"] = ParamBound(20.0, 300.0, mode="log")

    # --- Adaptation strengths (nA/Hz) ---
    b["J_adapt_pyr"] = ParamBound(0.001, 0.2, mode="log")
    b["J_adapt_som"] = ParamBound(0.001, 0.2, mode="lin")

    # --- GABA modulation (dimensionless) ---
    b["g_gaba_base"] = ParamBound(0.1, 5.0, mode="lin")
    b["g_alpha7"]    = ParamBound(0.1, 5.0, mode="lin")

    # --- Synaptic weights (nA/Hz) ---
    _W_LO = 0.001  # Lower bound: small but nonzero.
    # We also use jacobian loss to ensure the connection isn't nothing between each populations.

    _W_HI = w_hi if w_hi is not None else 0.01  # Upper bound: tightened for bistable regime search

    # Synaptic weights capped at 0.01 nA/Hz
    b["w_ep"]  = ParamBound(_W_LO,  _W_HI, mode="log")
    b["w_ev"]  = ParamBound(_W_LO, _W_HI, mode="log")
    b["w_sp"]  = ParamBound(_W_LO,  _W_HI, mode="log")
    b["w_vp"]  = ParamBound(_W_LO,  _W_HI, mode="log")
    b["w_vs"]  = ParamBound(_W_LO,  _W_HI, mode="log")
    b["w_se"]  = ParamBound(_W_LO,  _W_HI, mode="log")
    b["w_es"]  = ParamBound(_W_LO,  _W_HI, mode="log")
    b["w_pp"]  = ParamBound(_W_LO,  _W_HI, mode="log")

<<<<<<< HEAD
    # J_NMDA: NMDA recurrent coupling with wider bounds due to gating saturation
    b["J_NMDA"] = ParamBound(0.05, 2.0, mode="log")
=======
    b["w_ps"] = ParamBound(0.0, 5.0 * base.w_pe, mode="log")
>>>>>>> origin/main

    # w_pe: DIVISIVE (shunting) inhibition — enters denominator as 1 + g_gaba*w_pe*r_pv.
    b["w_pe"] = ParamBound(_W_LO, _W_HI, mode="log")

    # --- External tonic drives (nA) ---
    # We set the same lower bound of 0.01 nA for all I0_x and the same upper bound of 0.6 nA for interneurons
    # Pyr have higher upper bound as we consider that it can receive stronger external drive (e.g., from other brain areas) than interneurons.
    b["I0_pyr"] = ParamBound(0.01, 1.5, mode="lin")
    b["I0_pv"]  = ParamBound(0.01, 0.6, mode="lin") 
    b["I0_som"] = ParamBound(0.01, 0.6, mode="lin")
    b["I0_vip"] = ParamBound(0.01, 0.6, mode="lin")

    # --- nAChR cholinergic currents (nA) ---
    b["I_alpha7_pv"]  = ParamBound(0.01, 0.5, mode="lin")
    b["I_alpha7_som"] = ParamBound(0.01, 0.5, mode="lin")
    b["I_beta2_som"]  = ParamBound(0.01, 0.5, mode="lin")
    b["I_alpha5_vip"] = ParamBound(0.01, 0.5, mode="lin")

    return b