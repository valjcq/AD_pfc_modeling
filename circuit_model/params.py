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
    sigma_s: float = 0.0   # Noise amplitude (Hz); 0 for deterministic init/testing
    # Relative noise amplitude: std of noise current injected into PYR = sigma_noise * I_ext_pyr.
    # The noise enters the transfer function (current-space), so it is naturally scaled by the
    # drive strength and filtered through the transfer function slope.
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
    # w_pe enters as denominator: denom = 1 + g_gaba * w_pe * r_pv.
    # Meaningful shunting requires g_gaba * w_pe * r_pv ~ 0.2–1.
    # At r_pv ~ 4 Hz, g_gaba ~ 1: w_pe ~ 0.05–0.25 nA/Hz (much larger than additive weights).
    # Default set to 0.05 so the J[PYR,PV] connectivity threshold is met at baseline.
    w_pe: float = 0.05    # PV -> PYR: Perisomatic shunting inhibition
    w_pp: float = 0.002   # PV -> PV:  Self-inhibition

    # --- Connections FROM SOM (inhibitory, dendritic / subtractive) ---
    w_se: float = 0.002   # SOM -> PYR: Dendritic inhibition
    w_sp: float = 0.002   # SOM -> PV:  Cross-inhibition

    # --- Connections FROM VIP (inhibitory, disinhibitory) ---
    w_vp: float = 0.002   # VIP -> PV:  Weak disinhibition of PV
    w_vs: float = 0.002   # VIP -> SOM: Core disinhibition pathway (VIP→SOM→PYR)

    # =========================================================================
    # EXTERNAL CURRENTS
    # =========================================================================
    # Each population receives baseline + receptor-mediated currents

    # --- PYR external input ---
    # I0_pyr must be > Theta_e ≈ 0.403 nA so PYR operates above threshold.
    # Working init: I_syn* ≈ 0.428 nA  (= I0_pyr - 0.012 nA from small weight contributions)
    # Chosen at z≈1.2 (30% below W&W asymptote) to give a 30% Turing window for the ring.
    I0_pyr: float = 0.44   # Baseline tonic drive (nA)

    # --- PV external input ---
    # I0_pv must be > Theta_i ≈ 0.288 nA.
    # Working init: I_syn* ≈ 0.338 nA  (= I0_pv - 0.012 nA)
    I0_pv: float = 0.35            # Baseline tonic drive (nA)
    I_alpha7_pv: float = 0.0       # alpha7 nAChR current (nA); 0 at baseline, fitted for ACh condition

    # --- SOM external input ---
    # Working init: I_syn* ≈ 0.355 nA  (= I0_som + 0.005 nA)
    I0_som: float = 0.35           # Baseline tonic drive (nA)
    I_alpha7_som: float = 0.0      # alpha7 nAChR current (nA)
    I_beta2_som: float = 0.0       # beta2 nAChR current (nA)

    # --- VIP external input ---
    # Working init: I_syn* ≈ 0.347 nA  (= I0_vip + 0.017 nA)
    I0_vip: float = 0.33           # Baseline tonic drive (nA)
    I_alpha5_vip: float = 0.0      # alpha5 nAChR current (nA)

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
    # When trans_enabled=True, a transient current = trans_factor * I0_pop is applied
    # to ALL populations during [trans_start_ms, trans_start_ms + trans_duration_ms)
    # trans_factor is a multiplier (e.g., 0.2 means +20% of baseline I0)
    trans_factor: float = 0.2          # Transient as fraction of each population's I0
    trans_start_ms: float = 1000.0     # When transient starts (ms)
    trans_duration_ms: float = 500.0   # Duration of transient pulse (ms)
    trans_enabled: bool = False        # Whether to use time-dependent transient

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

    def _in_transient_window(self, t_ms: float) -> bool:
        """Check if time t_ms is within the transient window."""
        if not self.trans_enabled:
            return False
        trans_end_ms = self.trans_start_ms + self.trans_duration_ms
        return self.trans_start_ms <= t_ms < trans_end_ms

    def I_ext_pyr(self) -> float:
        """Total external current to PYR (static, no transient)."""
        return self.I0_pyr

    def I_ext_pyr_at_time(self, t_ms: float) -> float:
        """Total external current to PYR at time t_ms (with transient if enabled)."""
        base = self.I0_pyr
        if self._in_transient_window(t_ms):
            return base + self.trans_factor * self.I0_pyr
        return base

    def I_ext_pv(self) -> float:
        """Total external current to PV (with alpha7 modulation, no transient)."""
        return self.I0_pv + self.act_alpha7 * self.I_alpha7_pv

    def I_ext_pv_at_time(self, t_ms: float) -> float:
        """Total external current to PV at time t_ms (with transient if enabled)."""
        base = self.I0_pv + self.act_alpha7 * self.I_alpha7_pv
        if self._in_transient_window(t_ms):
            return base + self.trans_factor * self.I0_pv
        return base

    def I_ext_som(self) -> float:
        """Total external current to SOM (with alpha7 and beta2 modulation, no transient)."""
        return (
            self.I0_som
            + self.act_alpha7 * self.I_alpha7_som
            + self.act_beta2 * self.I_beta2_som
        )

    def I_ext_som_at_time(self, t_ms: float) -> float:
        """Total external current to SOM at time t_ms (with transient if enabled)."""
        base = (
            self.I0_som
            + self.act_alpha7 * self.I_alpha7_som
            + self.act_beta2 * self.I_beta2_som
        )
        if self._in_transient_window(t_ms):
            return base + self.trans_factor * self.I0_som
        return base

    def I_ext_vip(self) -> float:
        """Total external current to VIP (with alpha5 modulation, no transient)."""
        return self.I0_vip + self.act_alpha5 * self.I_alpha5_vip

    def I_ext_vip_at_time(self, t_ms: float) -> float:
        """Total external current to VIP at time t_ms (with transient if enabled)."""
        base = self.I0_vip + self.act_alpha5 * self.I_alpha5_vip
        if self._in_transient_window(t_ms):
            return base + self.trans_factor * self.I0_vip
        return base


@dataclass(frozen=True)
class ParamBound:
    """Search bounds for a single parameter."""
    lo: float
    hi: float
    mode: Literal["lin", "log"] = "log"


def default_bounds(base: CircuitParams) -> dict[str, ParamBound]:
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

    Working init: I_syn* ≈ 0.49 nA (PYR), 0.34–0.36 nA (interneurons).
    All I0_x lower bounds are set ABOVE the W&W threshold so the network
    is never initialised in the silent (below-threshold) regime.
    """
    b: dict[str, ParamBound] = {}

    # --- Time constants (ms) — tau_s fixed at 20 ms, not optimised ---
    # Working WT solution: tau_adapt_pyr=600, tau_adapt_som=150.
    b["tau_adapt_pyr"] = ParamBound(200.0, 1200.0, mode="log")
    b["tau_adapt_som"] = ParamBound(20.0, 300.0, mode="log")

    # --- Adaptation strengths (nA/Hz) ---
    # Working WT solution: J_adapt_pyr=0.002, J_adapt_som~0.001. Tightened upper bound.
    b["J_adapt_pyr"] = ParamBound(0.001, 0.2, mode="log")
    b["J_adapt_som"] = ParamBound(0.001, 0.2, mode="lin")

    # --- GABA modulation (dimensionless) ---
    # Working WT solution: g_gaba_base=1.0, g_alpha7~0. Tightened from [0.5,20] and [0,20].
    b["g_gaba_base"] = ParamBound(0.5, 5.0, mode="lin")
    b["g_alpha7"]    = ParamBound(0.0, 5.0, mode="lin")

    # --- Synaptic weights (nA/Hz) ---
    # Bounds tightened 2026-04-13 based on Monte Carlo viable region analysis (4.07% → target 15%+).
    # Previous uniform [0, 0.5] for all synaptic weights created 96% dead zone above best-fit values.
    # New strategy: set lower bound = 0.001 nA/Hz (neuronal noise floor), upper bound = 3× best-fit (or 0.5 if already large).
    # Best-fit reference from best_params.json WT solution:
    #   Small weights (< 0.06): w_ep=0.0043, w_ev=0.0009, w_sp=0.0528, w_vp=0.0919, w_vs=0.3405
    #   Medium weights (0.06–0.5): w_se=0.4877, w_es=0.0353, w_pp=0.3616
    #   Divisive w_pe=0.0267
    # J_NMDA (recurrent): larger bounds [0.05, 2.0] due to gating saturation.
    _W_LO = 0.001  # Lower bound: biological noise floor

    # Small excitatory weights with tightened upper bounds
    b["w_ep"]  = ParamBound(0.001, 0.10, mode="log")   # 3× best-fit 0.0043
    b["w_ev"]  = ParamBound(0.0005, 0.02, mode="log")  # best-fit 0.000935 requires lower_min=0.0005 (half of best-fit), upper=3×best-fit≈0.003, rounded to 0.02
    b["w_sp"]  = ParamBound(0.001, 0.15, mode="log")   # 3× best-fit 0.0528
    b["w_vp"]  = ParamBound(0.001, 0.25, mode="log")   # 3× best-fit 0.0919
    b["w_vs"]  = ParamBound(0.001, 0.5,  mode="log")   # 3× best-fit 0.3405

    # Larger weights: add lower bound minimum, keep upper at 0.5
    b["w_se"]  = ParamBound(_W_LO, 0.5, mode="log")    # best-fit 0.4877 — already near ceiling
    b["w_es"]  = ParamBound(_W_LO, 0.5, mode="log")    # best-fit 0.0353
    b["w_pp"]  = ParamBound(_W_LO, 0.5, mode="log")    # best-fit 0.3616

    # J_NMDA: NMDA recurrent coupling with wider bounds due to gating saturation
    b["J_NMDA"] = ParamBound(0.05, 2.0, mode="log")

    # w_pe: DIVISIVE (shunting) inhibition — enters denominator as 1 + g_gaba*w_pe*r_pv.
    # Working WT solution: w_pe=0.0267. Lower bound tightened from 0 → 0.001; upper kept at 1.0.
    b["w_pe"] = ParamBound(_W_LO, 1.0, mode="log")

    # --- External tonic drives (nA) ---
    # Working WT solution: I0_pyr=0.530, I0_pv=0.480, I0_som=0.528, I0_vip=0.198.
    # Tightened 2026-04-13: I0_pv was at 92% of range [0.1, 0.6] (norm_pos=0.917), causing 96% waste below.
    # APP solution: I0_pyr=0.435, I0_inh=0.255–0.265. Tightened from [0.01, 2.0] → [0.1, 0.8].
    b["I0_pyr"] = ParamBound(0.1, 1.5, mode="lin")
    b["I0_pv"]  = ParamBound(0.30, 0.65, mode="lin")  # Raise lower bound; extend upper 15% above best-fit 0.480
    b["I0_som"] = ParamBound(0.1, 0.6, mode="lin")
    b["I0_vip"] = ParamBound(0.1, 0.6, mode="lin")

    # Transient stimulus (dimensionless fraction of I0_pyr), centered near working init 0.2.
    b["trans_factor"] = ParamBound(0.0, 1.0, mode="lin")  # TODO: Remove from optimization (not a circuit parameter, but a stimulus parameter used for testing transient response)

    # --- nAChR cholinergic currents (nA) ---
    # These add to I0_x; should be comparable fraction of (I0_x - Theta_x).
    # Working init is 0 for all receptor-mediated currents. Tightened from [0, 2] → [0, 0.5].
    b["I_alpha7_pv"]  = ParamBound(0.0, 0.5, mode="lin")
    b["I_alpha7_som"] = ParamBound(0.0, 0.5, mode="lin")
    b["I_beta2_som"]  = ParamBound(0.0, 0.5, mode="lin")
    b["I_alpha5_vip"] = ParamBound(0.0, 0.5, mode="lin")

    return b