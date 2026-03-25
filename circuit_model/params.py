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
    # This prevents runaway excitation and creates bistable UP/DOWN state dynamics
    J_adapt_pyr: float = 0.270443  # PYR adaptation strength (moderate)
    J_adapt_som: float = 0.0       # SOM adaptation strength (off by default)

    # =========================================================================
    # NOISE
    # =========================================================================
    sigma_s: float = 5.88856  # Noise amplitude (std dev of Gaussian noise input)

    # =========================================================================
    # GABA SCALING (Inhibitory gain modulation)
    # =========================================================================
    # Total GABA scaling = g_gaba_base + g_alpha7
    # This multiplies inhibitory weights, implementing gain control
    g_gaba_base: float = 3.93207  # Baseline GABA scaling
    g_alpha7: float = 0.95607     # alpha7 nAChR-dependent GABA enhancement

    # =========================================================================
    # SYNAPTIC WEIGHTS
    # =========================================================================
    # Notation: w_XY = weight from Y to X (e=PYR, p=PV, s=SOM, v=VIP)

    # --- Connections FROM PYR (excitatory) ---
    w_ee: float = 6.27108   # PYR -> PYR: Recurrent excitation (maintains persistent activity)
    w_ep: float = 42.5334   # PYR -> PV:  Drives fast feedback inhibition
    w_es: float = 6.56939   # PYR -> SOM: Recruits dendritic inhibition
    w_ev: float = 2.9622e-06  # PYR -> VIP: Very weak (VIP driven by other inputs)

    # --- Connections FROM PV (inhibitory, perisomatic) ---
    w_pe: float = 2.22239   # PV -> PYR: Perisomatic inhibition (divisive, shunting)
    w_pp: float = 105.44    # PV -> PV:  Self-inhibition (limits PV firing rate)

    # --- Connections FROM SOM (inhibitory, dendritic) ---
    w_se: float = 2.61788   # SOM -> PYR: Dendritic inhibition (subtractive)
    w_sp: float = 6.12585e-06  # SOM -> PV: Very weak cross-inhibition

    # --- Connections FROM VIP (inhibitory, disinhibitory) ---
    w_vp: float = 0.0105234  # VIP -> PV:  Weak disinhibition of PV
    w_vs: float = 1.27414    # VIP -> SOM: Core disinhibition pathway (VIP->SOM->PYR)

    # =========================================================================
    # EXTERNAL CURRENTS
    # =========================================================================
    # Each population receives baseline + receptor-mediated currents

    # --- PYR external input ---
    I0_pyr: float = 1.7854 + 5.03758   # Baseline tonic drive

    # --- PV external input ---
    I0_pv: float = 5.58459        # Baseline tonic drive
    I_alpha7_pv: float = 9.90322  # alpha7 nAChR-mediated current (cholinergic enhancement)

    # --- SOM external input ---
    I0_som: float = 5.48551        # Baseline tonic drive
    I_alpha7_som: float = 5.84835  # alpha7 nAChR-mediated current
    I_beta2_som: float = 9.05679   # beta2 nAChR-mediated current (alpha4beta2 receptors on SOM)

    # --- VIP external input ---
    I0_vip: float = 7.57337        # Baseline tonic drive
    I_alpha5_vip: float = 1.44659  # alpha5 nAChR-mediated current (alpha4beta2alpha5 on VIP)

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
    # TRANSFER FUNCTION PARAMETERS (Wong-Wang)
    # =========================================================================
    # Each population has its own threshold (Theta) and gain (alpha)
    # g is shared curvature across all populations

    Theta_pyr: float = 7.0   # PYR threshold
    alpha_pyr: float = 1.9   # PYR gain

    Theta_pv: float = 7.0    # PV threshold
    alpha_pv: float = 2.6    # PV gain

    Theta_som: float = 7.0   # SOM threshold
    alpha_som: float = 1.5   # SOM gain

    Theta_vip: float = 7.0   # VIP threshold
    alpha_vip: float = 1.2   # VIP gain

    g: float = 1.0  # Transfer function curvature (shared across all populations)

    # Output scaling factors (Koukouli et al. 2025, Table 1)
    A_pyr: float = 4.2   # PYR max firing rate scale
    A_pv:  float = 10.1  # PV  max firing rate scale
    A_som: float = 17.1  # SOM max firing rate scale
    A_vip: float = 15.5  # VIP max firing rate scale

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

    Bounds are set based on:
    - Biological plausibility (e.g., time constants in reasonable range)
    - Numerical stability (e.g., weights not too large)
    - Prior knowledge from literature

    Parameters are searched in "log" space (logarithmic) when they span
    orders of magnitude, or "lin" space (linear) otherwise.

    Some weights have minimum values to prevent the optimizer from
    finding "degenerate" solutions where KO conditions have no effect
    because the relevant pathway is already silenced.
    """
    b: dict[str, ParamBound] = {}

    # Time constants
    b["tau_s"] = ParamBound(5.0, 100.0, mode="log")
    b["tau_adapt_pyr"] = ParamBound(100.0, 800.0, mode="log")
    b["tau_adapt_som"] = ParamBound(50.0, 2000.0, mode="log")

    # Adaptation strengths
    b["J_adapt_pyr"] = ParamBound(0.1, 5.0, mode="log")
    b["J_adapt_som"] = ParamBound(0.1, 5.0, mode="log")

    # Noise and GABA
    b["sigma_s"] = ParamBound(0.0, 10.0, mode="lin")
    b["g_gaba_base"] = ParamBound(0.0, 5.0, mode="lin")
    b["g_alpha7"] = ParamBound(0.0, 5.0, mode="lin")

    def w_range(x: float, *, min_val: float = 0.1, hi_factor: float = 5.0) -> ParamBound:
        hi = max(min_val, hi_factor * x)
        lo = min_val
        return ParamBound(lo, hi, mode="log")

    # All weights have a floor of 0.1 — no connection can be fully silenced.
    # Degeneracy (functionally near-zero connections) is instead discouraged
    # via the Jacobian connectivity penalty in the loss function.
    for name in ["w_ee", "w_pe", "w_ep", "w_pp", "w_se", "w_es", "w_vs"]:
        b[name] = w_range(getattr(base, name))

    # w_ev, w_sp, w_vp have near-zero defaults so 5×default ≈ 0 — explicit range
    b["w_ev"] = ParamBound(0.1, 10.0, mode="log")
    b["w_sp"] = ParamBound(0.1, 10.0, mode="log")
    b["w_vp"] = ParamBound(0.1, 10.0, mode="log")

    # External currents
    b["I0_pyr"] = ParamBound(0.0, 10.0, mode="lin")
    b["trans_factor"] = ParamBound(0.0, 1.0, mode="lin")  # Transient as fraction of I0 (0-100%)

    b["I0_pv"] = ParamBound(0.0, 15.0, mode="lin")
    b["I_alpha7_pv"] = ParamBound(0.0, 10.0, mode="lin")

    b["I0_som"] = ParamBound(0.0, 10.0, mode="lin")
    b["I_alpha7_som"] = ParamBound(0.0, 10.0, mode="lin")
    b["I_beta2_som"] = ParamBound(0.0, 10.0, mode="lin")

    b["I0_vip"] = ParamBound(0.0, 10.0, mode="lin")
    b["I_alpha5_vip"] = ParamBound(0.0, 10.0, mode="lin")

    # Transfer function parameters
    for name in ["Theta_pyr", "Theta_pv", "Theta_som", "Theta_vip"]:
        b[name] = ParamBound(0.0, 20.0, mode="lin")
    for name in ["alpha_pyr", "alpha_pv", "alpha_som", "alpha_vip"]:
        b[name] = ParamBound(0.05, 10.0, mode="log")

    b["g"] = ParamBound(0.1, 10.0, mode="log")

    # Output scaling factors
    b["A_pyr"] = ParamBound(0.5, 50.0, mode="log")
    b["A_pv"]  = ParamBound(0.5, 50.0, mode="log")
    b["A_som"] = ParamBound(0.5, 50.0, mode="log")
    b["A_vip"] = ParamBound(0.5, 50.0, mode="log")

    return b
