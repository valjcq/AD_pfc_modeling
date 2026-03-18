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
    tau_s: float = 37.3479          # Synaptic time constant (all populations)
    tau_adapt_pyr: float = 186.602  # PYR adaptation time constant (~200ms)
    tau_adapt_som: float = 2320.51  # SOM adaptation time constant (~2.3s, much slower)

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION
    # =========================================================================
    # Adaptation provides negative feedback: high firing -> builds up I_adapt -> reduces firing
    # This prevents runaway excitation and creates bistable UP/DOWN state dynamics
    J_adapt_pyr: float = 0.270443  # PYR adaptation strength (moderate)
    J_adapt_som: float = 27.2356   # SOM adaptation strength (strong, slow kinetics)

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
    w_ps: float = 2.22239   # PV -> SOM: Cross-inhibition between interneuron types This connection doesn't exist in the schematic diagram but is included in the code ? 

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
    # g_e/g_i control curvature for excitatory/inhibitory populations

    Theta_pyr: float = 5.01691   # PYR threshold
    alpha_pyr: float = 0.685403  # PYR gain

    Theta_pv: float = 16.3771    # PV threshold
    alpha_pv: float = 1.47638    # PV gain

    Theta_som: float = 5.88155   # SOM threshold
    alpha_som: float = 0.817185  # SOM gain

    Theta_vip: float = 13.9068   # VIP threshold
    alpha_vip: float = 0.100998  # VIP gain

    g_e: float = 0.377039  # Curvature for excitatory (PYR)
    g_i: float = 0.400125  # Curvature for inhibitory (PV, SOM, VIP)

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
    b["tau_adapt_pyr"] = ParamBound(50.0, 5000.0, mode="log")
    b["tau_adapt_som"] = ParamBound(50.0, 5000.0, mode="log")

    # Adaptation strengths
    b["J_adapt_pyr"] = ParamBound(0.0, 50.0, mode="lin")
    b["J_adapt_som"] = ParamBound(0.0, 50.0, mode="lin")

    # Noise and GABA
    b["sigma_s"] = ParamBound(0.0, 10.0, mode="lin")
    b["g_gaba_base"] = ParamBound(0.0, 5.0, mode="lin")
    b["g_alpha7"] = ParamBound(0.0, 5.0, mode="lin")

    def w_range(x: float, *, min_val: float = 1e-6) -> ParamBound:
        hi = max(1e-6, 5.0 * x)
        lo = min_val if x > 0 else 0.0
        return ParamBound(lo, hi, mode="log")

    # Standard weight ranges
    for name in ["w_ee", "w_pe", "w_ep", "w_pp", "w_vp", "w_sp", "w_ev"]:
        b[name] = w_range(getattr(base, name))

    # Keep a few weights away from zero to avoid KO-insensitive solutions
    b["w_se"] = w_range(base.w_se, min_val=0.1)
    b["w_es"] = w_range(base.w_es, min_val=0.5)
    b["w_vs"] = w_range(base.w_vs, min_val=0.5)

    b["w_ps"] = ParamBound(0.0, 5.0 * base.w_pe, mode="log")

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

    b["g_e"] = ParamBound(0.1, 10.0, mode="log")
    b["g_i"] = ParamBound(0.1, 10.0, mode="log")

    return b
