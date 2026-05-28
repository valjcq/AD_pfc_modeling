"""
Circuit model parameters and bounds definitions.

This module contains:
- CircuitParams: All parameters for the 5-population PFC circuit model
  (PYR, SOM, PV, VIP, NDNF)
- ParamBound: Search bounds for optimization
- default_bounds: Default parameter search ranges

Population order convention throughout the codebase:
    [PYR, SOM, PV, VIP, NDNF]  (indices 0..4)

Weight naming convention:
    w_XY = weight FROM population X TO population Y
    e = PYR, p = PV, s = SOM, v = VIP, n = NDNF
    Examples: w_es = PYR → SOM, w_ne = NDNF → PYR, w_sn = SOM → NDNF
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal


@dataclass(frozen=True)
class CircuitParams:
    """
    All parameters for the 5-population PFC circuit model.

    Population order: PYR, SOM, PV, VIP, NDNF.

    NDNF (added 2026-05): subtractive dendritic inhibitor (like SOM).
        - Expresses α7 and β2 nAChRs (same as SOM)
        - Receives:  SOM → NDNF (w_sn, subtractive)
                     [no PYR → NDNF in this model]
        - Projects:  NDNF → PYR dendrites (w_ne, subtractive)
                     NDNF → PV  (w_np, subtractive)
                     NDNF → VIP (w_nv, subtractive)
        - Transfer:  Wong-Wang + hyperbolic soft ceiling R_MAX_NDNF
    """

    # =========================================================================
    # TIME CONSTANTS (ms)
    # =========================================================================
    tau_s: float = 20.0
    tau_adapt_pyr: float = 600.0
    tau_adapt_som: float = 150.0

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION
    # =========================================================================
    J_adapt_pyr: float = 0.002
    J_adapt_som: float = 0.0

    # =========================================================================
    # NOISE
    # =========================================================================
    sigma_noise: float = 0.3

    # =========================================================================
    # GABA SCALING
    # =========================================================================
    # Total GABA scaling = g_gaba_base + g_alpha7 * mean(act_alpha7_pv,
    #                                                   act_alpha7_som,
    #                                                   act_alpha7_ndnf)
    g_gaba_base: float = 1.0
    g_alpha7: float = 0.0

    # =========================================================================
    # SYNAPTIC WEIGHTS  (nA/Hz)
    # =========================================================================

    # --- Connections FROM PYR (excitatory) ---
    J_NMDA: float = 0.3   # PYR -> PYR (NMDA-gated)
    w_ep: float = 0.002   # PYR -> PV
    w_es: float = 0.002   # PYR -> SOM
    w_ev: float = 0.002   # PYR -> VIP

    # --- Connections FROM PV (inhibitory, perisomatic / DIVISIVE) ---
    w_pe: float = 0.002   # PV -> PYR
    w_pp: float = 0.002   # PV -> PV

    # --- Connections FROM SOM (inhibitory, dendritic / subtractive) ---
    w_se: float = 0.002   # SOM -> PYR
    w_sp: float = 0.002   # SOM -> PV
    w_sn: float = 0.002   # SOM -> NDNF

    # --- Connections FROM VIP (inhibitory, disinhibitory) ---
    w_vp: float = 0.002   # VIP -> PV
    w_vs: float = 0.002   # VIP -> SOM

    # --- Connections FROM NDNF (inhibitory, subtractive) ---
    w_ne: float = 0.002   # NDNF -> PYR (dendrites)
    w_np: float = 0.002   # NDNF -> PV
    w_nv: float = 0.002   # NDNF -> VIP

    # =========================================================================
    # EXTERNAL CURRENTS (nA)
    # =========================================================================

    I0_pyr: float = 0.44

    I0_pv: float = 0.35
    I_alpha7_pv: float = 0.20

    I0_som: float = 0.35
    I_alpha7_som: float = 0.20
    I_beta2_som: float = 0.20

    I0_vip: float = 0.35
    I_alpha5_vip: float = 0.20

    # NDNF receives α7 + β2 currents like SOM
    I0_ndnf: float = 0.35           # placeholder, TODO: refine from literature
    I_alpha7_ndnf: float = 0.20
    I_beta2_ndnf: float = 0.20

    # =========================================================================
    # RECEPTOR ACTIVATION MULTIPLIERS (knockouts)
    # =========================================================================
    # α7 is now per-cell so we can simulate cell-type-specific α7 knockouts.
    # Global α7-KO = all three set to 0 (and g_alpha7 is multiplied by the mean,
    # so it also vanishes). The mean-scaling means a single selective α7-KO
    # reduces g_alpha7 by ~1/3.
    act_alpha7_pv:   float = 1.0
    act_alpha7_som:  float = 1.0
    act_alpha7_ndnf: float = 1.0
    act_beta2: float = 1.0   # affects SOM and NDNF
    act_alpha5: float = 1.0  # affects VIP

    # =========================================================================
    # TRANSIENT CURRENT TIMING
    # =========================================================================
    trans_factor: float = 0.2
    trans_start_ms: float = 1000.0
    trans_duration_ms: float = 500.0
    trans_enabled: bool = False

    trans2_factor: float = -0.3
    trans2_start_ms: float = 3000.0
    trans2_duration_ms: float = 500.0
    trans2_enabled: bool = False

    # =========================================================================
    # TRANSFER FUNCTION PARAMETERS (Wong-Wang 2006, fixed)
    # =========================================================================

    alpha_pyr: float = 310.0
    Theta_pyr: float = 125.0 / 310.0
    g_exc:     float = 0.16

    alpha_pv:  float = 615.0
    Theta_pv:  float = 177.0 / 615.0

    alpha_som: float = 615.0
    Theta_som: float = 177.0 / 615.0

    alpha_vip: float = 615.0
    Theta_vip: float = 177.0 / 615.0

    alpha_ndnf: float = 615.0
    Theta_ndnf: float = 177.0 / 615.0

    g_inh: float = 0.087

    # =========================================================================
    # Helpers
    # =========================================================================

    def act_alpha7_mean(self) -> float:
        """Mean of per-cell α7 activation — used to scale g_alpha7."""
        return (self.act_alpha7_pv + self.act_alpha7_som + self.act_alpha7_ndnf) / 3.0

    def g_gaba(self) -> float:
        """Total GABA scaling factor. g_alpha7 is gated by mean α7 activation."""
        return self.g_gaba_base + self.g_alpha7 * self.act_alpha7_mean()

    def _transient_delta(self, t_ms: float) -> float:
        delta = 0.0
        if self.trans_enabled:
            if self.trans_start_ms <= t_ms < self.trans_start_ms + self.trans_duration_ms:
                delta += self.trans_factor
        if self.trans2_enabled:
            if self.trans2_start_ms <= t_ms < self.trans2_start_ms + self.trans2_duration_ms:
                delta += self.trans2_factor
        return delta

    # --- External-current accessors (with per-cell α7 modulation) ---

    def I_ext_pyr(self) -> float:
        return self.I0_pyr

    def I_ext_pyr_at_time(self, t_ms: float) -> float:
        delta = self._transient_delta(t_ms)
        return self.I0_pyr + delta * self.I0_pyr

    def I_ext_pv(self) -> float:
        return self.I0_pv + self.act_alpha7_pv * self.I_alpha7_pv

    def I_ext_pv_at_time(self, t_ms: float) -> float:
        return self.I_ext_pv()

    def I_ext_som(self) -> float:
        return (
            self.I0_som
            + self.act_alpha7_som * self.I_alpha7_som
            + self.act_beta2 * self.I_beta2_som
        )

    def I_ext_som_at_time(self, t_ms: float) -> float:
        return self.I_ext_som()

    def I_ext_vip(self) -> float:
        return self.I0_vip + self.act_alpha5 * self.I_alpha5_vip

    def I_ext_vip_at_time(self, t_ms: float) -> float:
        return self.I_ext_vip()

    def I_ext_ndnf(self) -> float:
        return (
            self.I0_ndnf
            + self.act_alpha7_ndnf * self.I_alpha7_ndnf
            + self.act_beta2 * self.I_beta2_ndnf
        )

    def I_ext_ndnf_at_time(self, t_ms: float) -> float:
        return self.I_ext_ndnf()


@dataclass(frozen=True)
class ParamBound:
    """Search bounds for a single parameter."""
    lo: float
    hi: float
    mode: Literal["lin", "log"] = "log"


def default_bounds(base: CircuitParams, w_hi: float | None = None) -> dict[str, ParamBound]:
    """Default search bounds for the 5-population model."""
    b: dict[str, ParamBound] = {}

    # --- Time constants (ms) ---
    b["tau_adapt_pyr"] = ParamBound(200.0, 1200.0, mode="log")
    b["tau_adapt_som"] = ParamBound(20.0, 300.0, mode="log")

    # --- Adaptation strengths (nA/Hz) ---
    b["J_adapt_pyr"] = ParamBound(0.001, 0.2, mode="log")
    b["J_adapt_som"] = ParamBound(0.001, 0.2, mode="lin")

    # --- GABA modulation ---
    b["g_gaba_base"] = ParamBound(0.1, 5.0, mode="lin")
    b["g_alpha7"]    = ParamBound(0.1, 5.0, mode="lin")

    # --- Synaptic weights (nA/Hz) ---
    _W_LO = 0.001
    _W_HI = w_hi if w_hi is not None else 0.01

    for w_name in (
        "w_ep", "w_es", "w_ev",
        "w_pe", "w_pp",
        "w_se", "w_sp",
        "w_vp", "w_vs",
        "w_ne", "w_np", "w_nv",
    ):
        b[w_name] = ParamBound(_W_LO, _W_HI, mode="log")

    # SOM->NDNF is the only brake on NDNF firing, so give it more headroom.
    b["w_sn"]   = ParamBound(_W_LO, 0.05, mode="log")
    b["J_NMDA"] = ParamBound(0.05, 2.0, mode="log")

    # --- External tonic drives (nA) ---
    b["I0_pyr"]  = ParamBound(0.01, 1.5, mode="lin")
    b["I0_pv"]   = ParamBound(0.01, 0.6, mode="lin")
    b["I0_som"]  = ParamBound(0.01, 0.6, mode="lin")
    b["I0_vip"]  = ParamBound(0.01, 0.6, mode="lin")
    # NDNF lacks an excitatory PYR input, so it would self-drive easily if I0_ndnf is large.
    # Tightened upper bound so the optimizer can actually reach the low (~2.5 Hz) NDNF target.
    b["I0_ndnf"] = ParamBound(0.01, 0.25, mode="lin")

    # --- nAChR cholinergic currents (nA) ---
    b["I_alpha7_pv"]   = ParamBound(0.01, 0.5, mode="lin")
    b["I_alpha7_som"]  = ParamBound(0.01, 0.5, mode="lin")
    b["I_beta2_som"]   = ParamBound(0.01, 0.5, mode="lin")
    b["I_alpha5_vip"]  = ParamBound(0.01, 0.5, mode="lin")
    b["I_alpha7_ndnf"] = ParamBound(0.01, 0.5, mode="lin")
    b["I_beta2_ndnf"]  = ParamBound(0.01, 0.5, mode="lin")

    return b
