#!/usr/bin/env python3
"""
circuit_model_parameter_search.py
Nevergrad-based optimization of a 4-population rate model (PYR, SOM, PV, VIP) to match
target mean firing rates. Optional "knockout" conditions (act_* = 0) can also be
included in the objective.
Equations:
  tau_s * dr/dt = -r + Phi(I_det) + sigma_s * xi(t)
Transfer function (Wong–Wang form):
  Phi(I) = u / (1 - exp(-g*u)),  u = c*(I - theta)
Requires: numpy, nevergrad
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Literal, Optional

import argparse
import json
import os

import nevergrad as ng
import numpy as np


NoiseType = Literal["none", "white", "ou"]


def phi_wong_wang(I: Any, *, theta: float, c: float, g: float) -> np.ndarray:
    """Wong–Wang transfer function with basic numerical safeguards."""
    if g <= 0:
        raise ValueError("g must be > 0")
    if c < 0:
        raise ValueError("c must be >= 0")

    I = np.asarray(I, dtype=float)
    u = c * (I - theta)
    z = g * u

    # denom = 1 - exp(-z) computed stably via expm1; cap to avoid overflow
    denom = -np.expm1(np.minimum(-z, 700.0))

    eps = 1e-8
    out = np.where(np.abs(z) < eps, 1.0 / g + u / 2.0, u / denom)
    return np.maximum(out, 0.0)


@dataclass(frozen=True)
class CircuitParams:
    # Time constants (ms)
    tau_s: float = 37.3479
    tau_adapt_pyr: float = 186.602
    tau_adapt_som: float = 2320.51

    # Adaptation strengths
    J_adapt_pyr: float = 0.270443
    J_adapt_som: float = 27.2356

    # Noise strength
    sigma_s: float = 5.88856

    # GABA scaling
    g_gaba_base: float = 3.93207
    g_alpha7: float = 0.95607

    # Synaptic weights (pre -> post)
    w_ee: float = 6.27108
    w_pe: float = 2.22239
    w_se: float = 2.61788

    w_ep: float = 42.5334
    w_pp: float = 105.44
    w_vp: float = 0.0105234
    w_sp: float = 6.12585e-06

    w_es: float = 6.56939
    w_vs: float = 1.27414

    w_ev: float = 2.9622e-06
    w_ps: float = 2.22239
    w_vv: float = 24.7962  # VIP -> VIP

    # External currents
    I0_pyr: float = 1.7854
    I_trans: float = 5.03758

    I0_pv: float = 5.58459
    I_alpha7_pv: float = 9.90322

    I0_som: float = 5.48551
    I_alpha7_som: float = 5.84835
    I_beta2_som: float = 9.05679

    I0_vip: float = 7.57337
    I_alpha5_vip: float = 1.44659

    # Receptor activation multipliers (used for KO conditions)
    act_alpha7: float = 1.0
    act_beta2: float = 1.0
    act_alpha5: float = 1.0

    # Transfer parameters
    Theta_pyr: float = 5.01691
    alpha_pyr: float = 0.685403

    Theta_pv: float = 16.3771
    alpha_pv: float = 1.47638

    Theta_som: float = 5.88155
    alpha_som: float = 0.817185

    Theta_vip: float = 13.9068
    alpha_vip: float = 0.100998

    g_e: float = 0.377039
    g_i: float = 0.400125

    def g_gaba(self) -> float:
        return self.g_gaba_base + self.g_alpha7

    def I_ext_pyr(self) -> float:
        return self.I0_pyr + self.I_trans

    def I_ext_pv(self) -> float:
        return self.I0_pv + self.act_alpha7 * self.I_alpha7_pv

    def I_ext_som(self) -> float:
        return (
            self.I0_som
            + self.act_alpha7 * self.I_alpha7_som
            + self.act_beta2 * self.I_beta2_som
        )

    def I_ext_vip(self) -> float:
        return self.I0_vip + self.act_alpha5 * self.I_alpha5_vip


@dataclass
class SimulationResult:
    t_ms: np.ndarray
    r: np.ndarray        # (n_steps, 4): [pyr, som, pv, vip]
    I_adapt: np.ndarray  # (n_steps, 2): [pyr, som]


def simulate_circuit(
    params: CircuitParams,
    T_ms: float,
    dt_ms: float = 0.1,
    r0: Optional[np.ndarray] = None,
    I_adapt0: Optional[np.ndarray] = None,
    *,
    seed: Optional[int] = None,
    noise_type: NoiseType = "none",
    tau_noise_ms: float = 5.0,
) -> SimulationResult:
    if T_ms <= 0 or dt_ms <= 0:
        raise ValueError("T_ms and dt_ms must be > 0")

    n_steps = int(np.floor(T_ms / dt_ms)) + 1
    t = np.linspace(0.0, dt_ms * (n_steps - 1), n_steps, dtype=float)

    r = np.zeros((n_steps, 4), dtype=float)
    I_adapt = np.zeros((n_steps, 2), dtype=float)

    if r0 is None:
        r[0] = np.array([0.1, 0.1, 0.1, 0.1], dtype=float)
    else:
        r0 = np.asarray(r0, dtype=float)
        if r0.shape != (4,):
            raise ValueError("r0 must have shape (4,)")
        r[0] = r0

    if I_adapt0 is None:
        I_adapt[0] = 0.0
    else:
        I_adapt0 = np.asarray(I_adapt0, dtype=float)
        if I_adapt0.shape != (2,):
            raise ValueError("I_adapt0 must have shape (2,)")
        I_adapt[0] = I_adapt0

    rng = np.random.default_rng(seed)
    xi_state = np.zeros(4, dtype=float)

    ggaba = params.g_gaba()

    for k in range(n_steps - 1):
        r_pyr, r_som, r_pv, r_vip = r[k]
        Iap, Ias = I_adapt[k]

        # xi(t)
        if params.sigma_s == 0.0 or noise_type == "none":
            xi = np.zeros(4, dtype=float)
        elif noise_type == "white":
            xi = rng.standard_normal(4)
        elif noise_type == "ou":
            if tau_noise_ms <= 0:
                raise ValueError("tau_noise_ms must be > 0 for OU noise")
            xi_state += (-xi_state / tau_noise_ms) * dt_ms + np.sqrt(
                2.0 * dt_ms / tau_noise_ms
            ) * rng.standard_normal(4)
            xi = xi_state
        else:
            raise ValueError(f"Unknown noise_type: {noise_type!r}")

        # Deterministic inputs (argument to Phi)
        denom = 1.0 + ggaba * params.w_pe * r_pv
        I_pyr = (params.w_ee * r_pyr) / denom - ggaba * params.w_se * r_som - Iap + params.I_ext_pyr()
        I_som = params.w_es * r_pyr - ggaba * params.w_ps * r_pv - params.w_vs * r_vip - Ias + params.I_ext_som()
        I_pv = (
            params.w_ep * r_pyr
            - ggaba * params.w_pp * r_pv
            - ggaba * params.w_sp * r_som
            - params.w_vp * r_vip
            + params.I_ext_pv()
        )
        I_vip = params.w_ev * r_pyr - params.w_vv * r_vip + params.I_ext_vip()

        Phi = np.array(
            [
                phi_wong_wang(I_pyr, theta=params.Theta_pyr, c=params.alpha_pyr, g=params.g_e).item(),
                phi_wong_wang(I_som, theta=params.Theta_som, c=params.alpha_som, g=params.g_i).item(),
                phi_wong_wang(I_pv, theta=params.Theta_pv, c=params.alpha_pv, g=params.g_i).item(),
                phi_wong_wang(I_vip, theta=params.Theta_vip, c=params.alpha_vip, g=params.g_i).item(),
            ],
            dtype=float,
        )

        # Rate update (noise after Phi)
        dr = (-r[k] + Phi + params.sigma_s * xi) / params.tau_s
        r[k + 1] = np.maximum(r[k] + dt_ms * dr, 0.0)

        # Adaptation (PYR + SOM)
        dIap = (-Iap + params.J_adapt_pyr * r_pyr) / params.tau_adapt_pyr
        dIas = (-Ias + params.J_adapt_som * r_som) / params.tau_adapt_som
        I_adapt[k + 1, 0] = Iap + dt_ms * dIap
        I_adapt[k + 1, 1] = Ias + dt_ms * dIas

    return SimulationResult(t_ms=t, r=r, I_adapt=I_adapt)


def mean_rates(result: SimulationResult, burn_in_ms: float, window_ms: float) -> np.ndarray:
    dt = float(result.t_ms[1] - result.t_ms[0])
    start = int(np.floor(burn_in_ms / dt))

    if window_ms <= 0:
        rr = result.r[start:]
    else:
        end = result.r.shape[0]
        window_steps = int(np.floor(window_ms / dt))
        rr = result.r[max(start, end - window_steps) : end]

    return np.mean(rr, axis=0)


@dataclass(frozen=True)
class TargetRates:
    mean_r_pyr: float
    mean_r_som: float
    mean_r_pv: float
    mean_r_vip: float

    # Optional knockout targets
    alpha7_ko_pyr: Optional[float] = None
    alpha5_ko_pyr: Optional[float] = None
    beta2_ko_pyr: Optional[float] = None

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.mean_r_pyr, self.mean_r_som, self.mean_r_pv, self.mean_r_vip],
            dtype=float,
        )


@dataclass(frozen=True)
class FitConfig:
    T_ms: float = 2500.0
    dt_ms: float = 0.1
    burn_in_ms: float = 1800.0
    window_ms: float = 500.0

    n_trials: int = 8
    init_rate_scale: float = 0.2

    noise_type: NoiseType = "none"
    tau_noise_ms: float = 5.0

    max_rate: float = 200.0

    ko_min_effect_penalty: float = 5.0
    ko_wrong_direction_penalty: float = 10.0


@dataclass(frozen=True)
class ParamBound:
    lo: float
    hi: float
    mode: Literal["lin", "log"] = "log"


def default_bounds(base: CircuitParams) -> dict[str, ParamBound]:
    b: dict[str, ParamBound] = {}

    b["tau_s"] = ParamBound(5.0, 100.0, mode="log")
    b["tau_adapt_pyr"] = ParamBound(50.0, 5000.0, mode="log")
    b["tau_adapt_som"] = ParamBound(50.0, 5000.0, mode="log")

    b["J_adapt_pyr"] = ParamBound(0.0, 50.0, mode="lin")
    b["J_adapt_som"] = ParamBound(0.0, 50.0, mode="lin")

    b["sigma_s"] = ParamBound(0.0, 10.0, mode="lin")
    b["g_gaba_base"] = ParamBound(0.0, 5.0, mode="lin")
    b["g_alpha7"] = ParamBound(0.0, 5.0, mode="lin")

    def w_range(x: float, *, min_val: float = 1e-6) -> ParamBound:
        hi = max(1e-6, 5.0 * x)
        lo = min_val if x > 0 else 0.0
        return ParamBound(lo, hi, mode="log")

    for name in ["w_ee", "w_pe", "w_ep", "w_pp", "w_vp", "w_sp", "w_ev"]:
        b[name] = w_range(getattr(base, name))

    # Keep a few weights away from zero to avoid KO-insensitive solutions.
    b["w_se"] = w_range(base.w_se, min_val=0.1)
    b["w_es"] = w_range(base.w_es, min_val=0.5)
    b["w_vs"] = w_range(base.w_vs, min_val=0.5)

    b["w_ps"] = ParamBound(0.0, 5.0 * base.w_pe, mode="log")
    b["w_vv"] = ParamBound(0.0, 5.0 * max(base.w_vv, 1.0), mode="log")

    b["I0_pyr"] = ParamBound(0.0, 10.0, mode="lin")
    b["I_trans"] = ParamBound(0.0, 10.0, mode="lin")

    b["I0_pv"] = ParamBound(0.0, 15.0, mode="lin")
    b["I_alpha7_pv"] = ParamBound(0.0, 10.0, mode="lin")

    b["I0_som"] = ParamBound(0.0, 10.0, mode="lin")
    b["I_alpha7_som"] = ParamBound(0.0, 10.0, mode="lin")
    b["I_beta2_som"] = ParamBound(0.0, 10.0, mode="lin")

    b["I0_vip"] = ParamBound(0.0, 10.0, mode="lin")
    b["I_alpha5_vip"] = ParamBound(0.0, 10.0, mode="lin")

    for name in ["Theta_pyr", "Theta_pv", "Theta_som", "Theta_vip"]:
        b[name] = ParamBound(0.0, 20.0, mode="lin")
    for name in ["alpha_pyr", "alpha_pv", "alpha_som", "alpha_vip"]:
        b[name] = ParamBound(0.05, 10.0, mode="log")

    b["g_e"] = ParamBound(0.1, 10.0, mode="log")
    b["g_i"] = ParamBound(0.1, 10.0, mode="log")

    return b


def build_nevergrad_parametrization(
    base: CircuitParams,
    bounds: dict[str, ParamBound],
    freeze: Optional[set[str]] = None,
) -> ng.p.Dict:
    freeze = freeze or set()
    params_dict: dict[str, Any] = {}

    for f in fields(CircuitParams):
        name = f.name

        if name in freeze or name not in bounds:
            params_dict[name] = getattr(base, name)
            continue

        bound = bounds[name]
        if bound.mode == "log" and bound.lo > 0:
            params_dict[name] = ng.p.Log(lower=bound.lo, upper=bound.hi)
        else:
            params_dict[name] = ng.p.Scalar(lower=bound.lo, upper=bound.hi)

    return ng.p.Dict(**params_dict)


def params_from_ng_dict(ng_dict: dict[str, Any], base: CircuitParams) -> CircuitParams:
    allowed = {f.name for f in fields(CircuitParams)}
    clean = {k: v for k, v in ng_dict.items() if k in allowed}
    return replace(base, **clean)


def loss_from_means(
    means: np.ndarray,
    target: TargetRates,
    *,
    near_zero_threshold: float = 0.1,
    near_zero_weight: float = 10.0,
) -> float:
    tgt = target.as_array()
    denom = np.maximum(np.abs(tgt), 1e-3)
    rel = (means - tgt) / denom
    mse = float(np.mean(rel**2))

    below = np.maximum(near_zero_threshold - means, 0.0)
    near_zero = float(np.sum((below / near_zero_threshold) ** 2))
    return mse + near_zero_weight * near_zero


def loss_from_ko_pyr(
    pyr_mean: float,
    target_pyr: float,
    base_pyr: float,
    *,
    near_zero_threshold: float = 0.1,
    near_zero_weight: float = 10.0,
    min_effect_weight: float = 5.0,
    wrong_direction_weight: float = 10.0,
) -> float:
    denom = max(abs(target_pyr), 1e-3)
    mse = ((pyr_mean - target_pyr) / denom) ** 2

    below = max(near_zero_threshold - pyr_mean, 0.0)
    near_zero = (below / near_zero_threshold) ** 2

    expected = target_pyr - base_pyr
    actual = pyr_mean - base_pyr
    exp_mag = abs(expected)
    act_mag = abs(actual)

    min_effect = 0.0
    wrong_dir = 0.0
    if exp_mag > 0.1:
        ratio = act_mag / exp_mag
        min_effect = max(0.0, 1.0 - ratio) ** 2

        same_sign = (expected > 0 and actual > 0) or (expected < 0 and actual < 0) or act_mag < 0.01
        if not same_sign:
            wrong_dir = (act_mag / exp_mag) ** 2

    return mse + near_zero_weight * near_zero + min_effect_weight * min_effect + wrong_direction_weight * wrong_dir


def run_trials(params: CircuitParams, cfg: FitConfig, base_seed: int) -> tuple[bool, np.ndarray]:
    rng = np.random.default_rng(base_seed)
    means_trials: list[np.ndarray] = []

    for _ in range(cfg.n_trials):
        r0 = cfg.init_rate_scale * rng.lognormal(mean=0.0, sigma=0.6, size=4)
        seed = int(rng.integers(0, 2**31 - 1))

        res = simulate_circuit(
            params,
            T_ms=cfg.T_ms,
            dt_ms=cfg.dt_ms,
            r0=r0,
            seed=seed,
            noise_type=cfg.noise_type,
            tau_noise_ms=cfg.tau_noise_ms,
        )
        m = mean_rates(res, burn_in_ms=cfg.burn_in_ms, window_ms=cfg.window_ms)

        if not np.all(np.isfinite(m)) or np.any(m > cfg.max_rate):
            return False, m

        means_trials.append(m)

    means = np.mean(np.stack(means_trials, axis=0), axis=0)
    return True, means


@dataclass
class KOMeans:
    alpha7_ko: Optional[np.ndarray] = None
    alpha5_ko: Optional[np.ndarray] = None
    beta2_ko: Optional[np.ndarray] = None


ConditionResult = tuple[str, bool, np.ndarray]  # (name, ok, means)


def run_condition(args: tuple[str, CircuitParams, FitConfig, int]) -> ConditionResult:
    name, params, cfg, seed = args
    ok, means = run_trials(params, cfg, seed)
    return name, ok, means


def evaluate_params(
    params: CircuitParams,
    target: TargetRates,
    cfg: FitConfig,
    *,
    rng: np.random.Generator,
    executor: Optional[ProcessPoolExecutor] = None,
) -> tuple[float, np.ndarray, KOMeans]:
    conditions: list[tuple[str, CircuitParams, FitConfig, int]] = [
        ("base", params, cfg, int(rng.integers(0, 2**31 - 1))),
    ]

    # α7 KO: remove α7-mediated currents AND α7-dependent GABA scaling (set g_alpha7 -> 0).
    if target.alpha7_ko_pyr is not None:
        conditions.append(
            (
                "alpha7_ko",
                replace(params, act_alpha7=0.0, g_alpha7=0.0),
                cfg,
                int(rng.integers(0, 2**31 - 1)),
            )
        )
    if target.alpha5_ko_pyr is not None:
        conditions.append(("alpha5_ko", replace(params, act_alpha5=0.0), cfg, int(rng.integers(0, 2**31 - 1))))
    if target.beta2_ko_pyr is not None:
        conditions.append(("beta2_ko", replace(params, act_beta2=0.0), cfg, int(rng.integers(0, 2**31 - 1))))

    if executor is not None and len(conditions) > 1:
        results = list(executor.map(run_condition, conditions))
    else:
        results = [run_condition(c) for c in conditions]

    ko_means = KOMeans()
    base_means = np.zeros(4, dtype=float)

    for name, ok, means in results:
        if not ok:
            return 1e9, base_means, ko_means
        if name == "base":
            base_means = means
        elif name == "alpha7_ko":
            ko_means.alpha7_ko = means
        elif name == "alpha5_ko":
            ko_means.alpha5_ko = means
        elif name == "beta2_ko":
            ko_means.beta2_ko = means

    total = loss_from_means(base_means, target)
    base_pyr = float(base_means[0])

    if target.alpha7_ko_pyr is not None and ko_means.alpha7_ko is not None:
        total += loss_from_ko_pyr(
            float(ko_means.alpha7_ko[0]),
            target.alpha7_ko_pyr,
            base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )
    if target.alpha5_ko_pyr is not None and ko_means.alpha5_ko is not None:
        total += loss_from_ko_pyr(
            float(ko_means.alpha5_ko[0]),
            target.alpha5_ko_pyr,
            base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )
    if target.beta2_ko_pyr is not None and ko_means.beta2_ko is not None:
        total += loss_from_ko_pyr(
            float(ko_means.beta2_ko[0]),
            target.beta2_ko_pyr,
            base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )

    return total, base_means, ko_means


@dataclass(frozen=True)
class Candidate:
    loss: float
    means: np.ndarray
    ko_means: KOMeans
    params: CircuitParams


def format_params_as_code(params: CircuitParams) -> str:
    lines = ["CircuitParams("]
    for f in fields(CircuitParams):
        name = f.name
        val = getattr(params, name)
        if isinstance(val, float):
            lines.append(f"    {name}={val:.6g},")
        else:
            lines.append(f"    {name}={val!r},")
    lines.append(")")
    return "\n".join(lines)


def log_best_result(path: str, step: int, cand: Candidate, target: TargetRates) -> None:
    entry = {
        "step": step,
        "loss": cand.loss,
        "target": asdict(target),
        "means": {
            "pyr": float(cand.means[0]),
            "som": float(cand.means[1]),
            "pv": float(cand.means[2]),
            "vip": float(cand.means[3]),
        },
        "ko_means": {
            "alpha7_ko": cand.ko_means.alpha7_ko.tolist() if cand.ko_means.alpha7_ko is not None else None,
            "alpha5_ko": cand.ko_means.alpha5_ko.tolist() if cand.ko_means.alpha5_ko is not None else None,
            "beta2_ko": cand.ko_means.beta2_ko.tolist() if cand.ko_means.beta2_ko is not None else None,
        },
        "params": asdict(cand.params),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_params_json(path: str) -> CircuitParams:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    base = CircuitParams()
    allowed = {fld.name for fld in fields(CircuitParams)}
    clean = {k: d[k] for k in d if k in allowed}
    return replace(base, **clean)


def save_params_json(path: str, params: CircuitParams) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=2, sort_keys=True)


def parse_freeze_list(s: str) -> set[str]:
    return {x.strip() for x in s.split(",") if x.strip()}


def nevergrad_optimize(
    target: TargetRates,
    *,
    base: CircuitParams,
    bounds: dict[str, ParamBound],
    fit_cfg: FitConfig,
    n_samples: int,
    top_k: int,
    seed: Optional[int],
    freeze: Optional[set[str]] = None,
    early_stop_loss: Optional[float] = 1e-4,
    log_file: Optional[str] = None,
    log_interval: int = 50,
    n_workers: Optional[int] = None,
) -> list[Candidate]:
    rng = np.random.default_rng(seed)

    parametrization = build_nevergrad_parametrization(base, bounds, freeze)
    optimizer = ng.optimizers.TwoPointsDE(
        parametrization=parametrization,
        budget=n_samples,
        num_workers=1,
    )

    if seed is not None:
        optimizer.parametrization.random_state = np.random.RandomState(seed)

    n_conditions = 1 + sum(
        [
            target.alpha7_ko_pyr is not None,
            target.alpha5_ko_pyr is not None,
            target.beta2_ko_pyr is not None,
        ]
    )
    use_parallel = n_conditions > 1 and (n_workers is None or n_workers not in (0, 1))

    if n_workers is None:
        max_workers = min(n_conditions, os.cpu_count() or 4)
    else:
        max_workers = min(n_conditions, n_workers)

    if log_file:
        open(log_file, "w", encoding="utf-8").close()

    pool_cm = ProcessPoolExecutor(max_workers=max_workers) if use_parallel else nullcontext(None)

    best: list[Candidate] = []

    with pool_cm as executor:
        if use_parallel:
            print(f"Using {max_workers} workers for {n_conditions} conditions")

        last_step = 0
        stopped_early = False

        for step in range(1, n_samples + 1):
            last_step = step
            x = optimizer.ask()
            params = params_from_ng_dict(x.value, base)

            L, means, ko_means = evaluate_params(params, target, fit_cfg, rng=rng, executor=executor)
            optimizer.tell(x, L)

            cand = Candidate(loss=L, means=means, ko_means=ko_means, params=params)

            ko_str = ""
            if ko_means.alpha7_ko is not None:
                ko_str += f" a7KO_pyr={ko_means.alpha7_ko[0]:.4g}"
            if ko_means.alpha5_ko is not None:
                ko_str += f" a5KO_pyr={ko_means.alpha5_ko[0]:.4g}"
            if ko_means.beta2_ko is not None:
                ko_str += f" b2KO_pyr={ko_means.beta2_ko[0]:.4g}"

            print(
                f"[{step}/{n_samples}] loss={L:.6g} "
                f"means=[pyr={means[0]:.4g}, som={means[1]:.4g}, pv={means[2]:.4g}, vip={means[3]:.4g}]"
                f"{ko_str}"
            )

            if len(best) < top_k:
                best.append(cand)
                best.sort(key=lambda c: c.loss)
            elif L < best[-1].loss:
                best[-1] = cand
                best.sort(key=lambda c: c.loss)

            if log_file and step % log_interval == 0 and best:
                log_best_result(log_file, step, best[0], target)

            if early_stop_loss is not None and best and best[0].loss <= early_stop_loss:
                if log_file:
                    log_best_result(log_file, step, best[0], target)
                stopped_early = True
                break

        if log_file and best and (not stopped_early) and last_step % log_interval != 0:
            log_best_result(log_file, last_step, best[0], target)

    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Optimize circuit parameters to match target mean rates.")
    p.add_argument("--target_pyr", type=float, required=True)
    p.add_argument("--target_som", type=float, required=True)
    p.add_argument("--target_pv", type=float, required=True)
    p.add_argument("--target_vip", type=float, required=True)

    p.add_argument("--target_alpha7_ko_pyr", type=float, default=None)
    p.add_argument("--target_alpha5_ko_pyr", type=float, default=None)
    p.add_argument("--target_beta2_ko_pyr", type=float, default=None)

    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--early_stop_loss", type=float, default=1e-4)

    p.add_argument("--T_ms", type=float, default=2500.0)
    p.add_argument("--dt_ms", type=float, default=0.1)
    p.add_argument("--burn_in_ms", type=float, default=1800.0)
    p.add_argument("--window_ms", type=float, default=500.0)
    p.add_argument("--n_trials", type=int, default=8)
    p.add_argument("--init_rate_scale", type=float, default=0.2)
    p.add_argument("--max_rate", type=float, default=200.0)

    p.add_argument("--noise_type", choices=["none", "white", "ou"], default="none")
    p.add_argument("--tau_noise_ms", type=float, default=5.0)

    p.add_argument("--ko_min_effect_penalty", type=float, default=5.0)
    p.add_argument("--ko_wrong_direction_penalty", type=float, default=10.0)

    p.add_argument("--base_params_json", type=str, default="")
    p.add_argument("--freeze", type=str, default="")
    p.add_argument("--save_best_json", type=str, default="")
    p.add_argument("--log_file", type=str, default="")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--n_workers", type=int, default=None)

    args = p.parse_args()

    target = TargetRates(
        mean_r_pyr=args.target_pyr,
        mean_r_som=args.target_som,
        mean_r_pv=args.target_pv,
        mean_r_vip=args.target_vip,
        alpha7_ko_pyr=args.target_alpha7_ko_pyr,
        alpha5_ko_pyr=args.target_alpha5_ko_pyr,
        beta2_ko_pyr=args.target_beta2_ko_pyr,
    )

    base = load_params_json(args.base_params_json) if args.base_params_json else CircuitParams()
    bounds = default_bounds(base)

    fit_cfg = FitConfig(
        T_ms=args.T_ms,
        dt_ms=args.dt_ms,
        burn_in_ms=args.burn_in_ms,
        window_ms=args.window_ms,
        n_trials=args.n_trials,
        init_rate_scale=args.init_rate_scale,
        noise_type=args.noise_type,
        tau_noise_ms=args.tau_noise_ms,
        max_rate=args.max_rate,
        ko_min_effect_penalty=args.ko_min_effect_penalty,
        ko_wrong_direction_penalty=args.ko_wrong_direction_penalty,
    )

    freeze = parse_freeze_list(args.freeze)

    best = nevergrad_optimize(
        target,
        base=base,
        bounds=bounds,
        fit_cfg=fit_cfg,
        n_samples=args.n_samples,
        top_k=args.top_k,
        seed=args.seed,
        freeze=freeze,
        early_stop_loss=args.early_stop_loss,
        log_file=args.log_file or None,
        log_interval=args.log_interval,
        n_workers=args.n_workers,
    )

    if not best:
        raise RuntimeError("Optimization returned no candidates.")

    print("\n" + "=" * 60)
    print("TOP RESULTS")
    print("=" * 60)
    for i, c in enumerate(best, start=1):
        pyr, som, pv, vip = c.means.tolist()
        ko_str = ""
        if c.ko_means.alpha7_ko is not None:
            ko_str += f" a7KO_pyr={c.ko_means.alpha7_ko[0]:.4g}"
        if c.ko_means.alpha5_ko is not None:
            ko_str += f" a5KO_pyr={c.ko_means.alpha5_ko[0]:.4g}"
        if c.ko_means.beta2_ko is not None:
            ko_str += f" b2KO_pyr={c.ko_means.beta2_ko[0]:.4g}"
        print(
            f"rank {i:02d}: loss={c.loss:.3e} "
            f"means=[pyr={pyr:.4g}, som={som:.4g}, pv={pv:.4g}, vip={vip:.4g}]"
            f"{ko_str}"
        )

    print("\nBest parameter set:\n")
    print(format_params_as_code(best[0].params))

    if args.save_best_json:
        save_params_json(args.save_best_json, best[0].params)
        print(f"\nSaved best params to: {args.save_best_json}")


if __name__ == "__main__":
    main()