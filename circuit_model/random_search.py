"""
Random parameter search utilities for bistable regime discovery.

This module samples parameter sets from existing bounds, evaluates bistability
with the nullcline-based tools, then simulates low/high states for bistable
hits and logs results to JSONL.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional
import json
import os

import numpy as np
from tqdm import tqdm

from .bistable_loss import BistableConfig, bistable_loss
from .params import CircuitParams, ParamBound
from .simulation import NoiseType, mean_rates, simulate_circuit


@dataclass(frozen=True)
class RandomBistableSearchConfig:
    """Configuration for random bistable search."""

    n_samples: int = 100000
    seed: int = 0
    show_every: int = 1000
    output_jsonl: str = "figs/optim/random_bistable_hits.jsonl"
    summary_txt: str = "figs/optim/random_bistable_summary.txt"
    append: bool = False
    max_hits: Optional[int] = None
    n_workers: Optional[int] = 10

    # Simulation settings for validating low/high fixed-point states.
    sim_T_ms: float = 2500.0
    sim_dt_ms: float = 0.1
    sim_burn_in_ms: float = 1800.0
    sim_window_ms: float = 500.0
    sim_noise_type: NoiseType = "none"
    sim_tau_noise_ms: float = 5.0


# Module-level storage for multiprocessing worker initialization
_mp_args: Optional[tuple] = None


def _sample_param(bound: ParamBound, rng: np.random.Generator) -> float:
    """Sample one parameter value according to bound mode."""
    if bound.mode == "log" and bound.lo > 0.0:
        lo = float(np.log(bound.lo))
        hi = float(np.log(bound.hi))
        return float(np.exp(rng.uniform(lo, hi)))
    return float(rng.uniform(bound.lo, bound.hi))


def _sample_params(
    base: CircuitParams,
    bounds: dict[str, ParamBound],
    freeze: set[str],
    rng: np.random.Generator,
) -> CircuitParams:
    """Sample a full CircuitParams by replacing unfrozen bounded fields."""
    updates: dict[str, float] = {}
    for name, bound in bounds.items():
        if name in freeze:
            continue
        updates[name] = _sample_param(bound, rng)
    return replace(base, **updates)


def _simulate_state_rates(
    params: CircuitParams,
    r0: np.ndarray,
    cfg: RandomBistableSearchConfig,
    seed: int,
) -> np.ndarray:
    """Simulate one state and return mean firing rates [PYR, SOM, PV, VIP]."""
    res = simulate_circuit(
        params,
        T_ms=cfg.sim_T_ms,
        dt_ms=cfg.sim_dt_ms,
        r0=r0,
        seed=seed,
        noise_type=cfg.sim_noise_type,
        tau_noise_ms=cfg.sim_tau_noise_ms,
        use_transient=False,
    )
    return mean_rates(res, burn_in_ms=cfg.sim_burn_in_ms, window_ms=cfg.sim_window_ms)


def _init_worker(
    base: CircuitParams,
    bounds: dict[str, ParamBound],
    freeze: set[str],
    bistable_cfg: BistableConfig,
    search_cfg: RandomBistableSearchConfig,
) -> None:
    """Initialize worker process with shared configuration."""
    global _mp_args
    _mp_args = (base, bounds, freeze, bistable_cfg, search_cfg)


def _evaluate_sample(
    sample_idx: int,
    rng_seed: int,
) -> tuple[int, Optional[dict]]:
    """
    Evaluate a single parameter sample for bistability.
    
    Returns (sample_idx, result_dict) where result_dict is None if not bistable,
    or a dict with all hit data if bistable. Used by ProcessPoolExecutor workers.
    """
    global _mp_args
    if _mp_args is None:
        raise RuntimeError("Worker not initialized; _mp_args is None")
    
    base, bounds, freeze, bistable_cfg, search_cfg = _mp_args
    
    rng = np.random.default_rng(rng_seed)
    params_i = _sample_params(base, bounds, freeze, rng)
    
    try:
        _, comp = bistable_loss(params_i, bistable_cfg, return_components=True)
    except Exception:
        return sample_idx, None
    
    is_bistable = comp.get("n_stable", 0) >= 2
    if not is_bistable:
        return sample_idx, None
    
    r_high = comp.get("r_high_fp")
    som_high = comp.get("r_som_high_fp")
    pv_high = comp.get("r_pv_high_fp")
    vip_high = comp.get("r_vip_high_fp")
    if r_high is None or som_high is None or pv_high is None or vip_high is None:
        return sample_idx, None
    
    low_r0 = np.array([
        float(comp["r_low_fp"]),
        float(comp["r_som_fp"]),
        float(comp["r_pv_fp"]),
        float(comp["r_vip_fp"]),
    ], dtype=float)
    high_r0 = np.array([
        float(r_high),
        float(som_high),
        float(pv_high),
        float(vip_high),
    ], dtype=float)
    
    low_seed = int(rng.integers(0, 2**31 - 1))
    high_seed = int(rng.integers(0, 2**31 - 1))
    low_means = _simulate_state_rates(params_i, low_r0, search_cfg, low_seed)
    high_means = _simulate_state_rates(params_i, high_r0, search_cfg, high_seed)
    
    result_dict = {
        "sample_index": sample_idx,
        "seed": search_cfg.seed,
        "fixed_points_hz": {
            "low": {
                "pyr": float(comp["r_low_fp"]),
                "som": float(comp["r_som_fp"]),
                "pv": float(comp["r_pv_fp"]),
                "vip": float(comp["r_vip_fp"]),
            },
            "high": {
                "pyr": float(r_high),
                "som": float(som_high),
                "pv": float(pv_high),
                "vip": float(vip_high),
            },
        },
        "simulated_means_hz": {
            "low_state": {
                "pyr": float(low_means[0]),
                "som": float(low_means[1]),
                "pv": float(low_means[2]),
                "vip": float(low_means[3]),
            },
            "high_state": {
                "pyr": float(high_means[0]),
                "som": float(high_means[1]),
                "pv": float(high_means[2]),
                "vip": float(high_means[3]),
            },
        },
        "bistable_components": {
            "L_total": float(comp.get("L_total", 0.0)),
            "L_bistab": float(comp.get("L_bistab", 0.0)),
            "L_rate": float(comp.get("L_rate", 0.0)),
            "L_rate_high": float(comp.get("L_rate_high", 0.0)),
            "L_margin": float(comp.get("L_margin", 0.0)),
            "L_jac": float(comp.get("L_jac", 0.0)),
            "L_peak": float(comp.get("L_peak", 0.0)),
            "n_stable": int(comp.get("n_stable", 0)),
            "n_unstable": int(comp.get("n_unstable", 0)),
            "n_spurious": int(comp.get("n_spurious", 0)),
        },
        "params": asdict(params_i),
    }
    
    return sample_idx, result_dict


def run_random_bistable_search(
    *,
    base: CircuitParams,
    bounds: dict[str, ParamBound],
    freeze: set[str],
    bistable_cfg: BistableConfig,
    search_cfg: RandomBistableSearchConfig,
) -> dict[str, float | int | str]:
    """
    Run random search and log only bistable hits.

    Supports both serial and parallel (multiprocessing) evaluation.
    Uses tqdm for progress tracking with ETA.

    Returns a compact summary dictionary with counts and output paths.
    """
    if search_cfg.n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    out_path = Path(search_cfg.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(search_cfg.summary_txt)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(search_cfg.seed)
    mode = "a" if search_cfg.append else "w"

    n_hits = 0
    n_errors = 0
    n_completed = 0

    # Determine effective worker count
    n_workers = search_cfg.n_workers
    if n_workers is None:
        n_workers = 10
    n_workers = min(n_workers, search_cfg.n_samples, os.cpu_count() or 4)

    # Generate seeds for all samples upfront (consistent across serial/parallel)
    sample_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(search_cfg.n_samples)]

    with open(out_path, mode, encoding="utf-8") as fout:
        # Use multiprocessing if workers > 1 and samples > 1
        if n_workers > 1 and search_cfg.n_samples > 1:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(base, bounds, freeze, bistable_cfg, search_cfg),
            ) as executor:
                # Submit all jobs
                futures = {
                    executor.submit(_evaluate_sample, i + 1, sample_seeds[i]): i + 1
                    for i in range(search_cfg.n_samples)
                }

                with tqdm(
                    total=search_cfg.n_samples,
                    desc="Bistable search",
                    unit="sample",
                    dynamic_ncols=True,
                ) as pbar:
                    for future in as_completed(futures):
                        sample_idx, result_dict = future.result()
                        n_completed += 1

                        if result_dict is not None:
                            # Bistable hit: update hit index and write
                            n_hits += 1
                            result_dict["hit_index"] = n_hits
                            fout.write(json.dumps(result_dict) + "\n")
                        else:
                            n_errors += 1

                        pbar.update(1)
                        pbar.set_postfix({"hits": n_hits, "errors": n_errors})

                        # Early stop if max_hits reached
                        if search_cfg.max_hits is not None and n_hits >= search_cfg.max_hits:
                            pbar.close()
                            executor.shutdown(wait=False)
                            break
        else:
            # Serial execution (workers=1 or n_samples=1)
            with tqdm(
                total=search_cfg.n_samples,
                desc="Bistable search",
                unit="sample",
                dynamic_ncols=True,
            ) as pbar:
                for sample_idx in range(1, search_cfg.n_samples + 1):
                    sample_idx_ret, result_dict = _evaluate_sample(sample_idx, sample_seeds[sample_idx - 1])
                    n_completed += 1

                    if result_dict is not None:
                        n_hits += 1
                        result_dict["hit_index"] = n_hits
                        fout.write(json.dumps(result_dict) + "\n")
                    else:
                        n_errors += 1

                    pbar.update(1)
                    pbar.set_postfix({"hits": n_hits, "errors": n_errors})

                    if search_cfg.max_hits is not None and n_hits >= search_cfg.max_hits:
                        tqdm.write(f"Reached max_hits={search_cfg.max_hits}; stopping early.")
                        break

    summary_lines = [
        "=" * 70,
        "RANDOM BISTABLE SEARCH SUMMARY",
        "=" * 70,
        f"samples_evaluated: {n_completed}",
        f"samples_requested: {search_cfg.n_samples}",
        f"seed: {search_cfg.seed}",
        f"workers: {n_workers}",
        f"bistable_hits: {n_hits}",
        f"evaluation_errors: {n_errors}",
        f"hit_rate: {100.0 * n_hits / max(n_completed, 1):.6f}%",
        f"hits_jsonl: {out_path}",
        f"summary_txt: {summary_path}",
        f"simulation: T_ms={search_cfg.sim_T_ms}, dt_ms={search_cfg.sim_dt_ms}, "
        f"burn_in_ms={search_cfg.sim_burn_in_ms}, window_ms={search_cfg.sim_window_ms}, "
        f"noise={search_cfg.sim_noise_type}",
        "=" * 70,
    ]

    with open(summary_path, "w", encoding="utf-8") as fsum:
        fsum.write("\n".join(summary_lines) + "\n")

    return {
        "samples_requested": int(search_cfg.n_samples),
        "samples_evaluated": int(n_completed),
        "bistable_hits": int(n_hits),
        "evaluation_errors": int(n_errors),
        "hit_rate_pct": float(100.0 * n_hits / max(n_completed, 1)),
        "workers_used": int(n_workers),
        "hits_jsonl": str(out_path),
        "summary_txt": str(summary_path),
    }
