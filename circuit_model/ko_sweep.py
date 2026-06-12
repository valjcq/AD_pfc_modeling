"""
Combinatorial knockout sweep.

Simulates every combination of the three global nicotinic-receptor knockouts
(α7, α5, β2) at a fixed noise level and produces box plots of the firing-rate
distribution for all five populations.

The 2³ = 8 combinations (WT + 7 KOs) are split into two groups:

  - **target**     : KO conditions that entered the optimization loss
                     (α7, α5, β2 single KOs and the α7β2 double KO).
  - **prediction** : KO combinations the model was *not* fit to
                     (α7α5, α5β2, α7α5β2) — genuine model predictions.

WT is shown as a neutral baseline. The target/prediction split is auto-detected
from the fit's ``log.jsonl`` (its ``target`` entry) when available, otherwise it
falls back to the canonical target set.

Usage:
    python -m circuit_model ko-sweep [options]
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from .params import CircuitParams
from .simulation import simulate_circuit, mean_rates
from .plotting import POPULATION_NAMES, POPULATION_COLORS
from .study import StudyConfig


# =============================================================================
# KO COMBINATIONS
# =============================================================================

# Greek symbols for compact plot labels.
_RECEPTOR_SYMBOL = {"a7": "α7", "a5": "α5", "b2": "β2"}

# Mapping from a combination (frozenset of receptor keys) to the fit-target key
# that exercised it during optimization. WT (empty set) is the fitted baseline.
_COMBO_TO_TARGET_KEY: dict[frozenset[str], str] = {
    frozenset({"a7"}): "alpha7_ko_pyr",
    frozenset({"a5"}): "alpha5_ko_pyr",
    frozenset({"b2"}): "beta2_ko_pyr",
    frozenset({"a7", "b2"}): "alpha7_beta2_ko_pyr",
}

# Canonical fallback when no fit log is available: which combos were targets.
_DEFAULT_TARGET_COMBOS: set[frozenset[str]] = set(_COMBO_TO_TARGET_KEY.keys())


@dataclass(frozen=True)
class KOCombo:
    """A single knockout combination over the three global receptors."""
    receptors: frozenset[str]   # subset of {"a7", "a5", "b2"}
    category: str               # "WT" | "target" | "prediction"

    @property
    def name(self) -> str:
        """Compact label, e.g. 'WT' or 'α7+β2 KO'."""
        if not self.receptors:
            return "WT"
        order = ["a7", "a5", "b2"]
        return "+".join(_RECEPTOR_SYMBOL[r] for r in order if r in self.receptors) + " KO"

    @property
    def category_tag(self) -> str:
        """Human-readable category tag for the axis, e.g. 'target' / 'predicted'."""
        return {"WT": "baseline", "target": "fit target",
                "prediction": "predicted"}[self.category]

    @property
    def axis_label(self) -> str:
        """Two-line x-tick label: KO description over its fit category."""
        return f"{self.name}\n({self.category_tag})"


def enumerate_combos(target_combos: set[frozenset[str]]) -> list[KOCombo]:
    """Enumerate all 8 receptor-KO combinations, ordered by KO count.

    ``target_combos`` is the set of combinations that were optimization targets;
    every other non-WT combination is labelled a prediction.
    """
    receptors = ["a7", "a5", "b2"]
    combos: list[KOCombo] = []
    # Order: WT, singles, doubles, triple (i.e. by number of knocked-out receptors).
    for k in range(len(receptors) + 1):
        from itertools import combinations
        for subset in combinations(receptors, k):
            fs = frozenset(subset)
            if not fs:
                category = "WT"
            elif fs in target_combos:
                category = "target"
            else:
                category = "prediction"
            combos.append(KOCombo(receptors=fs, category=category))
    return combos


def detect_target_combos(log_path: Optional[Path]) -> set[frozenset[str]]:
    """Detect which KO combinations were optimization targets from a fit log.

    Reads the first line of ``log_path`` (a JSONL where each entry carries a
    ``target`` dict) and returns the set of combinations whose target value is
    present and non-null. Falls back to the canonical target set when the log is
    missing or unreadable.
    """
    if log_path is None or not log_path.exists():
        return set(_DEFAULT_TARGET_COMBOS)
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                target = json.loads(line).get("target", {})
                break
            else:
                return set(_DEFAULT_TARGET_COMBOS)
    except (OSError, json.JSONDecodeError):
        return set(_DEFAULT_TARGET_COMBOS)

    detected = {
        combo
        for combo, key in _COMBO_TO_TARGET_KEY.items()
        if target.get(key) is not None
    }
    return detected or set(_DEFAULT_TARGET_COMBOS)


def apply_ko_combo(base_params: CircuitParams, combo: KOCombo) -> CircuitParams:
    """Apply a receptor-KO combination to the base parameters.

    Matches the KO construction used during optimization
    (:func:`circuit_model.optimization._build_conditions`): a global α7 KO zeroes
    α7 activation on every cell type; α5/β2 KOs zero their single activation.
    """
    kwargs: dict[str, float] = {}
    if "a7" in combo.receptors:
        kwargs["act_alpha7_pv"] = 0.0
        kwargs["act_alpha7_som"] = 0.0
        kwargs["act_alpha7_ndnf"] = 0.0
    if "a5" in combo.receptors:
        kwargs["act_alpha5"] = 0.0
    if "b2" in combo.receptors:
        kwargs["act_beta2"] = 0.0
    return replace(base_params, **kwargs) if kwargs else base_params


# =============================================================================
# BATCH SIMULATION
# =============================================================================

@dataclass
class KOSweepResults:
    """Container for a combinatorial-KO sweep."""
    combos: list[KOCombo]
    population_names: list[str]
    data: dict[frozenset[str], np.ndarray]   # combo.receptors -> (n_runs, 5)
    config: StudyConfig


# Module-level state for ProcessPoolExecutor workers (params are pickled once).
_worker_args: Optional[tuple[CircuitParams, StudyConfig]] = None


def _init_worker(params: CircuitParams, cfg: StudyConfig) -> None:
    global _worker_args
    _worker_args = (params, cfg)


def _run_single_sim(seed: int) -> np.ndarray:
    if _worker_args is None:
        raise RuntimeError("Worker not initialized")
    params, cfg = _worker_args
    result = simulate_circuit(
        params,
        T_ms=cfg.T_ms,
        dt_ms=cfg.dt_ms,
        seed=seed,
        noise_type=cfg.noise_type,
        tau_noise_ms=cfg.tau_noise_ms,
        use_transient=False,
    )
    return mean_rates(result, burn_in_ms=cfg.burn_in_ms, window_ms=cfg.window_ms)


def run_params_batch(
    params: CircuitParams,
    cfg: StudyConfig,
    base_seed: int,
) -> np.ndarray:
    """Run ``cfg.n_runs`` simulations of a fixed parameter set, varying only the
    noise seed. Returns a (n_runs, 5) array of per-run mean rates."""
    rng = np.random.default_rng(base_seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(cfg.n_runs)]

    n_workers = cfg.n_workers or min(cfg.n_runs, os.cpu_count() or 4)

    if n_workers > 1 and cfg.n_runs > 1:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(params, cfg),
        ) as executor:
            futures = [executor.submit(_run_single_sim, s) for s in seeds]
            results = []
            with tqdm(total=cfg.n_runs, desc="Simulations", leave=False) as pbar:
                for future in futures:
                    results.append(future.result())
                    pbar.update()
    else:
        _init_worker(params, cfg)
        results = [_run_single_sim(s) for s in seeds]

    return np.array(results)


def run_ko_sweep(
    base_params: CircuitParams,
    cfg: StudyConfig,
    target_combos: set[frozenset[str]],
    base_seed: int = 0,
    verbose: bool = True,
) -> KOSweepResults:
    """Run the full combinatorial-KO sweep across all 8 combinations."""
    combos = enumerate_combos(target_combos)
    rng = np.random.default_rng(base_seed)
    data: dict[frozenset[str], np.ndarray] = {}

    for combo in combos:
        print(f"Running {combo.name:<10s} [{combo.category}]")
        params = apply_ko_combo(base_params, combo)
        seed = int(rng.integers(0, 2**31 - 1))
        data[combo.receptors] = run_params_batch(params, cfg, seed)
        if verbose:
            means = data[combo.receptors].mean(axis=0)
            print(f"  Mean rates: PYR={means[0]:.2f}, SOM={means[1]:.2f}, "
                  f"PV={means[2]:.2f}, VIP={means[3]:.2f}, NDNF={means[4]:.2f}")

    return KOSweepResults(
        combos=combos,
        population_names=list(POPULATION_NAMES),
        data=data,
        config=cfg,
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

# Box face / tick-label colors per category.
_CATEGORY_STYLE = {
    "WT":         dict(facecolor="#bdbdbd", tick="#444444", label="WT (baseline)"),
    "target":     dict(facecolor="#4c72b0", tick="#27508f", label="Fit target"),
    "prediction": dict(facecolor="#c44e52", tick="#9c2b30", label="Prediction (simulated only)"),
}


def plot_ko_sweep_boxplots(
    results: KOSweepResults,
    title: str = "Firing Rate Distribution by Receptor-KO Combination",
    figsize: tuple[float, float] = (16, 10),
    save_path: Optional[str] = None,
    show: bool = True,
    unit: str = "Hz",
):
    """Box plots of firing rates for all populations across every KO combination.

    Boxes are colored by category (WT / fit target / prediction) so the
    optimization targets are visually distinguished from model predictions.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from .plotting import _check_display_available

    combos = results.combos
    n_combos = len(combos)
    n_pops = len(POPULATION_NAMES)
    n_cols = 3
    n_rows = (n_pops + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()
    for i in range(n_pops, len(axes)):
        axes[i].set_visible(False)

    labels = [c.axis_label for c in combos]
    face_colors = [_CATEGORY_STYLE[c.category]["facecolor"] for c in combos]
    tick_colors = [_CATEGORY_STYLE[c.category]["tick"] for c in combos]

    for pop_idx, pop_name in enumerate(POPULATION_NAMES):
        ax = axes[pop_idx]
        data = [results.data[c.receptors][:, pop_idx] for c in combos]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="gray"),
            capprops=dict(color="gray"),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        for patch, fc in zip(bp["boxes"], face_colors):
            patch.set_facecolor(fc)
            patch.set_alpha(0.8)
            patch.set_edgecolor("black")

        ax.set_xticks(range(1, n_combos + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        # Color each tick label by its category so the target/predicted split is
        # legible directly on the axis, not only via the box fill.
        for ticklabel, tcolor in zip(ax.get_xticklabels(), tick_colors):
            ticklabel.set_color(tcolor)

        all_vals = np.concatenate(data)
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = max((vmax - vmin) * 0.2, 0.05 * max(vmax, 1e-6))
        ax.set_ylabel(f"Rate ({unit})", fontsize=10)
        ax.set_ylim(max(0, vmin - margin), vmax + margin)

        ax.set_title(pop_name, fontsize=13, fontweight="bold",
                     color=POPULATION_COLORS[pop_name])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    # Shared legend for the target/prediction distinction.
    legend_handles = [
        Patch(facecolor=style["facecolor"], edgecolor="black", label=style["label"])
        for style in _CATEGORY_STYLE.values()
    ]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=11,
               frameon=True, title="KO category")

    fig.suptitle(title, fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        if _check_display_available():
            plt.show(block=True)
        elif not save_path:
            fallback = f"ko_sweep_boxplots_{results.config.noise_type}.png"
            fig.savefig(fallback, dpi=150, bbox_inches="tight")
            print(f"No display available. Figure saved to: {fallback}")

    return fig
