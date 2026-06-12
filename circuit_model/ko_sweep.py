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


def _draw_boxplot_panel(
    ax,
    data: list[np.ndarray],
    *,
    labels: list[str],
    face_colors: list[str],
    tick_colors: list[str],
    pop_name: str,
    unit: str,
    label_fontsize: int = 8,
) -> None:
    """Draw one population's box plot onto ``ax``.

    Shared by the global (8-combo) and per-population (64-combo) sweeps so the
    box styling, category coloring and y-zoom behave identically.
    """
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

    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=label_fontsize)
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


def _category_legend_handles():
    """Legend patches for the WT / target / prediction categories."""
    from matplotlib.patches import Patch
    return [
        Patch(facecolor=style["facecolor"], edgecolor="black", label=style["label"])
        for style in _CATEGORY_STYLE.values()
    ]


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
    from .plotting import _check_display_available

    combos = results.combos
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
        data = [results.data[c.receptors][:, pop_idx] for c in combos]
        _draw_boxplot_panel(
            axes[pop_idx], data,
            labels=labels, face_colors=face_colors, tick_colors=tick_colors,
            pop_name=pop_name, unit=unit,
        )

    fig.legend(handles=_category_legend_handles(), loc="lower right", fontsize=11,
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


# =============================================================================
# PER-POPULATION KO SWEEP
# =============================================================================
#
# Receptors are expressed only on specific populations, so a knockout is a
# (population, receptor) "slot". There are 6 such slots in the model, giving
# 2^6 = 64 combinations. Each slot is knocked out by zeroing one CircuitParams
# field: α7/α5 via their activation multiplier (matching how optimization builds
# its KO conditions, incl. g_alpha7 mean-scaling); β2 via the population-specific
# current I_beta2_<pop>, since act_beta2 is shared across SOM and NDNF.

# (population, receptor, field_to_zero), ordered for stable enumeration.
KO_SLOTS: list[tuple[str, str, str]] = [
    ("PV",   "a7", "act_alpha7_pv"),
    ("SOM",  "a7", "act_alpha7_som"),
    ("SOM",  "b2", "I_beta2_som"),
    ("VIP",  "a5", "act_alpha5"),
    ("NDNF", "a7", "act_alpha7_ndnf"),
    ("NDNF", "b2", "I_beta2_ndnf"),
]

# slot key (pop, receptor) -> field to zero
_SLOT_FIELD: dict[tuple[str, str], str] = {(p, r): f for p, r, f in KO_SLOTS}

# Fit-target keys (in log.jsonl's `target`) -> the slot-set they knock out.
_TARGET_KEY_TO_SLOTSET: dict[str, frozenset[tuple[str, str]]] = {
    "alpha7_ko_pyr":        frozenset({("PV", "a7"), ("SOM", "a7"), ("NDNF", "a7")}),
    "alpha5_ko_pyr":        frozenset({("VIP", "a5")}),
    "beta2_ko_pyr":         frozenset({("SOM", "b2"), ("NDNF", "b2")}),
    "alpha7_beta2_ko_pyr":  frozenset({("PV", "a7"), ("SOM", "a7"), ("NDNF", "a7"),
                                       ("SOM", "b2"), ("NDNF", "b2")}),
    "alpha7_ndnf_ko_ndnf":  frozenset({("NDNF", "a7")}),
    "alpha7_pv_ko_pv":      frozenset({("PV", "a7")}),
}

_DEFAULT_TARGET_SLOTSETS: set[frozenset[tuple[str, str]]] = set(_TARGET_KEY_TO_SLOTSET.values())


def _slot_fragment(slot: tuple[str, str]) -> str:
    """Phenotype fragment for a slot, e.g. ('PV','a7') -> 'PV α7'."""
    pop, receptor = slot
    return f"{pop} {_RECEPTOR_SYMBOL[receptor]}"


@dataclass(frozen=True)
class PerPopCombo:
    """A per-population knockout combination over the 6 receptor slots."""
    slots: frozenset[tuple[str, str]]   # subset of _SLOT_FIELD keys
    category: str                       # "WT" | "target" | "prediction"

    @property
    def ko_count(self) -> int:
        return len(self.slots)

    @property
    def phenotype(self) -> str:
        """KO'd-slots-only label, e.g. 'PV α7 + NDNF β2' or 'WT'."""
        if not self.slots:
            return "WT"
        # Order fragments by the canonical slot order for stable labels.
        ordered = [_slot_fragment((p, r)) for p, r, _ in KO_SLOTS if (p, r) in self.slots]
        return " + ".join(ordered)

    @property
    def category_tag(self) -> str:
        return {"WT": "baseline", "target": "fit target",
                "prediction": "predicted"}[self.category]

    @property
    def axis_label(self) -> str:
        return f"{self.phenotype}\n({self.category_tag})"


def detect_target_slotsets(log_path: Optional[Path]) -> set[frozenset[tuple[str, str]]]:
    """Detect which per-population slot-sets were optimization targets.

    Mirrors :func:`detect_target_combos` but maps each present fit-target key to
    its (population, receptor) slot-set. Falls back to the canonical target set
    when the log is missing or unreadable.
    """
    if log_path is None or not log_path.exists():
        return set(_DEFAULT_TARGET_SLOTSETS)
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                target = json.loads(line).get("target", {})
                break
            else:
                return set(_DEFAULT_TARGET_SLOTSETS)
    except (OSError, json.JSONDecodeError):
        return set(_DEFAULT_TARGET_SLOTSETS)

    detected = {
        slotset
        for key, slotset in _TARGET_KEY_TO_SLOTSET.items()
        if target.get(key) is not None
    }
    return detected or set(_DEFAULT_TARGET_SLOTSETS)


def enumerate_perpop_combos(
    target_slotsets: set[frozenset[tuple[str, str]]],
    max_ko: Optional[int] = None,
) -> list[PerPopCombo]:
    """Enumerate all 2^6 per-population KO combinations.

    Ordered by number of knockouts then phenotype. ``max_ko`` optionally caps the
    number of simultaneous knockouts (None = all 6). A combo is a fit ``target``
    if its slot-set exactly matches a detected target set, ``WT`` if empty, else
    a ``prediction``.
    """
    from itertools import combinations

    slot_keys = [(p, r) for p, r, _ in KO_SLOTS]
    k_max = len(slot_keys) if max_ko is None else min(max_ko, len(slot_keys))
    combos: list[PerPopCombo] = []
    for k in range(k_max + 1):
        subsets = [frozenset(s) for s in combinations(slot_keys, k)]
        # Stable secondary ordering by phenotype text.
        for fs in sorted(subsets, key=lambda s: _phenotype_sort_key(s)):
            if not fs:
                category = "WT"
            elif fs in target_slotsets:
                category = "target"
            else:
                category = "prediction"
            combos.append(PerPopCombo(slots=fs, category=category))
    return combos


def _phenotype_sort_key(slots: frozenset[tuple[str, str]]) -> tuple:
    """Canonical-order sort key for a slot-set (used to order combos)."""
    index = {(p, r): i for i, (p, r, _) in enumerate(KO_SLOTS)}
    return tuple(sorted(index[s] for s in slots))


def apply_perpop_combo(base_params: CircuitParams, combo: PerPopCombo) -> CircuitParams:
    """Apply a per-population KO combination by zeroing each slot's field."""
    kwargs = {_SLOT_FIELD[slot]: 0.0 for slot in combo.slots}
    return replace(base_params, **kwargs) if kwargs else base_params


@dataclass
class PerPopSweepResults:
    """Container for a per-population combinatorial-KO sweep."""
    combos: list[PerPopCombo]
    population_names: list[str]
    data: dict[frozenset[tuple[str, str]], np.ndarray]   # combo.slots -> (n_runs, 5)
    config: StudyConfig


def run_perpop_sweep(
    base_params: CircuitParams,
    cfg: StudyConfig,
    target_slotsets: set[frozenset[tuple[str, str]]],
    base_seed: int = 0,
    max_ko: Optional[int] = None,
    verbose: bool = True,
) -> PerPopSweepResults:
    """Run the full per-population KO sweep across all combinations."""
    combos = enumerate_perpop_combos(target_slotsets, max_ko=max_ko)
    rng = np.random.default_rng(base_seed)
    data: dict[frozenset[tuple[str, str]], np.ndarray] = {}

    for i, combo in enumerate(combos, start=1):
        print(f"[{i:>2d}/{len(combos)}] {combo.phenotype:<32s} [{combo.category}]")
        params = apply_perpop_combo(base_params, combo)
        seed = int(rng.integers(0, 2**31 - 1))
        data[combo.slots] = run_params_batch(params, cfg, seed)
        if verbose:
            means = data[combo.slots].mean(axis=0)
            print(f"        PYR={means[0]:.2f} SOM={means[1]:.2f} PV={means[2]:.2f} "
                  f"VIP={means[3]:.2f} NDNF={means[4]:.2f}")

    return PerPopSweepResults(
        combos=combos,
        population_names=list(POPULATION_NAMES),
        data=data,
        config=cfg,
    )


# -----------------------------------------------------------------------------
# Per-population outputs: heatmap overview, faceted box plots, CSV
# -----------------------------------------------------------------------------

def plot_perpop_heatmap(
    results: PerPopSweepResults,
    save_path: str,
    title: str = "Per-population KO sweep — mean rate vs WT (log₂ fold-change)",
    clip: float = 3.0,
):
    """Compact overview: a (n_combos × 5) heatmap of log₂ fold-change vs WT.

    Diverging colormap centered at 0 (= WT level). Cells are annotated with the
    absolute mean rate (Hz). Target rows are marked with ★ and a colored label.
    """
    import matplotlib.pyplot as plt

    combos = results.combos
    pops = POPULATION_NAMES
    eps = 0.01

    wt_combo = next((c for c in combos if c.category == "WT"), combos[0])
    wt_means = results.data[wt_combo.slots].mean(axis=0)

    abs_rate = np.array([results.data[c.slots].mean(axis=0) for c in combos])  # (n,5)
    fold = np.log2((abs_rate + eps) / (wt_means + eps))
    fold = np.clip(fold, -clip, clip)

    n = len(combos)
    fig_h = max(4.0, 0.32 * n + 1.5)
    fig, ax = plt.subplots(figsize=(7.0, fig_h), constrained_layout=True)

    im = ax.imshow(fold, aspect="auto", cmap="RdBu_r", vmin=-clip, vmax=clip)

    ax.set_xticks(range(len(pops)))
    ax.set_xticklabels(pops, fontsize=10)
    ax.set_yticks(range(n))
    ylabels = [("★ " if c.category == "target" else "") + c.phenotype for c in combos]
    ax.set_yticklabels(ylabels, fontsize=7)
    for ticklabel, c in zip(ax.get_yticklabels(), combos):
        ticklabel.set_color(_CATEGORY_STYLE[c.category]["tick"])

    # Annotate each cell with the absolute mean rate (Hz).
    for i in range(n):
        for j in range(len(pops)):
            ax.text(j, i, f"{abs_rate[i, j]:.1f}", ha="center", va="center",
                    fontsize=5.5, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log₂(rate / WT)", fontsize=9)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Population (readout)", fontsize=10)
    ax.text(0.0, -0.06 / (fig_h / 6), "★ = fit target · row label color: "
            "gray WT / blue target / red prediction · cell value = mean Hz",
            transform=ax.transAxes, fontsize=7, color="#555555")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to: {save_path}")
    plt.close(fig)
    return fig


def plot_perpop_boxplots_faceted(
    results: PerPopSweepResults,
    out_dir: str,
    max_per_fig: int = 16,
    unit: str = "Hz",
):
    """Box plots grouped by number of simultaneous KOs, auto-paginated.

    Writes one PNG per KO-count group (split into ``_partN`` when a group exceeds
    ``max_per_fig`` boxes) into ``out_dir``. Returns the list of saved paths.
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict

    os.makedirs(out_dir, exist_ok=True)

    by_count: dict[int, list[PerPopCombo]] = defaultdict(list)
    for c in results.combos:
        by_count[c.ko_count].append(c)

    n_pops = len(POPULATION_NAMES)
    n_cols = 3
    n_rows = (n_pops + n_cols - 1) // n_cols
    saved: list[str] = []

    for k in sorted(by_count):
        group = by_count[k]
        # Paginate into chunks of at most max_per_fig.
        n_parts = (len(group) + max_per_fig - 1) // max_per_fig
        for part in range(n_parts):
            chunk = group[part * max_per_fig:(part + 1) * max_per_fig]
            labels = [c.axis_label for c in chunk]
            face_colors = [_CATEGORY_STYLE[c.category]["facecolor"] for c in chunk]
            tick_colors = [_CATEGORY_STYLE[c.category]["tick"] for c in chunk]

            # Widen the figure with the number of boxes so the rotated two-line
            # phenotype labels get enough horizontal room and never overlap.
            fig_w = max(16.0, 1.5 * len(chunk) + 4.0)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, 10),
                                     constrained_layout=True)
            axes = axes.flatten()
            for i in range(n_pops, len(axes)):
                axes[i].set_visible(False)

            for pop_idx, pop_name in enumerate(POPULATION_NAMES):
                data = [results.data[c.slots][:, pop_idx] for c in chunk]
                _draw_boxplot_panel(
                    axes[pop_idx], data,
                    labels=labels, face_colors=face_colors, tick_colors=tick_colors,
                    pop_name=pop_name, unit=unit,
                )

            fig.legend(handles=_category_legend_handles(), loc="lower right",
                       fontsize=11, frameon=True, title="KO category")
            ko_word = "knockout" if k == 1 else "knockouts"
            part_str = f"  (part {part + 1}/{n_parts})" if n_parts > 1 else ""
            kind = "WT baseline" if k == 0 else f"{k} simultaneous {ko_word}"
            fig.suptitle(f"Per-population KO sweep — {kind}{part_str}",
                         fontsize=14, fontweight="bold")

            suffix = f"_part{part + 1}" if n_parts > 1 else ""
            path = os.path.join(out_dir, f"ko{k}{suffix}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
            print(f"  saved {path}  ({len(chunk)} combos)")

    return saved


def write_perpop_csv(results: PerPopSweepResults, path: str) -> None:
    """Write a tidy CSV: one row per (combo, run) with per-population rates."""
    import csv

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["phenotype", "category", "ko_count", "run_idx",
                         *POPULATION_NAMES])
        for c in results.combos:
            arr = results.data[c.slots]  # (n_runs, 5)
            for run_idx in range(arr.shape[0]):
                writer.writerow([
                    c.phenotype, c.category, c.ko_count, run_idx,
                    *[f"{v:.6g}" for v in arr[run_idx]],
                ])
    print(f"Summary CSV saved to: {path}")
