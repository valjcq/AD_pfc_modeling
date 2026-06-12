"""
Loss evolution visualization for optimization runs.

The current optimisation loss (`_loss_from_results` in `optimization.py`) logs a
per-step `breakdown` with these buckets:

    base          -> 5 baseline wild-type firing-rate residuals
    global_ko     -> PYR rate under each global receptor KO
    selective_ko  -> NDNF / PV rate under their selective alpha7 KOs
    drug          -> per-drug measurements (Stage 2 only)
    total         -> weighted sum of the above

For plotting we expose only the residual families the fit actually minimises:

    wt    -> base                          ("target wild-type" baseline residual)
    ko    -> global_ko + selective_ko      (combined knockout residual)
    drug  -> drug                          (Stage 2, shown only when non-zero)

The legacy jacobian / turing / bistability / ach-ratio terms are no longer part
of the loss and are intentionally not plotted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


# Canonical residual components plotted, in display order.
COMPONENT_KEYS = ["wt", "ko", "drug"]

COMPONENT_COLORS = {
    "wt": "#1f77b4",
    "ko": "#ff7f0e",
    "drug": "#2ca02c",
    "total": "#000000",
}


def _extract_component_values(breakdown: dict) -> dict[str, float]:
    """Map a logged loss `breakdown` to the canonical residual components."""
    def _getf(key: str, default: float = 0.0) -> float:
        val = breakdown.get(key, default)
        return float(default) if val is None else float(val)

    return {
        "wt": _getf("base"),
        "ko": _getf("global_ko") + _getf("selective_ko"),
        "drug": _getf("drug"),
    }


def _pretty_component_name(name: str) -> str:
    return {
        "wt": "Wild-type (baseline)",
        "ko": "Knockout",
        "drug": "Drug",
    }.get(name, name.replace("_", " ").title())


def _drop_aberrant_initial_steps(
    steps: np.ndarray,
    total_losses: np.ndarray,
    components: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Drop aberrant first-step entries when they dominate scale.

    Heuristic:
    - consider all rows with minimal step value (usually step 1),
    - if their median total loss is > 5x median of subsequent rows,
      remove all those minimal-step rows.
    """
    if len(steps) < 3:
        return steps, total_losses, components

    first_step = np.min(steps)
    first_mask = steps == first_step
    later_mask = ~first_mask
    if not np.any(later_mask):
        return steps, total_losses, components

    first_med = float(np.median(total_losses[first_mask]))
    later_med = float(np.median(total_losses[later_mask]))
    if later_med <= 0:
        return steps, total_losses, components

    if first_med > 5.0 * later_med:
        steps = steps[later_mask]
        total_losses = total_losses[later_mask]
        for k in list(components.keys()):
            components[k] = components[k][later_mask]

    return steps, total_losses, components


def _load_log(
    log_file: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    """Load steps, total loss, and residual components from a JSONL log file.

    Returns (steps, total_losses, components, active_names) where `active_names`
    lists the residual components that are non-zero anywhere in the run.
    """
    steps: list[int] = []
    total_losses: list[float] = []
    comp_lists: dict[str, list[float]] = {k: [] for k in COMPONENT_KEYS}

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "step" not in entry or "loss" not in entry:
                continue  # skip non-standard records (e.g. Stage 2 drug logs)
            steps.append(entry["step"])
            total_losses.append(entry["loss"])
            comp_vals = _extract_component_values(entry.get("breakdown", {}))
            for k in COMPONENT_KEYS:
                comp_lists[k].append(comp_vals[k])

    steps_arr = np.array(steps)
    total_arr = np.array(total_losses, dtype=float)
    components = {k: np.array(v, dtype=float) for k, v in comp_lists.items()}
    steps_arr, total_arr, components = _drop_aberrant_initial_steps(
        steps_arr, total_arr, components
    )

    active_names = [k for k in COMPONENT_KEYS if np.any(components[k] > 0)]
    if not active_names:
        active_names = ["wt", "ko"]
    return steps_arr, total_arr, components, active_names


def should_plot_loss_evolution(step: int, log_interval: int = 50) -> bool:
    """Check if this step should trigger loss evolution plotting."""
    return step % log_interval == 0


def plot_total_loss(
    log_file: str,
    output_dir: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    dpi: int = 150,
) -> str:
    """Plot the total loss only, over optimisation steps (single panel, log-y).

    Args:
        log_file: Path to JSONL optimization log file.
        output_dir: Directory to save the plot. If None, uses parent of log_file.
        figsize: (width, height) in inches. Default: (7, 4.5).
        dpi: DPI for saved figure.

    Returns:
        Path to saved figure.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (7, 4.5)
    if output_dir is None:
        output_dir = str(Path(log_file).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    steps, total_losses, _components, _active = _load_log(log_file)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.plot(steps, total_losses, color=COMPONENT_COLORS["total"], linewidth=2)
    ax.set_xlabel("Optimisation step")
    ax.set_ylabel("Total loss")
    ax.set_title("Total loss evolution", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    best_idx = int(np.argmin(total_losses))
    best_loss = float(total_losses[best_idx])
    best_step = int(steps[best_idx])
    ax.text(0.98, 0.95, f"Best loss: {best_loss:.4g} (step {best_step})",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    output_file = Path(output_dir) / "loss_total.png"
    plt.savefig(str(output_file), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_file)


def plot_loss_evolution(
    log_file: str,
    output_dir: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    dpi: int = 100,
) -> str:
    """Plot total loss and its residual components over optimisation steps.

    Two panels (both log-y):
    - left:  total loss evolution.
    - right: residual components (wild-type baseline, knockout, drug).

    Args:
        log_file: Path to JSONL optimization log file.
        output_dir: Directory to save the plot. If None, uses parent of log_file.
        figsize: (width, height) in inches. Default: (12, 4.5).
        dpi: DPI for saved figure.

    Returns:
        Path to saved figure.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (12, 4.5)
    if output_dir is None:
        output_dir = str(Path(log_file).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    steps, total_losses, components, active_names = _load_log(log_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # --- Left: total loss ---
    ax1.plot(steps, total_losses, color=COMPONENT_COLORS["total"], linewidth=2,
             label="Total loss")
    ax1.set_xlabel("Optimisation step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Total loss evolution", fontsize=12, fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, which="both")

    # --- Right: residual components ---
    for name in active_names:
        ax2.plot(steps, np.maximum(components[name], 1e-12),
                 label=_pretty_component_name(name),
                 color=COMPONENT_COLORS.get(name, "#7f7f7f"), linewidth=1.8)
    ax2.set_xlabel("Optimisation step")
    ax2.set_ylabel("Residual loss")
    ax2.set_title("Residual loss components", fontsize=12, fontweight="bold")
    ax2.set_yscale("log")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    output_file = Path(output_dir) / "loss_evolution.png"
    plt.savefig(str(output_file), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_file)


def plot_loss_evolution_ratios(
    log_file: str,
    output_dir: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    dpi: int = 100,
) -> str:
    """Plot the share of each residual in the total loss, plus convergence.

    Two panels:
    - left:  stacked share (%) of each residual component in the total loss.
    - right: total-loss convergence (log-y) with a best-loss annotation.

    Args:
        log_file: Path to JSONL optimization log file.
        output_dir: Directory to save the plot. If None, uses parent of log_file.
        figsize: (width, height) in inches. Default: (12, 4.5).
        dpi: DPI for saved figure.

    Returns:
        Path to saved figure.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (12, 4.5)
    if output_dir is None:
        output_dir = str(Path(log_file).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    steps, total_losses, components, active_names = _load_log(log_file)
    total_safe = np.maximum(total_losses, 1e-10)
    ratio = {k: 100.0 * components[k] / total_safe for k in active_names}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # --- Left: stacked share of total ---
    stack_arrays = [ratio[name] for name in active_names]
    stack_labels = [_pretty_component_name(name) for name in active_names]
    stack_colors = [COMPONENT_COLORS.get(name, "#7f7f7f") for name in active_names]
    ax1.stackplot(steps, *stack_arrays, labels=stack_labels,
                  colors=stack_colors, alpha=0.75)
    ax1.set_xlabel("Optimisation step")
    ax1.set_ylabel("Share of total loss (%)")
    ax1.set_title("Residual share of total loss", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # --- Right: total-loss convergence ---
    ax2.semilogy(steps, total_losses, "o-", color="black", linewidth=2,
                 markersize=3, label="Total loss")
    ax2.set_xlabel("Optimisation step")
    ax2.set_ylabel("Loss (log scale)")
    ax2.set_title("Loss convergence", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(loc="best", fontsize=9)

    best_idx = int(np.argmin(total_losses))
    best_loss = float(total_losses[best_idx])
    best_step = int(steps[best_idx])
    improvement = ((total_losses[0] - best_loss) / total_losses[0] * 100.0
                   if total_losses[0] > 0 else 0.0)
    stats_text = f"Best loss: {best_loss:.4g} (step {best_step})\nImprovement: {improvement:.1f}%"
    ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    output_file = Path(output_dir) / "loss_evolution_ratios.png"
    plt.savefig(str(output_file), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_file)


def plot_loss_evolution_thesis(
    log_file: str,
    output_dir: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    dpi: int = 150,
) -> str:
    """Clean 2-panel loss figure for the thesis.

    Left panel:  total loss on log scale.
    Right panel: stacked residual components (wild-type, knockout, drug) on log
    scale.

    Args:
        log_file: Path to JSONL optimization log file.
        output_dir: Directory to save the plot. If None, uses parent of log_file.
        figsize: (width, height) in inches. Default: (10, 4).
        dpi: DPI for saved figure.

    Returns:
        Path to saved figure.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (10, 4)
    if output_dir is None:
        output_dir = str(Path(log_file).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    steps, total_losses, components, active_names = _load_log(log_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # --- Left: total loss (log scale) ---
    ax1.plot(steps, total_losses, color="black", linewidth=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Optimisation step")
    ax1.set_ylabel("Total loss")
    ax1.set_title("Total loss evolution")
    ax1.grid(True, alpha=0.3, which="both")

    # --- Right: stacked residual components (log scale) ---
    floor = 1e-10
    stack_arrays = [np.maximum(components[n], floor) for n in active_names]
    stack_labels = [_pretty_component_name(n) for n in active_names]
    stack_colors = [COMPONENT_COLORS.get(n, "#7f7f7f") for n in active_names]
    ax2.stackplot(steps, *stack_arrays, labels=stack_labels,
                  colors=stack_colors, alpha=0.75)
    ax2.set_yscale("log")
    ax2.set_xlabel("Optimisation step")
    ax2.set_ylabel("Residual loss (stacked)")
    ax2.set_title("Residual loss components")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    output_file = Path(output_dir) / "loss_evolution_thesis.pdf"
    plt.savefig(str(output_file), bbox_inches="tight")
    plt.close(fig)
    return str(output_file)
