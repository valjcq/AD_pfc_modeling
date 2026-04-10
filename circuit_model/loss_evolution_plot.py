"""
Loss evolution visualization for optimization runs.

This module provides functions to visualize loss components during optimization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def _extract_component_values(breakdown: dict) -> dict[str, float]:
    """Extract loss components from both legacy and ring log formats."""
    return {
        # Legacy optimize naming
        "firing_rate": float(breakdown.get("firing_rate", 0.0)),
        "ko_firing_rate": float(breakdown.get("ko_firing_rate", 0.0)),
        # Ring optimize naming
        "ring_rate": float(breakdown.get("ring_rate", 0.0)),
        "ko_penalty": float(breakdown.get("ko_penalty", 0.0)),
        # Shared / optional terms
        "jacobian": float(breakdown.get("jacobian", 0.0)),
        "turing": float(breakdown.get("turing", 0.0)),
        "ach_ratio": float(breakdown.get("ach_ratio", 0.0)),
        "spatial_uniformity": float(breakdown.get("spatial_uniformity", 0.0)),
        "bump": float(breakdown.get("bump", 0.0)),
    }


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


def should_plot_loss_evolution(step: int, log_interval: int = 50) -> bool:
    """Check if this step should trigger loss evolution plotting."""
    return step % log_interval == 0


def plot_loss_evolution(
    log_file: str,
    output_dir: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    dpi: int = 100,
) -> str:
    """
    Create a comprehensive loss evolution plot from a JSONL log file.
    
    The plot shows:
    - Global loss evolution (main panel, top-left)
    - Individual loss components (firing_rate, ko_firing_rate, jacobian, turing)
    - Ratio of each component to total loss
    
    Args:
        log_file: Path to JSONL optimization log file
        output_dir: Directory to save the plot. If None, uses parent of log_file
        figsize: Tuple of (width, height) in inches. Default: (16, 12)
        dpi: DPI for saved figure
        
    Returns:
        Path to saved figure
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    
    if figsize is None:
        figsize = (16, 12)
    
    if output_dir is None:
        output_dir = str(Path(log_file).parent)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load log file
    steps = []
    total_losses = []
    comp_keys = [
        "firing_rate", "ko_firing_rate", "ring_rate", "ko_penalty",
        "jacobian", "turing", "ach_ratio", "spatial_uniformity", "bump",
    ]
    comp_lists = {k: [] for k in comp_keys}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            steps.append(entry['step'])
            total_losses.append(entry['loss'])
            
            breakdown = entry.get('breakdown', {})
            comp_vals = _extract_component_values(breakdown)
            for k in comp_keys:
                comp_lists[k].append(comp_vals[k])
    
    steps = np.array(steps)
    total_losses = np.array(total_losses)
    components = {k: np.array(v, dtype=float) for k, v in comp_lists.items()}
    steps, total_losses, components = _drop_aberrant_initial_steps(steps, total_losses, components)

    # Legacy aliases for this figure layout
    fr_losses = components["firing_rate"] + components["ring_rate"]
    ko_losses = components["ko_firing_rate"] + components["ko_penalty"]
    jac_losses = components["jacobian"]
    turing_losses = components["turing"]
    ach_losses = components["ach_ratio"]
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Define colors for each component
    colors = {
        'firing_rate': '#1f77b4',      # blue
        'ko_firing_rate': '#ff7f0e',   # orange
        'jacobian': '#2ca02c',         # green
        'turing': '#d62728',           # red
        'ach_ratio': '#9467bd',        # purple
        'total': '#000000',            # black
    }
    
    # 1. Main plot: Total loss evolution
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(steps, total_losses, color=colors['total'], linewidth=2, label='Total Loss')
    ax1.set_xlabel('Optimization Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss Evolution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Stacked area plot: Loss components
    ax2 = plt.subplot(3, 3, 2)
    ax2.fill_between(steps, 0, fr_losses, alpha=0.5, label='Firing Rate', color=colors['firing_rate'])
    ax2.fill_between(steps, fr_losses, fr_losses + ko_losses, alpha=0.5, label='KO Firing Rate', color=colors['ko_firing_rate'])
    ax2.fill_between(steps, fr_losses + ko_losses, fr_losses + ko_losses + jac_losses, alpha=0.5, label='Jacobian', color=colors['jacobian'])
    ax2.fill_between(steps, fr_losses + ko_losses + jac_losses, 
                     fr_losses + ko_losses + jac_losses + turing_losses, 
                     alpha=0.5, label='Turing', color=colors['turing'])
    if np.any(ach_losses > 0):
        ax2.fill_between(steps, fr_losses + ko_losses + jac_losses + turing_losses,
                         fr_losses + ko_losses + jac_losses + turing_losses + ach_losses,
                         alpha=0.5, label='ACh Ratio', color=colors['ach_ratio'])
    ax2.set_xlabel('Optimization Step')
    ax2.set_ylabel('Loss (Stacked)')
    ax2.set_title('Loss Components (Stacked)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Individual component lines
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(steps, fr_losses, label='Firing Rate', color=colors['firing_rate'], linewidth=1.5)
    ax3.plot(steps, ko_losses, label='KO Firing Rate', color=colors['ko_firing_rate'], linewidth=1.5)
    ax3.plot(steps, jac_losses, label='Jacobian', color=colors['jacobian'], linewidth=1.5)
    ax3.plot(steps, turing_losses, label='Turing', color=colors['turing'], linewidth=1.5)
    if np.any(ach_losses > 0):
        ax3.plot(steps, ach_losses, label='ACh Ratio', color=colors['ach_ratio'], linewidth=1.5)
    ax3.set_xlabel('Optimization Step')
    ax3.set_ylabel('Loss Component Value')
    ax3.set_title('Individual Loss Components', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Calculate ratios (avoid division by zero)
    total_losses_safe = np.maximum(total_losses, 1e-10)
    fr_ratio = 100 * fr_losses / total_losses_safe
    ko_ratio = 100 * ko_losses / total_losses_safe
    jac_ratio = 100 * jac_losses / total_losses_safe
    turing_ratio = 100 * turing_losses / total_losses_safe
    ach_ratio = 100 * ach_losses / total_losses_safe
    
    # 4. Firing Rate component
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(steps, fr_losses, color=colors['firing_rate'], linewidth=2)
    ax4.fill_between(steps, 0, fr_losses, alpha=0.3, color=colors['firing_rate'])
    ax4.set_xlabel('Optimization Step')
    ax4.set_ylabel('Loss Value')
    ax4.set_title('Firing Rate Loss', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. KO Firing Rate component
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(steps, ko_losses, color=colors['ko_firing_rate'], linewidth=2)
    ax5.fill_between(steps, 0, ko_losses, alpha=0.3, color=colors['ko_firing_rate'])
    ax5.set_xlabel('Optimization Step')
    ax5.set_ylabel('Loss Value')
    ax5.set_title('KO Firing Rate Loss', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Jacobian component
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(steps, jac_losses, color=colors['jacobian'], linewidth=2)
    ax6.fill_between(steps, 0, jac_losses, alpha=0.3, color=colors['jacobian'])
    ax6.set_xlabel('Optimization Step')
    ax6.set_ylabel('Loss Value')
    ax6.set_title('Jacobian Loss', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Turing component
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(steps, turing_losses, color=colors['turing'], linewidth=2)
    ax7.fill_between(steps, 0, turing_losses, alpha=0.3, color=colors['turing'])
    ax7.set_xlabel('Optimization Step')
    ax7.set_ylabel('Loss Value')
    ax7.set_title('Turing Loss', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. ACh Ratio component (if present)
    ax8 = plt.subplot(3, 3, 8)
    if np.any(ach_losses > 0):
        ax8.plot(steps, ach_losses, color=colors['ach_ratio'], linewidth=2)
        ax8.fill_between(steps, 0, ach_losses, alpha=0.3, color=colors['ach_ratio'])
        ax8.set_xlabel('Optimization Step')
        ax8.set_ylabel('Loss Value')
        ax8.set_title('ACh Ratio Loss', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'No ACh Ratio Loss', ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('ACh Ratio Loss (not active)', fontsize=12, fontweight='bold')
        ax8.axis('off')
    
    # 9. Ratio pie chart (final state)
    ax9 = plt.subplot(3, 3, 9)
    final_fr = fr_losses[-1]
    final_ko = ko_losses[-1]
    final_jac = jac_losses[-1]
    final_tur = turing_losses[-1]
    final_ach = ach_losses[-1]
    
    final_values = [final_fr, final_ko, final_jac, final_tur]
    final_labels = ['Firing Rate', 'KO Firing Rate', 'Jacobian', 'Turing']
    final_colors_list = [colors['firing_rate'], colors['ko_firing_rate'], colors['jacobian'], colors['turing']]
    
    if final_ach > 0:
        final_values.append(final_ach)
        final_labels.append('ACh Ratio')
        final_colors_list.append(colors['ach_ratio'])
    
    # Filter out zero/small values for cleaner pie chart
    nonzero_idx = np.array(final_values) > 1e-6
    final_values_filtered = [v for v, keep in zip(final_values, nonzero_idx) if keep]
    final_labels_filtered = [l for l, keep in zip(final_labels, nonzero_idx) if keep]
    final_colors_filtered = [c for c, keep in zip(final_colors_list, nonzero_idx) if keep]
    
    if final_values_filtered:
        wedges, texts, autotexts = ax9.pie(final_values_filtered, labels=final_labels_filtered, 
                                             colors=final_colors_filtered, autopct='%1.1f%%',
                                             startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        ax9.set_title(f'Final Loss Breakdown (Step {steps[-1]})', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_dir) / 'loss_evolution.png'
    plt.savefig(str(output_file), dpi=dpi, bbox_inches='tight')

    plt.close(fig)
    
    return str(output_file)


def plot_loss_evolution_ratios(
    log_file: str,
    output_dir: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    dpi: int = 100,
) -> str:
    """
    Create a detailed ratio visualization of loss components.
    
    Shows each component as a percentage of total loss over optimization steps.
    
    Args:
        log_file: Path to JSONL optimization log file
        output_dir: Directory to save the plot. If None, uses parent of log_file
        figsize: Tuple of (width, height) in inches. Default: (14, 8)
        dpi: DPI for saved figure
        
    Returns:
        Path to saved figure
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    
    if figsize is None:
        figsize = (14, 8)
    
    if output_dir is None:
        output_dir = str(Path(log_file).parent)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load log file
    steps = []
    total_losses = []
    comp_keys = [
        "firing_rate", "ko_firing_rate", "ring_rate", "ko_penalty",
        "jacobian", "turing", "ach_ratio", "spatial_uniformity", "bump",
    ]
    comp_lists = {k: [] for k in comp_keys}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            steps.append(entry['step'])
            total_losses.append(entry['loss'])
            
            breakdown = entry.get('breakdown', {})
            comp_vals = _extract_component_values(breakdown)
            for k in comp_keys:
                comp_lists[k].append(comp_vals[k])
    
    steps = np.array(steps)
    total_losses = np.array(total_losses)
    components = {k: np.array(v, dtype=float) for k, v in comp_lists.items()}
    steps, total_losses, components = _drop_aberrant_initial_steps(steps, total_losses, components)

    # Merge legacy+ring aliases so ratio plots work for both log formats
    merged_components = {
        "rate": components["firing_rate"] + components["ring_rate"],
        "ko": components["ko_firing_rate"] + components["ko_penalty"],
        "jacobian": components["jacobian"],
        "turing": components["turing"],
        "ach_ratio": components["ach_ratio"],
        "spatial_uniformity": components["spatial_uniformity"],
        "bump": components["bump"],
    }

    active_names = [k for k, v in merged_components.items() if np.any(v > 0)]
    if not active_names:
        active_names = ["rate", "ko", "jacobian", "turing"]

    total_losses_safe = np.maximum(total_losses, 1e-10)
    ratio = {k: 100.0 * merged_components[k] / total_losses_safe for k in active_names}
    
    # Define colors
    colors = {
        'rate': '#1f77b4',
        'ko': '#ff7f0e',
        'jacobian': '#2ca02c',
        'turing': '#d62728',
        'ach_ratio': '#9467bd',
        'spatial_uniformity': '#8c564b',
        'bump': '#17becf',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    
    # 1. Stacked area chart of ratios
    ax = axes[0, 0]
    stack_arrays = [ratio[name] for name in active_names]
    stack_labels = [name.replace('_', ' ').title() for name in active_names]
    stack_colors = [colors.get(name, '#7f7f7f') for name in active_names]
    ax.stackplot(steps, *stack_arrays, labels=stack_labels, colors=stack_colors, alpha=0.7)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Percentage of Total Loss (%)')
    ax.set_title('Loss Component Ratios (Stacked %)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Individual ratio lines (full scale)
    ax = axes[0, 1]
    for name in active_names:
        ax.plot(steps, ratio[name], label=name.replace('_', ' ').title(), color=colors.get(name, '#7f7f7f'), linewidth=2)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Percentage of Total Loss (%)')
    ax.set_title('Loss Component Ratios (Line)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # 3. Zoomed ratio lines for small components (auto scale)
    ax = axes[1, 0]
    for name in active_names:
        ax.plot(steps, ratio[name], label=name.replace('_', ' ').title(), color=colors.get(name, '#7f7f7f'), linewidth=2)
    # Auto zoom: ignore the largest median contributor to reveal smaller curves
    medians = {k: float(np.median(v)) for k, v in ratio.items()}
    dominant = max(medians, key=medians.get)
    others = [ratio[k] for k in active_names if k != dominant]
    if others:
        ymax = max(5.0, min(100.0, np.percentile(np.concatenate(others), 99.0) * 1.2))
        ax.set_ylim(0.0, ymax)
    else:
        ax.set_ylim(0.0, 10.0)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Percentage of Total Loss (%)')
    ax.set_title('Ratios (Zoom on Small Components)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. Convergence metrics
    ax = axes[1, 1]
    ax.semilogy(steps, total_losses, 'o-', label='Total Loss', color='black', linewidth=2, markersize=4)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Loss Value (log scale)')
    ax.set_title('Loss Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=9)
    
    # Add statistics
    best_loss_idx = np.argmin(total_losses)
    best_loss = total_losses[best_loss_idx]
    best_step = steps[best_loss_idx]
    improvement = ((total_losses[0] - best_loss) / total_losses[0] * 100) if total_losses[0] > 0 else 0
    
    stats_text = f'Best Loss: {best_loss:.4g} (step {best_step})\nImprovement: {improvement:.1f}%'
    ax.text(0.98, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_dir) / 'loss_evolution_ratios.png'
    plt.savefig(str(output_file), dpi=dpi, bbox_inches='tight')

    plt.close(fig)
    
    return str(output_file)
