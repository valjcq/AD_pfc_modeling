"""
Loss evolution visualization for optimization runs.

This module provides functions to visualize loss components during optimization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


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
    fr_losses = []
    ko_losses = []
    jac_losses = []
    turing_losses = []
    ach_losses = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            steps.append(entry['step'])
            total_losses.append(entry['loss'])
            
            breakdown = entry.get('breakdown', {})
            fr_losses.append(breakdown.get('firing_rate', 0.0))
            ko_losses.append(breakdown.get('ko_firing_rate', 0.0))
            jac_losses.append(breakdown.get('jacobian', 0.0))
            turing_losses.append(breakdown.get('turing', 0.0))
            ach_losses.append(breakdown.get('ach_ratio', 0.0))
    
    steps = np.array(steps)
    total_losses = np.array(total_losses)
    fr_losses = np.array(fr_losses)
    ko_losses = np.array(ko_losses)
    jac_losses = np.array(jac_losses)
    turing_losses = np.array(turing_losses)
    ach_losses = np.array(ach_losses)
    
    # Skip the first step (typically has huge initial loss that breaks scaling)
    if len(steps) > 1:
        steps = steps[1:]
        total_losses = total_losses[1:]
        fr_losses = fr_losses[1:]
        ko_losses = ko_losses[1:]
        jac_losses = jac_losses[1:]
        turing_losses = turing_losses[1:]
        ach_losses = ach_losses[1:]
    
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
    print(f"✓ Loss evolution plot saved to {output_file}")
    
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
    import matplotlib.pyplot as plt
    
    if figsize is None:
        figsize = (14, 8)
    
    if output_dir is None:
        output_dir = str(Path(log_file).parent)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load log file
    steps = []
    total_losses = []
    fr_losses = []
    ko_losses = []
    jac_losses = []
    turing_losses = []
    ach_losses = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            steps.append(entry['step'])
            total_losses.append(entry['loss'])
            
            breakdown = entry.get('breakdown', {})
            fr_losses.append(breakdown.get('firing_rate', 0.0))
            ko_losses.append(breakdown.get('ko_firing_rate', 0.0))
            jac_losses.append(breakdown.get('jacobian', 0.0))
            turing_losses.append(breakdown.get('turing', 0.0))
            ach_losses.append(breakdown.get('ach_ratio', 0.0))
    
    steps = np.array(steps)
    total_losses = np.array(total_losses)
    fr_losses = np.array(fr_losses)
    ko_losses = np.array(ko_losses)
    jac_losses = np.array(jac_losses)
    turing_losses = np.array(turing_losses)
    ach_losses = np.array(ach_losses)
    
    # Skip the first step (typically has huge initial loss that breaks scaling)
    if len(steps) > 1:
        steps = steps[1:]
        total_losses = total_losses[1:]
        fr_losses = fr_losses[1:]
        ko_losses = ko_losses[1:]
        jac_losses = jac_losses[1:]
        turing_losses = turing_losses[1:]
        ach_losses = ach_losses[1:]
    
    # Calculate ratios
    total_losses_safe = np.maximum(total_losses, 1e-10)
    fr_ratio = 100 * fr_losses / total_losses_safe
    ko_ratio = 100 * ko_losses / total_losses_safe
    jac_ratio = 100 * jac_losses / total_losses_safe
    turing_ratio = 100 * turing_losses / total_losses_safe
    ach_ratio = 100 * ach_losses / total_losses_safe
    
    # Define colors
    colors = {
        'firing_rate': '#1f77b4',
        'ko_firing_rate': '#ff7f0e',
        'jacobian': '#2ca02c',
        'turing': '#d62728',
        'ach_ratio': '#9467bd',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    
    # 1. Stacked area chart of ratios
    ax = axes[0, 0]
    ax.stackplot(steps, fr_ratio, ko_ratio, jac_ratio, turing_ratio, 
                 labels=['Firing Rate', 'KO Firing Rate', 'Jacobian', 'Turing'],
                 colors=[colors['firing_rate'], colors['ko_firing_rate'], 
                        colors['jacobian'], colors['turing']],
                 alpha=0.7)
    if np.any(ach_losses > 0):
        ax.stackplot(steps, ach_ratio, labels=['ACh Ratio'], colors=[colors['ach_ratio']], alpha=0.7)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Percentage of Total Loss (%)')
    ax.set_title('Loss Component Ratios (Stacked %)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Individual ratio lines
    ax = axes[0, 1]
    ax.plot(steps, fr_ratio, label='Firing Rate', color=colors['firing_rate'], linewidth=2)
    ax.plot(steps, ko_ratio, label='KO Firing Rate', color=colors['ko_firing_rate'], linewidth=2)
    ax.plot(steps, jac_ratio, label='Jacobian', color=colors['jacobian'], linewidth=2)
    ax.plot(steps, turing_ratio, label='Turing', color=colors['turing'], linewidth=2)
    if np.any(ach_losses > 0):
        ax.plot(steps, ach_ratio, label='ACh Ratio', color=colors['ach_ratio'], linewidth=2)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Percentage of Total Loss (%)')
    ax.set_title('Loss Component Ratios (Line)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # 3. Total loss with component breakdown bars (sample every Nth step for clarity)
    ax = axes[1, 0]
    sample_every = max(1, len(steps) // 20)  # Show ~20 time points
    sample_indices = np.arange(0, len(steps), sample_every)
    if len(steps) - 1 not in sample_indices:
        sample_indices = np.append(sample_indices, len(steps) - 1)
    
    sample_steps = steps[sample_indices]
    x_pos = np.arange(len(sample_steps))
    
    sample_fr = fr_losses[sample_indices]
    sample_ko = ko_losses[sample_indices]
    sample_jac = jac_losses[sample_indices]
    sample_tur = turing_losses[sample_indices]
    sample_ach = ach_losses[sample_indices]
    
    width = 0.6
    ax.bar(x_pos, sample_fr, width, label='Firing Rate', color=colors['firing_rate'], alpha=0.8)
    ax.bar(x_pos, sample_ko, width, bottom=sample_fr, label='KO Firing Rate', 
           color=colors['ko_firing_rate'], alpha=0.8)
    ax.bar(x_pos, sample_jac, width, bottom=sample_fr+sample_ko, label='Jacobian', 
           color=colors['jacobian'], alpha=0.8)
    ax.bar(x_pos, sample_tur, width, bottom=sample_fr+sample_ko+sample_jac, label='Turing', 
           color=colors['turing'], alpha=0.8)
    if np.any(sample_ach > 0):
        ax.bar(x_pos, sample_ach, width, bottom=sample_fr+sample_ko+sample_jac+sample_tur, 
               label='ACh Ratio', color=colors['ach_ratio'], alpha=0.8)
    
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Component Breakdown (Sampled)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{int(s)}' for s in sample_steps], rotation=45, ha='right')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
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
    print(f"✓ Loss evolution ratios plot saved to {output_file}")
    
    plt.close(fig)
    
    return str(output_file)
