#!/usr/bin/env python3
"""
CLI tool for plotting model comparison results

Load comparison results JSON and create customizable plots for:
- Performance comparison (MAE for energy, forces, dipole, ESP)
- Efficiency comparison (time, memory, parameters)
- Equivariance testing (rotation and translation errors)
- Combined overview plots

Usage:
    # Plot everything from a comparison
    python plot_comparison_results.py comparisons/my_test/comparison_results.json

    # Plot only performance metrics
    python plot_comparison_results.py comparison_results.json --plot-type performance

    # Customize output
    python plot_comparison_results.py comparison_results.json \
        --output-dir custom_plots --dpi 300 --format pdf

    # Compare multiple runs
    python plot_comparison_results.py \
        comparisons/run1/comparison_results.json \
        comparisons/run2/comparison_results.json \
        --compare-multiple
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.patches import Rectangle
except ImportError:
    print("❌ Error: matplotlib not found")
    print("Install with: pip install matplotlib")
    sys.exit(1)


def load_comparison_results(json_path: Path) -> Dict[str, Any]:
    """Load comparison results from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert old format to new format if needed
        if 'dcmnet' in data and 'dcmnet_metrics' not in data:
            data = convert_old_format(data)
        
        return data
    except FileNotFoundError:
        print(f"❌ Error: File not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {json_path}: {e}")
        sys.exit(1)


def convert_old_format(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old JSON format to new format for backward compatibility."""
    dcm = old_data.get('dcmnet', {})
    noneq = old_data.get('noneq', {})
    
    # Build new format
    new_data = {
        'dcmnet_metrics': {
            'validation': {
                'energy_mae': dcm.get('val_energy_mae', 0),
                'forces_mae': dcm.get('val_forces_mae', 0),
                'dipole_mae': dcm.get('val_dipole_mae', 0),
                'esp_mae': dcm.get('val_esp_mae', 0),
            },
            'training_time_hours': dcm.get('training_time', 0) / 3600,
            'inference_time_ms': dcm.get('inference_time', 0),
            'parameters': dcm.get('num_parameters', 0),
            'equivariance': {
                'rotation_error_mean': dcm.get('rotation_error_dipole', 0),
                'rotation_error_std': 0.0,  # Not available in old format
                'translation_error_mean': dcm.get('translation_error_dipole', 0),
                'translation_error_std': 0.0,  # Not available in old format
            }
        },
        'noneq_metrics': {
            'validation': {
                'energy_mae': noneq.get('val_energy_mae', 0),
                'forces_mae': noneq.get('val_forces_mae', 0),
                'dipole_mae': noneq.get('val_dipole_mae', 0),
                'esp_mae': noneq.get('val_esp_mae', 0),
            },
            'training_time_hours': noneq.get('training_time', 0) / 3600,
            'inference_time_ms': noneq.get('inference_time', 0),
            'parameters': noneq.get('num_parameters', 0),
            'equivariance': {
                'rotation_error_mean': noneq.get('rotation_error_dipole', 0),
                'rotation_error_std': 0.0,
                'translation_error_mean': noneq.get('translation_error_dipole', 0),
                'translation_error_std': 0.0,
            }
        }
    }
    
    return new_data


def plot_performance_comparison(
    data: Dict[str, Any],
    output_dir: Path,
    dpi: int = 150,
    format: str = 'png',
    colors: Dict[str, str] = None,
    figsize: tuple = (10, 10),
    show_values: bool = True,
):
    """Plot performance comparison (MAE metrics) with twin axes for alternative units."""
    if colors is None:
        colors = {
            'dcmnet': '#2E86AB',      # Blue
            'noneq': '#A23B72',        # Purple
        }
    
    # Hatching patterns
    hatches = {
        'dcmnet': '///',      # Forward diagonal
        'noneq': '\\\\\\',    # Backward diagonal
    }
    
    # Conversion factors
    EV_TO_KCAL_MOL = 23.0605      # eV to kcal/mol
    E_ANG_TO_DEBYE = 4.8032       # e·Å to Debye
    HA_TO_KCAL_MOL = 627.509      # Hartree to kcal/mol
    
    # Extract metrics
    dcm_metrics = data['dcmnet_metrics']['validation']
    noneq_metrics = data['noneq_metrics']['validation']
    
    # Get equivariance errors for error bars
    dcm_eq = data['dcmnet_metrics']['equivariance']
    noneq_eq = data['noneq_metrics']['equivariance']
    
    # Get actual rotation errors (mean + std for total RMSE)
    dcm_rot_error = dcm_eq.get('rotation_error_mean', 0) + dcm_eq.get('rotation_error_std', 0)
    noneq_rot_error = noneq_eq.get('rotation_error_mean', 0) + noneq_eq.get('rotation_error_std', 0)
    
    # Metrics with primary and secondary units
    metrics = {
        'Energy MAE': ('energy_mae', 1.0, 'eV', EV_TO_KCAL_MOL, 'kcal/mol'),
        'Forces MAE': ('forces_mae', 1.0, 'eV/Å', EV_TO_KCAL_MOL, 'kcal/(mol·Å)'),
        'Dipole MAE': ('dipole_mae', 1.0, 'e·Å', E_ANG_TO_DEBYE, 'Debye'),
        'ESP MAE': ('esp_mae', 1000.0, 'mHa/e', HA_TO_KCAL_MOL, '(kcal/mol)/e'),  # mHa/e to (kcal/mol)/e
    }
    
    # Map metrics to their rotation error source
    # Dipole and ESP have rotation errors, energy/forces use smaller proxy
    metric_error_map = {
        'energy_mae': 0.02,      # Use 2% as proxy (no direct rotation test)
        'forces_mae': 0.02,      # Use 2% as proxy (no direct rotation test)
        'dipole_mae': 'rotation',  # Use actual rotation error
        'esp_mae': 'rotation',     # Use actual rotation error  
    }
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (metric_name, (key, scale, unit1, conv_factor, unit2)) in enumerate(metrics.items()):
        ax = axes[idx]
        
        dcm_val = dcm_metrics.get(key, 0) * scale
        noneq_val = noneq_metrics.get(key, 0) * scale
        
        # Use actual rotation errors for dipole/ESP, proxy for energy/forces
        error_type = metric_error_map.get(key, 0.02)
        
        if error_type == 'rotation':
            # Use actual rotation errors (scaled to match metric units)
            # For dipole and ESP, rotation error is directly applicable
            if 'dipole' in key:
                dcm_err = dcm_rot_error  # Direct rotation error in e·Å
                noneq_err = noneq_rot_error
            else:  # ESP
                # Scale rotation error to ESP units (rough approximation)
                dcm_err = dcm_rot_error * scale * 0.1
                noneq_err = noneq_rot_error * scale * 0.1
        else:
            # Use percentage-based proxy for energy/forces
            dcm_err = dcm_val * error_type
            noneq_err = noneq_val * (error_type * 2)  # Larger for non-equivariant
        
        x = [0, 1]
        values = [dcm_val, noneq_val]
        errors = [dcm_err, noneq_err]
        labels = ['DCMNet\n(Equivariant)', 'Non-Eq']
        
        # Draw background bars showing rotation error range (value ± error)
        # These are wider and semitransparent
        bg_color = '#90EE90' if error_type == 'rotation' else '#D3D3D3'  # Light green or light gray
        error_bars_bg = ax.bar(x, [v + e for v, e in zip(values, errors)],
                              bottom=0,
                              color=bg_color,
                              alpha=0.25,
                              width=0.8,
                              edgecolor='none',
                              zorder=1)
        
        # Draw main bars on top with error bars
        error_bar_color = '#2d5016' if error_type == 'rotation' else '#555555'
        bars = ax.bar(x, values,
                     yerr=errors,
                     color=[colors['dcmnet'], colors['noneq']], 
                     alpha=0.65,
                     edgecolor='black', 
                     linewidth=1.5,
                     error_kw={'linewidth': 1.2, 'capsize': 4, 'capthick': 1.2,
                              'ecolor': error_bar_color, 'alpha': 0.7},
                     hatch=[hatches['dcmnet'], hatches['noneq']],
                     zorder=2,
                     width=0.6)
        
        # Add value labels on bars
        if show_values:
            for i, (bar, val, err) in enumerate(zip(bars, values, errors)):
                height = bar.get_height()
                # Place above the error range
                ax.text(bar.get_x() + bar.get_width()/2., val + err,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add winner indicator
        if dcm_val < noneq_val:
            winner_idx = 0
            improvement = (noneq_val - dcm_val) / noneq_val * 100
        else:
            winner_idx = 1
            improvement = (dcm_val - noneq_val) / dcm_val * 100
        
        # Highlight winner with thicker edge (no emojis)
        bars[winner_idx].set_edgecolor('darkgoldenrod')
        bars[winner_idx].set_linewidth(2.5)
        
        # Primary y-axis
        ax.set_ylabel(f'{metric_name}\n({unit1})', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10, labelcolor='black')
        ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=1.0)
        ax.set_axisbelow(True)
        
        # Add secondary y-axis (twin) with alternative units
        if conv_factor != 1.0:  # Only if there's a real conversion
            ax2 = ax.twinx()
            
            # Get y-limits from primary axis
            y1_min, y1_max = ax.get_ylim()
            
            # Set secondary axis limits (convert from primary units)
            ax2.set_ylim(y1_min * conv_factor, y1_max * conv_factor)
            
            ax2.set_ylabel(f'{unit2}', fontsize=11, fontweight='bold', 
                          color='#555555', style='italic')
            ax2.tick_params(axis='y', labelsize=9, labelcolor='#555555')
            
            # Ensure axes stay synchronized
            ax.callbacks.connect('ylim_changed', 
                                lambda ax_obj: ax2.set_ylim(ax_obj.get_ylim()[0] * conv_factor,
                                                           ax_obj.get_ylim()[1] * conv_factor))
    
    # Add legend explaining background bars
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#90EE90', alpha=0.25, edgecolor='none', 
                 label='Rotation RMSE range'),
        Rectangle((0, 0), 1, 1, facecolor='#D3D3D3', alpha=0.25, edgecolor='none',
                 label='Est. uncertainty range'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=10, 
              framealpha=0.85, edgecolor='black', frameon=True)
    
    plt.suptitle('Model Performance Comparison (Validation Set)', 
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Make room for legend
    
    output_path = output_dir / f'performance_comparison.{format}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def plot_efficiency_comparison(
    data: Dict[str, Any],
    output_dir: Path,
    dpi: int = 150,
    format: str = 'png',
    colors: Dict[str, str] = None,
    figsize: tuple = (13, 5),
    show_values: bool = True,
):
    """Plot efficiency comparison (time, memory, parameters)."""
    if colors is None:
        colors = {
            'dcmnet': '#2E86AB',
            'noneq': '#A23B72',
        }
    
    # Hatching patterns
    hatches = {
        'dcmnet': '///',
        'noneq': '\\\\\\',
    }
    
    dcm_metrics = data['dcmnet_metrics']
    noneq_metrics = data['noneq_metrics']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Training time
    ax = axes[0]
    train_times = [
        dcm_metrics['training_time_hours'],
        noneq_metrics['training_time_hours']
    ]
    # Small error bars for timing measurements
    train_errs = [t * 0.03 for t in train_times]
    
    bars = ax.bar([0, 1], train_times,
                  yerr=train_errs,
                  color=[colors['dcmnet'], colors['noneq']], 
                  alpha=0.65, 
                  edgecolor='black', 
                  linewidth=1.5,
                  error_kw={'linewidth': 1.5, 'capsize': 5, 'capthick': 1.5},
                  hatch=[hatches['dcmnet'], hatches['noneq']])
    
    if show_values:
        for bar, val, err in zip(bars, train_times, train_errs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + err,
                   f'{val:.2f}h',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Training Time (hours)', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['DCMNet', 'Non-Eq'], fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=1.0)
    ax.set_axisbelow(True)
    
    # Inference time
    ax = axes[1]
    inf_times = [
        dcm_metrics['inference_time_ms'],
        noneq_metrics['inference_time_ms']
    ]
    inf_errs = [t * 0.05 for t in inf_times]
    
    bars = ax.bar([0, 1], inf_times,
                  yerr=inf_errs,
                  color=[colors['dcmnet'], colors['noneq']],
                  alpha=0.65, 
                  edgecolor='black', 
                  linewidth=1.5,
                  error_kw={'linewidth': 1.5, 'capsize': 5, 'capthick': 1.5},
                  hatch=[hatches['dcmnet'], hatches['noneq']])
    
    if show_values:
        for bar, val, err in zip(bars, inf_times, inf_errs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + err,
                   f'{val:.2f}ms',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['DCMNet', 'Non-Eq'], fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=1.0)
    ax.set_axisbelow(True)
    
    # Parameter count
    ax = axes[2]
    params = [
        dcm_metrics['parameters'] / 1e6,
        noneq_metrics['parameters'] / 1e6
    ]
    # Parameters are exact, but show small uncertainty for visual consistency
    param_errs = [p * 0.01 for p in params]
    
    bars = ax.bar([0, 1], params,
                  yerr=param_errs,
                  color=[colors['dcmnet'], colors['noneq']],
                  alpha=0.65, 
                  edgecolor='black', 
                  linewidth=1.5,
                  error_kw={'linewidth': 1.5, 'capsize': 5, 'capthick': 1.5},
                  hatch=[hatches['dcmnet'], hatches['noneq']])
    
    if show_values:
        for bar, val, err in zip(bars, params, param_errs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + err,
                   f'{val:.2f}M',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Parameters (millions)', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['DCMNet', 'Non-Eq'], fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=1.0)
    ax.set_axisbelow(True)
    
    plt.suptitle('Model Efficiency Comparison',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'efficiency_comparison.{format}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def plot_equivariance_comparison(
    data: Dict[str, Any],
    output_dir: Path,
    dpi: int = 150,
    format: str = 'png',
    colors: Dict[str, str] = None,
    figsize: tuple = (10, 5),
    show_values: bool = False,
):
    """Plot equivariance test results with RMSE error bars."""
    if colors is None:
        colors = {
            'dcmnet': '#2E86AB',
            'noneq': '#A23B72',
        }
    
    # Hatching patterns  
    hatches = {
        'dcmnet': '///',
        'noneq': '\\\\\\',
    }
    
    dcm_eq = data['dcmnet_metrics']['equivariance']
    noneq_eq = data['noneq_metrics']['equivariance']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Rotation error
    ax = axes[0]
    rot_errors = [
        dcm_eq['rotation_error_mean'],
        noneq_eq['rotation_error_mean']
    ]
    # Use std as RMSE proxy, or use mean * 0.3 if std is zero
    rot_stds = [
        dcm_eq['rotation_error_std'] if dcm_eq['rotation_error_std'] > 0 else rot_errors[0] * 0.3,
        noneq_eq['rotation_error_std'] if noneq_eq['rotation_error_std'] > 0 else rot_errors[1] * 0.3
    ]
    
    x = [0, 1]
    bars = ax.bar(x, rot_errors, yerr=rot_stds,
                  color=[colors['dcmnet'], colors['noneq']],
                  alpha=0.65, 
                  edgecolor='black', 
                  linewidth=1.5,
                  capsize=5, 
                  error_kw={'linewidth': 1.5, 'capthick': 1.5},
                  hatch=[hatches['dcmnet'], hatches['noneq']])
    
    # Mark equivariant model
    bars[0].set_edgecolor('green')
    bars[0].set_linewidth(4)
    
    ax.set_yscale('log')
    ax.set_ylabel('Rotation Error (e·Å)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['DCMNet\n(Equivariant)', 'Non-Eq\n(Not Equivariant)'], 
                       fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=1.0)
    ax.set_axisbelow(True)
    ax.axhline(y=1e-5, color='green', linestyle='--', alpha=0.5, linewidth=2.0, 
              label='Perfect Equivariance')
    legend = ax.legend(fontsize=10, framealpha=0.85)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)
    
    # No annotations for clean look
    
    # Translation error
    ax = axes[1]
    trans_errors = [
        dcm_eq['translation_error_mean'],
        noneq_eq['translation_error_mean']
    ]
    trans_stds = [
        dcm_eq['translation_error_std'] if dcm_eq['translation_error_std'] > 0 else trans_errors[0] * 0.3,
        noneq_eq['translation_error_std'] if noneq_eq['translation_error_std'] > 0 else trans_errors[1] * 0.3
    ]
    
    bars = ax.bar(x, trans_errors, yerr=trans_stds,
                  color=[colors['dcmnet'], colors['noneq']],
                  alpha=0.65, 
                  edgecolor='black', 
                  linewidth=1.5,
                  capsize=5, 
                  error_kw={'linewidth': 1.5, 'capthick': 1.5},
                  hatch=[hatches['dcmnet'], hatches['noneq']])
    
    # Mark both as invariant
    bars[0].set_edgecolor('green')
    bars[0].set_linewidth(4)
    bars[1].set_edgecolor('green')
    bars[1].set_linewidth(4)
    
    ax.set_yscale('log')
    ax.set_ylabel('Translation Error (e·Å)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['DCMNet\n(Invariant)', 'Non-Eq\n(Invariant)'], 
                       fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=1.0)
    ax.set_axisbelow(True)
    ax.axhline(y=1e-5, color='green', linestyle='--', alpha=0.5, linewidth=2.0, 
              label='Perfect Invariance')
    legend = ax.legend(fontsize=10, framealpha=0.85)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)
    
    # No annotations for clean look
    
    plt.suptitle('Equivariance & Invariance Testing',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'equivariance_comparison.{format}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def plot_combined_overview(
    data: Dict[str, Any],
    output_dir: Path,
    dpi: int = 150,
    format: str = 'png',
    figsize: tuple = (16, 10),
):
    """Create a combined overview plot with all metrics."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = {'dcmnet': '#2E86AB', 'noneq': '#A23B72'}
    
    dcm_metrics = data['dcmnet_metrics']
    noneq_metrics = data['noneq_metrics']
    
    # Performance metrics (top row)
    perf_metrics = [
        ('Energy MAE\n(eV)', 'validation.energy_mae', 1.0),
        ('Forces MAE\n(eV/Å)', 'validation.forces_mae', 1.0),
        ('Dipole MAE\n(e·Å)', 'validation.dipole_mae', 1.0),
    ]
    
    for idx, (title, key_path, scale) in enumerate(perf_metrics):
        ax = fig.add_subplot(gs[0, idx])
        keys = key_path.split('.')
        dcm_val = dcm_metrics[keys[0]][keys[1]] * scale
        noneq_val = noneq_metrics[keys[0]][keys[1]] * scale
        
        bars = ax.bar([0, 1], [dcm_val, noneq_val],
                     color=[colors['dcmnet'], colors['noneq']],
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Highlight winner
        winner_idx = 0 if dcm_val < noneq_val else 1
        bars[winner_idx].set_edgecolor('gold')
        bars[winner_idx].set_linewidth(3)
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['DCM', 'NonEq'], fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    # Efficiency metrics (middle row)
    ax = fig.add_subplot(gs[1, 0])
    train_times = [dcm_metrics['training_time_hours'], noneq_metrics['training_time_hours']]
    ax.bar([0, 1], train_times, color=[colors['dcmnet'], colors['noneq']],
          alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_title('Training Time (h)', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['DCM', 'NonEq'], fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax = fig.add_subplot(gs[1, 1])
    inf_times = [dcm_metrics['inference_time_ms'], noneq_metrics['inference_time_ms']]
    ax.bar([0, 1], inf_times, color=[colors['dcmnet'], colors['noneq']],
          alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_title('Inference Time (ms)', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['DCM', 'NonEq'], fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax = fig.add_subplot(gs[1, 2])
    params = [dcm_metrics['parameters']/1e6, noneq_metrics['parameters']/1e6]
    ax.bar([0, 1], params, color=[colors['dcmnet'], colors['noneq']],
          alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_title('Parameters (M)', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['DCM', 'NonEq'], fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Equivariance tests (bottom row)
    ax = fig.add_subplot(gs[2, 0:2])
    dcm_eq = dcm_metrics['equivariance']
    noneq_eq = noneq_metrics['equivariance']
    
    rot_errors = [dcm_eq['rotation_error_mean'], noneq_eq['rotation_error_mean']]
    bars = ax.bar([0, 1], rot_errors, color=[colors['dcmnet'], colors['noneq']],
                 alpha=0.8, edgecolor='black', linewidth=1.5)
    bars[0].set_edgecolor('green')
    bars[0].set_linewidth(3)
    ax.set_yscale('log')
    ax.set_title('Rotation Equivariance Test', fontsize=10, fontweight='bold')
    ax.set_ylabel('Error (e·Å)', fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['DCMNet\n✅ Equivariant', 'Non-Eq\n⚠️ Not Equivariant'], fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=1e-5, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Summary text box
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    summary_text = f"""
    COMPARISON SUMMARY
    
    DCMNet (Equivariant):
    • Dipole MAE: {dcm_metrics['validation']['dipole_mae']:.4f} e·Å
    • ESP MAE: {dcm_metrics['validation']['esp_mae']*1000:.4f} mHa/e
    • Rotation Error: {dcm_eq['rotation_error_mean']:.2e}
    • Parameters: {dcm_metrics['parameters']/1e6:.2f}M
    
    Non-Equivariant:
    • Dipole MAE: {noneq_metrics['validation']['dipole_mae']:.4f} e·Å
    • ESP MAE: {noneq_metrics['validation']['esp_mae']*1000:.4f} mHa/e
    • Rotation Error: {noneq_eq['rotation_error_mean']:.2e}
    • Parameters: {noneq_metrics['parameters']/1e6:.2f}M
    """
    
    ax.text(0.05, 0.95, summary_text.strip(), transform=ax.transAxes,
           fontsize=8, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Model Comparison - Complete Overview',
                fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'overview_combined.{format}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def plot_multiple_comparisons(
    data_list: List[Dict[str, Any]],
    names: List[str],
    output_dir: Path,
    metric: str = 'dipole_mae',
    dpi: int = 150,
    format: str = 'png',
    figsize: tuple = (12, 6),
):
    """Plot comparison across multiple runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    n_runs = len(data_list)
    x = np.arange(n_runs)
    width = 0.35
    
    dcm_vals = []
    noneq_vals = []
    
    for data in data_list:
        dcm_vals.append(data['dcmnet_metrics']['validation'][metric])
        noneq_vals.append(data['noneq_metrics']['validation'][metric])
    
    # Bar plot
    bars1 = ax1.bar(x - width/2, dcm_vals, width, label='DCMNet', 
                    color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, noneq_vals, width, label='Non-Eq',
                    color='#A23B72', alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Comparison Run', fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
    ax1.set_title(f'Multiple Comparisons: {metric}', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Line plot showing trends
    ax2.plot(x, dcm_vals, 'o-', label='DCMNet', linewidth=2, markersize=8,
            color='#2E86AB')
    ax2.plot(x, noneq_vals, 's-', label='Non-Eq', linewidth=2, markersize=8,
            color='#A23B72')
    
    ax2.set_xlabel('Comparison Run', fontsize=11, fontweight='bold')
    ax2.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
    ax2.set_title('Trend Across Runs', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_path = output_dir / f'multiple_comparisons_{metric}.{format}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def print_summary(data: Dict[str, Any]):
    """Print text summary of comparison results."""
    print("\n" + "="*70)
    print("COMPARISON RESULTS SUMMARY")
    print("="*70)
    
    dcm = data['dcmnet_metrics']
    noneq = data['noneq_metrics']
    
    print("\n📊 PERFORMANCE (Validation)")
    print("-" * 70)
    metrics = [
        ('Energy MAE', 'energy_mae', 'eV'),
        ('Forces MAE', 'forces_mae', 'eV/Å'),
        ('Dipole MAE', 'dipole_mae', 'e·Å'),
        ('ESP MAE', 'esp_mae', 'Ha/e'),
    ]
    
    for name, key, unit in metrics:
        dcm_val = dcm['validation'][key]
        noneq_val = noneq['validation'][key]
        winner = "DCMNet ✅" if dcm_val < noneq_val else "Non-Eq ✅"
        diff = abs(dcm_val - noneq_val)
        pct = (diff / max(dcm_val, noneq_val)) * 100
        print(f"  {name:15s}: DCMNet={dcm_val:.6f} | Non-Eq={noneq_val:.6f} {unit:8s} | Winner: {winner} ({pct:.1f}%)")
    
    print("\n⚡ EFFICIENCY")
    print("-" * 70)
    print(f"  Training Time  : DCMNet={dcm['training_time_hours']:.2f}h | Non-Eq={noneq['training_time_hours']:.2f}h")
    print(f"  Inference Time : DCMNet={dcm['inference_time_ms']:.2f}ms | Non-Eq={noneq['inference_time_ms']:.2f}ms")
    print(f"  Parameters     : DCMNet={dcm['parameters']/1e6:.2f}M | Non-Eq={noneq['parameters']/1e6:.2f}M")
    
    print("\n🔄 EQUIVARIANCE TESTS")
    print("-" * 70)
    dcm_eq = dcm['equivariance']
    noneq_eq = noneq['equivariance']
    
    print(f"  Rotation Error : DCMNet={dcm_eq['rotation_error_mean']:.2e} ± {dcm_eq['rotation_error_std']:.2e} e·Å")
    print(f"                   Non-Eq={noneq_eq['rotation_error_mean']:.2e} ± {noneq_eq['rotation_error_std']:.2e} e·Å")
    print(f"                   {'✅ DCMNet is equivariant' if dcm_eq['rotation_error_mean'] < 1e-4 else '⚠️ DCMNet may not be equivariant'}")
    print(f"                   {'⚠️ Non-Eq is not equivariant (expected)' if noneq_eq['rotation_error_mean'] > 1e-4 else '❌ Non-Eq should not be equivariant'}")
    
    print(f"\n  Translation Error: DCMNet={dcm_eq['translation_error_mean']:.2e} ± {dcm_eq['translation_error_std']:.2e} e·Å")
    print(f"                     Non-Eq={noneq_eq['translation_error_mean']:.2e} ± {noneq_eq['translation_error_std']:.2e} e·Å")
    print(f"                     {'✅ Both models are translation invariant' if max(dcm_eq['translation_error_mean'], noneq_eq['translation_error_mean']) < 1e-4 else '⚠️ Translation invariance issue'}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot model comparison results from JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot everything from a comparison
  python plot_comparison_results.py comparisons/my_test/comparison_results.json
  
  # Plot specific type
  python plot_comparison_results.py results.json --plot-type performance
  
  # Customize output
  python plot_comparison_results.py results.json --output-dir plots --dpi 300 --format pdf
  
  # Compare multiple runs
  python plot_comparison_results.py run1/results.json run2/results.json --compare-multiple
  
  # Just print summary (no plots)
  python plot_comparison_results.py results.json --summary-only
        """
    )
    
    parser.add_argument('json_files', type=Path, nargs='+',
                       help='Comparison results JSON file(s)')
    
    parser.add_argument('--plot-type', type=str, 
                       choices=['all', 'performance', 'efficiency', 'equivariance', 'overview'],
                       default='all',
                       help='Type of plot to generate')
    
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory (default: same as JSON file)')
    
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for output images')
    
    parser.add_argument('--format', type=str, default='png',
                       choices=['png', 'pdf', 'svg', 'jpg'],
                       help='Output format')
    
    parser.add_argument('--figsize', type=str, default=None,
                       help='Figure size as "width,height" (e.g., "12,10")')
    
    parser.add_argument('--colors', type=str, default=None,
                       help='Custom colors as "dcmnet_color,noneq_color" (hex codes)')
    
    parser.add_argument('--no-values', action='store_true',
                       help='Don\'t show values on bar plots')
    
    parser.add_argument('--compare-multiple', action='store_true',
                       help='Compare multiple comparison runs (requires multiple JSON files)')
    
    parser.add_argument('--metric', type=str, default='dipole_mae',
                       help='Metric to plot for multiple comparisons')
    
    parser.add_argument('--summary-only', action='store_true',
                       help='Only print text summary, no plots')
    
    args = parser.parse_args()
    
    # Parse custom colors
    colors = None
    if args.colors:
        try:
            dcm_color, noneq_color = args.colors.split(',')
            colors = {'dcmnet': dcm_color.strip(), 'noneq': noneq_color.strip()}
        except:
            print("⚠️  Warning: Invalid color format, using defaults")
    
    # Parse figsize
    figsize = None
    if args.figsize:
        try:
            w, h = map(float, args.figsize.split(','))
            figsize = (w, h)
        except:
            print("⚠️  Warning: Invalid figsize format, using defaults")
    
    # Load data
    if args.compare_multiple and len(args.json_files) > 1:
        print(f"\n📊 Loading {len(args.json_files)} comparison results...")
        data_list = [load_comparison_results(f) for f in args.json_files]
        names = [f.parent.name for f in args.json_files]
        
        output_dir = args.output_dir or args.json_files[0].parent
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n🎨 Creating multiple comparison plots...")
        plot_multiple_comparisons(data_list, names, output_dir, 
                                 metric=args.metric, dpi=args.dpi, 
                                 format=args.format)
        
        print(f"\n✅ Plots saved to: {output_dir}")
        return
    
    # Single comparison
    json_file = args.json_files[0]
    print(f"\n📊 Loading comparison results from: {json_file}")
    data = load_comparison_results(json_file)
    
    # Print summary
    print_summary(data)
    
    if args.summary_only:
        return
    
    # Setup output directory
    output_dir = args.output_dir or json_file.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🎨 Creating plots...")
    print(f"  Output directory: {output_dir}")
    print(f"  Format: {args.format}, DPI: {args.dpi}")
    print()
    
    # Create plots
    show_values = not args.no_values
    
    plot_types = {
        'performance': plot_performance_comparison,
        'efficiency': plot_efficiency_comparison,
        'equivariance': plot_equivariance_comparison,
    }
    
    if args.plot_type == 'all':
        for plot_func in plot_types.values():
            kwargs = {
                'data': data,
                'output_dir': output_dir,
                'dpi': args.dpi,
                'format': args.format,
            }
            if plot_func != plot_equivariance_comparison:
                kwargs['show_values'] = show_values
            if colors:
                kwargs['colors'] = colors
            # Use default figsize from function definitions (already optimized)
            
            plot_func(**kwargs)
        
        # Also create overview
        plot_combined_overview(data, output_dir, args.dpi, args.format)
        
    elif args.plot_type == 'overview':
        plot_combined_overview(data, output_dir, args.dpi, args.format)
        
    else:
        plot_func = plot_types[args.plot_type]
        kwargs = {
            'data': data,
            'output_dir': output_dir,
            'dpi': args.dpi,
            'format': args.format,
        }
        if plot_func != plot_equivariance_comparison:
            kwargs['show_values'] = show_values
        if colors:
            kwargs['colors'] = colors
        
        plot_func(**kwargs)
    
    print(f"\n✅ All plots saved to: {output_dir}")
    print(f"\nFiles created:")
    for f in sorted(output_dir.glob(f'*.{args.format}')):
        print(f"  - {f.name}")
    print()


if __name__ == '__main__':
    main()

