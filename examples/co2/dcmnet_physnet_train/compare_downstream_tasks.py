#!/usr/bin/env python3
"""
Comprehensive Downstream Task Comparison: DCMNet vs Non-Equivariant

Head-to-head comparison of trained models on all downstream tasks:
- Harmonic vibrational analysis (frequencies, IR)
- Molecular dynamics simulations
- IR spectra from MD (anharmonic)
- Raman spectra
- Charge distribution analysis over CO2 configurations

Usage:
    # Quick comparison
    python compare_downstream_tasks.py \
        --checkpoint-dcm comparisons/test1/dcmnet_equivariant/best_params.pkl \
        --checkpoint-noneq comparisons/test1/noneq_model/best_params.pkl \
        --quick
    
    # Full comparison
    python compare_downstream_tasks.py \
        --checkpoint-dcm dcmnet/best_params.pkl \
        --checkpoint-noneq noneq/best_params.pkl \
        --full \
        --md-steps 50000 \
        --output-dir downstream_comparison
    
    # With charge surface analysis
    python compare_downstream_tasks.py \
        --checkpoint-dcm dcmnet/best_params.pkl \
        --checkpoint-noneq noneq/best_params.pkl \
        --analyze-charges \
        --theta-range 160 180 20 \
        --r-range 1.0 1.3 20
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import argparse
import json
from typing import Dict, Tuple, Any, List
import time
from dataclasses import dataclass

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from ase import Atoms, units
from ase.optimize import BFGS

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from trainer import JointPhysNetDCMNet, JointPhysNetNonEquivariant
from dynamics_calculator import (
    JointPhysNetDCMNetCalculator,
    optimize_geometry,
    calculate_frequencies,
    calculate_ir_spectrum,
    run_molecular_dynamics,
    compute_ir_from_md,
)
from raman_calculator import (
    calculate_raman_spectrum,
    compute_polarizability_finite_field,
)

import e3x


@dataclass
class DownstreamMetrics:
    """Metrics from downstream tasks."""
    # Harmonic analysis
    frequencies: np.ndarray = None
    ir_intensities: np.ndarray = None
    raman_intensities: np.ndarray = None
    optimized_energy: float = None
    optimization_steps: int = None
    
    # MD results
    md_energy_std: float = None
    md_temperature_mean: float = None
    md_ir_frequencies: np.ndarray = None
    md_ir_intensities: np.ndarray = None
    md_time: float = None
    
    # Charge analysis
    charges_vs_geometry: Dict = None
    
    # Timing
    harmonic_time: float = None
    raman_time: float = None


def load_model_and_params(checkpoint_path: Path, is_noneq: bool = False):
    """Load model and parameters from checkpoint."""
    checkpoint_path = Path(checkpoint_path).resolve()  # Resolve to absolute path
    print(f"  Loading checkpoint: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}\n  Current dir: {Path.cwd()}")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint_data, dict) and 'params' in checkpoint_data:
        params = checkpoint_data['params']
    else:
        params = checkpoint_data
    
    # Load model config (required!)
    config_path = checkpoint_path.parent / 'model_config.pkl'
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}. Cannot determine model architecture.")
    
    with open(config_path, 'rb') as f:
        saved_config = pickle.load(f)
    
    print(f"  Loaded model config from: {config_path}")
    
    # Extract configs directly from saved file
    physnet_config = saved_config['physnet_config']
    mix_coulomb_energy = saved_config.get('mix_coulomb_energy', False)
    
    # Create model (different classes for equivariant vs non-equivariant)
    if is_noneq:
        # Non-equivariant model
        noneq_config = saved_config['noneq_config']
        
        model = JointPhysNetNonEquivariant(
            physnet_config=physnet_config,
            noneq_config=noneq_config,
            mix_coulomb_energy=mix_coulomb_energy,
        )
    else:
        # DCMNet (equivariant)
        dcmnet_config = saved_config['dcmnet_config']
        
        model = JointPhysNetDCMNet(
            physnet_config=physnet_config,
            dcmnet_config=dcmnet_config,
            mix_coulomb_energy=mix_coulomb_energy,
        )
    
    # The params should already be in the correct structure from training
    # Just ensure they're wrapped properly if needed
    if isinstance(params, dict) and 'params' not in params and 'physnet' in params:
        # Params are unwrapped, wrap them
        params = {'params': params}
    
    config = saved_config  # Return for reference
    
    return model, params, config


def create_co2_configurations(
    theta_values: np.ndarray,
    r1_values: np.ndarray,
    r2_values: np.ndarray
) -> List[Atoms]:
    """
    Create CO2 configurations varying internal coordinates.
    
    Parameters
    ----------
    theta_values : array
        Bond angles in degrees
    r1_values : array
        C-O bond 1 distances in Angstroms
    r2_values : array
        C-O bond 2 distances in Angstroms
    
    Returns
    -------
    configurations : List[Atoms]
        CO2 atoms at different geometries
    """
    configurations = []
    
    for theta in theta_values:
        for r1 in r1_values:
            for r2 in r2_values:
                # Build CO2 geometry
                # C at origin, O1 along +x, O2 at angle theta
                theta_rad = np.radians(theta)
                
                positions = np.array([
                    [0.0, 0.0, 0.0],                    # C
                    [r1, 0.0, 0.0],                     # O1
                    [r2 * np.cos(theta_rad),            # O2
                     r2 * np.sin(theta_rad), 0.0]
                ])
                
                atoms = Atoms('CO2', positions=positions)
                configurations.append(atoms)
    
    return configurations


def analyze_charges_vs_geometry(
    calculator: JointPhysNetDCMNetCalculator,
    theta_range: Tuple[float, float, int],
    r_range: Tuple[float, float, int],
    output_dir: Path
) -> Dict:
    """
    Analyze how charges vary with CO2 internal coordinates.
    
    Parameters
    ----------
    calculator : Calculator
        Model calculator
    theta_range : tuple
        (min_theta, max_theta, n_points) in degrees
    r_range : tuple
        (min_r, max_r, n_points) in Angstroms
    output_dir : Path
        Output directory
    
    Returns
    -------
    results : dict
        Charge analysis results
    """
    print("\n" + "="*70)
    print("CHARGE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Generate grids
    theta_min, theta_max, n_theta = theta_range
    r_min, r_max, n_r = r_range
    
    thetas = np.linspace(theta_min, theta_max, n_theta)
    r1s = np.linspace(r_min, r_max, n_r)
    r2s = np.linspace(r_min, r_max, n_r)
    
    print(f"\nScanning CO2 configuration space:")
    print(f"  Theta: {theta_min}° to {theta_max}° ({n_theta} points)")
    print(f"  R1:    {r_min} to {r_max} Å ({n_r} points)")
    print(f"  R2:    {r_min} to {r_max} Å ({n_r} points)")
    print(f"  Total configurations: {n_theta * n_r * n_r}")
    
    # Storage
    charges_c = []
    charges_o1 = []
    charges_o2 = []
    dipoles = []
    geometries = []
    
    # Scan configurations
    for theta in thetas:
        for r1 in r1s:
            for r2 in r2s:
                theta_rad = np.radians(theta)
                positions = np.array([
                    [0.0, 0.0, 0.0],
                    [r1, 0.0, 0.0],
                    [r2 * np.cos(theta_rad), r2 * np.sin(theta_rad), 0.0]
                ])
                
                atoms = Atoms('CO2', positions=positions)
                atoms.calc = calculator
                
                try:
                    charges = atoms.get_charges()
                    dipole = atoms.get_dipole_moment()
                    
                    charges_c.append(charges[0])
                    charges_o1.append(charges[1])
                    charges_o2.append(charges[2])
                    dipoles.append(dipole)
                    geometries.append((theta, r1, r2))
                except:
                    # Skip configurations that fail
                    pass
    
    results = {
        'thetas': thetas,
        'r1s': r1s,
        'r2s': r2s,
        'charges_c': np.array(charges_c),
        'charges_o1': np.array(charges_o1),
        'charges_o2': np.array(charges_o2),
        'dipoles': np.array(dipoles),
        'geometries': np.array(geometries),
    }
    
    print(f"\n  Computed charges for {len(geometries)} configurations")
    
    return results


def plot_charge_surfaces(
    results_dcm: Dict,
    results_noneq: Dict,
    output_dir: Path,
    dpi: int = 150
):
    """Plot charge distributions over CO2 configuration space."""
    
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.4)
    
    # Extract data
    geoms_dcm = results_dcm['geometries']
    geoms_noneq = results_noneq['geometries']
    
    thetas_dcm = geoms_dcm[:, 0]
    r1s_dcm = geoms_dcm[:, 1]
    
    # Plot 1-3: Charges vs theta (at equilibrium r)
    r_eq = 1.16  # Approximate equilibrium CO2 bond length
    
    for idx, (atom, charge_key) in enumerate([('C', 'charges_c'), 
                                               ('O1', 'charges_o1'), 
                                               ('O2', 'charges_o2')]):
        ax = fig.add_subplot(gs[0, idx])
        
        # Filter for equilibrium r
        mask_dcm = np.abs(r1s_dcm - r_eq) < 0.02
        mask_noneq = np.abs(geoms_noneq[:, 1] - r_eq) < 0.02
        
        ax.scatter(thetas_dcm[mask_dcm], results_dcm[charge_key][mask_dcm],
                  alpha=0.6, s=20, label='DCMNet', color='#2E86AB')
        ax.scatter(geoms_noneq[mask_noneq, 0], results_noneq[charge_key][mask_noneq],
                  alpha=0.6, s=20, label='Non-Eq', color='#A23B72')
        
        ax.set_xlabel('Angle (degrees)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Charge on {atom} (e)', fontsize=11, fontweight='bold')
        ax.set_title(f'{atom} Charge vs Angle', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
    
    # Plot 4-6: Charges vs r1 (at linear geometry)
    theta_linear = 180.0
    
    for idx, (atom, charge_key) in enumerate([('C', 'charges_c'), 
                                               ('O1', 'charges_o1'), 
                                               ('O2', 'charges_o2')]):
        ax = fig.add_subplot(gs[1, idx])
        
        # Filter for linear geometry
        mask_dcm = np.abs(thetas_dcm - theta_linear) < 2.0
        mask_noneq = np.abs(geoms_noneq[:, 0] - theta_linear) < 2.0
        
        ax.scatter(r1s_dcm[mask_dcm], results_dcm[charge_key][mask_dcm],
                  alpha=0.6, s=20, label='DCMNet', color='#2E86AB')
        ax.scatter(geoms_noneq[mask_noneq, 1], results_noneq[charge_key][mask_noneq],
                  alpha=0.6, s=20, label='Non-Eq', color='#A23B72')
        
        ax.set_xlabel('Bond Length (Å)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Charge on {atom} (e)', fontsize=11, fontweight='bold')
        ax.set_title(f'{atom} Charge vs Bond Length', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
    
    # Plot 7-9: 2D contour plots (theta vs r)
    from scipy.interpolate import griddata
    
    for idx, (atom, charge_key) in enumerate([('C', 'charges_c'), 
                                               ('O1', 'charges_o1'), 
                                               ('O2', 'charges_o2')]):
        ax = fig.add_subplot(gs[2, idx])
        
        # Create grid for contour plot
        theta_grid = np.linspace(thetas_dcm.min(), thetas_dcm.max(), 50)
        r_grid = np.linspace(r1s_dcm.min(), r1s_dcm.max(), 50)
        theta_mesh, r_mesh = np.meshgrid(theta_grid, r_grid)
        
        # Interpolate DCMNet charges
        charges_interp = griddata(
            (thetas_dcm, r1s_dcm),
            results_dcm[charge_key],
            (theta_mesh, r_mesh),
            method='cubic'
        )
        
        # Contour plot
        contour = ax.contourf(theta_mesh, r_mesh, charges_interp, 
                             levels=15, cmap='RdBu_r', alpha=0.7)
        ax.contour(theta_mesh, r_mesh, charges_interp, 
                  levels=15, colors='black', linewidths=0.5, alpha=0.3)
        
        plt.colorbar(contour, ax=ax, label='Charge (e)')
        
        ax.set_xlabel('Angle (degrees)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Bond Length (Å)', fontsize=11, fontweight='bold')
        ax.set_title(f'{atom} Charge Surface (DCMNet)', fontsize=12, fontweight='bold')
    
    plt.suptitle('Charge Distribution Analysis', fontsize=15, fontweight='bold')
    
    output_path = output_dir / 'charge_surfaces.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_spectroscopy_comparison(
    metrics_dcm: DownstreamMetrics,
    metrics_noneq: DownstreamMetrics,
    output_dir: Path,
    dpi: int = 150
):
    """Plot IR and Raman spectra comparison."""
    
    if not HAS_MATPLOTLIB:
        return
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    colors = {'dcm': '#2E86AB', 'noneq': '#A23B72'}
    
    # 1. Harmonic IR comparison
    ax = fig.add_subplot(gs[0, 0])
    if metrics_dcm.frequencies is not None:
        ax.stem(metrics_dcm.frequencies, metrics_dcm.ir_intensities,
               linefmt='C0-', markerfmt='C0o', label='DCMNet', basefmt=' ')
    if metrics_noneq.frequencies is not None:
        ax.stem(metrics_noneq.frequencies, metrics_noneq.ir_intensities,
               linefmt='C1-', markerfmt='C1s', label='Non-Eq', basefmt=' ')
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, fontweight='bold')
    ax.set_ylabel('IR Intensity (km/mol)', fontsize=12, fontweight='bold')
    ax.set_title('Harmonic IR Spectrum', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 2500)
    
    # 2. MD IR comparison
    ax = fig.add_subplot(gs[0, 1])
    if metrics_dcm.md_ir_frequencies is not None:
        ax.plot(metrics_dcm.md_ir_frequencies, metrics_dcm.md_ir_intensities,
               linewidth=1.5, label='DCMNet', color=colors['dcm'], alpha=0.7)
    if metrics_noneq.md_ir_frequencies is not None:
        ax.plot(metrics_noneq.md_ir_frequencies, metrics_noneq.md_ir_intensities,
               linewidth=1.5, label='Non-Eq', color=colors['noneq'], alpha=0.7)
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Intensity (arb. units)', fontsize=12, fontweight='bold')
    ax.set_title('IR from MD (Anharmonic)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 2500)
    
    # 3. Raman comparison
    ax = fig.add_subplot(gs[1, 0])
    if metrics_dcm.raman_intensities is not None:
        ax.stem(metrics_dcm.frequencies, metrics_dcm.raman_intensities,
               linefmt='C0-', markerfmt='C0o', label='DCMNet', basefmt=' ')
    if metrics_noneq.raman_intensities is not None:
        ax.stem(metrics_noneq.frequencies, metrics_noneq.raman_intensities,
               linefmt='C1-', markerfmt='C1s', label='Non-Eq', basefmt=' ')
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Raman Activity (Å⁴/amu)', fontsize=12, fontweight='bold')
    ax.set_title('Raman Spectrum', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 2500)
    
    # 4. Comparison statistics
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    
    # Build comparison text
    comparison_text = "SPECTROSCOPY COMPARISON\n" + "="*40 + "\n\n"
    
    if metrics_dcm.frequencies is not None:
        comparison_text += "Harmonic Frequencies (cm⁻¹):\n"
        comparison_text += f"  DCMNet:  {', '.join([f'{f:.1f}' for f in metrics_dcm.frequencies])}\n"
        comparison_text += f"  Non-Eq:  {', '.join([f'{f:.1f}' for f in metrics_noneq.frequencies])}\n\n"
    
    comparison_text += "Optimization:\n"
    comparison_text += f"  DCMNet:  {metrics_dcm.optimization_steps} steps, "
    comparison_text += f"E={metrics_dcm.optimized_energy:.6f} eV\n"
    comparison_text += f"  Non-Eq:  {metrics_noneq.optimization_steps} steps, "
    comparison_text += f"E={metrics_noneq.optimized_energy:.6f} eV\n\n"
    
    if metrics_dcm.md_temperature_mean is not None:
        comparison_text += "MD Statistics:\n"
        comparison_text += f"  DCMNet:  T={metrics_dcm.md_temperature_mean:.1f}K, "
        comparison_text += f"σ(E)={metrics_dcm.md_energy_std:.4f} eV\n"
        comparison_text += f"  Non-Eq:  T={metrics_noneq.md_temperature_mean:.1f}K, "
        comparison_text += f"σ(E)={metrics_noneq.md_energy_std:.4f} eV\n\n"
    
    comparison_text += "Timing:\n"
    comparison_text += f"  Harmonic: DCM={metrics_dcm.harmonic_time:.1f}s, "
    comparison_text += f"NonEq={metrics_noneq.harmonic_time:.1f}s\n"
    if metrics_dcm.md_time is not None:
        comparison_text += f"  MD: DCM={metrics_dcm.md_time:.1f}s, "
        comparison_text += f"NonEq={metrics_noneq.md_time:.1f}s\n"
    
    ax.text(0.05, 0.95, comparison_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Spectroscopy Comparison: DCMNet vs Non-Eq',
                fontsize=15, fontweight='bold')
    
    output_path = output_dir / 'spectroscopy_comparison.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def run_harmonic_analysis(
    calculator: JointPhysNetDCMNetCalculator,
    molecule: str = 'CO2'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Atoms, int, float]:
    """
    Run complete harmonic vibrational analysis.
    
    Returns
    -------
    frequencies : array
        Vibrational frequencies in cm^-1
    ir_intensities : array
        IR intensities in km/mol
    raman_intensities : array
        Raman activities in Å^4/amu
    optimized_atoms : Atoms
        Optimized geometry
    opt_steps : int
        Number of optimization steps
    opt_energy : float
        Optimized energy
    """
    start_time = time.time()
    
    # Create initial geometry
    if molecule == 'CO2':
        atoms = Atoms('CO2', positions=[[0,0,0], [1.16,0,0], [-1.16,0,0]])
    else:
        raise ValueError(f"Molecule {molecule} not implemented")
    
    atoms.calc = calculator
    
    # Optimize
    print(f"  Optimizing geometry...")
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=1e-4)
    opt_steps = opt.nsteps
    opt_energy = atoms.get_potential_energy()
    
    # Calculate frequencies
    print(f"  Calculating frequencies...")
    freqs, modes = calculate_frequencies(atoms, delta=0.01)
    
    # Calculate IR intensities
    print(f"  Calculating IR intensities...")
    ir_intensities = calculate_ir_spectrum(atoms, freqs, modes)
    
    # Calculate Raman
    print(f"  Calculating Raman activities...")
    try:
        raman_intensities = calculate_raman_spectrum(atoms, freqs, modes)
    except:
        raman_intensities = np.zeros_like(freqs)
        print("    Warning: Raman calculation failed, using zeros")
    
    elapsed = time.time() - start_time
    
    return freqs, ir_intensities, raman_intensities, atoms.copy(), opt_steps, opt_energy, elapsed


def run_md_analysis(
    calculator: JointPhysNetDCMNetCalculator,
    molecule: str = 'CO2',
    temperature: float = 300,
    nsteps: int = 10000,
    timestep: float = 0.5
) -> Dict:
    """
    Run molecular dynamics and extract IR spectrum.
    
    Returns
    -------
    results : dict
        MD results including IR spectrum from autocorrelation
    """
    start_time = time.time()
    
    # Create initial geometry
    if molecule == 'CO2':
        atoms = Atoms('CO2', positions=[[0,0,0], [1.16,0,0], [-1.16,0,0]])
    else:
        raise ValueError(f"Molecule {molecule} not implemented")
    
    atoms.calc = calculator
    
    print(f"  Running MD simulation ({nsteps} steps)...")
    md_results = run_molecular_dynamics(
        atoms, calculator,
        ensemble='nvt',
        temperature=temperature,
        timestep=timestep,
        nsteps=nsteps,
        save_dipoles=True,
        output_dir=None  # Don't save files
    )
    
    elapsed = time.time() - start_time
    md_results['md_time'] = elapsed
    
    # Compute IR from autocorrelation
    print(f"  Computing IR from MD...")
    try:
        ir_results = compute_ir_from_md(md_results, output_dir=None)
        md_results.update(ir_results)
    except:
        print("    Warning: IR from MD failed")
        md_results['ir_frequencies'] = None
        md_results['ir_intensities'] = None
    
    return md_results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive downstream task comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model checkpoints
    parser.add_argument('--checkpoint-dcm', type=Path, required=True,
                       help='DCMNet checkpoint path')
    parser.add_argument('--checkpoint-noneq', type=Path, required=True,
                       help='Non-equivariant checkpoint path')
    
    # Analysis options
    parser.add_argument('--check-only', action='store_true',
                       help='Quick smoke test: just load models and exit (~5 sec)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick analysis (harmonic only, ~5 min)')
    parser.add_argument('--full', action='store_true',
                       help='Full analysis (harmonic + MD + Raman, ~30-60 min)')
    parser.add_argument('--analyze-charges', action='store_true',
                       help='Analyze charges vs CO2 geometry')
    
    # MD parameters
    parser.add_argument('--md-steps', type=int, default=10000,
                       help='MD steps')
    parser.add_argument('--temperature', type=float, default=300,
                       help='MD temperature (K)')
    parser.add_argument('--timestep', type=float, default=0.5,
                       help='MD timestep (fs)')
    
    # Charge analysis parameters
    parser.add_argument('--theta-range', type=float, nargs=3, 
                       default=[160, 180, 20],
                       help='Theta range: min max npoints (degrees)')
    parser.add_argument('--r-range', type=float, nargs=3,
                       default=[1.0, 1.3, 20],
                       help='Bond length range: min max npoints (Angstroms)')
    
    # Output
    parser.add_argument('--output-dir', type=Path, 
                       default=Path('downstream_comparison'),
                       help='Output directory')
    parser.add_argument('--dpi', type=int, default=200,
                       help='Plot DPI')
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("DOWNSTREAM TASK COMPARISON: DCMNet vs Non-Equivariant")
    print("="*70)
    
    # Load models
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)
    
    print("\nDCMNet (Equivariant):")
    model_dcm, params_dcm, config_dcm = load_model_and_params(args.checkpoint_dcm, is_noneq=False)
    
    print("\nNon-Equivariant:")
    model_noneq, params_noneq, config_noneq = load_model_and_params(args.checkpoint_noneq, is_noneq=True)
    
    # Create calculators
    cutoff = config_dcm.get('cutoff', 6.0)
    calc_dcm = JointPhysNetDCMNetCalculator(model_dcm, params_dcm, cutoff=cutoff)
    calc_noneq = JointPhysNetDCMNetCalculator(model_noneq, params_noneq, cutoff=cutoff)
    
    print("\nCalculators created successfully")
    
    # Quick smoke test - just check models load
    if args.check_only:
        print("\n" + "="*70)
        print("CHECK-ONLY MODE: Models loaded successfully!")
        print("="*70)
        print("\nModel configurations:")
        print(f"\nDCMNet PhysNet config:")
        print(f"  natoms: {model_dcm.physnet_config['natoms']}")
        print(f"  cutoff: {model_dcm.physnet_config['cutoff']}")
        if hasattr(model_dcm, 'dcmnet_config'):
            print(f"\nDCMNet config:")
            print(f"  n_dcm: {model_dcm.dcmnet_config['n_dcm']}")
            print(f"  features: {model_dcm.dcmnet_config['features']}")
        
        print(f"\nNon-Eq PhysNet config:")
        print(f"  natoms: {model_noneq.physnet_config['natoms']}")
        print(f"  cutoff: {model_noneq.physnet_config['cutoff']}")
        if hasattr(model_noneq, 'noneq_config'):
            print(f"\nNon-Eq config:")
            print(f"  n_dcm: {model_noneq.noneq_config['n_dcm']}")
            print(f"  features: {model_noneq.noneq_config['features']}")
        
        print("\n✅ All checks passed! Models are ready to use.")
        print("Run without --check-only to perform actual analysis.")
        return
    
    # Initialize metrics
    metrics_dcm = DownstreamMetrics()
    metrics_noneq = DownstreamMetrics()
    
    # ========================================
    # 1. HARMONIC ANALYSIS
    # ========================================
    print("\n" + "="*70)
    print("1. HARMONIC VIBRATIONAL ANALYSIS")
    print("="*70)
    
    print("\nDCMNet:")
    (metrics_dcm.frequencies, metrics_dcm.ir_intensities, 
     metrics_dcm.raman_intensities, atoms_dcm, 
     metrics_dcm.optimization_steps, metrics_dcm.optimized_energy,
     metrics_dcm.harmonic_time) = run_harmonic_analysis(calc_dcm, 'CO2')
    
    print(f"  Frequencies: {', '.join([f'{f:.1f}' for f in metrics_dcm.frequencies])} cm⁻¹")
    print(f"  Optimization: {metrics_dcm.optimization_steps} steps, E={metrics_dcm.optimized_energy:.6f} eV")
    print(f"  Time: {metrics_dcm.harmonic_time:.1f}s")
    
    print("\nNon-Eq:")
    (metrics_noneq.frequencies, metrics_noneq.ir_intensities, 
     metrics_noneq.raman_intensities, atoms_noneq, 
     metrics_noneq.optimization_steps, metrics_noneq.optimized_energy,
     metrics_noneq.harmonic_time) = run_harmonic_analysis(calc_noneq, 'CO2')
    
    print(f"  Frequencies: {', '.join([f'{f:.1f}' for f in metrics_noneq.frequencies])} cm⁻¹")
    print(f"  Optimization: {metrics_noneq.optimization_steps} steps, E={metrics_noneq.optimized_energy:.6f} eV")
    print(f"  Time: {metrics_noneq.harmonic_time:.1f}s")
    
    # ========================================
    # 2. MOLECULAR DYNAMICS (if requested)
    # ========================================
    if args.full or not args.quick:
        print("\n" + "="*70)
        print("2. MOLECULAR DYNAMICS SIMULATIONS")
        print("="*70)
        
        print(f"\nParameters: T={args.temperature}K, steps={args.md_steps}, dt={args.timestep}fs")
        
        print("\nDCMNet:")
        md_dcm = run_md_analysis(calc_dcm, 'CO2', args.temperature, 
                                args.md_steps, args.timestep)
        metrics_dcm.md_energy_std = np.std(md_dcm['energies'])
        metrics_dcm.md_temperature_mean = np.mean(md_dcm['temperatures'])
        metrics_dcm.md_ir_frequencies = md_dcm.get('ir_frequencies')
        metrics_dcm.md_ir_intensities = md_dcm.get('ir_intensities')
        metrics_dcm.md_time = md_dcm['md_time']
        
        print(f"  T_avg={metrics_dcm.md_temperature_mean:.1f}K, σ(E)={metrics_dcm.md_energy_std:.4f} eV")
        print(f"  Time: {metrics_dcm.md_time:.1f}s")
        
        print("\nNon-Eq:")
        md_noneq = run_md_analysis(calc_noneq, 'CO2', args.temperature,
                                  args.md_steps, args.timestep)
        metrics_noneq.md_energy_std = np.std(md_noneq['energies'])
        metrics_noneq.md_temperature_mean = np.mean(md_noneq['temperatures'])
        metrics_noneq.md_ir_frequencies = md_noneq.get('ir_frequencies')
        metrics_noneq.md_ir_intensities = md_noneq.get('ir_intensities')
        metrics_noneq.md_time = md_noneq['md_time']
        
        print(f"  T_avg={metrics_noneq.md_temperature_mean:.1f}K, σ(E)={metrics_noneq.md_energy_std:.4f} eV")
        print(f"  Time: {metrics_noneq.md_time:.1f}s")
    
    # ========================================
    # 3. CHARGE SURFACE ANALYSIS (if requested)
    # ========================================
    if args.analyze_charges:
        print("\n" + "="*70)
        print("3. CHARGE DISTRIBUTION ANALYSIS")
        print("="*70)
        
        print("\nDCMNet:")
        charges_dcm = analyze_charges_vs_geometry(
            calc_dcm,
            tuple(args.theta_range),
            tuple(args.r_range),
            args.output_dir
        )
        metrics_dcm.charges_vs_geometry = charges_dcm
        
        print("\nNon-Eq:")
        charges_noneq = analyze_charges_vs_geometry(
            calc_noneq,
            tuple(args.theta_range),
            tuple(args.r_range),
            args.output_dir
        )
        metrics_noneq.charges_vs_geometry = charges_noneq
        
        # Plot charge surfaces
        print("\n  Creating charge surface plots...")
        plot_charge_surfaces(charges_dcm, charges_noneq, args.output_dir, args.dpi)
    
    # ========================================
    # 4. CREATE COMPARISON PLOTS
    # ========================================
    print("\n" + "="*70)
    print("4. GENERATING COMPARISON PLOTS")
    print("="*70)
    
    plot_spectroscopy_comparison(metrics_dcm, metrics_noneq, args.output_dir, args.dpi)
    
    # ========================================
    # 5. SAVE RESULTS
    # ========================================
    results = {
        'dcmnet': {
            'frequencies': metrics_dcm.frequencies.tolist() if metrics_dcm.frequencies is not None else None,
            'ir_intensities': metrics_dcm.ir_intensities.tolist() if metrics_dcm.ir_intensities is not None else None,
            'raman_intensities': metrics_dcm.raman_intensities.tolist() if metrics_dcm.raman_intensities is not None else None,
            'optimized_energy': float(metrics_dcm.optimized_energy) if metrics_dcm.optimized_energy is not None else None,
            'optimization_steps': metrics_dcm.optimization_steps,
            'md_energy_std': float(metrics_dcm.md_energy_std) if metrics_dcm.md_energy_std is not None else None,
            'md_temperature_mean': float(metrics_dcm.md_temperature_mean) if metrics_dcm.md_temperature_mean is not None else None,
            'harmonic_time': metrics_dcm.harmonic_time,
            'md_time': metrics_dcm.md_time,
        },
        'noneq': {
            'frequencies': metrics_noneq.frequencies.tolist() if metrics_noneq.frequencies is not None else None,
            'ir_intensities': metrics_noneq.ir_intensities.tolist() if metrics_noneq.ir_intensities is not None else None,
            'raman_intensities': metrics_noneq.raman_intensities.tolist() if metrics_noneq.raman_intensities is not None else None,
            'optimized_energy': float(metrics_noneq.optimized_energy) if metrics_noneq.optimized_energy is not None else None,
            'optimization_steps': metrics_noneq.optimization_steps,
            'md_energy_std': float(metrics_noneq.md_energy_std) if metrics_noneq.md_energy_std is not None else None,
            'md_temperature_mean': float(metrics_noneq.md_temperature_mean) if metrics_noneq.md_temperature_mean is not None else None,
            'harmonic_time': metrics_noneq.harmonic_time,
            'md_time': metrics_noneq.md_time,
        },
        'args': vars(args),
    }
    
    output_json = args.output_dir / 'downstream_results.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Saved: {output_json}")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print("\nHarmonic Frequencies:")
    if metrics_dcm.frequencies is not None:
        print(f"  DCMNet:  {', '.join([f'{f:.1f}' for f in metrics_dcm.frequencies])} cm⁻¹")
        print(f"  Non-Eq:  {', '.join([f'{f:.1f}' for f in metrics_noneq.frequencies])} cm⁻¹")
        
        # Compute differences
        freq_diffs = np.abs(metrics_dcm.frequencies - metrics_noneq.frequencies)
        print(f"  Max difference: {np.max(freq_diffs):.1f} cm⁻¹")
        print(f"  Mean difference: {np.mean(freq_diffs):.1f} cm⁻¹")
    
    print(f"\nOptimized Energies:")
    print(f"  DCMNet:  {metrics_dcm.optimized_energy:.6f} eV")
    print(f"  Non-Eq:  {metrics_noneq.optimized_energy:.6f} eV")
    print(f"  Difference: {abs(metrics_dcm.optimized_energy - metrics_noneq.optimized_energy):.6f} eV")
    
    if metrics_dcm.md_time is not None:
        print(f"\nMD Performance:")
        print(f"  DCMNet:  {metrics_dcm.md_time:.1f}s for {args.md_steps} steps")
        print(f"  Non-Eq:  {metrics_noneq.md_time:.1f}s for {args.md_steps} steps")
        print(f"  Speedup: {metrics_dcm.md_time / metrics_noneq.md_time:.2f}x")
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  - downstream_results.json")
    print(f"  - spectroscopy_comparison.png")
    if args.analyze_charges:
        print(f"  - charge_surfaces.png")


if __name__ == '__main__':
    main()

