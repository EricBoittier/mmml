#!/usr/bin/env python3
"""
ASE Calculator for Joint PhysNet-DCMNet Model

This script creates an ASE calculator from a trained joint PhysNet-DCMNet model
and performs:
1. Geometry optimization
2. Normal mode analysis (frequencies)
3. IR spectrum calculation from both PhysNet and DCMNet dipoles

Usage:
    python ase_calculator.py --checkpoint checkpoints/co2_joint/best_params.pkl \
                             --molecule co2 \
                             --optimize \
                             --frequencies \
                             --ir-spectra
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pickle
from typing import Dict, Tuple, Any

# Add mmml to path
repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp

try:
    import ase
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.optimize import BFGS, LBFGS
    from ase.vibrations import Vibrations
    from ase.thermochemistry import IdealGasThermo
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("⚠️  ASE not installed. Install with: pip install ase")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import the joint model
from trainer import JointPhysNetDCMNet


class JointPhysNetDCMNetCalculator(Calculator):
    """
    ASE Calculator for the joint PhysNet-DCMNet model.
    
    Provides:
    - Energy, forces (from PhysNet)
    - Dipole moment (from both PhysNet charges and DCMNet multipoles)
    - Atomic charges (from PhysNet)
    - Distributed charges (from DCMNet)
    
    Parameters
    ----------
    model : JointPhysNetDCMNet
        The joint model
    params : Any
        Trained model parameters
    cutoff : float
        Cutoff distance for edge list construction
    """
    
    implemented_properties = ['energy', 'forces', 'dipole', 'charges']
    
    def __init__(
        self,
        model: JointPhysNetDCMNet,
        params: Any,
        cutoff: float = 10.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.params = params
        self.cutoff = cutoff
        self.natoms = model.physnet_config['natoms']
        self.n_dcm = model.dcmnet_config['n_dcm']
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Calculate properties for the given atoms."""
        super().calculate(atoms, properties, system_changes)
        
        # Get atomic data
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        n_atoms = len(atoms)
        
        # Pad to model's natoms
        positions_padded = np.zeros((self.natoms, 3), dtype=np.float32)
        positions_padded[:n_atoms] = positions
        
        atomic_numbers_padded = np.zeros(self.natoms, dtype=np.int32)
        atomic_numbers_padded[:n_atoms] = atomic_numbers
        
        # Create atom mask
        atom_mask = np.zeros(self.natoms, dtype=np.float32)
        atom_mask[:n_atoms] = 1.0
        
        # Build edge list
        dst_idx_list = []
        src_idx_list = []
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < self.cutoff:
                        dst_idx_list.append(i)
                        src_idx_list.append(j)
        
        dst_idx = np.array(dst_idx_list, dtype=np.int32)
        src_idx = np.array(src_idx_list, dtype=np.int32)
        
        # Batch info (single molecule)
        batch_size = 1
        batch_segments = np.zeros(self.natoms, dtype=np.int32)
        batch_mask = np.ones(len(dst_idx), dtype=np.float32)
        
        # Convert to JAX arrays
        positions_jax = jnp.array(positions_padded)
        atomic_numbers_jax = jnp.array(atomic_numbers_padded)
        atom_mask_jax = jnp.array(atom_mask)
        dst_idx_jax = jnp.array(dst_idx)
        src_idx_jax = jnp.array(src_idx)
        batch_segments_jax = jnp.array(batch_segments)
        batch_mask_jax = jnp.array(batch_mask)
        
        # Model forward pass
        output = self.model.apply(
            self.params,
            atomic_numbers=atomic_numbers_jax,
            positions=positions_jax,
            dst_idx=dst_idx_jax,
            src_idx=src_idx_jax,
            batch_segments=batch_segments_jax,
            batch_size=batch_size,
            batch_mask=batch_mask_jax,
            atom_mask=atom_mask_jax,
        )
        
        # Extract results
        energy = float(output['energy'])
        forces = np.array(output['forces'])[:n_atoms]  # Remove padding
        
        # Compute dipoles from both methods
        dipole_physnet = np.array(output['dipoles'][0])  # (3,)
        
        # DCMNet dipole from distributed multipoles
        mono_dist = output['mono_dist'][:n_atoms]  # (n_atoms, n_dcm)
        dipo_dist = output['dipo_dist'][:n_atoms]  # (n_atoms, n_dcm, 3)
        dipole_dcmnet = np.sum(mono_dist[..., None] * dipo_dist, axis=(0, 1))  # (3,)
        
        # Atomic charges
        charges = np.array(output['charges_as_mono'])[:n_atoms]
        
        # Store results
        self.results = {
            'energy': energy,
            'forces': forces,
            'dipole': dipole_physnet,  # Default to PhysNet dipole
            'charges': charges,
            # Extra info
            'dipole_physnet': dipole_physnet,
            'dipole_dcmnet': dipole_dcmnet,
            'mono_dist': np.array(mono_dist),
            'dipo_dist': np.array(dipo_dist),
        }


def create_test_molecule(name: str = 'co2') -> Atoms:
    """
    Create a test molecule for calculations.
    
    Parameters
    ----------
    name : str
        Molecule name: 'co2', 'h2o', 'ch4', etc.
        
    Returns
    -------
    Atoms
        ASE Atoms object
    """
    if name.lower() == 'co2':
        # Linear CO2 molecule
        atoms = Atoms('CO2', 
                     positions=[[0.0, 0.0, 0.0],
                               [1.16, 0.0, 0.0],
                               [-1.16, 0.0, 0.0]])
    elif name.lower() == 'h2o':
        # Water molecule
        atoms = Atoms('H2O',
                     positions=[[0.0, 0.0, 0.0],
                               [0.96, 0.0, 0.0],
                               [-0.24, 0.93, 0.0]])
    elif name.lower() == 'ch4':
        # Methane (tetrahedral)
        atoms = Atoms('CH4',
                     positions=[[0.0, 0.0, 0.0],
                               [0.63, 0.63, 0.63],
                               [-0.63, -0.63, 0.63],
                               [-0.63, 0.63, -0.63],
                               [0.63, -0.63, -0.63]])
    else:
        raise ValueError(f"Unknown molecule: {name}")
    
    return atoms


def geometry_optimization(
    atoms: Atoms,
    calculator: Calculator,
    fmax: float = 0.05,
    optimizer: str = 'BFGS',
    logfile: str = None,
) -> Atoms:
    """
    Perform geometry optimization.
    
    Parameters
    ----------
    atoms : Atoms
        Initial structure
    calculator : Calculator
        ASE calculator
    fmax : float
        Force convergence criterion (eV/Å)
    optimizer : str
        Optimizer name: 'BFGS' or 'LBFGS'
    logfile : str
        Path to optimization log file
        
    Returns
    -------
    Atoms
        Optimized structure
    """
    print(f"\n{'#'*70}")
    print("# Geometry Optimization")
    print(f"{'#'*70}\n")
    
    atoms.calc = calculator
    
    print(f"Initial geometry:")
    print(f"  Positions:\n{atoms.get_positions()}")
    print(f"  Energy: {atoms.get_potential_energy():.6f} eV")
    print(f"  Max force: {np.abs(atoms.get_forces()).max():.6f} eV/Å")
    
    # Setup optimizer
    if optimizer.upper() == 'BFGS':
        opt = BFGS(atoms, logfile=logfile)
    elif optimizer.upper() == 'LBFGS':
        opt = LBFGS(atoms, logfile=logfile)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    print(f"\nOptimizing with {optimizer} (fmax={fmax} eV/Å)...")
    opt.run(fmax=fmax)
    
    print(f"\n✅ Optimization complete!")
    print(f"  Final positions:\n{atoms.get_positions()}")
    print(f"  Final energy: {atoms.get_potential_energy():.6f} eV")
    print(f"  Final max force: {np.abs(atoms.get_forces()).max():.6f} eV/Å")
    
    return atoms


def compute_frequencies(
    atoms: Atoms,
    calculator: Calculator,
    name: str = 'molecule',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute vibrational frequencies and normal modes.
    
    Parameters
    ----------
    atoms : Atoms
        Optimized structure
    calculator : Calculator
        ASE calculator
    name : str
        Name for vibrations directory
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequencies (cm^-1) and normal modes
    """
    print(f"\n{'#'*70}")
    print("# Vibrational Frequency Analysis")
    print(f"{'#'*70}\n")
    
    atoms.calc = calculator
    
    # Setup vibrations
    vib = Vibrations(atoms, name=name)
    
    print("Computing Hessian (finite differences)...")
    print("  This will take multiple energy+force evaluations...")
    vib.run()
    
    print("\nDiagonalizing Hessian...")
    vib.summary()
    
    # Get frequencies in cm^-1
    energies = vib.get_energies()
    frequencies = energies * 8065.54429  # eV to cm^-1
    
    # Filter out imaginary/translation/rotation modes
    real_freq_mask = frequencies > 10.0  # Only real vibrations > 10 cm^-1
    real_frequencies = frequencies[real_freq_mask]
    
    print(f"\n✅ Vibrational modes computed:")
    print(f"  Total modes: {len(frequencies)}")
    print(f"  Real vibrational modes: {len(real_frequencies)}")
    print(f"\nVibrational frequencies (cm^-1):")
    for i, freq in enumerate(real_frequencies):
        print(f"  Mode {i+1}: {freq:.2f} cm^-1")
    
    return frequencies, vib


def compute_ir_spectrum(
    atoms: Atoms,
    calculator: Calculator,
    vib: Vibrations,
    output_file: Path = None,
    dipole_source: str = 'physnet',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute IR spectrum from dipole derivatives.
    
    Parameters
    ----------
    atoms : Atoms
        Optimized structure
    calculator : Calculator
        Joint calculator
    vib : Vibrations
        Vibrations object with normal modes
    output_file : Path
        Path to save spectrum plot
    dipole_source : str
        'physnet' or 'dcmnet' for dipole source
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequencies (cm^-1) and intensities (km/mol)
    """
    print(f"\n{'#'*70}")
    print(f"# IR Spectrum Calculation ({dipole_source.upper()} dipole)")
    print(f"{'#'*70}\n")
    
    atoms.calc = calculator
    
    # Get equilibrium dipole
    atoms.get_potential_energy()  # Ensure calculation is done
    if dipole_source == 'physnet':
        dipole_eq = calculator.results['dipole_physnet']
    else:
        dipole_eq = calculator.results['dipole_dcmnet']
    
    print(f"Equilibrium dipole ({dipole_source}): {dipole_eq}")
    
    # Get frequencies
    energies = vib.get_energies()
    frequencies = energies * 8065.54429  # eV to cm^-1
    
    # Compute dipole derivatives for each mode
    print("\nComputing dipole derivatives...")
    
    n_atoms = len(atoms)
    n_modes = 3 * n_atoms
    dipole_derivs = np.zeros((n_modes, 3))
    
    # Small displacement for numerical derivative
    delta = 0.01  # Angstrom
    
    # Get normal modes
    modes = []
    for i in range(n_modes):
        mode = vib.get_mode(i)
        modes.append(mode)
    
    for mode_idx in range(n_modes):
        if mode_idx % 3 == 0:
            print(f"  Processing modes {mode_idx}-{min(mode_idx+2, n_modes-1)}...")
        
        mode = modes[mode_idx]
        
        # Displace along mode
        atoms_plus = atoms.copy()
        atoms_plus.positions += delta * mode
        atoms_plus.calc = calculator
        atoms_plus.get_potential_energy()
        
        if dipole_source == 'physnet':
            dipole_plus = calculator.results['dipole_physnet']
        else:
            dipole_plus = calculator.results['dipole_dcmnet']
        
        # Compute derivative
        dipole_derivs[mode_idx] = (dipole_plus - dipole_eq) / delta
    
    # Compute IR intensities
    # I ∝ |dμ/dQ|^2 where Q is normal mode coordinate
    intensities_au = np.sum(dipole_derivs**2, axis=1)  # Squared norm
    
    # Convert to km/mol (approximate conversion)
    # 1 (eÅ/amu^(1/2))^2 ≈ 42.255 km/mol
    conversion = 42.255
    intensities = intensities_au * conversion
    
    # Filter real vibrational modes
    real_mask = frequencies > 10.0
    real_frequencies = frequencies[real_mask]
    real_intensities = intensities[real_mask]
    
    print(f"\n✅ IR intensities computed")
    print(f"\nIR-active modes:")
    for i, (freq, intensity) in enumerate(zip(real_frequencies, real_intensities)):
        print(f"  Mode {i+1}: {freq:.2f} cm^-1, Intensity: {intensity:.2f} km/mol")
    
    # Create spectrum plot if matplotlib available
    if HAS_MATPLOTLIB and output_file is not None:
        plot_ir_spectrum(real_frequencies, real_intensities, output_file, dipole_source)
    
    return real_frequencies, real_intensities


def plot_ir_spectrum(
    frequencies: np.ndarray,
    intensities: np.ndarray,
    output_file: Path,
    dipole_source: str,
    broadening: float = 10.0,
) -> None:
    """
    Plot IR spectrum with Lorentzian broadening.
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequencies in cm^-1
    intensities : np.ndarray
        Intensities in km/mol
    output_file : Path
        Output file path
    dipole_source : str
        Dipole source name
    broadening : float
        Lorentzian HWHM in cm^-1
    """
    # Create smooth spectrum with Lorentzian broadening
    freq_range = np.linspace(0, max(frequencies) + 500, 2000)
    spectrum = np.zeros_like(freq_range)
    
    for freq, intensity in zip(frequencies, intensities):
        # Lorentzian: I(ν) = I₀ * (γ² / ((ν - ν₀)² + γ²))
        spectrum += intensity * (broadening**2 / ((freq_range - freq)**2 + broadening**2))
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Stick spectrum
    ax1.stem(frequencies, intensities, basefmt=' ', linefmt='b-', markerfmt='bo')
    ax1.set_xlabel('Frequency (cm⁻¹)')
    ax1.set_ylabel('Intensity (km/mol)')
    ax1.set_title(f'IR Spectrum - Stick ({dipole_source.upper()} dipole)')
    ax1.grid(True, alpha=0.3)
    
    # Broadened spectrum
    ax2.plot(freq_range, spectrum, 'b-', linewidth=2)
    ax2.fill_between(freq_range, spectrum, alpha=0.3)
    ax2.set_xlabel('Frequency (cm⁻¹)')
    ax2.set_ylabel('Intensity (a.u.)')
    ax2.set_title(f'IR Spectrum - Broadened (HWHM={broadening} cm⁻¹)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ IR spectrum plot saved: {output_file}")


def compare_ir_spectra(
    atoms: Atoms,
    calculator: Calculator,
    vib: Vibrations,
    output_dir: Path,
) -> None:
    """
    Compare IR spectra from PhysNet and DCMNet dipoles.
    
    Parameters
    ----------
    atoms : Atoms
        Optimized structure
    calculator : Calculator
        Joint calculator
    vib : Vibrations
        Vibrations object
    output_dir : Path
        Directory to save plots
    """
    print(f"\n{'='*70}")
    print("Comparing IR Spectra: PhysNet vs DCMNet Dipoles")
    print(f"{'='*70}\n")
    
    # Compute IR from PhysNet dipole
    freq_physnet, int_physnet = compute_ir_spectrum(
        atoms, calculator, vib,
        output_file=output_dir / 'ir_spectrum_physnet.png',
        dipole_source='physnet'
    )
    
    # Compute IR from DCMNet dipole
    freq_dcmnet, int_dcmnet = compute_ir_spectrum(
        atoms, calculator, vib,
        output_file=output_dir / 'ir_spectrum_dcmnet.png',
        dipole_source='dcmnet'
    )
    
    # Create comparison plot
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # PhysNet stick
        ax = axes[0, 0]
        ax.stem(freq_physnet, int_physnet, basefmt=' ', linefmt='b-', markerfmt='bo')
        ax.set_xlabel('Frequency (cm⁻¹)')
        ax.set_ylabel('Intensity (km/mol)')
        ax.set_title('PhysNet Dipole - Stick Spectrum')
        ax.grid(True, alpha=0.3)
        
        # DCMNet stick
        ax = axes[0, 1]
        ax.stem(freq_dcmnet, int_dcmnet, basefmt=' ', linefmt='r-', markerfmt='ro')
        ax.set_xlabel('Frequency (cm⁻¹)')
        ax.set_ylabel('Intensity (km/mol)')
        ax.set_title('DCMNet Dipole - Stick Spectrum')
        ax.grid(True, alpha=0.3)
        
        # PhysNet broadened
        ax = axes[1, 0]
        freq_range = np.linspace(0, max(freq_physnet.max(), freq_dcmnet.max()) + 500, 2000)
        spectrum_physnet = np.zeros_like(freq_range)
        for freq, intensity in zip(freq_physnet, int_physnet):
            spectrum_physnet += intensity * (100 / ((freq_range - freq)**2 + 100))
        ax.plot(freq_range, spectrum_physnet, 'b-', linewidth=2)
        ax.fill_between(freq_range, spectrum_physnet, alpha=0.3)
        ax.set_xlabel('Frequency (cm⁻¹)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title('PhysNet - Broadened')
        ax.grid(True, alpha=0.3)
        
        # DCMNet broadened
        ax = axes[1, 1]
        spectrum_dcmnet = np.zeros_like(freq_range)
        for freq, intensity in zip(freq_dcmnet, int_dcmnet):
            spectrum_dcmnet += intensity * (100 / ((freq_range - freq)**2 + 100))
        ax.plot(freq_range, spectrum_dcmnet, 'r-', linewidth=2)
        ax.fill_between(freq_range, spectrum_dcmnet, alpha=0.3, color='red')
        ax.set_xlabel('Frequency (cm⁻¹)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title('DCMNet - Broadened')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_path = output_dir / 'ir_spectrum_comparison.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Comparison plot saved: {comparison_path}")
    
    # Print comparison
    print(f"\nIntensity comparison:")
    print(f"  {'Mode':<6} {'Freq (cm⁻¹)':<15} {'PhysNet (km/mol)':<20} {'DCMNet (km/mol)':<20}")
    print(f"  {'-'*65}")
    for i, (f, ip, id) in enumerate(zip(freq_physnet, int_physnet, int_dcmnet)):
        print(f"  {i+1:<6} {f:<15.2f} {ip:<20.2f} {id:<20.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="ASE calculator for joint PhysNet-DCMNet model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to saved model parameters (.pkl)')
    parser.add_argument('--molecule', type=str, default='co2',
                       choices=['co2', 'h2o', 'ch4'],
                       help='Test molecule')
    parser.add_argument('--optimize', action='store_true',
                       help='Perform geometry optimization')
    parser.add_argument('--frequencies', action='store_true',
                       help='Compute vibrational frequencies')
    parser.add_argument('--ir-spectra', action='store_true',
                       help='Compute IR spectra from both dipole sources')
    parser.add_argument('--fmax', type=float, default=0.05,
                       help='Force convergence for optimization (eV/Å)')
    parser.add_argument('--output-dir', type=Path, default=Path('./ase_results'),
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    # Model configuration (only used if model_config.pkl not found)
    # These defaults match the typical CO2 training setup
    parser.add_argument('--natoms', type=int, default=60,
                       help='Model natoms parameter (ignored if model_config.pkl exists)')
    parser.add_argument('--max-atomic-number', type=int, default=28,
                       help='Max atomic number (ignored if model_config.pkl exists)')
    parser.add_argument('--physnet-features', type=int, default=64,
                       help='PhysNet features (ignored if model_config.pkl exists)')
    parser.add_argument('--physnet-iterations', type=int, default=5,
                       help='PhysNet iterations (ignored if model_config.pkl exists)')
    parser.add_argument('--physnet-basis', type=int, default=64,
                       help='PhysNet basis functions (ignored if model_config.pkl exists)')
    parser.add_argument('--physnet-cutoff', type=float, default=6.0,
                       help='PhysNet cutoff (ignored if model_config.pkl exists)')
    parser.add_argument('--dcmnet-features', type=int, default=32,
                       help='DCMNet features (ignored if model_config.pkl exists)')
    parser.add_argument('--dcmnet-iterations', type=int, default=2,
                       help='DCMNet iterations (ignored if model_config.pkl exists)')
    parser.add_argument('--dcmnet-basis', type=int, default=32,
                       help='DCMNet basis functions (ignored if model_config.pkl exists)')
    parser.add_argument('--dcmnet-cutoff', type=float, default=10.0,
                       help='DCMNet cutoff (ignored if model_config.pkl exists)')
    parser.add_argument('--n-dcm', type=int, default=3,
                       help='Distributed multipoles per atom (ignored if model_config.pkl exists)')
    parser.add_argument('--max-degree', type=int, default=2,
                       help='Maximum spherical harmonic degree (ignored if model_config.pkl exists)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ASE Calculator for Joint PhysNet-DCMNet Model")
    print("="*70)
    
    # Load model parameters
    if not args.checkpoint.exists():
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    print(f"\n📂 Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        params = pickle.load(f)
    print("✅ Parameters loaded")
    
    # Try to load model config (saved with newer checkpoints)
    config_file = args.checkpoint.parent / 'model_config.pkl'
    if config_file.exists():
        print(f"📂 Loading model config: {config_file}")
        with open(config_file, 'rb') as f:
            saved_config = pickle.load(f)
        physnet_config = saved_config['physnet_config']
        dcmnet_config = saved_config['dcmnet_config']
        mix_coulomb_energy = saved_config.get('mix_coulomb_energy', False)
        print("✅ Model config loaded from checkpoint")
    else:
        print("⚠️  Model config not found, using command-line arguments")
        print("   (This may cause errors if args don't match training)")
        mix_coulomb_energy = False
        
        physnet_config = {
            'features': args.physnet_features,
            'max_degree': 0,
            'num_iterations': args.physnet_iterations,
            'num_basis_functions': args.physnet_basis,
            'cutoff': args.physnet_cutoff,
            'max_atomic_number': args.max_atomic_number,
            'charges': True,
            'natoms': args.natoms,
            'total_charge': 0.0,
            'n_res': 3,
            'zbl': False,
            'use_energy_bias': True,
            'debug': False,
            'efa': False,
        }
        
        dcmnet_config = {
            'features': args.dcmnet_features,
            'max_degree': args.max_degree,
            'num_iterations': args.dcmnet_iterations,
            'num_basis_functions': args.dcmnet_basis,
            'cutoff': args.dcmnet_cutoff,
            'max_atomic_number': args.max_atomic_number,
            'n_dcm': args.n_dcm,
            'include_pseudotensors': False,
        }
    
    # Create model
    print("\n🔧 Building joint model...")
    print(f"  PhysNet: {physnet_config['features']} features, {physnet_config['num_iterations']} iterations")
    print(f"  DCMNet: {dcmnet_config['features']} features, {dcmnet_config['num_iterations']} iterations")
    print(f"  Max atomic number: {physnet_config['max_atomic_number']}")
    print(f"  Natoms: {physnet_config['natoms']}")
    
    model = JointPhysNetDCMNet(
        physnet_config=physnet_config,
        dcmnet_config=dcmnet_config,
        mix_coulomb_energy=mix_coulomb_energy,
    )
    
    print("✅ Model created")
    
    # Create calculator
    cutoff = max(args.physnet_cutoff, args.dcmnet_cutoff)
    calculator = JointPhysNetDCMNetCalculator(model, params, cutoff=cutoff)
    
    print(f"✅ ASE calculator created (cutoff={cutoff:.1f} Å)")
    
    # Create test molecule
    print(f"\n🧪 Creating test molecule: {args.molecule.upper()}")
    atoms = create_test_molecule(args.molecule)
    
    print(f"  Formula: {atoms.get_chemical_formula()}")
    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Initial positions:\n{atoms.get_positions()}")
    
    # Check if molecule fits in model
    model_natoms = physnet_config['natoms']
    if len(atoms) > model_natoms:
        print(f"\n❌ Error: Molecule has {len(atoms)} atoms but model supports max {model_natoms}")
        sys.exit(1)
    
    # Setup output directory
    args.output_dir.mkdir(exist_ok=True, parents=True)
    print(f"  Output directory: {args.output_dir}")
    
    # Geometry optimization
    if args.optimize:
        atoms = geometry_optimization(
            atoms, calculator, 
            fmax=args.fmax,
            logfile=str(args.output_dir / 'optimization.log')
        )
        
        # Save optimized geometry
        from ase.io import write
        opt_file = args.output_dir / f'{args.molecule}_optimized.xyz'
        write(opt_file, atoms)
        print(f"  💾 Saved optimized geometry: {opt_file}")
    
    # Frequency calculation
    vib = None
    if args.frequencies or args.ir_spectra:
        vib = compute_frequencies(
            atoms, calculator,
            name=str(args.output_dir / f'{args.molecule}_vib')
        )
    
    # IR spectra
    if args.ir_spectra:
        if vib is None:
            print("\n⚠️  Need to compute frequencies first for IR spectra")
            vib = compute_frequencies(
                atoms, calculator,
                name=str(args.output_dir / f'{args.molecule}_vib')
            )
        
        # Compare both methods
        compare_ir_spectra(atoms, calculator, vib, args.output_dir)
    
    print(f"\n{'='*70}")
    print("✅ ALL CALCULATIONS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {args.output_dir}")
    
    # Print final dipole comparison
    print(f"\nFinal dipole moments:")
    atoms.calc = calculator
    atoms.get_potential_energy()
    print(f"  PhysNet: {calculator.results['dipole_physnet']} D")
    print(f"  DCMNet:  {calculator.results['dipole_dcmnet']} D")
    print(f"\nAtomic charges (PhysNet):")
    charges = calculator.results['charges']
    for i, (symbol, charge) in enumerate(zip(atoms.get_chemical_symbols(), charges)):
        print(f"  {symbol}{i+1}: {charge:.4f} e")
    print(f"  Total charge: {charges.sum():.4f} e")


if __name__ == "__main__":
    main()

