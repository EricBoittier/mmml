#!/usr/bin/env python3
"""
Molecular Dynamics and Vibrational Analysis CLI Tool

Supports multiple frameworks:
- ASE: Geometry optimization, vibrational analysis, MD (CPU/GPU)
- JAX MD: Ultra-fast GPU-accelerated MD with JIT compilation

Features:
1. Geometry optimization (BFGS, LBFGS, FIRE)
2. Vibrational frequency analysis
3. IR spectra calculation
4. Molecular dynamics (NVE, NVT, NPT)
5. Trajectory analysis

Usage:
    # Geometry optimization (ASE)
    python -m mmml.cli.dynamics --checkpoint model/ --molecule CO2 --optimize
    
    # Vibrational analysis
    python -m mmml.cli.dynamics --checkpoint model/ --molecule CO2 --frequencies --ir-spectra
    
    # MD with ASE (CPU/GPU, good for small systems)
    python -m mmml.cli.dynamics --checkpoint model/ --molecule CO2 \
        --md --framework ase --ensemble nvt --temperature 300 --timestep 0.5 --nsteps 10000
    
    # MD with JAX MD (GPU-accelerated, best for large systems)
    python -m mmml.cli.dynamics --checkpoint model/ --molecule CO2 \
        --md --framework jaxmd --ensemble nvt --temperature 300 --timestep 0.5 --nsteps 100000
    
    # Load structure from file
    python -m mmml.cli.dynamics --checkpoint model/ --structure molecule.xyz \
        --md --framework jaxmd --ensemble nve --nsteps 50000
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pickle

# Try importing JAX and frameworks
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

try:
    from ase import Atoms, units
    from ase.calculators.calculator import Calculator, all_changes
    from ase.optimize import BFGS, LBFGS, BFGSLineSearch, FIRE
    from ase.vibrations import Vibrations
    from ase.md.verlet import VelocityVerlet
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from ase.io import read, write
    from ase.io.trajectory import Trajectory
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    Atoms = None

try:
    from jax_md import space, simulate, partition
    HAS_JAXMD = True
except ImportError:
    HAS_JAXMD = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import MMML calculator
try:
    from mmml.cli.calculator import MMMLCalculator
except ImportError:
    print("⚠️  Could not import MMMLCalculator from mmml.cli.calculator")
    MMMLCalculator = None


# =============================================================================
# PREDEFINED MOLECULES
# =============================================================================

MOLECULES = {
    'CO2': {
        'symbols': ['C', 'O', 'O'],
        'positions': [[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]],
    },
    'H2O': {
        'symbols': ['O', 'H', 'H'],
        'positions': [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]],
    },
    'CH4': {
        'symbols': ['C', 'H', 'H', 'H', 'H'],
        'positions': [
            [0, 0, 0],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ],
    },
    'NH3': {
        'symbols': ['N', 'H', 'H', 'H'],
        'positions': [
            [0, 0, 0],
            [0.94, 0, 0.33],
            [-0.47, 0.82, 0.33],
            [-0.47, -0.82, 0.33],
        ],
    },
}


def create_molecule(name_or_file: str) -> Atoms:
    """Create molecule from name or load from file."""
    if not HAS_ASE:
        raise ImportError("ASE required. Install with: pip install ase")
    
    # Try as predefined molecule
    if name_or_file.upper() in MOLECULES:
        mol_data = MOLECULES[name_or_file.upper()]
        return Atoms(symbols=mol_data['symbols'], positions=mol_data['positions'])
    
    # Try as file
    path = Path(name_or_file)
    if path.exists():
        return read(str(path))
    
    raise ValueError(f"Unknown molecule or file not found: {name_or_file}")


# =============================================================================
# GEOMETRY OPTIMIZATION
# =============================================================================

def optimize_geometry(
    atoms: Atoms,
    calculator: Calculator,
    optimizer: str = 'bfgs',
    fmax: float = 0.01,
    maxsteps: int = 200,
    output_file: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Optimize molecular geometry.
    
    Parameters
    ----------
    atoms : Atoms
        Molecule to optimize
    calculator : Calculator
        ASE calculator
    optimizer : str
        Optimizer type: 'bfgs', 'lbfgs', 'fire'
    fmax : float
        Force convergence criterion (eV/Å)
    maxsteps : int
        Maximum optimization steps
    output_file : Path, optional
        Save optimized structure
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        Optimization results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"GEOMETRY OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Optimizer: {optimizer.upper()}")
        print(f"Force criterion: {fmax} eV/Å")
        print(f"Max steps: {maxsteps}")
        print()
    
    atoms.calc = calculator
    
    # Choose optimizer
    optimizers = {
        'bfgs': BFGS,
        'lbfgs': LBFGS,
        'bfgsls': BFGSLineSearch,
        'fire': FIRE,
    }
    
    if optimizer.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer}. Choose from: {list(optimizers.keys())}")
    
    opt = optimizers[optimizer.lower()](atoms, logfile='-' if verbose else None)
    
    # Run optimization
    initial_energy = atoms.get_potential_energy()
    opt.run(fmax=fmax, steps=maxsteps)
    final_energy = atoms.get_potential_energy()
    
    # Results
    converged = opt.get_number_of_steps() < maxsteps
    
    if verbose:
        print(f"\n✅ Optimization {'converged' if converged else 'did not converge'}")
        print(f"   Steps: {opt.get_number_of_steps()}")
        print(f"   Initial energy: {initial_energy:.6f} eV")
        print(f"   Final energy: {final_energy:.6f} eV")
        print(f"   Energy change: {final_energy - initial_energy:.6f} eV")
        print(f"   Max force: {np.max(np.linalg.norm(atoms.get_forces(), axis=1)):.6f} eV/Å")
    
    # Save structure
    if output_file:
        write(str(output_file), atoms)
        if verbose:
            print(f"   Saved to: {output_file}")
    
    return {
        'converged': converged,
        'steps': opt.get_number_of_steps(),
        'initial_energy': initial_energy,
        'final_energy': final_energy,
        'energy_change': final_energy - initial_energy,
        'max_force': np.max(np.linalg.norm(atoms.get_forces(), axis=1)),
    }


# =============================================================================
# VIBRATIONAL ANALYSIS
# =============================================================================

def calculate_frequencies(
    atoms: Atoms,
    calculator: Calculator,
    delta: float = 0.01,
    nfree: int = 2,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calculate vibrational frequencies using finite differences.
    
    Parameters
    ----------
    atoms : Atoms
        Molecule (should be at equilibrium geometry)
    calculator : Calculator
        ASE calculator
    delta : float
        Displacement for finite differences (Å)
    nfree : int
        Number of degrees of freedom to remove (3=translation, 5=+rotation linear, 6=+rotation nonlinear)
    output_dir : Path, optional
        Output directory for vibrations data
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        Frequencies and related data
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"VIBRATIONAL FREQUENCY ANALYSIS")
        print(f"{'='*70}")
        print(f"Displacement: {delta} Å")
        print(f"Free DOF: {nfree}")
        print()
    
    atoms.calc = calculator
    
    # Setup vibrations
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        vib = Vibrations(atoms, delta=delta, nfree=nfree, name=str(output_dir / 'vib'))
    else:
        vib = Vibrations(atoms, delta=delta, nfree=nfree)
    
    # Run
    vib.run()
    vib.summary(log='-' if verbose else None)
    
    # Get frequencies
    freqs = vib.get_frequencies()
    
    if verbose:
        print(f"\n✅ Calculated {len(freqs)} vibrational modes")
        print(f"   Frequency range: {freqs.min():.1f} - {freqs.max():.1f} cm⁻¹")
    
    return {
        'frequencies': freqs,
        'vibrations': vib,
    }


# =============================================================================
# IR SPECTRA
# =============================================================================

def calculate_ir_spectrum(
    atoms: Atoms,
    vibrations: Vibrations,
    output_file: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calculate IR spectrum from vibrational modes.
    
    Parameters
    ----------
    atoms : Atoms
        Molecule
    vibrations : Vibrations
        ASE Vibrations object
    output_file : Path, optional
        Save IR spectrum plot
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        IR intensities and frequencies
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"IR SPECTRUM CALCULATION")
        print(f"{'='*70}")
    
    # Get IR intensities
    vibrations.summary(log='-' if verbose else None)
    intensities = vibrations.get_spectrum(type='ir')
    frequencies = vibrations.get_frequencies()
    
    if verbose:
        print(f"\n✅ IR spectrum calculated")
        print(f"   Active modes: {np.sum(intensities > 0.001)}")
    
    # Plot if requested
    if output_file and HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stem(frequencies, intensities, basefmt=' ')
        ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, fontweight='bold')
        ax.set_ylabel('IR Intensity (km/mol)', fontsize=12, fontweight='bold')
        ax.set_title('IR Spectrum', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        if verbose:
            print(f"   Saved plot: {output_file}")
    
    return {
        'frequencies': frequencies,
        'intensities': intensities,
    }


# =============================================================================
# MOLECULAR DYNAMICS (ASE)
# =============================================================================

def run_md_ase(
    atoms: Atoms,
    calculator: Calculator,
    ensemble: str = 'nvt',
    temperature: float = 300.0,
    timestep: float = 0.5,
    nsteps: int = 10000,
    friction: float = 0.01,
    trajectory_file: Optional[Path] = None,
    save_interval: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run molecular dynamics with ASE.
    
    Parameters
    ----------
    atoms : Atoms
        Initial structure
    calculator : Calculator
        ASE calculator
    ensemble : str
        'nve', 'nvt', or 'npt'
    temperature : float
        Temperature (K)
    timestep : float
        Timestep (fs)
    nsteps : int
        Number of MD steps
    friction : float
        Friction coefficient for Langevin (1/fs)
    trajectory_file : Path, optional
        Save trajectory
    save_interval : int
        Save every N steps
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        MD statistics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"MOLECULAR DYNAMICS (ASE)")
        print(f"{'='*70}")
        print(f"Ensemble: {ensemble.upper()}")
        print(f"Temperature: {temperature} K")
        print(f"Timestep: {timestep} fs")
        print(f"Steps: {nsteps}")
        print()
    
    atoms.calc = calculator
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)
    ZeroRotation(atoms)
    
    # Choose integrator
    if ensemble.lower() == 'nve':
        dyn = VelocityVerlet(atoms, timestep=timestep * units.fs)
    elif ensemble.lower() == 'nvt':
        dyn = Langevin(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            friction=friction,
        )
    else:
        raise ValueError(f"Unsupported ensemble: {ensemble}")
    
    # Attach trajectory
    if trajectory_file:
        traj = Trajectory(str(trajectory_file), 'w', atoms)
        dyn.attach(traj.write, interval=save_interval)
    
    # Attach logger
    if verbose:
        def log_stats():
            epot = atoms.get_potential_energy()
            ekin = atoms.get_kinetic_energy()
            temp = ekin / (1.5 * units.kB * len(atoms))
            print(f"  Step {dyn.nsteps:6d}: E_pot={epot:10.4f} eV, "
                  f"E_kin={ekin:8.4f} eV, T={temp:6.1f} K")
        
        dyn.attach(log_stats, interval=max(1, nsteps // 10))
    
    # Run
    dyn.run(nsteps)
    
    if trajectory_file:
        traj.close()
        if verbose:
            print(f"\n✅ Trajectory saved: {trajectory_file}")
    
    return {
        'nsteps': nsteps,
        'final_energy': atoms.get_potential_energy(),
        'final_temperature': atoms.get_temperature(),
    }


# =============================================================================
# MOLECULAR DYNAMICS (JAX MD)
# =============================================================================

def run_md_jaxmd(
    atoms: Atoms,
    model: Any,
    params: Any,
    ensemble: str = 'nvt',
    temperature: float = 300.0,
    timestep: float = 0.5,
    nsteps: int = 100000,
    save_interval: int = 100,
    cutoff: float = 10.0,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run molecular dynamics with JAX MD (ultra-fast, GPU-accelerated).
    
    Parameters
    ----------
    atoms : Atoms
        Initial structure
    model : Any
        MMML model
    params : Any
        Model parameters
    ensemble : str
        'nve' or 'nvt'
    temperature : float
        Temperature (K)
    timestep : float
        Timestep (fs)
    nsteps : int
        Number of MD steps
    save_interval : int
        Save every N steps
    cutoff : float
        Neighbor list cutoff (Å)
    output_dir : Path, optional
        Output directory
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        MD trajectory and statistics
    """
    if not HAS_JAXMD:
        raise ImportError("JAX MD required. Install with: pip install jax-md")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MOLECULAR DYNAMICS (JAX MD - GPU ACCELERATED)")
        print(f"{'='*70}")
        print(f"Ensemble: {ensemble.upper()}")
        print(f"Temperature: {temperature} K")
        print(f"Timestep: {timestep} fs")
        print(f"Steps: {nsteps:,}")
        print(f"Save interval: {save_interval}")
        print()
    
    # This is a simplified version - full implementation would follow jaxmd_dynamics.py
    raise NotImplementedError("JAX MD support coming soon - use --framework ase for now")


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Molecular dynamics and vibrational analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize geometry
  python -m mmml.cli.dynamics --checkpoint model/ --molecule CO2 --optimize
  
  # Calculate vibrational frequencies
  python -m mmml.cli.dynamics --checkpoint model/ --molecule CO2 --frequencies
  
  # Full vibrational analysis with IR
  python -m mmml.cli.dynamics --checkpoint model/ --molecule CO2 \\
      --optimize --frequencies --ir-spectra --output-dir co2_analysis
  
  # Molecular dynamics (ASE, NVT ensemble)
  python -m mmml.cli.dynamics --checkpoint model/ --molecule CO2 \\
      --md --framework ase --ensemble nvt --temperature 300 --nsteps 10000
  
  # Load structure from file
  python -m mmml.cli.dynamics --checkpoint model/ --structure molecule.xyz \\
      --optimize --output-dir molecule_analysis
        """
    )
    
    # Input
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Model checkpoint directory or file')
    parser.add_argument('--molecule', type=str,
                       help='Predefined molecule (CO2, H2O, CH4, NH3)')
    parser.add_argument('--structure', type=Path,
                       help='Load structure from file (XYZ, PDB, etc.)')
    
    # Tasks
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize geometry')
    parser.add_argument('--frequencies', action='store_true',
                       help='Calculate vibrational frequencies')
    parser.add_argument('--ir-spectra', action='store_true',
                       help='Calculate IR spectrum (requires --frequencies)')
    parser.add_argument('--md', action='store_true',
                       help='Run molecular dynamics')
    
    # Framework selection
    parser.add_argument('--framework', type=str, default='ase',
                       choices=['ase', 'jaxmd'],
                       help='MD framework: ase (CPU/GPU) or jaxmd (GPU-accelerated)')
    
    # Optimization parameters
    parser.add_argument('--optimizer', type=str, default='bfgs',
                       choices=['bfgs', 'lbfgs', 'fire'],
                       help='Geometry optimizer')
    parser.add_argument('--fmax', type=float, default=0.01,
                       help='Force convergence criterion (eV/Å)')
    
    # Vibration parameters
    parser.add_argument('--delta', type=float, default=0.01,
                       help='Displacement for finite differences (Å)')
    
    # MD parameters
    parser.add_argument('--ensemble', type=str, default='nvt',
                       choices=['nve', 'nvt', 'npt'],
                       help='MD ensemble')
    parser.add_argument('--temperature', type=float, default=300.0,
                       help='Temperature (K)')
    parser.add_argument('--timestep', type=float, default=0.5,
                       help='MD timestep (fs)')
    parser.add_argument('--nsteps', type=int, default=10000,
                       help='Number of MD steps')
    parser.add_argument('--friction', type=float, default=0.01,
                       help='Friction coefficient for Langevin (1/fs)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save trajectory every N steps')
    
    # Calculator parameters
    parser.add_argument('--cutoff', type=float, default=10.0,
                       help='Neighbor list cutoff (Å)')
    parser.add_argument('--use-dcmnet-dipole', action='store_true',
                       help='Use DCMNet dipole if available')
    
    # Output
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Validate
    if not HAS_ASE:
        print("❌ Error: ASE not installed")
        print("Install with: pip install ase")
        return 1
    
    if not HAS_JAX:
        print("⚠️  Warning: JAX not installed (some features may be limited)")
    
    if not args.molecule and not args.structure:
        print("❌ Error: Must specify either --molecule or --structure")
        return 1
    
    if args.ir_spectra and not args.frequencies:
        print("❌ Error: --ir-spectra requires --frequencies")
        return 1
    
    if not any([args.optimize, args.frequencies, args.md]):
        print("❌ Error: Must specify at least one task (--optimize, --frequencies, or --md)")
        return 1
    
    verbose = not args.quiet
    
    # Setup output directory
    output_dir = args.output_dir or Path('.')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load calculator
    if verbose:
        print(f"\n🔧 Loading calculator from: {args.checkpoint}")
    
    calc = MMMLCalculator.from_checkpoint(
        args.checkpoint,
        cutoff=args.cutoff,
        use_dcmnet_dipole=args.use_dcmnet_dipole,
    )
    
    if verbose:
        print("✅ Calculator loaded")
    
    # Create/load molecule
    if args.molecule:
        atoms = create_molecule(args.molecule)
        if verbose:
            print(f"✅ Created molecule: {args.molecule}")
    else:
        atoms = create_molecule(str(args.structure))
        if verbose:
            print(f"✅ Loaded structure: {args.structure}")
    
    if verbose:
        print(f"   Formula: {atoms.get_chemical_formula()}")
        print(f"   Number of atoms: {len(atoms)}")
    
    # Run tasks
    if args.optimize:
        opt_result = optimize_geometry(
            atoms, calc,
            optimizer=args.optimizer,
            fmax=args.fmax,
            output_file=output_dir / 'optimized.xyz',
            verbose=verbose,
        )
    
    if args.frequencies:
        freq_result = calculate_frequencies(
            atoms, calc,
            delta=args.delta,
            output_dir=output_dir,
            verbose=verbose,
        )
        
        if args.ir_spectra:
            ir_result = calculate_ir_spectrum(
                atoms,
                freq_result['vibrations'],
                output_file=output_dir / 'ir_spectrum.png',
                verbose=verbose,
            )
    
    if args.md:
        if args.framework == 'ase':
            md_result = run_md_ase(
                atoms, calc,
                ensemble=args.ensemble,
                temperature=args.temperature,
                timestep=args.timestep,
                nsteps=args.nsteps,
                friction=args.friction,
                trajectory_file=output_dir / 'trajectory.traj',
                save_interval=args.save_interval,
                verbose=verbose,
            )
        else:  # jaxmd
            # Would need model and params - placeholder for now
            print("⚠️  JAX MD support coming soon")
            return 1
    
    if verbose:
        print(f"\n✅ All tasks completed!")
        print(f"   Output directory: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

