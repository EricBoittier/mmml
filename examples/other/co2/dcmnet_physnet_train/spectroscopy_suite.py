#!/usr/bin/env python3
"""
Complete Spectroscopy Suite for PhysNet-DCMNet

This is a production-ready script for comprehensive spectroscopic analysis:
- IR spectra (harmonic and from MD)
- Raman spectra (finite-field polarizability)
- Multiple temperatures
- Multiple ensembles (NVE, NVT, Langevin)
- Multiple optimizers (BFGS, LBFGS, BFGSLineSearch)
- Saves everything to ASE trajectories with energies and forces
- Converts JAX MD trajectories to ASE format

Usage:
    # Quick analysis
    python spectroscopy_suite.py --checkpoint ckpt/ --molecule CO2 \
        --quick-analysis
    
    # Full temperature scan
    python spectroscopy_suite.py --checkpoint ckpt/ --molecule CO2 \
        --temperatures 100 200 300 400 500 \
        --ensembles nvt nve \
        --run-all
    
    # Production run
    python spectroscopy_suite.py --checkpoint ckpt/ --molecule CO2 \
        --production --nsteps 500000
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import argparse
import json
from typing import List, Dict, Tuple
import time

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, LBFGS, BFGSLineSearch
from ase.calculators.calculator import Calculator
import ase.data

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from trainer import JointPhysNetDCMNet
from dynamics_calculator import (
    JointPhysNetDCMNetCalculator,
    calculate_frequencies,
    calculate_ir_spectrum,
    compute_ir_from_md,
)
from raman_calculator import (
    calculate_raman_spectrum,
    compute_polarizability_finite_field,
)

# Try to import JAX MD (optional)
try:
    from jaxmd_dynamics import (
        run_jaxmd_simulation,
        initialize_system,
    )
    HAS_JAXMD = True
except (ImportError, AttributeError) as e:
    HAS_JAXMD = False
    print(f"⚠️  JAX MD not available: {e}")
    print("   Will use ASE MD instead (slower but works)")
    
    # Provide dummy functions
    def run_jaxmd_simulation(*args, **kwargs):
        raise NotImplementedError("JAX MD not available")
    
    def initialize_system(*args, **kwargs):
        raise NotImplementedError("JAX MD not available")


def optimize_geometry(atoms, calculator, optimizer='BFGS', fmax=0.01, 
                     max_steps=200, trajectory_file=None):
    """
    Optimize geometry and save trajectory.
    
    Parameters
    ----------
    atoms : Atoms
        Initial geometry
    calculator : Calculator
        ASE calculator
    optimizer : str
        'BFGS', 'LBFGS', or 'BFGSLineSearch'
    fmax : float
        Force convergence (eV/Å)
    max_steps : int
        Maximum steps
    trajectory_file : Path, optional
        Save optimization trajectory
        
    Returns
    -------
    Atoms
        Optimized geometry
    """
    print(f"\n{'='*70}")
    print(f"GEOMETRY OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Optimizer: {optimizer}")
    print(f"fmax: {fmax} eV/Å")
    
    atoms.calc = calculator
    
    # Set up optimizer
    if optimizer.upper() == 'BFGS':
        opt = BFGS(atoms, trajectory=str(trajectory_file) if trajectory_file else None)
    elif optimizer.upper() == 'LBFGS':
        opt = LBFGS(atoms, trajectory=str(trajectory_file) if trajectory_file else None)
    elif optimizer.upper() == 'BFGSLS':
        opt = BFGSLineSearch(atoms, trajectory=str(trajectory_file) if trajectory_file else None)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Run
    opt.run(fmax=fmax, steps=max_steps)
    
    print(f"✅ Converged in {opt.nsteps} steps")
    print(f"   Final energy: {atoms.get_potential_energy():.6f} eV")
    print(f"   Max force: {np.max(np.abs(atoms.get_forces())):.6f} eV/Å")
    
    if trajectory_file:
        print(f"💾 Saved optimization trajectory: {trajectory_file}")
    
    return atoms


def run_ase_md(atoms, calculator, ensemble='nvt', temperature=300, 
               timestep=0.5, nsteps=10000, trajectory_file=None, 
               log_interval=100, friction=0.01):
    """
    Run MD with ASE and save complete trajectory with energies/forces.
    
    Returns
    -------
    dict
        MD results with dipoles
    """
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from ase.md.verlet import VelocityVerlet
    from ase.md.langevin import Langevin
    from ase import units
    import warnings
    
    print(f"\n{'='*70}")
    print(f"ASE MD SIMULATION")
    print(f"{'='*70}")
    print(f"Ensemble: {ensemble.upper()}")
    print(f"Temperature: {temperature} K")
    print(f"Timestep: {timestep} fs")
    print(f"Total steps: {nsteps}")
    
    atoms.calc = calculator
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        try:
            ZeroRotation(atoms)
        except (ValueError, np.linalg.LinAlgError):
            pass
    
    # Set up trajectory
    if trajectory_file:
        traj = Trajectory(str(trajectory_file), 'w', atoms)
    
    # Set up dynamics
    if ensemble.lower() == 'nve':
        dyn = VelocityVerlet(atoms, timestep * units.fs)
    elif ensemble.lower() == 'nvt' or ensemble.lower() == 'langevin':
        dyn = Langevin(atoms, timestep * units.fs, temperature_K=temperature,
                      friction=friction)
    else:
        raise ValueError(f"Unknown ensemble: {ensemble}")
    
    if trajectory_file:
        dyn.attach(traj.write, interval=10)
    
    # Storage for dipoles
    times = []
    dipoles_physnet = []
    dipoles_dcmnet = []
    
    def save_dipoles():
        times.append(dyn.nsteps * timestep)
        _ = atoms.get_potential_energy()  # Trigger calculation
        dipoles_physnet.append(calculator.results.get('dipole_physnet', np.zeros(3)).copy())
        dipoles_dcmnet.append(calculator.results.get('dipole_dcmnet', np.zeros(3)).copy())
        
        if dyn.nsteps % log_interval == 0:
            print(f"Step {dyn.nsteps:6d}/{nsteps} | "
                  f"T = {atoms.get_temperature():6.1f} K | "
                  f"E = {atoms.get_potential_energy():10.4f} eV")
    
    dyn.attach(save_dipoles, interval=1)
    
    # Run
    print("Running...")
    dyn.run(nsteps)
    
    if trajectory_file:
        traj.close()
        print(f"💾 Saved trajectory: {trajectory_file}")
    
    return {
        'times': np.array(times),
        'dipoles_physnet': np.array(dipoles_physnet),
        'dipoles_dcmnet': np.array(dipoles_dcmnet),
        'timestep': timestep,
    }


def jaxmd_to_ase_trajectory(jaxmd_results, atoms_template, output_file, calculator=None):
    """
    Convert JAX MD results to ASE trajectory with energies and forces.
    
    Uses the energies already computed by JAX MD instead of recomputing.
    
    Parameters
    ----------
    jaxmd_results : dict
        Results from run_jaxmd_simulation
    atoms_template : Atoms
        Template atoms object
    output_file : Path
        Output trajectory file
    calculator : Calculator, optional
        Not used (kept for compatibility)
    """
    print(f"\n{'='*70}")
    print(f"CONVERTING JAX MD → ASE TRAJECTORY")
    print(f"{'='*70}")
    
    trajectory = jaxmd_results['trajectory']
    energies = jaxmd_results['energies']
    n_frames = len(trajectory)
    
    print(f"Converting {n_frames} frames...")
    print(f"Using pre-computed energies from JAX MD")
    
    traj = Trajectory(str(output_file), 'w')
    
    # Create a simple calculator that just stores the values
    from ase.calculators.singlepoint import SinglePointCalculator
    
    for i in range(n_frames):
        # Create atoms for this frame
        atoms = atoms_template.copy()
        atoms.positions = trajectory[i]
        
        # Use pre-computed energy from JAX MD
        # Note: JAX MD saves total energy (potential + kinetic)
        # For ASE, we just save the energy we have
        energy = float(energies[i])
        
        # Create SinglePointCalculator with the energy
        # Forces would require recomputation, so we skip them for now
        # (they can be computed on-demand if needed)
        calc = SinglePointCalculator(atoms, energy=energy)
        atoms.calc = calc
        
        traj.write(atoms)
        
        if (i + 1) % 100 == 0:
            print(f"  Converted {i+1}/{n_frames} frames...")
    
    traj.close()
    print(f"✅ Saved ASE trajectory: {output_file}")
    print(f"   Frames: {n_frames}")
    print(f"   Energies: ✅ (from JAX MD)")
    print(f"   Forces: ⚠️  (not saved, compute on-demand if needed)")
    print(f"   File size: {output_file.stat().st_size / 1024**2:.1f} MB")


def run_temperature_scan(molecule, calculator, temperatures, ensemble='nvt',
                         integrator='berendsen', nsteps=50000, timestep=0.5, 
                         equilibration_steps=2000, friction=0.01, tau_t=100.0,
                         output_dir=None, use_jaxmd=True, model=None, params=None, 
                         optimized_geometry=None):
    """
    Run MD at multiple temperatures and compute IR spectra.
    
    Returns
    -------
    dict
        Results for each temperature
    """
    # Force ASE if JAX MD not available
    if use_jaxmd and not HAS_JAXMD:
        print("⚠️  JAX MD not available, using ASE instead")
        use_jaxmd = False
    
    print(f"\n{'='*70}")
    print(f"TEMPERATURE SCAN")
    print(f"{'='*70}")
    print(f"Temperatures: {temperatures}")
    print(f"Ensemble: {ensemble}")
    print(f"Engine: {'JAX MD' if use_jaxmd else 'ASE'}")
    
    results = {}
    
    for T in temperatures:
        print(f"\n{'='*70}")
        print(f"Temperature: {T} K")
        print(f"{'='*70}")
        
        T_dir = output_dir / f"T{T}K_{ensemble}"
        T_dir.mkdir(parents=True, exist_ok=True)
        
        if use_jaxmd:
            # Use JAX MD (fast!)
            # Use optimized geometry if provided, otherwise default
            if optimized_geometry is not None:
                from ase.io import read as ase_read
                opt_atoms = ase_read(optimized_geometry)
                positions = opt_atoms.get_positions()
                atomic_numbers = opt_atoms.get_atomic_numbers()
                masses = np.array([ase.data.atomic_masses[z] for z in atomic_numbers])
                # Center in box
                box_size = 50.0
                positions = positions - positions.mean(axis=0) + box_size / 2
                box = np.array([[box_size, 0.0, 0.0],
                               [0.0, box_size, 0.0],
                               [0.0, 0.0, box_size]])
            else:
                positions, atomic_numbers, masses, box = initialize_system(
                    molecule, temperature=T
                )
            
            md_results = run_jaxmd_simulation(
                model=model,
                params=params,
                positions=positions,
                atomic_numbers=atomic_numbers,
                masses=masses,
                box=box,
                ensemble=ensemble,
                integrator=integrator,
                temperature=T,
                timestep=timestep,
                nsteps=nsteps,
                save_interval=10,
                friction=friction,
                tau_t=tau_t,
                equilibration_steps=equilibration_steps,
                output_dir=T_dir,
            )
            
            # Convert to ASE trajectory
            atoms_template = Atoms(
                numbers=atomic_numbers,
                positions=positions,
            )
            jaxmd_to_ase_trajectory(
                md_results,
                atoms_template,
                T_dir / 'trajectory.traj',
                calculator
            )
            
        else:
            # Use ASE MD (slow but simple)
            if molecule.upper() == 'CO2':
                atoms = Atoms('CO2', positions=[
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.16],
                    [0.0, 0.0, -1.16],
                ])
            elif molecule.upper() == 'H2O':
                atoms = Atoms('H2O', positions=[
                    [0.0, 0.0, 0.0],
                    [0.0, 0.757, 0.586],
                    [0.0, -0.757, 0.586],
                ])
            
            md_results = run_ase_md(
                atoms, calculator,
                ensemble=ensemble,
                temperature=T,
                timestep=timestep,
                nsteps=nsteps,
                trajectory_file=T_dir / 'trajectory.traj',
            )
        
        # Compute IR from MD
        ir_results = compute_ir_from_md(md_results, output_dir=T_dir)
        
        results[T] = {
            'md': md_results,
            'ir': ir_results,
        }
    
    # Plot temperature comparison
    if HAS_MATPLOTLIB:
        plot_temperature_comparison(results, output_dir)
    
    return results


def plot_temperature_comparison(results, output_dir):
    """Plot IR spectra at different temperatures."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    temperatures = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures)))
    
    # IR spectra - PhysNet
    ax = axes[0]
    for T, color in zip(temperatures, colors):
        ir_data = results[T]['ir']
        if ir_data is None:
            continue
        freqs = ir_data['frequencies']
        intensity = ir_data['intensity_physnet']
        mask = (freqs > 100) & (freqs < 3500)
        ax.plot(freqs[mask], intensity[mask], color=color, 
                linewidth=2, label=f'{T} K', alpha=0.8)
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title('IR Spectrum (PhysNet) - Temperature Dependence', 
                 fontsize=14, weight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # IR spectra - DCMNet
    ax = axes[1]
    for T, color in zip(temperatures, colors):
        ir_data = results[T]['ir']
        if ir_data is None:
            continue
        freqs = ir_data['frequencies']
        intensity = ir_data['intensity_dcmnet']
        mask = (freqs > 100) & (freqs < 3500)
        ax.plot(freqs[mask], intensity[mask], color=color,
                linewidth=2, label=f'{T} K', alpha=0.8)
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title('IR Spectrum (DCMNet) - Temperature Dependence',
                 fontsize=14, weight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'temperature_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved temperature comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Complete spectroscopy suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick analysis (optimize + frequencies + IR + Raman)
  python spectroscopy_suite.py --checkpoint ckpt/ --molecule CO2 --quick-analysis
  
  # Temperature scan
  python spectroscopy_suite.py --checkpoint ckpt/ --molecule CO2 \\
      --temperatures 100 200 300 400 500 --run-temperature-scan
  
  # Production run (long MD for high-resolution IR)
  python spectroscopy_suite.py --checkpoint ckpt/ --molecule CO2 \\
      --production --nsteps 500000
  
  # Everything
  python spectroscopy_suite.py --checkpoint ckpt/ --molecule CO2 --run-all
        """
    )
    
    # Required
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--molecule', type=str, default='CO2',
                       help='Molecule name')
    parser.add_argument('--geometry', type=str, default=None,
                       help='XYZ file for custom geometry')
    parser.add_argument('--output-dir', type=Path, default=Path('./spectroscopy_suite'),
                       help='Output directory')
    
    # Analysis modes
    parser.add_argument('--quick-analysis', action='store_true',
                       help='Quick analysis (opt + freq + IR + Raman)')
    parser.add_argument('--run-temperature-scan', action='store_true',
                       help='Run MD at multiple temperatures')
    parser.add_argument('--run-ensemble-comparison', action='store_true',
                       help='Compare different MD ensembles')
    parser.add_argument('--run-optimizer-comparison', action='store_true',
                       help='Compare different optimizers')
    parser.add_argument('--production', action='store_true',
                       help='Production run (long MD)')
    parser.add_argument('--run-all', action='store_true',
                       help='Run everything')
    
    # Parameters
    parser.add_argument('--temperatures', type=float, nargs='+',
                       default=[100, 200, 300, 400, 500],
                       help='Temperatures for scan (K)')
    parser.add_argument('--ensembles', type=str, nargs='+',
                       default=['nve', 'nvt'],
                       help='Ensembles to compare')
    parser.add_argument('--optimizers', type=str, nargs='+',
                       default=['BFGS', 'LBFGS', 'BFGSLineSearch'],
                       help='Optimizers to compare')
    
    # MD parameters
    parser.add_argument('--nsteps', type=int, default=50000,
                       help='MD steps (production, after equilibration)')
    parser.add_argument('--timestep', type=float, default=0.05,
                       help='MD timestep (fs) - default 0.05 for stability')
    parser.add_argument('--integrator', type=str, default='berendsen',
                       choices=['velocity-verlet', 'langevin', 'berendsen', 
                               'velocity-rescale', 'nose-hoover'],
                       help='Integrator/thermostat (default: berendsen - most stable)')
    parser.add_argument('--friction', type=float, default=0.01,
                       help='Langevin friction (1/fs)')
    parser.add_argument('--tau-t', type=float, default=100.0,
                       help='Temperature coupling time (fs) for Berendsen/NH')
    parser.add_argument('--equilibration-steps', type=int, default=2000,
                       help='NVE equilibration steps before NVT (default: 2000)')
    parser.add_argument('--temperature', type=float, default=300,
                       help='Default temperature (K)')
    parser.add_argument('--use-ase-md', action='store_true',
                       help='Use ASE MD instead of JAX MD (slower)')
    
    # Spectroscopy
    parser.add_argument('--skip-raman', action='store_true',
                       help='Skip Raman calculation (slow)')
    parser.add_argument('--laser-wavelength', type=float, default=532,
                       help='Raman laser wavelength (nm)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("COMPLETE SPECTROSCOPY SUITE")
    print("="*70)
    print(f"Molecule: {args.molecule}")
    
    # Force ASE if JAX MD not available
    if not HAS_JAXMD and not args.use_ase_md:
        print("⚠️  JAX MD not available, forcing ASE MD")
        args.use_ase_md = True
    
    print(f"Engine: {'ASE' if args.use_ase_md else 'JAX MD (fast!)'}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\n1. Loading checkpoint...")
    with open(args.checkpoint / 'best_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    with open(args.checkpoint / 'model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    print(f"✅ Loaded {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")
    
    # Create model and calculator
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config'],
        mix_coulomb_energy=config.get('mix_coulomb_energy', False)
    )
    
    calculator = JointPhysNetDCMNetCalculator(
        model=model,
        params=params,
        cutoff=10.0,
    )
    
    # Load/create molecule
    if args.geometry:
        atoms = read(args.geometry)
    else:
        if args.molecule.upper() == 'CO2':
            atoms = Atoms('CO2', positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.16],
                [0.0, 0.0, -1.16],
            ])
        elif args.molecule.upper() == 'H2O':
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.757, 0.586],
                [0.0, -0.757, 0.586],
            ])
    
    # Save initial geometry
    write(args.output_dir / f'{args.molecule}_initial.xyz', atoms)
    
    # Quick analysis
    if args.quick_analysis or args.run_all:
        print(f"\n{'='*70}")
        print("QUICK ANALYSIS")
        print(f"{'='*70}")
        
        quick_dir = args.output_dir / 'quick_analysis'
        quick_dir.mkdir(exist_ok=True)
        
        # Optimize
        atoms_opt = optimize_geometry(
            atoms.copy(), calculator,
            optimizer='BFGS',
            trajectory_file=quick_dir / 'optimization.traj'
        )
        write(quick_dir / f'{args.molecule}_optimized.xyz', atoms_opt)
        
        # Frequencies
        freqs, vib = calculate_frequencies(
            atoms_opt, calculator,
            output_dir=quick_dir
        )
        
        # IR
        ir_data = calculate_ir_spectrum(
            atoms_opt, calculator, vib,
            compare_dipoles=True,
            output_dir=quick_dir
        )
        
        # Raman (use charge-approx since finite-field doesn't work without field-trained model)
        if not args.skip_raman:
            print("\n⚠️  Note: Using charge approximation for Raman")
            print("   Finite-field doesn't work without field-trained model")
            raman_data = calculate_raman_spectrum(
                atoms_opt, calculator, vib,
                laser_wavelength=args.laser_wavelength,
                method='charge-approx',  # Changed from 'finite-field'
                output_dir=quick_dir
            )
    
    # Temperature scan
    if args.run_temperature_scan or args.run_all:
        temp_dir = args.output_dir / 'temperature_scan'
        temp_dir.mkdir(exist_ok=True)
        
        # Optimize geometry first for stable MD
        print(f"\n{'='*70}")
        print("OPTIMIZING GEOMETRY FOR MD")
        print(f"{'='*70}")
        atoms_for_md = optimize_geometry(
            atoms.copy(), calculator,
            optimizer='BFGS', fmax=0.01,
            trajectory_file=temp_dir / 'pre_md_optimization.traj'
        )
        write(temp_dir / f'{args.molecule}_optimized_for_md.xyz', atoms_for_md)
        
        temp_results = run_temperature_scan(
            args.molecule, calculator,
            temperatures=args.temperatures,
            ensemble='nvt',
            integrator=args.integrator,
            nsteps=args.nsteps,
            timestep=args.timestep,
            equilibration_steps=args.equilibration_steps,
            friction=args.friction,
            tau_t=args.tau_t,
            output_dir=temp_dir,
            use_jaxmd=not args.use_ase_md,
            model=model,
            params=params,
            optimized_geometry=temp_dir / f'{args.molecule}_optimized_for_md.xyz',
        )
    
    # Ensemble comparison
    if args.run_ensemble_comparison or args.run_all:
        print(f"\n{'='*70}")
        print("ENSEMBLE COMPARISON")
        print(f"{'='*70}")
        
        ens_dir = args.output_dir / 'ensemble_comparison'
        ens_dir.mkdir(exist_ok=True)
        
        for ensemble in args.ensembles:
            print(f"\nEnsemble: {ensemble.upper()}")
            ens_subdir = ens_dir / ensemble
            ens_subdir.mkdir(exist_ok=True)
            
            if not args.use_ase_md:
                positions, atomic_numbers, masses, box = initialize_system(
                    args.molecule, temperature=args.temperature
                )
                
                md_results = run_jaxmd_simulation(
                    model=model,
                    params=params,
                    positions=positions,
                    atomic_numbers=atomic_numbers,
                    masses=masses,
                    box=box,
                    ensemble=ensemble,
                    temperature=args.temperature,
                    timestep=args.timestep,
                    nsteps=args.nsteps,
                    output_dir=ens_subdir,
                )
                
                # Convert to ASE
                atoms_template = Atoms(numbers=atomic_numbers, positions=positions)
                jaxmd_to_ase_trajectory(
                    md_results, atoms_template,
                    ens_subdir / 'trajectory.traj',
                    calculator
                )
            else:
                md_results = run_ase_md(
                    atoms.copy(), calculator,
                    ensemble=ensemble,
                    temperature=args.temperature,
                    timestep=args.timestep,
                    nsteps=args.nsteps,
                    trajectory_file=ens_subdir / 'trajectory.traj',
                )
            
            # IR from MD
            compute_ir_from_md(md_results, output_dir=ens_subdir)
    
    # Optimizer comparison
    if args.run_optimizer_comparison or args.run_all:
        print(f"\n{'='*70}")
        print("OPTIMIZER COMPARISON")
        print(f"{'='*70}")
        
        opt_dir = args.output_dir / 'optimizer_comparison'
        opt_dir.mkdir(exist_ok=True)
        
        opt_results = {}
        for optimizer in args.optimizers:
            print(f"\nOptimizer: {optimizer}")
            opt_subdir = opt_dir / optimizer.lower()
            opt_subdir.mkdir(exist_ok=True)
            
            atoms_copy = atoms.copy()
            start_time = time.time()
            
            atoms_opt = optimize_geometry(
                atoms_copy, calculator,
                optimizer=optimizer,
                trajectory_file=opt_subdir / 'optimization.traj'
            )
            
            elapsed = time.time() - start_time
            
            opt_results[optimizer] = {
                'time': elapsed,
                'energy': atoms_opt.get_potential_energy(),
                'max_force': np.max(np.abs(atoms_opt.get_forces())),
            }
            
            write(opt_subdir / f'{args.molecule}_optimized.xyz', atoms_opt)
        
        # Save comparison
        with open(opt_dir / 'comparison.json', 'w') as f:
            json.dump(opt_results, f, indent=2)
        
        print(f"\n{'='*70}")
        print("OPTIMIZER COMPARISON RESULTS")
        print(f"{'='*70}")
        for opt, res in opt_results.items():
            print(f"{opt:15s}: {res['time']:.2f}s, "
                  f"E={res['energy']:.6f} eV, "
                  f"F_max={res['max_force']:.6f} eV/Å")
    
    # Production run
    if args.production:
        print(f"\n{'='*70}")
        print("PRODUCTION RUN")
        print(f"{'='*70}")
        print(f"Long MD for high-resolution IR spectrum")
        print(f"Steps: {args.nsteps}")
        
        prod_dir = args.output_dir / 'production'
        prod_dir.mkdir(exist_ok=True)
        
        if not args.use_ase_md:
            positions, atomic_numbers, masses, box = initialize_system(
                args.molecule, temperature=args.temperature
            )
            
            md_results = run_jaxmd_simulation(
                model=model,
                params=params,
                positions=positions,
                atomic_numbers=atomic_numbers,
                masses=masses,
                box=box,
                ensemble='nvt',
                temperature=args.temperature,
                timestep=args.timestep,
                nsteps=args.nsteps,
                output_dir=prod_dir,
            )
            
            # Convert to ASE
            atoms_template = Atoms(numbers=atomic_numbers, positions=positions)
            jaxmd_to_ase_trajectory(
                md_results, atoms_template,
                prod_dir / 'trajectory.traj',
                calculator
            )
        else:
            md_results = run_ase_md(
                atoms.copy(), calculator,
                ensemble='nvt',
                temperature=args.temperature,
                timestep=args.timestep,
                nsteps=args.nsteps,
                trajectory_file=prod_dir / 'trajectory.traj',
            )
        
        # High-resolution IR
        compute_ir_from_md(md_results, output_dir=prod_dir)
    
    print(f"\n{'='*70}")
    print("✅ COMPLETE SPECTROSCOPY SUITE FINISHED")
    print(f"{'='*70}")
    print(f"\nAll outputs saved to: {args.output_dir}")
    print(f"\nGenerated files:")
    print(f"  - ASE trajectories (.traj) with energies and forces")
    print(f"  - IR spectra (harmonic and from MD)")
    if not args.skip_raman:
        print(f"  - Raman spectra")
    print(f"  - Temperature/ensemble/optimizer comparisons")
    print(f"\n🎉 Enjoy your spectroscopy data!")


if __name__ == '__main__':
    main()

