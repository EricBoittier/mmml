#!/usr/bin/env python3
"""
Molecular Dynamics and Vibrational Analysis using Joint PhysNet-DCMNet Model

Features:
1. Geometry optimization
2. Vibrational frequency analysis
3. IR spectra from both PhysNet and DCMNet dipoles
4. Molecular dynamics (NVE, NVT, NPT)
5. Trajectory analysis and visualization

Usage:
    # Vibrational analysis
    python dynamics_calculator.py --checkpoint ckpt/ --molecule CO2 \
        --frequencies --ir-spectra --optimize
    
    # Molecular dynamics
    python dynamics_calculator.py --checkpoint ckpt/ --molecule CO2 \
        --md --ensemble nvt --temperature 300 --timestep 0.5 --nsteps 10000
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import argparse
import warnings
from typing import Dict, Any, Optional

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp

try:
    import ase
    from ase import Atoms, units
    from ase.calculators.calculator import Calculator, all_changes
    from ase.optimize import BFGS, LBFGS, BFGSLineSearch
    from ase.vibrations import Vibrations
    from ase.thermochemistry import IdealGasThermo, HarmonicThermo
    from ase.md.verlet import VelocityVerlet
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from ase.io import read, write
    from ase.io.trajectory import Trajectory
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
    print("⚠️  Matplotlib not available. Plotting disabled.")

from trainer import JointPhysNetDCMNet
import e3x


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
    use_dcmnet_dipole : bool
        If True, use DCMNet dipole for properties; otherwise use PhysNet
    """
    
    implemented_properties = ['energy', 'forces', 'dipole', 'charges']
    
    def __init__(
        self,
        model: JointPhysNetDCMNet,
        params: Any,
        cutoff: float = 10.0,
        use_dcmnet_dipole: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.params = params
        self.cutoff = cutoff
        self.use_dcmnet_dipole = use_dcmnet_dipole
        self.natoms = model.physnet_config['natoms']
        self.n_dcm = model.dcmnet_config['n_dcm']
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Calculate properties for the given atoms."""
        super().calculate(atoms, properties, system_changes)
        
        # Get atomic data
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        n_atoms = len(atoms)
        
        # Build edge list
        dst_list = []
        src_list = []
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < self.cutoff:
                        dst_list.append(i)
                        src_list.append(j)
        
        dst_idx = np.array(dst_list, dtype=np.int32)
        src_idx = np.array(src_list, dtype=np.int32)
        
        # Prepare batch data
        batch_segments = np.zeros(n_atoms, dtype=np.int32)
        batch_mask = np.ones(len(dst_idx), dtype=np.float32)
        atom_mask = np.ones(n_atoms, dtype=np.float32)
        
        # Run model
        output = self.model.apply(
            self.params,
            atomic_numbers=jnp.array(atomic_numbers),
            positions=jnp.array(positions),
            dst_idx=jnp.array(dst_idx),
            src_idx=jnp.array(src_idx),
            batch_segments=jnp.array(batch_segments),
            batch_size=1,
            batch_mask=jnp.array(batch_mask),
            atom_mask=jnp.array(atom_mask),
        )
        
        # Extract results
        energy = float(output['energy'][0]) if output['energy'].ndim > 0 else float(output['energy'])
        forces = np.array(output['forces'][:n_atoms])
        dipole_physnet = np.array(output['dipoles'][0])
        charges_physnet = np.array(output['charges_as_mono'][:n_atoms])
        
        # DCMNet outputs for dipole calculation
        mono_dist = np.array(output['mono_dist'][:n_atoms])
        dipo_dist = np.array(output['dipo_dist'][:n_atoms])
        
        # Compute DCMNet dipole relative to COM
        import ase.data
        masses = np.array([ase.data.atomic_masses[z] for z in atomic_numbers])
        com = np.sum(positions * masses[:, None], axis=0) / masses.sum()
        dipo_rel_com = dipo_dist - com[None, None, :]
        dipole_dcmnet = np.sum(mono_dist[..., None] * dipo_rel_com, axis=(0, 1))
        
        # Store results
        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['dipole'] = dipole_dcmnet if self.use_dcmnet_dipole else dipole_physnet
        self.results['charges'] = charges_physnet
        
        # Store both dipoles for analysis
        self.results['dipole_physnet'] = dipole_physnet
        self.results['dipole_dcmnet'] = dipole_dcmnet
        self.results['charges_dcmnet'] = mono_dist
        self.results['positions_dcmnet'] = dipo_dist


def optimize_geometry(atoms, calculator, fmax=0.01, max_steps=200, optimizer='BFGS'):
    """
    Optimize molecular geometry.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object
    calculator : Calculator
        ASE calculator
    fmax : float
        Force convergence criterion (eV/Å)
    max_steps : int
        Maximum optimization steps
    optimizer : str
        Optimizer type: 'BFGS', 'LBFGS', or 'BFGSLineSearch'
        
    Returns
    -------
    Atoms
        Optimized geometry
    """
    atoms.calc = calculator
    
    print(f"\n{'='*70}")
    print(f"GEOMETRY OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Optimizer: {optimizer}")
    print(f"Force convergence: {fmax} eV/Å")
    print(f"Max steps: {max_steps}")
    
    # Select optimizer
    if optimizer.upper() == 'BFGS':
        opt = BFGS(atoms, logfile='optimization.log')
    elif optimizer.upper() == 'LBFGS':
        opt = LBFGS(atoms, logfile='optimization.log')
    elif optimizer.upper() == 'BFGSLS':
        opt = BFGSLineSearch(atoms, logfile='optimization.log')
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Run optimization
    opt.run(fmax=fmax, steps=max_steps)
    
    print(f"\n✅ Optimization converged in {opt.nsteps} steps")
    print(f"Final energy: {atoms.get_potential_energy():.6f} eV")
    print(f"Max force: {np.max(np.abs(atoms.get_forces())):.6f} eV/Å")
    
    return atoms


def calculate_frequencies(atoms, calculator, delta=0.01, nfree=2, output_dir=None):
    """
    Calculate vibrational frequencies using numerical Hessian.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object (should be at optimized geometry)
    calculator : Calculator
        ASE calculator
    delta : float
        Displacement for numerical differentiation (Å)
    nfree : int
        Number of degrees of freedom to remove (3 translations + 3 rotations)
    output_dir : Path, optional
        Directory for output files
        
    Returns
    -------
    tuple
        (frequencies in cm^-1, vibrations object)
    """
    atoms.calc = calculator
    
    print(f"\n{'='*70}")
    print(f"VIBRATIONAL FREQUENCY ANALYSIS")
    print(f"{'='*70}")
    print(f"Displacement: {delta} Å")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Expected modes: {3*len(atoms) - 6}")
    
    # Create vibrations object
    vib_dir = output_dir / 'vibrations' if output_dir else Path('vibrations')
    vib_dir.mkdir(parents=True, exist_ok=True)
    
    vib = Vibrations(atoms, name=str(vib_dir / 'vib'), delta=delta, nfree=nfree)
    
    # Run frequency calculation
    print("\nCalculating Hessian (this may take a while)...")
    vib.run()
    
    # Summarize results
    print("\n" + "="*70)
    vib.summary(log=sys.stdout)
    print("="*70)
    
    # Get frequencies
    freqs = vib.get_frequencies()
    
    return freqs, vib


def calculate_ir_spectrum(atoms, calculator, vib, delta=0.01, temp=300, 
                         compare_dipoles=True, output_dir=None):
    """
    Calculate IR spectrum from dipole derivatives.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object
    calculator : Calculator
        ASE calculator (should have dipole_physnet and dipole_dcmnet)
    vib : Vibrations
        Vibrations object from calculate_frequencies
    delta : float
        Displacement for numerical differentiation
    temp : float
        Temperature for intensity calculation (K)
    compare_dipoles : bool
        If True, compare PhysNet and DCMNet IR spectra
    output_dir : Path, optional
        Directory for output files
        
    Returns
    -------
    dict
        IR spectrum data
    """
    print(f"\n{'='*70}")
    print(f"IR SPECTRUM CALCULATION")
    print(f"{'='*70}")
    print(f"Temperature: {temp} K")
    
    atoms.calc = calculator
    freqs = vib.get_frequencies()
    
    # Calculate dipole derivatives for PhysNet
    print("\nCalculating dipole derivatives (PhysNet)...")
    dipole_deriv_physnet = np.zeros((len(freqs), 3))
    
    for i in range(len(freqs)):
        # Get mode from ASE Vibrations object
        mode = vib.get_mode(i)
        
        # Displace along mode
        atoms_plus = atoms.copy()
        atoms_plus.positions += delta * mode.reshape(-1, 3)
        atoms_plus.calc = calculator
        _ = atoms_plus.get_potential_energy()  # Trigger calculation
        dipole_plus = calculator.results.get('dipole_physnet', atoms_plus.get_dipole_moment())
        
        atoms_minus = atoms.copy()
        atoms_minus.positions -= delta * mode.reshape(-1, 3)
        atoms_minus.calc = calculator
        _ = atoms_minus.get_potential_energy()  # Trigger calculation
        dipole_minus = calculator.results.get('dipole_physnet', atoms_minus.get_dipole_moment())
        
        # Numerical derivative
        dipole_deriv_physnet[i] = (dipole_plus - dipole_minus) / (2 * delta)
    
    # Calculate intensities (proportional to |dμ/dQ|^2)
    intensities_physnet = np.sum(dipole_deriv_physnet**2, axis=1)
    
    results = {
        'frequencies': freqs,
        'intensities_physnet': intensities_physnet,
        'dipole_derivatives_physnet': dipole_deriv_physnet,
    }
    
    # Compare with DCMNet dipoles if requested
    if compare_dipoles:
        print("Calculating dipole derivatives (DCMNet)...")
        dipole_deriv_dcmnet = np.zeros((len(freqs), 3))
        
        for i in range(len(freqs)):
            mode = vib.get_mode(i)
            
            atoms_plus = atoms.copy()
            atoms_plus.positions += delta * mode.reshape(-1, 3)
            atoms_plus.calc = calculator
            _ = atoms_plus.get_potential_energy()  # Trigger calculation
            dipole_plus = calculator.results.get('dipole_dcmnet')
            
            atoms_minus = atoms.copy()
            atoms_minus.positions -= delta * mode.reshape(-1, 3)
            atoms_minus.calc = calculator
            _ = atoms_minus.get_potential_energy()
            dipole_minus = calculator.results.get('dipole_dcmnet')
            
            dipole_deriv_dcmnet[i] = (dipole_plus - dipole_minus) / (2 * delta)
        
        intensities_dcmnet = np.sum(dipole_deriv_dcmnet**2, axis=1)
        
        results['intensities_dcmnet'] = intensities_dcmnet
        results['dipole_derivatives_dcmnet'] = dipole_deriv_dcmnet
    
    # Print results
    print(f"\n{'Frequency (cm⁻¹)':>18} {'Intensity (PhysNet)':>22}", end='')
    if compare_dipoles:
        print(f" {'Intensity (DCMNet)':>22}")
    else:
        print()
    print("-" * 70)
    
    # Convert to real and filter positive frequencies
    freqs_real = np.real(freqs)
    for i, freq in enumerate(freqs_real):
        if freq > 100:  # Skip low frequencies (rotations/translations)
            print(f"{freq:18.2f} {intensities_physnet[i]:22.6f}", end='')
            if compare_dipoles:
                print(f" {intensities_dcmnet[i]:22.6f}")
            else:
                print()
    
    # Plot IR spectrum
    if HAS_MATPLOTLIB and output_dir:
        plot_ir_spectrum(results, output_dir, temp=temp)
    
    return results


def plot_ir_spectrum(ir_data, output_dir, temp=300, broadening=10):
    """
    Plot IR spectrum with Lorentzian broadening.
    
    Parameters
    ----------
    ir_data : dict
        IR spectrum data from calculate_ir_spectrum
    output_dir : Path
        Output directory
    temp : float
        Temperature (K)
    broadening : float
        Lorentzian broadening (cm^-1)
    """
    freqs = ir_data['frequencies']
    intensities_physnet = ir_data['intensities_physnet']
    
    # Convert to real (handles complex frequencies from ASE)
    freqs = np.real(freqs)
    
    # Filter positive frequencies (> 100 cm^-1 to skip rotations/translations)
    mask = freqs > 100
    freqs = freqs[mask]
    intensities_physnet = intensities_physnet[mask]
    
    if len(freqs) == 0:
        print("⚠️  No positive vibrational frequencies found. Skipping IR plot.")
        return
    
    # Create broadened spectrum
    freq_range = np.linspace(0, max(freqs) + 500, 2000)
    spectrum_physnet = np.zeros_like(freq_range)
    
    for freq, intensity in zip(freqs, intensities_physnet):
        # Lorentzian broadening
        spectrum_physnet += intensity * (broadening**2) / ((freq_range - freq)**2 + broadening**2)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Stick spectrum
    ax = axes[0]
    markerline, stemlines, baseline = ax.stem(freqs, intensities_physnet, 
                                               linefmt='C0-', markerfmt='C0o', basefmt=' ')
    markerline.set_label('PhysNet')
    
    if 'intensities_dcmnet' in ir_data:
        intensities_dcmnet = ir_data['intensities_dcmnet'][mask]
        # Offset slightly for visibility
        markerline, stemlines, baseline = ax.stem(freqs + 5, intensities_dcmnet,
                                                   linefmt='C1-', markerfmt='C1^', basefmt=' ')
        markerline.set_label('DCMNet')
        markerline.set_markersize(6)
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Intensity (arb. units)', fontsize=12, weight='bold')
    ax.set_title('IR Spectrum (Stick)', fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Broadened spectrum
    ax = axes[1]
    ax.plot(freq_range, spectrum_physnet, 'C0-', linewidth=2, label='PhysNet')
    
    if 'intensities_dcmnet' in ir_data:
        intensities_dcmnet = ir_data['intensities_dcmnet'][mask]
        spectrum_dcmnet = np.zeros_like(freq_range)
        for freq, intensity in zip(freqs, intensities_dcmnet):
            spectrum_dcmnet += intensity * (broadening**2) / ((freq_range - freq)**2 + broadening**2)
        ax.plot(freq_range, spectrum_dcmnet, 'C1--', linewidth=2, label='DCMNet')
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Intensity (arb. units)', fontsize=12, weight='bold')
    ax.set_title(f'IR Spectrum (Broadened, FWHM = {broadening} cm⁻¹)', fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'ir_spectrum.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved IR spectrum: {output_path}")
    
    # Save harmonic data to NPZ for frequency scaling analysis
    npz_path = output_dir / 'harmonic_ir.npz'
    np.savez(
        npz_path,
        frequencies=freqs,
        intensities_physnet=intensities_physnet,
        intensities_dcmnet=intensities_dcmnet if 'intensities_dcmnet' in ir_data else None,
        dipole_derivatives_physnet=ir_data.get('dipole_derivatives_physnet', None),
        dipole_derivatives_dcmnet=ir_data.get('dipole_derivatives_dcmnet', None),
    )
    print(f"✅ Saved harmonic data: {npz_path}")


def run_molecular_dynamics(atoms, calculator, ensemble='nvt', temperature=300,
                          timestep=0.5, nsteps=10000, log_interval=100,
                          traj_interval=10, save_dipoles=True, output_dir=None):
    """
    Run molecular dynamics simulation.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object
    calculator : Calculator
        ASE calculator
    ensemble : str
        MD ensemble: 'nve' or 'nvt'
    temperature : float
        Temperature (K) for NVT or initial temperature for NVE
    timestep : float
        MD timestep (fs)
    nsteps : int
        Number of MD steps
    log_interval : int
        Interval for logging energies
    traj_interval : int
        Interval for writing trajectory
    save_dipoles : bool
        If True, save dipole moments at every step for IR analysis
    output_dir : Path, optional
        Output directory
        
    Returns
    -------
    dict
        MD results (energies, temperatures, dipoles, etc.)
    """
    print(f"\n{'='*70}")
    print(f"MOLECULAR DYNAMICS SIMULATION")
    print(f"{'='*70}")
    print(f"Ensemble: {ensemble.upper()}")
    print(f"Temperature: {temperature} K")
    print(f"Timestep: {timestep} fs")
    print(f"Total steps: {nsteps}")
    print(f"Total time: {nsteps * timestep / 1000:.2f} ps")
    if save_dipoles:
        print(f"Saving dipoles every step for IR spectrum analysis")
    
    atoms.calc = calculator
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)  # Remove COM motion
    
    # Remove rotation (suppress warnings for linear molecules)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        try:
            ZeroRotation(atoms)
        except (ValueError, np.linalg.LinAlgError):
            # Linear molecules have only 2 rotational DOF
            pass
    
    # Set up trajectory file
    traj_dir = output_dir if output_dir else Path('.')
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_file = traj_dir / f'md_{ensemble}.traj'
    traj = Trajectory(str(traj_file), 'w', atoms)
    
    # Set up dynamics
    if ensemble.lower() == 'nve':
        dyn = VelocityVerlet(atoms, timestep * units.fs)
        print(f"NVE dynamics (constant energy)")
    elif ensemble.lower() == 'nvt':
        # Langevin thermostat
        friction = 0.01  # friction coefficient (1/fs)
        dyn = Langevin(atoms, timestep * units.fs, temperature_K=temperature,
                      friction=friction)
        print(f"NVT dynamics (Langevin thermostat, friction={friction:.3f} 1/fs)")
    else:
        raise ValueError(f"Unknown ensemble: {ensemble}")
    
    # Attach trajectory writer
    dyn.attach(traj.write, interval=traj_interval)
    
    # Storage for monitoring
    times = []
    energies_pot = []
    energies_kin = []
    energies_tot = []
    temperatures = []
    dipoles_physnet = [] if save_dipoles else None
    dipoles_dcmnet = [] if save_dipoles else None
    
    def log_status(a=atoms):
        """Log current MD status and save dipoles."""
        times.append(dyn.nsteps * timestep)
        energies_pot.append(a.get_potential_energy())
        energies_kin.append(a.get_kinetic_energy())
        energies_tot.append(energies_pot[-1] + energies_kin[-1])
        temperatures.append(a.get_temperature())
        
        # Save dipole moments if requested
        if save_dipoles:
            dipole_physnet = calculator.results.get('dipole_physnet', np.zeros(3))
            dipole_dcmnet = calculator.results.get('dipole_dcmnet', np.zeros(3))
            dipoles_physnet.append(dipole_physnet.copy())
            dipoles_dcmnet.append(dipole_dcmnet.copy())
        
        if dyn.nsteps % log_interval == 0:
            print(f"Step {dyn.nsteps:6d} | "
                  f"T = {temperatures[-1]:6.1f} K | "
                  f"Epot = {energies_pot[-1]:10.4f} eV | "
                  f"Ekin = {energies_kin[-1]:10.4f} eV | "
                  f"Etot = {energies_tot[-1]:10.4f} eV")
    
    # Attach logger
    dyn.attach(log_status, interval=1)
    
    # Run MD
    print("\nRunning MD...")
    dyn.run(nsteps)
    
    traj.close()
    
    print(f"\n✅ MD simulation complete")
    print(f"Trajectory saved: {traj_file}")
    
    # Collect results
    results = {
        'times': np.array(times),
        'energies_pot': np.array(energies_pot),
        'energies_kin': np.array(energies_kin),
        'energies_tot': np.array(energies_tot),
        'temperatures': np.array(temperatures),
        'timestep': timestep,
    }
    
    if save_dipoles:
        results['dipoles_physnet'] = np.array(dipoles_physnet)
        results['dipoles_dcmnet'] = np.array(dipoles_dcmnet)
        print(f"Saved {len(dipoles_physnet)} dipole snapshots")
    
    # Plot results
    if HAS_MATPLOTLIB and output_dir:
        plot_md_results(results, output_dir, ensemble=ensemble)
    
    return results


def compute_ir_from_md(md_data, output_dir=None, max_lag=None):
    """
    Compute IR spectrum from MD dipole autocorrelation function.
    
    This uses the fluctuation-dissipation theorem: the IR absorption
    is proportional to the Fourier transform of the dipole autocorrelation.
    
    Parameters
    ----------
    md_data : dict
        MD results containing 'dipoles_physnet', 'dipoles_dcmnet', 'timestep'
    output_dir : Path, optional
        Output directory for plots
    max_lag : int, optional
        Maximum lag for autocorrelation (default: len/2)
        
    Returns
    -------
    dict
        IR spectrum data
    """
    if 'dipoles_physnet' not in md_data:
        print("⚠️  No dipole data found. Run MD with save_dipoles=True")
        return None
    
    print(f"\n{'='*70}")
    print(f"IR SPECTRUM FROM MD")
    print(f"{'='*70}")
    
    dipoles_physnet = md_data['dipoles_physnet']
    dipoles_dcmnet = md_data['dipoles_dcmnet']
    timestep = md_data['timestep']  # in fs
    
    n_steps = len(dipoles_physnet)
    if max_lag is None:
        max_lag = n_steps  # Use ALL data for best frequency resolution!
    
    print(f"Timesteps: {n_steps}")
    print(f"Timestep: {timestep} fs")
    print(f"Max lag for autocorrelation: {max_lag} steps ({max_lag * timestep:.1f} fs)")
    print(f"  → Frequency resolution: {1.0 / (n_steps * timestep * 1e-15) / 3e10:.2f} cm⁻¹")
    
    # Compute dipole autocorrelation functions
    print("\nComputing dipole autocorrelation (FFT method for speed)...")
    
    def autocorrelation_fft(data, max_lag):
        """
        Compute autocorrelation using FFT (much faster!).
        
        O(N log N) instead of O(N²)
        """
        n = len(data)
        # Subtract mean
        data = data - np.mean(data, axis=0)
        
        # Zero-pad to next power of 2 for efficient FFT
        n_fft = 2**(int(np.ceil(np.log2(2*n - 1))))
        
        # Compute autocorrelation via FFT for each component
        acf = np.zeros((max_lag, 3))
        for i in range(3):
            # FFT of signal
            fft_data = np.fft.fft(data[:, i], n=n_fft)
            # Power spectrum
            power = fft_data * np.conj(fft_data)
            # Inverse FFT gives autocorrelation
            acf_full = np.fft.ifft(power).real[:n]
            # Normalize by number of overlapping points
            acf[:, i] = acf_full[:max_lag] / np.arange(n, n-max_lag, -1)
        
        # Average over xyz components
        return np.mean(acf, axis=1)
    
    acf_physnet = autocorrelation_fft(dipoles_physnet, max_lag)
    acf_dcmnet = autocorrelation_fft(dipoles_dcmnet, max_lag)
    
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(max_lag)
    acf_physnet_windowed = acf_physnet * window
    acf_dcmnet_windowed = acf_dcmnet * window
    
    # Fourier transform to get IR spectrum
    print("Computing Fourier transform...")
    
    # Zero-pad for better frequency resolution
    nfft = max_lag * 4
    spectrum_physnet = np.fft.rfft(acf_physnet_windowed, n=nfft)
    spectrum_dcmnet = np.fft.rfft(acf_dcmnet_windowed, n=nfft)
    
    # Get frequencies in cm^-1
    # freq (Hz) = n / (N * dt)
    # Convert fs to s: 1 fs = 1e-15 s
    # Convert Hz to cm^-1: 1 Hz = 1/(c*100) cm^-1, c = 2.998e8 m/s
    freq_hz = np.fft.rfftfreq(nfft, d=timestep * 1e-15)
    c_cm_per_s = 2.99792458e10  # speed of light in cm/s
    freqs_cm = freq_hz / c_cm_per_s
    
    # IR intensity formula with quantum and frequency corrections:
    # I(ω) ∝ ω × (1 + n(ω)) × |FT[C(t)]|²
    # where n(ω) = 1/(exp(ℏω/kT) - 1) is Bose-Einstein distribution
    
    # Constants
    hbar_eV = 6.582119569e-16  # eV·s
    kB_eV = 8.617333262e-5  # eV/K
    T_kelvin = 300  # Assume 300 K (could be parameter)
    
    # Angular frequency in rad/s
    omega = 2 * np.pi * freq_hz
    
    # Quantum occupation number (Bose-Einstein)
    # n(ω) = 1 / (exp(ℏω/kT) - 1)
    # Avoid division by zero at ω=0
    with np.errstate(divide='ignore', invalid='ignore'):
        n_bose = 1.0 / (np.exp(hbar_eV * omega / (kB_eV * T_kelvin)) - 1.0)
        n_bose = np.nan_to_num(n_bose, nan=0.0, posinf=0.0)
    
    # Frequency prefactor (ω for classical, or ω(1+n) for quantum)
    # Using quantum correction: ω × (1 + n(ω))
    freq_prefactor = omega * (1.0 + n_bose)
    
    # Apply to spectrum
    intensity_physnet = freq_prefactor * np.abs(spectrum_physnet)**2
    intensity_dcmnet = freq_prefactor * np.abs(spectrum_dcmnet)**2
    
    # Normalize
    intensity_physnet /= np.max(intensity_physnet)
    intensity_dcmnet /= np.max(intensity_dcmnet)
    
    results = {
        'frequencies': freqs_cm,
        'intensity_physnet': intensity_physnet,
        'intensity_dcmnet': intensity_dcmnet,
        'acf_physnet': acf_physnet,
        'acf_dcmnet': acf_dcmnet,
        'acf_times': np.arange(max_lag) * timestep,
    }
    
    print(f"\nFrequency range: 0 - {freqs_cm[-1]:.1f} cm⁻¹")
    print(f"Frequency resolution: {freqs_cm[1] - freqs_cm[0]:.2f} cm⁻¹")
    
    # Plot results
    if HAS_MATPLOTLIB and output_dir:
        plot_ir_from_md(results, output_dir)
    
    return results


def plot_ir_from_md(ir_data, output_dir):
    """
    Plot IR spectrum and autocorrelation from MD.
    
    Parameters
    ----------
    ir_data : dict
        IR spectrum data from compute_ir_from_md
    output_dir : Path
        Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Dipole autocorrelation - PhysNet
    ax = axes[0, 0]
    ax.plot(ir_data['acf_times'], ir_data['acf_physnet'], 'C0-', linewidth=1.5)
    ax.set_xlabel('Time (fs)', fontsize=11, weight='bold')
    ax.set_ylabel('Dipole ACF (e²·Å²)', fontsize=11, weight='bold')
    ax.set_title('PhysNet: Dipole Autocorrelation', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    # Dipole autocorrelation - DCMNet
    ax = axes[0, 1]
    ax.plot(ir_data['acf_times'], ir_data['acf_dcmnet'], 'C1-', linewidth=1.5)
    ax.set_xlabel('Time (fs)', fontsize=11, weight='bold')
    ax.set_ylabel('Dipole ACF (e²·Å²)', fontsize=11, weight='bold')
    ax.set_title('DCMNet: Dipole Autocorrelation', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    # IR spectrum - full range
    ax = axes[1, 0]
    freqs = ir_data['frequencies']
    mask = (freqs > 0) & (freqs < 5000)  # Focus on relevant range
    ax.plot(freqs[mask], ir_data['intensity_physnet'][mask], 'C0-', 
            linewidth=2, label='PhysNet', alpha=0.8)
    ax.plot(freqs[mask], ir_data['intensity_dcmnet'][mask], 'C1--', 
            linewidth=2, label='DCMNet', alpha=0.8)
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=11, weight='bold')
    ax.set_ylabel('Intensity (normalized)', fontsize=11, weight='bold')
    ax.set_title('IR Spectrum from MD (0-5000 cm⁻¹)', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5000)
    
    # IR spectrum - vibrational range (zoomed)
    ax = axes[1, 1]
    mask = (freqs > 500) & (freqs < 3500)
    ax.plot(freqs[mask], ir_data['intensity_physnet'][mask], 'C0-', 
            linewidth=2, label='PhysNet', alpha=0.8)
    ax.plot(freqs[mask], ir_data['intensity_dcmnet'][mask], 'C1--', 
            linewidth=2, label='DCMNet', alpha=0.8)
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=11, weight='bold')
    ax.set_ylabel('Intensity (normalized)', fontsize=11, weight='bold')
    ax.set_title('IR Spectrum from MD (500-3500 cm⁻¹)', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(500, 3500)
    
    plt.suptitle('IR Spectrum from Molecular Dynamics', fontsize=16, weight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'ir_spectrum_md.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved IR spectrum from MD: {output_path}")
    
    # Save MD spectrum data to NPZ for frequency scaling analysis
    npz_path = output_dir / 'md_ir_spectrum.npz'
    np.savez(
        npz_path,
        frequencies=ir_data['frequencies'],
        intensities=ir_data['intensity_physnet'],  # Use PhysNet as primary
        intensity_physnet=ir_data['intensity_physnet'],
        intensity_dcmnet=ir_data['intensity_dcmnet'],
        autocorrelation=ir_data['acf_physnet'],
        times=ir_data['acf_times'],
    )
    print(f"✅ Saved MD spectrum data: {npz_path}")


def plot_md_results(md_data, output_dir, ensemble='nvt'):
    """
    Plot MD simulation results.
    
    Parameters
    ----------
    md_data : dict
        MD results from run_molecular_dynamics
    output_dir : Path
        Output directory
    ensemble : str
        MD ensemble
    """
    times = md_data['times']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Energy vs time
    ax = axes[0, 0]
    ax.plot(times, md_data['energies_pot'], 'C0-', label='Potential', linewidth=1.5)
    ax.plot(times, md_data['energies_kin'], 'C1-', label='Kinetic', linewidth=1.5)
    ax.plot(times, md_data['energies_tot'], 'C2-', label='Total', linewidth=2)
    ax.set_xlabel('Time (fs)', fontsize=11, weight='bold')
    ax.set_ylabel('Energy (eV)', fontsize=11, weight='bold')
    ax.set_title('Energy vs Time', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Temperature vs time
    ax = axes[0, 1]
    ax.plot(times, md_data['temperatures'], 'C3-', linewidth=1.5)
    ax.axhline(np.mean(md_data['temperatures']), color='k', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(md_data["temperatures"]):.1f} K')
    ax.set_xlabel('Time (fs)', fontsize=11, weight='bold')
    ax.set_ylabel('Temperature (K)', fontsize=11, weight='bold')
    ax.set_title('Temperature vs Time', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Energy distribution
    ax = axes[1, 0]
    ax.hist(md_data['energies_tot'], bins=50, alpha=0.7, color='C2', edgecolor='black')
    ax.axvline(np.mean(md_data['energies_tot']), color='k', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(md_data["energies_tot"]):.4f} eV')
    ax.set_xlabel('Total Energy (eV)', fontsize=11, weight='bold')
    ax.set_ylabel('Count', fontsize=11, weight='bold')
    ax.set_title('Energy Distribution', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Temperature distribution
    ax = axes[1, 1]
    ax.hist(md_data['temperatures'], bins=50, alpha=0.7, color='C3', edgecolor='black')
    ax.axvline(np.mean(md_data['temperatures']), color='k', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(md_data["temperatures"]):.1f} K')
    ax.set_xlabel('Temperature (K)', fontsize=11, weight='bold')
    ax.set_ylabel('Count', fontsize=11, weight='bold')
    ax.set_title('Temperature Distribution', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{ensemble.upper()} Molecular Dynamics', fontsize=16, weight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'md_{ensemble}_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved MD results plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='MD and vibrational analysis')
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--molecule', type=str, default='CO2',
                       help='Molecule formula')
    parser.add_argument('--geometry', type=str, default=None,
                       help='XYZ file with initial geometry')
    parser.add_argument('--output-dir', type=Path, default=Path('./analysis'),
                       help='Output directory')
    
    # Optimization
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize geometry before analysis')
    parser.add_argument('--fmax', type=float, default=0.01,
                       help='Force convergence for optimization (eV/Å)')
    
    # Vibrational analysis
    parser.add_argument('--frequencies', action='store_true',
                       help='Calculate vibrational frequencies')
    parser.add_argument('--ir-spectra', action='store_true',
                       help='Calculate IR spectra')
    parser.add_argument('--vib-delta', type=float, default=0.01,
                       help='Displacement for vibrational analysis (Å)')
    
    # Molecular dynamics
    parser.add_argument('--md', action='store_true',
                       help='Run molecular dynamics')
    parser.add_argument('--ensemble', type=str, default='nvt', choices=['nve', 'nvt'],
                       help='MD ensemble')
    parser.add_argument('--temperature', type=float, default=300,
                       help='Temperature (K)')
    parser.add_argument('--timestep', type=float, default=0.5,
                       help='MD timestep (fs)')
    parser.add_argument('--nsteps', type=int, default=10000,
                       help='Number of MD steps')
    parser.add_argument('--ir-from-md', action='store_true', default=True,
                       help='Compute IR spectrum from MD dipole autocorrelation (default: True)')
    
    # Dipole choice
    parser.add_argument('--use-dcmnet-dipole', action='store_true',
                       help='Use DCMNet dipole instead of PhysNet')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Joint PhysNet-DCMNet: Dynamics & Vibrational Analysis")
    print("="*70)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\n1. Loading checkpoint from {args.checkpoint}")
    with open(args.checkpoint / 'best_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    with open(args.checkpoint / 'model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    print(f"✅ Loaded {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")
    
    # Create model
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config'],
        mix_coulomb_energy=config.get('mix_coulomb_energy', False)
    )
    
    # Create calculator
    calculator = JointPhysNetDCMNetCalculator(
        model=model,
        params=params,
        cutoff=10.0,
        use_dcmnet_dipole=args.use_dcmnet_dipole
    )
    
    # Load or create molecule
    print(f"\n2. Setting up molecule: {args.molecule}")
    
    if args.geometry:
        atoms = read(args.geometry)
        print(f"✅ Loaded geometry from {args.geometry}")
    else:
        # Create default molecule
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
        else:
            raise ValueError(f"Unknown molecule: {args.molecule}. Provide --geometry")
        
        print(f"✅ Using default {args.molecule} geometry")
    
    # Optimize geometry if requested
    if args.optimize:
        atoms = optimize_geometry(atoms, calculator, fmax=args.fmax)
        # Save optimized geometry
        write(args.output_dir / f'{args.molecule}_optimized.xyz', atoms)
        print(f"✅ Saved optimized geometry: {args.output_dir / f'{args.molecule}_optimized.xyz'}")
    
    # Vibrational analysis
    if args.frequencies or args.ir_spectra:
        freqs, vib = calculate_frequencies(atoms, calculator, 
                                          delta=args.vib_delta,
                                          output_dir=args.output_dir)
        
        if args.ir_spectra:
            ir_data = calculate_ir_spectrum(atoms, calculator, vib,
                                           delta=args.vib_delta,
                                           temp=args.temperature,
                                           compare_dipoles=True,
                                           output_dir=args.output_dir)
    
    # Molecular dynamics
    if args.md:
        md_results = run_molecular_dynamics(atoms, calculator,
                                           ensemble=args.ensemble,
                                           temperature=args.temperature,
                                           timestep=args.timestep,
                                           nsteps=args.nsteps,
                                           save_dipoles=True,
                                           output_dir=args.output_dir)
        
        # Compute IR spectrum from MD dipole autocorrelation
        if args.ir_from_md:
            print("\nAnalyzing MD trajectory for IR spectrum...")
            ir_md = compute_ir_from_md(md_results, output_dir=args.output_dir)
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

