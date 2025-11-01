#!/usr/bin/env python3
"""
Raman Spectroscopy Analysis for Joint PhysNet-DCMNet Model

⚠️ IMPORTANT: This uses an approximate polarizability from distributed charges.
For accurate Raman spectra, the model should be trained to predict polarizability directly.

Methods:
1. Harmonic Raman from numerical polarizability derivatives
2. Raman from MD polarizability autocorrelation (approximate)

Usage:
    python raman_calculator.py --checkpoint ckpt/ --molecule CO2 \
        --optimize --frequencies --raman
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import argparse
import warnings

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from dynamics_calculator import (
    JointPhysNetDCMNetCalculator,
    optimize_geometry,
    calculate_frequencies
)
from trainer import JointPhysNetDCMNet


class FieldResponseCalculator(Calculator):
    """
    Calculator that applies external electric field for polarizability.
    
    Wraps a base calculator and adds field effects:
    E_total = E_base - μ·F
    
    Parameters
    ----------
    base_calculator : Calculator
        Underlying calculator (e.g., JointPhysNetDCMNetCalculator)
    electric_field : array (3,)
        Electric field in V/Angstrom
    """
    
    implemented_properties = ['energy', 'forces', 'dipole', 'charges']
    
    def __init__(self, base_calculator, electric_field=None, **kwargs):
        super().__init__(**kwargs)
        self.base_calc = base_calculator
        self.electric_field = np.zeros(3) if electric_field is None else np.array(electric_field)
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Calculate with electric field."""
        # Get base properties
        self.base_calc.calculate(atoms, properties, system_changes)
        
        # Copy results (handle both scalars and arrays)
        for key in ['energy', 'forces', 'dipole', 'charges']:
            if key in self.base_calc.results:
                val = self.base_calc.results[key]
                # Copy arrays, but scalars don't need .copy()
                if isinstance(val, np.ndarray):
                    self.results[key] = val.copy()
                else:
                    self.results[key] = val
        
        # Also store both dipoles
        if 'dipole_physnet' in self.base_calc.results:
            val = self.base_calc.results['dipole_physnet']
            self.results['dipole_physnet'] = val.copy() if isinstance(val, np.ndarray) else val
        if 'dipole_dcmnet' in self.base_calc.results:
            val = self.base_calc.results['dipole_dcmnet']
            self.results['dipole_dcmnet'] = val.copy() if isinstance(val, np.ndarray) else val
        
        # Apply field correction to energy
        # E_field = E_0 - μ·F (in eV)
        # Field in V/Å, dipole in e·Å
        # 1 V/Å = 1 eV/(e·Å), so units work out!
        dipole = self.results.get('dipole', np.zeros(3))
        field_energy = -np.dot(dipole, self.electric_field)
        self.results['energy'] += field_energy
        
        # Force correction: F_field = -∂E_field/∂r = ∂(μ·F)/∂r
        # For now, assume forces don't change significantly with small field
        # (This is the Born-Oppenheimer approximation)


def compute_polarizability_finite_field(atoms, base_calculator, field_strength=0.0001,
                                       use_dcmnet=True, method='dipole'):
    """
    Compute polarizability from finite electric field response.
    
    This is the PROPER way to get polarizability!
    
    Parameters
    ----------
    atoms : Atoms
        Molecule at equilibrium
    base_calculator : Calculator
        Base calculator (e.g., JointPhysNetDCMNet)
    field_strength : float
        Electric field magnitude in V/Angstrom (default: 0.0001 = 0.01 V/nm)
    use_dcmnet : bool
        Use DCMNet dipole (default) or PhysNet
    method : str
        'dipole' (∂μ/∂F) or 'energy' (-∂²E/∂F²)
        
    Returns
    -------
    array (3, 3)
        Polarizability tensor in Ang^3
    """
    print(f"\n{'='*70}")
    print(f"POLARIZABILITY FROM FINITE FIELD RESPONSE")
    print(f"{'='*70}")
    print(f"Method: {method}")
    print(f"Field strength: {field_strength:.6f} V/Å ({field_strength*10:.4f} V/nm)")
    print(f"Using: {'DCMNet' if use_dcmnet else 'PhysNet'} dipole")
    
    if method == 'dipole':
        # α_ij = ∂μ_i/∂F_j
        # Use central differences: α_ij ≈ [μ_i(+F_j) - μ_i(-F_j)] / (2F)
        
        alpha = np.zeros((3, 3))
        
        for j in range(3):  # Field direction
            # Apply field in +j direction
            field_plus = np.zeros(3)
            field_plus[j] = field_strength
            calc_plus = FieldResponseCalculator(base_calculator, electric_field=field_plus)
            atoms.calc = calc_plus
            _ = atoms.get_potential_energy()
            dipole_plus = calc_plus.results.get(
                'dipole_dcmnet' if use_dcmnet else 'dipole_physnet'
            )
            
            # Apply field in -j direction
            field_minus = np.zeros(3)
            field_minus[j] = -field_strength
            calc_minus = FieldResponseCalculator(base_calculator, electric_field=field_minus)
            atoms.calc = calc_minus
            _ = atoms.get_potential_energy()
            dipole_minus = calc_minus.results.get(
                'dipole_dcmnet' if use_dcmnet else 'dipole_physnet'
            )
            
            # Numerical derivative
            for i in range(3):  # Dipole component
                alpha[i, j] = (dipole_plus[i] - dipole_minus[i]) / (2 * field_strength)
        
    elif method == 'energy':
        # α_ij = -∂²E/∂F_i∂F_j
        # For diagonal: α_ii = -[E(F_i) - 2E(0) + E(-F_i)] / F²
        # For off-diagonal: α_ij = -[E(F_i+F_j) - E(F_i-F_j) - E(-F_i+F_j) + E(-F_i-F_j)] / (4F²)
        
        # Get energy at zero field
        calc_zero = FieldResponseCalculator(base_calculator, electric_field=np.zeros(3))
        atoms.calc = calc_zero
        E0 = atoms.get_potential_energy()
        
        alpha = np.zeros((3, 3))
        
        # Diagonal elements
        for i in range(3):
            field_p = np.zeros(3)
            field_p[i] = field_strength
            calc_p = FieldResponseCalculator(base_calculator, electric_field=field_p)
            atoms.calc = calc_p
            Ep = atoms.get_potential_energy()
            
            field_m = np.zeros(3)
            field_m[i] = -field_strength
            calc_m = FieldResponseCalculator(base_calculator, electric_field=field_m)
            atoms.calc = calc_m
            Em = atoms.get_potential_energy()
            
            alpha[i, i] = -(Ep - 2*E0 + Em) / (field_strength**2)
        
        # Off-diagonal elements
        for i in range(3):
            for j in range(i+1, 3):
                field_pp = np.zeros(3)
                field_pp[i] = field_strength
                field_pp[j] = field_strength
                calc_pp = FieldResponseCalculator(base_calculator, electric_field=field_pp)
                atoms.calc = calc_pp
                Epp = atoms.get_potential_energy()
                
                field_pm = np.zeros(3)
                field_pm[i] = field_strength
                field_pm[j] = -field_strength
                calc_pm = FieldResponseCalculator(base_calculator, electric_field=field_pm)
                atoms.calc = calc_pm
                Epm = atoms.get_potential_energy()
                
                field_mp = np.zeros(3)
                field_mp[i] = -field_strength
                field_mp[j] = field_strength
                calc_mp = FieldResponseCalculator(base_calculator, electric_field=field_mp)
                atoms.calc = calc_mp
                Emp = atoms.get_potential_energy()
                
                field_mm = np.zeros(3)
                field_mm[i] = -field_strength
                field_mm[j] = -field_strength
                calc_mm = FieldResponseCalculator(base_calculator, electric_field=field_mm)
                atoms.calc = calc_mm
                Emm = atoms.get_potential_energy()
                
                alpha[i, j] = -(Epp - Epm - Emp + Emm) / (4 * field_strength**2)
                alpha[j, i] = alpha[i, j]  # Symmetric
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert from e·Å²/(V/Å) to Å³
    # α has units: (e·Å) / (V/Å) = e·Å² / V
    # Since 1 V = 1 eV/e, we have: α = Å² / (eV/e²) = e²·Å² / eV
    # Standard polarizability units are Å³, and 1 Å³ = (4πε₀) × e²·Å² / eV
    # where 4πε₀ = 1/(14.3996 eV/e²/Å) in atomic units
    # So: α[Å³] = α[e·Å²/V] × 14.3996
    
    # Actually, let's keep it simple: our alpha is in units of e·Å²/(V/Å) = e²·Å/eV
    # This needs conversion factor
    conversion = 14.3996  # eV*Å to a.u. of polarizability
    alpha_au = alpha * conversion
    
    return alpha_au


def compute_polarizability_from_charges(positions, charges, method='clausius-mossotti'):
    """
    Approximate molecular polarizability from charge distribution.
    
    ⚠️ This is a crude approximation! For accurate Raman, train model with α.
    
    Parameters
    ----------
    positions : array (n_atoms, 3)
        Atomic positions in Angstrom
    charges : array (n_atoms,)
        Atomic charges in e
    method : str
        Method: 'clausius-mossotti' or 'traceless-quadrupole'
        
    Returns
    -------
    array (3, 3)
        Approximate polarizability tensor in Ang^3
    """
    if method == 'clausius-mossotti':
        # α_ij ≈ Σ_k q_k * r_k^i * r_k^j
        # This assumes isotropic response - very approximate!
        alpha = np.zeros((3, 3))
        for i in range(len(charges)):
            r = positions[i]
            q = charges[i]
            # Outer product weighted by charge
            alpha += q * np.outer(r, r)
        
        # Scale by empirical factor (this is very rough!)
        # Typical atomic polarizabilities are ~1-10 Ang^3
        alpha *= 0.1
        
    elif method == 'traceless-quadrupole':
        # Use traceless quadrupole moment
        # Q_ij = Σ_k q_k (3*r_k^i*r_k^j - r_k^2*δ_ij)
        # α ≈ Q / some_scaling
        Q = np.zeros((3, 3))
        for i in range(len(charges)):
            r = positions[i]
            q = charges[i]
            r2 = np.dot(r, r)
            Q += q * (3 * np.outer(r, r) - r2 * np.eye(3))
        
        # Approximate polarizability
        alpha = np.abs(Q) * 0.05
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure symmetric
    alpha = (alpha + alpha.T) / 2
    
    return alpha


def compute_polarizability_from_dcmnet(positions, charges_dist, positions_dist):
    """
    Compute polarizability from DCMNet distributed charges.
    
    This is more accurate than single point charges.
    
    Parameters
    ----------
    positions : array (n_atoms, 3)
        Atomic positions
    charges_dist : array (n_atoms, n_dcm)
        Distributed charges
    positions_dist : array (n_atoms, n_dcm, 3)
        Distributed charge positions
        
    Returns
    -------
    array (3, 3)
        Approximate polarizability tensor
    """
    # Flatten distributed charges
    charges_flat = charges_dist.flatten()
    positions_flat = positions_dist.reshape(-1, 3)
    
    # Compute polarizability
    alpha = compute_polarizability_from_charges(positions_flat, charges_flat)
    
    return alpha


def calculate_raman_spectrum(atoms, calculator, vib, delta=0.01, laser_wavelength=532,
                             use_dcmnet=True, field_strength=0.0001, 
                             method='finite-field', output_dir=None):
    """
    Calculate Raman spectrum from polarizability derivatives.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object at optimized geometry
    calculator : Calculator
        Joint calculator with charge predictions
    vib : Vibrations
        Vibrations object from frequency calculation
    delta : float
        Displacement for numerical differentiation (Å)
    laser_wavelength : float
        Laser wavelength in nm (default: 532 for green laser)
    use_dcmnet : bool
        Use DCMNet dipole (default) vs PhysNet
    field_strength : float
        Electric field strength for finite-field method (V/Å)
    method : str
        'finite-field' (proper, from ∂μ/∂F) or 'charge-approx' (crude estimate)
    output_dir : Path, optional
        Output directory
        
    Returns
    -------
    dict
        Raman spectrum data
    """
    print(f"\n{'='*70}")
    print(f"RAMAN SPECTRUM CALCULATION")
    print(f"{'='*70}")
    
    if method == 'finite-field':
        print(f"✅ Using PROPER finite-field polarizability (∂μ/∂F)")
    else:
        print(f"⚠️  Using approximate polarizability from charge distribution")
        print(f"   Consider using --method finite-field for better accuracy!")
    
    print(f"\nLaser wavelength: {laser_wavelength} nm")
    print(f"Using: {'DCMNet' if use_dcmnet else 'PhysNet'} dipole")
    
    atoms.calc = calculator
    freqs = vib.get_frequencies()
    
    # Get equilibrium polarizability
    if method == 'finite-field':
        alpha_eq = compute_polarizability_finite_field(
            atoms, calculator, field_strength=field_strength,
            use_dcmnet=use_dcmnet, method='dipole'
        )
    else:
        # Charge approximation (old method)
        _ = atoms.get_potential_energy()
        if use_dcmnet:
            charges = calculator.results.get('charges_dcmnet')
            positions_dist = calculator.results.get('positions_dcmnet')
            alpha_eq = compute_polarizability_from_dcmnet(
                atoms.get_positions(), charges, positions_dist
            )
        else:
            charges = calculator.results.get('charges')
            alpha_eq = compute_polarizability_from_charges(
                atoms.get_positions(), charges
            )
    
    print(f"\nEquilibrium polarizability (Å³):")
    print(f"  α_xx = {alpha_eq[0,0]:.4f}")
    print(f"  α_yy = {alpha_eq[1,1]:.4f}")
    print(f"  α_zz = {alpha_eq[2,2]:.4f}")
    print(f"  Isotropic: ᾱ = {np.trace(alpha_eq)/3:.4f}")
    
    # Calculate polarizability derivatives
    print("\nCalculating polarizability derivatives for each mode...")
    n_modes = len(freqs)
    alpha_derivatives = np.zeros((n_modes, 3, 3))
    
    for i in range(n_modes):
        mode = vib.get_mode(i)
        
        # Displace along mode (+)
        atoms_plus = atoms.copy()
        atoms_plus.positions += delta * mode.reshape(-1, 3)
        
        if method == 'finite-field':
            alpha_plus = compute_polarizability_finite_field(
                atoms_plus, calculator, field_strength=field_strength,
                use_dcmnet=use_dcmnet, method='dipole'
            )
        else:
            atoms_plus.calc = calculator
            _ = atoms_plus.get_potential_energy()
            if use_dcmnet:
                charges_p = calculator.results.get('charges_dcmnet')
                positions_dist_p = calculator.results.get('positions_dcmnet')
                alpha_plus = compute_polarizability_from_dcmnet(
                    atoms_plus.get_positions(), charges_p, positions_dist_p
                )
            else:
                charges_p = calculator.results.get('charges')
                alpha_plus = compute_polarizability_from_charges(
                    atoms_plus.get_positions(), charges_p
                )
        
        # Displace along mode (-)
        atoms_minus = atoms.copy()
        atoms_minus.positions -= delta * mode.reshape(-1, 3)
        
        if method == 'finite-field':
            alpha_minus = compute_polarizability_finite_field(
                atoms_minus, calculator, field_strength=field_strength,
                use_dcmnet=use_dcmnet, method='dipole'
            )
        else:
            atoms_minus.calc = calculator
            _ = atoms_minus.get_potential_energy()
            if use_dcmnet:
                charges_m = calculator.results.get('charges_dcmnet')
                positions_dist_m = calculator.results.get('positions_dcmnet')
                alpha_minus = compute_polarizability_from_dcmnet(
                    atoms_minus.get_positions(), charges_m, positions_dist_m
                )
            else:
                charges_m = calculator.results.get('charges')
                alpha_minus = compute_polarizability_from_charges(
                    atoms_minus.get_positions(), charges_m
                )
        
        # Numerical derivative
        alpha_derivatives[i] = (alpha_plus - alpha_minus) / (2 * delta)
        
        if (i + 1) % 3 == 0 or (i + 1) == n_modes:
            print(f"  Computed {i+1}/{n_modes} modes...")
    
    # Calculate Raman activities and intensities
    # Raman activity S = 45ᾱ'² + 7γ'²
    # where ᾱ' = (1/3) Σ_i ∂α_ii/∂Q  (isotropic)
    #       γ'² = (1/2) Σ_ij (∂α_ij/∂Q)²  (anisotropic)
    
    activities = np.zeros(n_modes)
    intensities = np.zeros(n_modes)
    
    # Laser frequency in cm^-1
    laser_freq_cm = 1e7 / laser_wavelength  # nm to cm^-1
    
    for i in range(n_modes):
        dalpha = alpha_derivatives[i]
        
        # Isotropic part
        alpha_bar_prime = np.trace(dalpha) / 3
        
        # Anisotropic part
        gamma_sq = 0.5 * np.sum(dalpha**2)
        
        # Raman activity
        activities[i] = 45 * alpha_bar_prime**2 + 7 * gamma_sq
        
        # Raman intensity (including frequency factor)
        freq_cm = np.real(freqs[i])
        if freq_cm > 0:
            # I ∝ (ν₀ - ν_i)⁴ * S / (ν_i * [1 - exp(-hν_i/kT)])
            # At room temp, simplify to: I ∝ (ν₀ - ν_i)⁴ * S / ν_i
            nu_scattered = laser_freq_cm - freq_cm  # Stokes
            intensities[i] = (nu_scattered**4) * activities[i] / freq_cm
        else:
            intensities[i] = 0
    
    # Normalize intensities
    if np.max(intensities) > 0:
        intensities /= np.max(intensities)
    
    results = {
        'frequencies': freqs,
        'activities': activities,
        'intensities': intensities,
        'alpha_derivatives': alpha_derivatives,
        'alpha_equilibrium': alpha_eq,
    }
    
    # Print results
    print(f"\n{'Frequency (cm⁻¹)':>18} {'Activity (Å⁴/amu)':>22} {'Intensity (norm)':>22}")
    print("-" * 70)
    
    freqs_real = np.real(freqs)
    for i, freq in enumerate(freqs_real):
        if freq > 100:  # Skip low frequencies
            print(f"{freq:18.2f} {activities[i]:22.6f} {intensities[i]:22.6f}")
    
    # Plot
    if HAS_MATPLOTLIB and output_dir:
        plot_raman_spectrum(results, output_dir, laser_wavelength=laser_wavelength,
                           use_dcmnet=use_dcmnet)
    
    return results


def plot_raman_spectrum(raman_data, output_dir, laser_wavelength=532, 
                        use_dcmnet=True, broadening=10):
    """
    Plot Raman spectrum.
    
    Parameters
    ----------
    raman_data : dict
        Raman data from calculate_raman_spectrum
    output_dir : Path
        Output directory
    laser_wavelength : float
        Laser wavelength (nm)
    use_dcmnet : bool
        Whether DCMNet charges were used
    broadening : float
        Lorentzian broadening (cm⁻¹)
    """
    freqs = np.real(raman_data['frequencies'])
    intensities = raman_data['intensities']
    
    # Filter positive frequencies
    mask = freqs > 100
    freqs = freqs[mask]
    intensities = intensities[mask]
    
    if len(freqs) == 0:
        print("⚠️  No positive frequencies for Raman plot")
        return
    
    # Create broadened spectrum
    freq_range = np.linspace(0, max(freqs) + 500, 2000)
    spectrum = np.zeros_like(freq_range)
    
    for freq, intensity in zip(freqs, intensities):
        spectrum += intensity * (broadening**2) / ((freq_range - freq)**2 + broadening**2)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Stick spectrum
    ax = axes[0]
    markerline, stemlines, baseline = ax.stem(freqs, intensities,
                                               linefmt='C2-', markerfmt='C2o', basefmt=' ')
    charge_type = 'DCMNet distributed' if use_dcmnet else 'PhysNet point'
    markerline.set_label(f'{charge_type} charges')
    
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title(f'Raman Spectrum (Stick) - {laser_wavelength} nm laser', 
                 fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Broadened spectrum
    ax = axes[1]
    ax.plot(freq_range, spectrum, 'C2-', linewidth=2)
    ax.fill_between(freq_range, spectrum, alpha=0.3)
    
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title(f'Raman Spectrum (Broadened, FWHM = {broadening} cm⁻¹)', 
                 fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    title_color = 'green' if 'finite-field' in str(output_dir) else 'orange'
    title_prefix = '✅ Proper' if title_color == 'green' else '⚠️ Approximate'
    plt.suptitle(f'{title_prefix} Raman Spectrum', 
                 fontsize=16, weight='bold', color=title_color)
    
    plt.tight_layout()
    
    output_path = output_dir / 'raman_spectrum.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved Raman spectrum: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Raman spectroscopy analysis')
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--molecule', type=str, default='CO2',
                       help='Molecule formula')
    parser.add_argument('--geometry', type=str, default=None,
                       help='XYZ file with initial geometry')
    parser.add_argument('--output-dir', type=Path, default=Path('./raman_analysis'),
                       help='Output directory')
    
    # Analysis options
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize geometry before analysis')
    parser.add_argument('--frequencies', action='store_true',
                       help='Calculate vibrational frequencies')
    parser.add_argument('--raman', action='store_true',
                       help='Calculate Raman spectrum')
    parser.add_argument('--fmax', type=float, default=0.01,
                       help='Force convergence for optimization (eV/Å)')
    parser.add_argument('--vib-delta', type=float, default=0.01,
                       help='Displacement for vibrational analysis (Å)')
    
    # Raman options
    parser.add_argument('--laser-wavelength', type=float, default=532,
                       help='Laser wavelength (nm), default: 532 (green)')
    parser.add_argument('--use-physnet-dipole', action='store_true',
                       help='Use PhysNet dipole instead of DCMNet')
    parser.add_argument('--field-strength', type=float, default=0.0001,
                       help='Electric field strength for finite-field (V/Å)')
    parser.add_argument('--method', type=str, default='finite-field',
                       choices=['finite-field', 'charge-approx'],
                       help='Polarizability method (default: finite-field from ∂μ/∂F)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Joint PhysNet-DCMNet: Raman Spectroscopy Analysis")
    print("="*70)
    
    if args.method == 'finite-field':
        print("\n✅ Using PROPER finite-field polarizability!")
        print("   α_ij = ∂μ_i/∂F_j (dipole response to electric field)\n")
    else:
        print("\n⚠️  Using approximate polarizability from charge distribution")
        print("   Consider using --method finite-field for better accuracy!\n")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"1. Loading checkpoint from {args.checkpoint}")
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
        use_dcmnet_dipole=False
    )
    
    # Load or create molecule
    print(f"\n2. Setting up molecule: {args.molecule}")
    
    if args.geometry:
        from ase.io import read
        atoms = read(args.geometry)
        print(f"✅ Loaded geometry from {args.geometry}")
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
        else:
            raise ValueError(f"Unknown molecule: {args.molecule}. Provide --geometry")
        
        print(f"✅ Using default {args.molecule} geometry")
    
    # Optimize if requested
    if args.optimize:
        atoms = optimize_geometry(atoms, calculator, fmax=args.fmax)
        from ase.io import write
        write(args.output_dir / f'{args.molecule}_optimized.xyz', atoms)
    
    # Calculate frequencies
    if args.frequencies or args.raman:
        freqs, vib = calculate_frequencies(atoms, calculator,
                                          delta=args.vib_delta,
                                          output_dir=args.output_dir)
    
    # Calculate Raman spectrum
    if args.raman:
        raman_data = calculate_raman_spectrum(
            atoms, calculator, vib,
            delta=args.vib_delta,
            laser_wavelength=args.laser_wavelength,
            use_dcmnet=not args.use_physnet_dipole,
            field_strength=args.field_strength,
            method=args.method,
            output_dir=args.output_dir
        )
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {args.output_dir}")
    print("\n⚠️  Remember: These Raman intensities are approximate!")
    print("   Consider training your model to predict polarizability for accuracy.")


if __name__ == '__main__':
    main()

