#!/usr/bin/env python3
"""
Example script demonstrating DCMNet ASE calculator usage.

This script shows how to:
1. Load a trained DCMNet model
2. Create an ASE calculator
3. Compute electrostatic potential at grid points
4. Visualize ESP on a molecular surface

Usage:
    python example_dcmnet_calculator.py <checkpoint.pkl> [--molecule CO2]
"""

import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from ase import Atoms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mmml.dcmnet.dcmnet_ase import DCMNetCalculator


def create_co2():
    """Create a CO2 molecule."""
    return Atoms('CO2', positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]])


def create_water():
    """Create a water molecule."""
    return Atoms('H2O', positions=[
        [0, 0, 0],
        [0.757, 0.587, 0],
        [-0.757, 0.587, 0]
    ])


def create_grid_around_molecule(atoms, spacing=0.5, padding=3.0):
    """
    Create a 3D grid around a molecule.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Molecular structure
    spacing : float
        Grid spacing in Angstroms
    padding : float
        Padding around molecule in Angstroms
        
    Returns
    -------
    array_like
        Grid points, shape (n_points, 3)
    """
    positions = atoms.get_positions()
    min_coords = positions.min(axis=0) - padding
    max_coords = positions.max(axis=0) + padding
    
    x = np.arange(min_coords[0], max_coords[0] + spacing, spacing)
    y = np.arange(min_coords[1], max_coords[1] + spacing, spacing)
    z = np.arange(min_coords[2], max_coords[2] + spacing, spacing)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    
    return grid_points


def create_vdw_surface_grid(atoms, density=1.0, probe_radius=1.4):
    """
    Create a VDW surface grid (simplified version).
    
    This is a simplified implementation. For production use, consider
    using more sophisticated surface generation methods.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Molecular structure
    density : float
        Grid point density (points per Angstrom^2)
    probe_radius : float
        Probe radius for VDW surface
        
    Returns
    -------
    array_like
        VDW surface grid points, shape (n_points, 3)
    """
    from ase.data import vdw_radii
    
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    
    # Get VDW radii
    radii = np.array([vdw_radii[z] for z in atomic_numbers])
    
    # Create points on spheres around each atom
    surface_points = []
    for pos, radius in zip(positions, radii):
        # Surface radius = atom radius + probe radius
        surface_radius = radius + probe_radius
        
        # Generate points on sphere (simplified - uniform sampling)
        n_points = int(4 * np.pi * surface_radius**2 * density)
        if n_points < 10:
            n_points = 10
        
        # Use spherical coordinates
        theta = np.linspace(0, np.pi, int(np.sqrt(n_points)))
        phi = np.linspace(0, 2*np.pi, int(np.sqrt(n_points)))
        
        for t in theta:
            for p in phi:
                point = pos + surface_radius * np.array([
                    np.sin(t) * np.cos(p),
                    np.sin(t) * np.sin(p),
                    np.cos(t)
                ])
                surface_points.append(point)
    
    return np.array(surface_points)


def main():
    parser = argparse.ArgumentParser(
        description="Example: DCMNet ASE Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with CO2
    python example_dcmnet_calculator.py checkpoint.pkl
    
    # Use water molecule
    python example_dcmnet_calculator.py checkpoint.pkl --molecule H2O
    
    # Create ESP on VDW surface
    python example_dcmnet_calculator.py checkpoint.pkl --vdw-surface
        """
    )
    
    parser.add_argument(
        'checkpoint',
        type=Path,
        help='Path to checkpoint file (.pkl) containing model and params'
    )
    parser.add_argument(
        '--molecule',
        type=str,
        default='CO2',
        choices=['CO2', 'H2O'],
        help='Molecule to use for testing (default: CO2)'
    )
    parser.add_argument(
        '--vdw-surface',
        action='store_true',
        help='Compute ESP on VDW surface instead of regular grid'
    )
    parser.add_argument(
        '--grid-spacing',
        type=float,
        default=0.5,
        help='Grid spacing in Angstroms (default: 0.5)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file to save ESP data (NPZ format)'
    )
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    with open(args.checkpoint, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract model and params
    model = checkpoint.get('model')
    params = checkpoint.get('params') or checkpoint.get('best_params')
    
    if model is None or params is None:
        raise ValueError(
            f"Checkpoint must contain 'model' and 'params' keys. "
            f"Found keys: {list(checkpoint.keys())}"
        )
    
    # Get model parameters
    n_dcm = getattr(model, 'n_dcm', checkpoint.get('n_dcm', 6))
    cutoff = getattr(model, 'cutoff', checkpoint.get('cutoff', 10.0))
    
    print(f"Model parameters:")
    print(f"  n_dcm: {n_dcm}")
    print(f"  cutoff: {cutoff} Å")
    
    # Create calculator
    print("\nCreating DCMNet calculator...")
    calc = DCMNetCalculator(
        model=model,
        params=params,
        cutoff=cutoff,
        n_dcm=n_dcm
    )
    
    # Create molecule
    if args.molecule == 'CO2':
        atoms = create_co2()
    elif args.molecule == 'H2O':
        atoms = create_water()
    else:
        raise ValueError(f"Unknown molecule: {args.molecule}")
    
    print(f"\nMolecule: {args.molecule}")
    print(f"  Formula: {atoms.get_chemical_formula()}")
    print(f"  Number of atoms: {len(atoms)}")
    
    # Set calculator
    atoms.calc = calc
    
    # Calculate properties
    print("\nComputing distributed multipoles...")
    charges = atoms.get_charges()
    dipole = atoms.get_dipole_moment()
    
    print(f"\nResults:")
    print(f"  Atomic charges: {charges}")
    print(f"  Total charge: {charges.sum():.6f} e")
    print(f"  Molecular dipole: {dipole} Debye")
    print(f"  Dipole magnitude: {np.linalg.norm(dipole):.6f} Debye")
    
    # Get distributed multipoles
    multipoles = calc.get_distributed_multipoles()
    print(f"\nDistributed multipoles:")
    print(f"  Monopoles shape: {multipoles['monopoles'].shape}")
    print(f"  Dipole positions shape: {multipoles['dipole_positions'].shape}")
    print(f"  Monopole range: [{multipoles['monopoles'].min():.6f}, {multipoles['monopoles'].max():.6f}] e")
    
    # Create grid
    if args.vdw_surface:
        print("\nCreating VDW surface grid...")
        grid_points = create_vdw_surface_grid(atoms)
    else:
        print(f"\nCreating regular grid (spacing={args.grid_spacing} Å)...")
        grid_points = create_grid_around_molecule(atoms, spacing=args.grid_spacing)
    
    print(f"  Grid points: {len(grid_points)}")
    
    # Compute ESP
    print("\nComputing electrostatic potential...")
    esp = calc.get_electrostatic_potential(grid_points)
    
    print(f"\nESP statistics:")
    print(f"  Mean: {esp.mean():.6f} Ha/e ({esp.mean()*627.5:.3f} kcal/mol/e)")
    print(f"  Std: {esp.std():.6f} Ha/e ({esp.std()*627.5:.3f} kcal/mol/e)")
    print(f"  Min: {esp.min():.6f} Ha/e ({esp.min()*627.5:.3f} kcal/mol/e)")
    print(f"  Max: {esp.max():.6f} Ha/e ({esp.max()*627.5:.3f} kcal/mol/e)")
    
    # Save results if requested
    if args.output:
        print(f"\nSaving results to {args.output}...")
        np.savez(
            args.output,
            grid_points=grid_points,
            esp=esp,
            charges=charges,
            dipole=dipole,
            monopoles=multipoles['monopoles'],
            dipole_positions=multipoles['dipole_positions'],
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
        )
        print(f"✓ Saved to {args.output}")
    
    # Show sample ESP values
    print("\nSample ESP values at grid points:")
    n_show = min(10, len(grid_points))
    indices = np.linspace(0, len(grid_points)-1, n_show, dtype=int)
    for i, idx in enumerate(indices):
        point = grid_points[idx]
        esp_val = esp[idx]
        print(f"  Point {i+1}: [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}] Å -> "
              f"{esp_val:.6f} Ha/e ({esp_val*627.5:.3f} kcal/mol/e)")
    
    print("\n✅ All calculations completed successfully!")


if __name__ == '__main__':
    main()

