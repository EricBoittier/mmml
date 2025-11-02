#!/usr/bin/env python3
"""
Convert Molpro Outputs to Training Format

Parses Molpro output files and cube files to create NPZ training data.

Usage:
    # Convert single run
    python convert_molpro_to_training.py \
        --molpro-outputs ./logs/*.out \
        --cube-dir ./cubes \
        --output ./qm_training_data.npz
    
    # Merge with existing training data
    python convert_molpro_to_training.py \
        --molpro-outputs ./logs/*.out \
        --cube-dir ./cubes \
        --merge-with ../physnet_train_charges/energies_forces_dipoles_train.npz \
        --output ../physnet_train_charges/energies_forces_dipoles_train_v2.npz
"""

import sys
from pathlib import Path
import numpy as np
import argparse
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    from ase import Atoms
    from ase.io.cube import read_cube
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("⚠️  ASE not available, ESP parsing disabled")


def parse_molpro_geometry(output_text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract geometry from Molpro output.
    
    Returns
    -------
    atomic_numbers : np.ndarray
        Atomic numbers
    positions : np.ndarray
        Positions in Angstroms
    """
    # Look for final geometry section
    # Example:
    #  ATOMIC COORDINATES
    #  NR  ATOM    CHARGE       X              Y              Z
    #   1  C       6.00    0.000000000    0.000000000    0.000000000
    #   2  O1      8.00    0.000000000    0.000000000    2.192076946
    
    pattern = r'ATOMIC\s+COORDINATES.*?\n\s+NR\s+ATOM.*?\n(.*?)(?=\n\s*\n|\Z)'
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        raise ValueError("Could not find geometry in Molpro output")
    
    geom_text = match.group(1)
    
    atomic_numbers = []
    positions = []
    
    for line in geom_text.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 6:
            atom_symbol = parts[1].rstrip('0123456789')  # Remove numbers from O1, O2
            charge = float(parts[2])
            x = float(parts[3])
            y = float(parts[4])
            z = float(parts[5])
            
            # Convert Bohr to Angstrom (Molpro uses Bohr in this section)
            bohr_to_ang = 0.529177
            positions.append([x * bohr_to_ang, y * bohr_to_ang, z * bohr_to_ang])
            atomic_numbers.append(int(charge))
    
    return np.array(atomic_numbers, dtype=np.int32), np.array(positions)


def parse_molpro_energy(output_text: str) -> float:
    """Extract MP2 energy from Molpro output."""
    # Look for: !MP2 total energy                   -187.721234567
    pattern = r'!MP2\s+total\s+energy\s+([-+]?\d+\.\d+)'
    match = re.search(pattern, output_text)
    
    if match:
        return float(match.group(1))
    
    # Try alternative format
    pattern = r'MP2\s+energy\s*=\s*([-+]?\d+\.\d+)'
    match = re.search(pattern, output_text)
    
    if match:
        return float(match.group(1))
    
    raise ValueError("Could not find MP2 energy in output")


def parse_molpro_forces(output_text: str) -> np.ndarray:
    """
    Extract Cartesian forces from Molpro output.
    
    Molpro prints gradients (not forces), so we need to negate them.
    """
    # Look for gradient section
    # Example:
    #  Atom          dE/dx               dE/dy               dE/dz
    #    1      0.000000000000      0.000000000000      0.012345678
    #    2      0.000000000000      0.000000000000     -0.006172839
    
    pattern = r'Atom\s+dE/dx\s+dE/dy\s+dE/dz\s*\n(.*?)(?=\n\s*\n|\Z)'
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        raise ValueError("Could not find forces/gradients in output")
    
    grad_text = match.group(1)
    
    gradients = []
    for line in grad_text.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 4:
            grad_x = float(parts[1])
            grad_y = float(parts[2])
            grad_z = float(parts[3])
            gradients.append([grad_x, grad_y, grad_z])
    
    gradients = np.array(gradients)
    
    # Convert gradient to force (F = -dE/dR)
    forces = -gradients
    
    # Convert from Hartree/Bohr to eV/Angstrom
    hartree_bohr_to_ev_ang = 51.4220652  # Ha/Bohr -> eV/Å
    forces *= hartree_bohr_to_ev_ang
    
    return forces


def parse_molpro_dipole(output_text: str) -> np.ndarray:
    """
    Extract dipole moment from Molpro output.
    
    Returns dipole in Debye.
    """
    # Look for dipole moment section
    # Example:
    #  Dipole moment /Debye                   0.00000000     0.00000000     0.00000000
    
    pattern = r'Dipole\s+moment\s+/Debye\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)'
    match = re.search(pattern, output_text)
    
    if match:
        dx = float(match.group(1))
        dy = float(match.group(2))
        dz = float(match.group(3))
        return np.array([dx, dy, dz])
    
    # Try alternative format
    pattern = r'DIPOLE\s+MOMENT.*?\n.*?([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)'
    match = re.search(pattern, output_text, re.DOTALL)
    
    if match:
        dx = float(match.group(1))
        dy = float(match.group(2))
        dz = float(match.group(3))
        return np.array([dx, dy, dz])
    
    raise ValueError("Could not find dipole moment in output")


def parse_esp_cube(cube_file: Path, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse ESP cube file.
    
    Returns
    -------
    grid_points : np.ndarray
        ESP grid points in Angstrom (n_grid, 3)
    esp_values : np.ndarray
        ESP values in Hartree/e (n_grid,)
    """
    if not HAS_ASE:
        raise ImportError("ASE required for cube parsing")
    
    with open(cube_file, 'r') as f:
        data = read_cube(f)
    
    # data is a dict with keys: 'data', 'atoms'
    esp_data = data['data']  # 3D array
    atoms = data['atoms']
    
    # Get grid parameters from cube file metadata
    # Cube file format:
    # Line 3: natoms origin_x origin_y origin_z
    # Line 4: nx voxel_x 0 0
    # Line 5: ny 0 voxel_y 0
    # Line 6: nz 0 0 voxel_z
    
    with open(cube_file, 'r') as f:
        lines = f.readlines()
    
    # Parse origin and voxel vectors
    origin_line = lines[2].split()
    natoms = int(origin_line[0])
    origin = np.array([float(origin_line[i]) for i in range(1, 4)]) * 0.529177  # Bohr to Å
    
    nx_line = lines[3].split()
    ny_line = lines[4].split()
    nz_line = lines[5].split()
    
    nx = int(nx_line[0])
    ny = int(ny_line[0])
    nz = int(nz_line[0])
    
    voxel_x = np.array([float(nx_line[i]) for i in range(1, 4)]) * 0.529177
    voxel_y = np.array([float(ny_line[i]) for i in range(1, 4)]) * 0.529177
    voxel_z = np.array([float(nz_line[i]) for i in range(1, 4)]) * 0.529177
    
    # Generate grid points
    grid_points = []
    esp_values = []
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                point = origin + ix * voxel_x + iy * voxel_y + iz * voxel_z
                value = esp_data[ix, iy, iz]
                
                grid_points.append(point)
                esp_values.append(value)
    
    return np.array(grid_points), np.array(esp_values)


def process_molpro_output(output_file: Path, cube_dir: Optional[Path] = None) -> Dict:
    """Process a single Molpro output file."""
    print(f"  Processing {output_file.name}...")
    
    with open(output_file, 'r') as f:
        output_text = f.read()
    
    # Extract basic data
    atomic_numbers, positions = parse_molpro_geometry(output_text)
    energy_hartree = parse_molpro_energy(output_text)
    forces = parse_molpro_forces(output_text)
    dipole_debye = parse_molpro_dipole(output_text)
    
    # Convert energy to eV
    hartree_to_ev = 27.2114
    energy = energy_hartree * hartree_to_ev
    
    # Convert dipole to e·Å
    debye_to_e_ang = 0.2081943  # D to e·Å
    dipole = dipole_debye * debye_to_e_ang
    
    result = {
        'atomic_numbers': atomic_numbers,
        'positions': positions,
        'energy': energy,
        'forces': forces,
        'dipole': dipole,
    }
    
    # Parse ESP cube if available
    if cube_dir:
        # Try to find corresponding ESP cube
        # Expected naming: esp_r1_XXX_r2_YYY_ang_ZZZ.cube
        stem = output_file.stem.replace('co2_', '').replace('.molpro', '')
        esp_cube = cube_dir / 'esp' / f'esp_{stem}.cube'
        
        if esp_cube.exists():
            try:
                grid_points, esp_values = parse_esp_cube(esp_cube, positions)
                result['vdw_surface'] = grid_points
                result['esp'] = esp_values
            except Exception as e:
                print(f"    ⚠️  Failed to parse ESP cube: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Convert Molpro outputs to training format')
    
    parser.add_argument('--molpro-outputs', type=str, nargs='+', required=True,
                       help='Molpro output files (glob patterns supported)')
    parser.add_argument('--cube-dir', type=Path, default=None,
                       help='Directory with cube files (expects cubes/esp/ subdirectory)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output NPZ file')
    parser.add_argument('--merge-with', type=Path, default=None,
                       help='Existing training data to merge with')
    parser.add_argument('--skip-errors', action='store_true',
                       help='Skip files that fail to parse')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MOLPRO OUTPUT CONVERTER")
    print("="*70)
    
    # Find all output files
    output_files = []
    for pattern in args.molpro_outputs:
        output_files.extend(Path('.').glob(pattern))
    
    output_files = sorted(set(output_files))  # Remove duplicates
    print(f"\nFound {len(output_files)} Molpro output files")
    
    if not output_files:
        print("No files found. Exiting.")
        return
    
    # Process all files
    print(f"\nProcessing outputs...")
    results = []
    errors = []
    
    for output_file in output_files:
        try:
            result = process_molpro_output(output_file, args.cube_dir)
            results.append(result)
        except Exception as e:
            error_msg = f"{output_file.name}: {e}"
            errors.append(error_msg)
            if args.skip_errors:
                print(f"    ⚠️  {error_msg}")
            else:
                raise RuntimeError(f"Failed to process {output_file.name}: {e}")
    
    print(f"\n✅ Successfully processed {len(results)} files")
    if errors:
        print(f"⚠️  Skipped {len(errors)} files with errors")
    
    # Convert to NPZ format
    print(f"\nConverting to training format...")
    
    # Check if all have same atom types
    n_atoms = len(results[0]['atomic_numbers'])
    all_same_atoms = all(len(r['atomic_numbers']) == n_atoms for r in results)
    
    if not all_same_atoms:
        print("⚠️  WARNING: Not all molecules have the same number of atoms!")
        print("    This may cause issues during training.")
    
    # Determine if we have ESP data
    has_esp = 'vdw_surface' in results[0]
    
    # For energy/forces/dipole (physnet format)
    R = np.array([r['positions'] for r in results])
    Z = np.array([r['atomic_numbers'] for r in results])
    E = np.array([r['energy'] for r in results])
    F = np.array([r['forces'] for r in results])
    D = np.array([r['dipole'] for r in results])
    
    efd_data = {
        'R': R,
        'Z': Z,
        'E': E,
        'F': F,
        'D': D,
    }
    
    # For ESP data (dcmnet format)
    if has_esp:
        vdw_surfaces = []
        esp_values = []
        
        for r in results:
            if 'vdw_surface' in r:
                vdw_surfaces.append(r['vdw_surface'])
                esp_values.append(r['esp'])
            else:
                # Fill with zeros if ESP missing for this structure
                vdw_surfaces.append(np.zeros((100, 3)))  # Placeholder
                esp_values.append(np.zeros(100))
        
        # Find max grid size for padding
        max_grid_size = max(len(v) for v in vdw_surfaces)
        
        # Pad to same size
        vdw_padded = []
        esp_padded = []
        for vdw, esp in zip(vdw_surfaces, esp_values):
            if len(vdw) < max_grid_size:
                pad_size = max_grid_size - len(vdw)
                vdw = np.vstack([vdw, np.zeros((pad_size, 3))])
                esp = np.concatenate([esp, np.zeros(pad_size)])
            vdw_padded.append(vdw)
            esp_padded.append(esp)
        
        esp_data = {
            'vdw_surface': np.array(vdw_padded),
            'esp': np.array(esp_padded),
        }
    
    # Merge with existing data if requested
    if args.merge_with:
        print(f"\nMerging with existing data from {args.merge_with}...")
        existing = np.load(args.merge_with)
        
        # Merge E/F/D data
        efd_data = {
            'R': np.vstack([existing['R'], R]),
            'Z': np.vstack([existing['Z'], Z]),
            'E': np.concatenate([existing['E'], E]),
            'F': np.vstack([existing['F'], F]),
            'D': np.vstack([existing['D'], D]),
        }
        
        print(f"  Original: {len(existing['E'])} structures")
        print(f"  New: {len(E)} structures")
        print(f"  Total: {len(efd_data['E'])} structures")
    
    # Save
    print(f"\nSaving to {args.output}...")
    np.savez(args.output, **efd_data)
    
    if has_esp and not args.merge_with:
        esp_output = args.output.parent / (args.output.stem + '_esp.npz')
        print(f"Saving ESP data to {esp_output}...")
        np.savez(esp_output, **esp_data)
    
    print(f"\n{'='*70}")
    print("✅ CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  Energy/Forces/Dipole: {args.output}")
    if has_esp and not args.merge_with:
        print(f"  ESP data: {esp_output}")
    
    print(f"\nStatistics:")
    print(f"  Structures: {len(results)}")
    print(f"  Atoms/structure: {n_atoms}")
    print(f"  Energy range: [{E.min():.2f}, {E.max():.2f}] eV")
    print(f"  Force range: [{F.min():.2f}, {F.max():.2f}] eV/Å")
    print(f"  Dipole range: [{np.linalg.norm(D, axis=1).min():.3f}, {np.linalg.norm(D, axis=1).max():.3f}] e·Å")
    
    if has_esp:
        print(f"  ESP grid points/structure: {len(vdw_surfaces[0])}")
        esp_flat = np.concatenate([e for e in esp_values])
        print(f"  ESP range: [{esp_flat.min():.4f}, {esp_flat.max():.4f}] Ha/e")


if __name__ == '__main__':
    main()

