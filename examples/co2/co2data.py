"""
CO2 Dataset Exploration and Visualization

This script loads and visualizes energy, force, dipole, and ESP data for CO2 molecules.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path so we can import mmml
repo_root = Path(__file__).parent / "../.."
sys.path.insert(0, str(repo_root.resolve()))

from mmml.plotting import esp, distributions


def plot_force_distributions(forces):
    """
    Plot distributions of force components and norms.
    
    Parameters
    ----------
    forces : array_like
        Force array of shape (n_samples, n_atoms, 3)
    """
    n_samples, n_atoms, _ = forces.shape
    
    # Flatten to get all force components
    forces_flat = forces.reshape(-1, 3)
    fx, fy, fz = forces_flat[:, 0], forces_flat[:, 1], forces_flat[:, 2]
    force_norms = np.linalg.norm(forces_flat, axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Force Distributions', fontsize=16, fontweight='bold')
    
    # Force components
    axes[0, 0].hist(fx, bins=50, alpha=0.7, edgecolor='black', color='red')
    axes[0, 0].set_xlabel('Force X (Hartree/Bohr)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'X Component (μ={np.mean(fx):.3e}, σ={np.std(fx):.3e})')
    axes[0, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
    
    axes[0, 1].hist(fy, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[0, 1].set_xlabel('Force Y (Hartree/Bohr)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Y Component (μ={np.mean(fy):.3e}, σ={np.std(fy):.3e})')
    axes[0, 1].axvline(0, color='black', linestyle='--', alpha=0.5)
    
    axes[1, 0].hist(fz, bins=50, alpha=0.7, edgecolor='black', color='blue')
    axes[1, 0].set_xlabel('Force Z (Hartree/Bohr)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Z Component (μ={np.mean(fz):.3e}, σ={np.std(fz):.3e})')
    axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
    
    axes[1, 1].hist(force_norms, bins=50, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 1].set_xlabel('Force Norm (Hartree/Bohr)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Norms (μ={np.mean(force_norms):.3e}, σ={np.std(force_norms):.3e})')
    
    plt.tight_layout()
    return fig


def check_coordinate_units(positions, atomic_numbers, sample_idx=0):
    """
    Check what units the atomic coordinates are in by calculating bond lengths.
    
    For CO2:
    - C-O bond length is typically ~1.16 Angstrom or ~2.19 Bohr
    - O-O distance is typically ~2.32 Angstrom or ~4.38 Bohr
    
    Parameters
    ----------
    positions : array_like
        Atomic positions
    atomic_numbers : array_like
        Atomic numbers
    sample_idx : int
        Sample index to check
    """
    pos = positions[sample_idx]
    z_nums = atomic_numbers[sample_idx]
    
    # Remove dummy atoms
    valid_mask = z_nums > 0
    valid_pos = pos[valid_mask]
    valid_z = z_nums[valid_mask]
    
    print(f"\n{'='*60}")
    print(f"COORDINATE UNITS CHECK (Sample {sample_idx})")
    print(f"{'='*60}")
    print(f"Number of atoms: {len(valid_pos)}")
    print(f"Atomic numbers: {valid_z}")
    print("\nBond lengths:")
    
    # Calculate all pairwise distances
    for i in range(len(valid_pos)):
        for j in range(i+1, len(valid_pos)):
            dist = np.linalg.norm(valid_pos[i] - valid_pos[j])
            z1, z2 = int(valid_z[i]), int(valid_z[j])
            
            # Get element symbols
            elements = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
            elem1 = elements.get(z1, f'Z{z1}')
            elem2 = elements.get(z2, f'Z{z2}')
            
            print(f"  {elem1}-{elem2} ({i}-{j}): {dist:.4f}")
    
    # Expected values for CO2
    print("\nExpected bond lengths for CO2:")
    print("  C-O bond:")
    print("    ~1.16 Angstrom")
    print("    ~2.19 Bohr")
    print("  O-O distance:")
    print("    ~2.32 Angstrom")
    print("    ~4.38 Bohr")
    
    # Make educated guess
    co_bonds = []
    for i in range(len(valid_pos)):
        for j in range(i+1, len(valid_pos)):
            z1, z2 = int(valid_z[i]), int(valid_z[j])
            if (z1 == 6 and z2 == 8) or (z1 == 8 and z2 == 6):  # C-O bond
                dist = np.linalg.norm(valid_pos[i] - valid_pos[j])
                co_bonds.append(dist)
    
    if co_bonds:
        avg_co = np.mean(co_bonds)
        print(f"\nAverage C-O distance: {avg_co:.4f}")
        
        # Check units
        if 1.10 < avg_co < 1.25:
            print(f"→ Positions appear to be in ANGSTROM units")
            print(f"   (Expected C-O ~1.16 Å, measured {avg_co:.4f})")
        elif 2.0 < avg_co < 2.5:
            print(f"→ Positions appear to be in BOHR units")
            print(f"   (Expected C-O ~2.19 Bohr, measured {avg_co:.4f})")
        elif 0.95 < avg_co < 1.05:
            print(f"→ WARNING: Positions appear to be NORMALIZED/SCALED")
            print(f"   (Measured C-O = {avg_co:.4f}, expected ~1.16 Å or ~2.19 Bohr)")
            print(f"   These may be relative coordinates or a different unit system!")
        else:
            print(f"→ UNCLEAR units (C-O distance = {avg_co:.4f})")
            print(f"   Does not match standard Angstrom or Bohr values")
    
    # Show coordinate ranges
    print(f"\nCoordinate ranges:")
    print(f"  X: [{np.min(valid_pos[:, 0]):.4f}, {np.max(valid_pos[:, 0]):.4f}]")
    print(f"  Y: [{np.min(valid_pos[:, 1]):.4f}, {np.max(valid_pos[:, 1]):.4f}]")
    print(f"  Z: [{np.min(valid_pos[:, 2]):.4f}, {np.max(valid_pos[:, 2]):.4f}]")
    
    print(f"{'='*60}\n")


def plot_esp_3d_with_molecule(grids, esp_values, positions, atomic_numbers, sample_idx=0):
    """
    Plot 3D scatter of ESP grid points with molecular structure overlay using ASE.
    
    Parameters
    ----------
    grids : array_like
        Grid coordinates of shape (n_samples, n_points, 3)
    esp_values : array_like
        ESP values of shape (n_samples, n_points)
    positions : array_like
        Atomic positions of shape (n_samples, n_atoms, 3)
    atomic_numbers : array_like
        Atomic numbers of shape (n_samples, n_atoms)
    sample_idx : int
        Index of the sample to visualize
    """
    from mpl_toolkits.mplot3d import Axes3D
    try:
        from ase.data import covalent_radii
    except ImportError:
        print("ASE not available, creating plot without molecule overlay")
        return plot_esp_3d_scatter(grids, esp_values, sample_idx)
    
    coords = grids[sample_idx]
    esp = esp_values[sample_idx]
    
    # Get molecule structure
    pos = positions[sample_idx]
    z_nums = atomic_numbers[sample_idx]
    
    # Remove dummy atoms (Z=0)
    valid_mask = z_nums > 0
    valid_pos = pos[valid_mask]
    valid_z = z_nums[valid_mask]
    
    # Positions are NORMALIZED (C-O bond = 1.0 instead of ~1.16 Å)
    # The ESP grid coordinates are subsampled from a cube grid, so the grid center
    # isn't meaningful. Instead, we should place the molecule where it actually is.
    # 
    # Based on analysis: normalized coords have C-O = 1.0, real C-O ~= 1.16 Å
    # So: real_position = normalized_position * 1.16
    #
    # But this doesn't tell us WHERE in the grid the molecule sits.
    # Looking at the data: grid is 0-49 Å, molecule is tiny (~1 Å scale)
    # The molecule must be positioned based on some other information not in these files
    #
    # Best guess: place molecule near the grid origin (bottom corner)
    # OR: assume molecule is centered and the grid was built around it
    
    CO_BOND_LENGTH_ANGSTROM = 1.16
    valid_pos_angstrom = valid_pos * CO_BOND_LENGTH_ANGSTROM
    
    # THE ISSUE: We have 3 different coordinate systems:
    # 1. Normalized atomic coords (C-O = 1.0)
    # 2. ESP grid coords in the NPZ (0-49 range, unknown units)
    # 3. Original cube coords in Bohr (-6.125 to +6.125 Bohr)
    #
    # The vdw_grid appears to be in GRID INDEX space (0-49), not physical units!
    # We need to convert to physical coordinates using origin + axes * indices / (dims - 1)
    #
    # Actually, looking at split_npz_cube.py, the coords are computed as:
    # origin + i*axes[0] + j*axes[1] + k*axes[2]
    # where axes have unit spacing (1.0), so this gives 0-49
    #
    # But the original cube has spacing 0.25 Bohr, dimensions 50
    # So physical coord = origin_bohr + (grid_index / (dims-1)) * (dims-1) * spacing_bohr
    #                   = origin_bohr + grid_index * spacing_bohr
    #
    # Wait, the grid_axes in NPZ is [[1,0,0],[0,1,0],[0,0,1]], not [[0.25,0,0], ...]
    # This means the vdw_grid is already in PHYSICAL coordinates!
    # Let me just use origin as the reference point
    
    # The simplest approach: place molecule at grid center
    # Since we don't know the exact transformation, center both
    grid_center_coords = coords.mean(axis=0)
    mol_center_normalized = valid_pos.mean(axis=0)
    
    # Scale molecule to physical size (C-O = 1.16 Angstroms or 2.19 Bohr)
    # Check if grid is in Angstroms or Bohr based on extent
    grid_extent = coords.max(axis=0) - coords.min(axis=0)
    avg_extent = grid_extent.mean()
    
    if avg_extent > 40:  # If extent is ~49, likely Angstroms
        # Grid is in Angstroms, scale molecule to Angstroms
        CO_BOND_ANGSTROM = 1.16
        valid_pos_scaled = (valid_pos - mol_center_normalized) * CO_BOND_ANGSTROM + grid_center_coords
        coord_label = "Angstrom"
    else:  # If extent is ~12, likely Bohr
        # Grid is in Bohr, scale molecule to Bohr  
        CO_BOND_BOHR = 2.19
        valid_pos_scaled = (valid_pos - mol_center_normalized) * CO_BOND_BOHR + grid_center_coords
        coord_label = "Bohr"
    
    print(f"Grid extent: {avg_extent:.1f} → using {coord_label} units")
    print(f"Molecule centered at: {grid_center_coords}")
    
    fig = plt.figure(figsize=(18, 6))
    
    # Make colormap symmetric around 0
    max_abs = np.percentile(np.abs(esp), 98)
    vmin, vmax = -max_abs, max_abs
    
    # Three different viewing angles
    angles = [(30, 45), (30, 135), (60, -90)]
    titles = ['View 1 (30°, 45°)', 'View 2 (30°, 135°)', 'View 3 (60°, -90°)']
    
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Plot ESP grid points
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c=esp, cmap='bwr', s=1, vmin=vmin, vmax=vmax,
                           alpha=0.3, edgecolors='none')
        
        # Plot atoms
        for atom_idx in range(len(valid_pos_scaled)):
            pos_physical = valid_pos_scaled[atom_idx]
            atomic_num = int(valid_z[atom_idx])
            
            # Radius in same units as coordinates
            if coord_label == "Angstrom":
                radius = covalent_radii[atomic_num] * 0.5
            else:  # Bohr
                radius = covalent_radii[atomic_num] * 1.88973 * 0.5
            
            # Color by element
            if atomic_num == 6:  # Carbon
                color = 'gray'
                radius_scale = 1.2
            elif atomic_num == 8:  # Oxygen
                color = 'red'
                radius_scale = 1.0
            elif atomic_num == 1:  # Hydrogen
                color = 'white'
                radius_scale = 0.8
            else:
                color = 'blue'
                radius_scale = 1.0
            
            # Plot atom as sphere
            ax.scatter([pos_physical[0]], [pos_physical[1]], [pos_physical[2]],
                      c=color, s=radius * radius_scale * 200, alpha=0.9, 
                      edgecolors='black', linewidths=1.5, zorder=10)
        
        # Draw bonds (simple distance-based)
        for idx1 in range(len(valid_pos_scaled)):
            for idx2 in range(idx1+1, len(valid_pos_scaled)):
                pos1 = valid_pos_scaled[idx1]
                pos2 = valid_pos_scaled[idx2]
                dist = np.linalg.norm(pos1 - pos2)
                
                # Bond if distance < sum of covalent radii * 1.3
                z1, z2 = int(valid_z[idx1]), int(valid_z[idx2])
                if coord_label == "Angstrom":
                    max_bond_dist = (covalent_radii[z1] + covalent_radii[z2]) * 1.3
                else:  # Bohr
                    max_bond_dist = (covalent_radii[z1] + covalent_radii[z2]) * 1.88973 * 1.3
                
                if dist < max_bond_dist:
                    ax.plot([pos1[0], pos2[0]], 
                           [pos1[1], pos2[1]], 
                           [pos1[2], pos2[2]], 
                           'k-', linewidth=2, alpha=0.8, zorder=9)
        
        ax.set_xlabel(f'X ({coord_label})', fontsize=10)
        ax.set_ylabel(f'Y ({coord_label})', fontsize=10)
        ax.set_zlabel(f'Z ({coord_label})', fontsize=10)
        ax.set_title(titles[i], fontsize=11)
        ax.view_init(elev=elev, azim=azim)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label(f'ESP (Hartree/e per {coord_label})', fontsize=9)
    
    fig.suptitle(f'ESP with Molecular Structure (Sample {sample_idx}, {len(valid_pos_scaled)} atoms, {len(coords)} grid points)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_esp_3d_scatter(grids, esp_values, sample_idx=0):
    """
    Plot 3D scatter of all ESP grid points.
    
    Parameters
    ----------
    grids : array_like
        Grid coordinates of shape (n_samples, n_points, 3)
    esp_values : array_like
        ESP values of shape (n_samples, n_points)
    sample_idx : int
        Index of the sample to visualize
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    coords = grids[sample_idx]
    esp = esp_values[sample_idx]
    
    fig = plt.figure(figsize=(18, 6))
    
    # Make colormap symmetric around 0
    max_abs = np.percentile(np.abs(esp), 98)
    vmin, vmax = -max_abs, max_abs
    
    # Three different viewing angles
    angles = [(30, 45), (30, 135), (60, -90)]
    titles = ['View 1 (30°, 45°)', 'View 2 (30°, 135°)', 'View 3 (60°, -90°)']
    
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c=esp, cmap='bwr', s=1, vmin=vmin, vmax=vmax,
                           alpha=0.6)
        ax.set_xlabel('X (Bohr)', fontsize=10)
        ax.set_ylabel('Y (Bohr)', fontsize=10)
        ax.set_zlabel('Z (Bohr)', fontsize=10)
        ax.set_title(titles[i], fontsize=11)
        ax.view_init(elev=elev, azim=azim)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('ESP (Hartree/Bohr)', fontsize=9)
    
    fig.suptitle(f'3D ESP Grid Points (Sample {sample_idx}, {len(coords)} points)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_esp_slices(grids, esp_values, sample_idx=0, slice_position=0.0, slice_thickness=2.0):
    """
    Plot 2D slices of ESP through XY, XZ, and YZ planes.
    
    Parameters
    ----------
    grids : array_like
        Grid coordinates of shape (n_samples, n_points, 3)
    esp_values : array_like
        ESP values of shape (n_samples, n_points)
    sample_idx : int
        Index of the sample to visualize
    slice_position : float
        Position along the perpendicular axis for the slice
    slice_thickness : float
        Thickness of the slice (points within this distance are included)
    """
    coords = grids[sample_idx]
    esp = esp_values[sample_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'ESP Slices (Sample {sample_idx}, thickness={slice_thickness:.1f} Bohr)', 
                 fontsize=16, fontweight='bold')
    
    # Make colormap symmetric around 0
    max_abs = np.percentile(np.abs(esp), 98)
    vmin, vmax = -max_abs, max_abs
    
    # XY slice (fixed Z)
    z_slice_mask = np.abs(coords[:, 2] - slice_position) < slice_thickness
    if z_slice_mask.sum() > 0:
        xy_coords = coords[z_slice_mask][:, :2]
        xy_esp = esp[z_slice_mask]
        sc1 = axes[0].scatter(xy_coords[:, 0], xy_coords[:, 1], c=xy_esp, 
                             cmap='bwr', s=5, vmin=vmin, vmax=vmax, alpha=0.8,
                             edgecolors='none')
        axes[0].set_xlabel('X (Bohr)', fontsize=12)
        axes[0].set_ylabel('Y (Bohr)', fontsize=12)
        axes[0].set_title(f'XY Plane (Z ≈ {slice_position:.2f})\n{z_slice_mask.sum()} points', 
                         fontsize=11)
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        cbar1 = plt.colorbar(sc1, ax=axes[0])
        cbar1.set_label('ESP (Hartree/Bohr)', fontsize=10)
    
    # XZ slice (fixed Y)
    y_slice_mask = np.abs(coords[:, 1] - slice_position) < slice_thickness
    if y_slice_mask.sum() > 0:
        xz_coords = coords[y_slice_mask][:, [0, 2]]
        xz_esp = esp[y_slice_mask]
        sc2 = axes[1].scatter(xz_coords[:, 0], xz_coords[:, 1], c=xz_esp,
                             cmap='bwr', s=5, vmin=vmin, vmax=vmax, alpha=0.8,
                             edgecolors='none')
        axes[1].set_xlabel('X (Bohr)', fontsize=12)
        axes[1].set_ylabel('Z (Bohr)', fontsize=12)
        axes[1].set_title(f'XZ Plane (Y ≈ {slice_position:.2f})\n{y_slice_mask.sum()} points',
                         fontsize=11)
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        cbar2 = plt.colorbar(sc2, ax=axes[1])
        cbar2.set_label('ESP (Hartree/Bohr)', fontsize=10)
    
    # YZ slice (fixed X)
    x_slice_mask = np.abs(coords[:, 0] - slice_position) < slice_thickness
    if x_slice_mask.sum() > 0:
        yz_coords = coords[x_slice_mask][:, 1:]
        yz_esp = esp[x_slice_mask]
        sc3 = axes[2].scatter(yz_coords[:, 0], yz_coords[:, 1], c=yz_esp,
                             cmap='bwr', s=5, vmin=vmin, vmax=vmax, alpha=0.8,
                             edgecolors='none')
        axes[2].set_xlabel('Y (Bohr)', fontsize=12)
        axes[2].set_ylabel('Z (Bohr)', fontsize=12)
        axes[2].set_title(f'YZ Plane (X ≈ {slice_position:.2f})\n{x_slice_mask.sum()} points',
                         fontsize=11)
        axes[2].set_aspect('equal')
        axes[2].grid(True, alpha=0.3)
        cbar3 = plt.colorbar(sc3, ax=axes[2])
        cbar3.set_label('ESP (Hartree/Bohr)', fontsize=10)
    
    plt.tight_layout()
    return fig


def main():
    """Main function to load and visualize CO2 data."""
    # Set up data directory
    data_dir = Path(__file__).parent / "../../.." / "testdata"
    data_dir = data_dir.resolve()
    print(f"Data directory: {data_dir}\n")
    
    # Create output directory for plots
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load energy, forces, and dipoles
    print("Loading energies_forces_dipoles.npz...")
    try:
        energies_forces_dipoles = np.load(data_dir / "energies_forces_dipoles.npz")
    except FileNotFoundError:
        print(f"ERROR: Could not find energies_forces_dipoles.npz in {data_dir}")
        return
    
    print("Available keys:", list(energies_forces_dipoles.keys()))
    for key in energies_forces_dipoles.keys():
        print(f"  {key}: shape {energies_forces_dipoles[key].shape}")
    print()
    
    # Check atomic coordinates in this file too
    if 'R' in energies_forces_dipoles and 'Z' in energies_forces_dipoles:
        print("\n" + "="*60)
        print("Checking coordinates in energies_forces_dipoles.npz:")
        print("="*60)
        check_coordinate_units(energies_forces_dipoles['R'], energies_forces_dipoles['Z'], sample_idx=0)
    
    # Plot energy distribution
    if 'E' in energies_forces_dipoles or 'energies' in energies_forces_dipoles or 'energy' in energies_forces_dipoles:
        if 'E' in energies_forces_dipoles:
            energy_key = 'E'
        elif 'energies' in energies_forces_dipoles:
            energy_key = 'energies'
        else:
            energy_key = 'energy'
        energies = energies_forces_dipoles[energy_key]
        print(f"Energy statistics:")
        print(f"  Mean: {np.mean(energies):.6f} Hartree")
        print(f"  Std:  {np.std(energies):.6f} Hartree")
        print(f"  Min:  {np.min(energies):.6f} Hartree")
        print(f"  Max:  {np.max(energies):.6f} Hartree\n")
        
        # Plot energy histogram
        fig_energy = plt.figure(figsize=(10, 6))
        plt.hist(energies, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Energy (Hartree)')
        plt.ylabel('Frequency')
        plt.title('Energy Distribution')
        plt.tight_layout()
        fig_energy.savefig(output_dir / "energy_distribution.png", dpi=300, bbox_inches='tight')
        plt.close(fig_energy)
        print(f"✓ Saved energy_distribution.png")
    
    # Plot force distributions
    if 'F' in energies_forces_dipoles or 'forces' in energies_forces_dipoles or 'force' in energies_forces_dipoles:
        if 'F' in energies_forces_dipoles:
            force_key = 'F'
        elif 'forces' in energies_forces_dipoles:
            force_key = 'forces'
        else:
            force_key = 'force'
        forces = energies_forces_dipoles[force_key]
        print(f"\nForce statistics:")
        print(f"  Shape: {forces.shape}")
        forces_flat = forces.reshape(-1, 3)
        force_norms = np.linalg.norm(forces_flat, axis=1)
        print(f"  Mean norm: {np.mean(force_norms):.6e} Hartree/Bohr")
        print(f"  Std norm:  {np.std(force_norms):.6e} Hartree/Bohr\n")
        
        fig_forces = plot_force_distributions(forces)
        fig_forces.savefig(output_dir / "force_distributions.png", dpi=300, bbox_inches='tight')
        plt.close(fig_forces)
        print(f"✓ Saved force_distributions.png")
    
    # Plot dipole distribution
    if 'Dxyz' in energies_forces_dipoles or 'dipoles' in energies_forces_dipoles or 'dipole' in energies_forces_dipoles:
        if 'Dxyz' in energies_forces_dipoles:
            dipole_key = 'Dxyz'
        elif 'dipoles' in energies_forces_dipoles:
            dipole_key = 'dipoles'
        else:
            dipole_key = 'dipole'
        dipoles = energies_forces_dipoles[dipole_key]
        dipole_norms = np.linalg.norm(dipoles, axis=-1)
        print(f"\nDipole statistics:")
        print(f"  Mean norm: {np.mean(dipole_norms):.6f} Debye")
        print(f"  Std norm:  {np.std(dipole_norms):.6f} Debye\n")
        
        # Plot dipole histogram
        fig_dipole = plt.figure(figsize=(10, 6))
        plt.hist(dipole_norms, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Dipole Moment Norm (Debye)')
        plt.ylabel('Frequency')
        plt.title('Dipole Moment Distribution')
        plt.tight_layout()
        fig_dipole.savefig(output_dir / "dipole_distribution.png", dpi=300, bbox_inches='tight')
        plt.close(fig_dipole)
        print(f"✓ Saved dipole_distribution.png")
    
    # Load and visualize ESP grids
    print("\nLoading grids_esp.npz...")
    try:
        grids_esp = np.load(data_dir / "grids_esp.npz")
    except FileNotFoundError:
        print(f"ERROR: Could not find grids_esp.npz in {data_dir}")
        print("Saved all plots created so far.")
        return
    
    print("Available keys:", list(grids_esp.keys()))
    for key in grids_esp.keys():
        print(f"  {key}: shape {grids_esp[key].shape}")
    print()
    
    # Check coordinate units by calculating bond lengths
    if 'R' in grids_esp and 'Z' in grids_esp:
        check_coordinate_units(grids_esp['R'], grids_esp['Z'], sample_idx=0)
        
        # Also show ESP grid coordinate range for comparison
        grids_key = 'vdw_grid' if 'vdw_grid' in grids_esp else 'grids'
        if grids_key in grids_esp:
            grid_coords = grids_esp[grids_key][0]  # First sample
            print(f"ESP Grid coordinate ranges (sample 0):")
            print(f"  X: [{np.min(grid_coords[:, 0]):.4f}, {np.max(grid_coords[:, 0]):.4f}]")
            print(f"  Y: [{np.min(grid_coords[:, 1]):.4f}, {np.max(grid_coords[:, 1]):.4f}]")
            print(f"  Z: [{np.min(grid_coords[:, 2]):.4f}, {np.max(grid_coords[:, 2]):.4f}]")
            
            # Check if there's grid transformation info
            if 'grid_origin' in grids_esp and 'grid_axes' in grids_esp and 'grid_dims' in grids_esp:
                origin = grids_esp['grid_origin'][0]
                axes = grids_esp['grid_axes'][0]
                dims = grids_esp['grid_dims'][0]
                print(f"\nGrid transformation info:")
                print(f"  Origin: {origin}")
                print(f"  Dimensions: {dims}")
                print(f"  Axes:\n{axes}")
                
                # Try to transform atomic positions
                pos = grids_esp['R'][0]
                z_nums = grids_esp['Z'][0]
                valid_mask = z_nums > 0
                valid_pos = pos[valid_mask]
                
                # Transform: position_grid = origin + axes @ (dims * position_normalized)
                # The dims might scale the normalized coordinates
                scaled_pos = valid_pos * dims
                transformed_pos = origin + (axes @ scaled_pos.T).T
                print(f"\nScaled atomic positions (sample 0):")
                for i, sp in enumerate(scaled_pos):
                    print(f"  Atom {i}: {sp}")
                print(f"\nTransformed atomic positions (sample 0):")
                for i, tp in enumerate(transformed_pos):
                    print(f"  Atom {i}: {tp}")
                
                # Calculate bond lengths after transformation
                if len(transformed_pos) >= 2:
                    bond_01 = np.linalg.norm(transformed_pos[0] - transformed_pos[1])
                    print(f"\nBond length after transformation:")
                    print(f"  Atom 0-1: {bond_01:.4f}")
                    if 1.10 < bond_01 < 1.25:
                        print(f"  → This matches C-O in Angstroms (~1.16 Å) ✓")
                    elif 2.0 < bond_01 < 2.5:
                        print(f"  → This matches C-O in Bohr (~2.19 Bohr) ✓")
                
                # Check if transformed positions are within grid
                print(f"\nAre transformed atoms within grid?")
                for i, tp in enumerate(transformed_pos):
                    in_grid = (np.min(grid_coords[:, 0]) <= tp[0] <= np.max(grid_coords[:, 0]) and
                              np.min(grid_coords[:, 1]) <= tp[1] <= np.max(grid_coords[:, 1]) and
                              np.min(grid_coords[:, 2]) <= tp[2] <= np.max(grid_coords[:, 2]))
                    print(f"  Atom {i}: {in_grid}")
            
            print()
    
    # Plot ESP slices
    grids_key = 'vdw_grid' if 'vdw_grid' in grids_esp else 'grids'
    if grids_key in grids_esp and 'esp' in grids_esp:
        grids = grids_esp[grids_key]
        esp_vals = grids_esp['esp']
        
        # Get atomic positions and numbers for molecule overlay
        positions_esp = grids_esp['R'] if 'R' in grids_esp else None
        atomic_nums_esp = grids_esp['Z'] if 'Z' in grids_esp else None
        
        print(f"ESP statistics:")
        print(f"  Mean: {np.mean(esp_vals):.6e} Hartree/Bohr")
        print(f"  Std:  {np.std(esp_vals):.6e} Hartree/Bohr")
        print(f"  Min:  {np.min(esp_vals):.6e} Hartree/Bohr")
        print(f"  Max:  {np.max(esp_vals):.6e} Hartree/Bohr\n")
        
        # Plot ESP distribution
        fig_esp_dist = plt.figure(figsize=(10, 6))
        plt.hist(esp_vals.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('ESP (Hartree/Bohr)')
        plt.ylabel('Frequency')
        plt.title('ESP Distribution')
        plt.tight_layout()
        fig_esp_dist.savefig(output_dir / "esp_distribution.png", dpi=300, bbox_inches='tight')
        plt.close(fig_esp_dist)
        print(f"✓ Saved esp_distribution.png")
        
        # Plot ESP slices for first sample
        if len(grids) > 0:
            # Create slices with increased thickness to show more points
            fig_slices = plot_esp_slices(grids, esp_vals, sample_idx=0, 
                                         slice_position=0.0, slice_thickness=2.0)
            fig_slices.savefig(output_dir / "esp_slices.png", dpi=300, bbox_inches='tight')
            plt.close(fig_slices)
            print(f"✓ Saved esp_slices.png")
            
            # Create 3D scatter plot to see all ESP points
            fig_3d = plot_esp_3d_scatter(grids, esp_vals, sample_idx=0)
            fig_3d.savefig(output_dir / "esp_3d_scatter.png", dpi=300, bbox_inches='tight')
            plt.close(fig_3d)
            print(f"✓ Saved esp_3d_scatter.png")
            
            # Create 3D scatter plot WITH molecular structure overlay
            if positions_esp is not None and atomic_nums_esp is not None:
                fig_3d_mol = plot_esp_3d_with_molecule(grids, esp_vals, positions_esp, 
                                                        atomic_nums_esp, sample_idx=0)
                fig_3d_mol.savefig(output_dir / "esp_3d_with_molecule.png", dpi=300, bbox_inches='tight')
                plt.close(fig_3d_mol)
                print(f"✓ Saved esp_3d_with_molecule.png")
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*60}")
    print("Generated files:")
    print("  - energy_distribution.png")
    print("  - force_distributions.png")
    print("  - dipole_distribution.png")
    print("  - esp_distribution.png")
    print("  - esp_slices.png")
    print("  - esp_3d_scatter.png")
    print("  - esp_3d_with_molecule.png  (ESP with molecular structure overlay)")


if __name__ == "__main__":
    main()
