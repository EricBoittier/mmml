#!/usr/bin/env python3
"""
Quick 3D Matplotlib Visualization (no POV-Ray required)

Creates interactive or static 3D plots with:
- Molecular structure
- Distributed charges as colored spheres
- ESP on VDW surface

Usage:
    python matplotlib_3d_viz.py \
        --checkpoint checkpoints/co2_model \
        --structure co2.xyz \
        --output figure.png
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import argparse
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from ase.io import read


# Element colors (CPK)
ELEMENT_COLORS = {
    1: '#FFFFFF',   # H - white
    6: '#404040',   # C - dark gray
    7: '#3050F8',   # N - blue
    8: '#FF0D0D',   # O - red
    9: '#90E050',   # F - green
    15: '#FF8000',  # P - orange
    16: '#FFFF30',  # S - yellow
}

# Covalent radii (Angstroms) for plotting
RADII = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 
    15: 1.07, 16: 1.05,
}


def load_model(checkpoint_dir):
    """Load model from checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    
    with open(checkpoint_dir / 'model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from trainer import JointPhysNetDCMNet
    
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config'],
        mix_coulomb_energy=config.get('mix_coulomb_energy', False)
    )
    
    with open(checkpoint_dir / 'best_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    return model, params, config


def evaluate_molecule(atoms, model, params, config):
    """Evaluate molecule to get charges."""
    import e3x
    
    positions = jnp.array(atoms.get_positions())
    atomic_numbers = jnp.array(atoms.get_atomic_numbers())
    
    natoms_actual = len(atoms)
    natoms_padded = config['physnet_config']['natoms']
    batch_size = 1
    
    # Pad to model's expected size
    if natoms_actual < natoms_padded:
        pad_size = natoms_padded - natoms_actual
        positions = jnp.concatenate([positions, jnp.zeros((pad_size, 3))], axis=0)
        atomic_numbers = jnp.concatenate([atomic_numbers, jnp.zeros(pad_size, dtype=jnp.int32)], axis=0)
    
    # Create masks
    atom_mask = jnp.array([1.0] * natoms_actual + [0.0] * (natoms_padded - natoms_actual))
    batch_mask = jnp.ones(batch_size)
    batch_segments = jnp.repeat(jnp.arange(batch_size), natoms_padded)
    
    # Flatten for batch dimension
    positions_flat = positions.reshape(batch_size * natoms_padded, 3)
    atomic_numbers_flat = atomic_numbers.reshape(batch_size * natoms_padded)
    
    # Compute graph
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(natoms_padded)
    
    # Forward pass
    output = model.apply(
        {'params': params['params']},
        atomic_numbers_flat,
        positions_flat,
        dst_idx,
        src_idx,
        batch_segments,
        batch_size,
        batch_mask,
        atom_mask,
    )
    
    # Extract distributed charges
    n_dcm = config['dcmnet_config']['n_dcm']
    # mono_dist: (batch*natoms, n_dcm) - charge values
    # dipo_dist: (batch*natoms, n_dcm, 3) - charge positions
    mono_values = np.array(output['mono_dist'][:natoms_actual, :])  # (natoms, n_dcm)
    charge_positions = np.array(output['dipo_dist'][:natoms_actual, :, :])  # (natoms, n_dcm, 3)
    
    return {
        'distributed_mono': charge_positions.reshape(-1, 3),
        'distributed_mono_values': mono_values.reshape(-1),
        'energy': float(output['energy'][0]),
    }


def compute_esp_surface(atoms, charge_pos, charge_vals, n_points=2000):
    """Compute ESP on VDW surface."""
    vdw_radii = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52}
    
    # Fibonacci sphere
    indices = np.arange(n_points)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1 - (indices / float(n_points - 1)) * 2
    radius = np.sqrt(1 - y * y)
    theta = phi * indices
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    sphere = np.stack([x, y, z], axis=1)
    
    surface_points = []
    for pos, Z in zip(atoms.get_positions(), atoms.get_atomic_numbers()):
        r = vdw_radii.get(Z, 1.7) * 1.4
        surface_points.append(sphere * r + pos)
    
    surface_points = np.vstack(surface_points)
    
    # Compute ESP
    esp_values = np.zeros(len(surface_points))
    for q, r_q in zip(charge_vals, charge_pos):
        diff = surface_points - r_q
        dist = np.linalg.norm(diff, axis=1)
        mask = dist > 0.01
        esp_values[mask] += q / (dist[mask] * 1.88973)
    
    return surface_points, esp_values


def plot_molecule_3d(atoms, result, show_molecule='full', show_charges=False, show_esp=False, 
                     elev=20, azim=45, figsize=(12, 10)):
    """
    Create 3D matplotlib visualization.
    
    Parameters
    ----------
    show_molecule : str
        'full' - ball-and-stick (default)
        'wireframe' - thin bonds only
        'none' - no molecule
    show_charges : bool
        Show distributed charges
    show_esp : bool
        Show ESP surface
        
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    
    # Plot molecule if requested
    if show_molecule != 'none':
        if show_molecule == 'wireframe':
            # Wireframe: thin bonds, small atoms
            atom_radius_scale = 0.25
            bond_width = 2
            atom_alpha = 0.8
        else:  # 'full'
            # Full ball-and-stick
            atom_radius_scale = 1.0
            bond_width = 3
            atom_alpha = 0.9
        
        # Plot atoms
        for pos, Z in zip(positions, atomic_numbers):
            color = ELEMENT_COLORS.get(Z, '#808080')
            radius = RADII.get(Z, 0.7) * atom_radius_scale
            
            # Draw atom as sphere
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 10)
            x = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + pos[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
            
            ax.plot_surface(x, y, z, color=color, alpha=atom_alpha, 
                           linewidth=0, antialiased=True, shade=True)
        
        # Draw bonds
        cutoff = 1.8
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    ax.plot([positions[i, 0], positions[j, 0]],
                           [positions[i, 1], positions[j, 1]],
                           [positions[i, 2], positions[j, 2]],
                           'k-', linewidth=bond_width, alpha=0.6, zorder=0)
    
    # Plot distributed charges
    if show_charges and 'distributed_mono' in result:
        charge_pos = result['distributed_mono']
        charge_vals = result['distributed_mono_values']
        
        # Filter small charges
        mask = np.abs(charge_vals) > 0.005
        charge_pos = charge_pos[mask]
        charge_vals = charge_vals[mask]
        
        # Color map
        vmax = max(abs(charge_vals.min()), abs(charge_vals.max()), 0.01)
        norm = Normalize(vmin=-vmax, vmax=vmax)
        cmap = cm.get_cmap('RdBu_r')
        
        colors = cmap(norm(charge_vals))
        sizes = np.abs(charge_vals) * 2000 + 50
        
        scatter = ax.scatter(charge_pos[:, 0], charge_pos[:, 1], charge_pos[:, 2],
                            c=charge_vals, cmap='RdBu_r', s=sizes,
                            alpha=0.6, edgecolors='black', linewidths=0.5,
                            vmin=-vmax, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Charge (e)', fontsize=12, weight='bold')
    
    # Plot ESP surface
    if show_esp and 'distributed_mono' in result:
        print("Computing ESP surface...")
        surface_pos, esp_vals = compute_esp_surface(
            atoms, result['distributed_mono'], result['distributed_mono_values']
        )
        
        # Subsample
        step = max(1, len(surface_pos) // 1500)
        surface_pos = surface_pos[::step]
        esp_vals = esp_vals[::step]
        
        # Color map
        vmax_esp = max(abs(np.percentile(esp_vals, 5)),
                      abs(np.percentile(esp_vals, 95)), 0.001)
        
        scatter_esp = ax.scatter(
            surface_pos[:, 0], surface_pos[:, 1], surface_pos[:, 2],
            c=esp_vals, cmap='RdBu_r', s=20, alpha=0.4,
            vmin=-vmax_esp, vmax=vmax_esp
        )
        
        cbar_esp = plt.colorbar(scatter_esp, ax=ax, pad=0.15, shrink=0.8)
        cbar_esp.set_label('ESP (Ha/e)', fontsize=12, weight='bold')
    
    # Set view
    ax.view_init(elev=elev, azim=azim)
    
    # Labels
    ax.set_xlabel('X (Å)', fontsize=12, weight='bold')
    ax.set_ylabel('Y (Å)', fontsize=12, weight='bold')
    ax.set_zlabel('Z (Å)', fontsize=12, weight='bold')
    
    # Equal aspect ratio
    center = positions.mean(axis=0)
    max_range = np.abs(positions - center).max() * 1.5
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    plt.tight_layout()
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description='Quick 3D matplotlib visualization'
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--structure', type=str, required=True)
    parser.add_argument('--output', type=str, default='molecule_3d.png')
    parser.add_argument('--plot-type', choices=['molecule', 'molecule+charges', 'esp', 'molecule+esp', 'charges'],
                       default='molecule+charges',
                       help='Type of plot to generate')
    parser.add_argument('--molecule-style', choices=['full', 'wireframe'],
                       default='wireframe',
                       help='Molecule style (default: wireframe for charges/esp, full for molecule-only)')
    parser.add_argument('--interactive', action='store_true',
                       help='Show interactive plot window')
    parser.add_argument('--elev', type=float, default=20)
    parser.add_argument('--azim', type=float, default=45)
    parser.add_argument('--dpi', type=int, default=300)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("📊 Matplotlib 3D Molecular Visualization")
    print("="*60)
    
    # Load
    print(f"\n📂 Loading: {args.structure}")
    atoms = read(args.structure)
    print(f"   • {len(atoms)} atoms: {atoms.get_chemical_formula()}")
    
    print(f"\n🔧 Loading model: {args.checkpoint}")
    model, params, config = load_model(args.checkpoint)
    
    print(f"\n⚡ Evaluating...")
    result = evaluate_molecule(atoms, model, params, config)
    print(f"   • Energy: {result['energy']:.6f} eV")
    print(f"   • Charge sites: {len(result['distributed_mono'])}")
    
    # Determine what to show based on plot type
    if args.plot_type == 'molecule':
        show_mol = 'full'
        show_charges = False
        show_esp = False
    elif args.plot_type == 'molecule+charges':
        show_mol = args.molecule_style
        show_charges = True
        show_esp = False
    elif args.plot_type == 'charges':
        show_mol = 'none'
        show_charges = True
        show_esp = False
    elif args.plot_type == 'esp':
        show_mol = 'none'
        show_charges = False
        show_esp = True
    elif args.plot_type == 'molecule+esp':
        show_mol = 'wireframe'
        show_charges = False
        show_esp = True
    
    # Plot
    print(f"\n🎨 Creating {args.plot_type} visualization...")
    fig, ax = plot_molecule_3d(
        atoms, result,
        show_molecule=show_mol,
        show_charges=show_charges,
        show_esp=show_esp,
        elev=args.elev,
        azim=args.azim,
    )
    
    # Save
    print(f"\n💾 Saving to: {args.output}")
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    if args.interactive:
        print("\n👁 Opening interactive window...")
        plt.show()
    else:
        plt.close()
    
    print("\n✅ Done!\n")


if __name__ == '__main__':
    main()

