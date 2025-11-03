#!/usr/bin/env python3
"""
High-Quality POV-Ray Visualizations with Distributed Charges and ESP

Creates publication-quality molecular visualizations showing:
1. Ball-and-stick molecular structure
2. Distributed charges as colored spheres
3. ESP mapped on VDW surface

Usage:
    python povray_visualization.py \
        --checkpoint /path/to/checkpoint \
        --structure co2.xyz \
        --output-dir ./povray_figures \
        --show-charges \
        --show-esp
"""

import numpy as np
import argparse
from pathlib import Path
import pickle
import subprocess
import tempfile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import jax
import jax.numpy as jnp
from ase.io import read, write
from ase import Atoms
import ase.io
import warnings

# Atomic radii for ball-and-stick rendering (in Angstroms)
COVALENT_RADII = {
    1: 0.31,   # H
    6: 0.76,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    15: 1.07,  # P
    16: 1.05,  # S
}

# VDW radii for surface (in Angstroms)
VDW_RADII = {
    1: 1.20,   # H
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    15: 1.80,  # P
    16: 1.80,  # S
}

# Element colors (CPK coloring)
ELEMENT_COLORS = {
    1: (1.0, 1.0, 1.0),     # H - white
    6: (0.2, 0.2, 0.2),     # C - dark gray
    7: (0.0, 0.0, 1.0),     # N - blue
    8: (1.0, 0.0, 0.0),     # O - red
    15: (1.0, 0.5, 0.0),    # P - orange
    16: (1.0, 1.0, 0.0),    # S - yellow
}


def load_model(checkpoint_dir):
    """Load trained model."""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load config
    config_file = checkpoint_dir / 'model_config.pkl'
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
    
    # Import from trainer
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from trainer import JointPhysNetDCMNet
    
    # Create joint model
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config'],
        mix_coulomb_energy=config.get('mix_coulomb_energy', False)
    )
    
    # Load parameters
    params_file = checkpoint_dir / 'best_params.pkl'
    with open(params_file, 'rb') as f:
        params = pickle.load(f)
    
    return model, params, config


def evaluate_molecule(atoms, model, params, config):
    """
    Evaluate model on a molecule to get charges and ESP.
    
    Returns
    -------
    dict with:
        - atomic_charges: (natoms,) PhysNet atomic charges
        - distributed_mono: (n_charges, 3) monopole positions  
        - distributed_mono_values: (n_charges,) monopole charge values
        - vdw_surface: (n_surface, 3) VDW surface points
        - esp_surface: (n_surface,) ESP values at surface points
        - energy: float
    """
    import e3x
    
    # Prepare input
    positions = jnp.array(atoms.get_positions())
    atomic_numbers = jnp.array(atoms.get_atomic_numbers())
    
    # Pad to model's expected size
    natoms_actual = len(atoms)
    natoms_padded = config['physnet_config']['natoms']
    batch_size = 1
    
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
    
    # Extract charges
    atomic_charges = np.array(output['charges_as_mono'][:natoms_actual])
    
    # Get distributed charge positions and values
    n_dcm = config['dcmnet_config']['n_dcm']
    # mono_dist: (batch*natoms, n_dcm) - charge values
    # dipo_dist: (batch*natoms, n_dcm, 3) - charge positions
    mono_values = np.array(output['mono_dist'][:natoms_actual, :])  # (natoms, n_dcm)
    charge_positions = np.array(output['dipo_dist'][:natoms_actual, :, :])  # (natoms, n_dcm, 3)
    
    # Reshape for easier handling
    distributed_mono = charge_positions.reshape(-1, 3)  # (natoms*n_dcm, 3)
    distributed_mono_values = mono_values.reshape(-1)  # (natoms*n_dcm,)
    
    # Generate VDW surface for ESP visualization
    vdw_surface, esp_surface = compute_esp_on_vdw_surface(
        atoms, distributed_mono, distributed_mono_values
    )
    
    return {
        'atomic_charges': atomic_charges,
        'distributed_mono': distributed_mono,
        'distributed_mono_values': distributed_mono_values,
        'vdw_surface': vdw_surface,
        'esp_surface': esp_surface,
        'energy': float(output['energy'][0]),
    }


def compute_esp_on_vdw_surface(atoms, charge_positions, charge_values, n_points=2000, scale=1.4):
    """
    Compute ESP on VDW surface.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Molecular structure
    charge_positions : np.ndarray (n_charges, 3)
        Positions of point charges
    charge_values : np.ndarray (n_charges,)
        Charge values
    n_points : int
        Number of surface points
    scale : float
        VDW radius scaling factor
        
    Returns
    -------
    surface_points : np.ndarray (n_surface, 3)
        Surface point coordinates
    esp_values : np.ndarray (n_surface,)
        ESP values at surface points (in Ha/e)
    """
    from scipy.spatial.transform import Rotation
    
    # Generate uniform sphere points using Fibonacci spiral
    indices = np.arange(n_points)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    
    # Spherical coordinates
    y = 1 - (indices / float(n_points - 1)) * 2
    radius = np.sqrt(1 - y * y)
    theta = phi * indices
    
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    
    sphere_points = np.stack([x, y, z], axis=1)
    
    # Create surface by placing spheres at each atom
    surface_points = []
    atom_positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    
    for i, (pos, Z) in enumerate(zip(atom_positions, atomic_numbers)):
        vdw_radius = VDW_RADII.get(Z, 1.7) * scale
        atom_surface = sphere_points * vdw_radius + pos
        surface_points.append(atom_surface)
    
    surface_points = np.vstack(surface_points)
    
    # Compute ESP at surface points
    # V = sum_i q_i / r_i (in atomic units)
    # Convert to Ha/e: multiply by 1/(4*pi*epsilon_0) in atomic units = 1
    esp_values = np.zeros(len(surface_points))
    
    for q, r_q in zip(charge_values, charge_positions):
        diff = surface_points - r_q
        distances = np.linalg.norm(diff, axis=1)
        # Avoid division by zero
        mask = distances > 0.01
        esp_values[mask] += q / (distances[mask] * 1.88973)  # Convert Å to Bohr
    
    return surface_points, esp_values


def write_povray_scene(atoms, result, output_path, show_charges=True, show_esp=False,
                       camera_location=None, camera_look_at=None, rotation='0x,0y,0z',
                       charge_scale=0.3, bond_radius=0.15, atom_scale=0.4):
    """
    Write POV-Ray scene file.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Molecular structure
    result : dict
        Result from evaluate_molecule
    output_path : Path
        Output .pov file path
    show_charges : bool
        Whether to show distributed charges
    show_esp : bool
        Whether to show ESP surface
    camera_location : tuple or None
        Camera position (x, y, z)
    camera_look_at : tuple or None
        Camera look-at point (x, y, z)
    rotation : str
        ASE rotation string
    charge_scale : float
        Radius scaling for charge spheres
    bond_radius : float
        Radius of bond cylinders
    atom_scale : float
        Scaling factor for atomic radii
    """
    
    # Get molecule center
    center = atoms.get_positions().mean(axis=0)
    
    # Default camera settings
    if camera_location is None:
        camera_location = center + np.array([0, 0, 12])
    if camera_look_at is None:
        camera_look_at = center
    
    with open(output_path, 'w') as f:
        # Header and global settings
        f.write("""// POV-Ray scene file
// Generated by povray_visualization.py

#version 3.7;
#include "colors.inc"
#include "textures.inc"
#include "finish.inc"

global_settings {
    assumed_gamma 1.0
    max_trace_level 10
}

// Background
background { color White }

""")
        
        # Camera
        f.write(f"""// Camera
camera {{
    orthographic
    location <{camera_location[0]:.3f}, {camera_location[1]:.3f}, {camera_location[2]:.3f}>
    look_at <{camera_look_at[0]:.3f}, {camera_look_at[1]:.3f}, {camera_look_at[2]:.3f}>
    right x*10
    up y*10
}}

""")
        
        # Lights
        f.write(f"""// Lighting
light_source {{
    <{camera_location[0] + 5:.3f}, {camera_location[1] + 10:.3f}, {camera_location[2] + 5:.3f}>
    color White
    parallel
}}

light_source {{
    <{camera_location[0] - 5:.3f}, {camera_location[1]:.3f}, {camera_location[2] - 5:.3f}>
    color White * 0.5
    parallel
}}

""")
        
        # Atom and bond finish
        f.write("""// Material finishes
#declare atom_finish = finish {
    ambient 0.2
    diffuse 0.7
    specular 0.3
    roughness 0.02
    metallic 0.1
}

#declare bond_finish = finish {
    ambient 0.15
    diffuse 0.75
    specular 0.1
    roughness 0.05
}

#declare charge_finish = finish {
    ambient 0.3
    diffuse 0.6
    specular 0.2
    roughness 0.05
    metallic 0.2
}

""")
        
        # Write atoms
        f.write("// Atoms\n")
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        
        for i, (pos, Z) in enumerate(zip(positions, atomic_numbers)):
            color = ELEMENT_COLORS.get(Z, (0.5, 0.5, 0.5))
            radius = COVALENT_RADII.get(Z, 0.7) * atom_scale
            
            f.write(f"sphere {{\n")
            f.write(f"    <{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}>, {radius:.3f}\n")
            f.write(f"    pigment {{ color rgb <{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}> }}\n")
            f.write(f"    finish {{ atom_finish }}\n")
            f.write(f"}}\n\n")
        
        # Write bonds
        f.write("// Bonds\n")
        cutoff = 1.8  # Bond cutoff in Angstroms
        
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                pos_i = positions[i]
                pos_j = positions[j]
                distance = np.linalg.norm(pos_i - pos_j)
                
                if distance < cutoff:
                    # Draw bond as cylinder
                    mid = (pos_i + pos_j) / 2
                    color_i = ELEMENT_COLORS.get(atomic_numbers[i], (0.5, 0.5, 0.5))
                    color_j = ELEMENT_COLORS.get(atomic_numbers[j], (0.5, 0.5, 0.5))
                    
                    # First half
                    f.write(f"cylinder {{\n")
                    f.write(f"    <{pos_i[0]:.4f}, {pos_i[1]:.4f}, {pos_i[2]:.4f}>,\n")
                    f.write(f"    <{mid[0]:.4f}, {mid[1]:.4f}, {mid[2]:.4f}>,\n")
                    f.write(f"    {bond_radius:.3f}\n")
                    f.write(f"    pigment {{ color rgb <{color_i[0]:.3f}, {color_i[1]:.3f}, {color_i[2]:.3f}> }}\n")
                    f.write(f"    finish {{ bond_finish }}\n")
                    f.write(f"}}\n\n")
                    
                    # Second half
                    f.write(f"cylinder {{\n")
                    f.write(f"    <{mid[0]:.4f}, {mid[1]:.4f}, {mid[2]:.4f}>,\n")
                    f.write(f"    <{pos_j[0]:.4f}, {pos_j[1]:.4f}, {pos_j[2]:.4f}>,\n")
                    f.write(f"    {bond_radius:.3f}\n")
                    f.write(f"    pigment {{ color rgb <{color_j[0]:.3f}, {color_j[1]:.3f}, {color_j[2]:.3f}> }}\n")
                    f.write(f"    finish {{ bond_finish }}\n")
                    f.write(f"}}\n\n")
        
        # Write distributed charges as colored spheres
        if show_charges and 'distributed_mono' in result:
            f.write("// Distributed Charges\n")
            
            charge_pos = result['distributed_mono']
            charge_vals = result['distributed_mono_values']
            
            # Color map for charges: negative (blue) to positive (red)
            vmax = max(abs(charge_vals.min()), abs(charge_vals.max()))
            vmax = max(vmax, 0.01)  # Avoid division by zero
            norm = Normalize(vmin=-vmax, vmax=vmax)
            cmap = cm.get_cmap('RdBu_r')
            
            for pos, q in zip(charge_pos, charge_vals):
                # Skip very small charges
                if abs(q) < 0.001:
                    continue
                
                # Charge sphere radius proportional to charge magnitude
                radius = charge_scale * min(abs(q) * 5, 0.3)
                
                # Color based on charge value
                rgba = cmap(norm(q))
                color = rgba[:3]
                
                f.write(f"sphere {{\n")
                f.write(f"    <{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}>, {radius:.3f}\n")
                f.write(f"    pigment {{ color rgbf <{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}, 0.3> }}\n")
                f.write(f"    finish {{ charge_finish }}\n")
                f.write(f"}}\n")
            
            f.write("\n")
        
        # Write ESP surface (semi-transparent colored surface)
        if show_esp and 'vdw_surface' in result and 'esp_surface' in result:
            f.write("// ESP Surface\n")
            
            surface_pos = result['vdw_surface']
            esp_vals = result['esp_surface']
            
            # Color map for ESP
            vmax_esp = max(abs(np.percentile(esp_vals, 5)), abs(np.percentile(esp_vals, 95)))
            vmax_esp = max(vmax_esp, 0.001)
            norm_esp = Normalize(vmin=-vmax_esp, vmax=vmax_esp)
            cmap_esp = cm.get_cmap('RdBu_r')
            
            # Draw surface as many small spheres
            for pos, esp in zip(surface_pos[::10], esp_vals[::10]):  # Subsample for performance
                rgba = cmap_esp(norm_esp(esp))
                color = rgba[:3]
                
                f.write(f"sphere {{\n")
                f.write(f"    <{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}>, 0.08\n")
                f.write(f"    pigment {{ color rgbf <{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}, 0.7> }}\n")
                f.write(f"    finish {{ ambient 0.4 diffuse 0.5 }}\n")
                f.write(f"}}\n")
            
            f.write("\n")


def render_povray(pov_file, output_image, width=2400, height=1800, quality=11):
    """
    Render POV-Ray scene to image.
    
    Parameters
    ----------
    pov_file : Path
        Input .pov file
    output_image : Path
        Output image path (.png)
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    quality : int
        POV-Ray quality (0-11, higher is better)
    """
    cmd = [
        'povray',
        f'+I{pov_file}',
        f'+O{output_image}',
        f'+W{width}',
        f'+H{height}',
        f'+Q{quality}',
        '+A',  # Anti-aliasing
        '-D',  # No display
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Rendered: {output_image}")
    except subprocess.CalledProcessError as e:
        print(f"✗ POV-Ray rendering failed:")
        print(e.stderr.decode())
        raise
    except FileNotFoundError:
        print("✗ POV-Ray not found. Please install POV-Ray:")
        print("  Ubuntu/Debian: sudo apt install povray")
        print("  macOS: brew install povray")
        print("  Or download from: http://www.povray.org/")
        raise


def create_colorbar_legend(output_path, vmin, vmax, label, cmap='RdBu_r'):
    """Create a colorbar legend as a separate image."""
    fig, ax = plt.subplots(figsize=(6, 1))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=ax, orientation='horizontal')
    cb.set_label(label, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved colorbar: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create high-quality POV-Ray visualizations with distributed charges and ESP'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint directory')
    parser.add_argument('--structure', type=str, required=True,
                       help='Input structure file (xyz, cif, etc.)')
    parser.add_argument('--output-dir', type=str, default='./povray_figures',
                       help='Output directory for figures')
    parser.add_argument('--show-charges', action='store_true',
                       help='Show distributed charges as colored spheres')
    parser.add_argument('--show-esp', action='store_true',
                       help='Show ESP on VDW surface')
    parser.add_argument('--charge-scale', type=float, default=0.3,
                       help='Scale factor for charge sphere size')
    parser.add_argument('--bond-radius', type=float, default=0.15,
                       help='Radius of bond cylinders')
    parser.add_argument('--atom-scale', type=float, default=0.4,
                       help='Scale factor for atomic radii')
    parser.add_argument('--width', type=int, default=2400,
                       help='Output image width')
    parser.add_argument('--height', type=int, default=1800,
                       help='Output image height')
    parser.add_argument('--quality', type=int, default=11,
                       help='POV-Ray quality (0-11)')
    parser.add_argument('--views', type=str, nargs='+',
                       default=['front', 'side', 'top'],
                       help='Views to render (front, side, top, or custom)')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("POV-Ray Molecular Visualization")
    print("=" * 60)
    
    # Load structure
    print(f"\n📂 Loading structure: {args.structure}")
    atoms = read(args.structure)
    print(f"   {len(atoms)} atoms")
    
    # Load model
    print(f"\n🔧 Loading model: {args.checkpoint}")
    model, params, config = load_model(args.checkpoint)
    
    # Evaluate
    print(f"\n⚡ Evaluating molecule...")
    result = evaluate_molecule(atoms, model, params, config)
    print(f"   Energy: {result['energy']:.6f} eV")
    print(f"   Total charge (atomic): {result['atomic_charges'].sum():.4f} e")
    print(f"   Total charge (distributed): {result['distributed_mono_values'].sum():.4f} e")
    print(f"   Number of charge sites: {len(result['distributed_mono'])}")
    if 'vdw_surface' in result:
        print(f"   ESP surface points: {len(result['vdw_surface'])}")
        print(f"   ESP range: [{result['esp_surface'].min():.4f}, {result['esp_surface'].max():.4f}] Ha/e")
    
    # Define view angles
    view_angles = {
        'front': (0, 0, 12),
        'side': (12, 0, 0),
        'top': (0, 12, 0),
    }
    
    # Render each view
    print(f"\n🎨 Rendering views...")
    for view_name in args.views:
        if view_name in view_angles:
            camera_offset = view_angles[view_name]
        else:
            print(f"   Warning: Unknown view '{view_name}', skipping")
            continue
        
        center = atoms.get_positions().mean(axis=0)
        camera_location = center + np.array(camera_offset)
        
        # Create POV file
        pov_file = output_dir / f'molecule_{view_name}.pov'
        png_file = output_dir / f'molecule_{view_name}.png'
        
        print(f"\n   View: {view_name}")
        print(f"      Writing scene: {pov_file}")
        
        write_povray_scene(
            atoms, result, pov_file,
            show_charges=args.show_charges,
            show_esp=args.show_esp,
            camera_location=camera_location,
            camera_look_at=center,
            charge_scale=args.charge_scale,
            bond_radius=args.bond_radius,
            atom_scale=args.atom_scale,
        )
        
        # Render
        print(f"      Rendering to: {png_file}")
        render_povray(pov_file, png_file, args.width, args.height, args.quality)
    
    # Create colorbars
    if args.show_charges:
        vmax = max(abs(result['distributed_mono_values'].min()),
                  abs(result['distributed_mono_values'].max()))
        create_colorbar_legend(
            output_dir / 'colorbar_charges.png',
            -vmax, vmax,
            'Distributed Charge (e)',
            cmap='RdBu_r'
        )
    
    if args.show_esp:
        vmax_esp = max(abs(np.percentile(result['esp_surface'], 5)),
                      abs(np.percentile(result['esp_surface'], 95)))
        create_colorbar_legend(
            output_dir / 'colorbar_esp.png',
            -vmax_esp, vmax_esp,
            'Electrostatic Potential (Ha/e)',
            cmap='RdBu_r'
        )
    
    print("\n" + "=" * 60)
    print("✓ Visualization complete!")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

