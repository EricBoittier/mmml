#!/usr/bin/env python3
"""
Advanced POV-Ray Visualization using ASE's Native POV-Ray Writer

Creates publication-quality figures with:
- Ball-and-stick molecular structure (using ASE's POV-Ray writer)
- Distributed charges as colored translucent spheres
- ESP on VDW surface as colored mesh

Usage:
    python ase_povray_viz.py \
        --checkpoint checkpoints/co2_model \
        --structure co2.xyz \
        --output-dir ./figures_povray \
        --resolution high
"""

import numpy as np
import argparse
from pathlib import Path
import pickle
import subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import jax
import jax.numpy as jnp
from ase.io import read, write
from ase import Atoms
import ase.io


# Element colors (jmol coloring scheme)
JMOL_COLORS = {
    1: [1.000, 1.000, 1.000],  # H - white
    6: [0.565, 0.565, 0.565],  # C - gray
    7: [0.188, 0.314, 0.973],  # N - blue
    8: [1.000, 0.051, 0.051],  # O - red
    9: [0.565, 0.878, 0.314],  # F - green
    15: [1.000, 0.502, 0.000], # P - orange
    16: [1.000, 1.000, 0.188], # S - yellow
}


def load_model(checkpoint_dir):
    """Load trained model."""
    checkpoint_dir = Path(checkpoint_dir)
    
    config_file = checkpoint_dir / 'model_config.pkl'
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from trainer import JointPhysNetDCMNet
    
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config'],
        mix_coulomb_energy=config.get('mix_coulomb_energy', False)
    )
    
    params_file = checkpoint_dir / 'best_params.pkl'
    with open(params_file, 'rb') as f:
        params = pickle.load(f)
    
    return model, params, config


def evaluate_molecule(atoms, model, params, config, n_surface_points=5000):
    """Evaluate model and compute charges + ESP."""
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
    
    # Extract charges
    atomic_charges = np.array(output['charges_as_mono'][:natoms_actual])
    
    # Extract distributed charges
    n_dcm = config['dcmnet_config']['n_dcm']
    # mono_dist: (batch*natoms, n_dcm) - charge values
    # dipo_dist: (batch*natoms, n_dcm, 3) - charge positions
    mono_values = np.array(output['mono_dist'][:natoms_actual, :])  # (natoms, n_dcm)
    charge_positions = np.array(output['dipo_dist'][:natoms_actual, :, :])  # (natoms, n_dcm, 3)
    
    distributed_mono = charge_positions.reshape(-1, 3)
    distributed_mono_values = mono_values.reshape(-1)
    
    # Generate VDW surface
    vdw_surface, esp_surface = compute_esp_on_vdw_surface(
        atoms, distributed_mono, distributed_mono_values, n_points=n_surface_points
    )
    
    return {
        'atomic_charges': atomic_charges,
        'distributed_mono': distributed_mono,
        'distributed_mono_values': distributed_mono_values,
        'vdw_surface': vdw_surface,
        'esp_surface': esp_surface,
        'energy': float(output['energy'][0]),
    }


def compute_esp_on_vdw_surface(atoms, charge_positions, charge_values, n_points=5000, scale=1.4):
    """Compute ESP on molecular VDW surface."""
    
    # VDW radii (Angstroms)
    vdw_radii = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 15: 1.80, 16: 1.80}
    
    # Fibonacci sphere
    indices = np.arange(n_points)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    
    y = 1 - (indices / float(n_points - 1)) * 2
    radius = np.sqrt(1 - y * y)
    theta = phi * indices
    
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    sphere_points = np.stack([x, y, z], axis=1)
    
    # Create surface
    surface_points = []
    atom_positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    
    for pos, Z in zip(atom_positions, atomic_numbers):
        vdw_r = vdw_radii.get(Z, 1.7) * scale
        atom_surface = sphere_points * vdw_r + pos
        surface_points.append(atom_surface)
    
    surface_points = np.vstack(surface_points)
    
    # Compute ESP: V = sum q_i / r_i (atomic units)
    esp_values = np.zeros(len(surface_points))
    
    for q, r_q in zip(charge_values, charge_positions):
        diff = surface_points - r_q
        distances = np.linalg.norm(diff, axis=1)
        mask = distances > 0.01
        esp_values[mask] += q / (distances[mask] * 1.88973)  # Å to Bohr
    
    return surface_points, esp_values


def write_ase_povray_with_charges(atoms, result, output_path, 
                                  show_molecule='full', show_charges=False, show_esp=False,
                                  rotation='0x,0y,0z', canvas_width=2400,
                                  charge_radius=0.15, transparency=0.5):
    """
    Write POV-Ray scene using ASE's writer, then add charges and ESP.
    
    Parameters
    ----------
    show_molecule : str
        'full' - full ball-and-stick (default)
        'wireframe' - bonds only, small atoms
        'none' - no molecule (for ESP-only plots)
    show_charges : bool
        Show distributed charges as colored spheres
    show_esp : bool
        Show ESP on VDW surface
    """
    
    # Determine molecule rendering style
    if show_molecule == 'wireframe':
        # Small atoms, emphasis on bonds
        radii = [0.15] * len(atoms)  # Very small atoms for wireframe
    elif show_molecule == 'none':
        # Invisible atoms for ESP-only
        radii = [0.01] * len(atoms)
    else:  # 'full'
        # Standard ball-and-stick
        radii = [0.4] * len(atoms)
    
    # Colors using jmol scheme
    colors = []
    for Z in atoms.get_atomic_numbers():
        color = JMOL_COLORS.get(Z, [0.5, 0.5, 0.5])
        colors.append(color)
    
    # Determine bonds
    from ase.neighborlist import natural_cutoffs, NeighborList
    cutoffs = natural_cutoffs(atoms, mult=1.1)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    
    bondatoms = []
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j in indices:
            if i < j:  # Avoid duplicates
                bondatoms.append((i, j))
    
    povray_settings = {
        'transparent': False,
        'canvas_width': canvas_width,
        'camera_dist': 10.0,
        'camera_type': 'orthographic',
        'point_lights': [],  # We'll add custom lights
        'area_light': [(2., 3., 40.), 'White', .7, .7, 3, 3],
        'background': 'White',
        'textures': ['jmol'] * len(atoms),
        'celllinewidth': 0.0,
        'bondatoms': bondatoms,  # Bonds go in povray_settings
    }
    
    # Write base molecule
    ase.io.write(
        str(output_path),
        atoms,
        format='pov',
        rotation=rotation,
        radii=radii,
        colors=colors,
        povray_settings=povray_settings,
    )
    
    # Now append charges and ESP to the POV file
    with open(output_path, 'a') as f:
        f.write("\n\n// Distributed Charges and ESP\n")
        f.write("// Added by ase_povray_viz.py\n\n")
        
        # Charge finish
        f.write("""
#declare charge_texture = texture {
    pigment { rgbf <1, 1, 1, 0.5> }
    finish {
        ambient 0.3
        diffuse 0.6
        specular 0.4
        roughness 0.01
        phong 0.5
        phong_size 40
    }
}

#declare esp_texture = texture {
    finish {
        ambient 0.4
        diffuse 0.5
        specular 0.2
    }
}

""")
        
        # Add distributed charges
        if show_charges and 'distributed_mono' in result:
            charge_pos = result['distributed_mono']
            charge_vals = result['distributed_mono_values']
            
            # Color mapping
            vmax = max(abs(charge_vals.min()), abs(charge_vals.max()), 0.01)
            norm = Normalize(vmin=-vmax, vmax=vmax)
            cmap = cm.get_cmap('RdBu_r')
            
            for pos, q in zip(charge_pos, charge_vals):
                if abs(q) < 0.001:
                    continue
                
                # Size proportional to charge
                radius = charge_radius * min(abs(q) * 3 + 0.1, 0.4)
                
                rgba = cmap(norm(q))
                r, g, b = rgba[:3]
                
                f.write(f"sphere {{\n")
                f.write(f"  <{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}>, {radius:.4f}\n")
                f.write(f"  texture {{\n")
                f.write(f"    pigment {{ rgbf <{r:.3f}, {g:.3f}, {b:.3f}, {transparency:.2f}> }}\n")
                f.write(f"    finish {{\n")
                f.write(f"      ambient 0.3\n")
                f.write(f"      diffuse 0.6\n")
                f.write(f"      specular 0.4\n")
                f.write(f"      roughness 0.01\n")
                f.write(f"    }}\n")
                f.write(f"  }}\n")
                f.write(f"}}\n\n")
        
        # Add ESP surface
        if show_esp and 'vdw_surface' in result:
            surface_pos = result['vdw_surface']
            esp_vals = result['esp_surface']
            
            # Color mapping for ESP
            vmax_esp = max(abs(np.percentile(esp_vals, 5)),
                          abs(np.percentile(esp_vals, 95)), 0.001)
            norm_esp = Normalize(vmin=-vmax_esp, vmax=vmax_esp)
            cmap_esp = cm.get_cmap('RdBu_r')
            
            # Subsample for performance
            step = max(1, len(surface_pos) // 3000)
            
            for pos, esp in zip(surface_pos[::step], esp_vals[::step]):
                rgba = cmap_esp(norm_esp(esp))
                r, g, b = rgba[:3]
                
                f.write(f"sphere {{\n")
                f.write(f"  <{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}>, 0.06\n")
                f.write(f"  texture {{\n")
                f.write(f"    pigment {{ rgbf <{r:.3f}, {g:.3f}, {b:.3f}, 0.7> }}\n")
                f.write(f"    finish {{\n")
                f.write(f"      ambient 0.5\n")
                f.write(f"      diffuse 0.4\n")
                f.write(f"    }}\n")
                f.write(f"  }}\n")
                f.write(f"}}\n\n")


def render_povray(pov_file, output_image, width=2400, height=1800, quality=11, antialias=True):
    """Render POV-Ray scene."""
    cmd = [
        'povray',
        f'+I{pov_file}',
        f'+O{output_image}',
        f'+W{width}',
        f'+H{height}',
        f'+Q{quality}',
        '-D',  # No display
    ]
    
    if antialias:
        cmd.extend(['+A0.1', '+AM2'])  # Anti-aliasing
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   ✓ Rendered: {output_image.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ✗ POV-Ray error:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("\n✗ POV-Ray not found!")
        print("   Install with:")
        print("   • Ubuntu/Debian: sudo apt install povray")
        print("   • macOS: brew install povray")
        print("   • Download: http://www.povray.org/")
        return False


def create_colorbar(output_path, vmin, vmax, label, cmap='RdBu_r', horizontal=True):
    """Create standalone colorbar."""
    if horizontal:
        fig, ax = plt.subplots(figsize=(8, 0.8))
        orientation = 'horizontal'
    else:
        fig, ax = plt.subplots(figsize=(1.5, 6))
        orientation = 'vertical'
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation=orientation
    )
    cb.set_label(label, fontsize=16, weight='bold')
    cb.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='High-quality POV-Ray visualization with ASE'
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--structure', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./figures_povray')
    parser.add_argument('--plot-types', nargs='+',
                       default=['molecule', 'molecule+charges', 'esp', 'molecule+esp'],
                       help='Types of plots to generate: molecule, molecule+charges, esp, molecule+esp, charges, all')
    parser.add_argument('--molecule-style', choices=['full', 'wireframe'],
                       default='full',
                       help='Molecule rendering style for molecule+charges (default: full for standalone, wireframe for +charges)')
    parser.add_argument('--charge-radius', type=float, default=0.15)
    parser.add_argument('--transparency', type=float, default=0.5)
    parser.add_argument('--resolution', choices=['low', 'medium', 'high', 'ultra'],
                       default='high')
    parser.add_argument('--views', nargs='+',
                       default=['0x,0y,0z', '90x,0y,0z', '0x,90y,0z'])
    parser.add_argument('--n-surface-points', type=int, default=5000)
    
    args = parser.parse_args()
    
    # Resolution settings
    resolutions = {
        'low': (800, 600, 9),
        'medium': (1600, 1200, 10),
        'high': (2400, 1800, 11),
        'ultra': (3840, 2160, 11),
    }
    width, height, quality = resolutions[args.resolution]
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("🎨 ASE + POV-Ray High-Quality Molecular Visualization")
    print("="*70)
    
    # Load
    print(f"\n📂 Loading structure: {args.structure}")
    atoms = read(args.structure)
    print(f"   • {len(atoms)} atoms: {atoms.get_chemical_formula()}")
    
    print(f"\n🔧 Loading model: {args.checkpoint}")
    model, params, config = load_model(args.checkpoint)
    # Handle both 'F' and 'features' keys
    physnet_features = config['physnet_config'].get('F', config['physnet_config'].get('features', 'N/A'))
    print(f"   • PhysNet features: {physnet_features}")
    print(f"   • DCM sites per atom: {config['dcmnet_config']['n_dcm']}")
    
    # Evaluate
    print(f"\n⚡ Computing charges and ESP...")
    result = evaluate_molecule(atoms, model, params, config, 
                               n_surface_points=args.n_surface_points)
    
    print(f"   • Energy: {result['energy']:.6f} eV")
    print(f"   • Total charge (atomic): {result['atomic_charges'].sum():.6f} e")
    print(f"   • Total charge (distributed): {result['distributed_mono_values'].sum():.6f} e")
    print(f"   • Charge sites: {len(result['distributed_mono'])}")
    print(f"   • Charge range: [{result['distributed_mono_values'].min():.4f}, "
          f"{result['distributed_mono_values'].max():.4f}] e")
    
    if 'esp_surface' in result:
        print(f"   • ESP surface points: {len(result['esp_surface'])}")
        print(f"   • ESP range: [{result['esp_surface'].min():.4f}, "
              f"{result['esp_surface'].max():.4f}] Ha/e")
    
    # Render views
    print(f"\n🎬 Rendering {len(args.views)} view(s) at {args.resolution} resolution...")
    print(f"   Resolution: {width}×{height} pixels")
    
    for i, rotation in enumerate(args.views):
        view_name = f"view_{i+1}_{rotation.replace(',', '_')}"
        pov_file = output_dir / f'{view_name}.pov'
        png_file = output_dir / f'{view_name}.png'
        
        print(f"\n   📐 View {i+1}: {rotation}")
        print(f"      Writing POV scene...")
        
        write_ase_povray_with_charges(
            atoms, result, pov_file,
            show_charges=args.show_charges,
            show_esp=args.show_esp,
            rotation=rotation,
            canvas_width=width,
            charge_radius=args.charge_radius,
            transparency=args.transparency,
        )
        
        print(f"      Rendering (this may take a minute)...")
        success = render_povray(pov_file, png_file, width, height, quality)
        
        if not success:
            print(f"      ⚠ Skipping remaining renders")
            break
    
    # Create colorbars
    print(f"\n📊 Creating colorbars...")
    
    if args.show_charges:
        vmax_q = max(abs(result['distributed_mono_values'].min()),
                    abs(result['distributed_mono_values'].max()))
        create_colorbar(
            output_dir / 'colorbar_charges.png',
            -vmax_q, vmax_q,
            'Charge (e)',
            cmap='RdBu_r'
        )
        print(f"   ✓ Charge colorbar saved")
    
    if args.show_esp:
        vmax_esp = max(abs(np.percentile(result['esp_surface'], 5)),
                      abs(np.percentile(result['esp_surface'], 95)))
        create_colorbar(
            output_dir / 'colorbar_esp.png',
            -vmax_esp, vmax_esp,
            'ESP (Ha/e)',
            cmap='RdBu_r'
        )
        print(f"   ✓ ESP colorbar saved")
    
    print("\n" + "="*70)
    print(f"✅ Complete! Figures saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

