#!/usr/bin/env python3
"""
Advanced POV-Ray Visualization using ASE's Native POV-Ray Writer

Creates publication-quality figures with:
- Ball-and-stick molecular structure (using ASE's POV-Ray writer)
- Distributed charges as colored translucent spheres
- ESP on VDW surface as colored mesh
- Multiple camera orientations per run with optional composite outputs
- Soft, semi-transparent molecule styling with optional grid floor for scale cues

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
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import jax
import jax.numpy as jnp
from ase.io import read, write
from ase import Atoms
import ase.io

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # Pillow is optional; composite output requires it
    Image = ImageDraw = ImageFont = None


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
                                  charge_radius=0.15, transparency=0.5,
                                  molecule_transparency=0.5, camera_zoom=1.0):
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
    camera_zoom : float
        Camera zoom factor (< 1.0 zooms out, > 1.0 zooms in)
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
    
    # Calculate bounding box for better camera distance
    positions = atoms.get_positions()
    center = positions.mean(axis=0)
    max_extent = np.max(np.linalg.norm(positions - center, axis=1))
    camera_distance = max(15.0, max_extent * 4.0) / camera_zoom  # Adjustable zoom

    grid_offset = center[1] - max_extent - 0.75
    grid_spacing = max_extent if max_extent > 0 else 1.0
    grid_spacing = max(0.5, min(2.5, grid_spacing))

    bbox_padding = 1.0
    bbox_min = positions.min(axis=0) - bbox_padding
    bbox_max = positions.max(axis=0) + bbox_padding
    base_radius = max_extent * 0.025 if max_extent > 0 else 0.025
    bbox_radius = max(0.018, min(0.055, base_radius))

    bbox_corners = []
    for idx in range(8):
        x = bbox_min[0] if (idx & 1) == 0 else bbox_max[0]
        y = bbox_min[1] if (idx & 2) == 0 else bbox_max[1]
        z = bbox_min[2] if (idx & 4) == 0 else bbox_max[2]
        bbox_corners.append((x, y, z))

    edge_pairs = []
    for axis in range(3):
        bit = 1 << axis
        for idx in range(8):
            if (idx & bit) == 0:
                edge_pairs.append((idx, idx | bit))

    bbox_lines = []
    bbox_lines.append("#declare bbox_frame_texture = texture {\n")
    bbox_lines.append("    pigment { rgbf <0.70, 0.72, 0.78, 0.80> }\n")
    bbox_lines.append("    finish { ambient 0.08 diffuse 0.25 specular 0.04 roughness 0.08 }\n")
    bbox_lines.append("};\n\n")

    for start_idx, end_idx in edge_pairs:
        start = bbox_corners[start_idx]
        end = bbox_corners[end_idx]
        bbox_lines.append("cylinder {\n")
        bbox_lines.append(f"    <{start[0]:.4f}, {start[1]:.4f}, {start[2]:.4f}>, <{end[0]:.4f}, {end[1]:.4f}, {end[2]:.4f}>, {bbox_radius:.4f}\n")
        bbox_lines.append("    texture { bbox_frame_texture }\n")
        bbox_lines.append("}\n")

    corner_radius = bbox_radius * 1.35
    for corner in bbox_corners:
        bbox_lines.append("sphere {\n")
        bbox_lines.append(f"    <{corner[0]:.4f}, {corner[1]:.4f}, {corner[2]:.4f}>, {corner_radius:.4f}\n")
        bbox_lines.append("    texture { bbox_frame_texture }\n")
        bbox_lines.append("}\n")

    bbox_geometry = ''.join(bbox_lines)
    
    povray_settings = {
        'transparent': False,
        'canvas_width': canvas_width,
        'camera_dist': camera_distance,  # Dynamic based on molecule size
        'camera_type': 'orthographic',
        'point_lights': [],  # We'll add custom soft area lights
        # Don't include area_light to avoid ASE's default
        'background': 'White',
        'textures': ['jmol'] * len(atoms),
        'celllinewidth': 0.0,
        'bondatoms': bondatoms,
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
    
    apply_molecule_transparency(output_path, molecule_transparency)

    # Now append custom lighting, materials, and charges/ESP to the POV file
    with open(output_path, 'a') as f:
        f.write("\n\n// Enhanced rendering settings\n")
        f.write("// Added by ase_povray_viz.py\n\n")
        
        lighting_block = """
// Soft lighting environment
#declare light_color = rgb <1.0, 1.0, 1.0>;

// Main light (broad, gentle)
light_source {
    <24, 36, 42>
    color light_color * 0.45
    area_light <12, 0, 0>, <0, 12, 0>, 6, 6
    adaptive 1
    jitter
    fade_distance 75
    fade_power 2
}

// Fill light (diffuse bounce)
light_source {
    <-18, 22, -28>
    color light_color * 0.25
    area_light <10, 0, 0>, <0, 10, 0>, 5, 5
    adaptive 1
    jitter
    fade_distance 65
    fade_power 2
}

// Rim light (very soft accent)
light_source {
    <-8, -12, -35>
    color light_color * 0.18
    area_light <6, 0, 0>, <0, 6, 0>, 4, 4
    adaptive 1
    jitter
}

// Ambient illumination
global_settings {
    ambient_light rgb <0.20, 0.20, 0.22>
    max_trace_level 15
}

// Sky tint for subtle background colour
sky_sphere {
    pigment {
        gradient y
        color_map {
            [0.0 color rgb <0.92, 0.95, 0.98>]
            [1.0 color rgb <1.0, 1.0, 1.0>]
        }
    }
}
"""
        f.write(lighting_block)
        f.write("\n// Bounding box frame for scale reference\n")
        f.write(bbox_geometry)
        f.write("\n// Softer finish settings\n")
        f.write("""
#declare charge_finish = finish {
    ambient 0.25
    diffuse 0.55
    specular 0.35
    roughness 0.02
    metallic 0.05
    phong 0.35
    phong_size 50
    brilliance 1.05
}

#declare esp_finish = finish {
    ambient 0.28
    diffuse 0.55
    specular 0.18
    roughness 0.05
    brilliance 1.0
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
                pastel = 0.6
                r = r * pastel + (1 - pastel)
                g = g * pastel + (1 - pastel)
                b = b * pastel + (1 - pastel)
                alpha = min(0.9, max(0.1, transparency + 0.2))
                
                f.write(f"sphere {{\n")
                f.write(f"  <{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}>, {radius:.4f}\n")
                f.write(f"  texture {{\n")
                f.write(f"    pigment {{ rgbf <{r:.3f}, {g:.3f}, {b:.3f}, {alpha:.2f}> }}\n")
                f.write(f"    finish {{ charge_finish }}\n")
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
                pastel = 0.7
                r = r * pastel + (1 - pastel)
                g = g * pastel + (1 - pastel)
                b = b * pastel + (1 - pastel)
                
                f.write(f"sphere {{\n")
                f.write(f"  <{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}>, 0.06\n")
                f.write(f"  texture {{\n")
                f.write(f"    pigment {{ rgbf <{r:.3f}, {g:.3f}, {b:.3f}, 0.75> }}\n")
                f.write(f"    finish {{ esp_finish }}\n")
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


def create_composite_row(image_paths, output_path, labels=None, padding=40, background_color=(255, 255, 255)):
    """Create a horizontal composite of images with optional labels."""
    if not image_paths:
        raise ValueError('No images provided for composite creation.')
    if Image is None:
        raise RuntimeError('Pillow is required for composite rendering (pip install pillow).')

    opened_images = [Image.open(path) for path in image_paths]
    try:
        widths = [img.width for img in opened_images]
        heights = [img.height for img in opened_images]

        total_width = sum(widths) + padding * (len(opened_images) - 1)
        max_height = max(heights)

        composite = Image.new('RGB', (total_width, max_height), color=background_color)
        draw = ImageDraw.Draw(composite) if labels else None
        font = None
        if labels:
            try:
                font = ImageFont.truetype('DejaVuSans.ttf', 36)
            except Exception:
                font = ImageFont.load_default()

        x_offset = 0
        for idx, img in enumerate(opened_images):
            y_offset = (max_height - img.height) // 2
            composite.paste(img, (x_offset, y_offset))

            if labels and draw:
                label = labels[idx]
                text = label.replace(',', '  ')
                if hasattr(draw, 'textbbox'):
                    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                    text_width = right - left
                    text_height = bottom - top
                else:
                    text_width, text_height = draw.textsize(text, font=font)
                text_x = int(x_offset + (img.width - text_width) // 2)
                text_y = int(max(10, y_offset + 10))

                # Draw semi-transparent rectangle for readability
                rect_padding = 10
                rect_left = max(0, int(text_x - rect_padding))
                rect_top = max(0, int(text_y - rect_padding))
                rect_right = min(composite.width, int(text_x + text_width + rect_padding))
                rect_bottom = min(composite.height, int(text_y + text_height + rect_padding))
                overlay_width = max(1, rect_right - rect_left)
                overlay_height = max(1, rect_bottom - rect_top)
                overlay = Image.new('RGBA', (overlay_width, overlay_height), (255, 255, 255, 180))
                composite.paste(overlay, (rect_left, rect_top), overlay)
                draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

            x_offset += img.width + padding

        composite.save(output_path)
    finally:
        for img in opened_images:
            img.close()

    return output_path


def apply_molecule_transparency(pov_path: Path, transparency: float) -> None:
    """Inject uniform transparency for atoms and bonds in the base POV file."""
    target = Path(pov_path)
    if not target.exists():
        return

    transparency = max(0.0, min(1.0, transparency))
    with target.open('r') as fh:
        content = fh.read()

    split_marker = "// Enhanced rendering settings"
    if split_marker in content:
        header, tail = content.split(split_marker, 1)
    else:
        header, tail = content, ''

    trans_str = f"{transparency:.3f}"

    header = header.replace('transmit TRANS', f'transmit {trans_str}')

    def replace_transmit(match):
        try:
            value = float(match.group(2))
        except ValueError:
            return match.group(0)
        if value <= 0.95:
            return match.group(1) + trans_str
        return match.group(0)

    header = re.sub(r'(transmit\s+)([-+]?\d*\.?\d+)', replace_transmit, header)

    def replace_atom_arg(match):
        try:
            value = float(match.group(2))
        except ValueError:
            return match.group(0)
        if value <= 0.95:
            return match.group(1) + trans_str + match.group(3)
        return match.group(0)

    header = re.sub(r'(atom\([^,]+,[^,]+,[^,]+,\s*)([-+]?\d*\.?\d+)(\s*,)', replace_atom_arg, header)

    updated_content = header + (split_marker + tail if tail else '')

    if updated_content != content:
        with target.open('w') as fh:
            fh.write(updated_content)


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
    parser.add_argument('--molecule-transparency', type=float, default=0.5,
                       help='Uniform transparency applied to atoms and bonds (0 = opaque, 1 = invisible)')
    parser.add_argument('--camera-zoom', type=float, default=1.0,
                       help='Camera zoom factor (< 1.0 zooms out, > 1.0 zooms in)')
    parser.add_argument('--resolution', choices=['low', 'medium', 'high', 'ultra'],
                       default='high')
    parser.add_argument('--views', nargs='+', default=None,
                       metavar='ROTATION',
                       help=('Legacy option to provide multiple rotation strings, '
                             'e.g., --views 0x,0y,0z 90x,0y,0z'))
    parser.add_argument('--view', dest='view_list', action='append', default=None,
                       metavar='ROTATION',
                       help=('Add a single view rotation (can be repeated). '
                             'Format: "ax,by,cz" in degrees, e.g., --view 0x,0y,0z'))
    parser.add_argument('--n-surface-points', type=int, default=5000)
    parser.add_argument('--composite-row', action='store_true',
                       help='Combine rendered views for each plot into a single horizontal composite image')
    
    args = parser.parse_args()
    
    # Resolution settings
    resolutions = {
        'low': (800, 600, 9),
        'medium': (1600, 1200, 10),
        'high': (2400, 1800, 11),
        'ultra': (3840, 2160, 11),
    }
    width, height, quality = resolutions[args.resolution]

    default_views = ['0x,0y,0z', '90x,0y,0z', '0x,90y,0z']
    if args.view_list and args.views:
        print("\n⚠ Both --view and --views provided; using values from --view.")
    if args.view_list:
        views = args.view_list
    elif args.views:
        views = args.views
    else:
        views = default_views
    
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
    
    # Parse plot types
    plot_types = []
    if 'all' in args.plot_types:
        plot_types = ['molecule', 'molecule+charges', 'esp', 'molecule+esp', 'charges']
    else:
        plot_types = args.plot_types
    
    # Render views
    print(f"\n🎬 Rendering {len(views)} view(s) × {len(plot_types)} plot type(s)")
    print(f"   Resolution: {width}×{height} pixels")
    print(f"   Plot types: {', '.join(plot_types)}")
    
    rendered_images = {}

    for i, rotation in enumerate(views):
        print(f"\n   📐 View {i+1}/{len(views)}: {rotation}")
        
        for plot_type in plot_types:
            # Determine rendering parameters based on plot type
            if plot_type == 'molecule':
                show_mol = 'full'
                show_charges = False
                show_esp_flag = False
                suffix = 'molecule'
            elif plot_type == 'molecule+charges':
                show_mol = args.molecule_style  # User choice or wireframe
                show_charges = True
                show_esp_flag = False
                suffix = 'molecule_charges'
            elif plot_type == 'charges':
                show_mol = 'none'  # No molecule
                show_charges = True
                show_esp_flag = False
                suffix = 'charges_only'
            elif plot_type == 'esp':
                show_mol = 'none'
                show_charges = False
                show_esp_flag = True
                suffix = 'esp'
            elif plot_type == 'molecule+esp':
                show_mol = 'wireframe'  # Wireframe for ESP overlay
                show_charges = False
                show_esp_flag = True
                suffix = 'molecule_esp'
            else:
                print(f"      ⚠ Unknown plot type: {plot_type}")
                continue
            
            view_name = f"view_{i+1}_{rotation.replace(',', '_')}_{suffix}"
            pov_file = output_dir / f'{view_name}.pov'
            png_file = output_dir / f'{view_name}.png'
            
            print(f"      • {plot_type:20s} → {png_file.name}")
            
            write_ase_povray_with_charges(
                atoms, result, pov_file,
                show_molecule=show_mol,
                show_charges=show_charges,
                show_esp=show_esp_flag,
                rotation=rotation,
                canvas_width=width,
                charge_radius=args.charge_radius,
                transparency=args.transparency,
                molecule_transparency=args.molecule_transparency,
                camera_zoom=args.camera_zoom,
            )
            
            success = render_povray(pov_file, png_file, width, height, quality, antialias=True)
            if success:
                rendered_images.setdefault(suffix, []).append((rotation, png_file))
            else:
                print(f"      ⚠ POV-Ray failed, skipping remaining renders")
                return 1
    
    if args.composite_row:
        if Image is None:
            print("\n✗ Pillow not found — install with `pip install pillow` to enable composite rows.")
        else:
            print("\n🧩 Building composite rows...")
            for suffix, entries in rendered_images.items():
                if not entries:
                    continue

                rotations, paths = zip(*entries)
                composite_name = f'composite_{suffix}.png'
                composite_path = output_dir / composite_name
                try:
                    create_composite_row(paths, composite_path, labels=list(rotations))
                    print(f"   ✓ Composite saved: {composite_name}")
                except Exception as exc:
                    print(f"   ✗ Failed to create composite for {suffix}: {exc}")

    # Create colorbars
    print(f"\n📊 Creating colorbars...")
    
    if any('charges' in pt for pt in plot_types):
        vmax_q = max(abs(result['distributed_mono_values'].min()),
                    abs(result['distributed_mono_values'].max()))
        create_colorbar(
            output_dir / 'colorbar_charges.png',
            -vmax_q, vmax_q,
            'Charge (e)',
            cmap='RdBu_r'
        )
        print(f"   ✓ Charge colorbar saved")
    
    if any('esp' in pt for pt in plot_types):
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

