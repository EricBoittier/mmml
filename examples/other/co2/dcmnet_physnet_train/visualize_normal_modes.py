#!/usr/bin/env python3
"""
Visualize Normal Modes of Vibration

Creates visualizations of atomic motions for each vibrational frequency.
Generates trajectory files and vector plots for normal modes.

Usage:
    python visualize_normal_modes.py --raman-dir ./raman_analysis --output-dir ./mode_animations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
from ase.io import read, write
from ase.vibrations import Vibrations
import warnings

# Okabe-Ito palette
COLORS = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9', 
    'bluish_green': '#009E73',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'black': '#000000'
}


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def create_mode_animation(atoms, mode_vector, frequency, n_frames=30, amplitude=0.5):
    """
    Create trajectory showing one complete oscillation of a normal mode.
    
    Parameters
    ----------
    atoms : Atoms
        Equilibrium geometry
    mode_vector : array (natoms, 3)
        Normal mode displacement vectors
    frequency : float
        Vibrational frequency (cm⁻¹)
    n_frames : int
        Number of frames in animation
    amplitude : float
        Oscillation amplitude (Å)
    
    Returns
    -------
    traj : list of Atoms
        Trajectory showing the vibration
    """
    traj = []
    
    for i in range(n_frames):
        # Phase of oscillation
        phase = 2 * np.pi * i / n_frames
        
        # Displaced positions
        displaced_atoms = atoms.copy()
        displacement = amplitude * np.sin(phase) * mode_vector
        displaced_atoms.positions += displacement
        
        traj.append(displaced_atoms)
    
    return traj


def plot_mode_vectors_3d(atoms, mode_vector, frequency, output_file, mode_type=''):
    """
    Create 3D visualization of normal mode displacement vectors.
    
    Parameters
    ----------
    atoms : Atoms
        Equilibrium geometry
    mode_vector : array (natoms, 3)
        Normal mode displacement vectors (normalized)
    frequency : float
        Vibrational frequency (cm⁻¹)
    output_file : Path
        Output PNG file
    mode_type : str
        Mode description (e.g., 'Raman active', 'IR active')
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get positions and atomic numbers
    pos = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    
    # Center molecule
    com = pos.mean(axis=0)
    pos_centered = pos - com
    
    # Atom colors and sizes
    atom_colors = []
    atom_sizes = []
    for num in numbers:
        if num == 6:  # Carbon
            atom_colors.append(COLORS['black'])
            atom_sizes.append(300)
        elif num == 8:  # Oxygen
            atom_colors.append(COLORS['vermillion'])
            atom_sizes.append(250)
        else:
            atom_colors.append(COLORS['blue'])
            atom_sizes.append(200)
    
    # Plot atoms
    ax.scatter(pos_centered[:, 0], pos_centered[:, 1], pos_centered[:, 2],
              c=atom_colors, s=atom_sizes, alpha=0.8, edgecolors='black', linewidths=2)
    
    # Plot displacement vectors
    scale_factor = 0.5  # Å per unit displacement
    
    for i in range(len(atoms)):
        start = pos_centered[i]
        vec = mode_vector[i] * scale_factor
        
        # Normalize and scale for visibility
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 0.01:  # Only draw significant displacements
            arrow = Arrow3D([start[0], start[0] + vec[0]],
                           [start[1], start[1] + vec[1]],
                           [start[2], start[2] + vec[2]],
                           mutation_scale=20, lw=3,
                           arrowstyle='->', color=COLORS['bluish_green'], alpha=0.8)
            ax.add_artist(arrow)
    
    # Labels
    ax.set_xlabel('X (Å)', fontsize=11, weight='bold')
    ax.set_ylabel('Y (Å)', fontsize=11, weight='bold')
    ax.set_zlabel('Z (Å)', fontsize=11, weight='bold')
    
    title = f'Normal Mode: {frequency:.1f} cm⁻¹'
    if mode_type:
        title += f' ({mode_type})'
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    
    # Equal aspect ratio
    max_range = np.array([pos_centered[:, 0].max() - pos_centered[:, 0].min(),
                         pos_centered[:, 1].max() - pos_centered[:, 1].min(),
                         pos_centered[:, 2].max() - pos_centered[:, 2].min()]).max() / 2.0
    mid_x = (pos_centered[:, 0].max() + pos_centered[:, 0].min()) * 0.5
    mid_y = (pos_centered[:, 1].max() + pos_centered[:, 1].min()) * 0.5
    mid_z = (pos_centered[:, 2].max() + pos_centered[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range*1.5, mid_x + max_range*1.5)
    ax.set_ylim(mid_y - max_range*1.5, mid_y + max_range*1.5)
    ax.set_zlim(mid_z - max_range*1.5, mid_z + max_range*1.5)
    
    # Isometric view
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved vector plot: {output_file}")
    plt.close()


def create_mode_comparison_plot(atoms, vib, output_file):
    """
    Create comparison plot showing all normal modes.
    """
    freqs = vib.get_frequencies()
    
    # Filter real positive frequencies
    mask = np.real(freqs) > 100
    freqs_filtered = np.real(freqs[mask])
    mode_indices = np.where(mask)[0]
    
    n_modes = len(freqs_filtered)
    if n_modes == 0:
        return
    
    # Determine grid size
    ncols = min(3, n_modes)
    nrows = int(np.ceil(n_modes / ncols))
    
    fig = plt.figure(figsize=(6*ncols, 5*nrows))
    
    for plot_idx, mode_idx in enumerate(mode_indices):
        ax = fig.add_subplot(nrows, ncols, plot_idx + 1, projection='3d')
        
        # Get mode
        mode = vib.get_mode(mode_idx)  # Returns (natoms, 3) displacement
        freq = freqs[mode_idx]
        
        # Classify mode
        mode_type = ''
        if abs(freq - 1388.2) < 50:
            mode_type = 'Raman (strong)'
            mode_color = COLORS['bluish_green']
        elif abs(freq - 2349.2) < 50:
            mode_type = 'IR (strong)'
            mode_color = COLORS['vermillion']
        elif abs(freq - 667.4) < 50:
            mode_type = 'IR + Raman'
            mode_color = COLORS['orange']
        else:
            mode_type = 'Combination/Overtone'
            mode_color = COLORS['blue']
        
        # Get positions
        pos = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()
        com = pos.mean(axis=0)
        pos_centered = pos - com
        
        # Atom colors
        atom_colors = []
        for num in numbers:
            if num == 6:
                atom_colors.append(COLORS['black'])
            elif num == 8:
                atom_colors.append(COLORS['vermillion'])
            else:
                atom_colors.append(COLORS['blue'])
        
        # Plot atoms (smaller)
        ax.scatter(pos_centered[:, 0], pos_centered[:, 1], pos_centered[:, 2],
                  c=atom_colors, s=150, alpha=0.7, edgecolors='black', linewidths=1)
        
        # Plot displacement vectors
        scale = 0.3
        for i in range(len(atoms)):
            start = pos_centered[i]
            vec = mode[i] * scale
            
            if np.linalg.norm(vec) > 0.01:
                arrow = Arrow3D([start[0], start[0] + vec[0]],
                               [start[1], start[1] + vec[1]],
                               [start[2], start[2] + vec[2]],
                               mutation_scale=15, lw=2.5,
                               arrowstyle='->', color=mode_color, alpha=0.9)
                ax.add_artist(arrow)
        
        # Minimal labels
        ax.set_title(f'{freq:.1f} cm⁻¹\n{mode_type}', 
                    fontsize=11, weight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=45)
        
        # Equal aspect
        max_range = 1.5
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
    
    plt.suptitle('Normal Mode Displacement Vectors', 
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved mode comparison: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize normal mode atomic motions')
    parser.add_argument('--raman-dir', type=Path, required=True,
                       help='Directory containing Raman analysis')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory (default: <raman-dir>/mode_visualizations)')
    parser.add_argument('--create-trajectories', action='store_true',
                       help='Create .traj animation files for each mode')
    parser.add_argument('--amplitude', type=float, default=0.5,
                       help='Animation amplitude (Å)')
    parser.add_argument('--frames', type=int, default=30,
                       help='Number of frames per oscillation')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.raman_dir / 'mode_visualizations'
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("NORMAL MODE VISUALIZATION")
    print("="*70)
    print(f"\nSettings:")
    print(f"  Raman dir:   {args.raman_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Trajectories: {args.create_trajectories}")
    if args.create_trajectories:
        print(f"  Amplitude:    {args.amplitude} Å")
        print(f"  Frames:       {args.frames}")
    
    # Load geometry
    print(f"\n📂 Loading geometry...")
    xyz_file = args.raman_dir / 'CO2_optimized.xyz'
    if not xyz_file.exists():
        # Try parent directory
        xyz_file = args.raman_dir.parent / 'CO2_optimized.xyz'
    
    if xyz_file.exists():
        atoms = read(xyz_file)
        print(f"   ✓ Loaded from {xyz_file}")
    else:
        from ase import Atoms
        atoms = Atoms('CO2', positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
        print(f"   ⚠️  Using default CO₂ geometry")
    
    # Load vibrations
    print(f"\n📂 Loading vibrations...")
    vib_cache = args.raman_dir / 'vibrations'
    if not vib_cache.exists():
        print(f"   ❌ No vibrations directory found: {vib_cache}")
        return
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vib = Vibrations(atoms, name=str(vib_cache / 'vib'))
    
    try:
        freqs = vib.get_frequencies()
        print(f"   ✓ Loaded {len(freqs)} vibrational modes")
    except Exception as e:
        print(f"   ❌ Could not load frequencies: {e}")
        return
    
    # Filter real positive frequencies
    mask = np.real(freqs) > 100
    freqs_filtered = np.real(freqs[mask])
    mode_indices = np.where(mask)[0]
    
    print(f"\n📊 Found {len(freqs_filtered)} physical modes:")
    for mode_idx, freq in zip(mode_indices, freqs_filtered):
        # Classify mode
        if abs(freq - 1388.2) < 50:
            mode_type = 'ν₁ Symmetric Stretch (Raman strong)'
        elif abs(freq - 2349.2) < 50:
            mode_type = 'ν₃ Asymmetric Stretch (IR strong)'
        elif abs(freq - 667.4) < 50:
            mode_type = 'ν₂ Bending (IR + Raman weak)'
        else:
            mode_type = 'Combination/Overtone'
        
        print(f"  Mode {mode_idx}: {freq:7.1f} cm⁻¹ - {mode_type}")
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    
    # Overall comparison plot
    print(f"  Creating mode comparison plot...")
    create_mode_comparison_plot(atoms, vib, args.output_dir / 'all_modes.png')
    
    # Individual mode plots and trajectories
    for mode_idx, freq in zip(mode_indices, freqs_filtered):
        mode = vib.get_mode(mode_idx)
        
        # Classify
        if abs(freq - 1388.2) < 50:
            mode_name = 'v1_symmetric_stretch'
            mode_type = 'Raman (strong)'
        elif abs(freq - 2349.2) < 50:
            mode_name = 'v3_asymmetric_stretch'
            mode_type = 'IR (strong)'
        elif abs(freq - 667.4) < 50:
            mode_name = 'v2_bending'
            mode_type = 'IR + Raman'
        else:
            mode_name = f'mode_{mode_idx}'
            mode_type = 'Other'
        
        # Vector plot
        output_vec = args.output_dir / f'{mode_name}_{freq:.0f}cm_vectors.png'
        plot_mode_vectors_3d(atoms, mode, freq, output_vec, mode_type)
        
        # Trajectory animation
        if args.create_trajectories:
            traj = create_mode_animation(atoms, mode, freq, 
                                        n_frames=args.frames, 
                                        amplitude=args.amplitude)
            traj_file = args.output_dir / f'{mode_name}_{freq:.0f}cm_animation.traj'
            write(traj_file, traj)
            print(f"  ✓ Saved trajectory: {traj_file.name}")
    
    print(f"\n{'='*70}")
    print(f"✅ VISUALIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"\nFiles created:")
    print(f"  all_modes.png - Overview of all modes")
    print(f"  *_vectors.png - 3D displacement vectors for each mode")
    if args.create_trajectories:
        print(f"  *_animation.traj - Animation trajectories (view with 'ase gui')")
        print(f"\n💡 View animations:")
        print(f"   ase gui {args.output_dir}/*.traj")
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()

