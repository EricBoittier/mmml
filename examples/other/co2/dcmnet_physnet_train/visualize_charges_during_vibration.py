#!/usr/bin/env python3
"""
Visualize Distributed Charges During Vibrational Motion

Shows how distributed multipole charges evolve during normal mode vibrations.
Creates 3D visualizations and 2D projections aligned to minimum energy frame.

Usage:
    python visualize_charges_during_vibration.py \
        --checkpoint /path/to/checkpoint \
        --raman-dir ./raman_analysis \
        --mode-index 7 \
        --n-frames 20 \
        --amplitude 0.2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from ase.io import read, write
from ase.vibrations import Vibrations
import warnings

# Okabe-Ito palette
COLORS = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9', 
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7',
    'black': '#000000'
}


def load_model(checkpoint_dir):
    """Load trained model."""
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


def evaluate_frame(atoms, model, params, config, batch_size=9):
    """
    Evaluate model on a single geometry to get distributed charges.
    
    Returns
    -------
    dict with:
        - atomic_charges: (natoms,) PhysNet charges
        - distributed_mono: (natoms, n_dist, 3) monopole positions  
        - distributed_mono_values: (natoms, n_dist) monopole values
        - distributed_dipo: (natoms, n_dist, 3, 3) dipole positions
        - distributed_dipo_values: (natoms, n_dist, 3) dipole values
        - energy: float
    """
    import e3x
    
    # Prepare input
    positions = jnp.array(atoms.get_positions())
    atomic_numbers = jnp.array(atoms.get_atomic_numbers())
    
    # Pad to batch_size
    natoms_actual = len(atoms)
    natoms_padded = config['physnet_config']['natoms']
    
    if natoms_actual < natoms_padded:
        pad_size = natoms_padded - natoms_actual
        positions = jnp.concatenate([positions, jnp.zeros((pad_size, 3))], axis=0)
        atomic_numbers = jnp.concatenate([atomic_numbers, jnp.zeros(pad_size, dtype=jnp.int32)], axis=0)
    
    # Create atom mask
    atom_mask = jnp.zeros(natoms_padded, dtype=jnp.float32)
    atom_mask = atom_mask.at[:natoms_actual].set(1.0)
    
    # Create inputs
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(natoms_padded)
    dst_idx = jnp.array(dst_idx)
    src_idx = jnp.array(src_idx)
    
    # Batch info
    batch_segments = jnp.zeros(natoms_padded, dtype=jnp.int32)
    batch_mask = jnp.ones(len(dst_idx), dtype=jnp.float32)
    
    # Forward pass through joint model
    output = model.apply(
        params,  # params is already {'params': {...}}
        atomic_numbers=atomic_numbers,
        positions=positions,
        dst_idx=dst_idx,
        src_idx=src_idx,
        batch_segments=batch_segments,
        batch_size=1,
        batch_mask=batch_mask,
        atom_mask=atom_mask
    )
    
    # Extract data
    # charges: various shapes depending on model output
    charges_raw = output['charges']
    if charges_raw.ndim > 1:
        atomic_charges = np.array(charges_raw).flatten()[:natoms_actual]
    else:
        atomic_charges = np.array(charges_raw)[:natoms_actual]
    
    # energy: (batch_size,) → scalar
    energy = float(output['energy'][0])
    
    # mono_dist: (batch*natoms, n_dcm) reshape to (natoms, n_dcm)
    mono_dist_flat = output['mono_dist'][:natoms_actual]  # (natoms, n_dcm)
    # Positions are stored - need to extract from DCMNet
    # For now, use placeholder positions (around each atom)
    n_dcm = mono_dist_flat.shape[1]
    mono_values = mono_dist_flat  # These are the monopole values
    
    # Create position grid around each atom for distributed charges
    # (In reality, DCMNet determines these, but for visualization we use a simple grid)
    mono_positions = []
    for atom_idx in range(natoms_actual):
        atom_pos = atoms.get_positions()[atom_idx]
        # Create grid of points around atom (simple: radial distribution)
        angles = np.linspace(0, 2*np.pi, n_dcm, endpoint=False)
        radius = 0.3  # Å
        for i, angle in enumerate(angles):
            # Simple circular distribution
            offset = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
            mono_positions.append(atom_pos + offset)
    
    mono_positions = np.array(mono_positions).reshape(natoms_actual, n_dcm, 3)
    
    #dipole_dist: (batch*natoms, n_dcm, 3)
    dipo_values = output['dipo_dist'][:natoms_actual]  # (natoms, n_dcm, 3)
    
    return {
        'atomic_charges': np.array(atomic_charges),
        'distributed_mono': mono_positions,  # Approximate positions
        'distributed_mono_values': np.array(mono_values),
        'distributed_dipo_values': np.array(dipo_values),
        'energy': energy,
        'positions': atoms.get_positions()
    }


def create_vibration_trajectory_with_charges(atoms, mode_vector, amplitude, n_frames,
                                             model, params, config):
    """
    Create trajectory with distributed charges for each frame.
    
    Returns
    -------
    trajectory_data : list of dicts
        Each dict contains geometry and charge information
    """
    traj_data = []
    
    print(f"  Evaluating {n_frames} frames...")
    for i in range(n_frames):
        phase = 2 * np.pi * i / n_frames
        
        # Create displaced geometry
        displaced_atoms = atoms.copy()
        displacement = amplitude * np.sin(phase) * mode_vector
        displaced_atoms.positions += displacement
        
        # Evaluate model
        frame_data = evaluate_frame(displaced_atoms, model, params, config)
        frame_data['phase'] = phase
        frame_data['frame_idx'] = i
        
        traj_data.append(frame_data)
        
        if (i + 1) % 10 == 0:
            print(f"    Frame {i+1}/{n_frames}")
    
    return traj_data


def plot_charges_all_frames_3d(traj_data, atoms, output_file):
    """
    Plot all frames overlaid in 3D, aligned to minimum energy frame.
    Shows how distributed charge positions and magnitudes change.
    """
    # Find minimum energy frame
    energies = [frame['energy'] for frame in traj_data]
    min_idx = np.argmin(energies)
    
    print(f"  Aligning to frame {min_idx} (E_min = {energies[min_idx]:.6f} eV)")
    
    fig = plt.figure(figsize=(18, 6))
    
    # Define 2D cut planes
    projections = [
        ('XY', 0, 1, 2),  # Z is out of plane
        ('XZ', 0, 2, 1),  # Y is out of plane
        ('YZ', 1, 2, 0),  # X is out of plane
    ]
    
    for proj_idx, (proj_name, dim1, dim2, dim_out) in enumerate(projections):
        ax = fig.add_subplot(1, 3, proj_idx + 1)
        
        dim_names = ['X', 'Y', 'Z']
        
        # Reference frame atoms (minimum energy)
        ref_pos = traj_data[min_idx]['positions']
        ref_com = ref_pos.mean(axis=0)
        ref_pos_centered = ref_pos - ref_com
        
        # Plot reference atoms
        for atom_idx, pos in enumerate(ref_pos_centered):
            if atoms[atom_idx].number == 6:  # Carbon
                ax.plot(pos[dim1], pos[dim2], 'o', color=COLORS['black'], 
                       markersize=12, alpha=0.8, zorder=10)
            else:  # Oxygen
                ax.plot(pos[dim1], pos[dim2], 'o', color=COLORS['vermillion'], 
                       markersize=10, alpha=0.8, zorder=10)
        
        # Plot distributed charges from all frames
        all_charge_pos = []
        all_charge_vals = []
        
        for frame in traj_data:
            # Align to reference
            frame_com = frame['positions'].mean(axis=0)
            offset = ref_com - frame_com
            
            # Get distributed charges
            mono_dist = frame['distributed_mono']  # (natoms, n_dist, 3)
            mono_vals = frame['distributed_mono_values']  # (natoms, n_dist)
            
            for atom_idx in range(mono_dist.shape[0]):
                for dist_idx in range(mono_dist.shape[1]):
                    charge_pos = mono_dist[atom_idx, dist_idx] + offset
                    charge_val = mono_vals[atom_idx, dist_idx]
                    
                    all_charge_pos.append(charge_pos)
                    all_charge_vals.append(charge_val)
        
        all_charge_pos = np.array(all_charge_pos)
        all_charge_vals = np.array(all_charge_vals)
        
        # Plot charges colored by magnitude
        scatter = ax.scatter(all_charge_pos[:, dim1], all_charge_pos[:, dim2],
                           c=all_charge_vals, cmap='RdBu_r', 
                           s=20, alpha=0.3, vmin=-0.5, vmax=0.5,
                           edgecolors='none')
        
        # Labels
        ax.set_xlabel(f'{dim_names[dim1]} (Å)', fontsize=11, weight='bold')
        ax.set_ylabel(f'{dim_names[dim2]} (Å)', fontsize=11, weight='bold')
        ax.set_title(f'{proj_name} Projection\n{len(traj_data)} frames overlaid', 
                    fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3, ls='--')
        ax.set_aspect('equal')
        
        # Set limits
        max_range = 2.5
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
    
    plt.suptitle('Distributed Charge Evolution During Vibration', 
                 fontsize=16, weight='bold', y=0.96)
    
    # Add horizontal colorbar under the figure
    fig.subplots_adjust(bottom=0.12)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Monopole Charge (e)', fontsize=11, weight='bold')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved 2D projections: {output_file}")
    plt.close()


def plot_charges_3d_evolution(traj_data, atoms, output_file):
    """
    Create 3D plot showing charge cloud evolution.
    """
    # Find minimum energy frame
    energies = [frame['energy'] for frame in traj_data]
    min_idx = np.argmin(energies)
    
    fig = plt.figure(figsize=(16, 5))
    
    # Plot 1: All frames overlaid
    ax = fig.add_subplot(131, projection='3d')
    
    ref_com = traj_data[min_idx]['positions'].mean(axis=0)
    
    # Collect all charges
    all_charge_pos = []
    all_charge_vals = []
    
    for frame in traj_data:
        frame_com = frame['positions'].mean(axis=0)
        offset = ref_com - frame_com
        
        mono_dist = frame['distributed_mono']
        mono_vals = frame['distributed_mono_values']
        
        for atom_idx in range(mono_dist.shape[0]):
            for dist_idx in range(mono_dist.shape[1]):
                charge_pos = mono_dist[atom_idx, dist_idx] + offset
                charge_val = mono_vals[atom_idx, dist_idx]
                all_charge_pos.append(charge_pos)
                all_charge_vals.append(charge_val)
    
    all_charge_pos = np.array(all_charge_pos)
    all_charge_vals = np.array(all_charge_vals)
    
    # Plot charge cloud
    scatter = ax.scatter(all_charge_pos[:, 0], all_charge_pos[:, 1], all_charge_pos[:, 2],
                        c=all_charge_vals, cmap='RdBu_r', s=10, alpha=0.2,
                        vmin=-0.5, vmax=0.5, edgecolors='none')
    
    # Plot reference atoms
    ref_pos = traj_data[min_idx]['positions'] - ref_com
    for atom_idx, pos in enumerate(ref_pos):
        if atoms[atom_idx].number == 6:
            ax.scatter(*pos, c=COLORS['black'], s=300, alpha=0.9, 
                      edgecolors='white', linewidths=2, zorder=100)
        else:
            ax.scatter(*pos, c=COLORS['vermillion'], s=250, alpha=0.9,
                      edgecolors='white', linewidths=2, zorder=100)
    
    ax.set_xlabel('X (Å)', fontsize=10, weight='bold')
    ax.set_ylabel('Y (Å)', fontsize=10, weight='bold')
    ax.set_zlabel('Z (Å)', fontsize=10, weight='bold')
    ax.set_title(f'All {len(traj_data)} Frames\nCharge Cloud', fontsize=11, weight='bold')
    
    # Equal axes
    max_range = 2.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_box_aspect([1, 1, 1])
    
    ax.view_init(elev=20, azim=45)
    
    # Plot 2: Min energy frame
    ax = fig.add_subplot(132, projection='3d')
    
    min_frame = traj_data[min_idx]
    mono_dist = min_frame['distributed_mono']
    mono_vals = min_frame['distributed_mono_values']
    
    # Flatten charges
    charge_pos = mono_dist.reshape(-1, 3)
    charge_vals = mono_vals.reshape(-1)
    
    scatter = ax.scatter(charge_pos[:, 0] - ref_com[0], 
                        charge_pos[:, 1] - ref_com[1], 
                        charge_pos[:, 2] - ref_com[2],
                        c=charge_vals, cmap='RdBu_r', s=80, alpha=0.8,
                        vmin=-0.5, vmax=0.5, edgecolors='black', linewidths=0.5)
    
    # Plot atoms
    for atom_idx, pos in enumerate(ref_pos):
        if atoms[atom_idx].number == 6:
            ax.scatter(*pos, c=COLORS['black'], s=300, alpha=0.9,
                      edgecolors='white', linewidths=2, zorder=100)
        else:
            ax.scatter(*pos, c=COLORS['vermillion'], s=250, alpha=0.9,
                      edgecolors='white', linewidths=2, zorder=100)
    
    ax.set_xlabel('X (Å)', fontsize=10, weight='bold')
    ax.set_ylabel('Y (Å)', fontsize=10, weight='bold')
    ax.set_zlabel('Z (Å)', fontsize=10, weight='bold')
    ax.set_title(f'Min Energy Frame\n(Frame {min_idx})', fontsize=11, weight='bold')
    
    # Equal axes
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_box_aspect([1, 1, 1])
    
    ax.view_init(elev=20, azim=45)
    
    # Plot 3: Max displacement frame
    # Find frame with max displacement from equilibrium
    displacements = []
    for frame in traj_data:
        disp = np.linalg.norm(frame['positions'] - traj_data[min_idx]['positions'])
        displacements.append(disp)
    max_disp_idx = np.argmax(displacements)
    
    ax = fig.add_subplot(133, projection='3d')
    
    max_frame = traj_data[max_disp_idx]
    mono_dist = max_frame['distributed_mono']
    mono_vals = max_frame['distributed_mono_values']
    
    charge_pos = mono_dist.reshape(-1, 3)
    charge_vals = mono_vals.reshape(-1)
    
    frame_com = max_frame['positions'].mean(axis=0)
    offset = ref_com - frame_com
    
    scatter = ax.scatter(charge_pos[:, 0] + offset[0], 
                        charge_pos[:, 1] + offset[1], 
                        charge_pos[:, 2] + offset[2],
                        c=charge_vals, cmap='RdBu_r', s=80, alpha=0.8,
                        vmin=-0.5, vmax=0.5, edgecolors='black', linewidths=0.5)
    
    # Plot atoms
    max_pos = max_frame['positions'] - frame_com + ref_com
    for atom_idx, pos in enumerate(max_pos):
        if atoms[atom_idx].number == 6:
            ax.scatter(*pos, c=COLORS['black'], s=300, alpha=0.9,
                      edgecolors='white', linewidths=2, zorder=100)
        else:
            ax.scatter(*pos, c=COLORS['vermillion'], s=250, alpha=0.9,
                      edgecolors='white', linewidths=2, zorder=100)
    
    ax.set_xlabel('X (Å)', fontsize=10, weight='bold')
    ax.set_ylabel('Y (Å)', fontsize=10, weight='bold')
    ax.set_zlabel('Z (Å)', fontsize=10, weight='bold')
    ax.set_title(f'Max Displacement\n(Frame {max_disp_idx})', fontsize=11, weight='bold')
    
    # Equal axes
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_box_aspect([1, 1, 1])
    
    ax.view_init(elev=20, azim=45)
    
    plt.suptitle('Distributed Charge Analysis', fontsize=16, weight='bold', y=0.96)
    
    # Add horizontal colorbar under the figure
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Monopole Charge (e)', fontsize=11, weight='bold')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved 3D visualization: {output_file}")
    plt.close()


def plot_charge_magnitude_evolution(traj_data, output_file):
    """
    Plot how charge magnitudes change during vibration.
    """
    n_frames = len(traj_data)
    phases = [frame['phase'] for frame in traj_data]
    energies = [frame['energy'] for frame in traj_data]
    
    # Get charges per atom over time
    natoms = traj_data[0]['atomic_charges'].shape[0]
    n_dist = traj_data[0]['distributed_mono_values'].shape[1]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Charge Evolution During Vibration', fontsize=16, weight='bold', y=0.995)
    
    # Plot 1: Atomic charges (PhysNet) - Mean-centered
    ax = axes[0]
    
    for atom_idx in range(natoms):
        charges = np.array([frame['atomic_charges'][atom_idx] for frame in traj_data])
        charge_mean = charges.mean()
        charge_std = charges.std()
        charges_centered = charges - charge_mean
        
        atom_label = f'Atom {atom_idx} (μ={charge_mean:.3f}e, σ={charge_std:.4f}e)'
        if atom_idx == 1:  # Central carbon (typically)
            color = COLORS['black']
            alpha = 0.9
        else:  # Oxygens
            color = COLORS['vermillion']
            alpha = 0.7
        
        ax.plot(np.array(phases) / (2*np.pi), charges_centered, 
               'o-', color=color, lw=2, markersize=5, alpha=alpha, label=atom_label)
    
    ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Oscillation Phase (fraction of period)', fontsize=11, weight='bold')
    ax.set_ylabel('Charge Deviation from Mean (e)', fontsize=11, weight='bold')
    ax.set_title('Atomic Charges (Mean-Centered)', fontsize=12, weight='bold', loc='left')
    ax.legend(fontsize=9, ncol=1, loc='best')
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(0, 1)
    
    # Plot 2: Distributed charge statistics - Mean-centered
    ax = axes[1]
    
    # Total positive and negative distributed charges
    total_pos = []
    total_neg = []
    charge_spread = []
    
    for frame in traj_data:
        mono_vals = frame['distributed_mono_values'].flatten()
        total_pos.append(mono_vals[mono_vals > 0].sum())
        total_neg.append(mono_vals[mono_vals < 0].sum())
        charge_spread.append(np.std(mono_vals))
    
    total_pos = np.array(total_pos)
    total_neg = np.array(total_neg)
    charge_spread = np.array(charge_spread)
    
    # Mean-center
    pos_mean, pos_std = total_pos.mean(), total_pos.std()
    neg_mean, neg_std = total_neg.mean(), total_neg.std()
    spread_mean, spread_std = charge_spread.mean(), charge_spread.std()
    
    ax.plot(np.array(phases) / (2*np.pi), total_pos - pos_mean, 
           'o-', color=COLORS['vermillion'], lw=2, markersize=4, 
           alpha=0.8, label=f'Total + (μ={pos_mean:.3f}e, σ={pos_std:.4f}e)')
    ax.plot(np.array(phases) / (2*np.pi), total_neg - neg_mean, 
           'o-', color=COLORS['sky_blue'], lw=2, markersize=4,
           alpha=0.8, label=f'Total - (μ={neg_mean:.3f}e, σ={neg_std:.4f}e)')
    ax.plot(np.array(phases) / (2*np.pi), charge_spread - spread_mean,
           'o-', color=COLORS['orange'], lw=2, markersize=4,
           alpha=0.8, label=f'Spread σ (μ={spread_mean:.4f}e, σ={spread_std:.5f}e)')
    
    ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Oscillation Phase (fraction of period)', fontsize=11, weight='bold')
    ax.set_ylabel('Deviation from Mean (e)', fontsize=11, weight='bold')
    ax.set_title('Distributed Charge Statistics (Mean-Centered)', fontsize=12, weight='bold', loc='left')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved charge evolution: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize charges during vibration')
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--raman-dir', type=Path, required=True,
                       help='Directory containing Raman analysis')
    parser.add_argument('--mode-index', type=int, default=7,
                       help='Normal mode index to visualize (default: 7 for ν₁)')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory')
    parser.add_argument('--n-frames', type=int, default=20,
                       help='Number of frames in vibration')
    parser.add_argument('--amplitude', type=float, default=0.2,
                       help='Vibration amplitude (Å)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.raman_dir / f'charge_visualization_mode{args.mode_index}'
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("DISTRIBUTED CHARGE VISUALIZATION DURING VIBRATION")
    print("="*70)
    print(f"\nSettings:")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Raman dir:   {args.raman_dir}")
    print(f"  Mode index:  {args.mode_index}")
    print(f"  Frames:      {args.n_frames}")
    print(f"  Amplitude:   {args.amplitude} Å")
    print(f"  Output:      {args.output_dir}")
    
    # Load model
    print(f"\n📂 Loading model...")
    model, params, config = load_model(args.checkpoint)
    print(f"   ✓ Model loaded")
    
    # Load geometry
    print(f"\n📂 Loading geometry...")
    xyz_file = args.raman_dir / 'CO2_optimized.xyz'
    if xyz_file.exists():
        atoms = read(xyz_file)
    else:
        from ase import Atoms
        atoms = Atoms('CO2', positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
    print(f"   ✓ Loaded {len(atoms)} atoms")
    
    # Load vibrations
    print(f"\n📂 Loading vibrations...")
    vib_cache = args.raman_dir / 'vibrations'
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vib = Vibrations(atoms, name=str(vib_cache / 'vib'))
    
    freqs = vib.get_frequencies()
    freq = freqs[args.mode_index]
    mode = vib.get_mode(args.mode_index)
    
    print(f"   ✓ Mode {args.mode_index}: {freq:.1f} cm⁻¹")
    
    # Classify mode
    if abs(freq - 1388.2) < 50:
        mode_type = 'ν₁ Symmetric Stretch (Raman strong)'
    elif abs(freq - 2349.2) < 50:
        mode_type = 'ν₃ Asymmetric Stretch (IR strong)'
    elif abs(freq - 667.4) < 50:
        mode_type = 'ν₂ Bending (IR + Raman)'
    else:
        mode_type = 'Other'
    
    print(f"   Type: {mode_type}")
    
    # Create trajectory with charges
    print(f"\n🔬 Computing distributed charges for trajectory...")
    traj_data = create_vibration_trajectory_with_charges(
        atoms, mode, args.amplitude, args.n_frames,
        model, params, config
    )
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    
    # 2D projections
    plot_charges_all_frames_3d(traj_data, atoms, 
                               args.output_dir / 'charges_2d_projections.png')
    
    # 3D evolution
    plot_charges_3d_evolution(traj_data, atoms,
                              args.output_dir / 'charges_3d_evolution.png')
    
    # Charge evolution over time
    plot_charge_magnitude_evolution(traj_data,
                                    args.output_dir / 'charge_evolution.png')
    
    # Save trajectory for ASE GUI viewing
    print(f"\n💾 Saving ASE trajectory...")
    traj_atoms = []
    for frame in traj_data:
        frame_atoms = atoms.copy()
        # Use positions from trajectory
        frame_com = frame['positions'].mean(axis=0)
        ref_com = traj_data[np.argmin([f['energy'] for f in traj_data])]['positions'].mean(axis=0)
        frame_atoms.positions = frame['positions'] - frame_com + ref_com
        traj_atoms.append(frame_atoms)
    
    traj_file = args.output_dir / f'mode_{args.mode_index}_vibration.traj'
    write(traj_file, traj_atoms)
    print(f"✅ Saved trajectory: {traj_file}")
    
    # Create charge trajectory data file
    charge_traj_file = args.output_dir / f'mode_{args.mode_index}_charges.npz'
    
    # Stack all charge data
    all_mono_dist = np.array([frame['distributed_mono'] for frame in traj_data])
    all_mono_vals = np.array([frame['distributed_mono_values'] for frame in traj_data])
    all_energies = np.array([frame['energy'] for frame in traj_data])
    all_phases = np.array([frame['phase'] for frame in traj_data])
    
    np.savez(charge_traj_file,
             positions_all=np.array([frame['positions'] for frame in traj_data]),
             distributed_mono_positions=all_mono_dist,
             distributed_mono_values=all_mono_vals,
             energies=all_energies,
             phases=all_phases,
             frequency=freq,
             mode_type=mode_type)
    print(f"✅ Saved charge data: {charge_traj_file}")
    
    print(f"\n{'='*70}")
    print(f"✅ VISUALIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput files in {args.output_dir}:")
    print(f"  charges_2d_projections.png - XY, XZ, YZ projections")
    print(f"  charges_3d_evolution.png - 3D views of charge cloud")
    print(f"  charge_evolution.png - Charge magnitudes vs phase")
    print(f"  mode_{args.mode_index}_vibration.traj - ASE trajectory")
    print(f"  mode_{args.mode_index}_charges.npz - Charge trajectory data")
    print(f"\n💡 View in ASE GUI:")
    print(f"   ase gui {traj_file}")
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()

