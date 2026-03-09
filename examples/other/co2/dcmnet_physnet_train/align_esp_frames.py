#!/usr/bin/env python3
"""
Diagnostic script to check for coordinate frame misalignment between 
atom positions and ESP grids by performing rigid alignment.

If atom positions and ESP grids are in different reference frames,
a rotation+translation can significantly reduce ESP error.
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from trainer import (
    JointPhysNetDCMNet, 
    prepare_batch_data, 
    load_combined_data, 
    precompute_edge_lists,
    eval_step
)


def compute_esp_from_charges(charges, positions, grid_points):
    """
    Compute ESP from point charges.
    
    ESP(r) = Σ q_i / |r - r_i| (in atomic units with Å to Bohr conversion)
    """
    # charges: (n_charges,)
    # positions: (n_charges, 3)
    # grid_points: (n_grid, 3)
    
    # Compute distances: (n_grid, n_charges)
    diff = grid_points[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    
    # ESP in atomic units (Hartree/e)
    # Need to convert Å to Bohr: 1 Å = 1.88973 Bohr
    esp = np.sum(charges[None, :] / (distances * 1.88973 + 1e-10), axis=1)
    return esp


def align_molecule(atom_positions, charges, grid_points, esp_target):
    """
    Find optimal rotation and translation to align molecule with ESP grid.
    
    Minimizes RMSE between predicted ESP (from aligned charges) and target ESP.
    
    Parameters
    ----------
    atom_positions : ndarray (n_atoms, 3)
        Initial atom positions
    charges : ndarray (n_atoms,)
        Atomic charges
    grid_points : ndarray (n_grid, 3)
        ESP grid points
    esp_target : ndarray (n_grid,)
        Target ESP values
        
    Returns
    -------
    dict
        'rotation': Rotation matrix (3, 3)
        'translation': Translation vector (3,)
        'rmse_before': RMSE before alignment
        'rmse_after': RMSE after alignment
        'aligned_positions': Transformed atom positions
    """
    # Initial ESP (no alignment)
    esp_init = compute_esp_from_charges(charges, atom_positions, grid_points)
    rmse_init = np.sqrt(np.mean((esp_init - esp_target)**2))
    
    # Optimization variables: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    # Rotation represented as Euler angles
    def objective(params):
        # Extract parameters
        euler_angles = params[:3]  # Rotation (Euler angles)
        translation = params[3:6]  # Translation
        
        # Apply rotation
        rot = Rotation.from_euler('xyz', euler_angles)
        positions_rotated = rot.apply(atom_positions)
        
        # Apply translation
        positions_transformed = positions_rotated + translation
        
        # Compute ESP
        esp_pred = compute_esp_from_charges(charges, positions_transformed, grid_points)
        
        # RMSE
        rmse = np.sqrt(np.mean((esp_pred - esp_target)**2))
        return rmse
    
    # Initial guess: no rotation, no translation
    x0 = np.zeros(6)
    
    # Optimize
    result = minimize(objective, x0, method='Powell', 
                     options={'maxiter': 1000, 'ftol': 1e-6})
    
    # Extract optimal transformation
    euler_opt = result.x[:3]
    translation_opt = result.x[3:6]
    rot_opt = Rotation.from_euler('xyz', euler_opt)
    
    # Apply optimal transformation
    positions_aligned = rot_opt.apply(atom_positions) + translation_opt
    esp_aligned = compute_esp_from_charges(charges, positions_aligned, grid_points)
    rmse_aligned = np.sqrt(np.mean((esp_aligned - esp_target)**2))
    
    return {
        'rotation_matrix': rot_opt.as_matrix(),
        'rotation_euler': euler_opt,
        'translation': translation_opt,
        'rmse_before': rmse_init,
        'rmse_after': rmse_aligned,
        'aligned_positions': positions_aligned,
        'esp_before': esp_init,
        'esp_after': esp_aligned,
        'optimization_success': result.success,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check ESP coordinate frame alignment')
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--valid-efd', type=Path, required=True,
                       help='Validation EFD file')
    parser.add_argument('--valid-esp', type=Path, required=True,
                       help='Validation ESP file')
    parser.add_argument('--n-samples', type=int, default=5,
                       help='Number of samples to analyze')
    parser.add_argument('--cutoff', type=float, default=10.0,
                       help='Edge cutoff distance')
    args = parser.parse_args()
    
    print("="*70)
    print("ESP Coordinate Frame Alignment Analysis")
    print("="*70)
    
    # Load checkpoint
    print(f"\n1. Loading checkpoint from {args.checkpoint}")
    with open(args.checkpoint / 'best_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    with open(args.checkpoint / 'model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    # Create model
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config'],
        mix_coulomb_energy=config.get('mix_coulomb_energy', False)
    )
    
    n_dcm = config['dcmnet_config']['n_dcm']
    
    # Load data
    print(f"\n2. Loading data...")
    data = load_combined_data(args.valid_efd, args.valid_esp, verbose=False)
    data = precompute_edge_lists(data, cutoff=args.cutoff, verbose=False)
    
    print(f"\n3. Analyzing {args.n_samples} samples for coordinate alignment...")
    print("="*70)
    
    results = []
    
    for idx in range(min(args.n_samples, len(data['N']))):
        print(f"\n{'='*70}")
        print(f"Sample {idx}")
        print(f"{'='*70}")
        
        # Prepare batch
        batch = prepare_batch_data(data, np.array([idx]), cutoff=args.cutoff)
        
        # Get atom positions and charges
        n_atoms = int(batch['N'][0])
        atom_positions = np.array(batch['R'][:n_atoms])
        atomic_nums = np.array(batch['Z'][:n_atoms])
        
        print(f"\nAtoms: {n_atoms}")
        print(f"Atomic numbers: {atomic_nums}")
        print(f"Atom positions (Å):")
        for i, pos in enumerate(atom_positions):
            print(f"  Atom {i} (Z={int(atomic_nums[i])}): {pos}")
        
        # Get ESP grid
        grid_points = np.array(batch['vdw_surface'][0])
        esp_target = np.array(batch['esp'][0])
        
        print(f"\nESP grid: {len(grid_points)} points")
        print(f"Grid range: X=[{grid_points[:, 0].min():.3f}, {grid_points[:, 0].max():.3f}], "
              f"Y=[{grid_points[:, 1].min():.3f}, {grid_points[:, 1].max():.3f}], "
              f"Z=[{grid_points[:, 2].min():.3f}, {grid_points[:, 2].max():.3f}]")
        print(f"ESP range: [{esp_target.min():.6f}, {esp_target.max():.6f}] Ha/e")
        
        # Run model to get charges
        _, losses, output = eval_step(
            params=params,
            batch=batch,
            model_apply=model.apply,
            energy_w=1.0,
            forces_w=50.0,
            dipole_w=25.0,
            esp_w=10000.0,
            mono_w=100.0,
            batch_size=1,
            n_dcm=n_dcm,
            dipole_source='physnet',
            esp_min_distance=0.0,
            esp_max_value=1e10,
        )
        
        # Extract PhysNet charges
        charges_physnet = np.array(output['charges_as_mono'][:n_atoms])
        
        print(f"\nPhysNet charges: [{charges_physnet.min():.4f}, {charges_physnet.max():.4f}] e")
        print(f"Total charge: {charges_physnet.sum():.6f} e")
        
        # Perform alignment
        print(f"\nPerforming spatial alignment...")
        alignment = align_molecule(atom_positions, charges_physnet, grid_points, esp_target)
        
        print(f"\n{'─'*70}")
        print("ALIGNMENT RESULTS:")
        print(f"{'─'*70}")
        print(f"RMSE before alignment: {alignment['rmse_before']:.6f} Ha/e ({alignment['rmse_before']*627.5:.2f} kcal/mol/e)")
        print(f"RMSE after alignment:  {alignment['rmse_after']:.6f} Ha/e ({alignment['rmse_after']*627.5:.2f} kcal/mol/e)")
        improvement = (1 - alignment['rmse_after'] / alignment['rmse_before']) * 100
        print(f"Improvement: {improvement:.1f}%")
        
        print(f"\nOptimal rotation (Euler angles): {alignment['rotation_euler']} rad")
        print(f"  = {np.degrees(alignment['rotation_euler'])} degrees")
        print(f"Optimal translation: {alignment['translation']} Å")
        
        print(f"\nAligned atom positions:")
        for i, pos in enumerate(alignment['aligned_positions']):
            original = atom_positions[i]
            shift = pos - original
            print(f"  Atom {i}: {original} → {pos} (shift: {shift})")
        
        print(f"\nOptimization success: {alignment['optimization_success']}")
        
        results.append({
            'sample_idx': idx,
            'n_atoms': n_atoms,
            'rmse_before': alignment['rmse_before'],
            'rmse_after': alignment['rmse_after'],
            'improvement_pct': improvement,
            'rotation_degrees': np.degrees(alignment['rotation_euler']),
            'translation': alignment['translation'],
        })
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Row 0: ESP comparisons
        ax = axes[0, 0]
        ax.scatter(esp_target, alignment['esp_before'], alpha=0.5, s=20, color='red', label='Before')
        lims = [min(esp_target.min(), alignment['esp_before'].min()),
                max(esp_target.max(), alignment['esp_before'].max())]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel('Target ESP (Ha/e)')
        ax.set_ylabel('Predicted ESP (Ha/e)')
        ax.set_title(f'Before Alignment\nRMSE: {alignment["rmse_before"]:.6f} Ha/e')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.scatter(esp_target, alignment['esp_after'], alpha=0.5, s=20, color='green', label='After')
        lims = [min(esp_target.min(), alignment['esp_after'].min()),
                max(esp_target.max(), alignment['esp_after'].max())]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel('Target ESP (Ha/e)')
        ax.set_ylabel('Predicted ESP (Ha/e)')
        ax.set_title(f'After Alignment\nRMSE: {alignment["rmse_after"]:.6f} Ha/e')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        ax.scatter(esp_target, alignment['esp_before'], alpha=0.3, s=10, color='red', label='Before')
        ax.scatter(esp_target, alignment['esp_after'], alpha=0.3, s=10, color='green', label='After')
        lims = [min(esp_target.min(), alignment['esp_before'].min(), alignment['esp_after'].min()),
                max(esp_target.max(), alignment['esp_before'].max(), alignment['esp_after'].max())]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel('Target ESP (Ha/e)')
        ax.set_ylabel('Predicted ESP (Ha/e)')
        ax.set_title(f'Comparison\nImprovement: {improvement:.1f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Row 1: Spatial visualization
        ax = axes[1, 0]
        # Original positions
        ax.scatter(atom_positions[:, 0], atom_positions[:, 2], 
                  c='red', s=200, marker='o', edgecolors='black', linewidths=2, label='Original')
        for i, pos in enumerate(atom_positions):
            ax.text(pos[0], pos[2], f' Z={int(atomic_nums[i])}', fontsize=10)
        ax.scatter(grid_points[:, 0], grid_points[:, 2], c='gray', s=5, alpha=0.2, label='ESP grid')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Z (Å)')
        ax.set_title('Original Positions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        ax = axes[1, 1]
        # Aligned positions
        aligned_pos = alignment['aligned_positions']
        ax.scatter(aligned_pos[:, 0], aligned_pos[:, 2], 
                  c='green', s=200, marker='o', edgecolors='black', linewidths=2, label='Aligned')
        for i, pos in enumerate(aligned_pos):
            ax.text(pos[0], pos[2], f' Z={int(atomic_nums[i])}', fontsize=10)
        ax.scatter(grid_points[:, 0], grid_points[:, 2], c='gray', s=5, alpha=0.2, label='ESP grid')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Z (Å)')
        ax.set_title('Aligned Positions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        ax = axes[1, 2]
        # Overlay
        ax.scatter(atom_positions[:, 0], atom_positions[:, 2], 
                  c='red', s=200, marker='o', alpha=0.5, edgecolors='black', linewidths=2, label='Original')
        ax.scatter(aligned_pos[:, 0], aligned_pos[:, 2], 
                  c='green', s=200, marker='s', alpha=0.5, edgecolors='black', linewidths=2, label='Aligned')
        # Draw arrows showing transformation
        for i in range(len(atom_positions)):
            ax.arrow(atom_positions[i, 0], atom_positions[i, 2],
                    aligned_pos[i, 0] - atom_positions[i, 0],
                    aligned_pos[i, 2] - atom_positions[i, 2],
                    head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)
        ax.scatter(grid_points[:, 0], grid_points[:, 2], c='gray', s=5, alpha=0.2)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Z (Å)')
        ax.set_title(f'Transformation\nΔR={np.linalg.norm(alignment["translation"]):.3f} Å, θ={np.linalg.norm(alignment["rotation_euler"]):.3f} rad')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.suptitle(f'Sample {idx}: Spatial Alignment Analysis', fontsize=14, weight='bold')
        plt.tight_layout()
        
        # Save
        save_dir = args.checkpoint / 'alignment_analysis'
        save_dir.mkdir(exist_ok=True)
        plot_path = save_dir / f'alignment_sample_{idx}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✅ Saved alignment plot: {plot_path}")
    
    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY OF ALIGNMENT ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"{'Sample':<8} {'RMSE Before':<15} {'RMSE After':<15} {'Improvement':<12} {'|Translation|':<15} {'|Rotation|'}")
    print(f"{'':<8} {'(Ha/e)':<15} {'(Ha/e)':<15} {'(%)':<12} {'(Å)':<15} {'(deg)'}")
    print(f"{'─'*70}")
    
    for r in results:
        trans_mag = np.linalg.norm(r['translation'])
        rot_mag = np.linalg.norm(r['rotation_degrees'])
        print(f"{r['sample_idx']:<8} {r['rmse_before']:<15.6f} {r['rmse_after']:<15.6f} {r['improvement_pct']:<12.1f} {trans_mag:<15.3f} {rot_mag:<.2f}")
    
    # Average improvement
    avg_improvement = np.mean([r['improvement_pct'] for r in results])
    avg_trans = np.mean([np.linalg.norm(r['translation']) for r in results])
    avg_rot = np.mean([np.linalg.norm(r['rotation_degrees']) for r in results])
    
    print(f"{'─'*70}")
    print(f"{'AVERAGE':<8} {'':<15} {'':<15} {avg_improvement:<12.1f} {avg_trans:<15.3f} {avg_rot:<.2f}")
    
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"{'='*70}")
    
    if avg_improvement < 5:
        print("✅ GOOD: Alignment provides < 5% improvement")
        print("   → Atom positions and ESP grids are in the SAME coordinate frame")
        print("   → No systematic misalignment")
    elif avg_improvement < 20:
        print("⚠️  MODERATE: Alignment provides 5-20% improvement")
        print("   → Small coordinate frame mismatch exists")
        print("   → Consider checking data preprocessing")
    else:
        print("❌ SIGNIFICANT: Alignment provides > 20% improvement")
        print("   → Large coordinate frame mismatch!")
        print("   → Atom positions and ESP grids may be in DIFFERENT reference frames")
        print("   → CHECK: Were ESP grids computed in a different geometry optimization?")
        print("   → CHECK: Is there a coordinate transformation in data preparation?")
    
    if avg_trans > 1.0:
        print(f"\n⚠️  Large translation ({avg_trans:.2f} Å) suggests systematic offset")
    
    if avg_rot > 10.0:
        print(f"\n⚠️  Large rotation ({avg_rot:.1f}°) suggests orientation mismatch")
    
    print(f"\nAlignment plots saved to: {args.checkpoint / 'alignment_analysis'}")


if __name__ == '__main__':
    main()

