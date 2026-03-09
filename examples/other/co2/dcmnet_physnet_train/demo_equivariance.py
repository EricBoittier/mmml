#!/usr/bin/env python3
"""
Simple Equivariance Demonstration

This script provides a clear, visual demonstration of the difference between
equivariant (DCMNet) and non-equivariant models by applying rotations and
translations to a single test molecule.

Usage:
    python demo_equivariance.py --checkpoint-dcm path/to/dcmnet/best_params.pkl \
                                --checkpoint-noneq path/to/noneq/best_params.pkl \
                                --test-data valid.npz \
                                --test-esp grids_valid.npz
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pickle
from typing import Dict, Tuple

# Add mmml to path
repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation

# Import from trainer
from trainer import JointPhysNetDCMNet, JointPhysNetNonEquivariant, load_combined_data

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def create_rotation_matrix(axis: str, angle_degrees: float) -> np.ndarray:
    """Create rotation matrix around x, y, or z axis."""
    angle = np.radians(angle_degrees)
    c, s = np.cos(angle), np.sin(angle)
    
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis == 'z':
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unknown axis: {axis}")


def predict_properties(
    model: any,
    params: any,
    R: np.ndarray,
    Z: np.ndarray,
    N: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Predict molecular properties.
    
    Returns
    -------
    dict with keys:
        - dipole: (3,) dipole moment in e·Å
        - charges: (natoms,) atomic charges
        - distributed_charges: (natoms, n_dcm) distributed charges
        - distributed_positions: (natoms, n_dcm, 3) distributed charge positions
    """
    natoms = R.shape[1]
    batch_size = 1
    
    # Flatten
    positions_flat = R.reshape(-1, 3)
    atomic_numbers_flat = Z.reshape(-1)
    
    # Build edge list
    cutoff = 10.0
    n_atoms = int(N[0])
    dst_idx_list = []
    src_idx_list = []
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                dist = np.linalg.norm(R[0, i] - R[0, j])
                if dist < cutoff:
                    dst_idx_list.append(i)
                    src_idx_list.append(j)
    
    dst_idx = jnp.array(dst_idx_list, dtype=jnp.int32)
    src_idx = jnp.array(src_idx_list, dtype=jnp.int32)
    
    batch_segments = jnp.zeros(natoms, dtype=jnp.int32)
    batch_mask = jnp.ones(batch_size)
    atom_mask = (jnp.arange(natoms) < n_atoms).astype(jnp.float32)
    
    # Forward pass
    output = model.apply(
        params,
        atomic_numbers=jnp.array(atomic_numbers_flat),
        positions=jnp.array(positions_flat),
        dst_idx=dst_idx,
        src_idx=src_idx,
        batch_segments=batch_segments,
        batch_size=batch_size,
        batch_mask=batch_mask,
        atom_mask=atom_mask,
    )
    
    return {
        'dipole': np.array(output['dipoles_dcmnet'][0]),
        'charges': np.array(output['charges_as_mono']),
        'distributed_charges': np.array(output['mono_dist']),
        'distributed_positions': np.array(output['dipo_dist']),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate equivariance with rotations and translations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint-dcm', type=Path, required=True,
                       help='DCMNet checkpoint (best_params.pkl)')
    parser.add_argument('--checkpoint-noneq', type=Path, required=True,
                       help='Non-Equivariant checkpoint (best_params.pkl)')
    parser.add_argument('--test-data', type=Path, required=True,
                       help='Test EFD data (NPZ)')
    parser.add_argument('--test-esp', type=Path, required=True,
                       help='Test ESP data (NPZ)')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Index of test sample to use')
    parser.add_argument('--rotation-angle', type=float, default=90.0,
                       help='Rotation angle in degrees')
    parser.add_argument('--rotation-axis', type=str, default='z', choices=['x', 'y', 'z'],
                       help='Rotation axis')
    parser.add_argument('--translation', type=float, nargs=3, default=[5.0, 3.0, -2.0],
                       help='Translation vector (Angstroms)')
    parser.add_argument('--output-dir', type=Path, default=Path('equivariance_demo'),
                       help='Output directory')
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("EQUIVARIANCE DEMONSTRATION")
    print("="*70)
    
    # Load test data
    print(f"\n📁 Loading test data...")
    test_data = load_combined_data(args.test_data, args.test_esp, verbose=False)
    
    # Extract single sample
    idx = args.sample_idx
    R_orig = test_data['R'][idx:idx+1]
    Z = test_data['Z'][idx:idx+1]
    N = test_data['N'][idx:idx+1]
    n_atoms = int(N[0])
    
    print(f"   Selected sample {idx} with {n_atoms} atoms")
    
    # Create transformations
    rot_matrix = create_rotation_matrix(args.rotation_axis, args.rotation_angle)
    translation = np.array(args.translation)
    
    print(f"\n🔄 Transformations:")
    print(f"   Rotation: {args.rotation_angle}° around {args.rotation_axis}-axis")
    print(f"   Translation: {translation} Å")
    
    # Apply transformations
    R_rot = np.einsum('ij,snj->sni', rot_matrix, R_orig)
    R_trans = R_orig + translation[None, None, :]
    R_both = np.einsum('ij,snj->sni', rot_matrix, R_orig) + translation[None, None, :]
    
    # Load models
    print(f"\n🤖 Loading models...")
    
    with open(args.checkpoint_dcm, 'rb') as f:
        params_dcm = pickle.load(f)
    print(f"   ✅ DCMNet (equivariant)")
    
    with open(args.checkpoint_noneq, 'rb') as f:
        params_noneq = pickle.load(f)
    print(f"   ✅ Non-Equivariant")
    
    # Infer model configs from checkpoint
    # This is a simplified version - in practice you'd save configs with checkpoints
    natoms = R_orig.shape[1]
    max_atomic_number = int(np.max(Z))
    
    physnet_config = {
        'features': 64,
        'max_degree': 0,
        'num_iterations': 3,
        'num_basis_functions': 64,
        'cutoff': 6.0,
        'max_atomic_number': max_atomic_number,
        'charges': True,
        'natoms': natoms,
        'total_charge': 0.0,
        'n_res': 3,
        'zbl': False,
        'use_energy_bias': True,
        'debug': False,
        'efa': False,
    }
    
    dcmnet_config = {
        'features': 128,
        'max_degree': 2,
        'num_iterations': 2,
        'num_basis_functions': 64,
        'cutoff': 10.0,
        'max_atomic_number': max_atomic_number,
        'n_dcm': 3,
        'include_pseudotensors': False,
    }
    
    noneq_config = {
        'features': 128,
        'n_dcm': 3,
        'max_atomic_number': max_atomic_number,
        'num_layers': 3,
        'max_displacement': 1.0,
    }
    
    model_dcm = JointPhysNetDCMNet(
        physnet_config=physnet_config,
        dcmnet_config=dcmnet_config,
        mix_coulomb_energy=False,
    )
    
    model_noneq = JointPhysNetNonEquivariant(
        physnet_config=physnet_config,
        noneq_config=noneq_config,
        mix_coulomb_energy=False,
    )
    
    # Run predictions
    print(f"\n🔮 Running predictions...")
    
    print(f"\n   DCMNet (Equivariant):")
    pred_dcm_orig = predict_properties(model_dcm, params_dcm, R_orig, Z, N)
    print(f"      Original:     dipole = {pred_dcm_orig['dipole']}")
    
    pred_dcm_rot = predict_properties(model_dcm, params_dcm, R_rot, Z, N)
    dipole_expected = rot_matrix @ pred_dcm_orig['dipole']
    print(f"      Rotated:      dipole = {pred_dcm_rot['dipole']}")
    print(f"      Expected:     dipole = {dipole_expected}")
    error_rot = np.linalg.norm(pred_dcm_rot['dipole'] - dipole_expected)
    print(f"      Error:        {error_rot:.8f} e·Å  {'✅' if error_rot < 1e-4 else '⚠️'}")
    
    pred_dcm_trans = predict_properties(model_dcm, params_dcm, R_trans, Z, N)
    error_trans = np.linalg.norm(pred_dcm_trans['dipole'] - pred_dcm_orig['dipole'])
    print(f"      Translated:   dipole = {pred_dcm_trans['dipole']}")
    print(f"      Error:        {error_trans:.8f} e·Å  {'✅' if error_trans < 1e-4 else '⚠️'}")
    
    print(f"\n   Non-Equivariant:")
    pred_noneq_orig = predict_properties(model_noneq, params_noneq, R_orig, Z, N)
    print(f"      Original:     dipole = {pred_noneq_orig['dipole']}")
    
    pred_noneq_rot = predict_properties(model_noneq, params_noneq, R_rot, Z, N)
    dipole_expected_noneq = rot_matrix @ pred_noneq_orig['dipole']
    print(f"      Rotated:      dipole = {pred_noneq_rot['dipole']}")
    print(f"      Expected:     dipole = {dipole_expected_noneq}")
    error_rot_noneq = np.linalg.norm(pred_noneq_rot['dipole'] - dipole_expected_noneq)
    print(f"      Error:        {error_rot_noneq:.8f} e·Å  {'⚠️' if error_rot_noneq > 0.01 else '✅'}")
    
    pred_noneq_trans = predict_properties(model_noneq, params_noneq, R_trans, Z, N)
    error_trans_noneq = np.linalg.norm(pred_noneq_trans['dipole'] - pred_noneq_orig['dipole'])
    print(f"      Translated:   dipole = {pred_noneq_trans['dipole']}")
    print(f"      Error:        {error_trans_noneq:.8f} e·Å  {'✅' if error_trans_noneq < 1e-4 else '⚠️'}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    print("✅ = Pass (near-zero error)")
    print("⚠️  = Fail (significant error)\n")
    
    print(f"Rotation Equivariance ({args.rotation_angle}° around {args.rotation_axis}):")
    print(f"  DCMNet:        {error_rot:.2e}  {'✅ EQUIVARIANT' if error_rot < 1e-4 else '⚠️  NOT EQUIVARIANT'}")
    print(f"  Non-Eq:        {error_rot_noneq:.2e}  {'⚠️  NOT EQUIVARIANT (expected)' if error_rot_noneq > 0.01 else '✅'}")
    
    print(f"\nTranslation Invariance ({translation} Å):")
    print(f"  DCMNet:        {error_trans:.2e}  {'✅ INVARIANT' if error_trans < 1e-4 else '⚠️  NOT INVARIANT'}")
    print(f"  Non-Eq:        {error_trans_noneq:.2e}  {'✅ INVARIANT' if error_trans_noneq < 1e-4 else '⚠️  NOT INVARIANT'}")
    
    print(f"\n{'='*70}")
    print("KEY INSIGHTS:")
    print(f"{'='*70}\n")
    
    print("1. ROTATION EQUIVARIANCE:")
    print("   - DCMNet: Near-zero error → Guaranteed by architecture")
    print("   - Non-Eq: Large error → Must learn from data")
    print()
    print("2. TRANSLATION INVARIANCE:")
    print("   - Both: Near-zero error → Both use relative coordinates")
    print()
    print("3. PRACTICAL IMPLICATIONS:")
    print("   - DCMNet: Works well with limited rotational coverage")
    print("   - Non-Eq: Needs diverse rotations in training data")
    
    # Visualize if matplotlib available
    if HAS_MATPLOTLIB:
        print(f"\n📊 Generating visualizations...")
        
        fig = plt.figure(figsize=(16, 10))
        
        # Plot 1: Original molecule
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.scatter(R_orig[0, :n_atoms, 0], R_orig[0, :n_atoms, 1], R_orig[0, :n_atoms, 2],
                   c=Z[0, :n_atoms], cmap='tab10', s=200, alpha=0.8)
        ax1.quiver(0, 0, 0, *pred_dcm_orig['dipole'], color='red', arrow_length_ratio=0.2, linewidth=3)
        ax1.set_title(f'Original Molecule\nDCMNet Dipole', fontsize=12)
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Y (Å)')
        ax1.set_zlabel('Z (Å)')
        
        # Plot 2: Rotated molecule
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.scatter(R_rot[0, :n_atoms, 0], R_rot[0, :n_atoms, 1], R_rot[0, :n_atoms, 2],
                   c=Z[0, :n_atoms], cmap='tab10', s=200, alpha=0.8)
        ax2.quiver(0, 0, 0, *pred_dcm_rot['dipole'], color='red', arrow_length_ratio=0.2, linewidth=3, label='Predicted')
        ax2.quiver(0, 0, 0, *dipole_expected, color='blue', arrow_length_ratio=0.2, linewidth=3, linestyle='--', label='Expected')
        ax2.set_title(f'Rotated {args.rotation_angle}° (DCMNet)\nError: {error_rot:.2e}', fontsize=12)
        ax2.set_xlabel('X (Å)')
        ax2.set_ylabel('Y (Å)')
        ax2.set_zlabel('Z (Å)')
        ax2.legend()
        
        # Plot 3: Translated molecule
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        ax3.scatter(R_trans[0, :n_atoms, 0], R_trans[0, :n_atoms, 1], R_trans[0, :n_atoms, 2],
                   c=Z[0, :n_atoms], cmap='tab10', s=200, alpha=0.8)
        ax3.quiver(*translation, *pred_dcm_trans['dipole'], color='red', arrow_length_ratio=0.2, linewidth=3)
        ax3.set_title(f'Translated (DCMNet)\nError: {error_trans:.2e}', fontsize=12)
        ax3.set_xlabel('X (Å)')
        ax3.set_ylabel('Y (Å)')
        ax3.set_zlabel('Z (Å)')
        
        # Plot 4: Non-Eq original
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        ax4.scatter(R_orig[0, :n_atoms, 0], R_orig[0, :n_atoms, 1], R_orig[0, :n_atoms, 2],
                   c=Z[0, :n_atoms], cmap='tab10', s=200, alpha=0.8)
        ax4.quiver(0, 0, 0, *pred_noneq_orig['dipole'], color='green', arrow_length_ratio=0.2, linewidth=3)
        ax4.set_title(f'Original Molecule\nNon-Eq Dipole', fontsize=12)
        ax4.set_xlabel('X (Å)')
        ax4.set_ylabel('Y (Å)')
        ax4.set_zlabel('Z (Å)')
        
        # Plot 5: Non-Eq rotated
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        ax5.scatter(R_rot[0, :n_atoms, 0], R_rot[0, :n_atoms, 1], R_rot[0, :n_atoms, 2],
                   c=Z[0, :n_atoms], cmap='tab10', s=200, alpha=0.8)
        ax5.quiver(0, 0, 0, *pred_noneq_rot['dipole'], color='green', arrow_length_ratio=0.2, linewidth=3, label='Predicted')
        ax5.quiver(0, 0, 0, *dipole_expected_noneq, color='blue', arrow_length_ratio=0.2, linewidth=3, linestyle='--', label='Expected')
        ax5.set_title(f'Rotated {args.rotation_angle}° (Non-Eq)\nError: {error_rot_noneq:.2e}', fontsize=12)
        ax5.set_xlabel('X (Å)')
        ax5.set_ylabel('Y (Å)')
        ax5.set_zlabel('Z (Å)')
        ax5.legend()
        
        # Plot 6: Non-Eq translated
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        ax6.scatter(R_trans[0, :n_atoms, 0], R_trans[0, :n_atoms, 1], R_trans[0, :n_atoms, 2],
                   c=Z[0, :n_atoms], cmap='tab10', s=200, alpha=0.8)
        ax6.quiver(*translation, *pred_noneq_trans['dipole'], color='green', arrow_length_ratio=0.2, linewidth=3)
        ax6.set_title(f'Translated (Non-Eq)\nError: {error_trans_noneq:.2e}', fontsize=12)
        ax6.set_xlabel('X (Å)')
        ax6.set_ylabel('Y (Å)')
        ax6.set_zlabel('Z (Å)')
        
        plt.tight_layout()
        output_path = args.output_dir / 'equivariance_demo.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {output_path}")
    
    print(f"\n{'='*70}")
    print("✅ DEMONSTRATION COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

