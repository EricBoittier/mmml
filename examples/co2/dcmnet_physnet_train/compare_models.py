#!/usr/bin/env python3
"""
Head-to-Head Comparison: Equivariant vs Non-Equivariant Models

This script:
1. Trains both DCMNet (equivariant) and Non-Equivariant models
2. Tests equivariance with rotations and translations
3. Compares performance, speed, and memory usage
4. Generates detailed comparison plots and metrics

Usage:
    python compare_models.py --train-efd train.npz --train-esp esp_train.npz \
                            --valid-efd valid.npz --valid-esp esp_valid.npz \
                            --epochs 50 --comparison-name my_comparison
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pickle
import json
import time
from typing import Dict, Tuple, Any, List
from dataclasses import dataclass, asdict

# Add mmml to path
repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from e3x import so3

# Import training script components
from trainer import (
    JointPhysNetDCMNet,
    JointPhysNetNonEquivariant,
    load_combined_data,
    train_model,
    create_optimizer,
    get_recommended_optimizer_config,
    LossTerm,
)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class ModelMetrics:
    """Store metrics for a single model."""
    name: str
    training_time: float
    inference_time: float
    memory_usage_mb: float
    num_parameters: int
    
    # Validation metrics
    val_energy_mae: float
    val_forces_mae: float
    val_dipole_mae: float
    val_esp_mae: float
    
    # Equivariance test results
    rotation_error_dipole: float
    rotation_error_esp: float
    translation_error_dipole: float
    translation_error_esp: float


def apply_rotation(positions: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Apply rotation matrix to positions.
    
    Parameters
    ----------
    positions : np.ndarray
        Shape (n_samples, n_atoms, 3)
    rotation_matrix : np.ndarray
        Shape (3, 3)
    
    Returns
    -------
    np.ndarray
        Rotated positions
    """
    positions_jax = jnp.asarray(positions)
    rotation_jax = jnp.asarray(rotation_matrix, dtype=positions_jax.dtype)
    rotated = jnp.einsum('ij,...j->...i', rotation_jax, positions_jax)
    if isinstance(positions, np.ndarray):
        return np.asarray(rotated)
    return rotated


def apply_translation(positions: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Apply translation to positions.
    
    Parameters
    ----------
    positions : np.ndarray
        Shape (n_samples, n_atoms, 3)
    translation : np.ndarray
        Shape (3,)
    
    Returns
    -------
    np.ndarray
        Translated positions
    """
    positions_jax = jnp.asarray(positions)
    translation_jax = jnp.asarray(translation, dtype=positions_jax.dtype)
    translated = positions_jax + translation_jax
    if isinstance(positions, np.ndarray):
        return np.asarray(translated)
    return translated


def generate_random_rotation(key: jax.Array) -> jnp.ndarray:
    """Generate a random 3D rotation matrix using e3x."""
    return so3.random_rotation(key)


def test_equivariance(
    model: Any,
    params: Any,
    test_data: Dict[str, np.ndarray],
    num_test_samples: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Test model equivariance under rotations and translations.
    
    Parameters
    ----------
    model : JointPhysNetDCMNet or JointPhysNetNonEquivariant
        Model to test
    params : Any
        Model parameters
    test_data : Dict[str, np.ndarray]
        Test dataset
    num_test_samples : int
        Number of samples to test
    seed : int
        Random seed
    
    Returns
    -------
    Dict[str, float]
        Equivariance error metrics
    """
    rng = np.random.RandomState(seed)
    jax_key = jax.random.PRNGKey(seed)
    
    # Select test samples
    n_available = len(test_data['E'])
    test_indices = rng.choice(n_available, size=min(num_test_samples, n_available), replace=False)
    
    rotation_errors_dipole = []
    rotation_errors_esp = []
    translation_errors_dipole = []
    translation_errors_esp = []
    energy_errors = []
    forces_errors = []
    dipole_errors = []
    esp_errors = []
    
    print(f"\n{'='*70}")
    print(f"Testing Equivariance on {len(test_indices)} samples")
    print(f"{'='*70}\n")
    
    for idx in test_indices:
        # Extract single sample
        R = test_data['R'][idx:idx+1]  # (1, natoms, 3)
        Z = test_data['Z'][idx:idx+1]  # (1, natoms)
        N = test_data['N'][idx:idx+1]  # (1,)
        vdw_surface = test_data['vdw_surface'][idx:idx+1]  # (1, ngrid, 3)
        n_atoms = int(N[0])
        
        # Original prediction
        output_orig = predict_single(model, params, R, Z, N, vdw_surface)
        
        # Test rotation
        jax_key, rot_key, trans_key = jax.random.split(jax_key, 3)
        rot_matrix = generate_random_rotation(rot_key)
        R_rot = apply_rotation(R, rot_matrix)
        vdw_rot = apply_rotation(vdw_surface, rot_matrix)
        
        output_rot = predict_single(model, params, R_rot, Z, N, vdw_rot)
        
        # For equivariant model: rotated output should equal rotation of original output
        # Dipole should rotate
        dipole_orig = jnp.asarray(output_orig['dipole'])
        dipole_rot = jnp.asarray(output_rot['dipole'])
        dipole_expected = jnp.einsum('ij,j->i', rot_matrix, dipole_orig)
        rotation_error_dipole = jnp.linalg.norm(dipole_rot - dipole_expected)
        rotation_errors_dipole.append(float(rotation_error_dipole))
        
        # ESP should be identical at rotated grid points
        esp_orig = jnp.asarray(output_orig['esp'])
        esp_rot = jnp.asarray(output_rot['esp'])
        rotation_error_esp = jnp.mean(jnp.abs(esp_rot - esp_orig))
        rotation_errors_esp.append(float(rotation_error_esp))
        
        # Test translation (both models should be translation invariant)
        translation = 5.0 * jax.random.normal(trans_key, (3,), dtype=rot_matrix.dtype)
        R_trans = apply_translation(R, translation)
        vdw_trans = apply_translation(vdw_surface, translation)
        
        output_trans = predict_single(model, params, R_trans, Z, N, vdw_trans)
        
        # Dipole should be identical (molecule-centered)
        dipole_trans = jnp.asarray(output_trans['dipole'])
        translation_error_dipole = jnp.linalg.norm(dipole_trans - dipole_orig)
        translation_errors_dipole.append(float(translation_error_dipole))
        
        # ESP should be identical
        esp_trans = jnp.asarray(output_trans['esp'])
        translation_error_esp = jnp.mean(jnp.abs(esp_trans - esp_orig))
        translation_errors_esp.append(float(translation_error_esp))

        # Compare against reference quantities
        true_energy = jnp.asarray(test_data['E'][idx])
        energy_error = jnp.abs(jnp.asarray(output_orig['energy']) - true_energy)
        energy_errors.append(float(energy_error))

        true_dipole = jnp.asarray(test_data['Dxyz'][idx])
        dipole_mae = jnp.mean(jnp.abs(dipole_orig - true_dipole))
        dipole_errors.append(float(dipole_mae))

        true_forces = jnp.asarray(test_data['F'][idx])
        if true_forces.ndim == 1:
            true_forces = true_forces.reshape(-1, 3)
        forces_pred = jnp.asarray(output_orig['forces'])
        forces_mae = jnp.mean(
            jnp.abs(forces_pred[:n_atoms] - true_forces[:n_atoms])
        )
        forces_errors.append(float(forces_mae))

        true_esp = jnp.asarray(test_data['esp'][idx])
        esp_mae = jnp.mean(jnp.abs(esp_orig - true_esp))
        esp_errors.append(float(esp_mae))
    
    results = {
        'rotation_error_dipole': float(np.mean(rotation_errors_dipole)),
        'rotation_error_esp': float(np.mean(rotation_errors_esp)),
        'translation_error_dipole': float(np.mean(translation_errors_dipole)),
        'translation_error_esp': float(np.mean(translation_errors_esp)),
        'rotation_error_dipole_std': float(np.std(rotation_errors_dipole)),
        'rotation_error_esp_std': float(np.std(rotation_errors_esp)),
        'translation_error_dipole_std': float(np.std(translation_errors_dipole)),
        'translation_error_esp_std': float(np.std(translation_errors_esp)),
        'energy_mae': float(np.mean(energy_errors)),
        'forces_mae': float(np.mean(forces_errors)),
        'dipole_mae': float(np.mean(dipole_errors)),
        'esp_mae': float(np.mean(esp_errors)),
        'energy_mae_std': float(np.std(energy_errors)),
        'forces_mae_std': float(np.std(forces_errors)),
        'dipole_mae_std': float(np.std(dipole_errors)),
        'esp_mae_std': float(np.std(esp_errors)),
    }
    
    print(f"Rotation Equivariance:")
    print(f"  Dipole error: {results['rotation_error_dipole']:.6f} ± {results['rotation_error_dipole_std']:.6f} e·Å")
    print(f"  ESP error:    {results['rotation_error_esp']:.6f} ± {results['rotation_error_esp_std']:.6f} Ha/e")
    print(f"\nTranslation Invariance:")
    print(f"  Dipole error: {results['translation_error_dipole']:.6f} ± {results['translation_error_dipole_std']:.6f} e·Å")
    print(f"  ESP error:    {results['translation_error_esp']:.6f} ± {results['translation_error_esp_std']:.6f} Ha/e")
    print(f"\nReference Property Errors:")
    print(f"  Energy MAE: {results['energy_mae']:.6f} ± {results['energy_mae_std']:.6f} eV")
    print(f"  Forces MAE: {results['forces_mae']:.6f} ± {results['forces_mae_std']:.6f} eV/Å")
    print(f"  Dipole MAE: {results['dipole_mae']:.6f} ± {results['dipole_mae_std']:.6f} e·Å")
    print(f"  ESP MAE:    {results['esp_mae']:.6f} ± {results['esp_mae_std']:.6f} Ha/e")
    
    return results


def predict_single(
    model: Any,
    params: Any,
    R: np.ndarray,
    Z: np.ndarray,
    N: np.ndarray,
    vdw_surface: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Make prediction for a single sample.
    
    Parameters
    ----------
    model : JointPhysNetDCMNet or JointPhysNetNonEquivariant
        Model
    params : Any
        Model parameters
    R : np.ndarray
        Positions (1, natoms, 3)
    Z : np.ndarray
        Atomic numbers (1, natoms)
    N : np.ndarray
        Number of atoms (1,)
    vdw_surface : np.ndarray
        ESP grid points (1, ngrid, 3)
    
    Returns
    -------
    Dict[str, np.ndarray]
        Predictions including dipole and ESP
    """
    import e3x
    
    natoms = R.shape[1]
    batch_size = 1
    
    # Flatten for model input
    positions_flat = R.reshape(-1, 3)
    atomic_numbers_flat = Z.reshape(-1)
    
    # Build edge list
    cutoff = 10.0  # Use max cutoff
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
    
    # Batch segments and masks
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
    
    # Compute ESP
    mono_dist = output['mono_dist'].reshape(batch_size, natoms, -1)
    dipo_dist = output['dipo_dist'].reshape(batch_size, natoms, -1, 3)
    
    # Calculate ESP at grid points
    esp_pred = calculate_esp(
        mono_dist[0],  # (natoms, n_dcm)
        dipo_dist[0],  # (natoms, n_dcm, 3)
        vdw_surface[0],  # (ngrid, 3)
        atom_mask,
    )
    
    forces = output['forces'].reshape(batch_size, natoms, 3)

    return {
        'dipole': np.array(output['dipoles_dcmnet'][0]),
        'esp': np.array(esp_pred),
        'energy': np.array(output['energy'][0]),
        'forces': np.array(forces[0]),
    }


def calculate_esp(
    charges: jnp.ndarray,
    positions: jnp.ndarray,
    grid_points: jnp.ndarray,
    atom_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate ESP at grid points from distributed charges.
    
    Parameters
    ----------
    charges : jnp.ndarray
        Charge values (natoms, n_dcm)
    positions : jnp.ndarray
        Charge positions (natoms, n_dcm, 3)
    grid_points : jnp.ndarray
        Grid points (ngrid, 3)
    atom_mask : jnp.ndarray
        Atom mask (natoms,)
    
    Returns
    -------
    jnp.ndarray
        ESP values (ngrid,)
    """
    natoms, n_dcm = charges.shape
    ngrid = grid_points.shape[0]
    
    # Flatten charges and positions
    charges_flat = charges.reshape(-1)  # (natoms*n_dcm,)
    positions_flat = positions.reshape(-1, 3)  # (natoms*n_dcm, 3)
    
    # Expand atom mask
    atom_mask_expanded = jnp.repeat(atom_mask, n_dcm)  # (natoms*n_dcm,)
    charges_masked = charges_flat * atom_mask_expanded
    
    # Compute distances: (ngrid, natoms*n_dcm)
    diff = grid_points[:, None, :] - positions_flat[None, :, :]  # (ngrid, natoms*n_dcm, 3)
    distances = jnp.linalg.norm(diff, axis=-1)  # (ngrid, natoms*n_dcm)
    
    # Avoid division by zero
    distances = jnp.where(distances < 1e-6, 1e6, distances)
    
    # ESP = sum_i q_i / r_i (in atomic units, distances in Angstrom -> Bohr)
    distances_bohr = distances * 1.88973
    esp = jnp.sum(charges_masked[None, :] / distances_bohr, axis=1)  # (ngrid,)
    
    return esp


def measure_inference_time(
    model: Any,
    params: Any,
    test_data: Dict[str, np.ndarray],
    num_samples: int = 100,
) -> float:
    """Measure average inference time per sample."""
    indices = np.random.choice(len(test_data['E']), size=min(num_samples, len(test_data['E'])), replace=False)
    
    times = []
    for idx in indices:
        R = test_data['R'][idx:idx+1]
        Z = test_data['Z'][idx:idx+1]
        N = test_data['N'][idx:idx+1]
        vdw_surface = test_data['vdw_surface'][idx:idx+1]
        
        start = time.time()
        _ = predict_single(model, params, R, Z, N, vdw_surface)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return float(np.mean(times))


def count_parameters(params: Any) -> int:
    """Count total number of parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def plot_comparison(
    metrics_dcm: ModelMetrics,
    metrics_noneq: ModelMetrics,
    save_dir: Path,
):
    """Generate comparison plots."""
    if not HAS_MATPLOTLIB:
        print("⚠️  Matplotlib not available, skipping plots")
        return
    
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Performance metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics_names = ['Energy MAE', 'Forces MAE', 'Dipole MAE', 'ESP MAE']
    dcm_values = [
        metrics_dcm.val_energy_mae,
        metrics_dcm.val_forces_mae,
        metrics_dcm.val_dipole_mae,
        metrics_dcm.val_esp_mae,
    ]
    noneq_values = [
        metrics_noneq.val_energy_mae,
        metrics_noneq.val_forces_mae,
        metrics_noneq.val_dipole_mae,
        metrics_noneq.val_esp_mae,
    ]
    
    for idx, (ax, name, dcm_val, noneq_val) in enumerate(zip(axes.flat, metrics_names, dcm_values, noneq_values)):
        x = np.arange(2)
        values = [dcm_val, noneq_val]
        bars = ax.bar(x, values, color=['#2E86AB', '#A23B72'])
        ax.set_xticks(x)
        ax.set_xticklabels(['DCMNet\n(Equivariant)', 'Non-Equivariant'])
        ax.set_ylabel('MAE')
        ax.set_title(name)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_dir / 'performance_comparison.png'}")
    
    # 2. Computational efficiency
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training time
    ax = axes[0]
    x = np.arange(2)
    times = [metrics_dcm.training_time / 3600, metrics_noneq.training_time / 3600]
    bars = ax.bar(x, times, color=['#2E86AB', '#A23B72'])
    ax.set_xticks(x)
    ax.set_xticklabels(['DCMNet', 'Non-Eq'])
    ax.set_ylabel('Hours')
    ax.set_title('Training Time')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}h', ha='center', va='bottom')
    
    # Inference time
    ax = axes[1]
    times_ms = [metrics_dcm.inference_time * 1000, metrics_noneq.inference_time * 1000]
    bars = ax.bar(x, times_ms, color=['#2E86AB', '#A23B72'])
    ax.set_xticks(x)
    ax.set_xticklabels(['DCMNet', 'Non-Eq'])
    ax.set_ylabel('Milliseconds')
    ax.set_title('Inference Time (per sample)')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, times_ms):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}ms', ha='center', va='bottom')
    
    # Parameters
    ax = axes[2]
    params = [metrics_dcm.num_parameters / 1e6, metrics_noneq.num_parameters / 1e6]
    bars = ax.bar(x, params, color=['#2E86AB', '#A23B72'])
    ax.set_xticks(x)
    ax.set_xticklabels(['DCMNet', 'Non-Eq'])
    ax.set_ylabel('Millions')
    ax.set_title('Number of Parameters')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}M', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'efficiency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_dir / 'efficiency_comparison.png'}")
    
    # 3. Equivariance test results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Rotation errors
    ax = axes[0]
    categories = ['Dipole', 'ESP']
    dcm_rot = [metrics_dcm.rotation_error_dipole, metrics_dcm.rotation_error_esp]
    noneq_rot = [metrics_noneq.rotation_error_dipole, metrics_noneq.rotation_error_esp]
    
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, dcm_rot, width, label='DCMNet (Equivariant)', color='#2E86AB')
    ax.bar(x + width/2, noneq_rot, width, label='Non-Equivariant', color='#A23B72')
    ax.set_xlabel('Property')
    ax.set_ylabel('Error after Rotation')
    ax.set_title('Rotation Equivariance Test\n(Lower is better, near-zero is perfect)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    # Translation errors
    ax = axes[1]
    dcm_trans = [metrics_dcm.translation_error_dipole, metrics_dcm.translation_error_esp]
    noneq_trans = [metrics_noneq.translation_error_dipole, metrics_noneq.translation_error_esp]
    
    ax.bar(x - width/2, dcm_trans, width, label='DCMNet', color='#2E86AB')
    ax.bar(x + width/2, noneq_trans, width, label='Non-Equivariant', color='#A23B72')
    ax.set_xlabel('Property')
    ax.set_ylabel('Error after Translation')
    ax.set_title('Translation Invariance Test\n(Both should be near-zero)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'equivariance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_dir / 'equivariance_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DCMNet (equivariant) vs Non-Equivariant models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train-efd', type=Path, required=True,
                       help='Training energies/forces/dipoles NPZ file')
    parser.add_argument('--train-esp', type=Path, required=True,
                       help='Training ESP grids NPZ file')
    parser.add_argument('--valid-efd', type=Path, required=True,
                       help='Validation energies/forces/dipoles NPZ file')
    parser.add_argument('--valid-esp', type=Path, required=True,
                       help='Validation ESP grids NPZ file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for each model')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Model arguments (use same for fair comparison)
    parser.add_argument('--physnet-features', type=int, default=64,
                       help='PhysNet features')
    parser.add_argument('--dcmnet-features', type=int, default=128,
                       help='DCMNet/Non-Eq features')
    parser.add_argument('--n-dcm', type=int, default=3,
                       help='Distributed charges per atom')
    
    # Comparison arguments
    parser.add_argument('--comparison-name', type=str, default='model_comparison',
                       help='Name for comparison results')
    parser.add_argument('--output-dir', type=Path, default=Path('comparisons'),
                       help='Output directory for results')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and load existing checkpoints')
    parser.add_argument('--equivariance-samples', type=int, default=20,
                       help='Number of samples for equivariance testing')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output_dir / args.comparison_name
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("HEAD-TO-HEAD MODEL COMPARISON")
    print("="*70)
    print(f"\nComparison: {args.comparison_name}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print(f"\n{'#'*70}")
    print("# Loading Data")
    print(f"{'#'*70}\n")
    
    train_data = load_combined_data(args.train_efd, args.train_esp, verbose=True)
    valid_data = load_combined_data(args.valid_efd, args.valid_esp, verbose=True)
    
    print(f"\n✅ Data loaded:")
    print(f"  Training samples: {len(train_data['E'])}")
    print(f"  Validation samples: {len(valid_data['E'])}")
    
    # Get dataset properties
    natoms = train_data['R'].shape[1]
    max_atomic_number = int(max(np.max(train_data['Z']), np.max(valid_data['Z'])))
    
    # Shared PhysNet config
    physnet_config = {
        'features': args.physnet_features,
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
    
    # Get recommended optimizer settings
    recommended_config = get_recommended_optimizer_config(
        dataset_size=len(train_data['E']),
        num_features=args.physnet_features + args.dcmnet_features,
        num_atoms=natoms,
        optimizer_name='adamw',
    )
    
    # Model configurations
    dcmnet_config = {
        'features': args.dcmnet_features,
        'max_degree': 2,
        'num_iterations': 2,
        'num_basis_functions': 64,
        'cutoff': 10.0,
        'max_atomic_number': max_atomic_number,
        'n_dcm': args.n_dcm,
        'include_pseudotensors': False,
    }
    
    noneq_config = {
        'features': args.dcmnet_features,  # Use same for fair comparison
        'n_dcm': args.n_dcm,
        'max_atomic_number': max_atomic_number,
        'num_layers': 3,
        'max_displacement': 1.0,
    }
    
    results = {}
    
    # ==================================================================
    # Train/Load DCMNet (Equivariant)
    # ==================================================================
    print(f"\n{'#'*70}")
    print("# DCMNet (Equivariant) Model")
    print(f"{'#'*70}\n")
    
    dcm_ckpt_dir = output_dir / 'dcmnet_equivariant'
    dcm_ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    model_dcm = JointPhysNetDCMNet(
        physnet_config=physnet_config,
        dcmnet_config=dcmnet_config,
        mix_coulomb_energy=False,
    )
    
    if args.skip_training and (dcm_ckpt_dir / 'best_params.pkl').exists():
        print(f"⏩ Loading existing checkpoint: {dcm_ckpt_dir / 'best_params.pkl'}")
        with open(dcm_ckpt_dir / 'best_params.pkl', 'rb') as f:
            params_dcm = pickle.load(f)
        training_time_dcm = 0.0
    else:
        print("🏋️  Training DCMNet...")
        start_time = time.time()
        
        # Create loss terms for training
        dipole_terms = (
            LossTerm(source='physnet', weight=25.0, metric='l2', name='physnet'),
        )
        esp_terms = (
            LossTerm(source='dcmnet', weight=10000.0, metric='l2', name='dcmnet'),
        )
        
        params_dcm = train_model(
            model=model_dcm,
            train_data=train_data,
            valid_data=valid_data,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=recommended_config['learning_rate'],
            weight_decay=recommended_config['weight_decay'],
            energy_w=10.0,
            forces_w=50.0,
            dipole_w=25.0,
            esp_w=10000.0,
            mono_w=100.0,
            n_dcm=args.n_dcm,
            cutoff=10.0,
            seed=args.seed,
            ckpt_dir=dcm_ckpt_dir.parent,
            name=dcm_ckpt_dir.name,
            print_freq=5,
            dipole_terms=dipole_terms,
            esp_terms=esp_terms,
            optimizer_name='adamw',
            optimizer_kwargs={'b1': 0.9, 'b2': 0.999},
        )
        
        training_time_dcm = time.time() - start_time
        print(f"\n✅ DCMNet training completed in {training_time_dcm/3600:.2f} hours")
    
    # ==================================================================
    # Train/Load Non-Equivariant
    # ==================================================================
    print(f"\n{'#'*70}")
    print("# Non-Equivariant Model")
    print(f"{'#'*70}\n")
    
    noneq_ckpt_dir = output_dir / 'noneq_model'
    noneq_ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    model_noneq = JointPhysNetNonEquivariant(
        physnet_config=physnet_config,
        noneq_config=noneq_config,
        mix_coulomb_energy=False,
    )
    
    if args.skip_training and (noneq_ckpt_dir / 'best_params.pkl').exists():
        print(f"⏩ Loading existing checkpoint: {noneq_ckpt_dir / 'best_params.pkl'}")
        with open(noneq_ckpt_dir / 'best_params.pkl', 'rb') as f:
            params_noneq = pickle.load(f)
        training_time_noneq = 0.0
    else:
        print("🏋️  Training Non-Equivariant model...")
        start_time = time.time()
        
        # Create loss terms for training (same as DCMNet for fair comparison)
        dipole_terms = (
            LossTerm(source='physnet', weight=25.0, metric='l2', name='physnet'),
        )
        esp_terms = (
            LossTerm(source='dcmnet', weight=10000.0, metric='l2', name='dcmnet'),
        )
        
        params_noneq = train_model(
            model=model_noneq,
            train_data=train_data,
            valid_data=valid_data,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=recommended_config['learning_rate'],
            weight_decay=recommended_config['weight_decay'],
            energy_w=10.0,
            forces_w=50.0,
            dipole_w=25.0,
            esp_w=10000.0,
            mono_w=100.0,
            n_dcm=args.n_dcm,
            cutoff=10.0,
            seed=args.seed,
            ckpt_dir=noneq_ckpt_dir.parent,
            name=noneq_ckpt_dir.name,
            print_freq=5,
            dipole_terms=dipole_terms,
            esp_terms=esp_terms,
            optimizer_name='adamw',
            optimizer_kwargs={'b1': 0.9, 'b2': 0.999},
        )
        
        training_time_noneq = time.time() - start_time
        print(f"\n✅ Non-Equivariant training completed in {training_time_noneq/3600:.2f} hours")
    
    # ==================================================================
    # Run Equivariance Tests
    # ==================================================================
    print(f"\n{'#'*70}")
    print("# Equivariance Testing")
    print(f"{'#'*70}")
    
    print("\n--- DCMNet (Equivariant) ---")
    equivariance_dcm = test_equivariance(
        model_dcm,
        params_dcm,
        valid_data,
        num_test_samples=args.equivariance_samples,
        seed=args.seed,
    )
    
    print("\n--- Non-Equivariant ---")
    equivariance_noneq = test_equivariance(
        model_noneq,
        params_noneq,
        valid_data,
        num_test_samples=args.equivariance_samples,
        seed=args.seed,
    )
    
    # ==================================================================
    # Measure Performance
    # ==================================================================
    print(f"\n{'#'*70}")
    print("# Performance Benchmarking")
    print(f"{'#'*70}\n")
    
    print("Measuring inference times...")
    inference_time_dcm = measure_inference_time(model_dcm, params_dcm, valid_data, num_samples=50)
    inference_time_noneq = measure_inference_time(model_noneq, params_noneq, valid_data, num_samples=50)
    
    print(f"  DCMNet:        {inference_time_dcm*1000:.2f} ms/sample")
    print(f"  Non-Eq:        {inference_time_noneq*1000:.2f} ms/sample")
    print(f"  Speedup:       {inference_time_dcm/inference_time_noneq:.2f}×")
    
    num_params_dcm = count_parameters(params_dcm)
    num_params_noneq = count_parameters(params_noneq)
    
    print(f"\nParameter counts:")
    print(f"  DCMNet:        {num_params_dcm:,} parameters")
    print(f"  Non-Eq:        {num_params_noneq:,} parameters")
    print(f"  Reduction:     {(1 - num_params_noneq/num_params_dcm)*100:.1f}%")
    
    # Load validation metrics from history (with fallback for missing files)
    history_dcm_path = dcm_ckpt_dir / 'history.json'
    history_noneq_path = noneq_ckpt_dir / 'history.json'
    
    if history_dcm_path.exists():
        with open(history_dcm_path, 'r') as f:
            history_dcm = json.load(f)
        best_epoch_dcm = np.argmin(history_dcm['val_loss'])
        val_energy_mae_dcm = history_dcm['val_energy_mae'][best_epoch_dcm]
        val_forces_mae_dcm = history_dcm['val_forces_mae'][best_epoch_dcm]
        val_dipole_mae_dcm = history_dcm['val_dipole_mae'][best_epoch_dcm]
        val_esp_mae_dcm = history_dcm['val_esp_mae'][best_epoch_dcm]
    else:
        print(f"\n⚠️  Warning: No history file found for DCMNet at {history_dcm_path}")
        print("   Using placeholder values (0.0) for validation metrics.")
        print("   Rerun comparison to get full metrics.")
        val_energy_mae_dcm = 0.0
        val_forces_mae_dcm = 0.0
        val_dipole_mae_dcm = 0.0
        val_esp_mae_dcm = 0.0
    
    if history_noneq_path.exists():
        with open(history_noneq_path, 'r') as f:
            history_noneq = json.load(f)
        best_epoch_noneq = np.argmin(history_noneq['val_loss'])
        val_energy_mae_noneq = history_noneq['val_energy_mae'][best_epoch_noneq]
        val_forces_mae_noneq = history_noneq['val_forces_mae'][best_epoch_noneq]
        val_dipole_mae_noneq = history_noneq['val_dipole_mae'][best_epoch_noneq]
        val_esp_mae_noneq = history_noneq['val_esp_mae'][best_epoch_noneq]
    else:
        print(f"\n⚠️  Warning: No history file found for Non-Eq at {history_noneq_path}")
        print("   Using placeholder values (0.0) for validation metrics.")
        print("   Rerun comparison to get full metrics.")
        val_energy_mae_noneq = 0.0
        val_forces_mae_noneq = 0.0
        val_dipole_mae_noneq = 0.0
        val_esp_mae_noneq = 0.0
    
    # Create metrics objects
    metrics_dcm = ModelMetrics(
        name='DCMNet (Equivariant)',
        training_time=training_time_dcm,
        inference_time=inference_time_dcm,
        memory_usage_mb=0.0,  # Would need profiling
        num_parameters=num_params_dcm,
        val_energy_mae=val_energy_mae_dcm,
        val_forces_mae=val_forces_mae_dcm,
        val_dipole_mae=val_dipole_mae_dcm,
        val_esp_mae=val_esp_mae_dcm,
        rotation_error_dipole=equivariance_dcm['rotation_error_dipole'],
        rotation_error_esp=equivariance_dcm['rotation_error_esp'],
        translation_error_dipole=equivariance_dcm['translation_error_dipole'],
        translation_error_esp=equivariance_dcm['translation_error_esp'],
    )
    
    metrics_noneq = ModelMetrics(
        name='Non-Equivariant',
        training_time=training_time_noneq,
        inference_time=inference_time_noneq,
        memory_usage_mb=0.0,
        num_parameters=num_params_noneq,
        val_energy_mae=val_energy_mae_noneq,
        val_forces_mae=val_forces_mae_noneq,
        val_dipole_mae=val_dipole_mae_noneq,
        val_esp_mae=val_esp_mae_noneq,
        rotation_error_dipole=equivariance_noneq['rotation_error_dipole'],
        rotation_error_esp=equivariance_noneq['rotation_error_esp'],
        translation_error_dipole=equivariance_noneq['translation_error_dipole'],
        translation_error_esp=equivariance_noneq['translation_error_esp'],
    )
    
    # ==================================================================
    # Generate Comparison Report
    # ==================================================================
    print(f"\n{'#'*70}")
    print("# Final Comparison Report")
    print(f"{'#'*70}\n")
    
    print("VALIDATION PERFORMANCE:")
    print(f"  Energy MAE:  DCMNet={metrics_dcm.val_energy_mae:.4f}, Non-Eq={metrics_noneq.val_energy_mae:.4f}")
    print(f"  Forces MAE:  DCMNet={metrics_dcm.val_forces_mae:.4f}, Non-Eq={metrics_noneq.val_forces_mae:.4f}")
    print(f"  Dipole MAE:  DCMNet={metrics_dcm.val_dipole_mae:.4f}, Non-Eq={metrics_noneq.val_dipole_mae:.4f}")
    print(f"  ESP MAE:     DCMNet={metrics_dcm.val_esp_mae:.4f}, Non-Eq={metrics_noneq.val_esp_mae:.4f}")
    
    print("\nEQUIVARIANCE (Rotation):")
    print(f"  Dipole:      DCMNet={metrics_dcm.rotation_error_dipole:.6f}, Non-Eq={metrics_noneq.rotation_error_dipole:.6f}")
    print(f"  ESP:         DCMNet={metrics_dcm.rotation_error_esp:.6f}, Non-Eq={metrics_noneq.rotation_error_esp:.6f}")
    print("  ⚠️  DCMNet should have near-zero rotation error (equivariant)")
    print("  ⚠️  Non-Eq will have larger error (not equivariant)")
    
    print("\nINVARIANCE (Translation):")
    print(f"  Dipole:      DCMNet={metrics_dcm.translation_error_dipole:.6f}, Non-Eq={metrics_noneq.translation_error_dipole:.6f}")
    print(f"  ESP:         DCMNet={metrics_dcm.translation_error_esp:.6f}, Non-Eq={metrics_noneq.translation_error_esp:.6f}")
    print("  ✅ Both should have near-zero translation error")
    
    print("\nCOMPUTATIONAL EFFICIENCY:")
    if training_time_dcm > 0 and training_time_noneq > 0:
        speedup = training_time_dcm / training_time_noneq
        print(f"  Training:    DCMNet={training_time_dcm/3600:.2f}h, Non-Eq={training_time_noneq/3600:.2f}h ({speedup:.2f}× speedup)")
    print(f"  Inference:   DCMNet={inference_time_dcm*1000:.2f}ms, Non-Eq={inference_time_noneq*1000:.2f}ms ({inference_time_dcm/inference_time_noneq:.2f}× speedup)")
    print(f"  Parameters:  DCMNet={num_params_dcm:,}, Non-Eq={num_params_noneq:,} ({(1-num_params_noneq/num_params_dcm)*100:.1f}% reduction)")
    
    # Save results
    # Convert Path objects to strings for JSON serialization
    args_dict = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            args_dict[key] = str(value)
        else:
            args_dict[key] = value
    
    results = {
        'dcmnet': asdict(metrics_dcm),
        'noneq': asdict(metrics_noneq),
        'args': args_dict,
    }
    
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_dir / 'comparison_results.json'}")
    
    # Generate plots
    print(f"\n{'#'*70}")
    print("# Generating Comparison Plots")
    print(f"{'#'*70}\n")
    
    plot_comparison(metrics_dcm, metrics_noneq, output_dir)
    
    print(f"\n{'='*70}")
    print("✅ COMPARISON COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - comparison_results.json")
    print(f"  - performance_comparison.png")
    print(f"  - efficiency_comparison.png")
    print(f"  - equivariance_comparison.png")
    print(f"  - dcmnet_equivariant/ (checkpoint)")
    print(f"  - noneq_model/ (checkpoint)")


if __name__ == '__main__':
    main()

