#!/usr/bin/env python3
"""
Evaluate Model on Train/Valid/Test Splits and Extract Geometric Features

Computes errors for energy, forces, dipoles, ESP across data splits,
and extracts geometric features (r1, r2, angle, r1+r2, r1*r2) for plotting.

Usage:
    python evaluate_splits.py \
        --checkpoint ./ckpts/model \
        --train-efd ./data/train_efd.npz \
        --valid-efd ./data/valid_efd.npz \
        --test-efd ./data/test_efd.npz \
        --train-esp ./data/train_esp.npz \
        --valid-esp ./data/valid_esp.npz \
        --test-esp ./data/test_esp.npz \
        --output-dir ./evaluation
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import argparse
from typing import Dict, List, Tuple
import pandas as pd

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from trainer import JointPhysNetDCMNet, prepare_batch_data
from mmml.dcmnet.dcmnet.electrostatics import calc_esp


def compute_bond_angle_features(positions: np.ndarray, atomic_numbers: np.ndarray) -> Dict:
    """
    Compute geometric features for CO2 molecule.
    
    For CO2: C-O1-O2
    Returns:
        r1: C-O1 distance
        r2: C-O2 distance
        angle: O1-C-O2 angle (degrees)
        r1_plus_r2: r1 + r2
        r1_times_r2: r1 * r2
    """
    n_atoms = len(atomic_numbers)
    
    if n_atoms != 3:
        raise ValueError(f"Expected 3 atoms for CO2, got {n_atoms}")
    
    # Find C and O positions
    c_idx = None
    o_indices = []
    for i, z in enumerate(atomic_numbers):
        if z == 6:  # Carbon
            c_idx = i
        elif z == 8:  # Oxygen
            o_indices.append(i)
    
    if c_idx is None or len(o_indices) != 2:
        raise ValueError("Could not identify CO2 structure (need 1 C, 2 O)")
    
    c_pos = positions[c_idx]
    o1_pos = positions[o_indices[0]]
    o2_pos = positions[o_indices[1]]
    
    # Compute distances
    r1 = np.linalg.norm(o1_pos - c_pos)
    r2 = np.linalg.norm(o2_pos - c_pos)
    
    # Compute angle
    vec1 = o1_pos - c_pos
    vec2 = o2_pos - c_pos
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    angle = np.arccos(cos_angle) * 180 / np.pi  # Convert to degrees
    
    # Derived features
    r1_plus_r2 = r1 + r2
    r1_times_r2 = r1 * r2
    
    return {
        'r1': r1,
        'r2': r2,
        'angle': angle,
        'r1_plus_r2': r1_plus_r2,
        'r1_times_r2': r1_times_r2,
    }


def evaluate_split(params, model, efd_data, esp_data, split_name: str, 
                   cutoff: float = 10.0, natoms: int = None, batch_size: int = 100) -> pd.DataFrame:
    """
    Evaluate model on a data split and return DataFrame with errors and features.
    """
    print(f"\nEvaluating {split_name} split ({len(efd_data['E'])} structures)...")
    
    # Extract data
    R = efd_data['R']  # (n_structures, n_atoms, 3)
    Z = efd_data['Z']  # (n_structures, n_atoms)
    E_true = efd_data['E']
    F_true = efd_data['F']
    
    # Handle different dipole key names
    if 'D' in efd_data:
        D_true = efd_data['D']
    elif 'Dxyz' in efd_data:
        D_true = efd_data['Dxyz']
    else:
        print(f"  Warning: No dipole data found. Available keys: {list(efd_data.keys())}")
        D_true = None
    
    # ESP data
    vdw_surface = esp_data['vdw_surface'] if 'vdw_surface' in esp_data else None
    esp_true = esp_data['esp'] if 'esp' in esp_data else None
    
    # Pre-compute edge lists for all structures
    print(f"  Pre-computing edge lists...")
    all_edge_lists = []
    for i in range(len(R)):
        positions = R[i]
        n_atoms = len(positions)
        
        dst_list = []
        src_list = []
        for a in range(n_atoms):
            for b in range(n_atoms):
                if a != b:
                    dist = np.linalg.norm(positions[a] - positions[b])
                    if dist < cutoff:
                        dst_list.append(a)
                        src_list.append(b)
        
        all_edge_lists.append({
            'dst_idx': np.array(dst_list, dtype=np.int32),
            'src_idx': np.array(src_list, dtype=np.int32),
        })
    
    # Evaluate in batches
    results = []
    
    for batch_start in range(0, len(R), batch_size):
        batch_end = min(batch_start + batch_size, len(R))
        batch_indices = np.arange(batch_start, batch_end)
        
        if (batch_start // batch_size + 1) % 10 == 0:
            print(f"  Processing batch {batch_start // batch_size + 1}...")
        
        for idx in batch_indices:
            positions = R[idx]
            atomic_numbers = Z[idx]
            n_atoms = len(positions)
            
            # Get edge list (built from original positions, not padded)
            edge_data = all_edge_lists[idx]
            dst_idx = edge_data['dst_idx']
            src_idx = edge_data['src_idx']
            
            # Prepare batch data - no padding, use actual molecule size
            batch_segments = np.zeros(n_atoms, dtype=np.int32)
            batch_mask = np.ones(len(dst_idx), dtype=np.float32)
            atom_mask = np.ones(n_atoms, dtype=np.float32)
            
            # Run model
            output = model.apply(
                params,
                atomic_numbers=jnp.array(atomic_numbers),
                positions=jnp.array(positions),
                dst_idx=jnp.array(dst_idx),
                src_idx=jnp.array(src_idx),
                batch_segments=jnp.array(batch_segments),
                batch_size=1,
                batch_mask=jnp.array(batch_mask),
                atom_mask=jnp.array(atom_mask),
            )
            
            # Extract predictions
            E_pred = float(output['energy'][0])
            F_pred = np.array(output['forces'][:n_atoms])
            D_physnet = np.array(output['dipoles'][0])
            charges_physnet = np.array(output['charges_as_mono'][:n_atoms])
            
            # DCMNet outputs
            mono_dist = np.array(output['mono_dist'][:n_atoms])
            dipo_dist = np.array(output['dipo_dist'][:n_atoms])
            
            # Compute DCMNet dipole
            import ase.data
            masses = np.array([ase.data.atomic_masses[z] for z in atomic_numbers])
            com = np.sum(positions * masses[:, None], axis=0) / masses.sum()
            dipo_rel_com = dipo_dist - com[None, None, :]
            D_dcmnet = np.sum(mono_dist[..., None] * dipo_rel_com, axis=(0, 1))
            
            # Compute errors
            E_error = E_pred - E_true[idx]
            F_error = F_pred - F_true[idx]
            F_norm_error = np.linalg.norm(F_pred) - np.linalg.norm(F_true[idx])
            
            # Dipole errors (if available)
            if D_true is not None:
                D_physnet_error = D_physnet - D_true[idx]
                D_physnet_norm_error = np.linalg.norm(D_physnet) - np.linalg.norm(D_true[idx])
                
                D_dcmnet_error = D_dcmnet - D_true[idx]
                D_dcmnet_norm_error = np.linalg.norm(D_dcmnet) - np.linalg.norm(D_true[idx])
                
                dipole_true_norm = np.linalg.norm(D_true[idx])
            else:
                D_physnet_norm_error = np.nan
                D_dcmnet_norm_error = np.nan
                dipole_true_norm = np.nan
            
            # ESP errors (if available)
            esp_error_physnet = None
            esp_error_dcmnet = None
            esp_rmse_physnet = None
            esp_rmse_dcmnet = None
            
            if vdw_surface is not None and esp_true is not None:
                grid_points = vdw_surface[idx]
                esp_true_vals = esp_true[idx]
                
                # Filter valid grid points (remove padding zeros)
                valid_mask = np.any(grid_points != 0, axis=1)
                grid_points = grid_points[valid_mask]
                esp_true_vals = esp_true_vals[valid_mask]
                
                if len(grid_points) > 0:
                    # PhysNet ESP (from point charges)
                    esp_pred_physnet = np.sum(
                        charges_physnet[None, :] / (
                            np.linalg.norm(
                                grid_points[:, None, :] - positions[None, :, :],
                                axis=2
                            ) * 1.88973 + 1e-10
                        ),
                        axis=1
                    )
                    
                    # DCMNet ESP (from distributed charges)
                    # Compute distributed charge positions
                    n_dcm = mono_dist.shape[1]
                    positions_dcmnet_list = []
                    for atom_idx in range(n_atoms):
                        for dcm_idx in range(n_dcm):
                            # Distributed charge position relative to atom
                            dist_pos = positions[atom_idx] + dipo_dist[atom_idx, dcm_idx]
                            positions_dcmnet_list.append(dist_pos)
                    positions_dcmnet_flat = np.array(positions_dcmnet_list)
                    charges_dcmnet_flat = mono_dist.flatten()
                    
                    esp_pred_dcmnet = calc_esp(
                        jnp.array(positions_dcmnet_flat),
                        jnp.array(charges_dcmnet_flat),
                        jnp.array(grid_points)
                    )
                    esp_pred_dcmnet = np.array(esp_pred_dcmnet)
                    
                    # Compute RMSE
                    esp_rmse_physnet = np.sqrt(np.mean((esp_pred_physnet - esp_true_vals)**2))
                    esp_rmse_dcmnet = np.sqrt(np.mean((esp_pred_dcmnet - esp_true_vals)**2))
            
            # Compute geometric features
            features = compute_bond_angle_features(positions, atomic_numbers)
            
            # Store results
            results.append({
                'split': split_name,
                'index': idx,
                # Geometric features
                'r1': features['r1'],
                'r2': features['r2'],
                'angle': features['angle'],
                'r1_plus_r2': features['r1_plus_r2'],
                'r1_times_r2': features['r1_times_r2'],
                # Energy
                'energy_true': E_true[idx],
                'energy_pred': E_pred,
                'energy_error': E_error,
                'energy_abs_error': abs(E_error),
                # Forces
                'force_norm_true': np.linalg.norm(F_true[idx]),
                'force_norm_pred': np.linalg.norm(F_pred),
                'force_norm_error': F_norm_error,
                'force_norm_abs_error': abs(F_norm_error),
                'force_max_abs_error': np.max(np.abs(F_error)),
                # Dipole (PhysNet)
                'dipole_physnet_norm_true': dipole_true_norm,
                'dipole_physnet_norm_pred': np.linalg.norm(D_physnet),
                'dipole_physnet_norm_error': D_physnet_norm_error,
                'dipole_physnet_norm_abs_error': abs(D_physnet_norm_error) if not np.isnan(D_physnet_norm_error) else np.nan,
                # Dipole (DCMNet)
                'dipole_dcmnet_norm_true': dipole_true_norm,
                'dipole_dcmnet_norm_pred': np.linalg.norm(D_dcmnet),
                'dipole_dcmnet_norm_error': D_dcmnet_norm_error,
                'dipole_dcmnet_norm_abs_error': abs(D_dcmnet_norm_error) if not np.isnan(D_dcmnet_norm_error) else np.nan,
                # ESP
                'esp_rmse_physnet': esp_rmse_physnet if esp_rmse_physnet is not None else np.nan,
                'esp_rmse_dcmnet': esp_rmse_dcmnet if esp_rmse_dcmnet is not None else np.nan,
            })
    
    df = pd.DataFrame(results)
    print(f"  ✅ Evaluated {len(results)} structures")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on data splits')
    
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Model checkpoint directory')
    
    # Data splits
    parser.add_argument('--train-efd', type=Path, required=True,
                       help='Training energy/forces/dipole data')
    parser.add_argument('--valid-efd', type=Path, required=True,
                       help='Validation energy/forces/dipole data')
    parser.add_argument('--test-efd', type=Path, default=None,
                       help='Test energy/forces/dipole data (optional)')
    
    parser.add_argument('--train-esp', type=Path, required=True,
                       help='Training ESP data')
    parser.add_argument('--valid-esp', type=Path, required=True,
                       help='Validation ESP data')
    parser.add_argument('--test-esp', type=Path, default=None,
                       help='Test ESP data (optional)')
    
    parser.add_argument('--cutoff', type=float, default=10.0,
                       help='Cutoff distance for edge list')
    parser.add_argument('--natoms', type=int, default=None,
                       help='[DEPRECATED] No longer needed - model handles variable sizes dynamically')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=Path, default=Path('./evaluation'),
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MODEL EVALUATION ON DATA SPLITS")
    print("="*70)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n1. Loading model from {args.checkpoint}...")
    with open(args.checkpoint / 'best_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    with open(args.checkpoint / 'model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config'],
        mix_coulomb_energy=config.get('mix_coulomb_energy', False)
    )
    
    print(f"✅ Loaded model with {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")
    
    # Load data splits
    print(f"\n2. Loading data splits...")
    splits_efd = {
        'train': np.load(args.train_efd),
        'valid': np.load(args.valid_efd),
    }
    splits_esp = {
        'train': np.load(args.train_esp),
        'valid': np.load(args.valid_esp),
    }
    
    if args.test_efd and args.test_esp:
        splits_efd['test'] = np.load(args.test_efd)
        splits_esp['test'] = np.load(args.test_esp)
        print(f"✅ Loaded train/valid/test splits")
    else:
        print(f"✅ Loaded train/valid splits")
    
    # Evaluate each split
    print(f"\n3. Evaluating model on splits...")
    all_results = []
    
    for split_name in splits_efd.keys():
        df = evaluate_split(
            params, model,
            splits_efd[split_name],
            splits_esp[split_name],
            split_name,
            cutoff=args.cutoff,
            natoms=args.natoms,
            batch_size=args.batch_size,
        )
        all_results.append(df)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV for R
    csv_file = args.output_dir / 'evaluation_results.csv'
    combined_df.to_csv(csv_file, index=False)
    print(f"\n✅ Saved evaluation results: {csv_file}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    for split_name in splits_efd.keys():
        split_df = combined_df[combined_df['split'] == split_name]
        print(f"\n{split_name.upper()} split ({len(split_df)} structures):")
        print(f"  Energy MAE: {split_df['energy_abs_error'].mean():.6f} eV")
        print(f"  Force norm MAE: {split_df['force_norm_abs_error'].mean():.6f} eV/Å")
        
        dipole_phys = split_df['dipole_physnet_norm_abs_error'].dropna()
        dipole_dcm = split_df['dipole_dcmnet_norm_abs_error'].dropna()
        if len(dipole_phys) > 0:
            print(f"  Dipole (PhysNet) norm MAE: {dipole_phys.mean():.6f} e·Å")
            print(f"  Dipole (DCMNet) norm MAE: {dipole_dcm.mean():.6f} e·Å")
        else:
            print(f"  Dipole data: Not available")
        
        esp_phys = split_df['esp_rmse_physnet'].dropna()
        esp_dcm = split_df['esp_rmse_dcmnet'].dropna()
        if len(esp_phys) > 0:
            print(f"  ESP RMSE (PhysNet): {esp_phys.mean():.6f} Ha/e")
            print(f"  ESP RMSE (DCMNet): {esp_dcm.mean():.6f} Ha/e")
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nNext step: Run R plotting script:")
    print(f"  Rscript plot_evaluation_results.R {args.output_dir}")


if __name__ == '__main__':
    main()

