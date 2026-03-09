#!/usr/bin/env python3
"""
Gradient Boosting Model for Predicting Atom-Centered Charges

This script trains a gradient boosting regressor to predict atomic monopoles
(charges) from molecular geometry (R, Z).

Input: R (atomic positions), Z (atomic numbers)
Output: mono (atomic charges per atom)

For CO2: predicts charges for atoms 0 (C), 1 (O), 2 (O)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle
import argparse

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def compute_molecular_features(R: np.ndarray, Z: np.ndarray, n_atoms: int = 3) -> np.ndarray:
    """
    Compute molecular features from atomic positions and numbers.
    
    For CO2 (3 atoms), computes:
    - Atomic numbers for each atom
    - Interatomic distances
    - Bond angles
    - Center of mass
    - Molecular geometry descriptors
    
    Parameters
    ----------
    R : np.ndarray
        Atomic positions, shape (n_samples, n_atoms, 3) or (n_atoms, 3)
    Z : np.ndarray
        Atomic numbers, shape (n_samples, n_atoms) or (n_atoms,)
    n_atoms : int
        Number of atoms (3 for CO2)
        
    Returns
    -------
    np.ndarray
        Feature matrix, shape (n_samples, n_features)
    """
    # Handle single sample vs batch
    if R.ndim == 2:
        R = R[None, :, :]
        Z = Z[None, :]
    
    n_samples = R.shape[0]
    features_list = []
    
    for i in range(n_samples):
        r = R[i]  # (n_atoms, 3)
        z = Z[i]  # (n_atoms,)
        
        # Only use valid atoms (Z > 0)
        valid_mask = z > 0
        r_valid = r[valid_mask]
        z_valid = z[valid_mask]
        n_valid = np.sum(valid_mask)
        
        sample_features = []
        
        # 1. Atomic numbers (one-hot or direct)
        for atom_idx in range(min(n_atoms, n_valid)):
            if atom_idx < n_valid:
                sample_features.append(float(z_valid[atom_idx]))
            else:
                sample_features.append(0.0)
        
        # 2. Interatomic distances
        # For CO2: we need distances: C-O1, C-O2, O1-O2
        n_pairs_needed = n_atoms * (n_atoms - 1) // 2
        if n_valid >= 2:
            # All pairwise distances
            pair_count = 0
            for i_atom in range(min(n_atoms, n_valid)):
                for j_atom in range(i_atom + 1, min(n_atoms, n_valid)):
                    dist = np.linalg.norm(r_valid[i_atom] - r_valid[j_atom])
                    sample_features.append(dist)
                    pair_count += 1
            # Pad remaining pairs with zeros
            for _ in range(n_pairs_needed - pair_count):
                sample_features.append(0.0)
        else:
            # Pad with zeros if not enough atoms
            sample_features.extend([0.0] * n_pairs_needed)
        
        # 3. Bond angles (for CO2: C-O-O angle)
        if n_valid >= 3:
            # For CO2: C (idx 0) - O (idx 1) - O (idx 2)
            # Angle at C atom
            vec1 = r_valid[1] - r_valid[0]  # C -> O1
            vec2 = r_valid[2] - r_valid[0]  # C -> O2
            
            # Normalize
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                sample_features.append(angle)
            else:
                sample_features.append(0.0)
        else:
            sample_features.append(0.0)
        
        # 4. Center of mass
        if n_valid > 0:
            com = np.mean(r_valid, axis=0)
            sample_features.extend(com.tolist())
        else:
            sample_features.extend([0.0, 0.0, 0.0])
        
        # 5. Molecular size descriptors
        if n_valid > 0:
            # Radius of gyration
            com = np.mean(r_valid, axis=0)
            rg = np.sqrt(np.mean(np.sum((r_valid - com)**2, axis=1)))
            sample_features.append(rg)
            
            # Maximum distance between atoms
            if n_valid >= 2:
                max_dist = 0.0
                for i_atom in range(n_valid):
                    for j_atom in range(i_atom + 1, n_valid):
                        dist = np.linalg.norm(r_valid[i_atom] - r_valid[j_atom])
                        max_dist = max(max_dist, dist)
                sample_features.append(max_dist)
            else:
                sample_features.append(0.0)
        else:
            sample_features.extend([0.0, 0.0])
        
        features_list.append(sample_features)
    
    return np.array(features_list)


def load_charge_data(csv_file: Path, scheme: Optional[str] = None, level: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load charge data from CSV file.
    
    Expected CSV format:
    r1, r2, ang, level, atom_index, atom, scheme, value
    
    Parameters
    ----------
    csv_file : Path
        Path to CSV file
    scheme : str, optional
        Charge scheme to use (e.g., 'Hirshfeld', 'VDD'). If None, uses first available.
    level : str, optional
        Quantum chemistry level to use (e.g., 'hf', 'mp2'). If None, uses all levels.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (R, Z, mono) where:
        - R: atomic positions (n_samples, 3, 3) - assuming CO2
        - Z: atomic numbers (n_samples, 3)
        - mono: atomic charges (n_samples, 3)
    """
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} rows from {csv_file}")
    print(f"Available schemes: {df['scheme'].unique()}")
    print(f"Available levels: {df['level'].unique()}")
    
    # Filter by level if specified
    if level is not None:
        df = df[df['level'] == level]
        print(f"Using level: {level}, {len(df)} rows")
    
    # Filter by scheme if specified
    if scheme is not None:
        df = df[df['scheme'] == scheme]
        print(f"Using scheme: {scheme}, {len(df)} rows")
    else:
        # Use first available scheme
        scheme = df['scheme'].iloc[0]
        print(f"Using first available scheme: {scheme}")
    
    # Group by geometry (r1, r2, ang, level) to get unique conformations
    # Note: level (hf, mp2) might have different charges for same geometry
    geometry_cols = ['r1', 'r2', 'ang', 'level']
    unique_geometries = df[geometry_cols].drop_duplicates()
    
    print(f"Found {len(unique_geometries)} unique geometry+level combinations")
    
    R_list = []
    Z_list = []
    mono_list = []
    
    for _, geom_row in unique_geometries.iterrows():
        # Get all atoms for this geometry and level
        geom_data = df[
            (df['r1'] == geom_row['r1']) &
            (df['r2'] == geom_row['r2']) &
            (df['ang'] == geom_row['ang']) &
            (df['level'] == geom_row['level']) &
            (df['scheme'] == scheme)
        ].sort_values('atom_index')
        
        if len(geom_data) < 3:
            continue
        
        # Extract atomic numbers
        atom_names = geom_data['atom'].values[:3]
        Z = np.array([6 if a == 'C' else 8 if a == 'O' else 1 for a in atom_names])  # C=6, O=8
        
        # Extract charges
        charges = geom_data['value'].values[:3]
        
        # Reconstruct positions from geometry
        # For CO2: C at origin, O1 at r1 along x-axis, O2 at r2 with angle
        r1 = geom_row['r1']
        r2 = geom_row['r2']
        angle_deg = geom_row['ang']
        angle_rad = np.deg2rad(angle_deg)
        
        # C at origin
        R = np.zeros((3, 3))
        R[0] = [0.0, 0.0, 0.0]  # C
        
        # O1 at r1 along x-axis
        R[1] = [r1, 0.0, 0.0]  # O1
        
        # O2 at r2 with angle from x-axis
        R[2] = [r2 * np.cos(angle_rad), r2 * np.sin(angle_rad), 0.0]  # O2
        
        R_list.append(R)
        Z_list.append(Z)
        mono_list.append(charges)
    
    R = np.array(R_list)
    Z = np.array(Z_list)
    mono = np.array(mono_list)
    
    print(f"\nPrepared data:")
    print(f"  R shape: {R.shape}")
    print(f"  Z shape: {Z.shape}")
    print(f"  mono shape: {mono.shape}")
    
    return R, Z, mono


def train_charge_predictor(
    R: np.ndarray,
    Z: np.ndarray,
    mono: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    save_path: Optional[Path] = None
) -> Dict:
    """
    Train gradient boosting models to predict atomic charges.
    
    Trains separate models for each atom (0, 1, 2 for CO2).
    
    Parameters
    ----------
    R : np.ndarray
        Atomic positions, shape (n_samples, n_atoms, 3)
    Z : np.ndarray
        Atomic numbers, shape (n_samples, n_atoms)
    mono : np.ndarray
        Atomic charges, shape (n_samples, n_atoms)
    test_size : float
        Fraction of data for testing
    random_state : int
        Random seed
    n_estimators : int
        Number of boosting stages
    learning_rate : float
        Learning rate
    max_depth : int
        Maximum depth of trees
    save_path : Path, optional
        Path to save the trained models
        
    Returns
    -------
    Dict
        Dictionary containing trained models and metrics
    """
    print("\n" + "="*70)
    print("Training Gradient Boosting Charge Predictors")
    print("="*70)
    
    # Compute features
    print("\nComputing molecular features...")
    X = compute_molecular_features(R, Z, n_atoms=Z.shape[1])
    print(f"Feature matrix shape: {X.shape}")
    
    n_atoms = mono.shape[1]
    models = {}
    results = {}
    
    # Train separate model for each atom
    for atom_idx in range(n_atoms):
        print(f"\n{'='*70}")
        print(f"Training model for atom {atom_idx} (Z={Z[0, atom_idx]})")
        print(f"{'='*70}")
        
        y = mono[:, atom_idx]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Charge range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"  Charge mean: {y.mean():.4f}, std: {y.std():.4f}")
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            subsample=0.8,
            verbose=1
        )
        
        print("\n  Training...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"\n  Training Metrics:")
        print(f"    MAE:  {train_mae:.6f}")
        print(f"    RMSE: {train_rmse:.6f}")
        print(f"    R²:   {train_r2:.6f}")
        
        print(f"\n  Test Metrics:")
        print(f"    MAE:  {test_mae:.6f}")
        print(f"    RMSE: {test_rmse:.6f}")
        print(f"    R²:   {test_r2:.6f}")
        
        models[f'atom_{atom_idx}'] = model
        results[f'atom_{atom_idx}'] = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }
    
    # Save models if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Saving models to {save_path}")
        print(f"{'='*70}")
        
        joblib.dump(models, save_path)
        
        # Also save metadata
        metadata = {
            'n_atoms': n_atoms,
            'feature_names': ['Z0', 'Z1', 'Z2', 'dist_01', 'dist_02', 'dist_12', 
                            'angle', 'com_x', 'com_y', 'com_z', 'rg', 'max_dist'],
            'results': results
        }
        metadata_path = save_path.parent / f"{save_path.stem}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved models and metadata")
    
    return {
        'models': models,
        'results': results,
        'feature_matrix': X
    }


def predict_charges(
    models: Dict,
    R: np.ndarray,
    Z: np.ndarray
) -> np.ndarray:
    """
    Predict atomic charges using trained models.
    
    Parameters
    ----------
    models : Dict
        Dictionary of trained models (from train_charge_predictor)
    R : np.ndarray
        Atomic positions, shape (n_samples, n_atoms, 3) or (n_atoms, 3)
    Z : np.ndarray
        Atomic numbers, shape (n_samples, n_atoms) or (n_atoms,)
        
    Returns
    -------
    np.ndarray
        Predicted charges, shape (n_samples, n_atoms)
    """
    # Compute features
    X = compute_molecular_features(R, Z, n_atoms=Z.shape[-1])
    
    # Predict for each atom
    n_atoms = len([k for k in models.keys() if k.startswith('atom_')])
    predictions = []
    
    for atom_idx in range(n_atoms):
        model = models[f'atom_{atom_idx}']
        pred = model.predict(X)
        predictions.append(pred)
    
    return np.array(predictions).T


def main():
    parser = argparse.ArgumentParser(
        description="Train gradient boosting model to predict atomic charges from molecular geometry"
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to CSV file with charge data'
    )
    parser.add_argument(
        '--scheme',
        type=str,
        default=None,
        help='Charge scheme to use (e.g., Hirshfeld, VDD). If not specified, uses first available.'
    )
    parser.add_argument(
        '--level',
        type=str,
        default=None,
        help='Quantum chemistry level to use (e.g., hf, mp2). If not specified, uses all levels.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='charge_predictor_model.pkl',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of boosting stages'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=5,
        help='Maximum depth of trees'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Load data
    print("="*70)
    print("Loading Charge Data")
    print("="*70)
    R, Z, mono = load_charge_data(Path(args.data), scheme=args.scheme, level=args.level)
    
    # Train models
    results = train_charge_predictor(
        R=R,
        Z=Z,
        mono=mono,
        test_size=args.test_size,
        random_state=args.seed,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        save_path=Path(args.output)
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nModels saved to: {args.output}")
    print("\nTo use the model:")
    print("  from train_charge_predictor import predict_charges")
    print("  import joblib")
    print("  models = joblib.load('charge_predictor_model.pkl')")
    print("  charges = predict_charges(models, R, Z)")


if __name__ == '__main__':
    main()

