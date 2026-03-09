#!/usr/bin/env python3
"""
Example: Using the trained charge predictor as an imputation function

This shows how to use the gradient boosting charge predictor with DCMNet training.
"""

import numpy as np
import jax.numpy as jnp
import joblib
from pathlib import Path
from typing import Dict

from train_charge_predictor import predict_charges


def create_mono_imputation_fn_from_gb(model_path: Path):
    """
    Create a monopole imputation function from a trained gradient boosting model.
    
    This can be used with DCMNet training to impute missing monopoles.
    
    Parameters
    ----------
    model_path : Path
        Path to saved gradient boosting model (.pkl file)
        
    Returns
    -------
    callable
        Imputation function compatible with prepare_batches
    """
    # Load models
    models = joblib.load(model_path)
    
    # Import here to avoid circular imports
    from train_charge_predictor import compute_molecular_features
    
    def imputation_fn(batch: Dict) -> jnp.ndarray:
        """
        Impute monopoles for a batch.
        
        Parameters
        ----------
        batch : dict
            Batch dictionary containing 'Z', 'R', 'dst_idx', 'src_idx', 'batch_segments'
            
        Returns
        -------
        jnp.ndarray
            Atomic monopoles with shape (batch_size * num_atoms,)
        """
        # Extract batch info
        R_flat = batch["R"]  # (batch_size * num_atoms, 3)
        Z_flat = batch["Z"]  # (batch_size * num_atoms,)
        batch_segments = batch["batch_segments"]  # (batch_size * num_atoms,)
        
        # Infer batch_size and num_atoms
        batch_size = int(jnp.max(batch_segments) + 1)
        num_atoms = len(R_flat) // batch_size
        
        # Reshape to (batch_size, num_atoms, 3) and (batch_size, num_atoms)
        R = R_flat.reshape(batch_size, num_atoms, 3)
        Z = Z_flat.reshape(batch_size, num_atoms)
        
        # Convert to numpy for sklearn
        R_np = np.array(R)
        Z_np = np.array(Z)
        
        # Process each molecule separately (for CO2, we only use first 3 atoms)
        # The model was trained on CO2, so we extract the first 3 atoms
        n_atoms_model = 3  # CO2 has 3 atoms
        predictions = []
        
        for i in range(batch_size):
            # Extract first 3 atoms (assuming CO2 structure)
            R_mol = R_np[i, :n_atoms_model, :]  # (3, 3)
            Z_mol = Z_np[i, :n_atoms_model]    # (3,)
            
            # Compute features for this molecule
            X_mol = compute_molecular_features(R_mol, Z_mol, n_atoms=n_atoms_model)  # (1, n_features)
            
            # Predict for each atom
            mol_charges = []
            for atom_idx in range(n_atoms_model):
                model = models[f'atom_{atom_idx}']
                charge = model.predict(X_mol)[0]  # Single prediction
                mol_charges.append(charge)
            
            # Pad with zeros if num_atoms > 3
            if num_atoms > n_atoms_model:
                mol_charges.extend([0.0] * (num_atoms - n_atoms_model))
            
            predictions.extend(mol_charges)
        
        return jnp.array(predictions)
    
    return imputation_fn


# Example usage
if __name__ == '__main__':
    # Example: Load model and create imputation function
    model_path = Path('charge_predictor_model.pkl')
    
    if model_path.exists():
        print("Creating imputation function from trained model...")
        mono_imputation_fn = create_mono_imputation_fn_from_gb(model_path)
        
        print("✅ Imputation function created!")
        print("\nUsage in training:")
        print("  from train_charge_predictor_usage import create_mono_imputation_fn_from_gb")
        print("  mono_imputation_fn = create_mono_imputation_fn_from_gb('charge_predictor_model.pkl')")
        print("  train_model(..., mono_imputation_fn=mono_imputation_fn)")
    else:
        print(f"❌ Model not found: {model_path}")
        print("Train the model first:")
        print("  python train_charge_predictor.py --data examples/co2/detailed_charges/df_charges_long.csv --scheme Hirshfeld --output charge_predictor_model.pkl")
