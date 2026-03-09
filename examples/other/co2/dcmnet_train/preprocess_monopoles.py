#!/usr/bin/env python3
"""
Preprocess NPZ files by imputing monopoles and saving updated copies.

This script loads NPZ files, uses a monopole imputation function to predict
charges for all samples, and saves new NPZ files with the imputed monopoles
included. This is more efficient than imputing during training.

Usage:
    python preprocess_monopoles.py \
        --input train.npz valid.npz \
        --output train_with_mono.npz valid_with_mono.npz \
        --imputation_fn charge_predictor_MBIS_raw.pkl \
        --batch_size 500
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import List, Optional, Union
import sys

# Add parent directory to path to import from examples
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_charge_predictor_usage import create_mono_imputation_fn_from_gb
from mmml.dcmnet.dcmnet.training import create_mono_imputation_fn


def load_npz_file(filepath: Path) -> dict:
    """Load an NPZ file and return as dictionary."""
    print(f"Loading {filepath}...")
    data = np.load(filepath, allow_pickle=True)
    return {k: np.array(v) for k, v in data.items()}


def save_npz_file(data: dict, filepath: Path, compressed: bool = True):
    """Save dictionary to NPZ file."""
    print(f"Saving {filepath}...")
    if compressed:
        np.savez_compressed(filepath, **data)
    else:
        np.savez(filepath, **data)
    print(f"✓ Saved {filepath} ({filepath.stat().st_size / 1024 / 1024:.2f} MB)")


def impute_monopoles_for_npz(
    data: dict,
    mono_imputation_fn,
    num_atoms: int = 60,
    batch_size: int = 500,
    verbose: bool = True,
) -> np.ndarray:
    """
    Impute monopoles for all samples in an NPZ data dictionary.
    
    Parameters
    ----------
    data : dict
        Data dictionary from NPZ file
    mono_imputation_fn : callable
        Function to impute monopoles. Takes batch dict, returns monopoles.
    num_atoms : int
        Number of atoms per system (padded size)
    batch_size : int
        Batch size for imputation
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    np.ndarray
        Imputed monopoles with shape (n_samples, num_atoms)
    """
    n_samples = len(data["R"])
    
    # Infer actual number of atoms from data
    actual_num_atoms = num_atoms
    if "Z" in data and len(data["Z"]) > 0:
        first_z = data["Z"][0]
        if isinstance(first_z, (list, tuple, np.ndarray)):
            # Find first non-zero element to determine actual atoms
            non_zero_mask = np.array(first_z) != 0
            if np.any(non_zero_mask):
                actual_num_atoms = int(np.sum(non_zero_mask))
    
    if verbose:
        print(f"  Imputing monopoles for {n_samples} samples...")
        print(f"  Using num_atoms={num_atoms} (padded), actual_atoms={actual_num_atoms} per molecule")
    
    imputed_monopoles = []
    
    # Process in batches
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_indices = list(range(i, end_idx))
        actual_batch_size = len(batch_indices)
        
        if verbose and i % (batch_size * 10) == 0:
            print(f"  Processing batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}...")
        
        # Create batch dict
        batch_dict = {}
        
        # Extract batch data
        batch_R = data["R"][batch_indices]  # (batch_size, num_atoms, 3)
        batch_Z = data["Z"][batch_indices]  # (batch_size, num_atoms)
        
        # Flatten for batch format
        batch_dict["R"] = jnp.array(batch_R.reshape(-1, 3))
        batch_dict["Z"] = jnp.array(batch_Z.reshape(-1))
        
        # Create message passing indices
        import e3x
        batch_segments = jnp.repeat(jnp.arange(actual_batch_size), num_atoms)
        offsets = jnp.arange(actual_batch_size) * num_atoms
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
        dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
        src_idx = (src_idx + offsets[:, None]).reshape(-1)
        
        batch_dict["dst_idx"] = dst_idx
        batch_dict["src_idx"] = src_idx
        batch_dict["batch_segments"] = batch_segments
        
        # Impute monopoles
        try:
            imputed_batch = mono_imputation_fn(batch_dict)
            
            # Infer atoms per molecule from output
            atoms_per_molecule = imputed_batch.size // actual_batch_size
            
            if atoms_per_molecule < actual_num_atoms:
                raise ValueError(
                    f"Imputation function returned too few atoms: got {atoms_per_molecule} atoms per molecule, "
                    f"but need at least {actual_num_atoms} (actual atoms)"
                )
            
            # Reshape and extract actual atoms
            imputed_reshaped = imputed_batch.reshape(actual_batch_size, atoms_per_molecule)
            imputed_actual = imputed_reshaped[:, :actual_num_atoms]
            
            # Pad to num_atoms
            if actual_num_atoms < num_atoms:
                padding = jnp.zeros((actual_batch_size, num_atoms - actual_num_atoms), dtype=imputed_actual.dtype)
                imputed_padded = jnp.concatenate([imputed_actual, padding], axis=1)
            else:
                imputed_padded = imputed_actual[:, :num_atoms]
            
            imputed_monopoles.append(np.array(imputed_padded))
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to impute monopoles for batch {i//batch_size + 1} "
                f"(samples {i} to {end_idx-1}): {e}"
            ) from e
    
    # Concatenate all batches
    all_imputed = np.concatenate(imputed_monopoles, axis=0)
    
    if verbose:
        imputed_mean = float(np.mean(np.abs(all_imputed)))
        imputed_std = float(np.std(all_imputed))
        print(f"  ✓ Monopole imputation complete. Mean_abs={imputed_mean:.6f} e, Std={imputed_std:.6f} e")
    
    return all_imputed


def preprocess_npz_files(
    input_files: List[Path],
    output_files: List[Path],
    mono_imputation_fn,
    num_atoms: int = 60,
    batch_size: int = 500,
    verbose: bool = True,
):
    """
    Preprocess multiple NPZ files by imputing monopoles.
    
    Parameters
    ----------
    input_files : List[Path]
        List of input NPZ file paths
    output_files : List[Path]
        List of output NPZ file paths (must match input_files length)
    mono_imputation_fn : callable
        Function to impute monopoles
    num_atoms : int
        Number of atoms per system (padded size)
    batch_size : int
        Batch size for imputation
    verbose : bool
        Whether to print progress
    """
    if len(input_files) != len(output_files):
        raise ValueError(f"Number of input files ({len(input_files)}) must match output files ({len(output_files)})")
    
    for input_file, output_file in zip(input_files, output_files):
        print(f"\n{'='*70}")
        print(f"Processing: {input_file.name} -> {output_file.name}")
        print(f"{'='*70}")
        
        # Load data
        data = load_npz_file(input_file)
        
        # Check if monopoles already exist and are non-zero
        if "mono" in data:
            mono_sum = np.abs(data["mono"]).sum()
            if mono_sum > 1e-6:
                print(f"  ⚠ Monopoles already present (sum_abs={mono_sum:.2e}). Overwriting with imputed values.")
        
        # Impute monopoles
        imputed_mono = impute_monopoles_for_npz(
            data,
            mono_imputation_fn,
            num_atoms=num_atoms,
            batch_size=batch_size,
            verbose=verbose,
        )
        
        # Update data dictionary
        data["mono"] = imputed_mono
        
        # Save updated NPZ file
        save_npz_file(data, output_file)
        
        print(f"✓ Completed: {output_file.name}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess NPZ files by imputing monopoles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using gradient boosting model
    python preprocess_monopoles.py \\
        --input train.npz valid.npz \\
        --output train_with_mono.npz valid_with_mono.npz \\
        --gb_model charge_predictor_MBIS_raw.pkl \\
        --batch_size 500
    
    # Using DCMNet model
    python preprocess_monopoles.py \\
        --input train.npz valid.npz \\
        --output train_with_mono.npz valid_with_mono.npz \\
        --dcmnet_model model.pkl \\
        --dcmnet_params params.pkl \\
        --batch_size 500
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Input NPZ file(s) to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        nargs="+",
        required=True,
        help="Output NPZ file(s) (must match input count)"
    )
    parser.add_argument(
        "--gb_model",
        type=str,
        default=None,
        help="Path to gradient boosting model (.pkl file)"
    )
    parser.add_argument(
        "--dcmnet_model",
        type=str,
        default=None,
        help="Path to DCMNet model (for create_mono_imputation_fn)"
    )
    parser.add_argument(
        "--dcmnet_params",
        type=str,
        default=None,
        help="Path to DCMNet parameters (required if --dcmnet_model is used)"
    )
    parser.add_argument(
        "--num_atoms",
        type=int,
        default=60,
        help="Number of atoms per system (padded size), default: 60"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Batch size for imputation, default: 500"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress information"
    )
    
    args = parser.parse_args()
    
    # Create imputation function
    print("="*70)
    print("Creating monopole imputation function...")
    print("="*70)
    
    if args.gb_model:
        print(f"Loading gradient boosting model: {args.gb_model}")
        mono_imputation_fn = create_mono_imputation_fn_from_gb(Path(args.gb_model))
        print("✓ Gradient boosting imputation function created")
    elif args.dcmnet_model:
        if args.dcmnet_params is None:
            raise ValueError("--dcmnet_params is required when using --dcmnet_model")
        print(f"Loading DCMNet model: {args.dcmnet_model}")
        print(f"Loading DCMNet parameters: {args.dcmnet_params}")
        # Note: This would require loading the model class
        # For now, we'll assume gradient boosting is the primary use case
        raise NotImplementedError("DCMNet model loading not yet implemented in this script")
    else:
        raise ValueError("Must provide either --gb_model or --dcmnet_model")
    
    # Process files
    input_files = [Path(f) for f in args.input]
    output_files = [Path(f) for f in args.output]
    
    # Create output directories if needed
    for output_file in output_files:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    preprocess_npz_files(
        input_files=input_files,
        output_files=output_files,
        mono_imputation_fn=mono_imputation_fn,
        num_atoms=args.num_atoms,
        batch_size=args.batch_size,
        verbose=args.verbose or True,
    )
    
    print("\n" + "="*70)
    print("Preprocessing Complete!")
    print("="*70)
    print("\nUpdated NPZ files with imputed monopoles:")
    for output_file in output_files:
        print(f"  - {output_file}")


if __name__ == "__main__":
    main()

