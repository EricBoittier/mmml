#!/usr/bin/env python3
"""
Convert NPZ molecular datasets to packed memmap format.

This script converts standard NPZ format (with padded arrays) to the packed
memory-mapped format used by PackedMemmapLoader.

Usage:
    python convert_npz_to_packed_memmap.py \
        --input data.npz \
        --output packed_memmap_dir \
        --verbose

Input NPZ should contain:
    - Z: (N, Amax) atomic numbers
    - R: (N, Amax, 3) positions in Angstrom
    - F: (N, Amax, 3) forces in kcal/mol/Å (optional)
    - E: (N,) energies in kcal/mol
    - N: (N,) number of atoms per molecule (optional, will be inferred)
    - Qtot: (N,) total charges (optional)
"""

import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


def validate_npz_data(data: dict) -> None:
    """Validate NPZ data has required fields."""
    required = ["Z", "R", "E"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    
    n_mols = len(data["Z"])
    for key in ["R", "E"]:
        if key in data and len(data[key]) != n_mols:
            raise ValueError(f"Field {key} has inconsistent length")


def infer_n_atoms(Z: np.ndarray) -> np.ndarray:
    """Infer number of atoms per molecule from atomic numbers."""
    # Count non-zero elements (assuming 0 means padding)
    return np.sum(Z > 0, axis=1).astype(np.int32)


def pack_arrays(
    Z: np.ndarray,
    R: np.ndarray,
    F: np.ndarray,
    N: np.ndarray,
) -> tuple:
    """
    Pack padded arrays into compact format.
    
    Parameters
    ----------
    Z : np.ndarray
        (N_mols, Amax) atomic numbers
    R : np.ndarray
        (N_mols, Amax, 3) positions
    F : np.ndarray
        (N_mols, Amax, 3) forces
    N : np.ndarray
        (N_mols,) number of atoms per molecule
        
    Returns
    -------
    tuple
        (Z_pack, R_pack, F_pack, offsets, n_atoms)
    """
    n_mols = len(Z)
    total_atoms = int(N.sum())
    
    # Preallocate packed arrays
    Z_pack = np.zeros(total_atoms, dtype=np.int32)
    R_pack = np.zeros((total_atoms, 3), dtype=np.float32)
    F_pack = np.zeros((total_atoms, 3), dtype=np.float32)
    offsets = np.zeros(n_mols + 1, dtype=np.int64)
    
    # Pack molecules
    current_offset = 0
    for i in range(n_mols):
        n = int(N[i])
        Z_pack[current_offset:current_offset + n] = Z[i, :n]
        R_pack[current_offset:current_offset + n] = R[i, :n]
        F_pack[current_offset:current_offset + n] = F[i, :n]
        current_offset += n
        offsets[i + 1] = current_offset
    
    return Z_pack, R_pack, F_pack, offsets, N


def convert_npz_to_packed_memmap(
    input_path: str,
    output_path: str,
    verbose: bool = True,
) -> None:
    """
    Convert NPZ file to packed memmap format.
    
    Parameters
    ----------
    input_path : str
        Path to input NPZ file
    output_path : str
        Path to output directory (will be created)
    verbose : bool, optional
        Print progress information, by default True
    """
    if verbose:
        print("=" * 80)
        print("Converting NPZ to Packed Memmap Format")
        print("=" * 80)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print()
    
    # Load NPZ data
    if verbose:
        print("Loading NPZ file...")
    data = dict(np.load(input_path))
    validate_npz_data(data)
    
    n_mols = len(data["Z"])
    if verbose:
        print(f"  Found {n_mols:,} molecules")
    
    # Get or infer number of atoms
    if "N" in data:
        N = data["N"].astype(np.int32)
    else:
        if verbose:
            print("  Inferring atom counts from Z...")
        N = infer_n_atoms(data["Z"])
    
    total_atoms = int(N.sum())
    if verbose:
        print(f"  Total atoms: {total_atoms:,}")
        print(f"  Avg atoms/molecule: {total_atoms / n_mols:.1f}")
        print(f"  Max atoms/molecule: {N.max()}")
        print()
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pack arrays
    if verbose:
        print("Packing arrays...")
    
    Z = data["Z"].astype(np.int32)
    R = data["R"].astype(np.float32)
    
    # Handle optional forces
    if "F" in data:
        F = data["F"].astype(np.float32)
    else:
        if verbose:
            print("  Warning: No forces (F) found, filling with zeros")
        F = np.zeros_like(R)
    
    Z_pack, R_pack, F_pack, offsets, n_atoms = pack_arrays(Z, R, F, N)
    
    # Handle energy
    E = data["E"].astype(np.float64)
    
    # Handle optional total charge
    if "Qtot" in data:
        Qtot = data["Qtot"].astype(np.float64)
    else:
        if verbose:
            print("  Warning: No total charges (Qtot) found, filling with zeros")
        Qtot = np.zeros(n_mols, dtype=np.float64)
    
    # Compute packing efficiency
    padded_atoms = n_mols * Z.shape[1]
    efficiency = total_atoms / padded_atoms * 100
    savings = (1 - total_atoms / padded_atoms) * 100
    
    if verbose:
        print(f"  Packing efficiency: {efficiency:.1f}%")
        print(f"  Space savings: {savings:.1f}%")
        print()
    
    # Write packed arrays
    if verbose:
        print("Writing packed memmap files...")
        files_to_write = [
            ("offsets.npy", offsets, "metadata"),
            ("n_atoms.npy", n_atoms, "metadata"),
            ("Z_pack.int32", Z_pack, "binary"),
            ("R_pack.f32", R_pack, "binary"),
            ("F_pack.f32", F_pack, "binary"),
            ("E.f64", E, "binary"),
            ("Qtot.f64", Qtot, "binary"),
        ]
    else:
        files_to_write = [
            ("offsets.npy", offsets, "metadata"),
            ("n_atoms.npy", n_atoms, "metadata"),
            ("Z_pack.int32", Z_pack, "binary"),
            ("R_pack.f32", R_pack, "binary"),
            ("F_pack.f32", F_pack, "binary"),
            ("E.f64", E, "binary"),
            ("Qtot.f64", Qtot, "binary"),
        ]
    
    for filename, array, fmt in tqdm(files_to_write, disable=not verbose):
        filepath = output_dir / filename
        if fmt == "metadata":
            np.save(filepath, array)
        else:
            array.tofile(filepath)
    
    # Compute file sizes
    total_size = sum(
        os.path.getsize(output_dir / fname)
        for fname, _, _ in files_to_write
    )
    
    if verbose:
        print()
        print("Conversion complete!")
        print(f"  Output size: {total_size / 1024**3:.2f} GB")
        print(f"  Files written to: {output_dir}")
        print()
        print("Files created:")
        for filename, _, _ in files_to_write:
            size_mb = os.path.getsize(output_dir / filename) / 1024**2
            print(f"  - {filename:20s} ({size_mb:>10.2f} MB)")
        print()
        print("Verification:")
        print(f"  Number of molecules: {n_mols:,}")
        print(f"  Total atoms: {total_atoms:,}")
        print(f"  Offsets range: 0 to {offsets[-1]:,}")
        print()
        print("Next steps:")
        print(f"  1. Verify data: python -c \"from mmml.data.packed_memmap_loader import PackedMemmapLoader; loader = PackedMemmapLoader('{output_path}', batch_size=1); print(loader)\"")
        print(f"  2. Train model: python train_physnet_memmap.py --data_path {output_path}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NPZ molecular dataset to packed memmap format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convert_npz_to_packed_memmap.py --input data.npz --output packed_data
  
  # Convert multiple files
  for file in train.npz valid.npz test.npz; do
    python convert_npz_to_packed_memmap.py --input $file --output ${file%.npz}_packed
  done
  
  # Quiet mode
  python convert_npz_to_packed_memmap.py --input data.npz --output packed_data --no-verbose
        """,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input NPZ file path",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for packed memmap files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print progress information (default: True)",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_false",
        dest="verbose",
        help="Suppress progress information",
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if not args.input.endswith(".npz"):
        print(f"Warning: Input file doesn't have .npz extension: {args.input}")
    
    # Convert
    try:
        convert_npz_to_packed_memmap(
            args.input,
            args.output,
            verbose=args.verbose,
        )
        return 0
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

