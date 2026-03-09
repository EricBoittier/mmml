#!/usr/bin/env python3
"""
Convert NPZ time series files to other formats (CSV, HDF5, etc.).

Usage:
    # Convert NPZ to CSV (one file per replica)
    python convert_timeseries_npz.py energy_timeseries.npz --format csv --output-dir csv_output/
    
    # Convert NPZ to CSV (single file with replica columns)
    python convert_timeseries_npz.py energy_timeseries.npz --format csv --output single_file.csv
    
    # Convert NPZ to HDF5
    python convert_timeseries_npz.py energy_timeseries.npz --format hdf5 --output timeseries.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def npz_to_csv_per_replica(npz_path: Path, output_dir: Path) -> None:
    """Convert NPZ to CSV files, one per replica."""
    data = np.load(npz_path)
    
    # Get dimensions
    if "total_energy" in data:
        n_steps, n_replicas = data["total_energy"].shape
    elif "potential_energy" in data:
        n_steps, n_replicas = data["potential_energy"].shape
    else:
        raise ValueError("No energy data found in NPZ file")
    
    # Get time/step
    time = data.get("time_fs", np.arange(n_steps))
    step = data.get("step", np.arange(n_steps))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write one CSV per replica
    for replica in range(n_replicas):
        csv_path = output_dir / f"replica_{replica:02d}.csv"
        
        # Collect columns
        columns = {
            "step": step,
            "time_fs": time,
        }
        
        # Add energy columns
        if "total_energy" in data:
            columns["total_energy_eV"] = data["total_energy"][:, replica]
        if "potential_energy" in data:
            columns["potential_energy_eV"] = data["potential_energy"][:, replica]
        if "kinetic_energy" in data:
            columns["kinetic_energy_eV"] = data["kinetic_energy"][:, replica]
        if "temperature" in data:
            columns["temperature_K"] = data["temperature"][:, replica]
        if "max_force" in data:
            columns["max_force_eV_per_A"] = data["max_force"][:, replica]
        
        # Write header
        header = ",".join(columns.keys())
        
        # Write data
        rows = np.column_stack([columns[k] for k in columns.keys()])
        np.savetxt(csv_path, rows, delimiter=",", header=header, comments="", fmt="%.8e")
        
        print(f"✅ Saved replica {replica} to {csv_path}")
    
    data.close()


def npz_to_csv_single(npz_path: Path, output_path: Path) -> None:
    """Convert NPZ to a single CSV file with replica columns."""
    data = np.load(npz_path)
    
    # Get dimensions
    if "total_energy" in data:
        n_steps, n_replicas = data["total_energy"].shape
    elif "potential_energy" in data:
        n_steps, n_replicas = data["potential_energy"].shape
    else:
        raise ValueError("No energy data found in NPZ file")
    
    # Get time/step
    time = data.get("time_fs", np.arange(n_steps))
    step = data.get("step", np.arange(n_steps))
    
    # Collect all columns
    columns = {
        "step": step,
        "time_fs": time,
    }
    
    # Add energy columns (one column per replica)
    if "total_energy" in data:
        for r in range(n_replicas):
            columns[f"total_energy_rep{r:02d}_eV"] = data["total_energy"][:, r]
    if "potential_energy" in data:
        for r in range(n_replicas):
            columns[f"potential_energy_rep{r:02d}_eV"] = data["potential_energy"][:, r]
    if "kinetic_energy" in data:
        for r in range(n_replicas):
            columns[f"kinetic_energy_rep{r:02d}_eV"] = data["kinetic_energy"][:, r]
    if "temperature" in data:
        for r in range(n_replicas):
            columns[f"temperature_rep{r:02d}_K"] = data["temperature"][:, r]
    if "max_force" in data:
        for r in range(n_replicas):
            columns[f"max_force_rep{r:02d}_eV_per_A"] = data["max_force"][:, r]
    
    # Write header
    header = ",".join(columns.keys())
    
    # Write data
    rows = np.column_stack([columns[k] for k in columns.keys()])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, rows, delimiter=",", header=header, comments="", fmt="%.8e")
    
    print(f"✅ Saved to {output_path}")
    data.close()


def npz_to_hdf5(npz_path: Path, output_path: Path) -> None:
    """Convert NPZ to HDF5 format."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 conversion. Install with: pip install h5py")
    
    data = np.load(npz_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, "w") as f:
        for key in data.keys():
            f.create_dataset(key, data=data[key])
    
    print(f"✅ Saved to {output_path}")
    data.close()


def csv_to_npz(csv_path: Path, output_path: Path, replica: int | None = None) -> None:
    """
    Convert CSV back to NPZ format.
    
    Parameters
    ----------
    csv_path : Path
        Input CSV file
    output_path : Path
        Output NPZ file
    replica : int, optional
        If provided, assumes CSV has single replica data and sets this replica index
    """
    # Load CSV
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    
    # Try to read header
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
    
    # Build dictionary
    npz_data = {}
    for i, col_name in enumerate(header):
        npz_data[col_name] = data[:, i]
    
    # If single replica, reshape arrays
    if replica is not None:
        # Reshape single-replica arrays to (n_steps, 1)
        for key in list(npz_data.keys()):
            if key not in ["step", "time_fs"]:
                npz_data[key] = npz_data[key].reshape(-1, 1)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **npz_data)
    
    print(f"✅ Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert NPZ time series files to other formats")
    parser.add_argument("input", type=Path, help="Input NPZ file")
    parser.add_argument("--format", type=str, choices=["csv", "hdf5", "npz"], default="csv",
                       help="Output format")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output file path (for single CSV or HDF5)")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory (for CSV per replica)")
    parser.add_argument("--replica", type=int, default=None,
                       help="Replica index (for CSV to NPZ conversion)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    if args.format == "csv":
        if args.output:
            # Single CSV file
            npz_to_csv_single(args.input, args.output)
        elif args.output_dir:
            # CSV per replica
            npz_to_csv_per_replica(args.input, args.output_dir)
        else:
            # Default: CSV per replica in same directory
            output_dir = args.input.parent / f"{args.input.stem}_csv"
            npz_to_csv_per_replica(args.input, output_dir)
    
    elif args.format == "hdf5":
        output_path = args.output or args.input.with_suffix(".h5")
        npz_to_hdf5(args.input, output_path)
    
    elif args.format == "npz":
        # CSV to NPZ
        output_path = args.output or args.input.with_suffix(".npz")
        csv_to_npz(args.input, output_path, args.replica)


if __name__ == "__main__":
    main()

