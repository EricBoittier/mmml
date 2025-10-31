#!/usr/bin/env python3
"""Split a large NPZ dataset containing cube-based ESP into two files.

Outputs:
  1) energies_forces_dipoles.npz: contains E, F, dipoles (D/Dxyz) and standard R/Z/N
  2) grids_esp.npz: contains subsampled esp and vdw_grid (XYZ) per-sample, plus R/Z/N, Dxyz, and Q (if present)

Grid handling:
  - By default, downsample to NGRID points per sample (default 3000) after
    trimming extreme ESP values by percentile. This keeps output compact and
    avoids building massive per-sample grids.

Notes:
  - This script re-saves arrays; ensure you have free disk space >= input size.
  - For very large `cube_potential`, writing will take time and memory. Consider
    running on a machine with sufficient RAM and fast disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


def compute_grid_xyz(origin: np.ndarray, axes: np.ndarray, dims: np.ndarray, dtype=None) -> np.ndarray:
    """Compute XYZ coordinates for a cube grid.

    Parameters
    ----------
    origin : (3,) array
        Origin of the grid.
    axes : (3, 3) array
        Grid axis vectors. Coordinate is origin + i*axes[0] + j*axes[1] + k*axes[2].
    dims : (3,) array-like of ints
        Number of points along each axis (Nx, Ny, Nz).
    dtype : numpy dtype or None
        Dtype for the returned array. If None, uses float64.

    Returns
    -------
    grid : (Nx*Ny*Nz, 3) array
        Flattened coordinates in numpy C-order corresponding to indexing='ij'.
    """
    origin = np.asarray(origin, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    dims = np.asarray(dims, dtype=np.int64)

    nx, ny, nz = [int(d) for d in dims]
    I, J, K = np.meshgrid(
        np.arange(nx, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        np.arange(nz, dtype=np.float64),
        indexing="ij",
    )
    # Broadcast to (..., 3)
    coords = (
        origin
        + I[..., None] * axes[0]
        + J[..., None] * axes[1]
        + K[..., None] * axes[2]
    )
    grid = coords.reshape(nx * ny * nz, 3)
    if dtype is not None:
        grid = grid.astype(dtype, copy=False)
    return grid


def all_equal(arr: np.ndarray) -> bool:
    """Return True if every row equals the first row (using exact equality)."""
    if arr.ndim == 1:
        return True
    return np.all(arr == arr[0])


def all_close(arr: np.ndarray, rtol=1e-8, atol=1e-12) -> bool:
    """Return True if every row is close to the first row."""
    if arr.ndim == 1:
        return True
    return np.allclose(arr, arr[0], rtol=rtol, atol=atol)


def select_indices_uniform(
    esp: np.ndarray,
    n_select: int,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Select indices uniformly at random after trimming extremes by percentiles.

    If the number of candidates after trimming is < n_select, fallback to the
    full set without trimming.
    """
    if rng is None:
        rng = np.random.default_rng()

    lo = np.percentile(esp, low_pct)
    hi = np.percentile(esp, high_pct)
    mask = (esp >= lo) & (esp <= hi)
    cand = np.nonzero(mask)[0]
    if cand.size >= n_select:
        return rng.choice(cand, size=n_select, replace=False)
    # Fallback to full range
    return rng.choice(np.arange(esp.size), size=n_select, replace=False)


def coords_from_indices(
    origin: np.ndarray,
    axes: np.ndarray,
    dims: np.ndarray,
    lin_idx: np.ndarray,
    dtype,
) -> np.ndarray:
    """Compute coordinates for given linear indices without building full grid."""
    nx, ny, nz = [int(d) for d in np.asarray(dims, dtype=np.int64)]
    lin_idx = np.asarray(lin_idx, dtype=np.int64)

    ij = lin_idx // (ny * nz)
    rem = lin_idx % (ny * nz)
    jj = rem // nz
    kk = rem % nz

    origin = np.asarray(origin, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)

    coords = (
        origin
        + ij[:, None] * axes[0]
        + jj[:, None] * axes[1]
        + kk[:, None] * axes[2]
    )
    return coords.astype(dtype, copy=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Path to input .npz (e.g., co2.1000.npz)")
    p.add_argument(
        "--out-props",
        type=Path,
        default=Path("energies_forces_dipoles.npz"),
        help="Output npz for E/F/dipoles + R/Z/N",
    )
    p.add_argument(
        "--out-grid",
        type=Path,
        default=Path("grids_esp.npz"),
        help="Output npz for esp + vdw_grid + R/Z/N",
    )
    p.add_argument(
        "--ngrid",
        type=int,
        default=3000,
        help="Number of ESP grid points to keep per sample",
    )
    p.add_argument(
        "--grid-dtype",
        choices=["float64", "float32"],
        default="float64",
        help="Dtype for vdw_grid coordinates",
    )
    p.add_argument(
        "--esp-dtype",
        choices=["float64", "float32"],
        default="float32",
        help="Dtype for esp values in output",
    )
    p.add_argument(
        "--allow-per-sample",
        action="store_true",
        help=(
            "If grid metadata differs per sample, allow computing per-sample vdw_grid. "
            "Warning: can be extremely memory/disk heavy."
        ),
    )
    p.add_argument(
        "--trim-low",
        type=float,
        default=1.0,
        help="Lower percentile for ESP trimming (ignore more-negative values)",
    )
    p.add_argument(
        "--trim-high",
        type=float,
        default=99.0,
        help="Upper percentile for ESP trimming (ignore more-positive values)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible subsampling",
    )
    p.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed for outputs (slower, smaller)",
    )
    args = p.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {input_path}")
    # Note: np.load on .npz does not provide true memmap for members; large arrays will be loaded.
    data = np.load(input_path)

    keys = set(data.keys())
    required_std = {"R", "Z", "N"}
    missing_std = [k for k in required_std if k not in keys]
    if missing_std:
        print(f"Warning: missing standard keys: {missing_std}")

    # Identify fields
    E = data.get("E", None)
    F = data.get("F", None)
    D = data.get("D", None)
    Dxyz = data.get("Dxyz", None)
    Q = data.get("Q", None)
    R = data.get("R", None)
    Z = data.get("Z", None)
    N = data.get("N", None)

    cube_potential = data.get("cube_potential", None)
    cp_dims = data.get("cube_potential_dimensions", None)
    cp_origin = data.get("cube_potential_origin", None)
    cp_axes = data.get("cube_potential_axes", None)

    if cube_potential is None:
        print("Error: cube_potential not found in input.", file=sys.stderr)
        sys.exit(2)

    if cp_dims is None or cp_origin is None or cp_axes is None:
        print("Error: cube_potential_* metadata (dimensions/origin/axes) missing.", file=sys.stderr)
        sys.exit(2)

    # Determine shapes
    n_samples = int(cube_potential.shape[0]) if cube_potential.ndim >= 1 else 1
    # cp_dims can be (n_samples, 3) or (3,)
    cp_dims_arr = np.array(cp_dims)
    if cp_dims_arr.ndim == 1:
        cp_dims_arr = np.broadcast_to(cp_dims_arr, (n_samples, 3))

    cp_origin_arr = np.array(cp_origin)
    if cp_origin_arr.ndim == 1:
        cp_origin_arr = np.broadcast_to(cp_origin_arr, (n_samples, 3))

    cp_axes_arr = np.array(cp_axes)
    # Normalize axes shape to (n_samples, 3, 3) if possible
    if cp_axes_arr.ndim == 1 and cp_axes_arr.shape[0] == 9:
        cp_axes_arr = cp_axes_arr.reshape(3, 3)
    if cp_axes_arr.ndim == 2:
        # Could be (n_samples, 9) or already (3,3)
        if cp_axes_arr.shape == (3, 3):
            cp_axes_arr = np.broadcast_to(cp_axes_arr, (n_samples, 3, 3))
        elif cp_axes_arr.shape[1] == 9:
            cp_axes_arr = cp_axes_arr.reshape(cp_axes_arr.shape[0], 3, 3)
        else:
            # Assume it's (n_samples, 3) which is invalid for axes; raise informative error
            raise ValueError(
                f"Unexpected cube_potential_axes shape {cp_axes_arr.shape}; expected (3,3) or (n,9) or (n,3,3)."
            )
    elif cp_axes_arr.ndim == 3:
        # already (n_samples, 3, 3) or (1,3,3)
        if cp_axes_arr.shape[0] != n_samples:
            cp_axes_arr = np.broadcast_to(cp_axes_arr, (n_samples, 3, 3))

    # Decide on shared vs per-sample grid
    has_shared_grid = (
        all_equal(cp_dims_arr)
        and all_close(cp_origin_arr)
        and all_close(cp_axes_arr)
    )

    grid_dtype = np.float64 if args.grid_dtype == "float64" else np.float32

    # Subsample per-sample without materializing full per-sample grids
    rng = np.random.default_rng(args.seed)
    esp_dtype = np.float64 if args.esp_dtype == "float64" else np.float32
    grid_dtype = np.float64 if args.grid_dtype == "float64" else np.float32

    print(
        f"Subsampling ESP grid to {args.ngrid} points per sample "
        f"(trim {args.trim_low}-{args.trim_high} percentiles)."
    )
    # Preallocate outputs
    esp_out = np.empty((n_samples, args.ngrid), dtype=esp_dtype)
    grid_out = np.empty((n_samples, args.ngrid, 3), dtype=grid_dtype)

    # Determine expected full grid length from first dims
    ngrid_expected0 = int(np.prod(cp_dims_arr[0]))
    lastdim = cube_potential.shape[-1]
    if lastdim != ngrid_expected0:
        print(
            f"Warning: cube_potential last dim {lastdim} != product(dims[0]) {ngrid_expected0}. "
            "Proceeding, but check metadata consistency.",
            file=sys.stderr,
        )

    # Iterate samples; avoid keeping large temporaries
    for i in range(n_samples):
        if i % 50 == 0:
            print(f"  sample {i}/{n_samples}")
        esp_i = np.asarray(cube_potential[i])
        idx = select_indices_uniform(
            esp_i, args.ngrid, low_pct=args.trim_low, high_pct=args.trim_high, rng=rng
        )
        esp_out[i] = esp_i[idx].astype(esp_dtype, copy=False)
        grid_out[i] = coords_from_indices(
            cp_origin_arr[i], cp_axes_arr[i], cp_dims_arr[i], idx, dtype=grid_dtype
        )

    # Build outputs
    props_out: Dict[str, np.ndarray] = {}
    grids_out: Dict[str, np.ndarray] = {}

    # Standard keys in both
    for k, v in (("R", R), ("Z", Z), ("N", N)):
        if v is not None:
            props_out[k] = v
            grids_out[k] = v

    # Energies, forces, dipoles
    if E is not None:
        props_out["E"] = E
    if F is not None:
        props_out["F"] = F
    # Prefer Dxyz if present; else D
    if Dxyz is not None:
        props_out["Dxyz"] = Dxyz
    elif D is not None:
        props_out["D"] = D

    # Grids + ESP (subsampled)
    grids_out["esp"] = esp_out
    grids_out["vdw_grid"] = grid_out
    # Include dipoles and charges in the grids file as requested
    if Dxyz is not None:
        grids_out["Dxyz"] = Dxyz
    elif D is not None:
        grids_out["D"] = D
    if Q is not None:
        grids_out["Q"] = Q
    # Also keep grid metadata for provenance (original full-grid info)
    grids_out["grid_dims"] = cp_dims_arr
    grids_out["grid_origin"] = cp_origin_arr
    grids_out["grid_axes"] = cp_axes_arr

    # Save
    save_fn = np.savez_compressed if args.compress else np.savez

    print(f"Writing: {args.out_props}")
    save_fn(args.out_props, **props_out)

    print(f"Writing: {args.out_grid}")
    save_fn(args.out_grid, **grids_out)

    print("Done.")


if __name__ == "__main__":
    main()
