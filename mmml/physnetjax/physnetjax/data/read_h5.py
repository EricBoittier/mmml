"""
HDF5 data reader for PhysNetJAX.

Reads molecular structure data from HDF5 archives (e.g. qcell_dimers.h5)
and produces dictionaries compatible with the PhysNetJAX training pipeline.

Each HDF5 file is organized as:
    mol_000001, mol_000002, ... (structure groups)
    metadata                    (shared metadata)

Each structure group contains datasets such as:
    atomic_numbers, positions, total_forces, formation_energy, dipole, charge, etc.
"""

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import jax
import numpy as np
import orbax.checkpoint

from mmml.physnetjax.physnetjax.data.data import get_choices, make_dicts

# Orbax checkpointer for dataset caching
_dataset_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


def _cache_key(
    filepath: Path,
    natoms: int,
    energy_key: str,
    force_key: str,
    dipole_key: str,
    max_structures: Optional[int],
) -> str:
    """Build a deterministic hash string for cache directory naming."""
    parts = f"{filepath.resolve()}|{natoms}|{energy_key}|{force_key}|{dipole_key}|{max_structures}"
    return hashlib.sha256(parts.encode()).hexdigest()[:16]


def _get_cache_dir(filepath: Path, cache_dir: Optional[Path], **key_kwargs) -> Path:
    """Return the orbax cache directory for a given set of load parameters."""
    if cache_dir is None:
        cache_dir = filepath.parent / ".h5_cache"
    h = _cache_key(filepath, **key_kwargs)
    return cache_dir / f"{filepath.stem}_{h}"


def load_h5(
    filepath: str | Path,
    natoms: int,
    energy_key: str = "formation_energy",
    force_key: str = "total_forces",
    dipole_key: str = "dipole",
    max_structures: Optional[int] = None,
    cache: bool = True,
    cache_dir: Optional[str | Path] = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Load molecular data from an HDF5 file into PhysNetJAX-compatible arrays.

    On the first call the HDF5 file is read structure-by-structure and the
    processed arrays are saved to an orbax checkpoint.  Subsequent calls with
    the same parameters load directly from the cache, which is much faster.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file.
    natoms : int
        Maximum number of atoms per structure. Arrays are zero-padded to this size.
    energy_key : str, optional
        HDF5 dataset name to use for energies, by default 'formation_energy'.
        Other options: 'total_energy', 'kinetic_energy', etc.
    force_key : str, optional
        HDF5 dataset name to use for forces, by default 'total_forces'.
        Other options: 'hellmann_feynman_forces', 'ionic_forces', etc.
    dipole_key : str, optional
        HDF5 dataset name to use for dipoles, by default 'dipole'.
    max_structures : int or None, optional
        Maximum number of structures to load. None loads all structures.
    cache : bool, optional
        Whether to cache the processed arrays via orbax, by default True.
    cache_dir : str or Path or None, optional
        Directory to store the orbax cache.  Defaults to ``<h5_parent>/.h5_cache/``.
    verbose : bool, optional
        Print progress information during loading.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'R': positions, shape (n_samples, natoms, 3)
        - 'Z': atomic numbers, shape (n_samples, natoms)
        - 'F': forces, shape (n_samples, natoms, 3)
        - 'E': energies, shape (n_samples, 1)
        - 'N': number of atoms per sample, shape (n_samples, 1)
        - 'D': dipole vectors, shape (n_samples, 3) (if available)
        - 'Q': total charge, shape (n_samples, 1) (if available)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    # ------------------------------------------------------------------
    # Try loading from orbax cache
    # ------------------------------------------------------------------
    if cache:
        cache_path = _get_cache_dir(
            filepath,
            Path(cache_dir) if cache_dir is not None else None,
            natoms=natoms,
            energy_key=energy_key,
            force_key=force_key,
            dipole_key=dipole_key,
            max_structures=max_structures,
        )
        if cache_path.exists():
            if verbose:
                print(f"Loading from orbax cache: {cache_path}")
            data = _dataset_checkpointer.restore(cache_path)
            # orbax may restore as jax arrays; convert back to numpy
            data = {k: np.asarray(v) for k, v in data.items()}
            if verbose:
                _print_data_summary(data, filepath.name)
            return data

    # ------------------------------------------------------------------
    # Read from HDF5
    # ------------------------------------------------------------------
    all_R = []
    all_Z = []
    all_F = []
    all_E = []
    all_N = []
    all_D = []
    all_Q = []
    has_dipoles = None
    has_charge = None

    with h5py.File(filepath, "r") as f:
        # Collect molecule group names (sorted for deterministic ordering)
        mol_keys = sorted([k for k in f.keys() if k.startswith("mol_")])

        if max_structures is not None:
            mol_keys = mol_keys[:max_structures]

        if verbose:
            print(f"Loading {len(mol_keys)} structures from {filepath.name}")
            # Print available datasets from first molecule
            if mol_keys:
                print(f"Available datasets in '{mol_keys[0]}':")
                for ds_name in sorted(f[mol_keys[0]].keys()):
                    ds = f[mol_keys[0]][ds_name]
                    print(f"  {ds_name}: shape={ds.shape}, dtype={ds.dtype}")

        n_skipped = 0
        for i, mol_name in enumerate(mol_keys):
            grp = f[mol_name]

            # Read atomic numbers and determine real atom count
            atomic_numbers = grp["atomic_numbers"][()]
            n_atoms = len(atomic_numbers)

            if n_atoms > natoms:
                n_skipped += 1
                if verbose and n_skipped <= 5:
                    print(
                        f"  Skipping {mol_name}: {n_atoms} atoms > natoms={natoms}"
                    )
                continue

            # Positions: (n_atoms, 3) -> zero-pad to (natoms, 3)
            positions = grp["positions"][()]
            R_padded = np.zeros((natoms, 3), dtype=np.float64)
            R_padded[:n_atoms] = positions

            # Atomic numbers: (n_atoms,) -> zero-pad to (natoms,)
            Z_padded = np.zeros(natoms, dtype=np.int32)
            Z_padded[:n_atoms] = atomic_numbers

            # Forces: (n_atoms, 3) -> zero-pad to (natoms, 3)
            if force_key in grp:
                forces = grp[force_key][()]
                F_padded = np.zeros((natoms, 3), dtype=np.float64)
                F_padded[:n_atoms] = forces
            else:
                F_padded = np.zeros((natoms, 3), dtype=np.float64)

            # Energy: scalar -> (1,)
            if energy_key in grp:
                energy = float(grp[energy_key][()])
            else:
                raise KeyError(
                    f"Energy key '{energy_key}' not found in {mol_name}. "
                    f"Available keys: {list(grp.keys())}"
                )

            # Dipole: (3,) -- optional
            if has_dipoles is None:
                has_dipoles = dipole_key in grp
            if has_dipoles and dipole_key in grp:
                dipole = grp[dipole_key][()]
                all_D.append(dipole)

            # Total charge: scalar -- optional
            if has_charge is None:
                has_charge = "charge" in grp
            if has_charge and "charge" in grp:
                charge = float(grp["charge"][()])
                all_Q.append(charge)

            all_R.append(R_padded)
            all_Z.append(Z_padded)
            all_F.append(F_padded)
            all_E.append(energy)
            all_N.append(n_atoms)

            if verbose and (i + 1) % 10000 == 0:
                print(f"  Loaded {i + 1}/{len(mol_keys)} structures...")

        if verbose and n_skipped > 0:
            print(f"  Skipped {n_skipped} structures with > {natoms} atoms")

    n_samples = len(all_R)
    if n_samples == 0:
        raise ValueError(
            f"No structures loaded from {filepath}. "
            f"Check that natoms={natoms} is large enough."
        )

    data = {
        "R": np.array(all_R, dtype=np.float64),                 # (n_samples, natoms, 3)
        "Z": np.array(all_Z, dtype=np.int32),                    # (n_samples, natoms)
        "F": np.array(all_F, dtype=np.float64),                  # (n_samples, natoms, 3)
        "E": np.array(all_E, dtype=np.float64).reshape(-1, 1),   # (n_samples, 1)
        "N": np.array(all_N, dtype=np.int32).reshape(-1, 1),     # (n_samples, 1)
    }

    if has_dipoles and all_D:
        data["D"] = np.array(all_D, dtype=np.float64)  # (n_samples, 3)

    if has_charge and all_Q:
        data["Q"] = np.array(all_Q, dtype=np.float64).reshape(-1, 1)  # (n_samples, 1)

    # ------------------------------------------------------------------
    # Save to orbax cache
    # ------------------------------------------------------------------
    if cache:
        cache_path = _get_cache_dir(
            filepath,
            Path(cache_dir) if cache_dir is not None else None,
            natoms=natoms,
            energy_key=energy_key,
            force_key=force_key,
            dipole_key=dipole_key,
            max_structures=max_structures,
        )
        if verbose:
            print(f"Saving orbax cache to: {cache_path}")
        from flax.training import orbax_utils
        save_args = orbax_utils.save_args_from_target(data)
        _dataset_checkpointer.save(cache_path, data, save_args=save_args)
        if verbose:
            print("  Cache saved.")

    if verbose:
        _print_data_summary(data, filepath.name)

    return data


def _print_data_summary(data: Dict[str, np.ndarray], name: str) -> None:
    """Print a summary of loaded data shapes and statistics."""
    n_samples = len(data["R"])
    print(f"\nLoaded {n_samples} structures from {name}")
    print("Array shapes:")
    for k, v in data.items():
        print(f"  {k}: {v.shape}  dtype={v.dtype}")
    print(f"Max atoms in any structure: {int(np.max(data['N']))}")
    print(f"Energy range: [{data['E'].min():.4f}, {data['E'].max():.4f}] eV")
    print(f"Energy std: {data['E'].std():.4f} eV")
    if "Q" in data:
        unique_q = np.unique(data["Q"])
        print(f"Charges present: {unique_q.tolist()}")


def prepare_h5_datasets(
    key,
    filepath: str | Path,
    train_size: int,
    valid_size: int,
    natoms: Optional[int] = None,
    energy_key: str = "formation_energy",
    force_key: str = "total_forces",
    dipole_key: str = "dipole",
    max_structures: Optional[int] = None,
    cache: bool = True,
    cache_dir: Optional[str | Path] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    """
    Load an HDF5 file and split into train/validation dictionaries.

    This is the main entry point for using HDF5 data with PhysNetJAX training.
    It produces train_data and valid_data dicts that can be passed directly
    to ``train_model()``.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for shuffling and splitting.
    filepath : str or Path
        Path to the HDF5 file.
    train_size : int
        Number of training samples.
    valid_size : int
        Number of validation samples.
    natoms : int or None, optional
        Maximum number of atoms per structure (padding size).
        If None, automatically determined from the largest molecule in the file.
    energy_key : str, optional
        HDF5 dataset name for energies, by default 'formation_energy'.
    force_key : str, optional
        HDF5 dataset name for forces, by default 'total_forces'.
    dipole_key : str, optional
        HDF5 dataset name for dipoles, by default 'dipole'.
    max_structures : int or None, optional
        Maximum number of structures to load from the file.
    cache : bool, optional
        Whether to cache the processed arrays via orbax, by default True.
    cache_dir : str or Path or None, optional
        Directory to store the orbax cache.  Defaults to ``<h5_parent>/.h5_cache/``.
    verbose : bool, optional
        Print progress and shape information.

    Returns
    -------
    train_data : dict
        Training data dictionary with keys R, Z, F, E, N, (D), (Q).
    valid_data : dict
        Validation data dictionary with keys R, Z, F, E, N, (D), (Q).
    natoms : int
        The padding size used (= max atoms in the dataset when auto-detected).
        Use this value for ``EF(natoms=...)`` and ``train_model(num_atoms=...)``.

    Examples
    --------
    >>> import jax
    >>> from mmml.physnetjax.physnetjax.models.model import EF
    >>> from mmml.physnetjax.physnetjax.training.training import train_model
    >>> key = jax.random.PRNGKey(42)
    >>> train_data, valid_data, natoms = prepare_h5_datasets(
    ...     key,
    ...     filepath="/path/to/qcell_dimers.h5",
    ...     train_size=1000,
    ...     valid_size=200,
    ... )
    >>> model = EF(natoms=natoms, ...)
    >>> train_model(key, model, train_data, valid_data, num_atoms=natoms, ...)
    """
    # Auto-detect natoms if not provided
    if natoms is None:
        natoms = _detect_natoms(filepath, max_structures=max_structures, verbose=verbose)

    # Load all data from HDF5 (uses orbax cache when available)
    data_dict = load_h5(
        filepath=filepath,
        natoms=natoms,
        energy_key=energy_key,
        force_key=force_key,
        dipole_key=dipole_key,
        max_structures=max_structures,
        cache=cache,
        cache_dir=cache_dir,
        verbose=verbose,
    )

    n_samples = len(data_dict["R"])

    # Validate sizes
    total_requested = train_size + valid_size
    if total_requested > n_samples:
        raise ValueError(
            f"Requested {train_size} train + {valid_size} valid = {total_requested} "
            f"samples, but only {n_samples} structures available in the file."
        )

    # Build parallel lists for make_dicts (matches the interface in data.py)
    keys = list(data_dict.keys())
    data = [data_dict[k] for k in keys]

    # Random train/valid split
    train_choice, valid_choice = get_choices(key, n_samples, train_size, valid_size)
    train_data, valid_data = make_dicts(data, keys, train_choice, valid_choice)

    if verbose:
        print(f"\nTrain/Valid split:")
        print(f"  Training samples:   {train_size}")
        print(f"  Validation samples: {valid_size}")
        print(f"  natoms (padding):   {natoms}")

    return train_data, valid_data, natoms


def _detect_natoms(
    filepath: str | Path,
    max_structures: Optional[int] = None,
    verbose: bool = False,
) -> int:
    """
    Scan an HDF5 file to find the maximum number of atoms across all structures.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file.
    max_structures : int or None
        If set, only scan the first N structures.
    verbose : bool
        Print detection info.

    Returns
    -------
    int
        Maximum atom count found in the file.
    """
    filepath = Path(filepath)
    max_n = 0
    with h5py.File(filepath, "r") as f:
        mol_keys = sorted([k for k in f.keys() if k.startswith("mol_")])
        if max_structures is not None:
            mol_keys = mol_keys[:max_structures]
        for mol_name in mol_keys:
            n = len(f[mol_name]["atomic_numbers"][()])
            if n > max_n:
                max_n = n
    if max_n == 0:
        raise ValueError(f"No molecule groups found in {filepath}")
    if verbose:
        print(f"Auto-detected natoms={max_n} from {filepath.name}")
    return max_n


if __name__ == "__main__":
    """
    Verification script: load an HDF5 file and print shapes/statistics.

    Usage:
        python -m mmml.physnetjax.physnetjax.data.read_h5 /path/to/qcell_dimers.h5 [natoms]

    If natoms is omitted it is auto-detected from the file.
    """
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python read_h5.py <path_to_h5_file> [natoms]")
        print("  natoms: max atoms per structure (auto-detected if omitted)")
        sys.exit(1)

    h5_path = sys.argv[1]
    natoms = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"Loading: {h5_path}")
    if natoms is not None:
        print(f"natoms:  {natoms} (user-specified)")
    else:
        natoms = _detect_natoms(h5_path, verbose=True)
        print(f"natoms:  {natoms} (auto-detected)")
    print("=" * 60)

    t0 = time.perf_counter()
    data = load_h5(h5_path, natoms=natoms, verbose=True)
    t1 = time.perf_counter()
    print(f"\nFirst load took {t1 - t0:.2f}s")

    # Second load should hit the orbax cache
    print("\n" + "=" * 60)
    print("Loading again (should use orbax cache)...")
    t0 = time.perf_counter()
    data = load_h5(h5_path, natoms=natoms, verbose=True)
    t1 = time.perf_counter()
    print(f"Cached load took {t1 - t0:.2f}s")

    print("\n" + "=" * 60)
    print("Verification complete.")
    print(f"Total structures: {len(data['R'])}")
    print(f"Keys: {list(data.keys())}")

    # Quick sanity checks
    print("\nSanity checks:")
    n_real = data["N"].flatten()
    print(f"  Atoms per structure: min={n_real.min()}, max={n_real.max()}, "
          f"mean={n_real.mean():.1f}")
    print(f"  Energy: mean={data['E'].mean():.4f}, std={data['E'].std():.4f} eV")

    if "Q" in data:
        unique_q = np.unique(data["Q"])
        print(f"  Charges present: {unique_q.tolist()}")

    # Check that padding is correct (Z should be 0 beyond N atoms)
    sample_idx = 0
    n = int(data["N"][sample_idx])
    z = data["Z"][sample_idx]
    assert np.all(z[n:] == 0), "Padding check failed: non-zero Z beyond N atoms"
    assert np.all(z[:n] > 0), "Padding check failed: zero Z within first N atoms"
    print(f"  Padding check passed (sample 0: {n} atoms, Z={z[:n]})")

    # Demo train/valid split with auto-detected natoms
    print("\n" + "=" * 60)
    print("Demo train/valid split (natoms auto-detected):")
    key = jax.random.PRNGKey(42)
    n_total = len(data["R"])
    n_train = min(100, n_total // 2)
    n_valid = min(50, n_total // 4)

    train_data, valid_data, detected_natoms = prepare_h5_datasets(
        key, h5_path, train_size=n_train, valid_size=n_valid,
        verbose=True,
    )
    print(f"  natoms returned: {detected_natoms}")
    print(f"  train_data keys: {list(train_data.keys())}")
    print(f"  valid_data keys: {list(valid_data.keys())}")
    for k in train_data:
        print(f"  train {k}: {train_data[k].shape}")
    for k in valid_data:
        print(f"  valid {k}: {valid_data[k].shape}")

    print(f"\nUse these values for model and training:")
    print(f"  model = EF(natoms={detected_natoms}, ...)")
    print(f"  train_model(..., num_atoms={detected_natoms}, ...)")
