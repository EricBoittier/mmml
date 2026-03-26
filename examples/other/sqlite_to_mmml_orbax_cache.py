#!/usr/bin/env python3
"""
Read a QCML-style SQLite database (APSW) and write an Orbax PyTree cache compatible
with MMML / PhysNetJAX spooky training (same layout as :func:`load_h5` in
``read_h5.py``).

Per-structure SQLite columns (expected order in ``SELECT * FROM data``):
  id, charge, spin/unpaired, Z blob, R blob, energy, F blob, D blob

MMML padded dict keys
---------------------
  R  (n_mol, natoms, 3) float64 — positions [Å]
  Z  (n_mol, natoms) int32 — atomic numbers (0 padding)
  F  (n_mol, natoms, 3) float64 — forces [eV/Å]
  E  (n_mol, 1) float64 — energy [eV]
  N  (n_mol, 1) int32 — real atom count per structure
  Q  (n_mol, 1) float64 — total charge
  S  (n_mol, 1) float64 — spin multiplicity (see --spin-mode)
  D  (n_mol, 3) float64 — dipole [e·Å], optional if all rows have dipoles

Usage
-----
  python examples/other/sqlite_to_mmml_orbax_cache.py /path/to/data.db \\
    --cache-dir /path/to/.sqlite_cache

Restore in Python::

  import orbax.checkpoint as ocp
  import numpy as np
  data = ocp.PyTreeCheckpointer().restore("/path/to/cache_dir")
  data = {k: np.asarray(v) for k, v in data.items()}

Prerequisites: apsw, numpy, jax, flax, orbax
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import apsw
except ImportError as e:
    raise ImportError("This script requires apsw: pip install apsw") from e


def _deblob(buffer: bytes, dtype: np.dtype, shape: Optional[Sequence[int]] = None) -> np.ndarray:
    array = np.frombuffer(buffer, dtype=dtype)
    if not np.little_endian:
        array = array.byteswap()
    if shape is not None:
        array.shape = tuple(shape)
    return np.copy(array)


def _unpack_data_tuple(data: Tuple[Any, ...]) -> Tuple[np.ndarray, ...]:
    """Unpack one ``data`` row; indices follow QCML blob layout."""
    n = len(data[3]) // 4  # int32 = 4 bytes
    q = np.asarray([0.0 if data[1] is None else float(data[1])], dtype=np.float32)
    s = np.asarray([0.0 if data[2] is None else float(data[2])], dtype=np.float32)
    z = _deblob(data[3], dtype=np.int32, shape=(n,))
    r = _deblob(data[4], dtype=np.float32, shape=(n, 3))
    e = np.asarray([0.0 if data[5] is None else float(data[5])], dtype=np.float32)
    f = _deblob(data[6], dtype=np.float32, shape=(n, 3))
    if data[7] is None:
        d = None
    else:
        d = _deblob(data[7], dtype=np.float32, shape=(1, 3))
    return q, s, z, r, e, f, d


def _spin_for_mmml(s_raw: float, mode: str) -> float:
    """
    Map SQLite scalar ``s`` to MMML ``S`` (multiplicity used by spooky / qcell).

    ``unpaired_plus_one``: interpret ``s`` as number of unpaired electrons;
    multiplicity ≈ n_unpaired + 1 (singlet 0→1, doublet 1→2, …).

    ``as_is``: use ``s`` directly (if your DB already stores multiplicity).
    """
    if mode == "as_is":
        return float(s_raw)
    if mode == "unpaired_plus_one":
        return float(s_raw) + 1.0
    raise ValueError(f"Unknown spin mode: {mode}")


def sqlite_rows_to_mmml_arrays(
    database: Path,
    natoms: int,
    max_structures: Optional[int] = None,
    charge_filter: Optional[float] = None,
    spin_mode: str = "unpaired_plus_one",
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Load all structures from SQLite into one padded dict (MMML / load_h5 format).
    """
    conn = apsw.Connection(str(database), flags=apsw.SQLITE_OPEN_READONLY)
    cur = conn.cursor()

    all_r: List[np.ndarray] = []
    all_z: List[np.ndarray] = []
    all_f: List[np.ndarray] = []
    all_e: List[float] = []
    all_n: List[int] = []
    all_q: List[float] = []
    all_s: List[float] = []
    all_d: List[np.ndarray] = []
    has_dipole: Optional[bool] = None
    n_skipped_large = 0
    n_charge_filtered = 0

    query = "SELECT * FROM data ORDER BY id"
    rows = cur.execute(query)
    count = 0
    for row in rows:
        if max_structures is not None and count >= max_structures:
            break
        q, s, z, r, e, f, d = _unpack_data_tuple(row)
        n_atoms = int(z.shape[0])
        if n_atoms > natoms:
            n_skipped_large += 1
            if verbose and n_skipped_large <= 5:
                print(f"  Skip row id={row[0]}: {n_atoms} atoms > natoms={natoms}")
            continue

        charge = float(q[0])
        if charge_filter is not None and abs(charge - charge_filter) > 1e-6:
            n_charge_filtered += 1
            continue

        r64 = np.asarray(r, dtype=np.float64)
        z32 = np.asarray(z, dtype=np.int32)
        f64 = np.asarray(f, dtype=np.float64)

        r_pad = np.zeros((natoms, 3), dtype=np.float64)
        r_pad[:n_atoms] = r64
        z_pad = np.zeros(natoms, dtype=np.int32)
        z_pad[:n_atoms] = z32
        f_pad = np.zeros((natoms, 3), dtype=np.float64)
        f_pad[:n_atoms] = f64

        all_r.append(r_pad)
        all_z.append(z_pad)
        all_f.append(f_pad)
        all_e.append(float(e[0]))
        all_n.append(n_atoms)
        all_q.append(charge)
        all_s.append(_spin_for_mmml(float(s[0]), spin_mode))

        dip_present = d is not None
        if has_dipole is None:
            has_dipole = dip_present
        elif has_dipole != dip_present:
            raise ValueError(
                "Inconsistent dipole: some rows have NULL dipole blob, some do not."
            )
        if dip_present:
            all_d.append(np.asarray(d, dtype=np.float64).reshape(3))

        count += 1

    if not all_r:
        raise ValueError(
            "No structures loaded. Check natoms, charge_filter, or database contents."
        )

    if verbose and n_skipped_large:
        print(f"  Skipped {n_skipped_large} structures with > {natoms} atoms")
    if verbose and n_charge_filtered:
        print(f"  Filtered out {n_charge_filtered} structures (charge != {charge_filter})")

    out: Dict[str, np.ndarray] = {
        "R": np.array(all_r, dtype=np.float64),
        "Z": np.array(all_z, dtype=np.int32),
        "F": np.array(all_f, dtype=np.float64),
        "E": np.array(all_e, dtype=np.float64).reshape(-1, 1),
        "N": np.array(all_n, dtype=np.int32).reshape(-1, 1),
        "Q": np.array(all_q, dtype=np.float64).reshape(-1, 1),
        "S": np.array(all_s, dtype=np.float64).reshape(-1, 1),
    }
    if has_dipole and all_d:
        out["D"] = np.array(all_d, dtype=np.float64)
    return out


def _cache_key_sqlite(
    filepath: Path,
    natoms: int,
    max_structures: Optional[int],
    charge_filter: Optional[float],
    spin_mode: str,
) -> str:
    parts = (
        f"{filepath.resolve()}|{natoms}|{max_structures}|{charge_filter}|{spin_mode}"
    )
    return hashlib.sha256(parts.encode()).hexdigest()[:16]


def _get_cache_dir(
    filepath: Path,
    cache_dir: Optional[Path],
    natoms: int,
    max_structures: Optional[int],
    charge_filter: Optional[float],
    spin_mode: str,
) -> Path:
    if cache_dir is None:
        cache_dir = filepath.parent / ".sqlite_cache"
    h = _cache_key_sqlite(
        filepath, natoms, max_structures, charge_filter, spin_mode
    )
    return cache_dir / f"{filepath.stem}_{h}"


def max_atoms_in_sqlite(database: Path) -> int:
    """Single pass: maximum atom count from Z blobs (for --natoms auto)."""
    conn = apsw.Connection(str(database), flags=apsw.SQLITE_OPEN_READONLY)
    cur = conn.cursor()
    m = 0
    for row in cur.execute("SELECT * FROM data ORDER BY id"):
        n = len(row[3]) // 4
        if n > m:
            m = n
    return m


def load_or_save_sqlite_orbax_cache(
    database: Path,
    natoms: int,
    cache_dir: Optional[Path] = None,
    max_structures: Optional[int] = None,
    charge_filter: Optional[float] = None,
    spin_mode: str = "unpaired_plus_one",
    cache: bool = True,
    verbose: bool = False,
) -> Tuple[Dict[str, np.ndarray], Path]:
    """
    Load MMML dict from SQLite, or restore from Orbax cache if present.

    Returns (data dict, cache path used or that would be used).
    """
    import orbax.checkpoint
    from flax.training import orbax_utils

    cache_path = _get_cache_dir(
        database, cache_dir, natoms, max_structures, charge_filter, spin_mode
    )
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    if cache and cache_path.exists():
        if verbose:
            print(f"Loading from orbax cache: {cache_path}")
        data = checkpointer.restore(cache_path)
        data = {k: np.asarray(v) for k, v in data.items()}
        return data, cache_path

    data = sqlite_rows_to_mmml_arrays(
        database,
        natoms=natoms,
        max_structures=max_structures,
        charge_filter=charge_filter,
        spin_mode=spin_mode,
        verbose=verbose,
    )

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Saving orbax cache to: {cache_path}")
        save_args = orbax_utils.save_args_from_target(data)
        checkpointer.save(cache_path, data, save_args=save_args)
        if verbose:
            print("  Cache saved.")

    return data, cache_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build MMML-compatible Orbax dataset cache from a QCML SQLite DB."
    )
    p.add_argument("database", type=str, help="Path to .sqlite / .db file")
    p.add_argument(
        "--natoms",
        type=int,
        default=None,
        help="Pad/truncate slot size (max atoms). Default: auto from DB.",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Orbax cache root. Default: <parent of db>/.sqlite_cache/",
    )
    p.add_argument("--max-structures", type=int, default=None)
    p.add_argument(
        "--charge-filter",
        type=float,
        default=None,
        help="If set, keep only structures with this total charge.",
    )
    p.add_argument(
        "--spin-mode",
        choices=("unpaired_plus_one", "as_is"),
        default="unpaired_plus_one",
        help=(
            "How to map SQLite spin column to MMML S: "
            "'unpaired_plus_one' treats it as # unpaired e⁻ → multiplicity n+1; "
            "'as_is' uses the value as multiplicity already."
        ),
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Only print summary (still loads DB; use with small tests).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    db = Path(args.database).resolve()
    if not db.is_file():
        raise FileNotFoundError(db)

    natoms = args.natoms
    if natoms is None:
        natoms = max_atoms_in_sqlite(db)
        if args.verbose:
            print(f"Auto natoms (max atoms in DB) = {natoms}")

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None

    if args.no_save:
        data = sqlite_rows_to_mmml_arrays(
            db,
            natoms=natoms,
            max_structures=args.max_structures,
            charge_filter=args.charge_filter,
            spin_mode=args.spin_mode,
            verbose=args.verbose,
        )
        cache_path = _get_cache_dir(
            db, cache_dir, natoms, args.max_structures, args.charge_filter, args.spin_mode
        )
        print(f"Would write cache to: {cache_path}")
    else:
        data, cache_path = load_or_save_sqlite_orbax_cache(
            db,
            natoms=natoms,
            cache_dir=cache_dir,
            max_structures=args.max_structures,
            charge_filter=args.charge_filter,
            spin_mode=args.spin_mode,
            cache=True,
            verbose=args.verbose,
        )

    print(f"Samples: {len(data['E'])}")
    print(f"Cache directory: {cache_path}")
    print("Shapes:")
    for k, v in data.items():
        print(f"  {k}: {v.shape}  {v.dtype}")


if __name__ == "__main__":
    main()
