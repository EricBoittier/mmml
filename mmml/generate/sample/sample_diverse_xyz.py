#!/usr/bin/env python
"""
Select the top-N diverse structures from one or more multi-frame XYZ files.

Diversity is measured in SOAP descriptor space (dscribe): greedy farthest-point
sampling (k-center style) on column-standardized inner-averaged SOAP vectors.

Writes a compressed NPZ compatible with other MMML trajectories::

    R: list of (n_i, 3) position arrays (float64)
    Z: list of chemical-symbol lists (one list per frame)
    N: list of atom counts (int)

Optional provenance::

    source_path: object array of source XYZ path strings
    frame_index: int64 index of the frame within that XYZ
    diversity_method: str (fixed)
    soap_params_json: str with SOAP hyperparameters

Example::

    python -m mmml.generate.sample.sample_diverse_xyz \\
        bench/*.xyz -n 64 -o sampled.npz --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SPECIES = ["H", "C", "O"]
DEFAULT_R_CUT = 10.0
DEFAULT_N_MAX = 5
DEFAULT_L_MAX = 0
DEFAULT_SIGMA = 0.5


def load_xyz_frames(path: Path) -> list[Any]:
    if not path.exists() or path.stat().st_size == 0:
        if path.exists() and path.stat().st_size == 0:
            logger.warning("Skipping empty file: %s", path)
        return []
    from ase.io import read as ase_read

    try:
        traj = ase_read(str(path), index=":")
    except (OSError, ValueError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return []
    if isinstance(traj, list):
        return traj
    return [traj]


def _make_soap(
    species: list[str],
    r_cut: float,
    n_max: int,
    l_max: int,
    sigma: float,
):
    from dscribe.descriptors import SOAP

    return SOAP(
        species=species,
        periodic=False,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        average="inner",
    )


def com_center_positions(atoms: Any) -> np.ndarray:
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    com = np.sum(pos * masses[:, None], axis=0) / np.sum(masses)
    return pos - com


def frame_signature(atoms: Any) -> tuple[int, tuple[str, ...]]:
    syms = tuple(atoms.get_chemical_symbols())
    return len(syms), syms


def collect_frames_from_paths(paths: list[Path]) -> tuple[list[Any], list[str], list[int]]:
    """Load all frames; require identical (n_atoms, element tuple) across pool."""
    all_atoms: list[Any] = []
    sources: list[str] = []
    indices: list[int] = []
    ref_sig: tuple[int, tuple[str, ...]] | None = None

    for path in paths:
        p = path.resolve()
        frames = load_xyz_frames(p)
        for fi, atoms in enumerate(frames):
            sig = frame_signature(atoms)
            if ref_sig is None:
                ref_sig = sig
            elif sig != ref_sig:
                raise ValueError(
                    f"Inconsistent stoichiometry/order: {p} frame {fi} has "
                    f"n={sig[0]} symbols={sig[1]!r}, expected n={ref_sig[0]} "
                    f"symbols={ref_sig[1]!r}. All XYZs must match atom count and order."
                )
            all_atoms.append(atoms)
            sources.append(str(p))
            indices.append(fi)

    return all_atoms, sources, indices


def soap_matrix(
    atoms_list: list[Any],
    *,
    species: list[str],
    r_cut: float,
    n_max: int,
    l_max: int,
    sigma: float,
) -> np.ndarray:
    soap = _make_soap(species, r_cut, n_max, l_max, sigma)
    rows: list[np.ndarray] = []
    from ase import Atoms as ASEAtoms

    for atoms in atoms_list:
        pos_c = com_center_positions(atoms)
        a = ASEAtoms(symbols=atoms.get_chemical_symbols(), positions=pos_c)
        desc = soap.create(a)
        rows.append(np.asarray(desc, dtype=np.float64).ravel())
    return np.vstack(rows)


def greedy_farthest_first(
    X: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """k-center greedy on rows of X (Euclidean), standardized internally."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    if n == 0:
        raise ValueError("no structures to sample")
    mu = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-12
    Xs = (X - mu) / sig
    if k >= n:
        return np.arange(n, dtype=np.int64)

    first = int(rng.integers(n))
    selected = [first]
    min_dist = np.linalg.norm(Xs - Xs[first], axis=1)
    min_dist[first] = -np.inf

    for _ in range(k - 1):
        idx = int(np.argmax(min_dist))
        if not np.isfinite(min_dist[idx]) or min_dist[idx] < 0:
            break
        selected.append(idx)
        new_d = np.linalg.norm(Xs - Xs[idx], axis=1)
        min_dist = np.minimum(min_dist, new_d)
        sel = np.array(selected, dtype=np.int64)
        min_dist[sel] = -np.inf

    return np.array(selected, dtype=np.int64)


def write_sampled_npz(
    out_path: Path,
    atoms_list: list[Any],
    chosen: np.ndarray,
    *,
    sources: list[str],
    frame_indices: list[int],
    soap_meta: dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    R_list = []
    Z_list = []
    N_list = []
    src_out: list[str] = []
    idx_out: list[int] = []

    for j in chosen:
        at = atoms_list[int(j)]
        R_list.append(np.asarray(at.get_positions(), dtype=np.float64))
        Z_list.append(list(at.get_chemical_symbols()))
        N_list.append(len(at))
        src_out.append(sources[int(j)])
        idx_out.append(int(frame_indices[int(j)]))

    np.savez_compressed(
        out_path,
        R=R_list,
        Z=Z_list,
        N=N_list,
        source_path=np.asarray(src_out, dtype=object),
        frame_index=np.asarray(idx_out, dtype=np.int64),
        diversity_method=np.array("greedy_farthest_soap"),
        soap_params_json=np.array(json.dumps(soap_meta)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Save top-N diverse structures from XYZ files to sampled.npz (SOAP space)."
    )
    p.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more multi-frame XYZ files (same stoichiometry and atom order).",
    )
    p.add_argument(
        "-n",
        "--n-structures",
        type=int,
        required=True,
        dest="n_structures",
        metavar="N",
        help="Number of diverse structures to keep.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sampled.npz"),
        help="Output NPZ path (default: sampled.npz).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the first farthest-point seed (default: 0).",
    )
    p.add_argument(
        "--species",
        type=str,
        default=",".join(DEFAULT_SPECIES),
        help="Comma-separated chemical symbols for SOAP (default: H,C,O).",
    )
    p.add_argument("--r-cut", type=float, default=DEFAULT_R_CUT)
    p.add_argument("--n-max", type=int, default=DEFAULT_N_MAX)
    p.add_argument("--l-max", type=int, default=DEFAULT_L_MAX)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    paths = [p.resolve() for p in args.inputs]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")

    species = [s.strip() for s in args.species.split(",") if s.strip()]

    atoms_list, sources, frame_indices = collect_frames_from_paths(paths)
    if not atoms_list:
        raise SystemExit("No frames loaded from inputs.")

    n_pool = len(atoms_list)
    k = min(args.n_structures, n_pool)
    if k < args.n_structures:
        logger.warning(
            "Requested %d structures but only %d available; saving %d.",
            args.n_structures,
            n_pool,
            k,
        )

    try:
        X = soap_matrix(
            atoms_list,
            species=species,
            r_cut=args.r_cut,
            n_max=args.n_max,
            l_max=args.l_max,
            sigma=args.sigma,
        )
    except ImportError as e:
        raise SystemExit(
            "dscribe is required. Install with: pip install -e '.[quantum]'"
        ) from e

    rng = np.random.default_rng(args.seed)
    chosen = greedy_farthest_first(X, k, rng)

    soap_meta = {
        "species": species,
        "r_cut": args.r_cut,
        "n_max": args.n_max,
        "l_max": args.l_max,
        "sigma": args.sigma,
        "average": "inner",
        "periodic": False,
        "pool_size": n_pool,
        "n_selected": int(len(chosen)),
        "seed": args.seed,
    }

    write_sampled_npz(
        args.output,
        atoms_list,
        chosen,
        sources=sources,
        frame_indices=frame_indices,
        soap_meta=soap_meta,
    )

    print(f"Wrote {args.output.resolve()} ({len(chosen)} structures from pool of {n_pool}).")


if __name__ == "__main__":
    main()
