"""Ragged (unpadded) SPICE-α conversion for full-dataset efield training.

The padded converter (:mod:`mmml.data.spice_alpha`) pads every frame to the
largest system, which for full SPICE-α (3-110 atoms, 1.8M frames) is mostly
padding. Here atoms are stored concatenated with per-frame offsets:

    Z (A,) int8, R (A, 3) float32 Å, F (A, 3) float32 eV/Å      A = total atoms
    N (n,) int16 atoms per frame, offsets (n + 1,) int64
    E (n,) float64 eV, D (n, 3) float32 e·Å, Q (n,) float32 e,
    polar (n, 3, 3) float32 Bohr³, mol (n,) int64 molecule id, subset (n,) int8

Molecule ids let :func:`split_ragged` hold out whole molecules, so conformers
of a training molecule never appear in validation/test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from mmml.data.spice_alpha import iter_spice_alpha_frames
from mmml.data.units import polar_e_angstrom2_per_volt_to_bohr3

RAGGED_FORMAT = "mmml-spice-alpha-ragged-v1"
PER_FRAME_KEYS = ("N", "E", "D", "Q", "polar", "mol", "subset")
PER_ATOM_KEYS = ("Z", "R", "F")

# SPICE-α file stem -> subset name (MACE-MDP Fig. 2 naming)
SUBSETS: dict[str, str] = {
    "DES370K_Dimers": "dimers",
    "DES370K_Monomers": "monomers",
    "SPICE_PubChem_1_2": "pubchem",
    "SPICE_PubChem_3_4": "pubchem",
    "SPICE_PubChem_5_6": "pubchem",
    "SPICE_PubChem_7_8": "pubchem",
    "SPICE_PubChem_9_10": "pubchem",
    "SPICE_Solvated_PubChem": "solvated_pubchem",
    "SPICE_Water": "water",
    "SPICE_amino_acid_ligand": "amino_acid_ligand",
    "SPICE_di_peptide": "dipeptides",
    "SPICE_solvated_amino_acid": "solvated_amino_acid",
}
SUBSET_IDS: dict[str, int] = {name: i for i, name in enumerate(sorted(set(SUBSETS.values())))}


def convert_hdf5_ragged(
    path: Path | str,
    out: Path | str,
    *,
    mol_id_base: int,
    neutral_tol: float = 0.5,
    max_abs_polar_bohr3: float = 1.0e4,
) -> dict[str, int]:
    """Convert one SPICE-α HDF5 file to a ragged NPZ; returns frame counts.

    Keeps neutral frames only (|sum MBIS charges| < ``neutral_tol``, as MACE-MDP)
    and drops frames with non-finite or corrupt (|α_ij| > ``max_abs_polar_bohr3``)
    polarizabilities or non-finite dipoles.
    """
    import h5py

    path = Path(path)
    subset = SUBSETS[path.stem]
    Z, R, F = [], [], []
    N, E, D, Q, P, mol = [], [], [], [], [], []
    group_ids: dict[str, int] = {}
    stats = {"seen": 0, "charged": 0, "bad_polar": 0, "bad_dipole": 0, "kept": 0}
    with h5py.File(path, "r") as h5:
        for fr in iter_spice_alpha_frames(h5, neutral_only=False):
            stats["seen"] += 1
            if fr.Q is not None and abs(fr.Q) > neutral_tol:
                stats["charged"] += 1
                continue
            if fr.polar is None:
                stats["bad_polar"] += 1
                continue
            polar = polar_e_angstrom2_per_volt_to_bohr3(np.asarray(fr.polar, dtype=np.float64))
            if not np.isfinite(polar).all() or np.abs(polar).max() > max_abs_polar_bohr3:
                stats["bad_polar"] += 1
                continue
            if not np.isfinite(fr.D).all():
                stats["bad_dipole"] += 1
                continue
            gid = group_ids.setdefault(fr.group, mol_id_base + len(group_ids))
            Z.append(fr.Z.astype(np.int8))
            R.append(fr.R.astype(np.float32))
            F.append(fr.F.astype(np.float32))
            N.append(fr.Z.shape[0])
            E.append(fr.E)
            D.append(fr.D)
            Q.append(0.0 if fr.Q is None else fr.Q)
            P.append(polar)
            mol.append(gid)
            stats["kept"] += 1
    n = len(N)
    out_arrays = {
        "Z": np.concatenate(Z) if n else np.zeros((0,), np.int8),
        "R": np.concatenate(R) if n else np.zeros((0, 3), np.float32),
        "F": np.concatenate(F) if n else np.zeros((0, 3), np.float32),
        "N": np.asarray(N, dtype=np.int16),
        "E": np.asarray(E, dtype=np.float64),
        "D": np.asarray(D, dtype=np.float32).reshape(n, 3),
        "Q": np.asarray(Q, dtype=np.float32),
        "polar": np.asarray(P, dtype=np.float32).reshape(n, 3, 3),
        "mol": np.asarray(mol, dtype=np.int64),
        "subset": np.full((n,), SUBSET_IDS[subset], dtype=np.int8),
    }
    meta = {"format": RAGGED_FORMAT, "source": path.name, "subset": subset,
            "subset_ids": SUBSET_IDS, "n_molecules": len(group_ids), "stats": stats,
            "units": {"R": "angstrom", "E": "ev", "F": "ev_angstrom", "D": "e_angstrom",
                      "Q": "e", "polar": "bohr3"}}
    np.savez(out, **out_arrays, meta=np.array(json.dumps(meta)))
    return stats


def offsets_from_counts(N: np.ndarray) -> np.ndarray:
    off = np.zeros((len(N) + 1,), dtype=np.int64)
    np.cumsum(np.asarray(N, dtype=np.int64), out=off[1:])
    return off


def load_ragged(paths: Sequence[Path | str]) -> dict[str, np.ndarray]:
    """Concatenate ragged NPZ shards; adds ``offsets``."""
    parts = [np.load(p) for p in paths]
    data = {k: np.concatenate([p[k] for p in parts]) for k in PER_ATOM_KEYS + PER_FRAME_KEYS}
    data["offsets"] = offsets_from_counts(data["N"])
    return data


def take_frames(data: Mapping[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Subset a ragged dict by frame indices (keeps order of ``idx``)."""
    idx = np.asarray(idx, dtype=np.int64)
    off = data["offsets"]
    atom_idx = np.concatenate([np.arange(off[i], off[i + 1]) for i in idx]) if len(idx) else np.zeros((0,), np.int64)
    out = {k: np.asarray(data[k])[atom_idx] for k in PER_ATOM_KEYS}
    out.update({k: np.asarray(data[k])[idx] for k in PER_FRAME_KEYS})
    out["offsets"] = offsets_from_counts(out["N"])
    return out


def split_ragged(
    data: Mapping[str, np.ndarray],
    *,
    valid_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 0,
    min_molecules_for_mol_split: int = 50,
) -> dict[str, np.ndarray]:
    """Frame indices per split, holding out whole molecules.

    Subsets with fewer than ``min_molecules_for_mol_split`` molecules (SPICE
    Water is one 1000-conformer group) are split by frame instead.
    """
    rng = np.random.default_rng(seed)
    subset = np.asarray(data["subset"])
    mol = np.asarray(data["mol"])
    idx = {"train": [], "valid": [], "test": []}
    for s in np.unique(subset):
        frames = np.flatnonzero(subset == s)
        mols = np.unique(mol[frames])
        if len(mols) >= min_molecules_for_mol_split:
            perm = rng.permutation(mols)
            n_v = max(1, int(round(len(mols) * valid_frac)))
            n_t = max(1, int(round(len(mols) * test_frac)))
            role = {m: "valid" for m in perm[:n_v]}
            role.update({m: "test" for m in perm[n_v:n_v + n_t]})
            for f in frames:
                idx[role.get(mol[f], "train")].append(f)
        else:
            perm = rng.permutation(frames)
            n_v = int(round(len(frames) * valid_frac))
            n_t = int(round(len(frames) * test_frac))
            idx["valid"].extend(perm[:n_v])
            idx["test"].extend(perm[n_v:n_v + n_t])
            idx["train"].extend(perm[n_v + n_t:])
    return {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in idx.items()}


def fit_atomic_energy_refs(data: Mapping[str, np.ndarray], max_z: int = 119) -> np.ndarray:
    """Least-squares per-element reference energies e_Z (eV), indexed by Z."""
    off = data["offsets"]
    frame_of_atom = np.repeat(np.arange(len(data["N"])), data["N"].astype(np.int64))
    counts = np.zeros((len(data["N"]), max_z), dtype=np.float64)
    np.add.at(counts, (frame_of_atom, data["Z"].astype(np.int64)), 1.0)
    present = counts.sum(axis=0) > 0
    sol, *_ = np.linalg.lstsq(counts[:, present], np.asarray(data["E"], np.float64), rcond=None)
    e_ref = np.zeros((max_z,), dtype=np.float64)
    e_ref[present] = sol
    assert off[-1] == len(data["Z"])
    return e_ref
