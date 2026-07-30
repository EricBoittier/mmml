"""Mechanical-embedding ML region helpers (shared by md-system + umbrella).

Restrict PhysNet / SpookyNet to a solute complex (e.g. AMM1+CH3CL) while
``mm_nonbonded`` owns solute–solvent and solvent–solvent pairs. Merging the
ML-region ``mol_id`` drops solute–solute MM pairs from the intermolecular list.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mmml.md.system import MolecularSystem

__all__ = [
    "apply_ml_resnames_mechanical_embedding",
    "compact_mol_id",
    "merge_ml_region_mol_id",
    "parse_ml_resnames",
    "per_atom_residue_names",
    "resolve_ml_region_indices",
]


def parse_ml_resnames(raw: Any) -> tuple[str, ...] | None:
    """Normalize CLI string / YAML list into an upper-cased residue-name tuple."""
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = [str(p).strip() for p in raw if str(p).strip()]
    else:
        raise TypeError(f"ml_resnames must be str or sequence; got {type(raw)!r}")
    if not parts:
        return None
    return tuple(parts)


def resolve_ml_region_indices(
    resnames: Sequence[str],
    ml_resnames: Sequence[str],
) -> np.ndarray:
    """Return atom indices whose residue name is in ``ml_resnames`` (case-insensitive)."""
    want = {str(r).strip().upper() for r in ml_resnames}
    idx = [
        i
        for i, name in enumerate(resnames)
        if str(name).strip().upper() in want
    ]
    if not idx:
        raise ValueError(
            f"no atoms match ml_resnames={sorted(want)}; "
            f"available residues={sorted({str(r).strip().upper() for r in resnames})}"
        )
    return np.asarray(idx, dtype=np.int32)


def merge_ml_region_mol_id(
    mol_id: np.ndarray,
    ml_indices: Sequence[int],
) -> np.ndarray:
    """Assign one shared ``mol_id`` to all ML-region atoms (exclude solute–solute MM)."""
    out = np.asarray(mol_id, dtype=np.int32).copy()
    ml = np.asarray(list(ml_indices), dtype=np.int32)
    if ml.size == 0:
        raise ValueError("ml_indices must be non-empty")
    if int(np.min(ml)) < 0 or int(np.max(ml)) >= out.shape[0]:
        raise ValueError("ml_indices out of range for mol_id")
    shared = int(np.min(out[ml]))
    out[ml] = shared
    return out


def compact_mol_id(mol_id: np.ndarray) -> np.ndarray:
    """Renumber molecule ids to contiguous ``0..K-1`` (drop unused ids after merge)."""
    _, compacted = np.unique(np.asarray(mol_id, dtype=np.int32), return_inverse=True)
    return np.asarray(compacted, dtype=np.int32)


def per_atom_residue_names(system: MolecularSystem) -> list[str]:
    """Expand per-molecule ``metadata['residue_names']`` to one label per atom."""
    per_mol = list(system.metadata.get("residue_names") or [])
    if not per_mol:
        raise ValueError(
            "ml_resnames requires metadata['residue_names'] on the MolecularSystem"
        )
    if len(per_mol) == int(system.n_atoms):
        return [str(x) for x in per_mol]
    if not system.monomer_indices:
        raise ValueError("ml_resnames requires system.monomer_indices")
    if len(per_mol) != len(system.monomer_indices):
        raise ValueError(
            f"residue_names length {len(per_mol)} != "
            f"n_molecules {len(system.monomer_indices)}"
        )
    resnames = [""] * int(system.n_atoms)
    for mol_ix, group in enumerate(system.monomer_indices):
        name = str(per_mol[mol_ix])
        for a in np.asarray(group, dtype=int):
            resnames[int(a)] = name
    return resnames


def apply_ml_resnames_mechanical_embedding(
    system: MolecularSystem,
    ml_resnames: Sequence[str],
) -> tuple[MolecularSystem, dict[str, dict], np.ndarray]:
    """Restrict ``ml_intra`` to the solute complex and drop solute–solute MM pairs.

    Returns ``(system, term_kwargs, ml_indices)`` where ``term_kwargs`` is suitable
    for :func:`mmml.md.assemble.assemble_and_run`.
    """
    from mmml.md.builders._topology import monomer_indices_from_mol_id

    resnames = per_atom_residue_names(system)
    ml_indices = resolve_ml_region_indices(resnames, ml_resnames)
    mol_id = merge_ml_region_mol_id(system.mol_id, ml_indices)
    # Merge can leave unused molecule ids (e.g. former CH3CL id); compact so
    # monomer_indices_from_mol_id does not emit empty groups.
    mol_id = compact_mol_id(mol_id)
    monomers = monomer_indices_from_mol_id(mol_id)
    new_system = MolecularSystem(
        R=system.R,
        Z=system.Z,
        box=system.box,
        mol_id=mol_id,
        monomer_indices=monomers,
        water_indices=system.water_indices,
        psf_path=system.psf_path,
        ff_params=system.ff_params,
        metadata={
            **dict(system.metadata),
            "ml_atom_indices": ml_indices.tolist(),
            "ml_resnames": [str(r) for r in ml_resnames],
        },
    )
    term_kwargs = {"ml_intra": {"monomer_indices": [ml_indices]}}
    return new_system, term_kwargs, ml_indices
