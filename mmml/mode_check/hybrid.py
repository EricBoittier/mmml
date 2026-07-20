"""Attach a hybrid ML/MM ASE calculator with a live CHARMM PSF (vacuum)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from .bonds import monomer_slices
from .config import HybridModeCheckSetup
from .geometry import composition_n_monomers, load_atoms_xyz


def com_separations_along_chain(
    positions: np.ndarray,
    atoms_per_monomer: list[int],
) -> list[float]:
    """Nearest-neighbor COM distances for monomers placed along a chain."""
    pos = np.asarray(positions, dtype=float)
    coms = [pos[sl].mean(axis=0) for sl in monomer_slices(atoms_per_monomer)]
    return [
        float(np.linalg.norm(coms[i + 1] - coms[i])) for i in range(len(coms) - 1)
    ]


def reposition_monomers_along_x(
    atoms: Atoms,
    atoms_per_monomer: list[int],
    *,
    separation_A: float,
) -> np.ndarray:
    """Re-center each monomer COM along +x at the requested spacing (in-place)."""
    pos = np.asarray(atoms.get_positions(), dtype=float).copy()
    offsets = np.cumsum([0, *[int(n) for n in atoms_per_monomer]])
    sep = float(separation_A)
    for i in range(len(atoms_per_monomer)):
        s, e = int(offsets[i]), int(offsets[i + 1])
        block = pos[s:e]
        com = block.mean(axis=0)
        pos[s:e] = block - com + np.array([sep * i, 0.0, 0.0], dtype=float)
    atoms.set_positions(pos)
    return pos


def place_monomers_along_x(
    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]],
    ordered_residue_names: list[str],
    atoms_per_monomer: list[int],
    *,
    separation_A: float,
) -> np.ndarray:
    """Stack PSF-ordered monomer templates along +x (COM separation).

    ``residue_geometries[res]`` is ``(coords, atom_names, Z)`` in PSF atom order.
    """
    offsets = np.cumsum([0, *[int(n) for n in atoms_per_monomer]])
    n_atoms = int(offsets[-1])
    placed = np.zeros((n_atoms, 3), dtype=float)
    sep = float(separation_A)
    for i, residue in enumerate(ordered_residue_names):
        s, e = int(offsets[i]), int(offsets[i + 1])
        block = np.asarray(residue_geometries[residue][0], dtype=float).copy()
        if block.shape != (e - s, 3):
            raise RuntimeError(
                f"Geometry shape mismatch for {residue} monomer {i}: "
                f"{block.shape} vs expected ({e - s}, 3)"
            )
        com = block.mean(axis=0)
        placed[s:e] = block - com + np.array([sep * i, 0.0, 0.0], dtype=float)
    return placed


def min_intermolecular_distance_A(
    positions: np.ndarray,
    atoms_per_monomer: list[int],
) -> float | None:
    """Minimum atom–atom distance between different monomers (Å)."""
    if len(atoms_per_monomer) < 2:
        return None
    pos = np.asarray(positions, dtype=float)
    slices = monomer_slices(atoms_per_monomer)
    min_d = float("inf")
    for i in range(len(slices)):
        for j in range(i + 1, len(slices)):
            a = pos[slices[i]]
            b = pos[slices[j]]
            d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
            min_d = min(min_d, float(np.min(d)))
    return None if not np.isfinite(min_d) else min_d


def assert_resolved_vacuum_geometry(
    positions: np.ndarray,
    atoms_per_monomer: list[int],
    *,
    min_intramolecular_distance_A: float = 0.2,
    min_intermolecular_distance_threshold_A: float = 1.2,
) -> None:
    """Fail fast if CHARMM/IC left coincident atoms (null IC table symptom).

    Also rejects unoriented close dimers whose atom–atom contacts fall below
    ``min_intermolecular_distance_threshold_A`` (COM spacing ≠ safe spacing).
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise RuntimeError(f"expected (N,3) positions, got shape {pos.shape}")
    if not np.all(np.isfinite(pos)):
        raise RuntimeError("non-finite coordinates in vacuum cluster geometry")
    if float(np.max(np.ptp(pos, axis=0))) <= 1.0e-4 and int(sum(atoms_per_monomer)) > 1:
        raise RuntimeError(
            "Unresolved vacuum cluster geometry (all atoms coincident). "
            "CHARMM IC build likely returned a null table; monomer templates "
            "must be placed explicitly after PSF generation."
        )
    offsets = np.cumsum([0, *[int(n) for n in atoms_per_monomer]])
    for i in range(len(atoms_per_monomer)):
        s, e = int(offsets[i]), int(offsets[i + 1])
        block = pos[s:e]
        if block.shape[0] < 2:
            continue
        # Pairwise distances within the monomer (upper triangle).
        d = np.linalg.norm(block[:, None, :] - block[None, :, :], axis=-1)
        iu = np.triu_indices(block.shape[0], k=1)
        min_d = float(np.min(d[iu])) if iu[0].size else float("inf")
        if min_d < float(min_intramolecular_distance_A):
            raise RuntimeError(
                f"Monomer {i} has min intramolecular distance {min_d:.3e} Å "
                f"(<{min_intramolecular_distance_A} Å). Geometry is collapsed; "
                "refusing to run mode-check."
            )
    d_ij = min_intermolecular_distance_A(pos, [int(n) for n in atoms_per_monomer])
    if d_ij is not None and d_ij < float(min_intermolecular_distance_threshold_A):
        raise RuntimeError(
            f"Min inter-monomer atom distance is {d_ij:.3f} Å "
            f"(<{min_intermolecular_distance_threshold_A} Å). Unoriented templates "
            "at small COM spacing often clash; use --far / --monomer-separation 15 "
            "for numerical checks, or pass an oriented --xyz for interacting dimers."
        )


def build_psf_and_attach_hybrid(
    setup: HybridModeCheckSetup,
    *,
    write_psf_to: Path | None = None,
) -> tuple[Atoms, list[int], dict[str, Any]]:
    """Build vacuum geometry + CHARMM PSF, attach hybrid calculator.

    Returns ``(atoms, atoms_per_monomer, meta)``.

    When ``setup.xyz`` is unset, monomer templates come from the md-pbc-suite
    make-res geometries (PSF atom order), placed along +x. When ``xyz`` is set,
    composition must match that atom order.
    """
    from mmml.cli.base import resolve_checkpoint_paths
    from mmml.cli.run.md_pbc_suite.ase import (
        _build_cluster_psf_from_composition,
        _factory_mmml,
        _residue_geometries_for_composition,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import write_charmm_psf
    from mmml.interfaces.pycharmmInterface.nbonds_config import apply_vacuum_nbonds
    import pycharmm.coor as coor
    import pandas as pd

    composition = [(str(r).upper(), int(n)) for r, n in setup.composition]
    n_mol = composition_n_monomers(composition)
    do_mm = bool(setup.do_mm)
    if n_mol < 2:
        do_mm = False

    # make-res geometries first (each call resets CHARMM), then build the cluster PSF.
    residue_geometries = _residue_geometries_for_composition(composition)
    z_psf, _atom_names, atoms_per, residue_labels = _build_cluster_psf_from_composition(
        composition,
        residue_geometries=residue_geometries,
    )
    apply_vacuum_nbonds(nbxmod=5)

    if setup.xyz is not None:
        atoms, _ = load_atoms_xyz(setup.xyz)
        z_at = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        if list(z_at) != list(np.asarray(z_psf, dtype=int)):
            raise RuntimeError(
                f"XYZ Z {z_at.tolist()} does not match PSF Z {list(map(int, z_psf))}; "
                "atom order must match CHARMM residue topology"
            )
        if len(atoms) != int(sum(atoms_per)):
            raise RuntimeError(
                f"XYZ natoms={len(atoms)} != PSF layout sum={int(sum(atoms_per))}"
            )
        placed = np.asarray(atoms.get_positions(), dtype=float)
    else:
        # Do NOT read CHARMM coords after ic.build(): TIP3/CGENFF often has a null
        # IC table (BILDC warning), leaving all atoms at the origin. Place the
        # already-relaxed make-res templates explicitly (same as
        # ``_build_cluster_from_composition``).
        placed = place_monomers_along_x(
            residue_geometries,
            [str(x) for x in residue_labels],
            [int(n) for n in atoms_per],
            separation_A=float(setup.monomer_separation_A),
        )
        atoms = Atoms(numbers=np.asarray(z_psf, dtype=int), positions=placed, pbc=False)

    assert_resolved_vacuum_geometry(placed, [int(n) for n in atoms_per])
    atoms.set_positions(placed)

    coor.set_positions(
        pd.DataFrame(np.asarray(atoms.get_positions(), float), columns=["x", "y", "z"])
    )

    psf_path = None
    if write_psf_to is not None:
        psf_path = Path(write_psf_to)
        psf_path.parent.mkdir(parents=True, exist_ok=True)
        write_charmm_psf(psf_path)

    base_ckpt_dir, _ = resolve_checkpoint_paths(Path(setup.checkpoint).resolve())
    atoms_per_list = [int(n) for n in atoms_per]
    atoms_per_arg: int | list[int]
    if len(set(atoms_per_list)) == 1:
        atoms_per_arg = int(atoms_per_list[0])
    else:
        atoms_per_arg = atoms_per_list

    calc = _factory_mmml(
        z=np.asarray(atoms.get_atomic_numbers(), dtype=int),
        r=atoms.get_positions(),
        n_mol=n_mol,
        atoms_per=atoms_per_arg,
        base_ckpt_dir=base_ckpt_dir,
        ml_cut=float(setup.ml_switch_width),
        mm_sw=float(setup.mm_switch_on),
        mm_cut=float(setup.mm_switch_width),
        cell_scalar=None,
        verbose=False,
        jax_md_capacity_multiplier=1.25,
        jax_md_capacity_growth_factor=1.5,
        jax_md_max_overflow_retries=4,
        jax_md_overflow_fallback_to_cell_list=True,
        jax_md_update_interval=1,
        jax_md_skin_distance=0.2,
        max_pairs=int(setup.max_pairs),
        do_ml=bool(setup.do_ml),
        do_ml_dimer=bool(setup.do_ml_dimer),
        do_mm=do_mm,
        timings={},
        lr_solver=str(setup.lr_solver),
        mm_charge_mode=str(setup.mm_charge_mode),
        # Mode-check needs trustworthy forces for FIRE / FD / vib; MD keeps
        # analytical forces (backprop=False) for throughput.
        backprop=True,
    )
    atoms.calc = calc
    com_seps = com_separations_along_chain(atoms.get_positions(), atoms_per_list)
    d_ij = min_intermolecular_distance_A(atoms.get_positions(), atoms_per_list)
    meta = {
        "composition": composition,
        "residue_labels": [str(x) for x in residue_labels],
        "atoms_per_monomer": atoms_per_list,
        "n_monomers": n_mol,
        "monomer_separation_A": float(setup.monomer_separation_A),
        "com_separations_A": com_seps,
        "min_intermolecular_distance_A": d_ij,
        "do_mm_effective": do_mm,
        "do_ml": bool(setup.do_ml),
        "do_ml_dimer": bool(setup.do_ml_dimer),
        "checkpoint": str(Path(setup.checkpoint)),
        "psf_path": str(psf_path) if psf_path is not None else None,
        "mm_charge_mode": str(setup.mm_charge_mode),
        "lr_solver": str(setup.lr_solver),
        "vacuum": True,
        "backprop": True,
    }
    return atoms, atoms_per_list, meta
