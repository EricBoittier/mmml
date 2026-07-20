"""Attach a hybrid ML/MM ASE calculator with a live CHARMM PSF (vacuum)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from .config import HybridModeCheckSetup
from .geometry import composition_n_monomers, load_atoms_xyz


def build_psf_and_attach_hybrid(
    setup: HybridModeCheckSetup,
    *,
    write_psf_to: Path | None = None,
) -> tuple[Atoms, list[int], dict[str, Any]]:
    """Build vacuum geometry + CHARMM PSF, attach hybrid calculator.

    Returns ``(atoms, atoms_per_monomer, meta)``.

    When ``setup.xyz`` is unset, geometries come from the md-pbc-suite residue
    builders (PSF atom order). When ``xyz`` is set, composition must match that
    atom order.
    """
    from mmml.cli.base import resolve_checkpoint_paths
    from mmml.cli.run.md_pbc_suite.ase import (
        _build_cluster_psf_from_composition,
        _factory_mmml,
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

    z_psf, _atom_names, atoms_per, residue_labels = _build_cluster_psf_from_composition(
        composition
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
    else:
        # Place each monomer COM along +x using the PSF-built coordinates.
        # ``_build_cluster_psf_from_composition`` already wrote IC coords into CHARMM;
        # read them back and separate monomers.
        pos = coor.get_positions().to_numpy(dtype=float)
        offsets = np.cumsum([0, *atoms_per])
        sep = float(setup.monomer_separation_A)
        placed = pos.copy()
        for i in range(n_mol):
            s, e = int(offsets[i]), int(offsets[i + 1])
            block = placed[s:e]
            com = block.mean(axis=0)
            placed[s:e] = block - com + np.array([sep * i, 0.0, 0.0])
        atoms = Atoms(numbers=np.asarray(z_psf, dtype=int), positions=placed, pbc=False)

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
    )
    atoms.calc = calc
    meta = {
        "composition": composition,
        "residue_labels": [str(x) for x in residue_labels],
        "atoms_per_monomer": atoms_per_list,
        "n_monomers": n_mol,
        "do_mm_effective": do_mm,
        "do_ml": bool(setup.do_ml),
        "do_ml_dimer": bool(setup.do_ml_dimer),
        "checkpoint": str(Path(setup.checkpoint)),
        "psf_path": str(psf_path) if psf_path is not None else None,
        "mm_charge_mode": str(setup.mm_charge_mode),
        "lr_solver": str(setup.lr_solver),
        "vacuum": True,
    }
    return atoms, atoms_per_list, meta
