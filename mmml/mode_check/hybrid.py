"""Attach a hybrid ML/MM ASE calculator with a live CHARMM PSF (vacuum)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from .config import HybridModeCheckSetup
from .geometry import (
    build_vacuum_cluster_from_molecules,
    composition_n_monomers,
    load_atoms_xyz,
)


def build_psf_and_attach_hybrid(
    setup: HybridModeCheckSetup,
    *,
    write_psf_to: Path | None = None,
) -> tuple[Atoms, list[int], dict[str, Any]]:
    """Build vacuum geometry + CHARMM PSF, attach hybrid calculator.

    Returns ``(atoms, atoms_per_monomer, meta)``.
    """
    from mmml.cli.base import resolve_checkpoint_paths
    from mmml.cli.run.md_pbc_suite.ase import (
        _factory_mmml,
        _reset_pycharmm_system,
        _read_cgenff_toppar,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        prepare_charmm_vacuum,
        write_charmm_psf,
    )
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        apply_vacuum_nbonds,
        ic_prm_fill,
    )
    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf
    import pycharmm.coor as coor
    import pycharmm.generate as gen
    import pycharmm.ic as ic
    import pycharmm.read as read
    import pandas as pd

    composition = list(setup.composition)
    n_mol = composition_n_monomers(composition)
    do_mm = bool(setup.do_mm)
    if n_mol < 2:
        do_mm = False

    if setup.xyz is not None:
        atoms, apm = load_atoms_xyz(setup.xyz)
        if apm is None:
            # Infer uniform layout from composition residue counts when possible.
            from mmml.analysis.dimer_molecules import MOLECULES

            apm = []
            for res, count in composition:
                key = str(res).upper()
                if key not in MOLECULES:
                    raise KeyError(
                        f"cannot infer atoms_per_monomer for {key}; pass layout via composition of known residues"
                    )
                n_at = len(MOLECULES[key])
                apm.extend([n_at] * int(count))
            if int(sum(apm)) != len(atoms):
                raise ValueError(
                    f"XYZ natoms={len(atoms)} does not match composition layout sum={sum(apm)}"
                )
        residue_labels = []
        for res, count in composition:
            residue_labels.extend([str(res).upper()] * int(count))
    else:
        atoms, apm, residue_labels = build_vacuum_cluster_from_molecules(
            composition,
            separation_A=float(setup.monomer_separation_A),
        )

    sequence_items: list[str] = []
    for res, count in composition:
        sequence_items.extend([str(res).upper()] * int(count))
    sequence = " ".join(sequence_items)

    _reset_pycharmm_system()
    prepare_charmm_vacuum()
    _read_cgenff_toppar()
    read.sequence_string(sequence)
    gen.new_segment(seg_name="CLST", setup_ic=True)
    ic_prm_fill(replace_all=True)
    ic.build()
    coor.set_positions(
        pd.DataFrame(np.asarray(atoms.get_positions(), float), columns=["x", "y", "z"])
    )
    apply_vacuum_nbonds(nbxmod=5)
    z_psf = np.asarray(get_Z_from_psf(), dtype=int)
    z_at = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    if list(z_psf) != list(z_at):
        raise RuntimeError(
            f"PSF Z {z_psf.tolist()} does not match geometry Z {z_at.tolist()}; "
            "check composition vs XYZ atom order"
        )
    psf_path = None
    if write_psf_to is not None:
        psf_path = Path(write_psf_to)
        psf_path.parent.mkdir(parents=True, exist_ok=True)
        write_charmm_psf(psf_path)

    base_ckpt_dir, _ = resolve_checkpoint_paths(Path(setup.checkpoint).resolve())
    atoms_per = [int(n) for n in apm]
    # _factory_mmml expects uniform atoms_per as int when possible
    atoms_per_arg: int | list[int]
    if len(set(atoms_per)) == 1:
        atoms_per_arg = int(atoms_per[0])
    else:
        atoms_per_arg = atoms_per

    calc = _factory_mmml(
        z=z_at,
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
        "composition": [(str(r), int(n)) for r, n in composition],
        "residue_labels": residue_labels,
        "atoms_per_monomer": atoms_per,
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
    return atoms, atoms_per, meta
