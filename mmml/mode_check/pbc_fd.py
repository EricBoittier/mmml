"""PBC cluster finite-difference force check (formerly ``check_fd.py`` main)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from .forces import force_fd_check


def run_pbc_cluster_fd(
    *,
    checkpoint: Path | None,
    residue: str = "MEOH",
    n_molecules: int = 10,
    spacing: float = 5.0,
    min_com_start_distance: float = 6.0,
    ml_cutoff: float = 0.1,
    mm_switch_on: float = 7.0,
    mm_cutoff: float = 5.0,
    fd_check_atoms: int = 3,
    fd_check_dx: float = 1e-3,
    max_pairs: int = 20_000,
    template_pdb: Path | None = None,
    charmm_pre_minimize: bool = False,
    charmm_sd_steps: int = 25,
    charmm_abnr_steps: int = 100,
    charmm_tolenr: float = 1e-3,
    charmm_tolgrd: float = 1e-3,
    jax_md_capacity_multiplier: float = 1.25,
    jax_md_capacity_growth_factor: float = 1.5,
    jax_md_max_overflow_retries: int = 4,
    jax_md_overflow_fallback_to_cell_list: bool = True,
    jax_md_update_interval: int = 1,
    jax_md_skin_distance: float = 0.2,
    lr_solver: str = "mic",
    ewald_omit_self: bool = False,
    mm_charge_mode: str | None = None,
) -> dict[str, Any]:
    """Build a PBC residue cluster, attach hybrid calc, run ``force_fd_check``."""
    from mmml.cli.base import resolve_checkpoint_paths
    from mmml.cli.run.md_pbc_suite.ase import (
        _cubic_box_length,
        _enforce_min_com_separation,
        _factory_mmml,
        _run_charmm_minimize,
    )
    from mmml.cli.run.md_pbc_suite.cluster import _build_psf_ordered_cluster

    if checkpoint is None:
        base_ckpt_dir, _ = resolve_checkpoint_paths(None)
    else:
        base_ckpt_dir, _ = resolve_checkpoint_paths(Path(checkpoint).expanduser().resolve())

    # None → residue-aware bundled template (TIP3/MEOH/ACO); do not force MEOH.
    pdb = (
        Path(template_pdb).expanduser().resolve() if template_pdb is not None else None
    )
    z, r0 = _build_psf_ordered_cluster(
        str(residue).upper(),
        int(n_molecules),
        float(spacing),
        template_pdb=pdb,
    )
    n_mol = int(n_molecules)
    atoms_per = len(z) // n_mol
    monomer_offsets = np.arange(0, n_mol + 1, dtype=int) * int(atoms_per)
    r0 = _enforce_min_com_separation(
        r0,
        monomer_offsets=monomer_offsets,
        min_com_distance=float(min_com_start_distance),
    )
    L = _cubic_box_length(r0, float(ml_cutoff))
    r_pbc = r0 - r0.mean(axis=0) + 0.5 * L
    atoms = Atoms(numbers=z, positions=r_pbc)
    atoms.set_cell([L, L, L])
    atoms.set_pbc(True)
    if charmm_pre_minimize:
        _run_charmm_minimize(
            atoms,
            nstep_sd=int(charmm_sd_steps),
            nstep_abnr=int(charmm_abnr_steps),
            tolenr=float(charmm_tolenr),
            tolgrd=float(charmm_tolgrd),
            timings={},
        )

    calc = _factory_mmml(
        z=z,
        r=atoms.get_positions(),
        n_mol=int(n_molecules),
        atoms_per=atoms_per,
        base_ckpt_dir=base_ckpt_dir,
        ml_cut=float(ml_cutoff),
        mm_sw=float(mm_switch_on),
        mm_cut=float(mm_cutoff),
        cell_scalar=L,
        verbose=False,
        jax_md_capacity_multiplier=float(jax_md_capacity_multiplier),
        jax_md_capacity_growth_factor=float(jax_md_capacity_growth_factor),
        jax_md_max_overflow_retries=int(jax_md_max_overflow_retries),
        jax_md_overflow_fallback_to_cell_list=bool(jax_md_overflow_fallback_to_cell_list),
        jax_md_update_interval=int(jax_md_update_interval),
        jax_md_skin_distance=float(jax_md_skin_distance),
        max_pairs=int(max_pairs),
        timings={},
        lr_solver=str(lr_solver),
        ewald_include_self=not bool(ewald_omit_self),
        mm_charge_mode=mm_charge_mode,
        backprop=True,
    )
    atoms.calc = calc
    result = force_fd_check(atoms, int(fd_check_atoms), float(fd_check_dx))
    result["box_A"] = float(L)
    result["n_molecules"] = float(n_molecules)
    result["residue"] = str(residue).upper()
    result["checkpoint"] = str(base_ckpt_dir)
    result["lr_solver"] = str(lr_solver)
    result["ewald_omit_self"] = bool(ewald_omit_self)
    return result


def write_fd_result(result: dict[str, Any], output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output
