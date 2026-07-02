#!/usr/bin/env python3
"""Export real CHARMM / Packmol structures for MkDocs figures.

Requires PyCHARMM (``CHARMM_HOME``) for tri-alanine; Packmol for ``make-box`` ACO.
ALAD falls back to the OpenMM benchmark PDB when CHARMM protein build is unavailable.

Run from repo root::

    export CHARMM_HOME=... CHARMM_LIB_DIR=... LD_LIBRARY_PATH=...
    uv run python scripts/export_docs_structure_assets.py

Writes under ``mmml/data/charmm/`` and ``mmml/data/structures/``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PACKMOL = REPO / "mmml" / "generate" / "packmol" / "packmol"

OPENMM_ALAD_PDB_URL = (
    "https://raw.githubusercontent.com/openmm/openmm/master/"
    "wrappers/python/tests/systems/alanine-dipeptide-implicit.pdb"
)


def export_trialanine_water_box(*, seed: int = 11) -> tuple[Path, Path]:
    """CHARMM CGENFF TRIA + TIP3 grid (same recipe as ``build_trialanine_water_box_in_charmm``)."""
    import os

    import pandas as pd
    from ase.io import read as ase_read, write as ase_write

    import pycharmm.coor as coor
    import pycharmm.generate as generate
    import pycharmm.ic as ic
    import pycharmm.read as read
    import pycharmm.settings as settings
    import pycharmm.write as write

    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        crystal_free_charmm_for_param_append,
        ensure_pycharmm_loaded,
        pycharmm,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )
    from mmml.interfaces.pycharmmInterface.nbonds_config import ic_prm_fill
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        TRIA_RESI_NAME,
        _grid_oxygen_sites,
        _load_cgenff_with_trialanine,
        _tip3_template,
        n_peptide_atoms_in_trialanine_box,
    )
    from mmml.paths import bundled_file

    ensure_pycharmm_loaded()
    workdir = REPO / ".docs_export" / "trialanine_water"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    n_waters = 10
    box_side_A = 28.0
    crystal_free_charmm_for_param_append()
    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    # Skip reset_block() here — it can block >100s on an empty session in some envs.

    _load_cgenff_with_trialanine()
    settings.set_verbosity(0)
    read.sequence_string(TRIA_RESI_NAME)
    generate.new_segment(seg_name="PEPT", setup_ic=True)
    ic_prm_fill(replace_all=True)
    ic.build()

    rng = np.random.default_rng(seed)
    peptide = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float).copy()
    peptide -= peptide.mean(axis=0)
    peptide += np.array([box_side_A / 2, box_side_A / 2, box_side_A / 2])
    coor.set_positions(pd.DataFrame(peptide, columns=["x", "y", "z"]))

    tip3 = _tip3_template()
    tip3_com = tip3.mean(axis=0)
    oxygen_sites = _grid_oxygen_sites(
        n_waters=n_waters,
        box_side_A=box_side_A,
        spacing_A=2.85,
        margin_A=3.0,
        existing=peptide,
        min_dist_A=2.4,
        rng=rng,
        water_template=tip3,
    )
    water_coords = np.vstack([site + (tip3 - tip3_com) for site in oxygen_sites])
    read.sequence_string(" ".join(["TIP3"] * n_waters))
    generate.new_segment(seg_name="SOLV", setup_ic=False)
    all_pos = np.vstack([peptide, water_coords])
    coor.set_positions(pd.DataFrame(all_pos, columns=["x", "y", "z"]))
    prepare_charmm_pbc(box_side_A)
    apply_pbc_nbonds(nbxmod=5, cubic_box_side_A=box_side_A)

    prev_cwd = os.getcwd()
    try:
        os.chdir(workdir)
        write.psf_card("trialanine-water.psf")
        write.coor_pdb("trialanine-water.pdb")
    finally:
        os.chdir(prev_cwd)

    psf_path = workdir / "trialanine-water.psf"
    pdb_local = workdir / "trialanine-water.pdb"
    atoms = ase_read(pdb_local)
    atoms.cell = np.diag([box_side_A, box_side_A, box_side_A])
    atoms.pbc = True
    atoms.info["comment"] = (
        f"CGENFF {TRIA_RESI_NAME} + {n_waters}× TIP3 "
        f"(build_trialanine_water_box_in_charmm, seed={seed})"
    )

    extxyz = bundled_file("data", "charmm", "trialanine-water-smoke.extxyz")
    pdb = bundled_file("data", "charmm", "trialanine-water-smoke.pdb")
    extxyz.parent.mkdir(parents=True, exist_ok=True)
    ase_write(extxyz, atoms)
    shutil.copy2(pdb_local, pdb)

    n_pep = n_peptide_atoms_in_trialanine_box(psf_path)
    print(
        f"trialanine-water: {all_pos.shape[0]} atoms "
        f"({n_pep} peptide + {n_waters * 3} water) -> {extxyz.name}"
    )
    return extxyz, pdb


def export_aco_make_box(*, n_molecules: int = 8, side_length: float = 22.0, seed: int = 42) -> Path:
    from ase.io import read as ase_read, write as ase_write

    from mmml.paths import bundled_file, default_aco_template_pdb

    workdir = REPO / ".docs_export" / "aco_make_box"
    if workdir.exists():
        shutil.rmtree(workdir)
    (workdir / "pdb").mkdir(parents=True)
    (workdir / "packmol").mkdir(parents=True)

    shutil.copy2(default_aco_template_pdb(), workdir / "pdb" / "initial.pdb")
    packmol_inp = workdir / "packmol" / "packmol.inp"
    packmol_inp.write_text(
        f"""#
seed {seed}
tolerance 2.0
filetype pdb
output pdb/init-packmol.pdb

structure pdb/initial.pdb
  number {n_molecules}
  inside box 0.0 0.0 0.0 {side_length} {side_length} {side_length}
end structure
"""
    )
    if not PACKMOL.is_file():
        raise FileNotFoundError(f"Packmol binary not found: {PACKMOL}")
    proc = subprocess.run(
        [str(PACKMOL), "-i", str(packmol_inp)],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Packmol failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )

    packed = workdir / "pdb" / "init-packmol.pdb"
    atoms = ase_read(packed)
    atoms.cell = np.diag([side_length, side_length, side_length])
    atoms.pbc = True
    out = bundled_file("data", "structures", "make-box-aco-8x22A.pdb")
    out.parent.mkdir(parents=True, exist_ok=True)
    ase_write(out, atoms, format="proteindatabank")
    print(f"make-box ACO: {len(atoms)} atoms -> {out.name}")
    return out


def export_alad_reference(*, prefer_charmm: bool = True) -> Path:
    from mmml.paths import bundled_file

    out_pdb = bundled_file("data", "charmm", "alad_reference.pdb")
    out_psf = bundled_file("data", "charmm", "alad_reference.psf")
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    if prefer_charmm:
        try:
            from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
            from mmml.interfaces.pycharmmInterface.protein_charmm_build import write_alad_artifacts

            ensure_pycharmm_loaded()
            workdir = REPO / ".docs_export" / "alad"
            if workdir.exists():
                shutil.rmtree(workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            pdb_path, psf_path, build = write_alad_artifacts(workdir, minimize=False)
            shutil.copy2(pdb_path, out_pdb)
            shutil.copy2(psf_path, out_psf)
            print(f"ALAD (CHARMM36): {build.n_atoms} atoms -> {out_pdb.name}")
            return out_pdb
        except Exception as exc:
            print(f"CHARMM ALAD export failed ({exc}); fetching OpenMM benchmark PDB", file=sys.stderr)

    urllib.request.urlretrieve(OPENMM_ALAD_PDB_URL, out_pdb)
    if out_psf.is_file():
        out_psf.unlink()
    print(f"ALAD (OpenMM benchmark): -> {out_pdb.name}")
    return out_pdb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-charmm",
        action="store_true",
        help="Skip CHARMM exports (trialanine); fetch ALAD from OpenMM only",
    )
    args = parser.parse_args()

    os.chdir(REPO)
    try:
        if not args.skip_charmm:
            export_trialanine_water_box()
        export_aco_make_box()
        export_alad_reference(prefer_charmm=not args.skip_charmm)
    finally:
        export_tmp = REPO / ".docs_export"
        if export_tmp.exists():
            shutil.rmtree(export_tmp, ignore_errors=True)

    print("export_docs_structure_assets: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
