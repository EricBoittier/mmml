#!/usr/bin/env python3
"""Export real CHARMM / Packmol structures for MkDocs figures.

Requires PyCHARMM (``CHARMM_HOME``) and the bundled Packmol binary.

Run from repo root::

    ./scripts/mmml-charmm-mpirun.sh python scripts/export_docs_structure_assets.py

Writes under ``mmml/data/charmm/`` and ``mmml/data/structures/``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "mmml" / "data"
CHARMM_DATA = DATA / "charmm"
STRUCT_DATA = DATA / "structures"
PACKMOL = REPO / "mmml" / "generate" / "packmol" / "packmol"


def export_trialanine_water_box(*, seed: int = 11) -> tuple[Path, Path]:
    from ase.io import read as ase_read, write as ase_write

    import pycharmm.write as write

    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        TRIA_RESI_NAME,
        build_trialanine_water_box_in_charmm,
        n_peptide_atoms_in_trialanine_box,
    )
    from mmml.paths import bundled_file

    ensure_pycharmm_loaded()
    workdir = REPO / ".docs_export" / "trialanine_water"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    box = build_trialanine_water_box_in_charmm(
        n_waters=10,
        box_side_A=28.0,
        seed=seed,
        workdir=workdir,
    )
    pdb_local = workdir / "trialanine-water-smoke.pdb"
    write.pdb(str(pdb_local))

    atoms = ase_read(pdb_local)
    side = float(box.box_side_A)
    atoms.cell = np.diag([side, side, side])
    atoms.pbc = True
    atoms.info["comment"] = (
        f"CGENFF {TRIA_RESI_NAME} + 10× TIP3 from build_trialanine_water_box_in_charmm "
        f"(seed={seed})"
    )

    extxyz = bundled_file("data", "charmm", "trialanine-water-smoke.extxyz")
    pdb = bundled_file("data", "charmm", "trialanine-water-smoke.pdb")
    extxyz.parent.mkdir(parents=True, exist_ok=True)
    ase_write(extxyz, atoms)
    shutil.copy2(pdb_local, pdb)

    n_pep = n_peptide_atoms_in_trialanine_box(box.psf_path)
    print(
        f"trialanine-water: {box.positions.shape[0]} atoms "
        f"({n_pep} peptide + {box.n_waters * 3} water) -> {extxyz.name}"
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

    monomer_pdb = default_aco_template_pdb()
    initial = workdir / "pdb" / "initial.pdb"
    shutil.copy2(monomer_pdb, initial)

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
    if not packed.is_file():
        raise FileNotFoundError(f"Packmol did not write {packed}")

    atoms = ase_read(packed)
    atoms.cell = np.diag([side_length, side_length, side_length])
    atoms.pbc = True
    out = bundled_file("data", "structures", "make-box-aco-8x22A.pdb")
    out.parent.mkdir(parents=True, exist_ok=True)
    ase_write(out, atoms, format="proteindatabank")
    print(f"make-box ACO: {len(atoms)} atoms -> {out.name}")
    return out


def export_alad_reference() -> Path:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.interfaces.pycharmmInterface.protein_charmm_build import write_alad_artifacts
    from mmml.paths import bundled_file

    ensure_pycharmm_loaded()
    workdir = REPO / ".docs_export" / "alad"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    pdb_path, psf_path, build = write_alad_artifacts(workdir, minimize=True)
    out_pdb = bundled_file("data", "charmm", "alad_reference.pdb")
    out_psf = bundled_file("data", "charmm", "alad_reference.psf")
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdb_path, out_pdb)
    shutil.copy2(psf_path, out_psf)
    print(f"ALAD: {build.n_atoms} atoms -> {out_pdb.name}, {out_psf.name}")
    return out_pdb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-alad",
        action="store_true",
        help="Skip protein ALAD export (no protein toppar)",
    )
    args = parser.parse_args()

    os.chdir(REPO)
    try:
        export_trialanine_water_box()
        export_aco_make_box()
        if not args.skip_alad:
            try:
                export_alad_reference()
            except FileNotFoundError as exc:
                print(f"ALAD export skipped: {exc}", file=sys.stderr)
    finally:
        export_tmp = REPO / ".docs_export"
        if export_tmp.exists():
            shutil.rmtree(export_tmp, ignore_errors=True)

    print("export_docs_structure_assets: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
