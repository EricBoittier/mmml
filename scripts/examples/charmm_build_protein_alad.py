#!/usr/bin/env python3
"""Build CHARMM36 alanine dipeptide (ACE–ALA–CT3) with PyCHARMM.

User-run on a CHARMM node (not CI / agent sessions):

  ./scripts/mmml-charmm-mpirun.sh python scripts/examples/charmm_build_protein_alad.py \\
    -o /tmp/alad_charmm

Writes ``alad.pdb``, ``alad.psf``, and prints CHARMM total energy. Feed the artifacts
to ``scripts/examples/jaxmd_protein_alad_energy.py``.

See ``docs/protein-force-fields.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CHARMM36 ALAD dipeptide")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("artifacts/protein/alad_charmm"),
        help="Directory for alad.pdb and alad.psf",
    )
    parser.add_argument(
        "--no-minimize",
        action="store_true",
        help="Skip ABNR minimization after IC build",
    )
    parser.add_argument(
        "--mini-steps",
        type=int,
        default=500,
        help="ABNR minimization steps when minimization is enabled",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded

        ensure_pycharmm_loaded()
        from mmml.interfaces.pycharmmInterface.protein_charmm_build import (
            charmm_total_energy_kcalmol,
            protein_toppar_paths,
            write_alad_artifacts,
        )
    except ImportError as exc:
        print(f"PyCHARMM not available: {exc}", file=sys.stderr)
        return 2

    toppar = protein_toppar_paths()
    print(f"Protein toppar: {toppar.rtf.name}, {toppar.prm.name}")

    pdb_path, psf_path, build = write_alad_artifacts(
        args.output_dir,
        minimize=not args.no_minimize,
    )
    e_tot = charmm_total_energy_kcalmol()
    print(f"ALAD atoms: {build.n_atoms}")
    print(f"CHARMM ENER total: {e_tot:.6f} kcal/mol")
    print(f"Wrote {pdb_path}")
    print(f"Wrote {psf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
