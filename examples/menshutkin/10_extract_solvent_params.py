#!/usr/bin/env python3
"""Extract CGenFF parameters for one solvent into a JSON the runtime can read.

CHARMM is used **once, offline** to resolve a single molecule's geometry,
charges, Lennard-Jones parameters and harmonic bonded terms; everything after
that is pure numpy/JAX-MD. This keeps the force field faithful to CGenFF while
removing live CHARMM state -- and its one-build-per-process limit -- from the
simulation itself.

Because of that limit this script builds exactly one residue and exits, so the
driver loops over solvents with one subprocess each.

    python examples/menshutkin/10_extract_solvent_params.py --residue MEOH \\
        --name methanol --density 792 --box-side 25

Writes ``examples/menshutkin/solvent_params/<name>.json``, which
``solvent_models.py`` picks up automatically.

Adding a solvent to the campaign is therefore: make sure its residue exists in
CGenFF (or supply an append RTF via MMML_CGENFF_EXTRA_RTF), run this once, done.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent
OUT_DIR = EXAMPLE_DIR / "solvent_params"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--residue", required=True, help="CGenFF residue name, e.g. MEOH")
    p.add_argument("--name", required=True, help="Campaign name, e.g. methanol")
    p.add_argument("--density", type=float, required=True, help="kg/m3 at 298 K")
    p.add_argument("--box-side", type=float, required=True, help="Default cube side (A)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded

    if not ensure_pycharmm_loaded():
        raise SystemExit("PyCHARMM not available (CHARMM_LIB_DIR / libcharmm.so)")

    import pycharmm.write as write

    from mmml.cli.run.md_pbc_suite.cluster import build_packmol_composition_cluster
    from mmml.interfaces.pycharmmInterface.cgenff_topology import default_cgenff_paths
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        load_bonded_system_from_psf,
        load_nonbonded_system_from_charmm,
    )

    work = args.out_dir / "_build"
    work.mkdir(parents=True, exist_ok=True)
    psf = work / f"{args.name}.psf"

    # A single molecule in a roomy box: we only want its internal geometry and
    # parameters, not a condensed-phase configuration.
    z, positions, sizes, residues = build_packmol_composition_cluster(
        composition=[(args.residue, 1)], seed=0,
        center=(10.0, 10.0, 10.0), cube_side=20.0,
        placement="cube", tolerance=2.0, quiet=True, verbose=False,
    )
    write.psf_card(str(psf))
    positions = np.asarray(positions, dtype=np.float64)
    z = np.asarray(z, dtype=np.int32)
    n = int(z.shape[0])
    print(f"{args.residue}: {n} atoms")

    _, prm = default_cgenff_paths()
    extra_prm = []
    import os

    if os.environ.get("MMML_CGENFF_EXTRA_PRM"):
        extra_prm = [Path(x) for x in os.environ["MMML_CGENFF_EXTRA_PRM"].split(":") if x]

    nb = load_nonbonded_system_from_charmm(psf, prm, *extra_prm)
    bonded_sys = load_bonded_system_from_psf(
        psf, positions, prm_file=prm, extra_prm_files=tuple(extra_prm)
    )
    topo, bp = bonded_sys.topology, bonded_sys.bonded

    charges = np.asarray(nb.charges, dtype=float)
    payload = {
        "name": args.name,
        "residue": args.residue,
        "Z": z.tolist(),
        # Centre the reference geometry; the builder places it by molecule centre.
        "geometry": (positions - positions.mean(axis=0)).tolist(),
        "charges": charges.tolist(),
        # CHARMM quotes epsilon negative; store the magnitude.
        "epsilon": np.abs(np.asarray(nb.epsilon, dtype=float)).tolist(),
        "rmin_half": np.asarray(nb.rmin, dtype=float).tolist(),
        "bonds": np.asarray(topo.bonds, dtype=int).tolist(),
        "bond_k": np.asarray(bp.bond_k, dtype=float).tolist(),
        "bond_r0": np.asarray(bp.bond_r0, dtype=float).tolist(),
        "angles": np.asarray(topo.angles, dtype=int).tolist(),
        "angle_k": np.asarray(bp.angle_k, dtype=float).tolist(),
        # cgenff_bonded stores theta0 in radians; the runtime wants degrees.
        "angle_theta0": np.rad2deg(np.asarray(bp.angle_theta0, dtype=float)).tolist(),
        "density_kg_m3": args.density,
        "box_side_A": args.box_side,
        "source": "CGenFF via PyCHARMM (10_extract_solvent_params.py)",
    }

    net = float(charges.sum())
    print(f"  net charge {net:+.6f} e")
    print(f"  {len(payload['bonds'])} bonds, {len(payload['angles'])} angles")
    if abs(net) > 1e-4:
        print(f"FAIL: {args.residue} is not neutral", file=sys.stderr)
        return 1
    if len(payload["bonds"]) == 0:
        print(f"FAIL: no bonds found for {args.residue}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
