#!/usr/bin/env python3
"""Evaluate alanine-dipeptide MM energy with jax-md or MMML JAX bonded loaders.

Examples (CPU recommended for smoke):

  # After charmm_build_protein_alad.py:
  JAX_PLATFORMS=cpu uv run python scripts/examples/jaxmd_protein_alad_energy.py \\
    --pdb /tmp/alad_charmm/alad.pdb \\
    --psf /tmp/alad_charmm/alad.psf \\
    --prm $CHARMM_HOME/toppar/par_all36m_prot.prm \\
    --loader mmml-bonded

  # jax-md OPLS-AA loader (bonded + optional nonbonded):
  JAX_PLATFORMS=cpu uv run python scripts/examples/jaxmd_protein_alad_energy.py \\
    --pdb /tmp/alad_charmm/alad.pdb \\
    --rtf $CHARMM_HOME/toppar/top_all36_prot.rtf \\
    --prm $CHARMM_HOME/toppar/par_all36m_prot.prm \\
    --loader jaxmd-oplsaa --nonbonded

See ``docs/protein-force-fields.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JAX protein MM energy smoke")
    parser.add_argument("--pdb", type=Path, required=True, help="Protein PDB coordinates")
    parser.add_argument(
        "--psf",
        type=Path,
        default=None,
        help="CHARMM PSF EXT (required for --loader mmml-bonded)",
    )
    parser.add_argument(
        "--rtf",
        type=Path,
        default=None,
        help="CHARMM RTF (required for --loader jaxmd-oplsaa)",
    )
    parser.add_argument(
        "--prm",
        type=Path,
        default=None,
        help="CHARMM PRM (protein or CGENFF); defaults from CHARMM_HOME for jaxmd-oplsaa",
    )
    parser.add_argument(
        "--loader",
        choices=("mmml-bonded", "jaxmd-oplsaa"),
        default="mmml-bonded",
        help="mmml-bonded: cgenff_topology PSF loader; jaxmd-oplsaa: jax_md.mm_forcefields.oplsaa",
    )
    parser.add_argument(
        "--nonbonded",
        action="store_true",
        help="Include jax-md OPLS-AA nonbonded terms (jaxmd-oplsaa only)",
    )
    parser.add_argument(
        "--box-side",
        type=float,
        default=50.0,
        help="Cubic box side (Å) for jax-md nonbonded (vacuum cluster)",
    )
    return parser.parse_args()


def _positions_from_pdb(pdb_path: Path) -> np.ndarray:
    from ase.io import read as ase_read

    return np.asarray(ase_read(pdb_path).get_positions(), dtype=np.float64)


def _mmml_bonded_energy(positions: np.ndarray, psf_path: Path, prm_path: Path | None) -> dict[str, float]:
    from mmml.interfaces.pycharmmInterface.cgenff_bonded import (
        bonded_energy_and_forces_from_system,
        bonded_energy_components_from_system,
    )
    from mmml.interfaces.pycharmmInterface.cgenff_topology import load_cgenff_bonded_from_psf

    system = load_cgenff_bonded_from_psf(
        psf_path,
        positions,
        prm_file=prm_path,
    )
    comp = bonded_energy_components_from_system(system, jnp.asarray(positions))
    bonded_total, _forces = bonded_energy_and_forces_from_system(
        system,
        jnp.asarray(positions),
        energy_unit="kcal/mol",
    )
    out = {k: float(v) for k, v in comp.items()}
    out["bonded_total"] = float(bonded_total)
    return out


def _jaxmd_oplsaa_energy(
    positions: np.ndarray,
    pdb_path: Path,
    rtf_path: Path,
    prm_path: Path,
    *,
    include_nonbonded: bool,
    box_side: float,
) -> dict[str, float]:
    from jax_md.mm_forcefields.base import NonbondedOptions
    from jax_md.mm_forcefields.nonbonded.electrostatics import CutoffCoulomb
    from jax_md.mm_forcefields.oplsaa import energy as oplsaa_energy
    from jax_md.mm_forcefields.oplsaa import load_charmm_system

    pos_j, topology, parameters = load_charmm_system(
        str(pdb_path),
        str(prm_path),
        str(rtf_path),
    )
    if not include_nonbonded:
        from jax_md.mm_forcefields.io.charmm import parse_rtf

        from mmml.interfaces.pycharmmInterface.cgenff_bonded import bonded_energy_components
        from mmml.interfaces.pycharmmInterface.cgenff_topology import (
            urey_arrays_for_topology_angles,
        )

        _, rtf_atoms, _, _, _ = parse_rtf(str(rtf_path))
        atom_types = tuple(atom.type for atom in rtf_atoms)
        urey_k, urey_r0 = urey_arrays_for_topology_angles(
            atom_types,
            topology.angles,
            prm_path,
        )
        comp = bonded_energy_components(
            pos_j,
            topology,
            parameters.bonded,
            urey_k=urey_k,
            urey_r0=urey_r0,
        )
        return {k: float(v) for k, v in comp.items()}

    box = jnp.array([float(box_side)] * 3, dtype=jnp.float64)
    coulomb = CutoffCoulomb(r_cut=12.0)
    nb_options = NonbondedOptions(r_cut=12.0, use_pbc=False, scale_14_lj=0.5, scale_14_coul=0.5)
    energy_fn, neighbor_fn, _disp, _shift = oplsaa_energy(
        topology,
        parameters,
        box,
        coulomb,
        nb_options,
    )
    nbrs = neighbor_fn.allocate(pos_j)
    terms = energy_fn(pos_j, nbrs)
    return {str(k): float(v) for k, v in terms.items()}


def main() -> int:
    args = _parse_args()
    if not args.pdb.is_file():
        print(f"PDB not found: {args.pdb}", file=sys.stderr)
        return 2

    positions = _positions_from_pdb(args.pdb)
    print(f"Atoms: {positions.shape[0]} from {args.pdb}")

    prm_path = args.prm
    if prm_path is None and args.loader == "jaxmd-oplsaa":
        from mmml.interfaces.pycharmmInterface.protein_charmm_build import protein_toppar_paths

        prm_path = protein_toppar_paths().prm
    if args.loader == "jaxmd-oplsaa":
        rtf_path = args.rtf
        if rtf_path is None:
            from mmml.interfaces.pycharmmInterface.protein_charmm_build import protein_toppar_paths

            rtf_path = protein_toppar_paths().rtf
        if prm_path is None or not prm_path.is_file():
            print("--prm or CHARMM_HOME protein toppar required for jaxmd-oplsaa", file=sys.stderr)
            return 2
        terms = _jaxmd_oplsaa_energy(
            positions,
            args.pdb,
            rtf_path,
            prm_path,
            include_nonbonded=bool(args.nonbonded),
            box_side=float(args.box_side),
        )
    else:
        if args.psf is None or not args.psf.is_file():
            print("--psf required for mmml-bonded", file=sys.stderr)
            return 2
        terms = _mmml_bonded_energy(positions, args.psf, prm_path)

    for key in sorted(terms):
        print(f"  {key}: {terms[key]:.6f} kcal/mol")
    if "total" in terms:
        print(f"JAX total: {terms['total']:.6f} kcal/mol")
    elif "bonded_total" in terms:
        print(f"MMML bonded total: {terms['bonded_total']:.6f} kcal/mol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
