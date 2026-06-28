"""Build symmetry-aware crystal structures with PyXtal and optional ASE relaxation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mmml.interfaces.aseInterface.pyxtal_optimize import optimize_ase_atoms
from mmml.interfaces.pyxtal_placement import (
    MolecularCrystalBuildRequest,
    ase_supercell,
    atoms_to_reference_npz,
    build_molecular_crystal_random,
    have_pyxtal,
    parse_stoichiometry,
    parse_supercell_reps,
    write_ase_structure,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build molecular crystals with PyXtal (space-group symmetry) and export "
            "ASE-compatible structures for optimization or MMML handoff."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--molecule",
        action="append",
        required=True,
        metavar="SPEC",
        help=(
            "Molecule specification (repeat for multi-component crystals): "
            "XYZ/CIF path, SMILES, or chemical formula understood by PyXtal"
        ),
    )
    parser.add_argument(
        "--stoichiometry",
        type=int,
        nargs="+",
        default=None,
        metavar="Z",
        help="Formula units per molecule species (same order as --molecule)",
    )
    parser.add_argument(
        "--z",
        dest="z_values",
        type=int,
        nargs="+",
        default=None,
        help="Alias for stoichiometry; one value repeats for all molecules",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=3,
        choices=(0, 1, 2, 3),
        help="Crystal dimensionality (0=cluster, 3=3D periodic)",
    )
    parser.add_argument(
        "--spg",
        "--space-group",
        dest="space_group",
        type=int,
        default=14,
        help="International space-group number",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=1.0,
        help="PyXtal volume factor passed to from_random",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible PyXtal trials",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=20,
        help="Maximum PyXtal from_random retries",
    )
    parser.add_argument(
        "--no-resort",
        action="store_true",
        help="Keep PyXtal atom order in ASE export (to_ase resort=False)",
    )
    parser.add_argument(
        "--supercell",
        type=str,
        default=None,
        metavar="NX,NY,NZ",
        help="Build supercell after generation (e.g. 2,2,2)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output path (.xyz, .extxyz, .cif, or .npz)",
    )
    parser.add_argument(
        "--format",
        dest="out_format",
        default=None,
        help="ASE output format override (default: inferred from --output suffix)",
    )
    opt = parser.add_argument_group("ASE optimization (optional)")
    opt.add_argument(
        "--optimize",
        action="store_true",
        help="Relax structure with ASE after PyXtal generation",
    )
    opt.add_argument(
        "--optimizer",
        choices=("bfgs", "fire", "lbfgs"),
        default="bfgs",
        help="ASE optimizer when --optimize is set",
    )
    opt.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="ASE force convergence (eV/Å)",
    )
    opt.add_argument(
        "--max-opt-steps",
        type=int,
        default=200,
        help="Maximum ASE optimizer steps",
    )
    opt.add_argument(
        "--fix-cell",
        action="store_true",
        help="Document intent to keep the unit cell fixed (positions-only relaxation)",
    )
    opt.add_argument(
        "--emt",
        action="store_true",
        help="Use ASE EMT calculator for --optimize (smoke tests only)",
    )
    opt.add_argument(
        "--quiet-opt",
        action="store_true",
        help="Suppress ASE optimizer log output",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _infer_format(path: Path, override: str | None) -> str | None:
    if override:
        return override
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("xyz", "extxyz", "cif", "json"):
        return suffix
    if suffix == "npz":
        return None
    return "extxyz"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not have_pyxtal():
        print(
            "Error: PyXtal is not installed. Install with: uv sync --extra chem",
            file=sys.stderr,
        )
        return 1

    stoich = parse_stoichiometry(
        args.molecule,
        args.stoichiometry,
        args.z_values,
    )
    request = MolecularCrystalBuildRequest(
        molecules=list(args.molecule),
        stoichiometry=stoich,
        dimension=int(args.dim),
        space_group=int(args.space_group),
        factor=float(args.factor),
        seed=args.seed,
        max_attempts=int(args.attempts),
        resort=not bool(args.no_resort),
    )
    result = build_molecular_crystal_random(request)
    atoms = result.atoms
    print(
        f"PyXtal build OK after {result.attempts} attempt(s); "
        f"spg={result.space_group} formula={result.formula or 'n/a'} "
        f"natoms={len(atoms)}",
        flush=True,
    )

    if args.supercell is not None:
        reps = parse_supercell_reps(args.supercell)
        atoms = ase_supercell(atoms, reps)
        print(f"Supercell {reps[0]}×{reps[1]}×{reps[2]} → natoms={len(atoms)}", flush=True)

    if args.optimize:
        if not args.emt and atoms.calc is None:
            print(
                "Error: --optimize requires --emt or a pre-attached atoms.calc "
                "(e.g. MMML/CHARMM calculator in a notebook).",
                file=sys.stderr,
            )
            return 1
        opt_result = optimize_ase_atoms(
            atoms,
            use_emt=bool(args.emt),
            optimizer=args.optimizer,
            fmax_ev_a=float(args.fmax),
            max_steps=int(args.max_opt_steps),
            fix_cell=bool(args.fix_cell),
            logfile=None if args.quiet_opt else "-",
        )
        atoms = opt_result.atoms
        energy_msg = (
            f", E={opt_result.energy_ev:.6f} eV"
            if opt_result.energy_ev is not None
            else ""
        )
        print(
            f"ASE {opt_result.optimizer} finished: fmax={opt_result.fmax_ev_a:.4f} eV/Å"
            f"{energy_msg}",
            flush=True,
        )

    out = Path(args.output).expanduser().resolve()
    if out.suffix.lower() == ".npz":
        atoms_to_reference_npz(atoms, out, label="pyxtal_build_crystal")
    else:
        write_ase_structure(atoms, out, format=_infer_format(out, args.out_format))
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
