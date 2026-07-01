"""Build symmetry-aware crystal structures with PyXtal and optional ASE relaxation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mmml.interfaces.aseInterface.pyxtal_optimize import optimize_ase_atoms
from mmml.interfaces.crystal_charmm import (
    LITERATURE_CRYSTAL_PRESETS,
    build_charmm_literature_supercell,
    build_literature_charmm_supercell,
)
from mmml.interfaces.pyxtal_placement import (
    MolecularCrystalBuildRequest,
    ase_supercell,
    atoms_to_reference_npz,
    build_molecular_crystal_random,
    crystal_mass_density_g_cm3,
    have_pyxtal,
    parse_stoichiometry,
    parse_supercell_reps,
    scale_atoms_cell_to_density,
    write_ase_structure,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build molecular crystals: literature CIF + make-res (CHARMM names) or "
            "PyXtal random placement with space-group symmetry."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    lit = parser.add_argument_group(
        "Literature CIF + make-res (recommended for DCM / benzene)"
    )
    lit.add_argument(
        "--literature",
        choices=sorted(LITERATURE_CRYSTAL_PRESETS),
        default=None,
        metavar="PRESET",
        help="Bundled experimental CIF preset: dcm (Pbcn) or benz (P2₁/c)",
    )
    lit.add_argument(
        "--from-cif",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override CIF path (requires --residue or --literature for residue name)",
    )
    lit.add_argument(
        "--residue",
        default=None,
        metavar="NAME",
        help="CHARMM residue (DCM, BENZ) when using --from-cif without --literature",
    )
    lit.add_argument(
        "--monomer-pdb",
        type=Path,
        default=None,
        metavar="PATH",
        help="make-res monomer PDB for atom-name mapping (default: pdb/<res>.pdb or bundled)",
    )
    lit.add_argument(
        "--min-box-side",
        type=float,
        default=28.0,
        metavar="ANG",
        help="Minimum supercell edge length (Å); default ≈2× CHARMM cutnb",
    )
    pyx = parser.add_argument_group("PyXtal random placement")
    pyx.add_argument(
        "-m",
        "--molecule",
        action="append",
        default=None,
        metavar="SPEC",
        help=(
            "Molecule specification (repeat for multi-component crystals): "
            "XYZ/CIF path, SMILES, or chemical formula understood by PyXtal"
        ),
    )
    pyx.add_argument(
        "--stoichiometry",
        type=int,
        nargs="+",
        default=None,
        metavar="Z",
        help="Formula units per molecule species (same order as --molecule)",
    )
    pyx.add_argument(
        "--z",
        dest="z_values",
        type=int,
        nargs="+",
        default=None,
        help="Alias for stoichiometry; one value repeats for all molecules",
    )
    pyx.add_argument(
        "--dim",
        type=int,
        default=3,
        choices=(0, 1, 2, 3),
        help="Crystal dimensionality (0=cluster, 3=3D periodic)",
    )
    pyx.add_argument(
        "--spg",
        "--space-group",
        dest="space_group",
        type=int,
        default=14,
        help="International space-group number",
    )
    pyx.add_argument(
        "--factor",
        type=float,
        default=1.0,
        help="PyXtal volume factor passed to from_random",
    )
    parser.add_argument(
        "--target-density-g-cm3",
        type=float,
        default=None,
        metavar="RHO",
        help=(
            "Scale cell to this mass density (g/cm³). Literature presets use CIF ρ "
            "unless this is set. Liquid DCM ≈ 1.326; crystal DCM ≈ 1.972"
        ),
    )
    pyx.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible PyXtal trials",
    )
    pyx.add_argument(
        "--attempts",
        type=int,
        default=20,
        help="Maximum PyXtal from_random retries",
    )
    pyx.add_argument(
        "--no-resort",
        action="store_true",
        help="Keep PyXtal atom order in ASE export (to_ase resort=False)",
    )
    parser.add_argument(
        "--supercell",
        type=str,
        default=None,
        metavar="NX,NY,NZ",
        help="Supercell repeats (literature: auto from --min-box-side if omitted)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output path (.pdb, .xyz, .extxyz, .cif, or .npz)",
    )
    parser.add_argument(
        "--format",
        dest="out_format",
        default=None,
        help="ASE output format override (default: inferred from --output suffix)",
    )
    opt = parser.add_argument_group("ASE optimization (optional, PyXtal path)")
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
    if suffix in ("xyz", "extxyz", "cif", "json", "pdb"):
        return suffix
    if suffix == "npz":
        return None
    return "extxyz"


def _resolve_literature_args(args: argparse.Namespace) -> tuple[str, Path]:
    if args.literature is not None:
        spec = LITERATURE_CRYSTAL_PRESETS[args.literature]
        residue = str(spec["residue"])
        cif = Path(spec["cif"]())
    elif args.from_cif is not None:
        if not args.residue:
            print(
                "Error: --from-cif requires --residue (e.g. DCM, BENZ).",
                file=sys.stderr,
            )
            raise SystemExit(2)
        residue = args.residue.strip().upper()
        cif = Path(args.from_cif).expanduser().resolve()
        if not cif.is_file():
            print(f"Error: CIF not found: {cif}", file=sys.stderr)
            raise SystemExit(1)
    else:
        raise ValueError("literature args not set")

    if args.from_cif is not None and args.literature is not None:
        cif = Path(args.from_cif).expanduser().resolve()
    return residue, cif


def _run_literature_build(args: argparse.Namespace) -> int:
    residue, cif_path = _resolve_literature_args(args)
    reps = (
        parse_supercell_reps(args.supercell)
        if args.supercell is not None
        else None
    )
    out = Path(args.output).expanduser().resolve()

    if args.literature is not None and args.from_cif is None:
        result = build_literature_charmm_supercell(
            args.literature,
            supercell_reps=reps,
            min_box_side_a=float(args.min_box_side) if reps is None else None,
            monomer_pdb=args.monomer_pdb,
            pdb_out=out if out.suffix.lower() == ".pdb" else None,
            target_density_g_cm3=args.target_density_g_cm3,
        )
    else:
        result = build_charmm_literature_supercell(
            residue=residue,
            cif_path=cif_path,
            supercell_reps=reps,
            min_box_side_a=float(args.min_box_side) if reps is None else None,
            monomer_pdb=args.monomer_pdb,
            pdb_out=out if out.suffix.lower() == ".pdb" else None,
            target_density_g_cm3=args.target_density_g_cm3,
        )

    atoms = result.atoms
    a, b, c = result.cell_lengths_a
    alpha, beta, gamma = result.cell_angles_deg
    rx, ry, rz = result.supercell_reps
    print(
        f"Literature crystal: {result.residue} from {cif_path.name}; "
        f"supercell {rx}×{ry}×{rz}; {result.n_molecules} molecules; "
        f"ρ={result.density_g_cm3:.4f} g/cm³",
        flush=True,
    )
    print(
        f"Box: a={a:.3f} b={b:.3f} c={c:.3f} Å; "
        f"α={alpha:.1f} β={beta:.1f} γ={gamma:.1f}°",
        flush=True,
    )
    print(f"Monomer template: {result.monomer_pdb}", flush=True)

    if out.suffix.lower() == ".pdb":
        if result.pdb_path != out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(result.pdb_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Wrote {out}", flush=True)
    elif out.suffix.lower() == ".npz":
        atoms_to_reference_npz(atoms, out, label="literature_charmm_crystal")
        print(f"Wrote {out}", flush=True)
    else:
        write_ase_structure(atoms, out, format=_infer_format(out, args.out_format))
        print(f"Wrote {out}", flush=True)
        if out.suffix.lower() != ".pdb":
            print(f"CHARMM PDB: {result.pdb_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.literature is not None or args.from_cif is not None:
        return _run_literature_build(args)

    if not args.molecule:
        print(
            "Error: provide --literature / --from-cif or at least one -m/--molecule.",
            file=sys.stderr,
        )
        return 2

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

    if args.target_density_g_cm3 is not None:
        rho_before = crystal_mass_density_g_cm3(atoms)
        scale = scale_atoms_cell_to_density(atoms, float(args.target_density_g_cm3))
        rho_after = crystal_mass_density_g_cm3(atoms)
        print(
            f"Density scale: {rho_before:.4f} → {rho_after:.4f} g/cm³ "
            f"(target {float(args.target_density_g_cm3):.4f}, cell×{scale:.4f})",
            flush=True,
        )

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
