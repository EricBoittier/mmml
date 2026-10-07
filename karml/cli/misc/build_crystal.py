"""Build symmetry-aware crystal structures with PyXtal and optional ASE relaxation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mmml.interfaces.aseInterface.pyxtal_optimize import optimize_ase_atoms
from mmml.interfaces.crystal_charmm import (
    DEFAULT_MIN_BOX_SIDE_A,
    LITERATURE_CRYSTAL_PRESETS,
    build_charmm_literature_supercell,
    build_literature_charmm_supercell,
    map_ase_crystal_to_charmm_pdb,
    suggest_supercell_reps,
    write_crystal_charmm_topology,
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
        "Literature CIF + make-res (recommended for DCM / benzene / acetone)"
    )
    lit.add_argument(
        "--literature",
        choices=sorted(LITERATURE_CRYSTAL_PRESETS),
        default=None,
        metavar="PRESET",
        help=(
            "Bundled experimental CIF preset: dcm / dcm133 (DCM Pbcn at 1.63 "
            "and 1.33 GPa), benz (P2₁/c), "
            "aco / aco5k / aco110k (acetone Pbca at 150, 5, 110 K), "
            "acocmcm (metastable acetone Cmcm)"
        ),
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
        help="CHARMM residue (DCM, BENZ) for --from-cif / --write-charmm on PyXtal path",
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
        default=DEFAULT_MIN_BOX_SIDE_A,
        metavar="ANG",
        help="Minimum supercell edge length (Å); default ≈2× CHARMM cutnb",
    )
    md = parser.add_argument_group("MD box sizing / CHARMM handoff")
    md.add_argument(
        "--box-size",
        "--side-length",
        dest="box_size",
        type=float,
        default=None,
        metavar="ANG",
        help=(
            "Cubic MD cell side length (Å). When --supercell is omitted, also drives "
            "auto tiling so each crystal edge ≥ this value. Used as CHARMM IMAGE side "
            "with --write-charmm."
        ),
    )
    md.add_argument(
        "--write-charmm",
        action="store_true",
        help=(
            "Write {stem}.pdb/.psf/.crd and {stem}_box.json via PyCHARMM GENERATE "
            "(cubic IMAGE). Prefer --literature for DCM/benzene."
        ),
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
        help=(
            "Supercell repeats (literature: auto from --box-size / --min-box-side "
            "if omitted; PyXtal: auto from --box-size if omitted)"
        ),
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


def effective_min_box_side_a(args: argparse.Namespace) -> float:
    """Supercell edge target: ``--box-size`` overrides ``--min-box-side``."""
    if args.box_size is not None:
        side = float(args.box_size)
        if side <= 0.0:
            raise ValueError(f"--box-size must be positive, got {side}")
        return side
    side = float(args.min_box_side)
    if side <= 0.0:
        raise ValueError(f"--min-box-side must be positive, got {side}")
    return side


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


def _charmm_side_length_a(
    args: argparse.Namespace,
    cell_lengths_a: tuple[float, float, float],
) -> float:
    """Cubic CHARMM IMAGE side; ``--box-size`` or max crystal edge."""
    max_edge = max(float(x) for x in cell_lengths_a)
    if args.box_size is not None:
        side = float(args.box_size)
        if side + 1e-6 < max_edge:
            raise ValueError(
                f"--box-size {side:.3f} Å is smaller than the largest supercell "
                f"edge ({max_edge:.3f} Å). Increase --box-size or reduce --supercell."
            )
        return side
    return max_edge


def _maybe_write_charmm(
    args: argparse.Namespace,
    *,
    charmm_pdb: Path,
    out_stem: Path,
    cell_lengths_a: tuple[float, float, float],
    n_molecules: int | None,
    residue: str | None,
) -> int:
    if not args.write_charmm:
        return 0
    try:
        side = _charmm_side_length_a(args, cell_lengths_a)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    try:
        paths = write_crystal_charmm_topology(
            charmm_pdb,
            out_stem,
            side_length_A=side,
            n_molecules=n_molecules,
            residue=residue,
        )
    except ModuleNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — surface CHARMM failures cleanly
        print(f"Error writing CHARMM topology: {exc}", file=sys.stderr)
        return 1
    print(
        f"CHARMM handoff: psf={paths.psf} crd={paths.crd} box.json={paths.box_json}",
        flush=True,
    )
    return 0


def _run_literature_build(args: argparse.Namespace) -> int:
    residue, cif_path = _resolve_literature_args(args)
    reps = (
        parse_supercell_reps(args.supercell)
        if args.supercell is not None
        else None
    )
    out = Path(args.output).expanduser().resolve()
    min_side = effective_min_box_side_a(args)

    if args.literature is not None and args.from_cif is None:
        result = build_literature_charmm_supercell(
            args.literature,
            supercell_reps=reps,
            min_box_side_a=min_side if reps is None else None,
            monomer_pdb=args.monomer_pdb,
            pdb_out=out if out.suffix.lower() == ".pdb" else None,
            target_density_g_cm3=args.target_density_g_cm3,
        )
    else:
        result = build_charmm_literature_supercell(
            residue=residue,
            cif_path=cif_path,
            supercell_reps=reps,
            min_box_side_a=min_side if reps is None else None,
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

    return _maybe_write_charmm(
        args,
        charmm_pdb=result.pdb_path,
        out_stem=out,
        cell_lengths_a=result.cell_lengths_a,
        n_molecules=result.n_molecules,
        residue=result.residue,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.box_size is not None and float(args.box_size) <= 0.0:
        print(f"Error: --box-size must be positive, got {args.box_size}", file=sys.stderr)
        return 2

    if args.literature is not None or args.from_cif is not None:
        return _run_literature_build(args)

    if not args.molecule:
        print(
            "Error: provide --literature / --from-cif or at least one -m/--molecule.",
            file=sys.stderr,
        )
        return 2

    if args.write_charmm and not args.residue:
        print(
            "Error: --write-charmm on the PyXtal path requires --residue "
            "(e.g. BENZ). Prefer --literature benz for MD-ready CHARMM files.",
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
    elif args.box_size is not None:
        lengths = tuple(float(x) for x in atoms.cell.cellpar()[:3])
        reps = suggest_supercell_reps(lengths, min_box_side_a=float(args.box_size))
        if reps != (1, 1, 1):
            atoms = ase_supercell(atoms, reps)
            print(
                f"Supercell {reps[0]}×{reps[1]}×{reps[2]} "
                f"(from --box-size {float(args.box_size):.3f} Å) → natoms={len(atoms)}",
                flush=True,
            )
        else:
            print(
                f"Unit cell already ≥ --box-size {float(args.box_size):.3f} Å "
                f"(edges {[round(x, 3) for x in lengths]})",
                flush=True,
            )

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

    if not args.write_charmm:
        return 0

    residue = str(args.residue).strip().upper()
    charmm_pdb = out.with_name(f"{out.stem}_charmm.pdb")
    try:
        map_ase_crystal_to_charmm_pdb(
            atoms,
            residue=residue,
            monomer_pdb=args.monomer_pdb,
            pdb_out=charmm_pdb,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error mapping PyXtal atoms to CHARMM PDB: {exc}", file=sys.stderr)
        return 1
    print(f"CHARMM PDB: {charmm_pdb}", flush=True)
    par = tuple(float(x) for x in atoms.cell.cellpar())
    cell_lengths = (par[0], par[1], par[2])
    # Molecule count from residue template size when available.
    n_mol = None
    try:
        from mmml.interfaces.crystal_charmm import (
            load_monomer_template,
            resolve_make_res_monomer_pdb,
        )

        tmpl = load_monomer_template(
            resolve_make_res_monomer_pdb(residue, monomer_pdb=args.monomer_pdb)
        )
        n_per = int(tmpl[2].shape[0])
        if n_per > 0 and len(atoms) % n_per == 0:
            n_mol = len(atoms) // n_per
    except Exception:  # noqa: BLE001
        n_mol = None
    return _maybe_write_charmm(
        args,
        charmm_pdb=charmm_pdb,
        out_stem=out,
        cell_lengths_a=cell_lengths,
        n_molecules=n_mol,
        residue=residue,
    )


if __name__ == "__main__":
    raise SystemExit(main())
