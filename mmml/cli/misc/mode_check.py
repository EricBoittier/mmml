"""CLI adapter for monomer / small-cluster mode checks (``mmml mode-check``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmml.mode_check import (
    HybridModeCheckSetup,
    ModeCheckConfig,
    build_psf_and_attach_hybrid,
    run_mode_check,
)
from mmml.mode_check.config import (
    DEFAULT_MONOMER_SEPARATION_A,
    FAR_MONOMER_SEPARATION_A,
)
from mmml.mode_check.cutoff_ladder import cutoff_region_stations
from mmml.mode_check.cutoff_sweep import run_cutoff_sweep
from mmml.mode_check.geometry import parse_composition_spec
from mmml.mode_check.pbc_fd import run_pbc_cluster_fd, write_fd_result


def _parse_checks(value: str) -> tuple[str, ...]:
    parts = tuple(p.strip() for p in value.split(",") if p.strip())
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one check name")
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml mode-check",
        description=(
            "Local force / vibrational diagnostics for monomers and small "
            "clusters (FD forces, X–H stretch scans, ASE vibrations, optional "
            "kick FFT), plus the PBC cluster FD check formerly in check_fd.py."
        ),
    )
    parser.add_argument(
        "--pbc-fd",
        action="store_true",
        help=(
            "Run the PBC residue-cluster analytic vs FD force check "
            "(legacy check_fd.py). Ignores vacuum local-mode flags."
        ),
    )
    parser.add_argument(
        "--composition",
        default=None,
        help="Residue composition for vacuum checks, e.g. TIP3:1 or TIP3:2",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="PhysNet / Spooky portable JSON or Orbax checkpoint ($MMML_CKPT / bundled)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for vacuum mode-check artifacts (required unless --pbc-fd)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mode_check/fd_force_check.json"),
        help="JSON path for --pbc-fd results",
    )
    parser.add_argument(
        "--xyz",
        type=Path,
        default=None,
        help="Optional geometry (XYZ/PDB); otherwise build from named monomers",
    )
    parser.add_argument(
        "--checks",
        type=_parse_checks,
        default=None,
        help=(
            "Comma-separated: minimize,fd,bond-scan,vibrations,kick "
            "(default: minimize,fd,bond-scan,vibrations; "
            "for --cutoff-sweep: fd,bond-scan so COM does not drift)"
        ),
    )
    parser.add_argument(
        "--include-mm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable hybrid MM (auto-disabled for single monomer; default: true)",
    )
    parser.add_argument(
        "--include-ml-dimer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable ML dimer term (default: on when n_monomers>=2)",
    )
    parser.add_argument("--mm-charge-mode", default="q0")
    parser.add_argument("--lr-solver", default="mic")
    parser.add_argument("--ml-switch-width", type=float, default=1.5)
    parser.add_argument("--mm-switch-on", type=float, default=6.0)
    parser.add_argument("--mm-switch-width", type=float, default=5.0)
    parser.add_argument(
        "--monomer-separation",
        type=float,
        default=None,
        help=(
            f"COM spacing between monomers in Å (default: "
            f"{FAR_MONOMER_SEPARATION_A:g} for n≥2, "
            f"{DEFAULT_MONOMER_SEPARATION_A:g} unused for monomers; "
            f"pass 2.8 only with an oriented --xyz)"
        ),
    )
    parser.add_argument(
        "--far",
        action="store_true",
        help=(
            f"Place monomers at {FAR_MONOMER_SEPARATION_A:g} Å COM separation "
            "(default for n≥2; beyond MM handoff for numerical / monomer-parity); "
            "conflicts with --monomer-separation / --cutoff-sweep"
        ),
    )
    parser.add_argument(
        "--cutoff-sweep",
        action="store_true",
        help=(
            "For n≥2: run mode-check at every COM station on the hybrid handoff "
            "ruler (pure ML, handoff mid, mm_switch_on, MM tail, beyond). "
            "Writes cutoff_sweep_summary.json. Conflicts with --far / "
            "--monomer-separation. Default checks: minimize,fd,bond-scan with "
            "per-monomer COM frozen."
        ),
    )
    parser.add_argument("--fd-atoms", type=int, default=3)
    parser.add_argument("--fd-dx", type=float, default=1e-3)
    parser.add_argument("--minimize-fmax", type=float, default=0.05)
    parser.add_argument("--minimize-steps", type=int, default=400)
    parser.add_argument(
        "--minimize-freeze-coms",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "During minimize, freeze each monomer COM (intramolecular relax only). "
            "Default: on for --cutoff-sweep, off otherwise."
        ),
    )
    parser.add_argument("--kick-steps", type=int, default=500)
    parser.add_argument("--kick-delta", type=float, default=0.03)
    # PBC FD cluster options (check_fd parity)
    parser.add_argument("--residue", default="MEOH", help="Residue for --pbc-fd")
    parser.add_argument("--n-molecules", type=int, default=10)
    parser.add_argument("--spacing", type=float, default=5.0)
    parser.add_argument("--min-com-start-distance", type=float, default=6.0)
    parser.add_argument(
        "--ml-cutoff",
        type=float,
        default=0.1,
        help="ML switch width for --pbc-fd (check_fd default: 0.1)",
    )
    parser.add_argument(
        "--pbc-mm-switch-on",
        type=float,
        default=7.0,
        help="MM switch-on for --pbc-fd (check_fd default: 7.0)",
    )
    parser.add_argument(
        "--pbc-mm-cutoff",
        type=float,
        default=5.0,
        help="MM switch width for --pbc-fd (check_fd default: 5.0)",
    )
    return parser


def _main_pbc_fd(args: argparse.Namespace) -> int:
    result = run_pbc_cluster_fd(
        checkpoint=args.checkpoint,
        residue=str(args.residue),
        n_molecules=int(args.n_molecules),
        spacing=float(args.spacing),
        min_com_start_distance=float(args.min_com_start_distance),
        ml_cutoff=float(args.ml_cutoff),
        mm_switch_on=float(args.pbc_mm_switch_on),
        mm_cutoff=float(args.pbc_mm_cutoff),
        fd_check_atoms=int(args.fd_atoms),
        fd_check_dx=float(args.fd_dx),
    )
    path = write_fd_result(result, Path(args.output))
    from mmml.utils.rich_report import print_colored_json

    print_colored_json(result)
    print(f"Wrote {path}")
    return 0


def _resolve_monomer_separation_A(
    args: argparse.Namespace,
    *,
    n_monomers: int,
) -> float:
    if args.far and args.monomer_separation is not None:
        raise SystemExit(
            "mmml mode-check: use only one of --far and --monomer-separation"
        )
    if args.far:
        return float(FAR_MONOMER_SEPARATION_A)
    if args.monomer_separation is not None:
        return float(args.monomer_separation)
    # Dimers default far: COM=2.8 Å with unoriented templates often clashes
    # and poisons OH stretch / vib numerics (see PhysNet tip3:2 mode-check).
    if int(n_monomers) >= 2:
        return float(FAR_MONOMER_SEPARATION_A)
    return float(DEFAULT_MONOMER_SEPARATION_A)


def _main_vacuum(args: argparse.Namespace) -> int:
    if not args.composition:
        build_parser().error("--composition is required for vacuum mode-check")
    if args.output_dir is None:
        build_parser().error("--output-dir is required for vacuum mode-check")
    if args.checkpoint is None:
        build_parser().error("--checkpoint is required for vacuum mode-check")

    composition = parse_composition_spec(args.composition)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    n_mol = sum(int(n) for _, n in composition)

    if args.cutoff_sweep:
        if n_mol < 2:
            build_parser().error("--cutoff-sweep requires at least two monomers")
        if args.far or args.monomer_separation is not None:
            build_parser().error(
                "--cutoff-sweep conflicts with --far / --monomer-separation"
            )
        if args.xyz is not None:
            build_parser().error(
                "--cutoff-sweep repositions monomers; omit --xyz "
                "(or use a single-distance run)"
            )
        checks = (
            tuple(args.checks)
            if args.checks is not None
            else ("minimize", "fd", "bond-scan")
        )
        freeze_coms = (
            True if args.minimize_freeze_coms is None else bool(args.minimize_freeze_coms)
        )
        setup = HybridModeCheckSetup(
            composition=tuple((str(r), int(n)) for r, n in composition),
            checkpoint=Path(args.checkpoint),
            do_mm=bool(args.include_mm),
            do_ml=True,
            do_ml_dimer=args.include_ml_dimer,
            ml_switch_width=float(args.ml_switch_width),
            mm_switch_on=float(args.mm_switch_on),
            mm_switch_width=float(args.mm_switch_width),
            mm_charge_mode=str(args.mm_charge_mode),
            lr_solver=str(args.lr_solver),
            monomer_separation_A=float(FAR_MONOMER_SEPARATION_A),
        )
        cfg = ModeCheckConfig(
            checks=checks,  # type: ignore[arg-type]
            fd_atoms=int(args.fd_atoms),
            fd_dx_A=float(args.fd_dx),
            minimize_fmax=float(args.minimize_fmax),
            minimize_steps=int(args.minimize_steps),
            minimize_freeze_monomer_coms=freeze_coms,
            kick_steps=int(args.kick_steps),
            kick_delta_A=float(args.kick_delta),
        )
        stations = cutoff_region_stations(
            ml_switch_width=float(args.ml_switch_width),
            mm_switch_on=float(args.mm_switch_on),
            mm_switch_width=float(args.mm_switch_width),
        )
        payload = run_cutoff_sweep(
            setup,
            output_dir=out,
            config=cfg,
            stations=stations,
        )
        # Compact stdout table.
        print(
            json.dumps(
                {
                    "summary": payload.get("summary"),
                    "boundaries": payload.get("boundaries"),
                    "stations": [
                        {
                            "label": r.get("label"),
                            "region": r.get("region"),
                            "com_A": r.get("com_A_requested"),
                            "min_atom_A": r.get("min_intermolecular_distance_A"),
                            "energy_eV": r.get("energy_eV"),
                            "max_force_eVA": r.get("max_force_eVA"),
                            "fd_max_abs": (r.get("fd") or {}).get(
                                "fd_force_max_abs_diff_eVA"
                            ),
                            "bond_nu_cm_from_E": r.get("bond_nu_cm_from_E"),
                            "skipped": r.get("skipped"),
                            "errors": r.get("errors"),
                        }
                        for r in payload.get("results", [])
                    ],
                    "ok": payload.get("ok"),
                },
                indent=2,
            )
        )
        return 0 if payload.get("ok") else 1

    sep = _resolve_monomer_separation_A(args, n_monomers=n_mol)

    setup = HybridModeCheckSetup(
        composition=tuple((str(r), int(n)) for r, n in composition),
        checkpoint=Path(args.checkpoint),
        do_mm=bool(args.include_mm),
        do_ml=True,
        do_ml_dimer=args.include_ml_dimer,
        ml_switch_width=float(args.ml_switch_width),
        mm_switch_on=float(args.mm_switch_on),
        mm_switch_width=float(args.mm_switch_width),
        mm_charge_mode=str(args.mm_charge_mode),
        lr_solver=str(args.lr_solver),
        monomer_separation_A=sep,
        xyz=Path(args.xyz) if args.xyz is not None else None,
    )
    atoms, apm, meta = build_psf_and_attach_hybrid(
        setup,
        write_psf_to=out / "cluster.psf",
    )
    # Persist the exact geometry used for numerical / position audits.
    from ase.io import write as ase_write

    ase_write(str(out / "cluster_initial.xyz"), atoms)
    checks = (
        tuple(args.checks)
        if args.checks is not None
        else ("minimize", "fd", "bond-scan", "vibrations")
    )
    freeze_coms = (
        False if args.minimize_freeze_coms is None else bool(args.minimize_freeze_coms)
    )
    cfg = ModeCheckConfig(
        checks=checks,  # type: ignore[arg-type]
        fd_atoms=int(args.fd_atoms),
        fd_dx_A=float(args.fd_dx),
        minimize_fmax=float(args.minimize_fmax),
        minimize_steps=int(args.minimize_steps),
        minimize_freeze_monomer_coms=freeze_coms,
        kick_steps=int(args.kick_steps),
        kick_delta_A=float(args.kick_delta),
        atoms_per_monomer=tuple(apm),
    )
    result = run_mode_check(
        atoms,
        cfg,
        output_dir=out,
        setup_meta=meta,
    )
    # Write post-minimize geometry when minimize ran (atoms mutated in place).
    ase_write(str(out / "cluster_final.xyz"), atoms)

    print(
        json.dumps(
            {
                "summary": str(out / "mode_check_summary.json"),
                "monomer_separation_A": meta.get("monomer_separation_A", sep),
                "com_separations_A": meta.get("com_separations_A"),
                "min_intermolecular_distance_A": meta.get(
                    "min_intermolecular_distance_A"
                ),
                "energy_eV": result.energy_eV,
                "max_force_eVA": result.max_force_eVA,
                "do_mm_effective": meta.get("do_mm_effective"),
                "fd": result.fd,
                "bond_nu_cm_from_E": {
                    k: v.get("nu_cm_from_E") for k, v in result.bond_scans.items()
                },
                "vib_max_cm": (result.vibrations or {}).get("max_cm"),
                "kick_fft_peak_cm": (result.kick or {}).get("fft_peak_cm"),
                "errors": result.errors,
            },
            indent=2,
        )
    )
    return 1 if result.errors else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pbc_fd:
        return _main_pbc_fd(args)
    return _main_vacuum(args)


if __name__ == "__main__":
    raise SystemExit(main())
