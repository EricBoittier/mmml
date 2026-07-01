"""``mmml health-check`` — validate MMML / PyCHARMM / JAX interface health."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

DEFAULT_CHECKS: tuple[str, ...] = (
    "core",
    "jax",
    "charmm",
    "mlpot",
    "packmol",
    "checkpoint",
    "mpi",
)

ALL_CHECKS: tuple[str, ...] = (*DEFAULT_CHECKS, "live")


@dataclass
class InterfaceCheck:
    name: str
    ok: bool
    summary: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HealthReport:
    ok: bool
    checks: list[InterfaceCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [c.to_dict() for c in self.checks]}


def _resolve_checks(
    *,
    only: Sequence[str] | None,
    skip: Sequence[str] | None,
    live: bool,
) -> list[str]:
    if only:
        names = [str(x).strip().lower() for x in only if str(x).strip()]
        unknown = sorted(set(names) - set(ALL_CHECKS))
        if unknown:
            raise ValueError(f"Unknown check(s): {', '.join(unknown)}")
    else:
        names = list(DEFAULT_CHECKS)
        if live and "live" not in names:
            names.append("live")
    if skip:
        skip_set = {str(x).strip().lower() for x in skip if str(x).strip()}
        names = [n for n in names if n not in skip_set]
    if live and "live" not in names:
        names.append("live")
    return names


def check_core() -> InterfaceCheck:
    check = InterfaceCheck(name="core", ok=True, summary="core Python imports")
    for mod in ("numpy", "jax", "e3x", "ase"):
        try:
            __import__(mod)
            check.details[f"import_{mod}"] = "ok"
        except Exception as exc:
            check.ok = False
            check.errors.append(f"import {mod}: {exc}")
    if check.ok:
        check.summary = "core imports (numpy, jax, e3x, ase)"
    return check


def check_jax(*, require_gpu: bool = False) -> InterfaceCheck:
    check = InterfaceCheck(name="jax", ok=True, summary="JAX runtime")
    try:
        import jax

        devices = [str(d) for d in jax.devices()]
        check.details["devices"] = devices
        check.details["default_backend"] = str(jax.default_backend())
        has_cuda = any("cuda" in d.lower() or "gpu" in d.lower() for d in devices)
        check.details["cuda_visible"] = has_cuda
        if require_gpu and not has_cuda:
            check.ok = False
            check.errors.append(
                "No JAX CUDA devices visible (load CUDA/cuDNN modules or uv sync --extra gpu)"
            )
    except Exception as exc:
        check.ok = False
        check.errors.append(f"JAX unavailable: {exc}")
        return check

    try:
        from mmml.utils.jax_gpu_warmup import (
            ensure_jax_cuda_runtime_libs,
            jax_cuda_runtime_libs_warning,
        )

        bundled = ensure_jax_cuda_runtime_libs(quiet=True)
        check.details["pip_cuda_runtime_libs"] = bool(bundled)
        warning = jax_cuda_runtime_libs_warning(prefix="health-check")
        if warning:
            check.warnings.append(warning)
    except Exception as exc:
        check.warnings.append(f"CUDA runtime lib probe failed: {exc}")

    if check.ok and check.details.get("cuda_visible"):
        check.summary = f"JAX CUDA ({', '.join(check.details['devices'])})"
    elif check.ok:
        check.summary = f"JAX CPU ({', '.join(check.details.get('devices', []))})"
    return check


def check_charmm() -> InterfaceCheck:
    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        charmm_lib_available,
        charmm_lib_links_mpi,
        _charmm_lib_path,
    )

    check = InterfaceCheck(name="charmm", ok=True, summary="PyCHARMM / libcharmm")
    lib_dir = (os.environ.get("CHARMM_LIB_DIR") or "").strip() or None
    check.details["CHARMM_LIB_DIR"] = lib_dir
    check.details["CHARMM_HOME"] = (os.environ.get("CHARMM_HOME") or "").strip() or None

    lib = _charmm_lib_path()
    if lib is None:
        check.ok = False
        check.errors.append("libcharmm.so not found under CHARMM_LIB_DIR")
        return check

    check.details["libcharmm"] = str(lib)
    check.details["mpi_linked"] = bool(charmm_lib_links_mpi())
    if not charmm_lib_available():
        check.ok = False
        check.errors.append("charmm_lib_available() returned False")
        return check

    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401

        check.details["import_pycharmm"] = "ok"
    except Exception as exc:
        check.ok = False
        check.errors.append(f"import_pycharmm: {exc}")
        return check

    if check.details["mpi_linked"]:
        check.warnings.append(
            "MPI-linked libcharmm — launch MLpot via ./scripts/mmml-charmm-mpirun.sh"
        )
    check.summary = f"libcharmm @ {lib}"
    return check


def check_mlpot_symbols() -> InterfaceCheck:
    check = InterfaceCheck(name="mlpot", ok=True, summary="MLpot libcharmm symbols")
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import check_mlpot_symbols

        missing = check_mlpot_symbols()
        if missing:
            check.ok = False
            check.errors.append(f"missing symbols: {', '.join(missing)}")
        else:
            check.details["symbols"] = [
                "mlpot_set_func",
                "mlpot_set_properties",
                "mlpot_unset",
            ]
            check.summary = "MLpot hooks present in libcharmm"
    except Exception as exc:
        check.ok = False
        check.errors.append(str(exc))
    return check


def check_packmol() -> InterfaceCheck:
    check = InterfaceCheck(name="packmol", ok=True, summary="Packmol executable")
    try:
        from mmml.interfaces.pycharmmInterface.packmol_placement import packmol_executable

        exe = Path(packmol_executable())
        check.details["path"] = str(exe)
        if not exe.is_file():
            check.ok = False
            check.errors.append(f"Packmol not found: {exe}")
        elif not os.access(exe, os.X_OK):
            check.ok = False
            check.errors.append(f"Packmol not executable: {exe}")
        else:
            check.summary = f"Packmol @ {exe}"
    except Exception as exc:
        check.ok = False
        check.errors.append(str(exc))
    return check


def check_checkpoint(path: Path | None) -> InterfaceCheck:
    check = InterfaceCheck(name="checkpoint", ok=True, summary="PhysNet checkpoint")
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_checkpoint

        ckpt = resolve_checkpoint(path)
        check.details["path"] = str(ckpt)
        check.summary = f"checkpoint @ {ckpt}"
        if ckpt.suffix.lower() == ".json":
            import json as _json

            payload = _json.loads(ckpt.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                check.warnings.append("checkpoint JSON root is not an object")
            else:
                check.details["json_keys"] = sorted(payload.keys())[:12]
    except FileNotFoundError as exc:
        check.ok = False
        check.errors.append(str(exc))
        check.warnings.append("Set MMML_CKPT or pass --checkpoint for MLpot runs")
    except Exception as exc:
        check.ok = False
        check.errors.append(f"checkpoint load: {exc}")
    return check


def check_mpi(*, strict: bool = False, tier2: bool = False, prelaunch: bool = False) -> InterfaceCheck:
    from mmml.cli.run.mpi_check import run_mpi_check

    mpi_report = run_mpi_check(strict=strict, prelaunch=prelaunch)
    check = InterfaceCheck(
        name="mpi",
        ok=mpi_report.ok,
        summary="OpenMPI / mpi4py / mpirun",
        warnings=list(mpi_report.warnings),
        errors=list(mpi_report.errors),
        details={
            "mpirun": mpi_report.mpirun_path,
            "mpi_linked": mpi_report.charmm_links_mpi,
            "under_mpirun": mpi_report.under_mpirun,
            "rank": mpi_report.mpi_rank,
            "size": mpi_report.mpi_size,
            "recommended_launch": mpi_report.recommended_launch,
        },
    )
    if tier2:
        from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_validate import (
            validate_tier2_spatial_mpi_env,
        )

        tier2_report = validate_tier2_spatial_mpi_env(
            strict=strict,
            prelaunch=prelaunch,
        )
        check.details["tier2"] = tier2_report.to_dict()
        if not tier2_report.ok:
            check.ok = False
            check.errors.extend(tier2_report.errors)
            check.warnings.extend(tier2_report.warnings)
    if check.ok:
        check.summary = f"MPI env OK (mpirun={mpi_report.mpirun_path or 'n/a'})"
    return check


def check_live_mlpot(*, checkpoint: Path | None, residue: str, n_molecules: int) -> InterfaceCheck:
    check = InterfaceCheck(
        name="live",
        ok=True,
        summary=f"MLpot ENER smoke ({residue}:{n_molecules})",
    )
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            all_atom_selection,
            build_ase_cluster,
            charmm_energy_row,
            check_mlpot_symbols,
            load_physnet_for_cluster,
            resolve_checkpoint,
            setup_charmm_nbonds,
        )

        missing = check_mlpot_symbols()
        if missing:
            check.ok = False
            check.errors.append(f"missing MLpot symbols: {', '.join(missing)}")
            return check

        ckpt = resolve_checkpoint(checkpoint)
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        import ase
        import pycharmm
        import pycharmm.energy as energy

        z, r = build_ase_cluster(residue, n_molecules, spacing=5.0)
        n_atoms = len(z)
        setup_charmm_nbonds()
        params, model = load_physnet_for_cluster(ckpt, n_atoms)
        model.natoms = n_atoms
        atoms = ase.Atoms(numbers=z, positions=r)
        from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_pyc

        pyCModel = get_pyc(params, model, atoms)
        mlpot = pycharmm.MLpot(
            ml_model=pyCModel,
            ml_Z=list(z),
            ml_selection=all_atom_selection(),
            ml_charge=0,
            ml_fq=True,
        )
        if not mlpot.is_set:
            check.ok = False
            check.errors.append("MLpot.is_set is False after registration")
            return check
        energy.show()
        terms = charmm_energy_row()
        check.details["n_atoms"] = n_atoms
        check.details["checkpoint"] = str(ckpt)
        check.details["energy_terms"] = {k: round(v, 4) for k, v in terms.items()}
        mlpot.unset_mlpot()
        check.summary = f"MLpot ENER OK ({residue}:{n_molecules}, {n_atoms} atoms)"
    except Exception as exc:
        check.ok = False
        check.errors.append(str(exc))
    return check


def run_health_check(
    *,
    only: Sequence[str] | None = None,
    skip: Sequence[str] | None = None,
    live: bool = False,
    checkpoint: Path | None = None,
    strict: bool = False,
    require_gpu: bool = False,
    tier2: bool = False,
    prelaunch: bool = False,
    live_residue: str = "DCM",
    live_n_molecules: int = 2,
) -> HealthReport:
    names = _resolve_checks(only=only, skip=skip, live=live)
    checks: list[InterfaceCheck] = []
    for name in names:
        if name == "core":
            checks.append(check_core())
        elif name == "jax":
            checks.append(check_jax(require_gpu=require_gpu))
        elif name == "charmm":
            checks.append(check_charmm())
        elif name == "mlpot":
            checks.append(check_mlpot_symbols())
        elif name == "packmol":
            checks.append(check_packmol())
        elif name == "checkpoint":
            checks.append(check_checkpoint(checkpoint))
        elif name == "mpi":
            checks.append(
                check_mpi(strict=strict, tier2=tier2, prelaunch=prelaunch)
            )
        elif name == "live":
            checks.append(
                check_live_mlpot(
                    checkpoint=checkpoint,
                    residue=live_residue,
                    n_molecules=live_n_molecules,
                )
            )
    ok = all(c.ok for c in checks)
    if strict:
        for c in checks:
            if c.warnings:
                ok = False
    return HealthReport(ok=ok, checks=checks)


def render_health_report(report: HealthReport) -> str:
    lines = [
        "MMML interface health check",
        "===========================",
        f"Status: {'OK' if report.ok else 'FAIL'}",
        "",
    ]
    for check in report.checks:
        mark = "OK" if check.ok else "FAIL"
        lines.append(f"[{mark}] {check.name}: {check.summary}")
        for key, value in check.details.items():
            if key == "energy_terms":
                lines.append(f"      energy: {value}")
            elif key in ("devices", "symbols", "json_keys"):
                lines.append(f"      {key}: {value}")
            elif key not in ("tier2",):
                lines.append(f"      {key}: {value}")
        for w in check.warnings:
            lines.append(f"      warn: {w}")
        for e in check.errors:
            lines.append(f"      error: {e}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml health-check",
        description=(
            "Validate MMML interface health before PyCHARMM / MLpot jobs: "
            "imports, JAX devices, libcharmm, MLpot symbols, Packmol, checkpoint, MPI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast preflight on a GPU node (no CHARMM energy eval):
  mmml health-check --require-gpu

  # Under the MPI launcher (recommended on MPI-linked libcharmm):
  MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh health-check --require-gpu --strict

  # Include live MLpot registration + CHARMM ENER on DCM:2:
  MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh health-check --live --checkpoint "$MMML_CKPT"

  # Smallest liquid-DCM density smoke (after modules + CHARMM build):
  mmml liquid-box --composition DCM:20 --target-density-g-cm3 1.326 \\
    --profile standard -o boxes/dcm20 --charmm-sd-steps 50 --charmm-abnr-steps 50
  MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh md-system \\
    --from-psf boxes/dcm20/model.psf --from-crd boxes/dcm20/model.crd \\
    --checkpoint "$MMML_CKPT" --md-stages mini --mini-nstep 20 --no-echeck --quiet

Checks: core, jax, charmm, mlpot, packmol, checkpoint, mpi (+ live with --live).
        """,
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="CHECK",
        help=f"Run subset of: {', '.join(ALL_CHECKS)}",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="CHECK",
        help="Skip checks from the default set.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live MLpot registration + CHARMM energy (implies charmm + checkpoint).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="PhysNet checkpoint (default: MMML_CKPT).",
    )
    parser.add_argument(
        "--live-residue",
        default="DCM",
        help="Residue for --live smoke (default: DCM).",
    )
    parser.add_argument(
        "--live-n-molecules",
        type=int,
        default=2,
        help="Monomer count for --live smoke (default: 2).",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if JAX does not see a CUDA device.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors.",
    )
    parser.add_argument(
        "--prelaunch",
        action="store_true",
        help="Relax MPI prelaunch warnings (serial health-check before mpirun).",
    )
    parser.add_argument(
        "--tier2",
        action="store_true",
        help="Also run spatial-MPI GPU checks inside the mpi section.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        maybe_rerun_mmml_under_mpirun,
        prepare_serial_charmm_mpi_env,
    )

    prepare_serial_charmm_mpi_env()
    parsed_argv = list(argv) if argv is not None else sys.argv[1:]
    rerun_code = maybe_rerun_mmml_under_mpirun(parsed_argv, subcommand="health-check")
    if rerun_code is not None:
        return int(rerun_code)

    args = build_parser().parse_args(parsed_argv)
    try:
        report = run_health_check(
            only=args.only,
            skip=args.skip,
            live=bool(args.live),
            checkpoint=args.checkpoint,
            strict=bool(args.strict),
            require_gpu=bool(args.require_gpu),
            tier2=bool(args.tier2),
            prelaunch=bool(args.prelaunch),
            live_residue=str(args.live_residue).strip().upper(),
            live_n_molecules=int(args.live_n_molecules),
        )
    except ValueError as exc:
        print(f"health-check: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_health_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
