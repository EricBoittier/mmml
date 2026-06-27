"""``mmml mpi-check`` — validate OpenMPI / CHARMM / mpi4py environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MpiCheckReport:
    ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    charmm_lib_dir: str | None = None
    charmm_lib: str | None = None
    charmm_links_mpi: bool = False
    mpirun_path: str | None = None
    under_mpirun: bool = False
    mpi_rank: int = 0
    mpi_size: int = 1
    mpi4py_available: bool = False
    mpi4py_initialized: bool = False
    jax_device: str | None = None
    omp_num_threads: str | None = None
    recommended_launch: str | None = None
    spatial_mpi_env: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_mpi_check(*, strict: bool = False) -> MpiCheckReport:
    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        _under_mpirun,
        charmm_lib_available,
        charmm_lib_links_mpi,
        charmm_mpirun_path,
        mpirun_launch_hint,
        _charmm_lib_path,
        _mpi4py_available,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size
    from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
        spatial_mpi_enabled,
    )

    report = MpiCheckReport(ok=True)
    report.charmm_lib_dir = (os.environ.get("CHARMM_LIB_DIR") or "").strip() or None
    lib = _charmm_lib_path()
    report.charmm_lib = str(lib) if lib is not None else None
    report.charmm_links_mpi = bool(charmm_lib_links_mpi())
    report.under_mpirun = bool(_under_mpirun())
    report.mpi_rank, report.mpi_size = mpi_rank_size()
    report.mpi4py_available = bool(_mpi4py_available())
    report.omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    report.spatial_mpi_env = bool(spatial_mpi_enabled())

    mpirun = charmm_mpirun_path()
    report.mpirun_path = str(mpirun) if mpirun is not None else None

    if not charmm_lib_available():
        report.errors.append("CHARMM_LIB_DIR does not contain libcharmm.so")
        report.ok = False
    elif report.charmm_links_mpi and mpirun is None:
        report.errors.append(
            "MPI-linked libcharmm.so but no matching mpirun found "
            "(set MMML_MPIRUN or OPENMPI_ROOT)"
        )
        report.ok = False

    if report.charmm_links_mpi and not report.under_mpirun:
        report.warnings.append(
            "Not under mpirun; use ./scripts/mmml-charmm-mpirun.sh for MLpot jobs"
        )

    if report.charmm_links_mpi and report.mpi4py_available is False:
        msg = "mpi4py not installed (optional for mpirun launch; needed for spatial gather tests)"
        if strict:
            report.errors.append(msg)
            report.ok = False
        else:
            report.warnings.append(msg)

    if report.mpi4py_available:
        try:
            from mpi4py import MPI

            report.mpi4py_initialized = bool(MPI.Is_initialized())
        except Exception as exc:
            report.warnings.append(f"mpi4py import issue: {exc}")

    try:
        import jax

        devs = [str(d) for d in jax.devices()]
        report.jax_device = ", ".join(devs) if devs else "none"
    except Exception as exc:
        report.warnings.append(f"JAX not available: {exc}")

    if report.charmm_links_mpi:
        report.recommended_launch = (
            "MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh md-system ..."
        )
        if report.spatial_mpi_env:
            report.recommended_launch = (
                "MMML_MPI_NP=4 MMML_MLPOT_SPATIAL_MPI=1 "
                "./scripts/mmml-charmm-mpirun.sh md-system --ml-spatial-mpi ..."
            )
    else:
        report.recommended_launch = "mmml md-system ...  # serial libcharmm"

    if strict and report.warnings:
        report.ok = False

    return report


def render_mpi_check_report(report: MpiCheckReport) -> str:
    lines = [
        "MMML MPI environment check",
        "==========================",
        f"Status: {'OK' if report.ok else 'FAIL'}",
        f"CHARMM_LIB_DIR: {report.charmm_lib_dir or '(unset)'}",
        f"libcharmm: {report.charmm_lib or '(not found)'}",
        f"MPI-linked build: {report.charmm_links_mpi}",
        f"mpirun: {report.mpirun_path or '(not found)'}",
        f"Under mpirun: {report.under_mpirun} (rank {report.mpi_rank}/{report.mpi_size})",
        f"mpi4py: {'yes' if report.mpi4py_available else 'no'}"
        + (f", initialized={report.mpi4py_initialized}" if report.mpi4py_available else ""),
        f"JAX devices: {report.jax_device or '(unknown)'}",
        f"OMP_NUM_THREADS: {report.omp_num_threads or '(unset)'}",
        f"Spatial MPI env: {report.spatial_mpi_env}",
    ]
    if report.recommended_launch:
        lines.append(f"Recommended: {report.recommended_launch}")
    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in report.warnings)
    if report.errors:
        lines.append("")
        lines.append("Errors:")
        lines.extend(f"  - {e}" for e in report.errors)
    if report.charmm_links_mpi:
        lines.append("")
        lines.append("Full mpirun hint:")
        lines.append(
            __import__(
                "mmml.interfaces.pycharmmInterface.charmm_mpi",
                fromlist=["mpirun_launch_hint"],
            ).mpirun_launch_hint()
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml mpi-check",
        description="Validate OpenMPI / CHARMM / mpi4py setup for PyCHARMM MLpot runs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON on stdout.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (non-zero exit).",
    )
    parser.add_argument(
        "--tier2",
        action="store_true",
        help="Also validate Tier 2 spatial MPI + GPU environment for MLpot.",
    )
    parser.add_argument(
        "--tier3",
        action="store_true",
        help="Survey Tier 3 DOMDEC + MLpot blockers (informational; production blocked).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from mmml.interfaces.pycharmmInterface.charmm_mpi import prepare_serial_charmm_mpi_env

    prepare_serial_charmm_mpi_env()
    args = build_parser().parse_args(argv)
    report = run_mpi_check(strict=bool(args.strict))
    tier2_ok = True
    tier2_report = None
    tier3_ok = True
    tier3_report = None
    if args.tier2:
        from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_validate import (
            validate_tier2_spatial_mpi_env,
        )

        tier2_report = validate_tier2_spatial_mpi_env(strict=bool(args.strict))
        tier2_ok = tier2_report.ok

    if args.tier3:
        from mmml.interfaces.pycharmmInterface.mlpot.tier3_domdec_validate import (
            validate_tier3_domdec_env,
        )

        tier3_report = validate_tier3_domdec_env(strict=bool(args.strict))
        tier3_ok = tier3_report.ok

    if args.json:
        payload = report.to_dict()
        if tier2_report is not None:
            payload["tier2"] = tier2_report.to_dict()
        if tier3_report is not None:
            payload["tier3"] = tier3_report.to_dict()
        print(json.dumps(payload, indent=2))
    else:
        print(render_mpi_check_report(report))
        if tier2_report is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_validate import (
                render_tier2_report,
            )

            print()
            print(render_tier2_report(tier2_report))
        if tier3_report is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.tier3_domdec_validate import (
                render_tier3_report,
            )

            print()
            print(render_tier3_report(tier3_report))
    return 0 if report.ok and tier2_ok and tier3_ok else 1


if __name__ == "__main__":
    sys.exit(main())
