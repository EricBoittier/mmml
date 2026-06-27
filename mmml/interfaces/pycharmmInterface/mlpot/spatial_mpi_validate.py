"""Tier 2 spatial MPI + GPU environment validation for MLpot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tier2MlpotMpiReport:
    """Readiness report for ``np>1`` + ``--ml-spatial-mpi`` MLpot runs."""

    ok: bool
    tier: str = "tier2_spatial_mpi"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    mpi_size: int = 1
    mpi_rank: int = 0
    under_mpirun: bool = False
    spatial_mpi_enabled: bool = False
    jax_gpu_count: int = 0
    mlpot_device: str = "unknown"
    defer_jax_warmup: bool = False
    charmm_links_mpi: bool = False
    recommended_launch: str | None = None
    tested_with_live_mlpot: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tier": self.tier,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "mpi_size": self.mpi_size,
            "mpi_rank": self.mpi_rank,
            "under_mpirun": self.under_mpirun,
            "spatial_mpi_enabled": self.spatial_mpi_enabled,
            "jax_gpu_count": self.jax_gpu_count,
            "mlpot_device": self.mlpot_device,
            "defer_jax_warmup": self.defer_jax_warmup,
            "charmm_links_mpi": self.charmm_links_mpi,
            "recommended_launch": self.recommended_launch,
            "tested_with_live_mlpot": self.tested_with_live_mlpot,
        }


def validate_tier2_spatial_mpi_env(
    *,
    strict: bool = False,
    prelaunch: bool = False,
) -> Tier2MlpotMpiReport:
    """Check env for Tier 2 spatial ML MPI (does not run MLpot).

    With ``prelaunch=True``, serial checks before ``mmml-charmm-mpirun.sh`` do not
    fail ``strict`` on expected warnings (spatial env unset, not under mpirun, etc.).
    """
    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        _under_mpirun,
        charmm_lib_links_mpi,
        charmm_mpirun_path,
        defer_jax_warmup_until_after_mlpot_sd,
    )
    from mmml.interfaces.pycharmmInterface.jax_device_policy import (
        mlpot_jax_device_name,
        mlpot_local_gpu_count,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size
    from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
        spatial_mpi_enabled,
    )

    report = Tier2MlpotMpiReport(ok=True)
    report.charmm_links_mpi = bool(charmm_lib_links_mpi())
    report.under_mpirun = bool(_under_mpirun())
    report.mpi_rank, report.mpi_size = mpi_rank_size()
    report.spatial_mpi_enabled = bool(spatial_mpi_enabled())
    report.mlpot_device = mlpot_jax_device_name()
    report.jax_gpu_count = int(mlpot_local_gpu_count())
    report.defer_jax_warmup = bool(defer_jax_warmup_until_after_mlpot_sd())

    if not report.spatial_mpi_enabled:
        report.warnings.append(
            "MMML_MLPOT_SPATIAL_MPI is off; Tier 2 uses rank-0 MLpot bridge (slow at np>1)"
        )

    if not report.charmm_links_mpi:
        report.warnings.append("libcharmm.so is not MPI-linked; mpirun optional")

    if report.charmm_links_mpi and charmm_mpirun_path() is None:
        report.errors.append("MPI-linked CHARMM but no matching mpirun (set MMML_MPIRUN)")
        report.ok = False

    if report.spatial_mpi_enabled and report.mpi_size < 2:
        report.warnings.append(
            f"spatial MPI enabled but mpi_size={report.mpi_size}; use MMML_MPI_NP>=2"
        )

    if report.spatial_mpi_enabled and report.mlpot_device == "gpu" and report.jax_gpu_count == 0:
        report.errors.append(
            "spatial MPI + MMML_MLPOT_DEVICE=gpu but JAX sees no GPU (uv sync --extra gpu)"
        )
        report.ok = False

    if report.spatial_mpi_enabled and report.mpi_size > 1:
        ml_gpus = int(os.environ.get("MMML_MLPOT_N_GPUS", "1") or "1")
        if ml_gpus > 1:
            report.errors.append(
                f"MMML_MLPOT_N_GPUS={ml_gpus} with mpi_size={report.mpi_size} "
                "oversubscribes GPUs; use --ml-gpu-count 1 and MMML_MPI_PIN_GPU_PER_RANK=1"
            )
            report.ok = False
        if report.jax_gpu_count > 0 and report.jax_gpu_count < report.mpi_size:
            report.warnings.append(
                f"mpi_size={report.mpi_size} but only {report.jax_gpu_count} JAX GPU(s) visible; "
                "ranks may share devices unless SLURM sets CUDA_VISIBLE_DEVICES per task"
            )

    omp = os.environ.get("OMP_NUM_THREADS")
    if report.charmm_links_mpi and omp not in (None, "1"):
        report.warnings.append(
            f"OMP_NUM_THREADS={omp}; MPI-linked CHARMM expects 1 (see MMML_CHARMM_OMP_THREADS)"
        )

    if report.charmm_links_mpi and not report.defer_jax_warmup and report.mlpot_device == "gpu":
        if report.under_mpirun:
            report.warnings.append(
                "JAX GPU warmup may run before MLpot SD; prefer mmml-charmm-mpirun.sh "
                "(defer_jax_warmup default on MPI builds)"
            )
        else:
            report.warnings.append(
                "Defer JAX until after MLpot SD applies under mpirun (launcher sets this)"
            )

    if os.environ.get("MMML_MLPOT_RANK0_BRIDGE", "1").strip().lower() in ("0", "false"):
        if report.spatial_mpi_enabled and report.mpi_size > 1:
            report.warnings.append(
                "MMML_MLPOT_RANK0_BRIDGE=0 with spatial MPI: every rank runs full MLpot (debug only)"
            )
        elif report.mpi_size > 1:
            report.errors.append(
                "MMML_MLPOT_RANK0_BRIDGE=0 at np>1 without spatial MPI duplicates MLpot incorrectly"
            )
            report.ok = False

    report.recommended_launch = (
        "MMML_MPI_NP=4 MMML_MLPOT_SPATIAL_MPI=1 MMML_MPI_PIN_GPU_PER_RANK=1 "
        "./scripts/mmml-charmm-mpirun.sh md-system --ml-spatial-mpi --ml-gpu-count 1 ..."
    )

    if strict and report.warnings:
        if prelaunch:
            blocking = [w for w in report.warnings if not _prelaunch_ok_warning(w)]
            if blocking:
                report.ok = False
        else:
            report.ok = False

    return report


def _prelaunch_ok_warning(message: str) -> bool:
    """Warnings that are normal for serial ``mpi-check`` before ``mpirun`` launch."""
    ok_fragments = (
        "MMML_MLPOT_SPATIAL_MPI is off",
        "spatial MPI enabled but mpi_size=",
        "Defer JAX until after MLpot SD applies under mpirun",
    )
    return any(fragment in message for fragment in ok_fragments)


def render_tier2_report(report: Tier2MlpotMpiReport, *, prelaunch: bool = False) -> str:
    lines = [
        "MMML Tier 2 spatial MPI + MLpot check",
        "====================================",
        f"Status: {'OK' if report.ok else 'FAIL'}",
        f"Under mpirun: {report.under_mpirun} (rank {report.mpi_rank}/{report.mpi_size})",
        f"Spatial MPI: {report.spatial_mpi_enabled}",
        f"CHARMM MPI-linked: {report.charmm_links_mpi}",
        f"MLpot JAX device: {report.mlpot_device} ({report.jax_gpu_count} GPU(s))",
        f"Defer JAX until after MLpot SD: {report.defer_jax_warmup}",
    ]
    if report.recommended_launch:
        lines.append(f"Recommended: {report.recommended_launch}")
    if report.warnings:
        lines.extend(["", "Warnings:"] + [f"  - {w}" for w in report.warnings])
    if report.errors:
        lines.extend(["", "Errors:"] + [f"  - {e}" for e in report.errors])
    if prelaunch and not report.under_mpirun:
        lines.extend(
            [
                "",
                "Pre-launch note: serial mpi-check is OK. For strict Tier 2 under launch:",
                "  MMML_MPI_NP=2 MMML_MLPOT_SPATIAL_MPI=1 ./scripts/mmml-charmm-mpirun.sh mpi-check --tier2 --strict",
            ]
        )
    lines.extend(
        [
            "",
            "Live MLpot Tier 2 smoke (CHARMM node):",
            "  MMML_MPI_NP=2 MMML_MLPOT_SPATIAL_MPI=1 ./scripts/mmml-charmm-mpirun.sh python \\",
            "    tests/functionality/mlpot/06_spatial_mpi_tier2_smoke.py",
        ]
    )
    return "\n".join(lines)
