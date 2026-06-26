"""Periodic external MM nonbond mode (ScaFaCoS Coulomb + CHARMM IMAGE VDW).

``periodic_external`` turns off JAX real-space LJ and Coulomb in the MLpot
callback and instead uses:

* **Coulomb** — ScaFaCoS ``libfcs`` (full periodic k-space + real-space split)
* **Lennard-Jones** — CHARMM switched VDW with IMAGE neighbor lists (not ScaFaCoS)

ScaFaCoS does not implement LJ; CHARMM is the supported periodic VDW backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

MmNonbondMode = Literal["jax_mic", "periodic_external"]

# Conservative margin between cluster extent and cubic box edge (Å).
PERIODIC_MM_BOX_MARGIN_A = 5.0
# Minimum cubic box for periodic external mode when extent is unknown at validate time.
PERIODIC_MM_MIN_BOX_A = 20.0


@dataclass(frozen=True)
class PeriodicMmConfig:
    """Resolved periodic external MM settings."""

    lr_solver: str
    scafacos_method: str
    charmm_vdw: bool = True

    @property
    def uses_scafacos(self) -> bool:
        return self.lr_solver == "scafacos"


def resolve_mm_nonbond_mode(args: Any | None) -> MmNonbondMode:
    raw = "jax_mic"
    if args is not None:
        raw = str(getattr(args, "mm_nonbond_mode", "jax_mic") or "jax_mic").strip().lower()
    if raw in ("jax_mic", "mic", "jax"):
        return "jax_mic"
    if raw in ("periodic_external", "periodic", "scafacos"):
        return "periodic_external"
    raise ValueError(
        f"mm_nonbond_mode must be jax_mic or periodic_external; got {raw!r}"
    )


def resolve_lr_solver_arg(args: Any | None) -> str | None:
    if args is None:
        return os.environ.get("MMML_LR_SOLVER")
    explicit = getattr(args, "lr_solver", None)
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip().lower()
    return os.environ.get("MMML_LR_SOLVER")


def resolve_periodic_charmm_vdw(args: Any | None) -> bool:
    """Whether periodic_external keeps CHARMM IMAGE VDW on (default True)."""
    if args is None:
        return True
    return bool(getattr(args, "periodic_charmm_vdw", True))


def build_periodic_mm_config(args: Any | None) -> PeriodicMmConfig | None:
    if resolve_mm_nonbond_mode(args) != "periodic_external":
        return None
    from mmml.interfaces.pycharmmInterface.long_range_backend import pick_lr_solver

    lr = pick_lr_solver(resolve_lr_solver_arg(args))
    if lr not in ("scafacos",):
        raise ValueError(
            f"periodic_external requires a working ScaFaCoS install "
            f"(lr_solver resolved to {lr!r}; set --lr-solver scafacos and SCAFACOS_LIB)"
        )
    method = str(
        getattr(args, "scafacos_method", None)
        or os.environ.get("SCAFACOS_METHOD", "p2nfft")
    ).strip()
    return PeriodicMmConfig(
        lr_solver=lr,
        scafacos_method=method,
        charmm_vdw=resolve_periodic_charmm_vdw(args),
    )


def validate_periodic_mm_args(
    args: Any,
    *,
    charmm_pbc: bool,
    mlpot_pbc: bool,
    box_side_A: float | None,
) -> PeriodicMmConfig:
    """Validate CLI/stage prerequisites for periodic external MM."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_charmm_use_pbc,
        resolve_mlpot_use_pbc,
    )

    if resolve_mm_nonbond_mode(args) != "periodic_external":
        raise RuntimeError("validate_periodic_mm_args called outside periodic_external mode")

    if getattr(args, "free_space", False):
        raise ValueError(
            "periodic_external MM requires PBC (--setup pbc_*); "
            "--free-space is incompatible"
        )
    if not charmm_pbc and not resolve_charmm_use_pbc(args):
        raise ValueError(
            "periodic_external MM requires CHARMM PBC (crystal + IMAGE). "
            "Use --setup pbc_nve|pbc_nvt|pbc_npt and a cubic --box-size."
        )
    if not mlpot_pbc and not resolve_mlpot_use_pbc(args):
        raise ValueError(
            "periodic_external MM requires ML MIC PBC (--setup pbc_*). "
            "Loose box-only PBC (--box-size without pbc_*) is not supported."
        )
    if box_side_A is None or float(box_side_A) <= 0.0:
        raise ValueError(
            "periodic_external MM requires a positive cubic box side (Å). "
            "Pass --box-size or use a setup that defines the cell."
        )

    cfg = build_periodic_mm_config(args)
    assert cfg is not None

    from mmml.interfaces.scafacosInterface.scafacos_session import have_scafacos

    if cfg.uses_scafacos and not have_scafacos():
        raise ValueError(
            "periodic_external with lr_solver=scafacos requires libfcs on "
            "LD_LIBRARY_PATH or SCAFACOS_LIB. See mmml/interfaces/scafacosInterface/README.md"
        )
    return cfg


def min_cubic_box_for_periodic_mm(
    *,
    box_side_A: float,
    cluster_extent_A: float | None = None,
) -> float:
    """Lower bound on cubic L for periodic external MM (Å)."""
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        PBC_NBOND_BOX_MARGIN_A,
        pbc_nbond_cutoffs,
    )

    cut = pbc_nbond_cutoffs(float(box_side_A))
    half_box_min = cut.cutnb + float(PBC_NBOND_BOX_MARGIN_A)
    extent_need = (
        float(cluster_extent_A) + float(PERIODIC_MM_BOX_MARGIN_A)
        if cluster_extent_A is not None
        else float(PERIODIC_MM_MIN_BOX_A)
    )
    return max(2.0 * half_box_min, extent_need, float(PERIODIC_MM_MIN_BOX_A))


def assert_periodic_mm_box_side(
    box_side_A: float,
    *,
    cluster_extent_A: float | None = None,
) -> None:
    """Raise when the cubic box is too small for CHARMM IMAGE + ScaFaCoS."""
    L = float(box_side_A)
    need = min_cubic_box_for_periodic_mm(
        box_side_A=L,
        cluster_extent_A=cluster_extent_A,
    )
    if L + 1e-6 < need:
        raise ValueError(
            f"periodic_external MM: cubic box L={L:.2f} Å is too small "
            f"(need L >= {need:.2f} Å for CHARMM cutnb and cluster/image margin). "
            f"Increase --box-size or Packmol spacing."
        )


def cluster_extent_from_positions(positions: np.ndarray) -> float:
    """Max distance from centroid (Å); crude image-free extent estimate."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pos.size == 0:
        return 0.0
    com = pos.mean(axis=0)
    return float(np.linalg.norm(pos - com, axis=1).max())


def periodic_mm_status_line(cfg: PeriodicMmConfig, *, box_side_A: float) -> str:
    lj = "CHARMM IMAGE VDW" if cfg.charmm_vdw else "none (CHARMM VDW off)"
    return (
        f"periodic_external MM: Coulomb={cfg.lr_solver} ({cfg.scafacos_method}), "
        f"LJ={lj}, JAX real-space MM off, L={float(box_side_A):.3f} Å"
    )
