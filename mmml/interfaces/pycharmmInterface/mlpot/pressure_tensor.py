"""NPT pressure-tensor reference, logging, and instantaneous CHARMM reports."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles

# CHARMM ``IUPTEN`` unit for piston-tensor time series (avoid 1–3 = DCD/restart).
DEFAULT_PRESSURE_TENSOR_LOG_UNIT = 29

@dataclass(frozen=True)
class NptPressureTensor:
    """Anisotropic CPT reference pressure tensor (``PR**`` keywords, atm)."""

    prxx: float
    pryy: float
    przz: float
    prxy: float = 0.0
    prxz: float = 0.0
    pryz: float = 0.0

    def is_isotropic(self, pref: float, *, tol: float = 1e-9) -> bool:
        p = float(pref)
        return (
            math.isclose(self.prxx, p, abs_tol=tol)
            and math.isclose(self.pryy, p, abs_tol=tol)
            and math.isclose(self.przz, p, abs_tol=tol)
            and math.isclose(self.prxy, 0.0, abs_tol=tol)
            and math.isclose(self.prxz, 0.0, abs_tol=tol)
            and math.isclose(self.pryz, 0.0, abs_tol=tol)
        )

    def as_dynamics_kwargs(self) -> dict[str, float]:
        return {
            "PRXX": self.prxx,
            "PRYY": self.pryy,
            "PRZZ": self.przz,
            "PRXY": self.prxy,
            "PRXZ": self.prxz,
            "PRYZ": self.pryz,
        }


def parse_npt_pressure_tensor(
    spec: str | None,
    *,
    isotropic_pref: float = 1.0,
) -> NptPressureTensor | None:
    """Parse ``xx,yy,zz,xy,xz,yz`` or return ``None`` for isotropic ``pref`` only."""
    if spec is None or not str(spec).strip():
        return None
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 6:
        raise ValueError(
            "npt-pressure-tensor requires 6 comma-separated values: "
            "xx,yy,zz,xy,xz,yz (atm)"
        )
    try:
        values = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(
            "npt-pressure-tensor values must be numeric (xx,yy,zz,xy,xz,yz)"
        ) from exc
    tensor = NptPressureTensor(*values)
    if tensor.is_isotropic(isotropic_pref):
        return None
    return tensor


def apply_npt_pressure_reference(
    kw: dict[str, Any],
    *,
    pref: float,
    pressure_tensor: NptPressureTensor | None,
) -> None:
    """Attach isotropic ``pref`` or anisotropic ``PR**`` CPT reference keywords."""
    if pressure_tensor is None:
        kw["pint pconst pref"] = float(pref)
        return
    kw["pint"] = True
    kw["pconst"] = True
    kw.update(pressure_tensor.as_dynamics_kwargs())


def apply_npt_pressure_log_kwargs(
    kw: dict[str, Any],
    *,
    interval_steps: int,
    iupten_unit: int = DEFAULT_PRESSURE_TENSOR_LOG_UNIT,
) -> None:
    """Enable CHARMM ``IUPTEN`` / ``IPTFRQ`` piston-tensor logging during ``dyna``."""
    interval = int(interval_steps)
    if interval <= 0:
        return
    kw["iptfrq"] = max(1, interval)
    kw["iupten"] = int(iupten_unit)


def attach_pressure_tensor_log(
    io: CharmmTrajectoryFiles,
    path: Path | str,
    *,
    unit: int = DEFAULT_PRESSURE_TENSOR_LOG_UNIT,
) -> Path:
    """Register a formatted pressure-tensor log opened by ``CharmmTrajectoryFiles``."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    io.pressure_tensor_log = log_path
    io.pressure_tensor_log_unit = int(unit)
    return log_path


def resolve_npt_cpt_pressure_options(args: argparse.Namespace) -> dict[str, Any]:
    """NPT barostat options shared by equi/prod dynamics builders."""
    pref = float(getattr(args, "npt_pressure", 1.0))
    tensor = parse_npt_pressure_tensor(
        getattr(args, "npt_pressure_tensor", None),
        isotropic_pref=pref,
    )
    return {
        "thermostat": getattr(args, "npt_thermostat", "hoover"),
        "pref": pref,
        "pgamma": float(getattr(args, "npt_pgamma", 5.0)),
        "pressure_tensor": tensor,
        "pressure_log_interval": int(
            getattr(args, "npt_pressure_log_interval", 0) or 0
        ),
    }


def npt_cpt_builder_options(args: argparse.Namespace) -> dict[str, Any]:
    """Keyword subset for :func:`build_cpt_*_dynamics` (excludes logging metadata)."""
    opts = resolve_npt_cpt_pressure_options(args)
    return {
        k: v
        for k, v in opts.items()
        if k not in ("pressure_log_interval",)
    }


def maybe_configure_stage_pressure_tensor_io(
    io: CharmmTrajectoryFiles,
    kw: dict[str, Any],
    *,
    log_path: Path | None,
    pressure_log_interval: int,
) -> None:
    """Wire high-frequency piston logging when ``pressure_log_interval > 0``."""
    if int(pressure_log_interval) <= 0 or log_path is None:
        return
    if not bool(kw.get("cpt")):
        return
    attach_pressure_tensor_log(io, log_path)
    apply_npt_pressure_log_kwargs(
        kw,
        interval_steps=pressure_log_interval,
        iupten_unit=io.pressure_tensor_log_unit,
    )


def report_instantaneous_pressure_tensor(
    temp: float,
    *,
    context: str = "NPT",
    quiet: bool = False,
    mlpot_ctx: Any | None = None,
) -> None:
    """Run CHARMM ``PRESSure INSTantaneous`` after an energy evaluation.

    Requires coordinates and virials in CHARMM memory (call after ``energy`` /
    MLpot refresh). Output goes to the CHARMM log.
    """
    if mlpot_ctx is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            refresh_mlpot_energy_and_grms,
        )

        refresh_mlpot_energy_and_grms(
            mlpot_ctx,
            context=f"{context} instantaneous pressure",
            quiet=quiet,
        )
    else:
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics import safe_energy_show

        safe_energy_show()

    import pycharmm

    t = float(temp)
    noprint = " noprint" if quiet else ""
    pycharmm.lingo.charmm_script(
        f"pressure instantaneous temperature {t:.6g}{noprint}\n"
    )
    if not quiet:
        print(
            f"{context}: CHARMM instantaneous pressure tensor "
            f"(T={t:.2f} K; see log above)",
            flush=True,
        )


def maybe_report_instantaneous_pressure_tensor(
    *,
    stage: str,
    temp: float,
    args: argparse.Namespace,
    use_cpt: bool,
    mlpot_ctx: Any | None = None,
) -> None:
    """One-off virial pressure tensor before equi/prod when CPT is active."""
    if not use_cpt or stage not in ("equi", "prod"):
        return
    if getattr(args, "skip_npt_pressure_report", False):
        return
    report_instantaneous_pressure_tensor(
        temp,
        context=stage.upper(),
        quiet=bool(getattr(args, "quiet", False)),
        mlpot_ctx=mlpot_ctx,
    )
