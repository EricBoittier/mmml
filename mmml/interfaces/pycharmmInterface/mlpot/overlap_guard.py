"""Inter-monomer overlap checks during PyCHARMM MLpot dynamics."""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

DynamicsOverlapAction = Literal["error", "warn", "rescue", "off"]


@dataclass(frozen=True)
class OverlapRescueConfig:
    """CHARMM bonded+VDW SD/ABNR while MLpot stays registered."""

    nstep_sd: int = 200
    nstep_abnr: int = 400
    nprint: int = 50
    tolenr: float = 1e-3
    tolgrd: float = 1e-3
    verbose: bool = False


@dataclass(frozen=True)
class DynamicsOverlapConfig:
    """Chunked dynamics overlap guard (see :func:`run_dynamics_with_io`)."""

    action: DynamicsOverlapAction = "rescue"
    min_distance_A: float = 1.5
    check_interval: int = 50
    n_monomers: int = 1
    use_pbc: bool = False
    fallback_box_side_A: float | None = None
    rescue: OverlapRescueConfig = field(default_factory=OverlapRescueConfig)

    @property
    def enabled(self) -> bool:
        return (
            self.action != "off"
            and float(self.min_distance_A) > 0.0
            and int(self.n_monomers) > 1
        )


def add_dynamics_overlap_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Dynamics overlap guard (PyCHARMM MLpot)")
    group.add_argument(
        "--dynamics-overlap-action",
        choices=("error", "warn", "rescue", "off"),
        default="rescue",
        help=(
            "On inter-monomer overlap during MD: rescue=CHARMM bonded+VDW mini "
            "(default), error=abort, warn=log only, off=disable."
        ),
    )
    group.add_argument(
        "--dynamics-overlap-charmm-sd-steps",
        type=int,
        default=200,
        help="CHARMM SD steps for overlap rescue (default: 200).",
    )
    group.add_argument(
        "--dynamics-overlap-charmm-abnr-steps",
        type=int,
        default=400,
        help="CHARMM ABNR steps for overlap rescue (default: 400).",
    )
    group.add_argument(
        "--dynamics-overlap-min-distance",
        type=float,
        default=1.5,
        metavar="ANG",
        help=(
            "Minimum allowed inter-monomer atom distance in Å during dynamics "
            "(default: 1.5; CHARMM close-contact warnings often appear near this)."
        ),
    )
    group.add_argument(
        "--dynamics-overlap-check-interval",
        type=int,
        default=50,
        help=(
            "Integration steps between overlap checks (default: 50, matches imgfrq). "
            "Per stage, the effective interval is the largest divisor of the stage "
            "step count not exceeding this value (avoids a short final chunk)."
        ),
    )


def resolve_dynamics_overlap_config(
    args: argparse.Namespace,
    *,
    n_monomers: int,
    use_pbc: bool,
    fallback_box_side_A: float | None = None,
) -> DynamicsOverlapConfig:
    action = str(
        getattr(args, "dynamics_overlap_action", "rescue")
    ).lower()
    if action not in ("error", "warn", "rescue", "off"):
        raise ValueError(f"unknown dynamics_overlap_action: {action!r}")

    min_dist = getattr(args, "dynamics_overlap_min_distance", None)
    if min_dist is None:
        min_dist = getattr(args, "min_intermonomer_atom_distance", 1.5)

    interval = int(getattr(args, "dynamics_overlap_check_interval", 50))
    if use_pbc and fallback_box_side_A is None:
        box_size = getattr(args, "box_size", None)
        if box_size is not None:
            fallback_box_side_A = float(box_size)
    rescue = OverlapRescueConfig(
        nstep_sd=int(getattr(args, "dynamics_overlap_charmm_sd_steps", 200)),
        nstep_abnr=int(getattr(args, "dynamics_overlap_charmm_abnr_steps", 400)),
        nprint=max(1, int(getattr(args, "dyn_nprint", 50))),
        tolenr=float(getattr(args, "charmm_tolenr", 1e-3)),
        tolgrd=float(getattr(args, "charmm_tolgrd", 1e-3)),
        verbose=not bool(getattr(args, "quiet", False)),
    )
    return DynamicsOverlapConfig(
        action=action,  # type: ignore[arg-type]
        min_distance_A=float(min_dist),
        check_interval=max(1, interval),
        n_monomers=int(n_monomers),
        use_pbc=bool(use_pbc),
        fallback_box_side_A=(
            float(fallback_box_side_A)
            if use_pbc and fallback_box_side_A is not None and float(fallback_box_side_A) > 0.0
            else None
        ),
        rescue=rescue,
    )


def monomer_offsets(n_atoms: int, n_monomers: int) -> np.ndarray:
    """Uniform monomer atom offsets (length ``n_monomers + 1``)."""
    n_atoms = int(n_atoms)
    n_monomers = int(n_monomers)
    if n_monomers <= 0:
        raise ValueError(f"n_monomers must be > 0, got {n_monomers}")
    if n_atoms % n_monomers != 0:
        raise ValueError(
            f"atom count {n_atoms} not divisible by n_monomers={n_monomers}"
        )
    per = n_atoms // n_monomers
    return np.arange(0, n_atoms + 1, per, dtype=int)


@lru_cache(maxsize=1)
def _assert_no_intermonomer_atom_overlap_fn():
    """Load geometry_checks without importing ``mmml.utils`` (pulls JAX)."""
    path = Path(__file__).resolve().parents[3] / "utils" / "geometry_checks.py"
    spec = importlib.util.spec_from_file_location("_mmml_geometry_checks", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load geometry checks from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.assert_no_intermonomer_atom_overlap


def _overlap_cell(
    *,
    use_pbc: bool,
    fallback_box_side_A: float | None = None,
) -> float | np.ndarray | None:
    if not use_pbc:
        return None
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        resolve_charmm_cubic_box_side_A,
    )

    side, _ = resolve_charmm_cubic_box_side_A(
        fallback_side_A=fallback_box_side_A,
    )
    return float(side)


def _overlap_check(
    config: DynamicsOverlapConfig,
    *,
    context: str,
) -> float:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    pos = get_charmm_positions_array()
    offsets = monomer_offsets(int(pos.shape[0]), config.n_monomers)
    cell = _overlap_cell(
        use_pbc=config.use_pbc,
        fallback_box_side_A=config.fallback_box_side_A,
    )
    assert_no_intermonomer_atom_overlap = _assert_no_intermonomer_atom_overlap_fn()
    return assert_no_intermonomer_atom_overlap(
        pos,
        offsets,
        min_distance=config.min_distance_A,
        cell=cell,
        context=context,
    )


def check_dynamics_overlap(
    config: DynamicsOverlapConfig,
    *,
    context: str,
    step: int | None = None,
    mlpot_ctx: "MlpotContext | None" = None,
) -> float:
    """Check current CHARMM coordinates; raise, warn, or rescue per ``config.action``."""
    if not config.enabled:
        return float("inf")

    label = context if step is None else f"{context} at step {step}"

    if config.action == "error":
        return _overlap_check(config, context=label)

    try:
        return _overlap_check(config, context=label)
    except RuntimeError as exc:
        if config.action == "rescue":
            if mlpot_ctx is None:
                raise RuntimeError(
                    f"{exc}; overlap rescue requires MlpotContext"
                ) from exc
            print(
                f"{exc}\nAttempting MLpot overlap rescue "
                f"(bonded+VDW SD={config.rescue.nstep_sd}, "
                f"ABNR={config.rescue.nstep_abnr})...",
                flush=True,
            )
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                minimize_overlap_rescue,
            )

            try:
                minimize_overlap_rescue(mlpot_ctx, config.rescue)
            except Exception as rescue_exc:
                raise RuntimeError(
                    f"{exc}; MLpot overlap rescue failed: {rescue_exc}"
                ) from rescue_exc
            try:
                return _overlap_check(
                    config,
                    context=f"{label} after overlap rescue",
                )
            except RuntimeError as still_bad:
                raise RuntimeError(
                    f"{still_bad}; overlap rescue "
                    f"(SD={config.rescue.nstep_sd}, ABNR={config.rescue.nstep_abnr}) "
                    f"did not separate monomers — try larger "
                    f"--dynamics-overlap-charmm-sd-steps / "
                    f"--dynamics-overlap-charmm-abnr-steps, "
                    f"increase Packmol spacing, or relax "
                    f"--dynamics-overlap-min-distance"
                ) from still_bad
        if config.action == "warn":
            print(f"WARNING: {exc}", flush=True)
            return float("nan")
        raise
