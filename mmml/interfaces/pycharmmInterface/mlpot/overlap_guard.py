"""Inter-monomer overlap checks during PyCHARMM MLpot dynamics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import numpy as np

DynamicsOverlapAction = Literal["error", "warn", "off"]


@dataclass(frozen=True)
class DynamicsOverlapConfig:
    """Chunked dynamics overlap guard (see :func:`run_dynamics_with_io`)."""

    action: DynamicsOverlapAction = "error"
    min_distance_A: float = 1.5
    check_interval: int = 50
    n_monomers: int = 1
    use_pbc: bool = False

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
        choices=("error", "warn", "off"),
        default="error",
        help=(
            "Abort or warn when inter-monomer atoms are closer than the overlap "
            "threshold during MD (default: error)."
        ),
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
        help="Integration steps between overlap checks (default: 50, matches imgfrq).",
    )


def resolve_dynamics_overlap_config(
    args: argparse.Namespace,
    *,
    n_monomers: int,
    use_pbc: bool,
) -> DynamicsOverlapConfig:
    action = str(
        getattr(args, "dynamics_overlap_action", "error")
    ).lower()
    if action not in ("error", "warn", "off"):
        raise ValueError(f"unknown dynamics_overlap_action: {action!r}")

    min_dist = getattr(args, "dynamics_overlap_min_distance", None)
    if min_dist is None:
        min_dist = getattr(args, "min_intermonomer_atom_distance", 1.5)

    interval = int(getattr(args, "dynamics_overlap_check_interval", 50))
    return DynamicsOverlapConfig(
        action=action,  # type: ignore[arg-type]
        min_distance_A=float(min_dist),
        check_interval=max(1, interval),
        n_monomers=int(n_monomers),
        use_pbc=bool(use_pbc),
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


def _overlap_cell(*, use_pbc: bool) -> float | np.ndarray | None:
    if not use_pbc:
        return None
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        get_charmm_cubic_box_side_A,
    )

    return float(get_charmm_cubic_box_side_A())


def check_dynamics_overlap(
    config: DynamicsOverlapConfig,
    *,
    context: str,
    step: int | None = None,
) -> float:
    """Check current CHARMM coordinates; raise or warn per ``config.action``."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
    )
    from mmml.utils.geometry_checks import assert_no_intermonomer_atom_overlap

    if not config.enabled:
        return float("inf")

    label = context if step is None else f"{context} at step {step}"
    pos = get_charmm_positions_array()
    offsets = monomer_offsets(int(pos.shape[0]), config.n_monomers)
    cell = _overlap_cell(use_pbc=config.use_pbc)

    if config.action == "error":
        return assert_no_intermonomer_atom_overlap(
            pos,
            offsets,
            min_distance=config.min_distance_A,
            cell=cell,
            context=label,
        )

    try:
        return assert_no_intermonomer_atom_overlap(
            pos,
            offsets,
            min_distance=config.min_distance_A,
            cell=cell,
            context=label,
        )
    except RuntimeError as exc:
        print(f"WARNING: {exc}", flush=True)
        return float("nan")
