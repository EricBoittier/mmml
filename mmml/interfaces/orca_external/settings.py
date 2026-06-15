"""Shared settings helpers for MMML ORCA external-tool CLIs."""

from __future__ import annotations

import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MmmlOrcaSettings:
    checkpoint: Path
    cutoff: float | None = None
    is_noneq: bool = False
    use_dcmnet_dipole: bool = False
    disable_physnet_point_coulomb: bool = False


def add_model_arguments(parser: ArgumentParser) -> None:
    """Register checkpoint and model inference flags on ``parser``."""
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("MMML_CHECKPOINT"),
        help="Path to MMML checkpoint pickle (or set MMML_CHECKPOINT).",
    )
    parser.add_argument("--cutoff", type=float, default=None, help="Neighbor-list cutoff (Å).")
    parser.add_argument("--noneq", action="store_true", help="Load a non-equivariant checkpoint.")
    parser.add_argument(
        "--use-dcmnet-dipole",
        action="store_true",
        help="Report DCMNet dipole in calculator results (not passed to ORCA).",
    )
    parser.add_argument(
        "--disable-physnet-point-coulomb",
        action="store_true",
        help="Disable PhysNet internal point-charge Coulomb term at inference.",
    )


def settings_from_namespace(
    args: Any,
    *,
    default_settings: MmmlOrcaSettings | None = None,
) -> MmmlOrcaSettings:
    """Build settings from parsed CLI args, optionally inheriting server defaults."""
    checkpoint_value = args.checkpoint
    if not checkpoint_value and default_settings is not None:
        checkpoint_value = str(default_settings.checkpoint)

    if not checkpoint_value:
        raise ValueError(
            "Missing checkpoint. Pass --checkpoint /path/to/epoch.pkl, set MMML_CHECKPOINT, "
            "or start the server with a default checkpoint."
        )

    checkpoint = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cutoff = args.cutoff
    if cutoff is None and default_settings is not None:
        cutoff = default_settings.cutoff

    base_noneq = default_settings.is_noneq if default_settings else False
    base_dcmnet = default_settings.use_dcmnet_dipole if default_settings else False
    base_disable_coulomb = (
        default_settings.disable_physnet_point_coulomb if default_settings else False
    )

    return MmmlOrcaSettings(
        checkpoint=checkpoint,
        cutoff=cutoff,
        is_noneq=bool(args.noneq) or base_noneq,
        use_dcmnet_dipole=bool(args.use_dcmnet_dipole) or base_dcmnet,
        disable_physnet_point_coulomb=bool(args.disable_physnet_point_coulomb) or base_disable_coulomb,
    )
