"""Lightweight argparse helpers for CHARMM MM pretreat flags.

Kept separate from ``cli_common`` so ``mmml md-system -h`` does not import the
large runtime module just to print help.
"""

from __future__ import annotations

from typing import Any

DEFAULT_CHARMM_MM_PRETREAT_DT_FS = 1.0


def add_charmm_mm_pretreat_physics_args(group: Any) -> None:
    """Pretreat integrator and bath flags (shared by staged CLI and md-system)."""
    group.add_argument(
        "--charmm-mm-pretreat-dt-fs",
        type=float,
        default=DEFAULT_CHARMM_MM_PRETREAT_DT_FS,
        metavar="FS",
        help=(
            "Pretreat CHARMM dynamics timestep in fs (default: 1.0). "
            "Independent of MLpot --dt-fs."
        ),
    )
    group.add_argument(
        "--charmm-mm-pretreat-temperature",
        type=float,
        default=None,
        metavar="K",
        help="Pretreat CHARMM heat/equi/prod temperature (default: --temperature).",
    )
    group.add_argument(
        "--charmm-mm-pretreat-pressure",
        type=float,
        default=None,
        metavar="ATM",
        help=(
            "Pretreat CHARMM NPT reference pressure (default: --npt-pressure or --pressure)."
        ),
    )
    group.add_argument(
        "--charmm-mm-pretreat-echeck",
        type=float,
        default=None,
        metavar="KCAL",
        help=(
            "ECHECK for pretreat CPT equi/prod and mini box equil (kcal/mol). "
            "Default: disabled. Use 0 or a negative value to keep ECHECK off."
        ),
    )
    group.add_argument(
        "--charmm-mm-pretreat-inbfrq",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Pretreat CHARMM nonbond list rebuild cadence (inbfrq). "
            "Default scales with --charmm-mm-pretreat-dt-fs (400 at 2 fs vs 50 for MLpot)."
        ),
    )
    group.add_argument(
        "--charmm-mm-pretreat-imgfrq",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Pretreat PBC image/HB list cadence (imgfrq/ihbfrq/ilbfrq). "
            "Default matches pretreat inbfrq."
        ),
    )
    group.add_argument(
        "--charmm-mm-pretreat-ixtfrq",
        type=int,
        default=None,
        metavar="N",
        help="Pretreat crystal transform cadence (ixtfrq; default scales with pretreat dt).",
    )
