#!/usr/bin/env python3
"""CHARMM-free PET-MAD periodic MD for liquid ethanol in a 32 Å cube.

Prefer the package command::

    export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
    mmml metatomic-pbc-md --ensemble nve --minimize-steps 60 --n-steps 400

This file stays as a thin wrapper so existing example scripts keep working.
"""

from __future__ import annotations

from mmml.cli.misc.metatomic_pbc_md import main
from mmml.md.metatomic_pbc import nve_conservation_stats

__all__ = ["main", "nve_conservation_stats"]


if __name__ == "__main__":
    raise SystemExit(main())
