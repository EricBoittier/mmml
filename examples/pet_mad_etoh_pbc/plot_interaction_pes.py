#!/usr/bin/env python3
"""CHARMM-free PET-MAD interaction slices and surfaces.

Prefer the package command::

    export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
    JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu mmml pet-interaction-pes
"""

from __future__ import annotations

from mmml.cli.misc.pet_interaction_pes import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
