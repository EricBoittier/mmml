#!/usr/bin/env python3
"""Audit a spice-α / efield-train Slurm job from the login node.

Does not start training, download data, or touch CHARMM.

    python scripts/spice_alpha/audit_efield_job.py \
      --ckpt $HOME/mmml/ckpts/spice_ef_polar_big \
      --log artifacts/spice_ef_polar_big/slurm-22826285.out \
      --job 22826285
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmml.models.efield.audit import main


if __name__ == "__main__":
    raise SystemExit(main())
