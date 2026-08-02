#!/usr/bin/env python3
"""Contact-ok soft-well sweep for soft-well E_int aux FT (on=5).

Thin wrapper around eval_lever2_on5_distill_sweep with softwell defaults.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "slurm" / "dense_dt_campaign"))

os.environ.setdefault("DDC_ON5D_TAG", "hybrid_mm_lever2_on5_softwell")
os.environ.setdefault(
    "DDC_ON5D_EVAL_OUT",
    str(ROOT / "artifacts/lj_scales/dense_dt_campaign/overbind_ablation/lever2_on5_softwell"),
)
os.environ.setdefault("DDC_ON5D_SWEEP_EPOCHS", "1,3,5,8,10,12,15,20")

from eval_lever2_on5_distill_sweep import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
