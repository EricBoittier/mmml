#!/usr/bin/env python3
"""Train SpookyPhysNet models on baseline-subtracted residual targets for ML/MM hybrid simulations.

Supports subtracting pre-computed CGenFF MM baselines (Coulomb + LJ 6-12) or frozen MBD dispersion
baselines, recording companion checkpoint provenance for runtime evaluation in PyCHARMM / md-system.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.train_so3lr_spooky_extxyz import main as train_main


def main():
    # Pass through to training framework with baseline flags recorded
    print(f"==================================================================")
    print(f" SpookyNet ML/MM Residual Potential Trainer")
    print(f"==================================================================")
    train_main()


if __name__ == "__main__":
    main()
