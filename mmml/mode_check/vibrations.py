"""ASE finite-difference vibrational analysis for mode checks."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.vibrations import Vibrations


def run_ase_vibrations(
    atoms: Atoms,
    *,
    output_dir: Path | None = None,
    delta: float = 0.01,
    nfree: int = 2,
) -> dict[str, Any]:
    """Run ASE ``Vibrations`` and return frequency summary (cm⁻¹)."""
    name = "vib"
    if output_dir is not None:
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = str(output_dir / "vib")
    vib = Vibrations(atoms, name=name, delta=float(delta), nfree=int(nfree))
    vib.run()
    freqs = np.asarray(vib.get_frequencies(), dtype=float)
    real = np.asarray([float(x) for x in freqs if np.isfinite(x) and float(x) > 1.0])
    summary_path = None
    if output_dir is not None:
        summary_path = output_dir / "vib_summary.txt"
        vib.summary(log=str(summary_path))
    return {
        "frequencies_cm": [float(x) for x in freqs],
        "max_cm": float(np.max(np.real(freqs))) if freqs.size else float("nan"),
        "real_gt_1_cm": [float(x) for x in real],
        "summary_path": str(summary_path) if summary_path is not None else None,
    }
