"""Pure plotting from completed or reloaded dimer-scan results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .result import ScanResult


def plot_energy(result: ScanResult, output: str | Path) -> Path:
    """Plot energy against distance, leaving visible gaps at failed points."""

    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    distance = np.asarray([record.distance_angstrom for record in result.records])
    energy = np.asarray(
        [record.energy_kcal_mol if record.energy_kcal_mol is not None else np.nan
         for record in result.records]
    )
    fig, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.axhline(0.0, color="0.75", linewidth=0.8)
    axis.plot(distance, energy, marker="o")
    axis.set_xlabel("Distance (Å)")
    axis.set_ylabel(f"{result.config.energy_definition.title()} energy (kcal/mol)")
    axis.set_title(f"{result.config.residues[0]}–{result.config.residues[1]} 1D scan")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
