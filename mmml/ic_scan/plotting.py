"""Plot energy profiles from saved IC-scan results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .result import ScanResult


def plot_energy_profiles(result: ScanResult, output_dir: Path) -> list[Path]:
    """Write one PNG per 1D scan job; skip multidimensional jobs silently."""

    if not any(record.energy_ev is not None for record in result.records):
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: list[Path] = []
    by_scan: dict[str, list] = {}
    for record in result.records:
        by_scan.setdefault(record.scan_name, []).append(record)

    for scan_name, records in by_scan.items():
        active = [item for item in records[0].active_dofs.split(",") if item]
        if len(active) != 1:
            continue
        dof_name = active[0]
        xs: list[float] = []
        ys: list[float] = []
        for record in records:
            if record.energy_ev is None:
                continue
            coords = json.loads(record.coordinates_json)
            xs.append(float(coords[dof_name]))
            ys.append(float(record.energy_ev))
        if len(xs) < 2:
            continue
        order = np.argsort(xs)
        xs_arr = np.asarray(xs)[order]
        ys_arr = np.asarray(ys)[order]
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        ax.plot(xs_arr, ys_arr, marker="o", linewidth=1.5)
        ax.set_xlabel(dof_name)
        ax.set_ylabel("Energy / eV")
        ax.set_title(f"IC scan: {scan_name}")
        fig.tight_layout()
        path = output_dir / f"energy_{scan_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
