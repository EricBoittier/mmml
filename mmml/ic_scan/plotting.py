"""Plot energy / force profiles from saved IC-scan results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .result import EV_TO_KCAL_MOL, ScanResult


def _1d_series(
    result: ScanResult,
    *,
    value_attr: str,
) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    """Return ``{scan_name: (dof_name, x, y)}`` for successful 1D scans."""

    by_scan: dict[str, list] = {}
    for record in result.records:
        by_scan.setdefault(record.scan_name, []).append(record)

    out: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for scan_name, records in by_scan.items():
        active = [item for item in records[0].active_dofs.split(",") if item]
        if len(active) != 1:
            continue
        dof_name = active[0]
        xs: list[float] = []
        ys: list[float] = []
        for record in records:
            value = getattr(record, value_attr, None)
            if value is None:
                continue
            coords = json.loads(record.coordinates_json)
            xs.append(float(coords[dof_name]))
            ys.append(float(value))
        if len(xs) < 2:
            continue
        order = np.argsort(xs)
        out[scan_name] = (
            dof_name,
            np.asarray(xs)[order],
            np.asarray(ys)[order],
        )
    return out


def plot_energy_profiles(result: ScanResult, output_dir: Path) -> list[Path]:
    """Write energy and max-|F| PNGs per 1D scan job; skip N-D jobs."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: list[Path] = []
    energy_series = _1d_series(result, value_attr="energy_ev")
    force_series = _1d_series(result, value_attr="max_force_ev_A")

    for scan_name, (dof_name, xs_arr, ys_arr) in energy_series.items():
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        ax.plot(xs_arr, ys_arr, marker="o", linewidth=1.5, label="E / eV")
        ax.set_xlabel(f"{dof_name} / deg")
        ax.set_ylabel("Energy / eV")
        ax.set_title(f"IC scan: {scan_name}")
        fig.tight_layout()
        path = output_dir / f"energy_{scan_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

        # Relative kcal/mol (min → 0) for ML comparisons
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        y_kcal = (ys_arr - ys_arr.min()) * EV_TO_KCAL_MOL
        ax.plot(xs_arr, y_kcal, marker="o", linewidth=1.5)
        ax.set_xlabel(f"{dof_name} / deg")
        ax.set_ylabel("ΔE / kcal·mol⁻¹")
        ax.set_title(f"IC scan: {scan_name}")
        fig.tight_layout()
        path = output_dir / f"energy_rel_kcal_{scan_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    for scan_name, (dof_name, xs_arr, ys_arr) in force_series.items():
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        ax.plot(xs_arr, ys_arr, marker="o", linewidth=1.5, color="C1")
        ax.set_xlabel(f"{dof_name} / deg")
        ax.set_ylabel("max |F| / eV·Å⁻¹")
        ax.set_title(f"IC scan forces: {scan_name}")
        fig.tight_layout()
        path = output_dir / f"maxforce_{scan_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_model_comparison(
    series: dict[str, ScanResult],
    output_dir: Path,
    *,
    scan_name: str | None = None,
) -> list[Path]:
    """Overlay energy (rel kcal/mol) and max-|F| vs φ for multiple models.

    ``series`` maps legend label → :class:`ScanResult`.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover 1D scan names present in all results
    scan_names: set[str] | None = None
    per_model_energy: dict[str, dict[str, tuple[str, np.ndarray, np.ndarray]]] = {}
    per_model_force: dict[str, dict[str, tuple[str, np.ndarray, np.ndarray]]] = {}
    for label, result in series.items():
        e = _1d_series(result, value_attr="energy_ev")
        f = _1d_series(result, value_attr="max_force_ev_A")
        per_model_energy[label] = e
        per_model_force[label] = f
        names = set(e) | set(f)
        scan_names = names if scan_names is None else (scan_names & names)

    if not scan_names:
        return []
    targets = sorted(scan_names) if scan_name is None else [scan_name]
    paths: list[Path] = []

    for name in targets:
        if name not in (scan_names or ()):
            continue
        # Energy overlay (relative kcal/mol)
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        dof_label = "φ"
        for label, scans in per_model_energy.items():
            if name not in scans:
                continue
            dof_label, xs, ys = scans[name]
            y_rel = (ys - ys.min()) * EV_TO_KCAL_MOL
            ax.plot(xs, y_rel, marker="o", linewidth=1.5, label=label)
        ax.set_xlabel(f"{dof_label} / deg")
        ax.set_ylabel("ΔE / kcal·mol⁻¹")
        ax.set_title(f"{name}: energy vs φ")
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"compare_energy_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

        # Max force overlay
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for label, scans in per_model_force.items():
            if name not in scans:
                continue
            dof_label, xs, ys = scans[name]
            ax.plot(xs, ys, marker="o", linewidth=1.5, label=label)
        ax.set_xlabel(f"{dof_label} / deg")
        ax.set_ylabel("max |F| / eV·Å⁻¹")
        ax.set_title(f"{name}: max |F| vs φ")
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"compare_maxforce_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths
