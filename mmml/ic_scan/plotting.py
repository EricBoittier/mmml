"""Plot energy / force profiles from saved IC-scan results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .result import EV_TO_KCAL_MOL, ScanResult

# Uniform dihedral axes so a 3-fold rotor is visually 3-fold (not an autoscale
# that stretches one well). Tick spacing matches the methyl period (120°).
DIHEDRAL_AXIS_MIN_DEG = -180.0
DIHEDRAL_AXIS_MAX_DEG = 180.0
DIHEDRAL_TICK_DEG = 60.0
DIHEDRAL_SYMMETRY_GUIDES_DEG = (-120.0, 0.0, 120.0)
REL_ENERGY_YLIM_PAD = 0.08


def style_dihedral_scan_axes(
    ax,
    *,
    y_max: float,
    y_min: float = 0.0,
    show_guides: bool = True,
) -> None:
    """Force ±180° x-limits, 60° ticks, and a shared padded y-range."""

    import numpy as np

    ax.set_xlim(DIHEDRAL_AXIS_MIN_DEG, DIHEDRAL_AXIS_MAX_DEG)
    ticks = np.arange(
        DIHEDRAL_AXIS_MIN_DEG,
        DIHEDRAL_AXIS_MAX_DEG + 0.5 * DIHEDRAL_TICK_DEG,
        DIHEDRAL_TICK_DEG,
    )
    ax.set_xticks(ticks)
    span = float(y_max) - float(y_min)
    pad = REL_ENERGY_YLIM_PAD * span if span > 0.0 else 0.05
    ax.set_ylim(float(y_min), float(y_max) + pad)
    if show_guides:
        for x in DIHEDRAL_SYMMETRY_GUIDES_DEG:
            ax.axvline(float(x), color="0.85", linewidth=0.7, zorder=0)


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

    dof_map = result.config.dof_map()

    def _apply_dihedral_x(ax, dof_name: str, *, y_max: float, y_min: float | None = None) -> None:
        dof = dof_map.get(dof_name)
        if dof is None or dof.kind != "dihedral":
            return
        kwargs = {"y_max": y_max}
        if y_min is not None:
            kwargs["y_min"] = y_min
        style_dihedral_scan_axes(ax, **kwargs)

    for scan_name, (dof_name, xs_arr, ys_arr) in energy_series.items():
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        ax.plot(xs_arr, ys_arr, marker="o", linewidth=1.5, label="E / eV")
        ax.set_xlabel(f"{dof_name} / deg")
        ax.set_ylabel("Energy / eV")
        ax.set_title(f"IC scan: {scan_name}")
        _apply_dihedral_x(ax, dof_name, y_max=float(np.max(ys_arr)), y_min=float(np.min(ys_arr)))
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
        _apply_dihedral_x(ax, dof_name, y_max=float(np.max(y_kcal)), y_min=0.0)
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
        _apply_dihedral_x(ax, dof_name, y_max=float(np.max(ys_arr)), y_min=0.0)
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
        y_hi = 0.0
        for label, scans in per_model_energy.items():
            if name not in scans:
                continue
            dof_label, xs, ys = scans[name]
            y_rel = (ys - ys.min()) * EV_TO_KCAL_MOL
            y_hi = max(y_hi, float(np.max(y_rel)))
            ax.plot(xs, y_rel, marker="o", linewidth=1.5, label=label)
        ax.set_xlabel(f"{dof_label} / deg")
        ax.set_ylabel("ΔE / kcal·mol⁻¹")
        ax.set_title(f"{name}: energy vs φ")
        dof = next(iter(series.values())).config.dof_map().get(dof_label)
        if dof is not None and dof.kind == "dihedral":
            style_dihedral_scan_axes(ax, y_max=y_hi, y_min=0.0)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"compare_energy_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

        # Max force overlay
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        f_hi = 0.0
        for label, scans in per_model_force.items():
            if name not in scans:
                continue
            dof_label, xs, ys = scans[name]
            f_hi = max(f_hi, float(np.max(ys)))
            ax.plot(xs, ys, marker="o", linewidth=1.5, label=label)
        ax.set_xlabel(f"{dof_label} / deg")
        ax.set_ylabel("max |F| / eV·Å⁻¹")
        ax.set_title(f"{name}: max |F| vs φ")
        dof = next(iter(series.values())).config.dof_map().get(dof_label)
        if dof is not None and dof.kind == "dihedral":
            style_dihedral_scan_axes(ax, y_max=f_hi, y_min=0.0)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"compare_maxforce_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths
