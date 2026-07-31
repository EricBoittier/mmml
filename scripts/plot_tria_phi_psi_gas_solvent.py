#!/usr/bin/env python3
"""Ramachandran comparison: gas-phase vs solvent-relaxed TRIA φ/ψ MM energies.

Inputs
------
- Gas CSV from ``scan_trialanine_phi_psi_pes.py`` (``phi_psi_pes.csv``)
- Solvent CSV from ``scan_trialanine_phi_psi_solvent.py`` (``phi_psi_solvent.csv``)

Either file alone is OK; both produce a side-by-side figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from mmml.utils.plotting.styles import apply_plot_style, default_cmap


def _wrap180(deg: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(deg, dtype=float)
    return ((arr + 180.0) % 360.0) - 180.0


def _relative_kcal(series: pd.Series) -> np.ndarray:
    v = np.asarray(series, dtype=float)
    finite = np.isfinite(v)
    if not np.any(finite):
        return v
    return v - np.nanmin(v)


def _contour_panel(
    ax,
    phi: np.ndarray,
    psi: np.ndarray,
    energy: np.ndarray,
    *,
    title: str,
    vmax: float,
    show_ylabel: bool,
) -> object:
    mask = np.isfinite(phi) & np.isfinite(psi) & np.isfinite(energy)
    phi, psi, energy = phi[mask], psi[mask], energy[mask]
    if phi.size < 3:
        ax.text(0.5, 0.5, "too few points", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return None

    xi = np.linspace(-180, 180, 200)
    yi = np.linspace(-180, 180, 200)
    xx, yy = np.meshgrid(xi, yi)
    zz = griddata((phi, psi), energy, (xx, yy), method="cubic")
    if zz is not None and np.any(np.isfinite(zz)):
        zz = np.where(np.isfinite(zz), zz, np.nanmax(energy))
        zz = gaussian_filter(zz, sigma=1.5)
        levels = np.linspace(0.0, vmax, 25)
        cs = ax.contourf(xx, yy, zz, levels=levels, cmap=default_cmap("sequential"), extend="max")
        ax.contour(xx, yy, zz, levels=8, colors="#222222", linewidths=0.25, alpha=0.35)
    else:
        cs = ax.scatter(phi, psi, c=energy, s=40, cmap=default_cmap("sequential"), vmin=0, vmax=vmax)

    ax.scatter(phi, psi, s=10, c="white", edgecolors="#333333", linewidths=0.3, alpha=0.7, zorder=3)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\phi$ (deg)")
    if show_ylabel:
        ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_title(title)
    return cs


def plot_gas_solvent(
    gas_csv: Path | None,
    solvent_csv: Path | None,
    out: Path,
    *,
    vmax: float = 40.0,
    style: str = "icml",
) -> Path:
    apply_plot_style(style)
    panels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    if gas_csv is not None and Path(gas_csv).is_file():
        g = pd.read_csv(gas_csv)
        panels.append(
            (
                "Gas (CHARMM MM min)",
                _wrap180(g["actual_phi_deg"]),
                _wrap180(g["actual_psi_deg"]),
                _relative_kcal(g["charmm_mm_min_energy_kcal_mol"]),
            )
        )
        if "ml_energy_eV" in g.columns and np.any(np.isfinite(g["ml_energy_eV"])):
            e_ml = 23.0609 * _relative_kcal(g["ml_energy_eV"])  # eV → kcal/mol
            panels.append(
                (
                    "Gas (ML min)",
                    _wrap180(g["actual_phi_deg"]),
                    _wrap180(g["actual_psi_deg"]),
                    e_ml,
                )
            )

    if solvent_csv is not None and Path(solvent_csv).is_file():
        s = pd.read_csv(solvent_csv)
        panels.append(
            (
                "Solvent (MM, φ/ψ constrained)",
                _wrap180(s["actual_phi_deg"]),
                _wrap180(s["actual_psi_deg"]),
                _relative_kcal(s["solvent_mm_min_energy_kcal_mol"]),
            )
        )

    if not panels:
        raise FileNotFoundError("Need at least one of gas_csv / solvent_csv with data")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    last_mappable = None
    for ax, (title, phi, psi, e) in zip(axes[0], panels):
        last_mappable = _contour_panel(
            ax, phi, psi, e, title=title, vmax=vmax, show_ylabel=(ax is axes[0, 0])
        )

    if last_mappable is not None:
        fig.colorbar(
            last_mappable,
            ax=axes.ravel().tolist(),
            shrink=0.85,
            label=r"$\Delta E$ (kcal/mol)",
        )
    fig.suptitle(r"Trialanine central $\phi$/$\psi$: gas vs solvent-relaxed", y=1.02)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _write_demo_csvs(out_dir: Path) -> tuple[Path, Path]:
    """Synthetic basins for figure smoke when no scan has been run yet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    phi = np.linspace(-165, 165, 12)
    psi = np.linspace(-165, 165, 12)
    rows_g: list[dict[str, float]] = []
    rows_s: list[dict[str, float]] = []
    for p in phi:
        for q in psi:
            # Rough alanine-like basins near (-60,-45) and (-60,135)
            e_gas = (
                8.0 * (1 - np.exp(-((p + 60) ** 2 + (q + 45) ** 2) / (2 * 35**2)))
                + 10.0 * (1 - np.exp(-((p + 60) ** 2 + (q - 135) ** 2) / (2 * 40**2)))
                + 0.002 * (p**2 + q**2) / 1000.0
            )
            e_sol = e_gas * 0.85 + 1.5 * np.sin(np.deg2rad(p)) ** 2
            rows_g.append(
                {
                    "phi_deg": p,
                    "psi_deg": q,
                    "actual_phi_deg": p,
                    "actual_psi_deg": q,
                    "charmm_mm_min_energy_kcal_mol": e_gas,
                    "ml_energy_eV": e_gas / 23.0609,
                }
            )
            rows_s.append(
                {
                    "phi_deg": p,
                    "psi_deg": q,
                    "actual_phi_deg": p,
                    "actual_psi_deg": q,
                    "solvent_mm_min_energy_kcal_mol": e_sol,
                }
            )
    g_path = out_dir / "phi_psi_pes.DEMO.csv"
    s_path = out_dir / "phi_psi_solvent.DEMO.csv"
    pd.DataFrame(rows_g).to_csv(g_path, index=False)
    pd.DataFrame(rows_s).to_csv(s_path, index=False)
    return g_path, s_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gas-csv", type=Path, default=None)
    parser.add_argument("--solvent-csv", type=Path, default=None)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("artifacts/tria_phi_psi_scan/figures/gas_vs_solvent.png"),
    )
    parser.add_argument("--vmax", type=float, default=40.0)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Write synthetic DEMO CSVs and plot them (no CHARMM required)",
    )
    args = parser.parse_args()

    gas_csv, solvent_csv = args.gas_csv, args.solvent_csv
    if args.demo:
        demo_dir = Path(args.output).parent / "demo_csv"
        gas_csv, solvent_csv = _write_demo_csvs(demo_dir)
        print(f"DEMO gas CSV → {gas_csv}")
        print(f"DEMO solvent CSV → {solvent_csv}")

    out = plot_gas_solvent(gas_csv, solvent_csv, args.output, vmax=float(args.vmax))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
