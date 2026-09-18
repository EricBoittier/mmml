"""ICML-style plots for ``mmml.analysis.interaction_pes`` JSON documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from mmml.analysis.interaction_pes import (
    DEFAULT_R_MAX_A,
    DEFAULT_R_MIN_A,
    ORIENTATION_HBOND,
    ORIENTATION_STACKED,
    PET_MAD_XS_RECEPTIVE_FIELD_A,
    mask_clash_energy,
)
from mmml.utils.plotting.styles import apply_plot_style, comparison_colors

# Okabe–Ito blue → grey → vermillion (interaction energy has a true zero).
OKABE_DIVERGING = LinearSegmentedColormap.from_list(
    "okabe_int",
    ["#0072B2", "#7FB4D3", "#E8E8E6", "#EBA07A", "#D55E00"],
)

SURFACE_CONTOUR_LEVELS_KCAL = (-2.0, -1.0, 1.0, 2.0)
SURFACE_COLOR_MAX_KCAL = 6.0
SLICE_Y_MAX_KCAL = 15.0
TRIMER_EINT_DISPLAY_MAX_KCAL = 40.0


def _style():
    return apply_plot_style("icml")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return path


def _slice_lookup(document: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["system"], row["orientation"]): row
        for row in document.get("dimer_slices", [])
    }


def plot_dimer_slices(document: Mapping[str, Any], output: Path | str) -> Path:
    """1D ``E_int(r)`` for each system: H-bond vs stacked, with the PET RF line."""
    style = _style()
    colors = comparison_colors(style, n=2)
    slices = _slice_lookup(document)
    systems = list(dict.fromkeys(row["system"] for row in document.get("dimer_slices", [])))
    if not systems:
        raise ValueError("document has no dimer_slices")
    n_col = len(systems)
    fig, axes = plt.subplots(1, n_col, figsize=(4.4 * n_col, 3.6), squeeze=False, sharey=False)
    rf = float(document.get("pet_receptive_field_angstrom", PET_MAD_XS_RECEPTIVE_FIELD_A))
    for ax, system in zip(axes[0], systems, strict=True):
        for orientation, color, ls in (
            (ORIENTATION_HBOND, colors[0], "-"),
            (ORIENTATION_STACKED, colors[1], "--"),
        ):
            row = slices.get((system, orientation))
            if row is None:
                continue
            energy = mask_clash_energy(row["e_int_kcal_mol"], row["min_contact_angstrom"])
            ax.plot(
                row["r_angstrom"],
                energy,
                color=color,
                linestyle=ls,
                marker="o",
                markersize=3.5,
                label=orientation,
            )
        ax.axhline(0.0, color="0.7", linewidth=0.8)
        ax.axvline(rf, color="0.45", linewidth=0.9, linestyle=":", label=f"PET RF ({rf:.0f} Å)")
        ax.set_xlim(DEFAULT_R_MIN_A, DEFAULT_R_MAX_A)
        ax.set_xlabel("COM distance (Å)")
        ax.set_title(f"{system} dimer")
        y0, y1 = ax.get_ylim()
        ax.set_ylim(min(y0, -1.0), min(max(y1, 1.0), SLICE_Y_MAX_KCAL))
        ax.legend(frameon=False, loc="best")
    axes[0][0].set_ylabel(r"$E_{\mathrm{int}}$ (kcal/mol)")
    fig.tight_layout()
    return _save(fig, Path(output))


def plot_dimer_surface(document: Mapping[str, Any], output: Path | str, *, index: int = 0) -> Path:
    """Heatmap + isolevels of one 2D ``E_int(r, theta)`` surface."""
    _style()
    surfaces = document.get("dimer_surfaces", [])
    if not surfaces:
        raise ValueError("document has no dimer_surfaces")
    surface = surfaces[index]
    r = np.asarray(surface["r_angstrom"], dtype=np.float64)
    theta = np.asarray(surface["theta_deg"], dtype=np.float64)
    z = mask_clash_energy(surface["e_int_kcal_mol"], surface["min_contact_angstrom"])
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        raise ValueError("surface has no non-clash samples")
    span = min(SURFACE_COLOR_MAX_KCAL, max(float(np.max(np.abs(finite))), 1.0))
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    mesh = ax.pcolormesh(
        r,
        theta,
        z,
        cmap=OKABE_DIVERGING,
        norm=TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span),
        shading="auto",
    )
    present = [
        level
        for level in SURFACE_CONTOUR_LEVELS_KCAL
        if float(np.nanmin(z)) < level < float(np.nanmax(z))
    ]
    if present:
        ax.contour(r, theta, np.ma.masked_invalid(z), levels=present, colors="0.15", linewidths=0.7)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$E_{\mathrm{int}}$ (kcal/mol)")
    ax.set_xlabel("COM distance (Å)")
    ax.set_ylabel("In-plane rotation of B (deg)")
    ax.set_title(f"{surface['system']} dimer $E_\\mathrm{{int}}(r,\\theta)$")
    fig.tight_layout()
    return _save(fig, Path(output))


def plot_trimer_mbe(document: Mapping[str, Any], output: Path | str) -> Path:
    """Trimer ``E_int``, pairwise reconstruction, and residual ``E3`` vs side length."""
    style = _style()
    colors = comparison_colors(style, n=3)
    rows = document.get("trimer_slices", [])
    if not rows:
        raise ValueError("document has no trimer_slices")
    n_col = len(rows)
    fig, axes = plt.subplots(1, n_col, figsize=(4.4 * n_col, 3.6), squeeze=False)
    rf = float(document.get("pet_receptive_field_angstrom", PET_MAD_XS_RECEPTIVE_FIELD_A))
    labels = (
        (r"$E_{\mathrm{int}}(ABC)$", "e_int_kcal_mol", colors[0], "-"),
        (r"$\sum E_{\mathrm{int}}(IJ)$", "e_pair_sum_kcal_mol", colors[1], "--"),
        (r"$E_3$", "e3_kcal_mol", colors[2], "-."),
    )
    for ax, row in zip(axes[0], rows, strict=True):
        r = np.asarray(row["r_angstrom"], dtype=np.float64)
        e_int = np.asarray(row["e_int_kcal_mol"], dtype=np.float64)
        keep = np.abs(e_int) <= TRIMER_EINT_DISPLAY_MAX_KCAL
        for label, key, color, ls in labels:
            y = np.asarray(row[key], dtype=np.float64).copy()
            y[~keep] = np.nan
            ax.plot(r, y, color=color, linestyle=ls, marker="o", markersize=3.5, label=label)
        ax.axhline(0.0, color="0.7", linewidth=0.8)
        ax.axvline(rf, color="0.45", linewidth=0.9, linestyle=":")
        ax.set_xlim(DEFAULT_R_MIN_A, DEFAULT_R_MAX_A)
        ax.set_xlabel("Trimer side / COM (Å)")
        ax.set_title(f"{row['system']} trimer")
        ax.legend(frameon=False, loc="best")
    axes[0][0].set_ylabel("Energy (kcal/mol)")
    fig.tight_layout()
    return _save(fig, Path(output))


def write_interaction_pes_figures(
    document: Mapping[str, Any],
    output_dir: Path | str,
    *,
    prefix: str = "pet_mad",
) -> dict[str, Path]:
    """Write the three campaign figures (PNG + PDF) under ``output_dir``."""
    out = Path(output_dir)
    paths = {
        "slices": plot_dimer_slices(document, out / f"{prefix}_dimer_slices.png"),
        "trimer": plot_trimer_mbe(document, out / f"{prefix}_trimer_mbe.png"),
    }
    if document.get("dimer_surfaces"):
        paths["surface"] = plot_dimer_surface(document, out / f"{prefix}_dimer_surface.png")
    return paths
