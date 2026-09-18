"""ICML-style plots for ``mmml.analysis.interaction_pes`` JSON documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mmml.analysis.interaction_pes import (
    ORIENTATION_ACCEPTOR_ACCEPTOR,
    ORIENTATION_CARBONYL,
    ORIENTATION_LINEAR_OH_O,
    ORIENTATION_METHYL,
    PET_MAD_XS_RECEPTIVE_FIELD_A,
)
from mmml.analysis.interaction_pes_geom import MOTIF_CYCLIC, MOTIF_LINEAR, ORIENTATION_LABELS
from mmml.utils.plotting.styles import apply_plot_style, comparison_colors

OKABE_DIVERGING = LinearSegmentedColormap.from_list(
    "okabe_int",
    ["#0072B2", "#7FB4D3", "#E8E8E6", "#EBA07A", "#D55E00"],
)

SURFACE_CONTOUR_LEVELS_KCAL = (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0)
SURFACE_COLOR_MAX_KCAL = 8.0
SLICE_Y_WELL_PAD_KCAL = 0.8
SLICE_Y_TOP_KCAL = 3.0
WALL_INSET_R_MAX_A = 3.3


def _style():
    return apply_plot_style("icml")


def _save(fig, path: Path, *, write_pdf: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _rf(document: Mapping[str, Any]) -> float:
    return float(document.get("pet_receptive_field_angstrom", PET_MAD_XS_RECEPTIVE_FIELD_A))


def _mark_rf(ax, rf: float, *, label: str | None = None) -> None:
    ax.axvline(rf, color="0.45", linewidth=0.9, linestyle=":", label=label)


def _eint_ylabel() -> str:
    return r"$E_{\mathrm{int}}=E(AB)-E(A)-E(B)$ (kcal/mol)"


def _slice_lookup(document: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["system"], row["orientation"]): row
        for row in document.get("dimer_slices", [])
    }


def _systems_in_slices(document: Mapping[str, Any]) -> list[str]:
    return list(dict.fromkeys(row["system"] for row in document.get("dimer_slices", [])))


def _pair_orientations(system: str) -> tuple[str, str]:
    if system == "acetone":
        return ORIENTATION_CARBONYL, ORIENTATION_METHYL
    return ORIENTATION_LINEAR_OH_O, ORIENTATION_ACCEPTOR_ACCEPTOR


def _orientation_legend(orientation: str) -> str:
    return ORIENTATION_LABELS.get(orientation, orientation).replace("$", "")


def plot_dimer_slices(
    document: Mapping[str, Any],
    output: Path | str,
    *,
    write_pdf: bool = True,
) -> Path:
    """1D ``E_int(r)`` with a visible repulsive limb, well markers, and PET RF."""
    style = _style()
    palette = comparison_colors(style, n=8)
    colors = (palette[5], palette[6])  # blue, vermillion (skip black)
    slices = _slice_lookup(document)
    systems = _systems_in_slices(document)
    if not systems:
        raise ValueError("document has no dimer_slices")
    n_col = len(systems)
    fig, axes = plt.subplots(1, n_col, figsize=(4.5 * n_col, 3.8), squeeze=False, sharey=False)
    rf = _rf(document)
    for ax, system in zip(axes[0], systems, strict=True):
        ori_pair = _pair_orientations(system)
        well_pts: list[tuple[float, float]] = []
        r_min, r_max = np.inf, 0.0
        all_energy: list[np.ndarray] = []
        for orientation, color, ls in (
            (ori_pair[0], colors[0], "-"),
            (ori_pair[1], colors[1], "--"),
        ):
            row = slices.get((system, orientation))
            if row is None:
                continue
            r = np.asarray(row["r_angstrom"], dtype=np.float64)
            energy = np.asarray(row["e_int_kcal_mol"], dtype=np.float64)
            all_energy.append(energy)
            r_min = min(r_min, float(np.min(r)))
            r_max = max(r_max, float(np.max(r)))
            ax.plot(
                r,
                energy,
                color=color,
                linestyle=ls,
                marker="o",
                markersize=3.2,
                label=ORIENTATION_LABELS.get(orientation, orientation),
            )
            well_r = row.get("well_r_angstrom")
            well_e = row.get("well_kcal_mol")
            if well_r is not None and well_e is not None and np.isfinite(well_e) and well_e < 0.0:
                ax.scatter([well_r], [well_e], color=color, s=36, zorder=5, marker="*")
                well_pts.append((float(well_r), float(well_e)))
        ax.axhline(0.0, color="0.7", linewidth=0.8)
        _mark_rf(ax, rf, label=f"PET RF ({rf:.0f} Å)")
        ax.set_xlim(min(r_min, 2.2), max(r_max, 12.0))
        scan_name = "O–O" if system != "acetone" else "site–site"
        ax.set_xlabel(f"{scan_name} $r$ (Å)")
        ax.set_title(f"{system} dimer")
        finite = np.concatenate(all_energy) if all_energy else np.array([])
        finite = finite[np.isfinite(finite)]
        y_lo = float(np.min(finite)) - SLICE_Y_WELL_PAD_KCAL if finite.size else -1.0
        y_hi = SLICE_Y_TOP_KCAL
        if finite.size:
            y_hi = max(SLICE_Y_TOP_KCAL, min(float(np.max(finite)) * 1.05, 10.0))
        if well_pts:
            r_e, e_min = min(well_pts, key=lambda p: p[1])
            ax.text(
                0.97,
                0.05,
                f"$r_e$={r_e:.2f} Å\n$E_\\mathrm{{min}}$={e_min:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=11,
                color="0.15",
            )
            y_lo = min(y_lo, min(p[1] for p in well_pts) - SLICE_Y_WELL_PAD_KCAL)
        ax.set_ylim(y_lo, y_hi)
        _add_wall_inset(ax, slices, system, ori_pair, colors)
        ax.legend(frameon=False, loc="upper right", fontsize=10)
    axes[0][0].set_ylabel(r"$E_{\mathrm{int}}$ (kcal/mol)")
    fig.text(
        0.01,
        0.01,
        r"$E_{\mathrm{int}}=E(AB)-E(A)-E(B)$. PET-MAD xs 1.5.0 (PBEsol); "
        r"water-dimer CCSD(T) $\approx-5$ kcal/mol is context only.",
        fontsize=9,
        color="0.35",
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.88, wspace=0.28)
    return _save(fig, Path(output), write_pdf=write_pdf)


def _add_wall_inset(
    ax,
    slices: Mapping[tuple[str, str], Mapping[str, Any]],
    system: str,
    ori_pair: tuple[str, str],
    colors: tuple[str, str],
) -> None:
    """Show the repulsive limb that the well-focused y-limits clip away."""
    inset = inset_axes(ax, width="37%", height="32%", loc="upper left", borderpad=0.6)
    drew = False
    y_hi = 1.0
    for orientation, color, ls in (
        (ori_pair[0], colors[0], "-"),
        (ori_pair[1], colors[1], "--"),
    ):
        row = slices.get((system, orientation))
        if row is None:
            continue
        r = np.asarray(row["r_angstrom"], dtype=np.float64)
        energy = np.asarray(row["e_int_kcal_mol"], dtype=np.float64)
        mask = r <= WALL_INSET_R_MAX_A
        if not np.any(mask):
            continue
        inset.plot(r[mask], energy[mask], color=color, linestyle=ls, marker="o", markersize=2.4)
        finite = energy[mask][np.isfinite(energy[mask])]
        if finite.size:
            y_hi = max(y_hi, float(np.max(finite)))
            drew = True
    if not drew:
        inset.remove()
        return
    inset.axhline(0.0, color="0.7", linewidth=0.6)
    inset.set_xlim(2.15, WALL_INSET_R_MAX_A)
    inset.set_ylim(-0.5, min(max(y_hi * 1.05, 4.0), 40.0))
    inset.set_title("wall", fontsize=8, pad=1)
    inset.tick_params(labelsize=7, length=2)
    inset.set_xticks([2.2, 2.8, 3.3])


def plot_dimer_angular(
    document: Mapping[str, Any],
    output: Path | str,
    *,
    write_pdf: bool = True,
) -> Path:
    """Angular slice at fixed $r_e$: in-plane vs out-of-plane donor–H–acceptor."""
    style = _style()
    palette = comparison_colors(style, n=8)
    colors = {"xz": palette[5], "yz": palette[6]}
    rows = document.get("dimer_angular", [])
    if not rows:
        raise ValueError("document has no dimer_angular")
    systems = list(dict.fromkeys(row["system"] for row in rows))
    fig, axes = plt.subplots(1, len(systems), figsize=(4.5 * len(systems), 3.8), squeeze=False)
    by_key = {(row["system"], row.get("plane", "xz")): row for row in rows}
    for ax, system in zip(axes[0], systems, strict=True):
        r_e = None
        for plane, ls, label in (
            ("xz", "-", "in-plane"),
            ("yz", "--", "out-of-plane"),
        ):
            row = by_key.get((system, plane))
            if row is None:
                continue
            r_e = row["r_angstrom"]
            ax.plot(
                row["theta_deg"],
                row["e_int_kcal_mol"],
                color=colors[plane],
                linestyle=ls,
                marker="o",
                markersize=3.2,
                label=label,
            )
            well_th = row.get("well_theta_deg")
            well_e = row.get("well_kcal_mol")
            if well_th is not None and well_e is not None:
                ax.scatter([well_th], [well_e], color=colors[plane], s=36, zorder=5, marker="*")
        ax.axhline(0.0, color="0.7", linewidth=0.8)
        ax.axvline(180.0, color="0.45", linewidth=0.8, linestyle=":", label="linear OH···O")
        ax.set_xlabel(r"donor–H–acceptor $\theta$ (deg)")
        title = f"{system} at $r_e$"
        if r_e is not None:
            title += f" = {float(r_e):.2f} Å"
        ax.set_title(title)
        ax.legend(frameon=False, loc="best", fontsize=10)
    axes[0][0].set_ylabel(r"$E_{\mathrm{int}}$ (kcal/mol)")
    fig.tight_layout()
    return _save(fig, Path(output), write_pdf=write_pdf)


def plot_dimer_surface(
    document: Mapping[str, Any],
    output: Path | str,
    *,
    index: int = 0,
    write_pdf: bool = True,
) -> Path:
    """Heatmap + isolevels of one 2D ``E_int(r, θ)`` surface with the minimum marked."""
    _style()
    surfaces = document.get("dimer_surfaces", [])
    if not surfaces:
        raise ValueError("document has no dimer_surfaces")
    surface = surfaces[index]
    r = np.asarray(surface["r_angstrom"], dtype=np.float64)
    theta = np.asarray(surface["theta_deg"], dtype=np.float64)
    z = np.asarray(surface["e_int_kcal_mol"], dtype=np.float64)
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        raise ValueError("surface has no finite samples")
    span = min(SURFACE_COLOR_MAX_KCAL, max(float(np.max(np.abs(finite))), 1.0))
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
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
    well_r = surface.get("well_r_angstrom")
    well_th = surface.get("well_theta_deg")
    well_e = surface.get("well_kcal_mol")
    if well_r is not None and well_th is not None:
        ax.scatter([well_r], [well_th], s=70, marker="*", color="0.05", zorder=5, label="min")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$E_{\mathrm{int}}$ (kcal/mol)")
    ax.set_xlabel(r"O–O $r$ (Å)")
    ax.set_ylabel(r"donor–H–acceptor $\theta$ (deg)")
    ax.set_title(f"{surface['system']} dimer $E_\\mathrm{{int}}(r,\\theta)$")
    if well_e is not None and well_r is not None:
        ax.text(
            0.03,
            0.04,
            f"$r_e$={float(well_r):.2f} Å, $\\theta_e$={float(well_th):.0f}°\n"
            f"$E_\\mathrm{{min}}$={float(well_e):.2f} kcal/mol",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            color="0.1",
        )
    fig.tight_layout()
    return _save(fig, Path(output), write_pdf=write_pdf)


def plot_trimer_mbe(
    document: Mapping[str, Any],
    output: Path | str,
    *,
    write_pdf: bool = True,
) -> Path:
    """Trimer ``E_int``, pairwise reconstruction, residual ``E3``, and ``E3/E_int``."""
    style = _style()
    palette = comparison_colors(style, n=8)
    rows = document.get("trimer_slices", [])
    if not rows:
        raise ValueError("document has no trimer_slices")
    systems = list(dict.fromkeys(row["system"] for row in rows))
    n_col = len(systems)
    fig, axes = plt.subplots(2, n_col, figsize=(4.6 * n_col, 6.2), squeeze=False, sharex="col")
    rf = _rf(document)
    energy_style = {
        MOTIF_LINEAR: {
            "e_int": (palette[5], "-"),
            "pair": (palette[1], "--"),
            "e3": (palette[6], "-."),
            "short": "linear",
        },
        MOTIF_CYCLIC: {
            "e_int": (palette[2], "-"),
            "pair": (palette[3], "--"),
            "e3": (palette[7], "-."),
            "short": "cyclic",
        },
    }
    by_sys: dict[str, list[dict[str, Any]]] = {name: [] for name in systems}
    for row in rows:
        by_sys[row["system"]].append(row)
    for col, system in enumerate(systems):
        ax = axes[0][col]
        ax_f = axes[1][col]
        ref_note = None
        for row in by_sys[system]:
            motif = row.get("motif", MOTIF_CYCLIC)
            sty = energy_style.get(motif, energy_style[MOTIF_CYCLIC])
            r = np.asarray(row["r_angstrom"], dtype=np.float64)
            short = sty["short"]
            e_int = np.asarray(row["e_int_kcal_mol"], dtype=np.float64)
            pair = np.asarray(row["e_pair_sum_kcal_mol"], dtype=np.float64)
            e3 = np.asarray(row["e3_kcal_mol"], dtype=np.float64)
            ax.plot(r, _clip_trace(e_int), color=sty["e_int"][0], linestyle="-", marker="o", markersize=3.0, label=rf"$E_\mathrm{{int}}$ {short}")
            ax.plot(r, _clip_trace(pair), color=sty["pair"][0], linestyle="--", marker="o", markersize=3.0, label=rf"$\sum IJ$ {short}")
            ax.plot(r, _clip_trace(e3), color=sty["e3"][0], linestyle="-.", marker="s", markersize=3.0, label=rf"$E_3$ {short}")
            frac = np.asarray(row.get("e3_over_eint", []), dtype=np.float64)
            if frac.size:
                shown = 100.0 * frac
                shown[np.abs(e_int) < 1.0] = np.nan
                shown[np.abs(shown) > 250.0] = np.nan
                ax_f.plot(r, shown, color=sty["e3"][0], linestyle="-.", marker="s", markersize=3.0, label=short)
            if motif == MOTIF_LINEAR or ref_note is None:
                ref_r = row.get("ref_r_angstrom")
                ref_e3 = float(row.get("ref_e3_kcal_mol", np.nan))
                ref_eint = float(row.get("ref_e_int_kcal_mol") or np.nan)
                ref_note = (short, ref_r, ref_e3, ref_eint, sty["e3"][0])
        if ref_note is not None:
            short, ref_r, ref_e3, ref_eint, color = ref_note
            ax.scatter([ref_r], [ref_e3], s=42, marker="D", color=color, zorder=5)
        ax.axhline(0.0, color="0.7", linewidth=0.8)
        ax_f.axhline(0.0, color="0.7", linewidth=0.8)
        _mark_rf(ax, rf)
        _mark_rf(ax_f, rf)
        ax.set_title(f"{system} trimer")
        ax.set_ylabel("Energy (kcal/mol)" if col == 0 else "")
        ax.set_ylim(-16.0, 22.0)
        ax_f.set_xlabel(r"O–O $r$ (Å)")
        ax_f.set_ylabel(r"$E_3/E_{\mathrm{int}}$ (%)" if col == 0 else "")
        ax_f.set_ylim(-150.0, 150.0)
        if ref_note is not None:
            short, ref_r, ref_e3, ref_eint, _color = ref_note
            if abs(ref_e3) >= 1.0:
                extra = "pairwise sum has the wrong sign" if np.isfinite(ref_eint) and ref_eint * ref_e3 < 0 else f"2-body misses $E_3$={ref_e3:.2f}"
                text = f"{short} at {float(ref_r):.2f} Å: $E_3$={ref_e3:.2f} kcal/mol\n{extra}"
            else:
                text = f"{short} at {float(ref_r):.2f} Å: $E_3$={ref_e3:.2f} kcal/mol\nnearly pairwise"
            ax.text(0.97, 0.04, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="0.15")
        ax.legend(frameon=False, loc="upper right", fontsize=7, ncol=2)
        ax_f.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    return _save(fig, Path(output), write_pdf=write_pdf)


def _clip_trace(values: np.ndarray, cap: float = 30.0) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64).copy()
    y[np.abs(y) > cap] = np.nan
    return y


def write_interaction_pes_figures(
    document: Mapping[str, Any],
    output_dir: Path | str,
    *,
    prefix: str = "pet_mad",
    write_pdf: bool = True,
) -> dict[str, Path]:
    """Write campaign figures (PNG, optional PDF) under ``output_dir``."""
    out = Path(output_dir)
    paths = {
        "slices": plot_dimer_slices(
            document, out / f"{prefix}_dimer_slices.png", write_pdf=write_pdf
        ),
        "trimer": plot_trimer_mbe(
            document, out / f"{prefix}_trimer_mbe.png", write_pdf=write_pdf
        ),
    }
    if document.get("dimer_angular"):
        paths["angular"] = plot_dimer_angular(
            document, out / f"{prefix}_dimer_angular.png", write_pdf=write_pdf
        )
    if document.get("dimer_surfaces"):
        paths["surface"] = plot_dimer_surface(
            document, out / f"{prefix}_dimer_surface.png", write_pdf=write_pdf
        )
    return paths
