#!/usr/bin/env python3
"""Plot 2D potential energy surfaces as heatmaps/contour plots.

For each (pair, backend) combination, interpolates the discrete (distance, offset)
grid onto a smooth surface and renders contour + scatter plots. Each figure
combines several linked views of the same pair: a filmstrip of ball-and-stick
geometries along the approach distance (coordinate 1, top), a second filmstrip
along the lateral offset (coordinate 2, below it) — both annotated with small
3D-projected arrows showing which direction each coordinate points on the
actual rotated structure — then the 2D interaction-energy heatmap per backend
(with dashed guide lines tying it back to the distance filmstrip, wrapped to
at most 4 panels per row) and a highlighted render of the located minimum per
backend directly below its heatmap. Optionally overlays a force-arrow field
on the geometries.

Input CSV must have columns:
    molecule_a, molecule_b, distance_angstrom, offset_angstrom,
    energy_kcal_mol, backend

Usage
-----
    python scripts/plot_2d_pes.py --csv results/dimer_scan_campaign/scan_results_new.csv
    python scripts/plot_2d_pes.py --csv foo.csv --backends xtb_gfn2 charmm
    python scripts/plot_2d_pes.py --csv foo.csv --forces-backend xtb
    python scripts/plot_2d_pes.py --csv foo.csv --forces-backend spookynet \\
        --forces-checkpoint examples/sppoky-epoch-0010_params.json
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from scipy.interpolate import PchipInterpolator, griddata
from scipy.ndimage import gaussian_filter

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from mmml.analysis.dimer_molecules import PAIR_SCAN_CONFIG, ORIENTED_MONOMERS
from mmml.analysis.dimer_scans import build_rigid_dimer_2d
from plot_utils import (
    BACKEND_COLORS,
    BACKEND_LABELS,
    MIN_SAFE_CONTACT_ANGSTROM,
    backend_cmap,
    flag_clashing_geometries,
    flag_energy_outliers,
    load_and_enrich,
    ordered_backends,
    render_dimer_atoms,
    robust_color_vmax,
)
from mmml.utils.plotting.styles import apply_plot_style

MAX_COLS = 4

# Figure identifiers used throughout the dimer-scan set; TIP3–TIP3 is (AA).
MONOMER_FIGURE_KEYS = {
    "TIP3": "A",
    "MEOH": "B",
    "ACE": "C",
    "DCM": "D",
    "BENZ": "E",
}

PANEL_LABELS = {
    "ccsd_def2svp_gpu4pyscf_cp": "CCSD/def2-SVP",
    "ccsd_def2svpd_gpu4pyscf_cp": "CCSD/def2-SVPD",
    "mp2_def2svp_gpu4pyscf_cp": "MP2/def2-SVP",
    "hf_def2svp_gpu4pyscf_cp": "HF/def2-SVP",
    "pbe0_def2svp_gpu4pyscf_cp": "PBE0/def2-SVP",
    "pbe0_def2svp_gpu4pyscf_d3bj_cp": "PBE0-D3BJ/def2-SVP",
}

COORD1_COLOR = "0.25"
COORD2_COLOR = "0.25"
MIN_TEXT_SIZE = 10.0
PANEL_TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 11
COLORBAR_LABEL_FONTSIZE = max(MIN_TEXT_SIZE, 10)
MAX_LEGEND_TEXT_SIZE = 16.0
MAX_LEGEND_COLUMNS = 8
MAX_COMPACT_LEGEND_ROWS = 2


def _pair_cfg(label_a: str, label_b: str) -> dict | None:
    pair = (label_a, label_b)
    if pair not in PAIR_SCAN_CONFIG:
        pair = (label_b, label_a)
        if pair not in PAIR_SCAN_CONFIG:
            return None
    return PAIR_SCAN_CONFIG[pair]


def _coord_axes_for(label_a: str, label_b: str) -> list[tuple[np.ndarray, str, str]] | None:
    """Direction vectors for the d and c scan coordinates, for the arrow overlay."""
    cfg = _pair_cfg(label_a, label_b)
    if cfg is None:
        return None
    return [
        (np.array([0.0, 0.0, 1.0]), COORD1_COLOR, "d"),
        (np.array(cfg["transverse_axis"], dtype=float), COORD2_COLOR, "c"),
    ]


def _atoms_snapshot(
    label_a: str, label_b: str, distance: float, offset: float
) -> tuple[object, tuple[np.ndarray, np.ndarray]] | tuple[None, None]:
    """Build an ASE Atoms object (+ fragment index arrays) for a given geometry."""
    pair = (label_a, label_b)
    if pair not in ORIENTED_MONOMERS:
        pair = (label_b, label_a)
        if pair not in ORIENTED_MONOMERS:
            return None, None
    monomers = ORIENTED_MONOMERS[pair]
    cfg = PAIR_SCAN_CONFIG[pair]
    atoms, fragments = build_rigid_dimer_2d(
        monomers["a"],
        monomers["b"],
        distance_angstrom=distance,
        offset_angstrom=offset,
        axis=(0, 0, 1),
        transverse_axis=cfg["transverse_axis"],
        center="none",
    )
    return atoms, fragments


def _forces_for(atoms, calc) -> np.ndarray | None:
    """Compute forces for *atoms* with *calc*, returning None on failure."""
    if atoms is None or calc is None:
        return None
    try:
        probe = atoms.copy()
        probe.calc = calc
        return np.asarray(probe.get_forces())
    except Exception as e:
        print(f"    Warning: force evaluation failed: {e}")
        return None


def _plot_distance_filmstrip(
    ax, label_a: str, label_b: str, distances: np.ndarray, n_snap: int = 5,
    forces_calc=None, forces_label: str | None = None,
) -> list[float]:
    """Row of ball-and-stick snapshots spanning the scanned distances (coordinate 1, offset=0).

    Returns the list of distances actually rendered, so callers can draw
    matching guide lines on the heatmap panels below, and a representative
    fixed distance can be picked for the offset filmstrip.
    """
    ax.set_axis_off()
    coord_axes = _coord_axes_for(label_a, label_b)
    dist_sorted = np.sort(np.unique(distances))
    if len(dist_sorted) == 0:
        return []
    n_snap = min(n_snap, len(dist_sorted))
    idx = np.linspace(0, len(dist_sorted) - 1, n_snap).round().astype(int)
    snap_distances = [dist_sorted[i] for i in sorted(set(idx))]
    n_snap = len(snap_distances)
    for j, d in enumerate(snap_distances):
        inset = ax.inset_axes([j / n_snap, 0.0, 1 / n_snap, 1.0])
        atoms_snap, fragments = _atoms_snapshot(label_a, label_b, d, 0.0)
        forces = _forces_for(atoms_snap, forces_calc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_dimer_atoms(
                inset, atoms_snap, fragments,
                forces=forces, coord_axes=coord_axes, title=rf"$d={d:.2f}$ Å",
            )
    if forces_calc is not None and forces_label:
        ax.text(
            0.5, -0.05, f"Force field: {forces_label}",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color="crimson",
        )
    return snap_distances


def _plot_offset_filmstrip(
    ax, label_a: str, label_b: str, offsets: np.ndarray, fixed_distance: float, n_snap: int = 5,
    forces_calc=None,
) -> list[float]:
    """Vertical column of ball-and-stick snapshots spanning the scanned offsets
    (coordinate 2, fixed distance) — stacked bottom-to-top so offset=0 sits at
    the bottom, matching the heatmap panels' y-axis orientation directly below.
    """
    ax.set_axis_off()
    coord_axes = _coord_axes_for(label_a, label_b)
    off_sorted = np.sort(np.unique(offsets))
    if len(off_sorted) == 0:
        return []
    n_snap = min(n_snap, len(off_sorted))
    idx = np.linspace(0, len(off_sorted) - 1, n_snap).round().astype(int)
    snap_offsets = [off_sorted[i] for i in sorted(set(idx))]
    n_snap = len(snap_offsets)
    for j, off in enumerate(snap_offsets):
        inset = ax.inset_axes([0.0, j / n_snap, 1.0, 1 / n_snap])
        atoms_snap, fragments = _atoms_snapshot(label_a, label_b, fixed_distance, off)
        forces = _forces_for(atoms_snap, forces_calc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_dimer_atoms(
                inset, atoms_snap, fragments,
                forces=forces, coord_axes=coord_axes, title=rf"$c={off:.2f}$ Å", title_fontsize=6,
            )
    return snap_offsets


def _inset_bounds_for(
    ax, min_d: float, min_o: float, size: float = 0.30, gap: float = 0.03,
) -> tuple[float, float, float, float]:
    """Axes-fraction (x0, y0, w, h) for a minimum-geometry inset near (min_d, min_o).

    Placed just off whichever side of the data point has room, so the inset
    sits visually "at" the minimum without covering the star marker itself,
    and clamped to stay fully inside the axes regardless of how close the
    minimum is to an edge.
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_frac = (min_d - xlim[0]) / (xlim[1] - xlim[0]) if xlim[1] != xlim[0] else 0.5
    y_frac = (min_o - ylim[0]) / (ylim[1] - ylim[0]) if ylim[1] != ylim[0] else 0.5
    x0 = x_frac + gap if x_frac < 0.5 else x_frac - size - gap
    y0 = y_frac + gap if y_frac < 0.5 else y_frac - size - gap
    x0 = min(max(x0, 0.0), 1.0 - size)
    y0 = min(max(y0, 0.0), 1.0 - size)
    return x0, y0, size, size


def _clean_interaction_data(df_pair: pd.DataFrame, backend: str, min_contact: float) -> pd.DataFrame:
    """Return a consistently referenced, clash/outlier-filtered PES."""
    df_be = df_pair[df_pair["backend"] == backend].copy()
    if df_be.empty:
        return df_be
    df_be = flag_clashing_geometries(df_be, min_contact=min_contact)
    df_be = df_be[~df_be["is_clash"]].copy()
    if df_be.empty:
        return df_be
    df_on = df_be[df_be["offset_angstrom"] == df_be["offset_angstrom"].min()]
    ref_source = df_on if not df_on.empty else df_be
    ref = ref_source.sort_values("distance_angstrom")["energy_kcal_mol"].iloc[-1]
    df_be["E_int"] = df_be["energy_kcal_mol"] - ref
    df_be = flag_energy_outliers(df_be, "E_int")
    df_be = df_be[~df_be["is_energy_outlier"]].copy()
    return df_be


def _plot_summary_panel(
    ax, df_pair: pd.DataFrame, backends: list[str], min_contact: float
) -> tuple[list, list[str]]:
    """Compare smooth distance curves across all lateral offsets."""
    line_styles = ["-", "--", ":", "-.", (0, (5, 1, 1, 1))]
    y_values = []
    x_values = []
    for backend in backends:
        df_be = _clean_interaction_data(df_pair, backend, min_contact)
        if df_be.empty:
            continue
        color = BACKEND_COLORS.get(backend)
        offsets = np.sort(df_be["offset_angstrom"].unique())
        # At most five representative lateral cuts keep the panel readable.
        offset_indices = np.unique(np.linspace(0, len(offsets) - 1, min(5, len(offsets))).round().astype(int))
        for style_idx, offset_idx in enumerate(offset_indices):
            curve = df_be[df_be["offset_angstrom"] == offsets[offset_idx]].sort_values("distance_angstrom")
            x = curve["distance_angstrom"].to_numpy()
            y = curve["E_int"].to_numpy()
            if len(x) < 2:
                continue
            x_smooth = np.linspace(x.min(), x.max(), 180)
            y_smooth = PchipInterpolator(x, y)(x_smooth) if len(x) >= 3 else np.interp(x_smooth, x, y)
            ax.plot(
                x_smooth, y_smooth, color=color, lw=1.4, alpha=0.9,
                ls=line_styles[style_idx],
                label=BACKEND_LABELS.get(backend, backend) if style_idx == 0 else None,
            )
            y_values.append(y_smooth)
            x_values.append(x_smooth)
    if y_values:
        values = np.concatenate(y_values)
        low = min(float(np.nanmin(values)), -1.0)
        high = max(float(np.nanmax(values)), 2.0)
        ax.set_ylim(low * 1.12, high * 1.12)
        # A symlog axis keeps the modest wells readable while retaining the
        # very steep walls of unstable methods in the same comparison.
        ax.set_yscale("symlog", linthresh=5.0)
    if x_values:
        xmin = min(float(x.min()) for x in x_values)
        xmax = max(float(x.max()) for x in x_values)
        ax.set_xlim(xmin, xmax + 0.55 * (xmax - xmin))
    ax.axhline(0.0, color="0.35", lw=0.7, zorder=0)
    ax.set_title("Summary — repulsive walls", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel(r"$d$ / Å", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("$E_{int}$ / kcal mol$^{-1}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    return ax.get_legend_handles_labels()


def _legend_width_points(labels: list[str], ncol: int, fontsize: float) -> float:
    """Estimated Matplotlib legend width, including handles and column gaps."""
    prop = FontProperties(size=fontsize)
    text_widths = [TextPath((0, 0), label, prop=prop).get_extents().width for label in labels]
    # Matplotlib fills legend entries row-wise.  A column needs room for its
    # widest label, its line handle, and the gap between them.
    column_widths = [
        max(text_widths[column::ncol], default=0.0) + fontsize * (1.6 + 0.45)
        for column in range(ncol)
    ]
    return sum(column_widths) + fontsize * 0.65 * max(ncol - 1, 0)


def _summary_legend_layout(fig, labels: list[str]) -> tuple[int, float]:
    """Maximise legend type while fitting the measured labels in few rows.

    The legend sits below the complete figure, so it can use the full canvas
    width.  Prefer at most two rows; only fall back to more rows when the
    minimum readable font cannot fit.
    """
    n_entries = len(labels)
    if n_entries == 0:
        return 1, MIN_TEXT_SIZE
    available_width = fig.get_size_inches()[0] * 72.0 * 0.93
    max_columns = min(MAX_LEGEND_COLUMNS, n_entries)
    for max_rows in range(1, n_entries + 1):
        if max_rows > MAX_COMPACT_LEGEND_ROWS and max_rows > 3:
            break
        min_columns = int(np.ceil(n_entries / max_rows))
        if min_columns > max_columns:
            continue
        for fontsize in np.arange(MAX_LEGEND_TEXT_SIZE, MIN_TEXT_SIZE - 0.01, -0.25):
            viable_columns = [
                ncol for ncol in range(min_columns, max_columns + 1)
                if _legend_width_points(labels, ncol, float(fontsize)) <= available_width
            ]
            if viable_columns:
                # More columns means fewer rows at the same maximised font.
                return max(viable_columns), float(fontsize)
    return max_columns, MIN_TEXT_SIZE


def _panel_label(backend: str) -> str:
    """Readable, compact label for a PES panel and its minimum annotation."""
    tuned_labels = {
        "spookynet_muon_ep7": "SpookyNet (Muon e7)",
        "spookynet_hybrid_muon_ep7": "Hybrid (Muon e7)",
        "spookynet_mbdzbl_ep2": "MBD+ZBL (e2)",
        "spookynet_hybrid_mbdzbl_ep2": "Hybrid (MBD+ZBL e2)",
        "spookynet_hybrid_step3000": "SpookyNet (hybrid train s3000)",
        "spookynet_hybrid_hybrid_step3000": "Hybrid decomposition (s3000)",
        "spookynet_hybrid_step3000_mbd": "Hybrid s3000 + MBD",
        "spookynet_hybrid_muon_epoch1_mbd": "Muon e1 + MBD",
        "spookynet_hybrid_step19800": "Hybrid s19800",
        "spookynet_hybrid_step19800_mbd": "Hybrid s19800 + MBD",
        "spookynet_hybrid_step36800": "Hybrid s36800",
        "spookynet_hybrid_step36800_mbd": "Hybrid s36800 + MBD",
    }
    return PANEL_LABELS.get(backend, tuned_labels.get(backend, BACKEND_LABELS.get(backend, backend)))


def _reference_energy(series: pd.Series, ref_dist: float | None = None) -> pd.Series:
    """Subtract the energy at the largest distance (or explicit *ref_dist*)."""
    if ref_dist is not None:
        idx = (series.index - ref_dist).abs().argmin()
        ref = series.iloc[idx]
    else:
        ref = series.iloc[-1]
    return series - ref


def plot_2d_pes_for_pair(
    df_pair: pd.DataFrame,
    label_a: str,
    label_b: str,
    backends: list[str],
    out_dir: Path,
    n_grid: int = 80,
    energy_clip_kcal: float | None = None,
    min_contact: float = MIN_SAFE_CONTACT_ANGSTROM,
    show_atoms: bool = True,
    forces_calc=None,
    forces_label: str | None = None,
    max_cols: int = MAX_COLS,
) -> None:
    """Render a 2D PES heatmap for one (pair, backend) set."""
    n_be = len(backends)
    if n_be == 0:
        return

    # Reserve the first panel for the cross-method summary.  With the standard
    # 11 backends this deliberately makes a 3 × 4 grid: model families first,
    # followed by SpookyNet/classical methods, then the reference methods.
    n_panels = n_be + 1
    n_cols = min(n_panels, max_cols)
    n_be_rows = -(-n_panels // n_cols)  # ceil

    film_h = 1.1   # distance filmstrip row height (matches the heatmaps' x-axis, across the top)
    film_w = 1.3   # offset filmstrip column width (matches the heatmaps' y-axis, down the side)
    heat_h = 4.5
    heat_w = 5.5

    if show_atoms:
        height_ratios = [film_h] + [heat_h] * n_be_rows
        width_ratios = [film_w] + [heat_w] * n_cols
        fig = plt.figure(
            figsize=(sum(width_ratios), sum(height_ratios)), constrained_layout=True,
        )
        gs = fig.add_gridspec(
            len(height_ratios), len(width_ratios),
            height_ratios=height_ratios, width_ratios=width_ratios,
        )
        ax_corner = fig.add_subplot(gs[0, 0])
        ax_corner.set_axis_off()
        ax_film_dist = fig.add_subplot(gs[0, 1:])
        ax_film_off = fig.add_subplot(gs[1:, 0])
        row_offset, col_offset = 1, 1
    else:
        height_ratios = [heat_h] * n_be_rows
        fig = plt.figure(figsize=(heat_w * n_cols, sum(height_ratios)), constrained_layout=True)
        gs = fig.add_gridspec(n_be_rows, n_cols, height_ratios=height_ratios)
        ax_film_dist = ax_film_off = None
        row_offset, col_offset = 0, 0

    pair_tag = f"{label_a}_{label_b}"
    pair_key = f"{MONOMER_FIGURE_KEYS.get(label_a, label_a[0])}{MONOMER_FIGURE_KEYS.get(label_b, label_b[0])}"
    fig.suptitle(
        f"({pair_key}) 2D PES: {label_a} + {label_b}",
        x=0.015, y=0.99, ha="left", fontsize=15, fontweight="bold",
    )

    snap_distances: list[float] = []
    if ax_film_dist is not None:
        all_distances = df_pair["distance_angstrom"].to_numpy()
        snap_distances = _plot_distance_filmstrip(
            ax_film_dist, label_a, label_b, all_distances,
            forces_calc=forces_calc, forces_label=forces_label,
        )
    if ax_film_off is not None:
        all_offsets = df_pair["offset_angstrom"].to_numpy()
        fixed_distance = snap_distances[len(snap_distances) // 2] if snap_distances else float(
            df_pair["distance_angstrom"].median()
        )
        _plot_offset_filmstrip(
            ax_film_off, label_a, label_b, all_offsets, fixed_distance,
            forces_calc=forces_calc,
        )

    summary_ax = fig.add_subplot(gs[row_offset, col_offset])
    summary_handles, summary_labels = _plot_summary_panel(
        summary_ax, df_pair, backends, min_contact
    )

    for i, backend in enumerate(backends):
        panel_index = i + 1
        row_block = panel_index // n_cols
        col = panel_index % n_cols
        heat_row = row_offset + row_block
        heat_col = col_offset + col
        ax = fig.add_subplot(gs[heat_row, heat_col])

        df_be = df_pair[df_pair["backend"] == backend].copy()
        if df_be.empty:
            ax.set_visible(False)
            continue

        df_be = flag_clashing_geometries(df_be, min_contact=min_contact)
        n_clash = int(df_be["is_clash"].sum())
        df_clash = df_be[df_be["is_clash"]]
        df_be = df_be[~df_be["is_clash"]].copy()
        if df_be.empty:
            ax.set_visible(False)
            continue

        # Reference energy at largest distance, offset=0 (on-axis)
        df_on = df_be[df_be["offset_angstrom"] == df_be["offset_angstrom"].min()]
        if not df_on.empty:
            ref = df_on.sort_values("distance_angstrom")["energy_kcal_mol"].iloc[-1]
        else:
            ref = df_be["energy_kcal_mol"].max()

        df_be["E_int"] = df_be["energy_kcal_mol"] - ref

        # A fixed geometric contact cutoff can still miss backend-specific
        # energetic blow-ups; catch those directly from E_int via a robust
        # (MAD-based) outlier test and exclude them the same way.
        df_be = flag_energy_outliers(df_be, "E_int")
        n_outlier = int(df_be["is_energy_outlier"].sum())
        df_outlier = df_be[df_be["is_energy_outlier"]]
        df_be = df_be[~df_be["is_energy_outlier"]]
        if df_be.empty:
            ax.set_visible(False)
            continue

        # Colour range from the *clean* raw scatter (not the clipped/
        # interpolated grid, whose spline can amplify a residual repulsive
        # wall): a percentile of THIS backend's own distribution, so a
        # backend with a genuinely much deeper/shallower well (e.g. an
        # undertrained model that's 4x overbound) gets its own appropriate
        # scale instead of being flattened by a fixed global ceiling.
        # --energy-clip is an optional hard cap for when you *do* want a
        # fixed, comparable scale across backends; default is fully
        # data-driven per backend.
        vmax = robust_color_vmax(df_be["E_int"].to_numpy(), ceiling=energy_clip_kcal)
        # Clip bound for interpolation stability only — kept a bit above vmax
        # so genuine (non-outlier) structure near the edge of the colour
        # range isn't itself washed out by the clip.
        clip_bound = max(vmax * 2.0, 1.0)
        if energy_clip_kcal is not None:
            clip_bound = min(energy_clip_kcal, clip_bound)

        min_d = min_o = min_e = None
        # Clash + outlier removal can leave the surface too sparse/degenerate
        # to interpolate meaningfully (e.g. a pair whose safe contact
        # distance is far outside most of the scanned range) — fall back to
        # a plain scatter rather than fitting a surface to a handful of points.
        n_excluded_total = n_clash + n_outlier
        # One shared, linear (sequential, non-diverging) colormap per model
        # family (ML / ab initio / empirical reference) — see backend_cmap
        # in plot_utils.py. A linear norm (not TwoSlopeNorm) means equal
        # energy differences map to equal colour differences throughout,
        # with no special white-centred treatment of E_int=0.
        panel_cmap = backend_cmap(backend)
        excluded = pd.concat([df_clash, df_outlier], ignore_index=True)
        surface_df = df_be.copy()
        if not excluded.empty:
            excluded = excluded.copy()
            excluded["E_int"] = vmax
            surface_df = pd.concat([surface_df, excluded], ignore_index=True)
        dist_vals = np.sort(surface_df["distance_angstrom"].unique())
        off_vals  = np.sort(surface_df["offset_angstrom"].unique())
        sparse_data = len(dist_vals) < 3 or len(off_vals) < 2 or len(surface_df) < 6

        if sparse_data:
            norm = Normalize(vmin=-vmax, vmax=vmax)
            sc = ax.scatter(
                df_be["distance_angstrom"],
                df_be["offset_angstrom"],
                c=df_be["E_int"],
                cmap=panel_cmap,
                norm=norm,
                s=80,
                edgecolors="k",
                linewidths=0.4,
            )
            plt.colorbar(sc, ax=ax, label="$E_{int}$ / kcal mol$^{-1}$")
            if n_excluded_total:
                ax.text(
                    0.5, 1.02,
                    f"only {len(df_be)} clean points after removing {n_excluded_total} "
                    "clashing/outlier geometries — showing raw points, no surface fit",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=6, color="firebrick",
                )
            if not df_be.empty:
                best = df_be.loc[df_be["E_int"].idxmin()]
                min_d, min_o, min_e = best["distance_angstrom"], best["offset_angstrom"], best["E_int"]
        else:
            # Interpolate directly from the clean scattered (distance, offset,
            # E_int) points onto a regular grid. Using scattered-data
            # interpolation (rather than pivoting to a rectangular grid first)
            # is essential here: dropping clash/outlier rows leaves holes in
            # what would otherwise be a rectangular grid, and a rectangular-
            # grid spline can't handle missing cells.
            points = surface_df[["distance_angstrom", "offset_angstrom"]].to_numpy()
            values = surface_df["E_int"].to_numpy()
            D_fine = np.linspace(dist_vals.min(), dist_vals.max(), n_grid)
            O_fine = np.linspace(off_vals.min(), off_vals.max(), n_grid)
            Dg, Og = np.meshgrid(D_fine, O_fine)

            try:
                Z_fine = griddata(points, values, (Dg, Og), method="linear")
                if np.all(np.isnan(Z_fine)):
                    raise ValueError("linear interpolation returned no valid surface")
            except Exception:
                Z_fine = np.full(Dg.shape, np.nan)
            # A light, mask-aware Gaussian pass softens the piecewise-linear
            # facets without leaking values across regions outside the sampled
            # surface.
            valid = np.isfinite(Z_fine)
            if np.any(valid):
                weights = gaussian_filter(valid.astype(float), sigma=0.8)
                blurred = gaussian_filter(np.where(valid, Z_fine, 0.0), sigma=0.8)
                Z_fine = np.where(weights > 1e-8, blurred / weights, np.nan)
            # Cells outside the convex hull of the *clean* points come back as
            # NaN — deliberately left as gaps rather than flat-filled by
            # nearest-neighbour extrapolation. A flat fill previously let the
            # global-minimum search land in a region with zero real data
            # (e.g. right where the clash/outlier filters had stripped every
            # point), reporting a fabricated minimum at an excluded geometry.
            # Keep the physical interpolated surface for locating/reporting the
            # minimum.  Clipping is a display-only operation; using the clipped
            # array made deep wells report exactly the colour-scale floor.
            Z_physical = Z_fine.copy()
            Z_fine = np.clip(Z_fine, -clip_bound, clip_bound)

            norm = Normalize(vmin=-vmax, vmax=vmax)

            cmap = panel_cmap
            im = ax.contourf(
                D_fine, O_fine, Z_fine,
                levels=30,
                cmap=cmap,
                norm=norm,
            )
            ax.contour(
                D_fine, O_fine, Z_fine,
                levels=[-2, -1, -0.5, -0.2, 0],
                colors="k",
                linewidths=0.6,
                linestyles="--",
                alpha=0.6,
            )
            cb = plt.colorbar(im, ax=ax, shrink=0.85)
            cb.set_label("$E_{int}$ / kcal mol$^{-1}$", fontsize=COLORBAR_LABEL_FONTSIZE)
            cb.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

            # Report and mark a real sampled geometry.  The smoothed surface is
            # presentation-only and can attenuate a narrow well or move its
            # apparent minimum between evaluated configurations.
            if df_be.empty:
                min_d = min_o = min_e = None
            else:
                best = df_be.loc[df_be["E_int"].idxmin()]
                min_d = float(best["distance_angstrom"])
                min_o = float(best["offset_angstrom"])
                min_e = float(best["E_int"])

        # Guide lines linking this heatmap back to the filmstrip snapshots above
        for d in snap_distances:
            ax.axvline(d, color="k", lw=0.5, ls=":", alpha=0.35, zorder=0)

        # A minimum can lie exactly on c=0.  Nudge its marker into the surface
        # so the full star remains visible rather than being clipped by the axis.
        if min_d is not None and min_o is not None and np.isfinite(min_e):
            ylo, yhi = ax.get_ylim()
            star_c = max(float(min_o), ylo + 0.045 * (yhi - ylo))
            ax.plot(min_d, star_c, "*", color="gold", markersize=13,
                    markeredgecolor="k", markeredgewidth=0.5, zorder=5, clip_on=False)

        ax.set_xlabel(r"$d$ / Å", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(r"$c$ / Å", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
        panel_title = _panel_label(backend)
        if min_e is not None and np.isfinite(min_e):
            panel_title += "\n" + rf"$E_{{\mathrm{{min}}}}={min_e:.2f}$ kcal mol$^{{-1}}$"
        ax.set_title(
            panel_title,
            loc="left",
            fontsize=PANEL_TITLE_FONTSIZE,
            fontweight="bold",
        )

        # Highlighted render of the located minimum, overlaid directly on the
        # surface as a small inset anchored near its actual (d, offset)
        # position — not a separate panel — with a dashed line connecting it
        # back to the gold star marker.
        if show_atoms and min_d is not None and np.isfinite(min_e):
            atoms_min, fragments_min = _atoms_snapshot(label_a, label_b, float(min_d), float(min_o))
            forces_min = _forces_for(atoms_min, forces_calc)
            x0, y0, w, h = _inset_bounds_for(ax, float(min_d), float(min_o))
            ax_min = ax.inset_axes([x0, y0, w, h])
            ax_min.patch.set_alpha(0.85)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                render_dimer_atoms(
                    ax_min, atoms_min, fragments_min, forces=forces_min,
                    coord_axes=_coord_axes_for(label_a, label_b),
                    title="", title_fontsize=9,
                )
            ax.annotate(
                "", xy=(min_d, min_o), xycoords="data",
                xytext=(x0 + w / 2, y0 + h / 2), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.7, ls="dashed", alpha=0.7),
                zorder=4,
            )

    if summary_handles:
        ncol, fontsize = _summary_legend_layout(fig, summary_labels)
        fig.legend(
            summary_handles,
            summary_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.018),
            ncol=ncol,
            fontsize=fontsize,
            frameon=False,
            handlelength=1.6,
            handletextpad=0.45,
            columnspacing=0.65,
            labelspacing=0.45,
        )

    out_path = out_dir / f"{pair_tag}_2d_pes.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _build_forces_calc(backend: str | None, checkpoint: Path | None):
    if backend is None or backend == "none":
        return None, None
    if backend == "xtb":
        from mmml.analysis.dimer_scans import make_xtb_calculator
        return make_xtb_calculator(method="GFN2-xTB"), "GFN2-xTB"
    if backend == "spookynet":
        if checkpoint is None:
            raise ValueError("--forces-checkpoint is required for --forces-backend spookynet")
        from mmml.models.spookynet_calc import SpookyNetCalculator
        return SpookyNetCalculator(checkpoint=checkpoint), "SpookyNet"
    raise ValueError(f"Unknown forces backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Input scan CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dimer_scan_campaign/pes_2d"),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Backends to plot (default: all found)",
    )
    parser.add_argument(
        "--energy-clip", type=float, default=None,
        help=(
            "Optional hard cap (±N kcal/mol) on the colour scale. Default: fully "
            "data-driven per backend (85th percentile of that backend's own clean "
            "E_int distribution) — set this only if you want a fixed, directly "
            "comparable scale across backends instead."
        ),
    )
    parser.add_argument(
        "--n-grid", type=int, default=80,
        help="Interpolation grid resolution (default 80)",
    )
    parser.add_argument(
        "--min-contact", type=float, default=MIN_SAFE_CONTACT_ANGSTROM,
        help=(
            "Exclude geometries whose fragment atoms are closer than this "
            f"(Å) from the colour scale/interpolation (default {MIN_SAFE_CONTACT_ANGSTROM}). "
            "distance_angstrom is an anchor-to-anchor separation, not atom-atom, "
            "so nominally 'close' scan points can have overlapping atoms."
        ),
    )
    parser.add_argument(
        "--no-atoms",
        action="store_true",
        help="Skip the ASE atoms filmstrips + minimum-geometry panels",
    )
    parser.add_argument(
        "--max-cols", type=int, default=MAX_COLS,
        help=f"Maximum backend panels per row before wrapping to a new row (default {MAX_COLS})",
    )
    parser.add_argument(
        "--forces-backend",
        choices=["none", "xtb", "spookynet"],
        default="none",
        help="Overlay a force-arrow field on the rendered geometries using this calculator",
    )
    parser.add_argument(
        "--forces-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path for --forces-backend spookynet",
    )
    args = parser.parse_args()
    apply_plot_style("icml")
    mpl.rcParams["text.usetex"] = False

    df = load_and_enrich(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forces_calc, forces_label = _build_forces_calc(args.forces_backend, args.forces_checkpoint)

    pairs = df[["molecule_a", "molecule_b"]].drop_duplicates().values
    print(f"Plotting 2D PES for {len(pairs)} pairs...")

    for label_a, label_b in pairs:
        df_pair = df[(df["molecule_a"] == label_a) & (df["molecule_b"] == label_b)]
        pair_backends = ordered_backends(df_pair, args.backends)
        plot_2d_pes_for_pair(
            df_pair, label_a, label_b, pair_backends, args.output_dir,
            n_grid=args.n_grid, energy_clip_kcal=args.energy_clip,
            min_contact=args.min_contact, show_atoms=not args.no_atoms,
            forces_calc=forces_calc, forces_label=forces_label,
            max_cols=args.max_cols,
        )

    print(f"\nAll 2D PES plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
