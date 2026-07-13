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
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from mmml.analysis.dimer_molecules import PAIR_SCAN_CONFIG, ORIENTED_MONOMERS
from mmml.analysis.dimer_scans import build_rigid_dimer_2d
from plot_utils import (
    BACKEND_CMAPS,
    BACKEND_LABELS,
    MIN_SAFE_CONTACT_ANGSTROM,
    flag_clashing_geometries,
    flag_energy_outliers,
    load_and_enrich,
    ordered_backends,
    render_dimer_atoms,
    robust_color_vmax,
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "figure.dpi": 150,
        "text.usetex": False,
    }
)

MAX_COLS = 4

COORD1_COLOR = "#d62728"  # distance (approach axis)
COORD2_COLOR = "#1f77b4"  # lateral offset (transverse axis)


def _pair_cfg(label_a: str, label_b: str) -> dict | None:
    pair = (label_a, label_b)
    if pair not in PAIR_SCAN_CONFIG:
        pair = (label_b, label_a)
        if pair not in PAIR_SCAN_CONFIG:
            return None
    return PAIR_SCAN_CONFIG[pair]


def _coord_axes_for(label_a: str, label_b: str) -> list[tuple[np.ndarray, str, str]] | None:
    """Direction vectors for coordinate 1 (distance) and 2 (offset), for the arrow overlay."""
    cfg = _pair_cfg(label_a, label_b)
    if cfg is None:
        return None
    return [
        (np.array([0.0, 0.0, 1.0]), COORD1_COLOR, "d"),
        (np.array(cfg["transverse_axis"], dtype=float), COORD2_COLOR, "off"),
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
                forces=forces, coord_axes=coord_axes, title=f"{d:.2f} Å",
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
                forces=forces, coord_axes=coord_axes, title=f"off={off:.2f} Å", title_fontsize=6,
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
    energy_clip_kcal: float = 5.0,
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

    n_cols = min(n_be, max_cols)
    n_be_rows = -(-n_be // n_cols)  # ceil

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
    fig.suptitle(
        f"2D PES: {label_a} + {label_b}",
        fontsize=13,
        fontweight="bold",
    )

    snap_distances: list[float] = []
    if ax_film_dist is not None:
        all_distances = df_pair["distance_angstrom"].to_numpy()
        snap_distances = _plot_distance_filmstrip(
            ax_film_dist, label_a, label_b, all_distances,
            forces_calc=forces_calc, forces_label=forces_label,
        )
        ax_film_dist.set_title(
            f"Coordinate 1 — approach distance (offset=0)  [arrow: d]",
            fontsize=9, color=COORD1_COLOR,
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
        ax_film_off.text(
            -0.28, 0.5,
            f"Coordinate 2 — lateral offset (fixed d={fixed_distance:.2f} Å)  [arrow: off]",
            transform=ax_film_off.transAxes, rotation=90, ha="center", va="center",
            fontsize=9, color=COORD2_COLOR,
        )

    for i, backend in enumerate(backends):
        row_block = i // n_cols
        col = i % n_cols
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
        # wall): a percentile so a handful of remaining steep points can't
        # blow out the whole scale, with a floor for near-flat surfaces.
        vmax = robust_color_vmax(df_be["E_int"].to_numpy(), ceiling=energy_clip_kcal)
        # Clip bound for interpolation stability only — kept a bit above vmax
        # so genuine (non-outlier) structure near the edge of the colour
        # range isn't itself washed out by the clip.
        clip_bound = min(energy_clip_kcal, max(vmax * 2.0, 1.0))

        dist_vals = np.sort(df_be["distance_angstrom"].unique())
        off_vals  = np.sort(df_be["offset_angstrom"].unique())

        min_d = min_o = min_e = None
        # Clash + outlier removal can leave the surface too sparse/degenerate
        # to interpolate meaningfully (e.g. a pair whose safe contact
        # distance is far outside most of the scanned range) — fall back to
        # a plain scatter rather than fitting a surface to a handful of points.
        n_excluded_total = n_clash + n_outlier
        sparse_data = len(dist_vals) < 3 or len(off_vals) < 2 or len(df_be) < 6

        if sparse_data:
            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
            sc = ax.scatter(
                df_be["distance_angstrom"],
                df_be["offset_angstrom"],
                c=df_be["E_int"],
                cmap=BACKEND_CMAPS.get(backend, "RdBu_r"),
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
                ax.plot(
                    min_d, min_o,
                    "*", color="gold", markersize=14, markeredgecolor="k", markeredgewidth=0.5,
                    label=f"min {min_e:.2f} kcal/mol", zorder=5,
                )
                ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
        else:
            # Interpolate directly from the clean scattered (distance, offset,
            # E_int) points onto a regular grid. Using scattered-data
            # interpolation (rather than pivoting to a rectangular grid first)
            # is essential here: dropping clash/outlier rows leaves holes in
            # what would otherwise be a rectangular grid, and a rectangular-
            # grid spline can't handle missing cells.
            points = df_be[["distance_angstrom", "offset_angstrom"]].to_numpy()
            values = df_be["E_int"].to_numpy()
            D_fine = np.linspace(dist_vals.min(), dist_vals.max(), n_grid)
            O_fine = np.linspace(off_vals.min(), off_vals.max(), n_grid)
            Dg, Og = np.meshgrid(D_fine, O_fine)

            try:
                Z_fine = griddata(points, values, (Dg, Og), method="linear")
            except Exception:
                Z_fine = np.full(Dg.shape, np.nan)
            # Cells outside the convex hull of the *clean* points come back as
            # NaN — deliberately left as gaps rather than flat-filled by
            # nearest-neighbour extrapolation. A flat fill previously let the
            # global-minimum search land in a region with zero real data
            # (e.g. right where the clash/outlier filters had stripped every
            # point), reporting a fabricated minimum at an excluded geometry.
            Z_fine = np.clip(Z_fine, -clip_bound, clip_bound)

            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

            cmap = BACKEND_CMAPS.get(backend, "RdBu_r")
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
            cb.set_label("$E_{int}$ / kcal mol$^{-1}$", fontsize=9)

            # Mark minimum (NaN-aware: gaps outside the clean-data hull must
            # never win the search)
            if np.all(np.isnan(Z_fine)):
                min_idx = None
            else:
                min_idx = np.unravel_index(np.nanargmin(Z_fine), Z_fine.shape)
            if min_idx is None:
                min_d = min_o = min_e = None
            else:
                min_d, min_o, min_e = D_fine[min_idx[1]], O_fine[min_idx[0]], Z_fine[min_idx]
                ax.plot(
                    min_d, min_o,
                    "*", color="gold", markersize=14, markeredgecolor="k", markeredgewidth=0.5,
                    label=f"min {min_e:.2f} kcal/mol",
                    zorder=5,
                )
                ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

            # Data point dots
            ax.scatter(
                df_be["distance_angstrom"],
                df_be["offset_angstrom"],
                c="w", s=15, edgecolors="k", linewidths=0.3, zorder=4,
            )

        # Guide lines linking this heatmap back to the filmstrip snapshots above
        for d in snap_distances:
            ax.axvline(d, color="k", lw=0.5, ls=":", alpha=0.35, zorder=0)

        if n_clash:
            # Excluded on geometric grounds (unphysically close atom-atom contact).
            ax.scatter(
                df_clash["distance_angstrom"],
                df_clash["offset_angstrom"],
                marker="x", c="red", s=40, linewidths=1.2, zorder=6,
                label=f"{n_clash} clashing (<{min_contact:.1f} Å contact)",
            )
        if n_outlier:
            # Excluded on energetic grounds (robust outlier in E_int even
            # though the geometric contact cutoff didn't flag it).
            ax.scatter(
                df_outlier["distance_angstrom"],
                df_outlier["offset_angstrom"],
                marker="+", c="darkorange", s=50, linewidths=1.4, zorder=6,
                label=f"{n_outlier} energy outlier",
            )
        if n_clash or n_outlier:
            ax.legend(fontsize=6, loc="lower right", framealpha=0.7)

        ax.set_xlabel("Centre distance / Å")
        ax.set_ylabel("Lateral offset / Å")
        ax.set_title(BACKEND_LABELS.get(backend, backend))

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
                    title=f"E={min_e:.2f} kcal/mol", title_fontsize=6.5,
                )
            ax.annotate(
                "", xy=(min_d, min_o), xycoords="data",
                xytext=(x0 + w / 2, y0 + h / 2), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.7, ls="dashed", alpha=0.7),
                zorder=4,
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
        "--energy-clip", type=float, default=5.0,
        help="Clip energies to ±N kcal/mol for colour scale (default 5.0)",
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

    df = load_and_enrich(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_backends = ordered_backends(df, args.backends)
    backends = all_backends

    forces_calc, forces_label = _build_forces_calc(args.forces_backend, args.forces_checkpoint)

    pairs = df[["molecule_a", "molecule_b"]].drop_duplicates().values
    print(f"Plotting 2D PES for {len(pairs)} pairs × {len(backends)} backends...")

    for label_a, label_b in pairs:
        df_pair = df[(df["molecule_a"] == label_a) & (df["molecule_b"] == label_b)]
        plot_2d_pes_for_pair(
            df_pair, label_a, label_b, backends, args.output_dir,
            n_grid=args.n_grid, energy_clip_kcal=args.energy_clip,
            min_contact=args.min_contact, show_atoms=not args.no_atoms,
            forces_calc=forces_calc, forces_label=forces_label,
            max_cols=args.max_cols,
        )

    print(f"\nAll 2D PES plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
