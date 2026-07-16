#!/usr/bin/env python3
"""Compare SpookyNet interaction-energy components across checkpoints.

Each output uses one row per model and the fixed columns
``total, neural, electrostatics, CGenFF LJ, ZBL, MBD``.  The shared-scale
figure deliberately uses the same colour normalization in every panel;
the companion full-range figure uses a symmetric-log normalization to make
both ordinary wells and catastrophic values visible.  Missing component
columns are labelled as unavailable and are never interpreted as zero.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmml.utils.plotting.styles import apply_plot_style


COMPONENTS = (
    ("total", "comp_Eint_kcal_mol"),
    ("neural", "comp_Eint_neural_energy_kcal_mol"),
    ("electrostatics", "comp_Eint_electrostatics_energy_kcal_mol"),
    ("CGenFF LJ", "comp_Eint_cgenff_vdw_energy_kcal_mol"),
    ("ZBL", "comp_Eint_zbl_repulsion_energy_kcal_mol"),
    ("MBD", "comp_Eint_mbd_energy_kcal_mol"),
)


def _parse_input(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("input must be LABEL=CSV") from exc
    return label, Path(path)


def _hybrid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[frame["backend"].str.contains("spookynet_hybrid", na=False)].copy()
    if rows.empty:
        raise ValueError("CSV contains no spookynet_hybrid rows")
    return rows


def _common_geometry(models: list[tuple[str, pd.DataFrame]], pair: tuple[str, str]) -> pd.MultiIndex:
    keys = ["distance_angstrom", "offset_angstrom"]
    common: pd.MultiIndex | None = None
    for _, frame in models:
        subset = frame[(frame.molecule_a == pair[0]) & (frame.molecule_b == pair[1])]
        index = pd.MultiIndex.from_frame(subset[keys].drop_duplicates())
        common = index if common is None else common.intersection(index)
    if common is None or common.empty:
        raise ValueError(f"no common geometry for {pair[0]}-{pair[1]}")
    return common


def _surface(frame: pd.DataFrame, pair: tuple[str, str], column: str, common: pd.MultiIndex):
    if column not in frame.columns or not frame[column].notna().any():
        return None
    subset = frame[(frame.molecule_a == pair[0]) & (frame.molecule_b == pair[1])].copy()
    subset = subset.set_index(["distance_angstrom", "offset_angstrom"]).loc[common].reset_index()
    pivot = subset.pivot(index="offset_angstrom", columns="distance_angstrom", values=column)
    return pivot.columns.to_numpy(), pivot.index.to_numpy(), pivot.to_numpy()


def _plot_pair(
    models: list[tuple[str, pd.DataFrame]],
    pair: tuple[str, str],
    output: Path,
    *,
    full_range: bool,
    vmin: float,
    vmax: float,
) -> None:
    common = _common_geometry(models, pair)
    if full_range:
        values = []
        for _, frame in models:
            for _, column in COMPONENTS:
                surface = _surface(frame, pair, column, common)
                if surface is not None:
                    values.append(surface[2][np.isfinite(surface[2])])
        magnitude = max(1.0, np.nanmax(np.abs(np.concatenate(values))))
        norm = mcolors.SymLogNorm(linthresh=1.0, vmin=-magnitude, vmax=magnitude)
        suffix = "full range (symmetric log)"
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)
        suffix = f"shared scale [{vmin:g}, {vmax:g}] kcal mol$^{{-1}}$"

    nrows, ncols = len(models), len(COMPONENTS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.15 * ncols, 1.75 * nrows + 0.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_under("#34205c")
    cmap.set_over("#5a160f")
    for row, (label, frame) in enumerate(models):
        for col, (component, column) in enumerate(COMPONENTS):
            ax = axes[row, col]
            surface = _surface(frame, pair, column, common)
            if surface is None:
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
            else:
                x, y, z = surface
                image = ax.pcolormesh(x, y, z, shading="nearest", cmap=cmap, norm=norm)
                finite = z[np.isfinite(z)]
                ax.text(
                    0.02,
                    0.04,
                    f"min {finite.min():.2g}\nmax {finite.max():.2g}",
                    transform=ax.transAxes,
                    fontsize=6,
                    va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
                )
                if not full_range:
                    saturated = np.count_nonzero((finite < vmin) | (finite > vmax))
                    if saturated:
                        ax.text(0.98, 0.96, f"{saturated} sat.", ha="right", va="top", transform=ax.transAxes, fontsize=6)
            if row == 0:
                ax.set_title(component)
            if col == 0:
                ax.set_ylabel(
                    f"{label}\noffset (Å)",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=12,
                    fontsize=7,
                    fontweight="bold",
                )
            if row == nrows - 1:
                ax.set_xlabel("separation (Å)")
    if image is not None:
        fig.colorbar(image, ax=axes, location="right", shrink=0.9, label="interaction energy (kcal mol$^{-1}$)", extend="both")
    fig.suptitle(f"{pair[0]}–{pair[1]} component surfaces — {suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=_parse_input, required=True, metavar="LABEL=CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"))
    parser.add_argument("--vmin", type=float, default=-10.0)
    parser.add_argument("--vmax", type=float, default=20.0)
    parser.add_argument("--html", type=Path, help="Optional inline-visualization HTML fragment")
    args = parser.parse_args()

    apply_plot_style("icml")
    models = [(label, _hybrid_rows(pd.read_csv(path))) for label, path in args.input]
    first = models[0][1]
    pairs = args.pair or list(first[["molecule_a", "molecule_b"]].drop_duplicates().itertuples(index=False, name=None))
    shared_outputs: list[tuple[tuple[str, str], Path]] = []
    for pair in pairs:
        stem = f"{pair[0]}_{pair[1]}"
        shared = args.output_dir / f"{stem}_components_shared.png"
        _plot_pair(models, pair, shared, full_range=False, vmin=args.vmin, vmax=args.vmax)
        _plot_pair(models, pair, args.output_dir / f"{stem}_components_full.png", full_range=True, vmin=args.vmin, vmax=args.vmax)
        shared_outputs.append((pair, shared))

    if args.html:
        figures = []
        for pair, path in shared_outputs:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            label = f"{pair[0]}–{pair[1]} shared-scale energy-component surfaces"
            figures.append(
                f'<figure><img src="data:image/png;base64,{encoded}" alt="{label}">'
                f'<figcaption class="text-small text-muted">{label}</figcaption></figure>'
            )
        fragment = (
            '<div id="mmml-component-energy-surfaces">\n'
            '<style>\n'
            '#mmml-component-energy-surfaces{display:grid;gap:16px;color:var(--foreground);}\n'
            '#mmml-component-energy-surfaces figure{margin:0;}\n'
            '#mmml-component-energy-surfaces img{display:block;width:100%;height:auto;}\n'
            '#mmml-component-energy-surfaces figcaption{margin-top:4px;}\n'
            '</style>\n'
            + "\n".join(figures)
            + '\n</div>\n'
        )
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(fragment)


if __name__ == "__main__":
    main()
