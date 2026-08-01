#!/usr/bin/env python3
"""ICML-styled DCM liquid structure validation figures.

Combines POV-Ray snapshots (PSF topological bonds), thermo time series with
marginal distributions, element-pair RDFs, and PSF bond/angle distributions
into a single validation summary — plus restyled standalone panels.

See docs/plotting-style-guide.md (Structural analysis + POV-Ray liquid).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from ase.io import read as ase_read
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmml.utils.domdec_psf_order import read_psf_atoms_and_bonds  # noqa: E402
from mmml.utils.plotting.styles import (  # noqa: E402
    LINE_STYLE_CYCLE,
    apply_plot_style,
    assert_no_text_overlap,
    comparison_colors,
    legend_outside,
    shared_axis_labels,
    status_color,
    timeseries_with_distribution,
)
from mmml.utils.plotting.trajectory_structure import (  # noqa: E402
    InternalCoordinates,
    element_pair_rdfs,
)

# Coordinate-type colors — same semantic mapping as plot_trajectory_structure.
_COORDINATE_TYPE_COLORS = {
    "Bond lengths": "#1A5276",
    "Angles": "#B9770E",
    "Dihedrals": "#1E8449",
}

# Soft cutoffs for "distorted" PSF bonds (Å); used only for bond-health diagnostics.
_BOND_SOFT_MAX_A = {"C-H": 1.40, "H-C": 1.40, "C-Cl": 2.15, "Cl-C": 2.15}


def psf_bond_pairs(psf_path: Path) -> list[tuple[int, int]]:
    _atoms, bonds = read_psf_atoms_and_bonds(psf_path)
    return [(int(i), int(j)) if i < j else (int(j), int(i)) for i, j in bonds]


def load_frames(h5_path: Path, numbers: np.ndarray, box: float) -> list[Atoms]:
    with h5py.File(h5_path, "r") as f:
        positions = np.asarray(f["positions"], dtype=float)
    cell = [box, box, box]
    frames: list[Atoms] = []
    for pos in positions:
        atoms = Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True)
        atoms.wrap()
        frames.append(atoms)
    return frames


def load_thermo(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        return {
            "time_ps": np.asarray(f["time_ps"], dtype=float),
            "temperature": np.asarray(f["temperature"], dtype=float),
            "potential_energy": np.asarray(f["potential_energy"], dtype=float),
            "total_energy": np.asarray(f["total_energy"], dtype=float),
            "kinetic_energy": np.asarray(f["kinetic_energy"], dtype=float),
        }


def internal_from_psf_bonds(
    frames: list[Atoms], bonds: list[tuple[int, int]]
) -> InternalCoordinates:
    """Bond/angle/dihedral distributions using PSF topology only."""
    neighbors: dict[int, set[int]] = defaultdict(set)
    for a, b in bonds:
        neighbors[a].add(b)
        neighbors[b].add(a)
    angles = sorted(
        (a, c, b)
        for c, bonded in neighbors.items()
        for a, b in combinations(sorted(bonded), 2)
    )
    dihedrals = sorted(
        {
            (oa, a, b, ob)
            for a, b in bonds
            for oa in neighbors[a] - {b}
            for ob in neighbors[b] - {a}
            if oa != ob
        }
    )
    symbols = frames[0].get_chemical_symbols()

    def label(idxs: tuple[int, ...]) -> str:
        return "-".join(symbols[i] for i in idxs)

    def collect(items, getter):
        out = {}
        for item in items:
            key = f"{label(item)} ({'-'.join(map(str, item))})"
            out[key] = np.asarray([getter(frame, item) for frame in frames], dtype=float)
        return out

    return InternalCoordinates(
        bonds=collect(bonds, lambda f, item: f.get_distance(*item, mic=True)),
        angles=collect(angles, lambda f, item: f.get_angle(*item, mic=True)),
        dihedrals=collect(dihedrals, lambda f, item: f.get_dihedral(*item, mic=True)),
    )


def group_by_chem(coords: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for key, vals in coords.items():
        chem = key.split(" (")[0]
        grouped[chem].append(vals)
    return {k: np.concatenate(v) for k, v in grouped.items()}


def bond_health_timeseries(
    frames: list[Atoms], bonds: list[tuple[int, int]]
) -> dict[str, np.ndarray]:
    """Per-frame PSF bond statistics for validation (means, max, n_bad)."""
    symbols = frames[0].get_chemical_symbols()
    n_frames = len(frames)
    mean_by_chem: dict[str, list[float]] = defaultdict(list)
    max_bond = np.zeros(n_frames)
    n_bad = np.zeros(n_frames, dtype=int)
    for fi, frame in enumerate(frames):
        lengths = []
        bad = 0
        by_chem_frame: dict[str, list[float]] = defaultdict(list)
        for i, j in bonds:
            r = float(frame.get_distance(i, j, mic=True))
            pair = {symbols[i], symbols[j]}
            if pair == {"C", "H"}:
                chem = "C-H"
            elif pair == {"C", "Cl"}:
                chem = "C-Cl"
            else:
                chem = f"{symbols[i]}-{symbols[j]}"
            lengths.append(r)
            by_chem_frame[chem].append(r)
            soft = _BOND_SOFT_MAX_A.get(chem)
            if soft is not None and r > soft:
                bad += 1
        max_bond[fi] = max(lengths) if lengths else float("nan")
        n_bad[fi] = bad
        for chem, vals in by_chem_frame.items():
            mean_by_chem[chem].append(float(np.mean(vals)))
    out: dict[str, np.ndarray] = {
        "max_bond": max_bond,
        "n_bad": n_bad.astype(float),
    }
    for chem, vals in mean_by_chem.items():
        out[f"mean_{chem}"] = np.asarray(vals, dtype=float)
    return out


def _imshow_png(ax, path: Path, title: str) -> None:
    ax.imshow(np.asarray(Image.open(path)))
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=6)


def plot_rdfs_icml(radii: np.ndarray, rdfs: dict[str, np.ndarray], output: Path) -> None:
    """Lower-triangle element-pair RDF grid with jmol colors + shared axis labels."""
    import matplotlib.patheffects as path_effects

    apply_plot_style("icml")
    elements = sorted({el for pair in rdfs for el in pair.split("-")})
    size = len(elements)
    fig, axes = plt.subplots(
        size, size, figsize=(3.2 * size, 3.1 * size), sharex=True, sharey=False,
    )
    # Large hspace: bonded pairs (C–Cl) peak ~60 while intermolecular pairs
    # peak ~4 — stacked panels with independent y-scales otherwise collide
    # on tick labels at the shared edge.
    fig.subplots_adjust(wspace=0.32, hspace=0.55, top=0.88, bottom=0.08, left=0.12, right=0.98)
    for row, element_a in enumerate(elements):
        for column, element_b in enumerate(elements):
            axis = axes[row, column]
            if column > row:
                # Invisible panels still contribute tick Text to overlap checks —
                # strip artists entirely rather than only set_visible(False).
                axis.axis("off")
                continue
            key = "-".join(sorted((element_a, element_b)))
            color_a = np.asarray(jmol_colors[atomic_numbers[element_a]])
            color_b = np.asarray(jmol_colors[atomic_numbers[element_b]])
            pair_color = color_a if element_a == element_b else 0.5 * (color_a + color_b)
            ls = "-" if row == column else LINE_STYLE_CYCLE[(row + column) % len(LINE_STYLE_CYCLE)]
            line = axis.plot(radii, rdfs[key], color=pair_color, linestyle=ls, linewidth=1.8)[0]
            if float(np.mean(pair_color)) > 0.88:
                line.set_path_effects(
                    [path_effects.Stroke(linewidth=3.2, foreground="#888888"), path_effects.Normal()]
                )
            ymax = float(np.nanmax(rdfs[key])) if len(rdfs[key]) else 1.0
            axis.set_ylim(0.0, ymax * 1.05 if ymax > 0 else 1.0)
            axis.locator_params(axis="y", nbins=4)
            axis.set_title(key, fontsize=11)
            axis.grid(alpha=0.18)
            if column != 0:
                axis.tick_params(labelleft=False)
            if row != size - 1:
                axis.tick_params(labelbottom=False)
    shared_axis_labels(fig, xlabel=r"$r$ (Å)", ylabel=r"$g(r)$")
    fig.suptitle("Element-pair radial distribution functions", y=0.96)
    # No legend — panel titles carry pair identity; linestyle marks homo/hetero.
    fig.canvas.draw()
    assert_no_text_overlap(fig)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_internal_by_type_icml(data: InternalCoordinates, output: Path) -> None:
    """Per chemical-type histograms; legends outside each column."""
    apply_plot_style("icml")
    groups = [
        (data.bonds, "Bond lengths", "Å", 50),
        (data.angles, "Angles", "degrees", 48),
        (data.dihedrals, "Dihedrals", "degrees", 72),
    ]
    # Drop empty coordinate classes (DCM has no proper dihedrals) so blank
    # axes don't contribute overlapping default tick labels.
    groups = [(c, t, u, b) for c, t, u, b in groups if c]
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n + 1.5, 4.6), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(wspace=0.55, top=0.88, bottom=0.16, left=0.08, right=0.78)
    for col, (ax, (coords, title, unit, bins)) in enumerate(zip(axes, groups)):
        grouped = group_by_chem(coords)
        # Panel title carries the figure context — avoid a fig.suptitle that
        # collides with legend_outside()'s layout pass on wide multi-column figs.
        ax.set_title(f"PSF {title.lower()}")
        colors = comparison_colors("icml", n=max(len(grouped), 1))
        for i, (chem, vals) in enumerate(sorted(grouped.items())):
            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colors[i % len(colors)],
                linestyle=LINE_STYLE_CYCLE[i % len(LINE_STYLE_CYCLE)],
                label=f"{chem} (n={len(vals):,})",
            )
        ax.set_xlabel(unit)
        if col == 0:
            ax.set_ylabel("Probability density")
        else:
            ax.tick_params(labelleft=False)
        ax.set_ylim(bottom=0.0)
        ax.locator_params(axis="y", nbins=4)
        ax.grid(alpha=0.18)
        # Outer-edge legends for a multi-column figure.
        legend_outside(ax, side="left" if col == 0 else "right", fontsize=8)
    fig.canvas.draw()
    assert_no_text_overlap(fig)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_validation_summary(
    *,
    thermo: dict[str, np.ndarray],
    radii: np.ndarray,
    rdfs: dict[str, np.ndarray],
    internal: InternalCoordinates,
    bond_ts: dict[str, np.ndarray],
    pov_images: list[tuple[Path, str]],
    output: Path,
    title: str,
) -> None:
    """One composition: POV strip + thermo + bond health + RDF/angles."""
    apply_plot_style("icml")
    colors = comparison_colors("icml", n=6)
    t = thermo["time_ps"]

    # Tall figure + large hspace: multi-row summaries with twin axes or
    # bottom legends routinely collide under tighter budgets.
    fig = plt.figure(figsize=(16.0, 22.0))
    outer = fig.add_gridspec(
        5,
        1,
        height_ratios=[1.15, 1.25, 1.15, 0.85, 1.25],
        hspace=0.65,
        top=0.93,
        bottom=0.04,
        left=0.08,
        right=0.82,
    )

    # --- Row 0: POV-Ray snapshots -------------------------------------------------
    pov_gs = outer[0].subgridspec(1, len(pov_images), wspace=0.08)
    for i, (path, panel_title) in enumerate(pov_images):
        ax = fig.add_subplot(pov_gs[0, i])
        if path.exists():
            _imshow_png(ax, path, panel_title)
        else:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"missing:\n{path.name}", ha="center", va="center")
            ax.set_title(panel_title, pad=10)

    # --- Row 1: thermo time series + marginals -----------------------------------
    thermo_gs = outer[1].subgridspec(1, 3, wspace=0.55)

    def _thermo_panel(slot, y, *, color, ylabel, title, center: bool, ylim=None):
        from matplotlib.ticker import MaxNLocator

        ax_s, ax_h = timeseries_with_distribution(
            fig,
            slot,
            t,
            y,
            color=color,
            ylabel=ylabel,
            xlabel=r"$t$ (ps)",
            center=center,
            bins=16,
            width_ratios=(3.4, 1.0),
        )
        ax_s.set_title(title, fontsize=11, pad=18)
        if ylim is not None:
            ax_s.set_ylim(*ylim)
        # prune='both' drops corner ticks that collide with the orthogonal axis.
        ax_s.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax_s.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        # Drop the hist "count" label — it collides with the next panel's ylabel
        # in a 3-up row; the series xlabel already anchors the row.
        ax_h.set_xlabel("")
        ax_h.tick_params(labelbottom=False)
        ax_h.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        return ax_s

    t_vals = thermo["temperature"]
    ax_t = _thermo_panel(
        thermo_gs[0, 0],
        t_vals,
        color=colors[0],
        ylabel=r"$T$ (K)",
        title=f"Temperature  (mean={t_vals.mean():.1f} K)",
        center=False,
        ylim=(float(t_vals.min()) - 10.0, float(t_vals.max()) + 10.0),
    )
    ax_t.axhline(300.0, color=status_color("neutral"), linewidth=0.9, linestyle=":")

    pe = thermo["potential_energy"]
    _thermo_panel(
        thermo_gs[0, 1],
        pe,
        color=colors[1],
        ylabel=r"$E_{\mathrm{pot}}-\overline{E}$",
        title=f"Potential energy  (std={pe.std():.1f})",
        center=True,
    )

    et = thermo["total_energy"]
    _thermo_panel(
        thermo_gs[0, 2],
        et,
        color=colors[2],
        ylabel=r"$E_{\mathrm{tot}}-\overline{E}$",
        title=f"Total energy  (std={et.std():.1f})",
        center=True,
    )

    # --- Row 2: bond-length means / max (no twin axis) ---------------------------
    bond_gs = outer[2].subgridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.55)
    ax_bh = fig.add_subplot(bond_gs[0, 0])
    series_specs = [
        ("mean_C-H", r"mean C–H", colors[0], "-"),
        ("mean_C-Cl", r"mean C–Cl", colors[1], "--"),
        ("max_bond", r"max PSF bond", status_color("serious"), "-."),
    ]
    for key, label, color, ls in series_specs:
        if key in bond_ts:
            ax_bh.plot(t, bond_ts[key], color=color, linestyle=ls, linewidth=1.7, label=label)
    from matplotlib.ticker import MaxNLocator

    ax_bh.set_xlabel(r"$t$ (ps)")
    ax_bh.set_ylabel(r"Bond length (Å)")
    ax_bh.set_title("PSF bond lengths over time", fontsize=11, pad=10)
    ax_bh.grid(alpha=0.18)
    ax_bh.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax_bh.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    legend_outside(ax_bh, side="right", fontsize=8)

    ax_bhist = fig.add_subplot(bond_gs[0, 1])
    bond_groups = group_by_chem(internal.bonds)
    for i, (chem, vals) in enumerate(sorted(bond_groups.items())):
        ax_bhist.hist(
            vals,
            bins=50,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=colors[i % len(colors)],
            linestyle=LINE_STYLE_CYCLE[i % len(LINE_STYLE_CYCLE)],
            label=chem,
        )
    ax_bhist.set_xlabel(r"Bond length (Å)")
    ax_bhist.set_ylabel("density")
    ax_bhist.set_title("Bond-length distributions", fontsize=11, pad=10)
    ax_bhist.set_ylim(bottom=0.0)
    y0, y1 = ax_bhist.get_ylim()
    ax_bhist.set_ylim(0.0, y1 * 1.18)
    ax_bhist.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    ax_bhist.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax_bhist.grid(alpha=0.18)
    legend_outside(ax_bhist, side="right", fontsize=8)

    # --- Row 3: soft-cutoff bond outliers (own panel; avoids twin-axis collisions)
    ax_bad = fig.add_subplot(outer[3])
    ax_bad.plot(
        t,
        bond_ts["n_bad"],
        color=status_color("critical"),
        linestyle="-",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label=r"$n$ soft-cutoff outliers / frame",
    )
    ax_bad.set_xlabel(r"$t$ (ps)")
    ax_bad.set_ylabel(r"$n$ distorted")
    ax_bad.set_title("PSF bond soft-cutoff outliers (C–H>1.40 Å, C–Cl>2.15 Å)", fontsize=11, pad=10)
    ax_bad.grid(alpha=0.18)
    ax_bad.set_ylim(bottom=0.0)
    y0, y1 = ax_bad.get_ylim()
    ax_bad.set_ylim(0.0, max(y1 * 1.15, 1.0))
    ax_bad.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both", integer=True))
    ax_bad.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    legend_outside(ax_bad, side="right", fontsize=8)

    # --- Row 4: key RDFs + angles ------------------------------------------------
    struct_gs = outer[4].subgridspec(1, 2, wspace=0.55)
    ax_rdf = fig.add_subplot(struct_gs[0, 0])
    # Intermolecular-leaning pairs first; bonded C–Cl kept as a reference spike.
    preferred = ["C-C", "Cl-Cl", "H-H", "Cl-H", "C-H", "C-Cl"]
    plotted = [k for k in preferred if k in rdfs]
    for i, key in enumerate(plotted):
        els = key.split("-")
        c_a = np.asarray(jmol_colors[atomic_numbers[els[0]]])
        c_b = np.asarray(jmol_colors[atomic_numbers[els[1]]])
        pair_color = c_a if els[0] == els[1] else 0.5 * (c_a + c_b)
        ax_rdf.plot(
            radii,
            rdfs[key],
            color=pair_color,
            linestyle=LINE_STYLE_CYCLE[i % len(LINE_STYLE_CYCLE)],
            linewidth=1.7,
            label=key,
        )
    ax_rdf.set_xlabel(r"$r$ (Å)")
    ax_rdf.set_ylabel(r"$g(r)$")
    ax_rdf.set_xlim(0.0, min(12.0, float(radii.max())))
    ax_rdf.set_ylim(bottom=0.0)
    ax_rdf.set_title("Element-pair RDFs", fontsize=11, pad=10)
    ax_rdf.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax_rdf.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax_rdf.grid(alpha=0.18)
    legend_outside(ax_rdf, side="right", fontsize=8)

    ax_ang = fig.add_subplot(struct_gs[0, 1])
    angle_groups = group_by_chem(internal.angles)
    ang_colors = comparison_colors("icml", n=max(len(angle_groups), 1))
    for i, (chem, vals) in enumerate(sorted(angle_groups.items())):
        ax_ang.hist(
            vals,
            bins=48,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=ang_colors[i % len(ang_colors)],
            linestyle=LINE_STYLE_CYCLE[i % len(LINE_STYLE_CYCLE)],
            label=chem,
        )
    ax_ang.axvline(109.5, color=status_color("neutral"), linewidth=0.9, linestyle=":")
    ax_ang.set_xlabel("Angle (degrees)")
    ax_ang.set_ylabel("density")
    ax_ang.set_title("Angle distributions (PSF)", fontsize=11, pad=10)
    ax_ang.set_ylim(bottom=0.0)
    y0, y1 = ax_ang.get_ylim()
    ax_ang.set_ylim(0.0, y1 * 1.18)
    ax_ang.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    ax_ang.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax_ang.grid(alpha=0.18)
    proxy = Line2D(
        [0], [0], color=status_color("neutral"), linestyle=":", linewidth=0.9, label=r"109.5°"
    )
    handles, labels = ax_ang.get_legend_handles_labels()
    legend_outside(
        ax_ang, handles=handles + [proxy], labels=labels + [r"109.5°"], side="right", fontsize=8
    )

    fig.suptitle(title, fontsize=12, y=0.978)

    fig.canvas.draw()
    # Small pad: axis-aligned tick bboxes often touch without a visual collision.
    assert_no_text_overlap(fig, padding_px=2.0)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_txt(
    path: Path,
    *,
    h5: Path,
    psf: Path,
    n_bonds: int,
    n_frames: int,
    box: float,
    radii: np.ndarray,
    rdfs: dict[str, np.ndarray],
    internal: InternalCoordinates,
    outputs: list[str],
) -> None:
    lines = [
        "DCM liquid structure validation",
        f"source: {h5}",
        f"psf: {psf} ({n_bonds} topological bonds)",
        f"frames: {n_frames}  box: {box} Å",
        "",
        "RDF first peaks (r > 1.5 Å):",
    ]
    mask = radii > 1.5
    for key, g in sorted(rdfs.items()):
        if not np.any(mask):
            continue
        i = int(np.argmax(g[mask]))
        lines.append(f"  {key:6s}  r={float(radii[mask][i]):5.2f} Å  g={float(g[mask][i]):5.2f}")
    lines.append("")
    lines.append("PSF bond length means (Å):")
    for chem, vals in sorted(group_by_chem(internal.bonds).items()):
        a = np.asarray(vals)
        lines.append(f"  {chem:8s}  mean={a.mean():.3f}  std={a.std():.3f}  n={len(a)}")
    lines.append("")
    lines.append("PSF angle means (deg):")
    for chem, vals in sorted(group_by_chem(internal.angles).items()):
        a = np.asarray(vals)
        lines.append(f"  {chem:12s}  mean={a.mean():.2f}  std={a.std():.2f}  n={len(a)}")
    lines += [
        "",
        "POV-Ray notes",
        "- Bonds: PSF !NBOND only — never covalent-radius inferred for liquid MD",
        "- Molecules MIC-wrapped before render so intramolecular bonds stay contiguous",
        "- conda povray needs +L to share/povray-3.7/include (see povray_wrap.sh)",
        "- Camera: ASE orthographic; jmol textures + covalent_radii * scale",
        "",
        "Run notes",
        f"- source H5: {h5}",
        "- Prefer --psf-angle-restraints on jaxmd liquid NVT to keep DCM tetrahedral",
        "",
        "Outputs",
    ]
    lines.extend(f"- {name}" for name in outputs)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        type=Path,
        default=ROOT / "artifacts/lj_scales/prod_20ps_dual/nvt20_gpu0/pbc_nvt_jaxmd_nvt.h5",
    )
    parser.add_argument(
        "--pdb",
        type=Path,
        default=ROOT / "artifacts/lj_scales/prod_20ps_dual/nvt20_gpu0/pbc_nvt_jaxmd_minimized.pdb",
    )
    parser.add_argument(
        "--psf",
        type=Path,
        default=ROOT / "artifacts/lj_scales/liquid_nvt/mini.psf",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/lj_scales/structure_analysis",
    )
    parser.add_argument("--box", type=float, default=30.0)
    parser.add_argument(
        "--pov-frame-ps",
        type=float,
        default=2.0,
        help="Preferred snapshot time (ps) for POV captions; images are reused if present.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override figure suptitle (default encodes box / frame count).",
    )
    args = parser.parse_args()

    apply_plot_style("icml")
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    template = ase_read(str(args.pdb))
    numbers = template.get_atomic_numbers()
    bonds = psf_bond_pairs(args.psf)
    print(f"PSF bonds: {len(bonds)} | atoms: {len(numbers)}")

    frames = load_frames(args.h5, numbers, args.box)
    thermo = load_thermo(args.h5)
    print(f"Loaded {len(frames)} frames from {args.h5}")

    print("Computing element-pair RDFs…")
    radii, rdfs = element_pair_rdfs(frames, r_max=12.0, bins=200)
    rdf_png = out / "element_pair_rdfs.png"
    plot_rdfs_icml(radii, rdfs, rdf_png)
    np.savez_compressed(out / "element_pair_rdfs.npz", radii=radii, **rdfs)
    print("wrote", rdf_png)

    print("Computing PSF bond/angle/dihedral distributions…")
    internal = internal_from_psf_bonds(frames, bonds)
    by_type_png = out / "internal_coordinates_psf_by_type.png"
    plot_internal_by_type_icml(internal, by_type_png)
    print("wrote", by_type_png)

    # Aggregate (all bonds one hist) — keep for parity with plot_internal, ICML titles.
    apply_plot_style("icml")
    pooled_groups = [
        (c, t, u, b)
        for c, t, u, b in (
            (internal.bonds, "Bond lengths", "Å", 40),
            (internal.angles, "Angles", "degrees", 45),
            (internal.dihedrals, "Dihedrals", "degrees", 72),
        )
        if c
    ]
    fig, axes = plt.subplots(
        1, len(pooled_groups), figsize=(5.2 * len(pooled_groups), 4.2), squeeze=False,
    )
    axes = axes[0]
    fig.subplots_adjust(wspace=0.45, top=0.88, bottom=0.15, left=0.10, right=0.98)
    for col, (ax, (coords, title, unit, bins)) in enumerate(zip(axes, pooled_groups)):
        color = _COORDINATE_TYPE_COLORS[title]
        vals = np.concatenate(list(coords.values()))
        ax.hist(vals, bins=bins, density=True, alpha=0.85, color=color)
        ax.set_title(f"PSF {title.lower()} ({len(coords)} coords)", pad=14)
        ax.set_xlabel(unit)
        if col == 0:
            ax.set_ylabel("Probability density")
        else:
            ax.tick_params(labelleft=False)
        # Keep the top y-tick below the title band (nbins ticks sit on ylim).
        ax.set_ylim(0.0, None)
        y0, y1 = ax.get_ylim()
        ax.set_ylim(0.0, y1 * 1.18)
        ax.locator_params(axis="y", nbins=3)
        ax.grid(alpha=0.18)
    fig.canvas.draw()
    assert_no_text_overlap(fig)
    pooled_png = out / "internal_coordinates_psf.png"
    fig.savefig(pooled_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pooled_png)

    bond_ts = bond_health_timeseries(frames, bonds)
    pov_images = [
        (out / "dcm120_liquid_psf_bonds.png", f"Liquid · PSF bonds · t≈{args.pov_frame_ps:g} ps"),
        (out / "dcm120_liquid_psf_bonds_top.png", "Liquid · top view"),
        (out / "dcm_monomer_psf_bonds.png", "DCM monomer (PSF bonds)"),
    ]

    summary_png = out / "validation_summary.png"
    print("Building validation summary figure…")
    plot_validation_summary(
        thermo=thermo,
        radii=radii,
        rdfs=rdfs,
        internal=internal,
        bond_ts=bond_ts,
        pov_images=pov_images,
        output=summary_png,
        title=(
            args.title
            or (
                f"DCM:120 liquid NVT validation  ·  PSF topology  ·  "
                f"box={args.box:g} Å  ·  {len(frames)} frames"
            )
        ),
    )
    print("wrote", summary_png)

    outputs = [
        "validation_summary.png",
        "dcm120_liquid_psf_bonds.png",
        "dcm120_liquid_psf_bonds_top.png",
        "dcm120_liquid_minimized_psf_bonds.png",
        "dcm_monomer_psf_bonds.png",
        "element_pair_rdfs.png / .npz",
        "internal_coordinates_psf.png",
        "internal_coordinates_psf_by_type.png",
    ]
    write_summary_txt(
        out / "SUMMARY.txt",
        h5=args.h5,
        psf=args.psf,
        n_bonds=len(bonds),
        n_frames=len(frames),
        box=args.box,
        radii=radii,
        rdfs=rdfs,
        internal=internal,
        outputs=outputs,
    )
    print("DONE →", out)


if __name__ == "__main__":
    main()
