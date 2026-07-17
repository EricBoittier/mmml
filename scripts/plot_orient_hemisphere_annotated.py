#!/usr/bin/env python3
"""Six-hemisphere orientation map + ASE overlays + cross-referenced 1D slices.

* Legend with body axes e0/e1/e2 and C=O
* Six hemispheres with coloured tag markers
* ASE dimer overlays **around** each hemisphere (tags that land there)
* COM→COM approach vector + body axes on every overlay (coloured + labeled)
* Same tagged geometries from multiple camera perspectives
* 1D slices with matching letter tags

    uv run python scripts/plot_orient_hemisphere_annotated.py \\
        --rays /Volumes/PortableSSD/DATA/acodcm/orient_6A/rays.csv \\
        --monomer /Volumes/PortableSSD/DATA/acodcm/pdb/aco.pdb \\
        --validate /Volumes/PortableSSD/DATA/acodcm/validate_ACO/rays_ACO.csv \\
        --out /Volumes/PortableSSD/DATA/acodcm/orient_plots/hemispheres
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import CubicSpline

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mmml.utils.plotting.styles import (  # noqa: E402
    STATUS_COLORS,
    apply_plot_style,
    comparison_colors,
    default_cmap,
    legend_outside,
    status_color,
)
from scripts.plot_orient_hemisphere_surfaces import (  # noqa: E402
    _aggregate_by_orientation,
    _draw_hemisphere,
    _energy_norm,
    _load_rays,
    fibonacci_sphere,
    quat_to_matrix,
    super_fibonacci,
)

EV_TO_KCAL = 23.0605

REF_TAGS: list[tuple[str, int, str]] = [
    ("A", 0, "clean-ish (validate ray 0)"),
    ("B", 2, "clean-ish (validate ray 2)"),
    ("C", 11, "deepest ML well (validate ray 59)"),
    ("D", 17, "dir8 cell can be repulsive"),
]

# Prefer these approach dirs when building overlays for a tag (matches validate where possible)
TAG_DIRS: dict[str, int] = {"A": 0, "B": 0, "C": 2, "D": 8}

AXIS_COLORS = ("#1A5276", "#943126", "#1E8449")
COM_COLOR = "#6C3483"

# Same annotation, three cameras
PERSPECTIVES: list[tuple[str, str]] = [
    ("front", "18x,-25y,5z"),
    ("side", "10x,-95y,0z"),
    ("top", "85x,10y,0z"),
]


def _load_plot_utils():
    path = _REPO / "scripts" / "plot_utils.py"
    spec = importlib.util.spec_from_file_location("mmml_plot_utils", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_monomer(path: Path) -> Atoms:
    if path.suffix == ".npz":
        raw = dict(np.load(path, allow_pickle=True))
        if "coords" in raw and "z" in raw:
            atoms = Atoms(numbers=np.asarray(raw["z"], dtype=int), positions=np.asarray(raw["coords"]))
        else:
            raise SystemExit(f"unrecognised monomer npz keys: {list(raw)}")
    else:
        atoms = read(str(path))
    atoms = atoms.copy()
    atoms.set_positions(atoms.get_positions() - atoms.get_positions().mean(axis=0))
    return atoms


def _co_bond_axis(atoms: Atoms) -> np.ndarray:
    z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    o_idx = int(np.where(z == 8)[0][0])
    c_idx = [i for i in range(len(z)) if z[i] == 6]
    d = np.linalg.norm(pos[c_idx] - pos[o_idx], axis=1)
    c = c_idx[int(np.argmin(d))]
    v = pos[o_idx] - pos[c]
    return v / np.linalg.norm(v)


def _dimer_atoms(
    mono: Atoms,
    *,
    direction: np.ndarray,
    quat: np.ndarray,
    r: float,
) -> tuple[Atoms, tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """Return dimer, fragments, COM_A, COM_B."""
    R = quat_to_matrix(quat)
    a = mono.get_positions()
    b = a @ R.T
    ra = a - 0.5 * r * direction
    rb = b + 0.5 * r * direction
    n = len(mono)
    atoms = Atoms(
        numbers=np.concatenate([mono.get_atomic_numbers(), mono.get_atomic_numbers()]),
        positions=np.vstack([ra, rb]),
    )
    com_a = ra.mean(axis=0)
    com_b = rb.mean(axis=0)
    return atoms, (np.arange(n), np.arange(n, 2 * n)), com_a, com_b


def _annotation_arrows(
    com_a: np.ndarray,
    com_b: np.ndarray,
    R_ori: np.ndarray,
    *,
    axis_scale: float = 1.6,
    highlight_axis: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    """COM→COM plus body axes from B; optionally emphasise one axis column."""
    arrows: list[tuple[np.ndarray, np.ndarray, str, str]] = [
        (com_a, com_b, COM_COLOR, "COM"),
    ]
    for k, (col, lab) in enumerate(zip(AXIS_COLORS, ("e0", "e1", "e2"), strict=True)):
        scale = axis_scale * (1.35 if highlight_axis == k else 1.0)
        arrows.append((com_b, com_b + scale * R_ori[:, k], col, lab))
    return arrows


def _r_for_tag(rays: dict[str, np.ndarray], ori: int, direction: int) -> float:
    m = (rays["orientation"] == ori) & (rays["direction"] == direction)
    if m.any():
        return float(rays["r_at_min"][m][0])
    m2 = rays["orientation"] == ori
    if m2.any():
        return float(np.median(rays["r_at_min"][m2]))
    return 5.5


def _validate_tables(path: Path) -> dict[int, dict]:
    rows = list(csv.DictReader(path.open()))
    out: dict[int, dict] = {}
    for ray in sorted({int(r["ray"]) for r in rows}):
        sub = sorted(
            [r for r in rows if int(r["ray"]) == ray],
            key=lambda r: float(r["r_com"]),
        )
        ori = int(sub[0]["orientation"])
        r = np.asarray([float(x["r_com"]) for x in sub])
        e_xtb = np.asarray([float(x["E_xtb"]) for x in sub]) * EV_TO_KCAL
        e_ml = np.asarray([float(x["E_ml"]) for x in sub]) * EV_TO_KCAL
        out[ori] = {
            "ray": ray,
            "direction": int(sub[0]["direction"]),
            "r": r,
            "e_xtb": e_xtb - e_xtb[-1],
            "e_ml": e_ml - e_ml[-1],
        }
    return out


def _tag_style() -> dict[str, str]:
    style = apply_plot_style("icml")
    cols = comparison_colors(style, n=len(REF_TAGS))
    return {tag: cols[i] for i, (tag, _, _) in enumerate(REF_TAGS)}


def _render_tag_overlay(
    ax,
    plot_utils,
    *,
    mono: Atoms,
    rays: dict[str, np.ndarray],
    quats: np.ndarray,
    dirs: np.ndarray,
    Rs: np.ndarray,
    tag: str,
    ori: int,
    tag_color: str,
    rotation: str,
    highlight_axis: int | None = None,
    title: str | None = None,
) -> None:
    d_idx = TAG_DIRS.get(tag, 0)
    r_use = _r_for_tag(rays, ori, d_idx)
    dim, frags, com_a, com_b = _dimer_atoms(
        mono, direction=dirs[d_idx], quat=quats[ori], r=r_use
    )
    arrows = _annotation_arrows(com_a, com_b, Rs[ori], highlight_axis=highlight_axis)
    plot_utils.render_dimer_atoms(
        ax,
        dim,
        frags,
        rotation=rotation,
        segment_arrows=arrows,
        title=title or f"{tag}: ori {ori}, dir {d_idx}",
        title_fontsize=7,
        label_color=tag_color,
        radii_scale=0.38,
    )
    # coloured frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(tag_color)
        spine.set_linewidth(2.0)


def plot_hemispheres_with_ase_ring(
    *,
    rays: dict[str, np.ndarray],
    mono: Atoms,
    out: Path,
    how: str = "min",
) -> None:
    """Six hemispheres, each surrounded by ASE overlays for tags on that hemi."""
    plot_utils = _load_plot_utils()
    apply_plot_style("icml")
    tag_color = _tag_style()
    quats = super_fibonacci(24)
    dirs = fibonacci_sphere(10)
    Rs = np.stack([quat_to_matrix(q) for q in quats], axis=0)
    axis_tips = [Rs[:, :, k] for k in range(3)]
    energy, _ = _aggregate_by_orientation(rays, how)
    cmap = default_cmap("diverging") if (energy.min() < 0 < energy.max()) else default_cmap(
        "sequential"
    )
    norm = _energy_norm(energy)

    fig = plt.figure(figsize=(16.5, 11.5))
    outer = GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.18)
    axis_names = ("e0", "e1", "e2")

    for col, (tips, name, acolor) in enumerate(zip(axis_tips, axis_names, AXIS_COLORS, strict=True)):
        for row, hemi in enumerate(("north", "south")):
            cell = outer[row, col]
            inner = cell.subgridspec(3, 3, wspace=0.08, hspace=0.18)
            ax3d = fig.add_subplot(inner[1, 1], projection="3d")
            _draw_hemisphere(
                ax3d,
                axis_points=tips,
                energy=energy,
                hemisphere=hemi,
                title=f"{name} — {hemi}",
                norm=norm,
                cmap=cmap,
                show_samples=True,
            )
            th = np.linspace(0, 2 * np.pi, 80)
            ax3d.plot(np.cos(th), np.sin(th), np.zeros_like(th), color=acolor, lw=1.6)

            # tags on this hemisphere
            on_hemi: list[tuple[str, int]] = []
            for tag, ori, _ in REF_TAGS:
                p = tips[ori]
                owns = (p[2] >= 0.0) if hemi == "north" else (p[2] < 0.0)
                if owns:
                    on_hemi.append((tag, ori))
                    ax3d.scatter(
                        [p[0]], [p[1]], [p[2]],
                        s=130, facecolors=tag_color[tag], edgecolors="white",
                        linewidths=1.1, depthshade=False, zorder=8,
                    )
                    ax3d.text(
                        p[0] * 1.35, p[1] * 1.35, p[2] * 1.35, tag,
                        color=tag_color[tag], fontsize=11, fontweight="bold",
                    )

            # ring slots around the 3D panel (skip center)
            slots = [
                (0, 0), (0, 1), (0, 2),
                (1, 0),         (1, 2),
                (2, 0), (2, 1), (2, 2),
            ]
            # fill with (tag, perspective) pairs — cycle perspectives for same tags
            fill: list[tuple[str, int, str, str]] = []
            for i, (tag, ori) in enumerate(on_hemi):
                for j, (pname, prot) in enumerate(PERSPECTIVES):
                    fill.append((tag, ori, pname, prot))
            # if few tags, still show all perspectives of each
            if not fill:
                # no tag on this hemi — show all tags from front as ghost context? skip
                for r, c in slots:
                    ax = fig.add_subplot(inner[r, c])
                    ax.set_axis_off()
                continue

            for slot_i, (r, c) in enumerate(slots):
                ax = fig.add_subplot(inner[r, c])
                if slot_i >= len(fill):
                    ax.set_axis_off()
                    continue
                tag, ori, pname, prot = fill[slot_i]
                _render_tag_overlay(
                    ax,
                    plot_utils,
                    mono=mono,
                    rays=rays,
                    quats=quats,
                    dirs=dirs,
                    Rs=Rs,
                    tag=tag,
                    ori=ori,
                    tag_color=tag_color[tag],
                    rotation=prot,
                    highlight_axis=col,
                    title=f"{tag} · {pname}",
                )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
    fig.colorbar(sm, cax=cax, label=f"{how} well depth (kcal/mol)")
    fig.suptitle(
        "Hemispheres with ASE overlays (COM vector + e0/e1/e2); "
        "same tags from front / side / top",
        fontsize=12,
        y=0.98,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_perspective_gallery(
    *,
    rays: dict[str, np.ndarray],
    mono: Atoms,
    out: Path,
) -> None:
    """Rows = tags A–D; columns = front / side / top. COM + axes on every panel."""
    plot_utils = _load_plot_utils()
    apply_plot_style("icml")
    tag_color = _tag_style()
    quats = super_fibonacci(24)
    dirs = fibonacci_sphere(10)
    Rs = np.stack([quat_to_matrix(q) for q in quats], axis=0)

    n_tag = len(REF_TAGS)
    n_persp = len(PERSPECTIVES)
    fig, axes = plt.subplots(
        n_tag,
        n_persp,
        figsize=(3.4 * n_persp, 3.1 * n_tag),
        squeeze=False,
    )
    for i, (tag, ori, note) in enumerate(REF_TAGS):
        for j, (pname, prot) in enumerate(PERSPECTIVES):
            ax = axes[i][j]
            _render_tag_overlay(
                ax,
                plot_utils,
                mono=mono,
                rays=rays,
                quats=quats,
                dirs=dirs,
                Rs=Rs,
                tag=tag,
                ori=ori,
                tag_color=tag_color[tag],
                rotation=prot,
                title=f"{tag} · {pname}  (ori {ori}, dir {TAG_DIRS[tag]})",
            )
            if j == 0:
                ax.text(
                    -0.08,
                    0.5,
                    note,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=7,
                    color=tag_color[tag],
                )

    # vector legend strip
    fig.text(
        0.5,
        0.01,
        "Arrows: purple COM = A->B approach  |  blue/red/green = body axes e0/e1/e2 from B COM",
        ha="center",
        fontsize=9,
        color=STATUS_COLORS["neutral"],
    )
    fig.suptitle("Tagged orientations from three perspectives (colour = tag)", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_annotated_dashboard(
    *,
    rays: dict[str, np.ndarray],
    mono: Atoms,
    validate: Path | None,
    out: Path,
    how: str = "min",
    slice_dirs: tuple[int, ...] = (0, 2, 8),
) -> None:
    """Compact dashboard: legend, hemispheres+tags, 1D slices (ASE on E(r))."""
    plot_utils = _load_plot_utils()
    apply_plot_style("icml")
    tag_color = _tag_style()
    quats = super_fibonacci(24)
    dirs = fibonacci_sphere(10)
    Rs = np.stack([quat_to_matrix(q) for q in quats], axis=0)
    axis_tips = [Rs[:, :, k] for k in range(3)]
    energy, _ = _aggregate_by_orientation(rays, how)
    cmap = default_cmap("diverging") if (energy.min() < 0 < energy.max()) else default_cmap(
        "sequential"
    )
    norm = _energy_norm(energy)
    co = _co_bond_axis(mono)
    val = _validate_tables(validate) if validate is not None else {}

    fig = plt.figure(figsize=(14.5, 12.0))
    gs = GridSpec(4, 6, figure=fig, height_ratios=[1.0, 1.15, 1.15, 1.35], hspace=0.4, wspace=0.32)

    ax_leg = fig.add_subplot(gs[0, 0:2])
    com0 = np.zeros(3)
    plot_utils.render_dimer_atoms(
        ax_leg,
        mono,
        rotation="25x,-35y,10z",
        segment_arrows=[
            (com0, 1.8 * np.array([1.0, 0, 0]), AXIS_COLORS[0], "e0"),
            (com0, 1.8 * np.array([0, 1.0, 0]), AXIS_COLORS[1], "e1"),
            (com0, 1.8 * np.array([0, 0, 1.0]), AXIS_COLORS[2], "e2"),
            (com0, 1.8 * co, STATUS_COLORS["warning"], "C=O"),
        ],
        title="Monomer B @ COM (identity)",
        title_fontsize=9,
    )

    ax_key = fig.add_subplot(gs[0, 2:4])
    ax_key.set_axis_off()
    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(0, 1)
    ax_key.set_title("Tags (colour + label)", fontsize=10, loc="left")
    y = 0.9
    for tag, ori, note in REF_TAGS:
        ax_key.add_patch(
            FancyBboxPatch(
                (0.02, y - 0.12), 0.12, 0.14, boxstyle="round,pad=0.02",
                facecolor=tag_color[tag], edgecolor="none",
            )
        )
        ax_key.text(0.08, y - 0.05, tag, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax_key.text(
            0.18, y - 0.05,
            f"ori {ori} · dir {TAG_DIRS[tag]} — {note}",
            ha="left", va="center", fontsize=8,
        )
        y -= 0.2
    ax_key.text(
        0.02, 0.02,
        "Purple COM arrow = monomer-A COM -> monomer-B COM.\n"
        "See hemisphere_ase_ring.png + perspectives_gallery.png for more views.",
        fontsize=7.5, color=STATUS_COLORS["neutral"], va="bottom",
    )

    # four tags, front view with COM
    ax_ex = fig.add_subplot(gs[0, 4:6])
    ax_ex.set_axis_off()
    for j, (tag, ori, _) in enumerate(REF_TAGS):
        inset = ax_ex.inset_axes([0.02 + (j % 2) * 0.49, 0.08 + (1 - j // 2) * 0.48, 0.46, 0.42])
        _render_tag_overlay(
            inset, plot_utils, mono=mono, rays=rays, quats=quats, dirs=dirs, Rs=Rs,
            tag=tag, ori=ori, tag_color=tag_color[tag],
            rotation=PERSPECTIVES[0][1], title=f"{tag} front",
        )

    axis_names = ("e0", "e1", "e2")
    for col, (tips, name, acolor) in enumerate(zip(axis_tips, axis_names, AXIS_COLORS, strict=True)):
        for row, hemi in enumerate(("north", "south")):
            ax = fig.add_subplot(gs[1 + row, col * 2: col * 2 + 2], projection="3d")
            _draw_hemisphere(
                ax, axis_points=tips, energy=energy, hemisphere=hemi,
                title=f"{name} — {hemi}", norm=norm, cmap=cmap, show_samples=True,
            )
            th = np.linspace(0, 2 * np.pi, 80)
            ax.plot(np.cos(th), np.sin(th), np.zeros_like(th), color=acolor, lw=1.4)
            for tag, ori, _ in REF_TAGS:
                p = tips[ori]
                owns = (p[2] >= 0.0) if hemi == "north" else (p[2] < 0.0)
                if not owns:
                    continue
                ax.scatter(
                    [p[0]], [p[1]], [p[2]], s=120,
                    facecolors=tag_color[tag], edgecolors="white",
                    linewidths=1.0, depthshade=False, zorder=8,
                )
                ax.text(
                    p[0] * 1.35, p[1] * 1.35, p[2] * 1.35, tag,
                    color=tag_color[tag], fontsize=11, fontweight="bold",
                )

    # 1D slices
    ax_s1 = fig.add_subplot(gs[3, 0:3])
    style = apply_plot_style("icml")
    dcols = comparison_colors(style, n=len(slice_dirs))
    oris = np.arange(24)
    for di, dcol in zip(slice_dirs, dcols, strict=True):
        e_row = np.full(24, np.nan)
        spur = np.zeros(24, dtype=bool)
        for o in oris:
            m = (rays["direction"] == di) & (rays["orientation"] == o)
            if m.any():
                e_row[o] = rays["e_min_kcal"][m][0]
                spur[o] = rays["n_min_ml"][m][0] > 1
        ax_s1.plot(oris, e_row, color=dcol, lw=1.6, label=f"dir {di}")
        ax_s1.scatter(oris[spur], e_row[spur], color=status_color("critical"), s=16, zorder=3)
    ax_s1.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.7)
    ymin, ymax = ax_s1.get_ylim()
    for tag, ori, _ in REF_TAGS:
        ax_s1.axvline(ori, color=tag_color[tag], ls=":", lw=1.0)
        ax_s1.text(
            ori, ymax - 0.08 * (ymax - ymin), tag,
            color=tag_color[tag], fontsize=10, fontweight="bold", ha="center", va="top",
        )
    ax_s1.set_xlabel("orientation index")
    ax_s1.set_ylabel("well depth (kcal/mol)")
    ax_s1.set_title("1D: e_min(orientation) at fixed approach dir")
    legend_outside(ax_s1, side="right")

    ax_s2 = fig.add_subplot(gs[3, 3:6])
    plotted = False
    for tag, ori, _ in REF_TAGS:
        if ori not in val:
            continue
        plotted = True
        t = val[ori]
        r_hi = np.linspace(float(t["r"].min()), float(t["r"].max()), 300)
        e_hi = CubicSpline(t["r"], t["e_xtb"], bc_type="natural")(r_hi)
        ax_s2.plot(r_hi, e_hi, color=tag_color[tag], lw=2.0, label=f"{tag}: xTB")
        ax_s2.plot(t["r"], t["e_ml"], color=tag_color[tag], lw=1.0, ls="--", alpha=0.75)
    if plotted:
        vtags = [(tag, ori) for tag, ori, _ in REF_TAGS if ori in val]
        for k, (tag, ori) in enumerate(vtags):
            t = val[ori]
            r_min = float(t["r"][np.argmin(t["e_xtb"])])
            inset = ax_s2.inset_axes([0.02 + k * 0.24, 0.52, 0.22, 0.45])
            _render_tag_overlay(
                inset, plot_utils, mono=mono, rays=rays, quats=quats, dirs=dirs, Rs=Rs,
                tag=tag, ori=ori, tag_color=tag_color[tag],
                rotation=PERSPECTIVES[0][1], title=f"{tag}",
            )
            ax_s2.annotate(
                tag, xy=(r_min, float(np.min(t["e_xtb"]))), xytext=(r_min, 3.2),
                color=tag_color[tag], fontsize=8, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color=tag_color[tag], lw=0.8),
            )
        ax_s2.plot([], [], color="k", ls="--", lw=1.0, label="ML (dashed)")
        ax_s2.set_ylim(-3.0, 5.0)
        ax_s2.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.7)
        ax_s2.set_xlabel("r_COM (A)")
        ax_s2.set_ylabel("binding E (kcal/mol)")
        ax_s2.set_title("1D: E(r) tagged oris (xTB solid, ML dashed)")
        legend_outside(ax_s2, side="right")
    else:
        ax_s2.set_axis_off()

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.38, 0.015, 0.28])
    fig.colorbar(sm, cax=cax, label=f"{how} well depth (kcal/mol)")
    fig.suptitle("ACO orientation map — tags, COM vectors, ASE overlays", fontsize=13, y=0.995)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_slice_strip(
    *,
    rays: dict[str, np.ndarray],
    mono: Atoms,
    out: Path,
    direction: int = 2,
) -> None:
    plot_utils = _load_plot_utils()
    apply_plot_style("icml")
    tag_color = _tag_style()
    quats = super_fibonacci(24)
    dirs = fibonacci_sphere(10)
    Rs = np.stack([quat_to_matrix(q) for q in quats], axis=0)
    oris = np.arange(24)
    e_row = np.full(24, np.nan)
    spur = np.zeros(24, dtype=bool)
    for o in oris:
        m = (rays["direction"] == direction) & (rays["orientation"] == o)
        if m.any():
            e_row[o] = rays["e_min_kcal"][m][0]
            spur[o] = rays["n_min_ml"][m][0] > 1

    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.plot(oris, e_row, color=STATUS_COLORS["neutral"], lw=1.8)
    ax.scatter(oris[~spur], e_row[~spur], c=e_row[~spur], cmap=default_cmap("diverging"), s=28, zorder=3)
    ax.scatter(
        oris[spur], e_row[spur], facecolors="none",
        edgecolors=status_color("critical"), s=40, linewidths=1.0, zorder=4, label="spurious",
    )
    ax.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.7)
    ax.set_xlabel("orientation index")
    ax.set_ylabel("well depth (kcal/mol)")
    ax.set_title(f"Direction {direction} — e_min(orientation) with ASE + COM")

    for k, (tag, ori, _) in enumerate(REF_TAGS):
        ax.axvline(ori, color=tag_color[tag], ls=":", lw=1.0)
        x0 = min(max(0.06 + (ori / 23.0) * 0.72, 0.05), 0.78)
        y0 = 0.55 if (np.isfinite(e_row[ori]) and e_row[ori] > np.nanmedian(e_row)) else 0.1
        inset = ax.inset_axes([x0, y0, 0.18, 0.4])
        # force this strip's direction for the overlay
        r_use = _r_for_tag(rays, ori, direction)
        dim, frags, com_a, com_b = _dimer_atoms(
            mono, direction=dirs[direction], quat=quats[ori], r=r_use
        )
        plot_utils.render_dimer_atoms(
            inset, dim, frags,
            rotation=PERSPECTIVES[k % len(PERSPECTIVES)][1],
            segment_arrows=_annotation_arrows(com_a, com_b, Rs[ori]),
            title=f"{tag}",
            title_fontsize=8,
            label_color=tag_color[tag],
            radii_scale=0.36,
        )
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color(tag_color[tag])
            spine.set_linewidth(1.8)

    legend_outside(ax, side="right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rays", type=Path, required=True)
    p.add_argument("--monomer", type=Path, required=True)
    p.add_argument("--validate", type=Path, default=None)
    p.add_argument("--how", choices=("min", "mean"), default="min")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rays = _load_rays(args.rays)
    mono = _load_monomer(args.monomer)

    plot_hemispheres_with_ase_ring(
        rays=rays, mono=mono, out=args.out / "hemisphere_ase_ring.png", how=args.how
    )
    plot_perspective_gallery(
        rays=rays, mono=mono, out=args.out / "perspectives_gallery.png"
    )
    plot_annotated_dashboard(
        rays=rays,
        mono=mono,
        validate=args.validate,
        out=args.out / "hemisphere_annotated_dashboard.png",
        how=args.how,
    )
    for d in (0, 2, 8):
        plot_slice_strip(
            rays=rays, mono=mono, out=args.out / f"slice_dir{d}_with_ase.png", direction=d
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
