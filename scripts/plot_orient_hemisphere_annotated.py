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
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from matplotlib import colors as mcolors
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
    _aggregate_by_direction,
    _aggregate_by_orientation,
    _draw_hemisphere,
    _energy_norm,
    _load_rays,
    fibonacci_sphere,
    quat_to_matrix,
    super_fibonacci,
)

EV_TO_KCAL = 23.0605


def _robust_lim(
    *arrays: np.ndarray,
    lo: float = 5.0,
    hi: float = 95.0,
    pad_frac: float = 0.10,
    include_zero: bool = False,
) -> tuple[float, float]:
    """Axis limits from percentiles so a few outliers don't dominate the view."""
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1.0, 1.0)
    lo_v = float(np.percentile(vals, lo))
    hi_v = float(np.percentile(vals, hi))
    if include_zero:
        lo_v = min(lo_v, 0.0)
        hi_v = max(hi_v, 0.0)
    span = hi_v - lo_v
    if span < 1e-9:
        span = max(abs(hi_v), 1.0) * 0.1
    return lo_v - pad_frac * span, hi_v + pad_frac * span


def _robust_lim_drop_spurious(
    rays: dict[str, np.ndarray],
    key: str,
    *,
    include_zero: bool = False,
) -> tuple[float, float]:
    """Prefer single-minimum rays for axis limits; fall back to all if too few."""
    vals = np.asarray(rays[key], dtype=float)
    spur = np.asarray(rays["n_min_ml"], dtype=int) > 1
    clean = vals[~spur]
    if np.isfinite(clean).sum() >= 8:
        return _robust_lim(clean, include_zero=include_zero)
    return _robust_lim(vals, include_zero=include_zero)


def _norm_from_limits(vmin: float, vmax: float) -> mcolors.TwoSlopeNorm | mcolors.Normalize:
    if vmin < 0.0 < vmax:
        return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


@dataclass(frozen=True)
class SharedScales:
    """Fixed colour + axis limits so ML / xTB flip pairs are comparable."""

    e_vmin: float
    e_vmax: float
    e_vcenter: float
    e_ymin: float
    e_ymax: float
    r_xmin: float
    r_xmax: float

    def energy_norm(self) -> mcolors.TwoSlopeNorm | mcolors.Normalize:
        eps = 1e-6 * max(self.e_vmax - self.e_vmin, 1e-6)
        vcenter = float(np.clip(self.e_vcenter, self.e_vmin + eps, self.e_vmax - eps))
        return mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=self.e_vmin, vmax=self.e_vmax)

    def energy_cmap(self):
        # Always diverging about the colour pivot (0 or the data median) so
        # deeper-than-typical and shallower-than-typical wells get distinct hues.
        return default_cmap("diverging")

    @property
    def e_ylim(self) -> tuple[float, float]:
        return self.e_ymin, self.e_ymax

    @property
    def r_xlim(self) -> tuple[float, float]:
        return self.r_xmin, self.r_xmax


def _clean_vals(rays: dict[str, np.ndarray], key: str) -> np.ndarray:
    vals = np.asarray(rays[key], dtype=float)
    spur = np.asarray(rays["n_min_ml"], dtype=int) > 1
    clean = vals[~spur]
    clean = clean[np.isfinite(clean)]
    if clean.size >= 8:
        return clean
    return vals[np.isfinite(vals)]


def build_shared_scales(
    *ray_sets: dict[str, np.ndarray],
    how: str = "min",
) -> SharedScales:
    """Joint robust colour/axis limits across one or more rays.csv tables.

    Colour limits come from the **map aggregates** (per-orientation / per-
    direction well depths) only.  Mixing in raw per-ray ``e_min`` pulls the
    window through zero (repulsive outliers) and collapses both ML and xTB
    onto a narrow band of a diverging colourbar.
    """
    color_chunks: list[np.ndarray] = []
    e_chunks: list[np.ndarray] = []
    r_chunks: list[np.ndarray] = []
    for rays in ray_sets:
        e_ori, _ = _aggregate_by_orientation(rays, how)
        e_dir = _aggregate_by_direction(rays, how)
        color_chunks.extend([e_ori, e_dir])
        e_chunks.append(_clean_vals(rays, "e_min_kcal"))
        r_chunks.append(_clean_vals(rays, "r_at_min"))

    color_vals = np.concatenate(color_chunks)
    color_norm = _energy_norm(color_vals, lo=5.0, hi=95.0)
    e_ymin, e_ymax = _robust_lim(*e_chunks, include_zero=True)
    r_xmin, r_xmax = _robust_lim(*r_chunks)
    return SharedScales(
        e_vmin=float(color_norm.vmin),
        e_vmax=float(color_norm.vmax),
        e_vcenter=float(color_norm.vcenter),
        e_ymin=e_ymin,
        e_ymax=e_ymax,
        r_xmin=r_xmin,
        r_xmax=r_xmax,
    )


def scales_for_rays(
    rays: dict[str, np.ndarray],
    *,
    how: str = "min",
    shared: SharedScales | None = None,
) -> SharedScales:
    return shared if shared is not None else build_shared_scales(rays, how=how)


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
    shared: SharedScales | None = None,
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
    sc = scales_for_rays(rays, how=how, shared=shared)
    cmap = sc.energy_cmap()
    norm = sc.energy_norm()

    fig = plt.figure(figsize=(22.0, 15.5))
    outer = GridSpec(2, 3, figure=fig, hspace=0.58, wspace=0.40)
    axis_names = ("e0", "e1", "e2")

    for col, (tips, name, acolor) in enumerate(zip(axis_tips, axis_names, AXIS_COLORS, strict=True)):
        for row, hemi in enumerate(("north", "south")):
            cell = outer[row, col]
            inner = cell.subgridspec(3, 3, wspace=0.18, hspace=0.28)
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
        figsize=(4.8 * n_persp, 4.2 * n_tag),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.58, wspace=0.45)
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
    shared: SharedScales | None = None,
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
    sc = scales_for_rays(rays, how=how, shared=shared)
    cmap = sc.energy_cmap()
    norm = sc.energy_norm()
    co = _co_bond_axis(mono)
    val = _validate_tables(validate) if validate is not None else {}

    fig = plt.figure(figsize=(20.0, 17.0))
    gs = GridSpec(
        4, 6, figure=fig,
        height_ratios=[1.0, 1.15, 1.15, 1.35],
        hspace=0.68, wspace=0.48,
    )

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
    s1_rows: list[np.ndarray] = []
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
        s1_rows.append(e_row)
    ax_s1.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.7)
    ymin, ymax = sc.e_ylim
    ax_s1.set_ylim(ymin, ymax)
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
        # Keep validate E(r) panel on the shared energy scale when available
        ax_s2.set_ylim(*sc.e_ylim)
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


def _xyz_to_lonlat(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors → lon/lat in degrees (equirectangular)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return lon, lat


def _equirect_field(
    tips: np.ndarray,
    energy: np.ndarray,
    *,
    n_lon: int = 180,
    n_lat: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RBF on the sphere, sampled onto an equirectangular lon/lat grid."""
    from scipy.interpolate import RBFInterpolator

    lon = np.linspace(-180.0, 180.0, n_lon)
    lat = np.linspace(-90.0, 90.0, n_lat)
    lon_g, lat_g = np.meshgrid(lon, lat)
    lon_r = np.radians(lon_g)
    lat_r = np.radians(lat_g)
    xyz = np.column_stack(
        [
            np.cos(lat_r.ravel()) * np.cos(lon_r.ravel()),
            np.cos(lat_r.ravel()) * np.sin(lon_r.ravel()),
            np.sin(lat_r.ravel()),
        ]
    )
    rbf = RBFInterpolator(tips, energy, kernel="thin_plate_spline", smoothing=0.15)
    field = rbf(xyz).reshape(lat_g.shape)
    return lon_g, lat_g, field


def plot_map_projections(
    *,
    rays: dict[str, np.ndarray],
    mono: Atoms,
    out: Path,
    how: str = "min",
    shared: SharedScales | None = None,
) -> None:
    """Flat equirectangular maps — 3 body-axis spheres (+ approach). ASE tags kept.

    Six hemispheres were only a 3D viewing split of *three* S²'s.  One map per
    sphere shows the whole surface without the N/S cut.
    """
    plot_utils = _load_plot_utils()
    apply_plot_style("icml")
    tag_color = _tag_style()
    quats = super_fibonacci(24)
    dirs = fibonacci_sphere(10)
    Rs = np.stack([quat_to_matrix(q) for q in quats], axis=0)
    axis_tips = [Rs[:, :, k] for k in range(3)]
    energy, _ = _aggregate_by_orientation(rays, how)
    e_dir = _aggregate_by_direction(rays, how)
    sc = scales_for_rays(rays, how=how, shared=shared)
    cmap = sc.energy_cmap()
    norm = sc.energy_norm()

    fig = plt.figure(figsize=(20.0, 15.0))
    gs = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[1.15, 1.15, 1.1],
        hspace=0.68,
        wspace=0.45,
    )

    body_panels = [
        ("e0 — equirectangular map", axis_tips[0], energy, 0),
        ("e1 — equirectangular map", axis_tips[1], energy, 1),
        ("e2 — equirectangular map", axis_tips[2], energy, 2),
    ]
    last_im = None
    for col, (title, tips, evals, ax_i) in enumerate(body_panels):
        ax = fig.add_subplot(gs[0, col])
        lon_g, lat_g, field = _equirect_field(tips, evals)
        last_im = ax.pcolormesh(lon_g, lat_g, field, cmap=cmap, norm=norm, shading="auto")
        lon_s, lat_s = _xyz_to_lonlat(tips)
        ax.scatter(
            lon_s, lat_s, c=evals, cmap=cmap, norm=norm,
            s=22, edgecolors="white", linewidths=0.4, zorder=3,
        )
        for tag, ori, _ in REF_TAGS:
            lo, la = _xyz_to_lonlat(tips[ori : ori + 1])
            ax.scatter(
                lo, la, s=90, facecolors=tag_color[tag],
                edgecolors="white", linewidths=1.0, zorder=5,
            )
            ax.annotate(
                tag, (lo[0], la[0]), textcoords="offset points", xytext=(5, 5),
                color=tag_color[tag], fontsize=11, fontweight="bold",
            )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("longitude (deg)")
        ax.set_ylabel("latitude (deg)" if col == 0 else "")
        ax.set_title(title, fontsize=10, color=AXIS_COLORS[ax_i])
        ax.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.5, ls=":")
        ax.axvline(0.0, color=STATUS_COLORS["neutral"], lw=0.5, ls=":")
        for side in ("bottom", "top", "left", "right"):
            ax.spines[side].set_color(AXIS_COLORS[ax_i])
            ax.spines[side].set_linewidth(1.5)

    # colourbar in the spare top-right cell
    ax_c = fig.add_subplot(gs[0, 3])
    ax_c.set_axis_off()
    if last_im is not None:
        fig.colorbar(last_im, ax=ax_c, fraction=0.8, pad=0.05, label=f"{how} well depth (kcal/mol)")

    # approach map (same colourscale as body-axis maps for flip parity)
    ax_u = fig.add_subplot(gs[1, 0])
    lon_g, lat_g, field = _equirect_field(dirs, e_dir)
    im_u = ax_u.pcolormesh(lon_g, lat_g, field, cmap=cmap, norm=norm, shading="auto")
    lon_s, lat_s = _xyz_to_lonlat(dirs)
    ax_u.scatter(
        lon_s, lat_s, c=e_dir, cmap=cmap, norm=norm,
        s=36, edgecolors="white", linewidths=0.5, zorder=3,
    )
    for d_idx, lab in ((0, "d0"), (2, "d2"), (8, "d8")):
        lo, la = _xyz_to_lonlat(dirs[d_idx : d_idx + 1])
        ax_u.annotate(
            lab, (lo[0], la[0]), textcoords="offset points", xytext=(4, 4),
            fontsize=8, fontweight="bold", color=STATUS_COLORS["serious"],
        )
    ax_u.set_xlim(-180, 180)
    ax_u.set_ylim(-90, 90)
    ax_u.set_xlabel("longitude (deg)")
    ax_u.set_ylabel("latitude (deg)")
    ax_u.set_title("approach û — equirectangular (+1 optional)", fontsize=10)
    fig.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04, label=f"{how} well depth")

    ax_note = fig.add_subplot(gs[1, 1:4])
    ax_note.set_axis_off()
    ax_note.set_xlim(0, 1)
    ax_note.set_ylim(0, 1)
    ax_note.text(0.0, 0.95, "How many maps do you need?", fontsize=12, fontweight="bold", va="top")
    ax_note.text(
        0.0,
        0.70,
        "• 6 hemispheres = 3 spheres x N/S view split (3D display only).\n"
        "• 3 flat maps (e0, e1, e2) cover the same SO(3) projection — usually enough.\n"
        "• +1 approach map (u-hat on S2) if direction-of-approach structure matters.\n"
        "• Still lost per axis-map: spin about that axis (twist DOF).\n"
        "• r is collapsed into the colour (well depth), not a map axis.\n"
        "• True SO(3) without that loss needs a 3D axis-angle ball or more panels.",
        fontsize=9,
        va="top",
    )
    ax_note.text(
        0.0,
        0.08,
        "Equirectangular: lon = atan2(y,x), lat = arcsin(z). "
        "Poles are stretched — read colours near +/-90 deg lat with care.",
        fontsize=8,
        color=STATUS_COLORS["neutral"],
        va="bottom",
    )

    for j, (tag, ori, _) in enumerate(REF_TAGS):
        ax = fig.add_subplot(gs[2, j])
        _render_tag_overlay(
            ax, plot_utils, mono=mono, rays=rays, quats=quats, dirs=dirs, Rs=Rs,
            tag=tag, ori=ori, tag_color=tag_color[tag],
            rotation=PERSPECTIVES[0][1],
            title=f"{tag}: map marker (front)",
        )

    fig.suptitle(
        "Flat map projections of the orientation spheres (3 enough; +1 for approach)",
        fontsize=12,
        y=0.98,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_projection_explainer(*, out: Path) -> None:
    """Backup diagram: 6D → what the maps are, and what 3 vs 6 means."""
    apply_plot_style("icml")
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    ax.set_title("Orientation maps - what they are (and are not)", fontsize=13, loc="left")

    def box(x, y, w, h, text, fc="#F4F6F7", ec="#5D6D7E"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.15", facecolor=fc, edgecolor=ec, lw=1.2
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=STATUS_COLORS["neutral"], lw=1.4),
        )

    box(0.3, 5.2, 2.4, 1.3, "6D rigid dimer\nr + û(S²) + R(SO(3))", fc="#EAF2F8")
    arrow(2.8, 5.85, 3.5, 5.85)
    box(3.5, 5.2, 2.6, 1.3, "Scan collapses r\nto well-depth colour", fc="#FEF9E7")
    arrow(6.2, 5.85, 6.9, 5.85)
    box(6.9, 5.2, 2.4, 1.3, "R = [e0 e1 e2]\n3 tips on S2", fc="#E8F8F5")
    arrow(9.4, 5.85, 10.1, 5.85)
    box(10.1, 5.2, 1.6, 1.3, "colour\n= e_min", fc="#F5EEF8")

    # two paths
    box(0.3, 2.8, 3.5, 1.8,
        "Path A - 6 hemispheres\n"
        "3 spheres x north/south\n"
        "(3D bowls; same data twice\n"
        "if you already have maps)",
        fc="#FDEDEC")
    box(4.3, 2.8, 3.5, 1.8,
        "Path B - 3 flat maps  *\n"
        "equirectangular e0,e1,e2\n"
        "whole sphere, no N/S split\n"
        "+ optional approach map",
        fc="#E9F7EF")
    box(8.3, 2.8, 3.4, 1.8,
        "Still lost on each map\n"
        "* spin about that axis\n"
        "* full SO(3) is 3D\n"
        "* r not a spatial axis",
        fc="#FCF3CF")

    arrow(2.0, 5.2, 2.0, 4.7)
    arrow(6.0, 5.2, 6.0, 4.7)

    ax.text(
        0.3,
        1.6,
        "How many do you really need?",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.3,
        0.35,
        "Minimum useful set: 3 body-axis maps (+ ASE tag overlays).\n"
        "Add the approach map if direction-of-approach structure matters (your dirs 2/5/6/8 story).\n"
        "Keep 6 hemispheres only if the 3D bowl view helps intuition - they are not extra information.\n"
        "For a single atlas page: 3 equirectangular maps + one perspective gallery of tags A-D.",
        fontsize=9,
        va="bottom",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_slice_strip(
    *,
    rays: dict[str, np.ndarray],
    mono: Atoms,
    out: Path,
    direction: int = 2,
    shared: SharedScales | None = None,
    how: str = "min",
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

    sc = scales_for_rays(rays, how=how, shared=shared)
    fig, ax = plt.subplots(figsize=(16.0, 6.0))
    ax.plot(oris, e_row, color=STATUS_COLORS["neutral"], lw=1.8)
    ax.scatter(
        oris[~spur], e_row[~spur],
        c=e_row[~spur], cmap=sc.energy_cmap(),
        norm=sc.energy_norm(), s=28, zorder=3,
    )
    ax.scatter(
        oris[spur], e_row[spur], facecolors="none",
        edgecolors=status_color("critical"), s=40, linewidths=1.0, zorder=4, label="spurious",
    )
    ax.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.7)
    ax.set_ylim(*sc.e_ylim)
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


def _quat_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    w = float(np.clip(w, -1.0, 1.0))
    theta = 2.0 * float(np.arccos(w))
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array([x, y, z]) / s
    return axis, theta


def _quat_to_ball(q: np.ndarray) -> np.ndarray:
    """SO(3) → unit ball via axis-angle: p = (θ/π) n̂."""
    axis, theta = _quat_to_axis_angle(q)
    return (theta / np.pi) * axis


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = float(np.arccos(dot))
    s = np.sin(theta)
    return (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / s


# Path graph among tags: top A–B, bottom C–D, left A–C, right C–B
PATH_EDGES: tuple[tuple[str, str], ...] = (("A", "B"), ("C", "D"), ("A", "C"), ("C", "B"))


def _tag_lookup() -> dict[str, tuple[int, int]]:
    """tag → (orientation, direction)."""
    return {t: (o, TAG_DIRS[t]) for t, o, _ in REF_TAGS}


def _path_samples(
    *,
    tag0: str,
    tag1: str,
    quats: np.ndarray,
    dirs: np.ndarray,
    rays: dict[str, np.ndarray],
    n: int,
) -> list[tuple[float, np.ndarray, np.ndarray, float, np.ndarray]]:
    """SLERP/lerp samples along a tag→tag path: (t, quat, û, r, R)."""
    lookup = _tag_lookup()
    o0, d0 = lookup[tag0]
    o1, d1 = lookup[tag1]
    q0, q1 = quats[o0], quats[o1]
    u0, u1 = dirs[d0], dirs[d1]
    r0 = _r_for_tag(rays, o0, d0)
    r1 = _r_for_tag(rays, o1, d1)
    out = []
    for k in range(n):
        t = k / max(n - 1, 1)
        q = _slerp(q0, q1, t)
        u = (1.0 - t) * u0 + t * u1
        u = u / np.linalg.norm(u)
        r = (1.0 - t) * r0 + t * r1
        out.append((t, q, u, r, quat_to_matrix(q)))
    return out


def _draw_axis_ball(
    ax,
    *,
    quats: np.ndarray,
    energy: np.ndarray,
    tag_color: dict[str, str],
    cmap,
    norm,
    title: str,
    elev: float = 22,
    azim: float = -55,
    path_edges: tuple[tuple[str, str], ...] | None = None,
    path_n: int = 24,
) -> None:
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 24)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="#D5D8DC", alpha=0.10, linewidth=0, shade=False)

    pts = np.stack([_quat_to_ball(q) for q in quats], axis=0)
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=energy, cmap=cmap, norm=norm, s=36, depthshade=False,
        edgecolors="white", linewidths=0.3,
    )

    lookup = _tag_lookup()
    if path_edges:
        for t0, t1 in path_edges:
            o0, o1 = lookup[t0][0], lookup[t1][0]
            curve = np.stack(
                [_quat_to_ball(_slerp(quats[o0], quats[o1], t)) for t in np.linspace(0, 1, path_n)],
                axis=0,
            )
            c0 = np.array(mcolors.to_rgb(tag_color[t0]))
            c1 = np.array(mcolors.to_rgb(tag_color[t1]))
            mid = mcolors.to_hex(0.5 * (c0 + c1))
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=mid, lw=2.2, alpha=0.9)

    for tag, ori, _ in REF_TAGS:
        p = pts[ori]
        ax.scatter(
            [p[0]], [p[1]], [p[2]],
            s=140, facecolors=tag_color[tag], edgecolors="white",
            linewidths=1.1, depthshade=False, zorder=5,
        )
        ax.text(
            p[0] * 1.15, p[1] * 1.15, p[2] * 1.15, tag,
            color=tag_color[tag], fontsize=10, fontweight="bold",
        )

    for vec, col, lab in (
        ([1.05, 0, 0], AXIS_COLORS[0], "x"),
        ([0, 1.05, 0], AXIS_COLORS[1], "y"),
        ([0, 0, 1.05], AXIS_COLORS[2], "z"),
    ):
        ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], color=col, lw=1.2)
        ax.text(vec[0], vec[1], vec[2], lab, color=col, fontsize=8)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(-1.15, 1.15)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def _draw_filmstrip(
    fig: plt.Figure,
    cell,
    *,
    plot_utils,
    mono: Atoms,
    rays: dict[str, np.ndarray],
    quats: np.ndarray,
    dirs: np.ndarray,
    tag0: str,
    tag1: str,
    tag_color: dict[str, str],
    n_frames: int,
    orientation: str,
    rotation: str = "16x,-28y,0z",
) -> None:
    """ASE morph frames along one path edge (horizontal or vertical strip)."""
    samples = _path_samples(
        tag0=tag0, tag1=tag1, quats=quats, dirs=dirs, rays=rays, n=n_frames,
    )
    if orientation == "horizontal":
        inner = cell.subgridspec(1, n_frames, wspace=0.18)
        axes = [fig.add_subplot(inner[0, i]) for i in range(n_frames)]
    else:
        inner = cell.subgridspec(n_frames, 1, hspace=0.22)
        axes = [fig.add_subplot(inner[i, 0]) for i in range(n_frames)]

    c0 = np.array(mcolors.to_rgb(tag_color[tag0]))
    c1 = np.array(mcolors.to_rgb(tag_color[tag1]))
    for ax, (t, q, u, r, R) in zip(axes, samples, strict=True):
        dim, frags, com_a, com_b = _dimer_atoms(mono, direction=u, quat=q, r=r)
        blend = mcolors.to_hex(c0 * (1 - t) + c1 * t)
        if t < 0.08:
            lab = tag0
        elif t > 0.92:
            lab = tag1
        else:
            lab = f"{t:.1f}"
        plot_utils.render_dimer_atoms(
            ax, dim, frags,
            rotation=rotation,
            segment_arrows=_annotation_arrows(com_a, com_b, R, axis_scale=1.25),
            title=lab,
            title_fontsize=7,
            label_color=blend,
            radii_scale=0.32,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(blend)
            spine.set_linewidth(1.5)
    # path name tucked under the strip (doesn't fight ASE titles)
    mid = axes[n_frames // 2]
    mid.text(
        0.5, -0.08, f"{tag0}–{tag1}",
        transform=mid.transAxes, ha="center", va="top",
        fontsize=8, fontweight="bold",
        color=mcolors.to_hex(0.5 * (c0 + c1)),
        clip_on=False,
    )


def _draw_equirect_panel(
    ax,
    *,
    tips: np.ndarray,
    evals: np.ndarray,
    title: str,
    cmap,
    norm,
    tag_color: dict[str, str],
    mark_tags: bool,
    spine_color: str | None = None,
    ylabel: bool = False,
) -> object:
    lon_g, lat_g, field = _equirect_field(tips, evals)
    im = ax.pcolormesh(lon_g, lat_g, field, cmap=cmap, norm=norm, shading="auto")
    lon_s, lat_s = _xyz_to_lonlat(tips)
    ax.scatter(
        lon_s, lat_s, c=evals, cmap=cmap, norm=norm,
        s=14, edgecolors="white", linewidths=0.3, zorder=3,
    )
    if mark_tags:
        for tag, ori, _ in REF_TAGS:
            lo, la = _xyz_to_lonlat(tips[ori : ori + 1])
            ax.scatter(
                lo, la, s=55, facecolors=tag_color[tag],
                edgecolors="white", linewidths=0.8, zorder=5,
            )
            ax.text(lo[0] + 4, la[0] + 4, tag, color=tag_color[tag], fontsize=8, fontweight="bold")
            # path chords on the flat map (great-circle approx in lon/lat)
        lookup = _tag_lookup()
        for t0, t1 in PATH_EDGES:
            o0, o1 = lookup[t0][0], lookup[t1][0]
            # only draw if tips are orientation tips (same length as quats)
            if tips.shape[0] < max(o0, o1) + 1:
                continue
            p0 = tips[o0]
            p1 = tips[o1]
            ts = np.linspace(0, 1, 40)
            chord = np.stack([(1 - t) * p0 + t * p1 for t in ts], axis=0)
            chord = chord / np.linalg.norm(chord, axis=1, keepdims=True)
            lon_c, lat_c = _xyz_to_lonlat(chord)
            # break lines that wrap across ±180
            dlon = np.abs(np.diff(lon_c))
            breaks = np.where(dlon > 180)[0]
            start = 0
            mid = mcolors.to_hex(
                0.5 * (np.array(mcolors.to_rgb(tag_color[t0])) + np.array(mcolors.to_rgb(tag_color[t1])))
            )
            for b in list(breaks) + [len(lon_c) - 1]:
                ax.plot(lon_c[start : b + 1], lat_c[start : b + 1], color=mid, lw=1.4, alpha=0.85, zorder=4)
                start = b + 1
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("lon (deg)", fontsize=8)
    if ylabel:
        ax.set_ylabel("lat (deg)", fontsize=8)
    if spine_color:
        ax.spines["bottom"].set_color(spine_color)
        ax.spines["bottom"].set_linewidth(2.0)
    return im


def plot_path_atlas(
    *,
    rays: dict[str, np.ndarray],
    mono: Atoms,
    out: Path,
    how: str = "min",
    n_h: int = 6,
    n_v: int = 5,
    shared: SharedScales | None = None,
) -> None:
    """Path-framed atlas: filmstrips A–B / C–D / A–C / C–B around surfaces."""
    plot_utils = _load_plot_utils()
    apply_plot_style("icml")
    tag_color = _tag_style()
    quats = super_fibonacci(24)
    dirs = fibonacci_sphere(10)
    Rs = np.stack([quat_to_matrix(q) for q in quats], axis=0)
    axis_tips = [Rs[:, :, k] for k in range(3)]
    energy, _ = _aggregate_by_orientation(rays, how)
    e_dir = _aggregate_by_direction(rays, how)
    sc = scales_for_rays(rays, how=how, shared=shared)
    cmap = sc.energy_cmap()
    norm = sc.energy_norm()
    style = apply_plot_style("icml")
    cut_colors = comparison_colors(style, n=4)

    fig = plt.figure(figsize=(24.0, 20.0))
    # Frame: corners = tags, edges = path filmstrips, center = surfaces
    outer = GridSpec(
        3, 3, figure=fig,
        height_ratios=[1.15, 4.6, 1.15],
        width_ratios=[1.15, 5.2, 1.15],
        hspace=0.28,
        wspace=0.22,
    )

    # ── Corners: endpoint ASE ─────────────────────────────────────────────
    corner_specs = (
        (0, 0, "A", 0, PERSPECTIVES[0][1]),
        (0, 2, "B", 2, PERSPECTIVES[1][1]),
        (2, 0, "C", 11, PERSPECTIVES[0][1]),
        (2, 2, "D", 17, PERSPECTIVES[1][1]),
    )
    for row, col, tag, ori, rot in corner_specs:
        ax = fig.add_subplot(outer[row, col])
        _render_tag_overlay(
            ax, plot_utils, mono=mono, rays=rays, quats=quats, dirs=dirs, Rs=Rs,
            tag=tag, ori=ori, tag_color=tag_color[tag],
            rotation=rot, title=tag,
        )

    # ── Edge filmstrips ───────────────────────────────────────────────────
    _draw_filmstrip(
        fig, outer[0, 1], plot_utils=plot_utils, mono=mono, rays=rays,
        quats=quats, dirs=dirs, tag0="A", tag1="B", tag_color=tag_color,
        n_frames=n_h, orientation="horizontal",
    )
    _draw_filmstrip(
        fig, outer[2, 1], plot_utils=plot_utils, mono=mono, rays=rays,
        quats=quats, dirs=dirs, tag0="C", tag1="D", tag_color=tag_color,
        n_frames=n_h, orientation="horizontal",
    )
    _draw_filmstrip(
        fig, outer[1, 0], plot_utils=plot_utils, mono=mono, rays=rays,
        quats=quats, dirs=dirs, tag0="A", tag1="C", tag_color=tag_color,
        n_frames=n_v, orientation="vertical", rotation="25x,-20y,0z",
    )
    # Right edge reads top→bottom toward the shared C–B path (B at top corner)
    _draw_filmstrip(
        fig, outer[1, 2], plot_utils=plot_utils, mono=mono, rays=rays,
        quats=quats, dirs=dirs, tag0="B", tag1="C", tag_color=tag_color,
        n_frames=n_v, orientation="vertical", rotation="25x,30y,0z",
    )

    # ── Center: surfaces + balls + cuts ───────────────────────────────────
    center = outer[1, 1].subgridspec(
        3, 4,
        height_ratios=[1.15, 1.15, 1.05],
        hspace=0.42,
        wspace=0.32,
    )

    last_im = _draw_equirect_panel(
        fig.add_subplot(center[0, 0]),
        tips=axis_tips[0], evals=energy, title="e0 map",
        cmap=cmap, norm=norm, tag_color=tag_color, mark_tags=True,
        spine_color=AXIS_COLORS[0], ylabel=True,
    )
    _draw_equirect_panel(
        fig.add_subplot(center[0, 1]),
        tips=axis_tips[1], evals=energy, title="e1 map",
        cmap=cmap, norm=norm, tag_color=tag_color, mark_tags=True,
        spine_color=AXIS_COLORS[1],
    )
    _draw_equirect_panel(
        fig.add_subplot(center[0, 2]),
        tips=axis_tips[2], evals=energy, title="e2 map",
        cmap=cmap, norm=norm, tag_color=tag_color, mark_tags=True,
        spine_color=AXIS_COLORS[2],
    )
    _draw_equirect_panel(
        fig.add_subplot(center[0, 3]),
        tips=dirs, evals=e_dir, title="approach û map",
        cmap=cmap, norm=norm, tag_color=tag_color, mark_tags=False,
    )

    # hemispheres (extra surface views) + axis-angle balls with paths
    for j, (hemi, tips, name) in enumerate(
        (
            ("north", axis_tips[0], "e0 N"),
            ("south", axis_tips[0], "e0 S"),
        )
    ):
        ax = fig.add_subplot(center[1, j], projection="3d")
        _draw_hemisphere(
            ax, axis_points=tips, energy=energy, hemisphere=hemi,
            title=name, norm=norm, cmap=cmap, show_samples=True,
        )
        for tag, ori, _ in REF_TAGS:
            p = tips[ori]
            owns = (p[2] >= 0.0) if hemi == "north" else (p[2] < 0.0)
            if owns:
                ax.scatter(
                    [p[0]], [p[1]], [p[2]], s=90,
                    facecolors=tag_color[tag], edgecolors="white",
                    linewidths=0.9, depthshade=False, zorder=8,
                )
                ax.text(p[0] * 1.25, p[1] * 1.25, p[2] * 1.25, tag, color=tag_color[tag], fontsize=9, fontweight="bold")

    for j, (title, elev, azim) in enumerate(
        (("SO(3) ball + paths", 22, -55), ("SO(3) ball · alt", 18, 40))
    ):
        ax = fig.add_subplot(center[1, 2 + j], projection="3d")
        _draw_axis_ball(
            ax, quats=quats, energy=energy, tag_color=tag_color,
            cmap=cmap, norm=norm, title=title, elev=elev, azim=azim,
            path_edges=PATH_EDGES,
        )

    # bottom center row: 2D cuts with path overlays on equator
    ax_c1 = fig.add_subplot(center[2, 0:2])
    oris = np.arange(24)
    for k, di in enumerate((0, 2, 5, 8)):
        e_row = np.full(24, np.nan)
        for o in oris:
            m = (rays["direction"] == di) & (rays["orientation"] == o)
            if m.any():
                e_row[o] = rays["e_min_kcal"][m][0]
        ax_c1.plot(oris, e_row, color=cut_colors[k], lw=1.5, label=f"dir {di}")
    ax_c1.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.6)
    ymin, ymax = sc.e_ylim
    ax_c1.set_ylim(ymin, ymax)
    for tag, ori, _ in REF_TAGS:
        ax_c1.axvline(ori, color=tag_color[tag], ls=":", lw=1.0)
        ax_c1.text(
            ori, ymax - 0.06 * (ymax - ymin), tag,
            color=tag_color[tag], fontsize=8, fontweight="bold", ha="center", va="top",
        )
    ax_c1.set_xlabel("orientation index")
    ax_c1.set_ylabel("well depth (kcal/mol)")
    ax_c1.set_title("e_min(orientation) | fixed dirs")
    legend_outside(ax_c1, side="right")

    ax_c2 = fig.add_subplot(center[2, 2])
    pts = np.stack([_quat_to_ball(q) for q in quats], axis=0)
    w = np.exp(-((pts[:, 2]) ** 2) / 0.08)
    scat = ax_c2.scatter(
        pts[:, 0], pts[:, 1], c=energy, cmap=cmap, norm=norm,
        s=16 + 70 * w, alpha=0.35 + 0.65 * w, edgecolors="white", linewidths=0.3,
    )
    lookup = _tag_lookup()
    for t0, t1 in PATH_EDGES:
        o0, o1 = lookup[t0][0], lookup[t1][0]
        curve = np.stack(
            [_quat_to_ball(_slerp(quats[o0], quats[o1], t)) for t in np.linspace(0, 1, 30)],
            axis=0,
        )
        mid = mcolors.to_hex(
            0.5 * (np.array(mcolors.to_rgb(tag_color[t0])) + np.array(mcolors.to_rgb(tag_color[t1])))
        )
        ax_c2.plot(curve[:, 0], curve[:, 1], color=mid, lw=1.8, alpha=0.9, zorder=4)
    for tag, ori, _ in REF_TAGS:
        p = pts[ori]
        ax_c2.scatter(
            [p[0]], [p[1]], s=90, facecolors=tag_color[tag],
            edgecolors="white", linewidths=1.0, zorder=5,
        )
        ax_c2.text(p[0] + 0.04, p[1] + 0.04, tag, color=tag_color[tag], fontsize=9, fontweight="bold")
    ax_c2.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=STATUS_COLORS["neutral"], lw=0.8))
    ax_c2.set_aspect("equal")
    ax_c2.set_xlim(-1.15, 1.15)
    ax_c2.set_ylim(-1.15, 1.15)
    ax_c2.set_xlabel("ball x")
    ax_c2.set_ylabel("ball y")
    ax_c2.set_title("ball equator + paths")
    fig.colorbar(scat, ax=ax_c2, fraction=0.046, pad=0.04)

    ax_c3 = fig.add_subplot(center[2, 3])
    spur = rays["n_min_ml"] > 1
    ax_c3.scatter(
        rays["r_at_min"][~spur], rays["e_min_kcal"][~spur],
        c=STATUS_COLORS["good"], s=16, alpha=0.7, label="single min", edgecolors="none",
    )
    ax_c3.scatter(
        rays["r_at_min"][spur], rays["e_min_kcal"][spur],
        c=status_color("critical"), s=18, alpha=0.7, label="spurious", edgecolors="none",
    )
    for tag, ori, _ in REF_TAGS:
        d_idx = TAG_DIRS[tag]
        m = (rays["orientation"] == ori) & (rays["direction"] == d_idx)
        if m.any():
            ax_c3.scatter(
                [rays["r_at_min"][m][0]], [rays["e_min_kcal"][m][0]],
                s=70, facecolors=tag_color[tag], edgecolors="black", linewidths=0.8, zorder=5,
            )
            ax_c3.text(
                rays["r_at_min"][m][0] + 0.05, rays["e_min_kcal"][m][0],
                tag, color=tag_color[tag], fontsize=8, fontweight="bold",
            )
    ax_c3.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.6)
    ax_c3.set_xlim(*sc.r_xlim)
    ax_c3.set_ylim(*sc.e_ylim)
    ax_c3.set_xlabel("r at min (A)")
    ax_c3.set_ylabel("well depth")
    ax_c3.set_title("(r, e) all rays")
    legend_outside(ax_c3, side="right")

    if last_im is not None:
        cax = fig.add_axes([0.935, 0.38, 0.012, 0.28])
        fig.colorbar(last_im, cax=cax, label=f"{how} well depth (kcal/mol)")

    fig.suptitle(
        "Path atlas: A–B (top) · C–D (bottom) · A–C (left) · C–B (right)  |  surfaces in center",
        fontsize=12,
        y=0.995,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_concentric_atlas(**kwargs) -> None:
    """Backward-compatible name → :func:`plot_path_atlas`."""
    plot_path_atlas(**kwargs)


def _labeled(path: Path, label: str) -> Path:
    """``foo.png`` + label ``xTB`` → ``foo_xTB.png`` (flip-pair friendly)."""
    return path.with_name(f"{path.stem}_{label}{path.suffix}")


def _stamp_label(fig: plt.Figure, label: str) -> None:
    """Large corner badge so ML / xTB pages are obvious when flipping."""
    color = "#1A5276" if label.upper() in ("ML", "HYBRID", "6A", "8A") else "#943126"
    fig.text(
        0.01,
        0.99,
        label,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none", alpha=0.95),
        zorder=100,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rays", type=Path, required=True)
    p.add_argument("--monomer", type=Path, required=True)
    p.add_argument("--validate", type=Path, default=None)
    p.add_argument("--how", choices=("min", "mean"), default="min")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--label",
        default="ML",
        help="Source badge + filename suffix (e.g. ML or xTB) for flip-pairs",
    )
    p.add_argument(
        "--match-rays",
        type=Path,
        default=None,
        help="Second rays.csv used to lock shared colour/axis scales (flip pairs)",
    )
    p.add_argument(
        "--atlas-only",
        action="store_true",
        help="Only write the concentric-cube atlas figure",
    )
    args = p.parse_args()
    label = str(args.label).strip()

    args.out.mkdir(parents=True, exist_ok=True)
    rays = _load_rays(args.rays)
    mono = _load_monomer(args.monomer)
    if args.match_rays is not None:
        shared = build_shared_scales(rays, _load_rays(args.match_rays), how=args.how)
        scale_path = args.out / "shared_scales.json"
        scale_path.write_text(json.dumps(asdict(shared), indent=2) + "\n")
        print(
            f"  shared scales: e_color=[{shared.e_vmin:.3f},{shared.e_vcenter:.3f},"
            f"{shared.e_vmax:.3f}]  "
            f"e_ylim=[{shared.e_ymin:.3f},{shared.e_ymax:.3f}]  "
            f"r_xlim=[{shared.r_xmin:.3f},{shared.r_xmax:.3f}]  -> {scale_path}"
        )
    else:
        shared = None

    _orig_savefig = plt.Figure.savefig

    def _savefig_with_badge(self, *a, **k):
        _stamp_label(self, label)
        if getattr(self, "_suptitle", None) is not None:
            old = self._suptitle.get_text()
            if f"[{label}]" not in old:
                self.suptitle(f"[{label}]  {old}", fontsize=self._suptitle.get_fontsize())
        return _orig_savefig(self, *a, **k)

    plt.Figure.savefig = _savefig_with_badge  # type: ignore[method-assign]
    try:
        plot_path_atlas(
            rays=rays,
            mono=mono,
            out=_labeled(args.out / "path_atlas.png", label),
            how=args.how,
            shared=shared,
        )
        if args.atlas_only:
            return 0

        plot_projection_explainer(
            out=_labeled(args.out / "projection_explainer.png", label)
        )
        plot_map_projections(
            rays=rays,
            mono=mono,
            out=_labeled(args.out / "equirectangular_maps.png", label),
            how=args.how,
            shared=shared,
        )
        plot_hemispheres_with_ase_ring(
            rays=rays,
            mono=mono,
            out=_labeled(args.out / "hemisphere_ase_ring.png", label),
            how=args.how,
            shared=shared,
        )
        plot_perspective_gallery(
            rays=rays,
            mono=mono,
            out=_labeled(args.out / "perspectives_gallery.png", label),
        )
        plot_annotated_dashboard(
            rays=rays,
            mono=mono,
            validate=args.validate,
            out=_labeled(args.out / "hemisphere_annotated_dashboard.png", label),
            how=args.how,
            shared=shared,
        )
        for d in (0, 2, 8):
            plot_slice_strip(
                rays=rays,
                mono=mono,
                out=_labeled(args.out / f"slice_dir{d}_with_ase.png", label),
                direction=d,
                shared=shared,
                how=args.how,
            )
    finally:
        plt.Figure.savefig = _orig_savefig  # type: ignore[method-assign]

    # index for flipping
    index = args.out / f"FLIP_{label}.md"
    index.write_text(
        "\n".join(
            [
                f"# {label} orientation figures",
                "",
                f"Flip pair: open the matching `*_ML.png` / `*_xTB.png` side by side.",
                "Colour/axis scales locked via `--match-rays` when provided (`shared_scales.json`).",
                "",
                f"- path_atlas_{label}.png",
                f"- equirectangular_maps_{label}.png",
                f"- hemisphere_ase_ring_{label}.png",
                f"- perspectives_gallery_{label}.png",
                f"- hemisphere_annotated_dashboard_{label}.png",
                f"- slice_dir{{0,2,8}}_with_ase_{label}.png",
                "",
                f"Source rays: `{args.rays}`",
                f"Match rays: `{args.match_rays}`" if args.match_rays else "Match rays: (none)",
                "",
            ]
        )
        + "\n"
    )
    print(f"  wrote {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
