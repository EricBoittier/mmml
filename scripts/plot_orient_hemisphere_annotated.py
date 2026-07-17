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
        # Drop clash spikes so a few 10^3 kcal wall hits don't set the window
        e_ori = e_ori[np.isfinite(e_ori) & (e_ori < 20.0)]
        e_dir = e_dir[np.isfinite(e_dir) & (e_dir < 20.0)]
        color_chunks.extend([e_ori, e_dir])
        e_clean = _clean_vals(rays, "e_min_kcal")
        e_chunks.append(e_clean[e_clean < 20.0] if e_clean.size else e_clean)
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
        r = float(rays["r_at_min"][m][0])
        if np.isfinite(r):
            return r
    m2 = rays["orientation"] == ori
    if m2.any():
        vals = np.asarray(rays["r_at_min"][m2], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            return float(np.median(vals))
    return 5.5


def _validate_tables(path: Path) -> dict[int, dict]:
    """orientation → full E(r_COM) table (last entry used as asymptote)."""
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


def _complete_edges(tags: list[tuple[str, int, str]]) -> tuple[tuple[str, str], ...]:
    """Every unordered pair among tags (A-B, A-C, …)."""
    labs = [t[0] for t in tags]
    edges: list[tuple[str, str]] = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            edges.append((labs[i], labs[j]))
    return tuple(edges)


def _antipode_dir(dirs: np.ndarray, di: int) -> int:
    """Nearest Fibonacci dir to −û[di] (homodimer exchange partner)."""
    return int(np.argmax(dirs @ (-dirs[di])))


def _uhat_exchange_pairs(
    rays: dict[str, np.ndarray],
    dirs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(e(+û), e(−û)) for same orientation — interaction exchange symmetry."""
    e_map, _ = _ray_lookup_maps(rays)
    xs: list[float] = []
    ys: list[float] = []
    oris = np.unique(rays["orientation"])
    for ori in oris:
        for di in range(len(dirs)):
            anti = _antipode_dir(dirs, di)
            if anti <= di:
                continue
            e1 = e_map.get((int(ori), di))
            e2 = e_map.get((int(ori), anti))
            if e1 is None or e2 is None:
                continue
            if not (np.isfinite(e1) and np.isfinite(e2)):
                continue
            xs.append(e1)
            ys.append(e2)
    return np.asarray(xs), np.asarray(ys)


def _tag_style(tags: list[tuple[str, int, str]] | None = None) -> dict[str, str]:
    tags = tags if tags is not None else REF_TAGS
    style = apply_plot_style("icml")
    cols = comparison_colors(style, n=max(len(tags), 1))
    return {tag: cols[i] for i, (tag, _, _) in enumerate(tags)}


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
    tag_dirs: dict[str, int] | None = None,
) -> None:
    d_idx = (tag_dirs or TAG_DIRS).get(tag, 0)
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


# Default path graph (atlas overrides with A-B, B-C, C-D chain)
PATH_EDGES: tuple[tuple[str, str], ...] = (("A", "B"), ("B", "C"), ("C", "D"))


def _tag_lookup(
    tags: list[tuple[str, int, str]] | None = None,
    tag_dirs: dict[str, int] | None = None,
) -> dict[str, tuple[int, int]]:
    """tag → (orientation, direction)."""
    tags = tags if tags is not None else REF_TAGS
    tag_dirs = tag_dirs if tag_dirs is not None else TAG_DIRS
    return {t: (o, tag_dirs[t]) for t, o, _ in tags}


def _select_surface_minima_tags(
    rays: dict[str, np.ndarray],
    *,
    quats: np.ndarray,
    dirs: np.ndarray,
    n_tags: int = 4,
    min_sep: float = 0.65,
) -> tuple[list[tuple[str, int, str]], dict[str, int]]:
    """Pick well-separated physical minima as tags A.., ordered by e0 longitude."""
    Rs = np.stack([quat_to_matrix(q) for q in quats], axis=0)
    phys = _physical_minima(rays)
    idx = np.flatnonzero(phys)
    if idx.size == 0:
        # fall back to legacy fixed tags
        return list(REF_TAGS), dict(TAG_DIRS)

    order = np.argsort(rays["e_min_kcal"][idx])
    picked: list[tuple[int, int, float, np.ndarray]] = []
    feats: list[np.ndarray] = []
    used_ori: set[int] = set()

    def _try_pick(require_new_ori: bool) -> None:
        for i in idx[order]:
            if len(picked) >= n_tags:
                return
            ori = int(rays["orientation"][i])
            di = int(rays["direction"][i])
            if require_new_ori and ori in used_ori:
                continue
            tip = Rs[ori][:, 0]
            u = dirs[di]
            feat = np.concatenate([tip, u])
            if feats and min(float(np.linalg.norm(feat - f)) for f in feats) < min_sep:
                continue
            picked.append((ori, di, float(rays["e_min_kcal"][i]), tip))
            feats.append(feat)
            used_ori.add(ori)

    _try_pick(require_new_ori=True)
    _try_pick(require_new_ori=False)
    # pad with next-deepest if still short
    if len(picked) < n_tags:
        have = {(o, d) for o, d, _, _ in picked}
        for i in idx[order]:
            ori = int(rays["orientation"][i])
            di = int(rays["direction"][i])
            if (ori, di) in have:
                continue
            tip = Rs[ori][:, 0]
            picked.append((ori, di, float(rays["e_min_kcal"][i]), tip))
            have.add((ori, di))
            if len(picked) >= n_tags:
                break

    # Order A..D by e0 tip longitude so the filmstrip chain walks the surface
    lons = [_xyz_to_lonlat(p[3][None, :])[0][0] for p in picked]
    order_lon = np.argsort(lons)
    picked = [picked[int(k)] for k in order_lon]

    labels = "ABCDEFGH"
    tags: list[tuple[str, int, str]] = []
    tag_dirs: dict[str, int] = {}
    for k, (ori, di, e, _tip) in enumerate(picked[:n_tags]):
        lab = labels[k]
        tags.append((lab, ori, f"min {e:.2f} kcal/mol"))
        tag_dirs[lab] = di
    return tags, tag_dirs


def _physical_minima(rays: dict[str, np.ndarray]) -> np.ndarray:
    """Boolean mask: single-minimum rays with finite well depth and r*."""
    spur = np.asarray(rays["n_min_ml"], dtype=int) > 1
    e = np.asarray(rays["e_min_kcal"], dtype=float)
    r = np.asarray(rays["r_at_min"], dtype=float)
    return (~spur) & np.isfinite(e) & np.isfinite(r)


def _top_physical_indices(
    rays: dict[str, np.ndarray],
    *,
    top_n: int,
) -> np.ndarray:
    """Indices of the deepest physical wells (most negative e_min)."""
    mask = _physical_minima(rays)
    idx = np.flatnonzero(mask)
    if idx.size == 0 or top_n <= 0:
        return np.asarray([], dtype=int)
    order = np.argsort(rays["e_min_kcal"][idx])
    return idx[order[: min(top_n, idx.size)]]


def _nearest_ori(quats: np.ndarray, q: np.ndarray) -> int:
    """Nearest super-Fibonacci orientation index (quaternion chord, sign-invariant)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    dots = np.abs(quats @ q)
    return int(np.argmax(dots))


def _nearest_dir(dirs: np.ndarray, u: np.ndarray) -> int:
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    return int(np.argmax(dirs @ u))


def _ray_lookup_maps(
    rays: dict[str, np.ndarray],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """(ori, dir) → e_min_kcal / r_at_min."""
    e_map: dict[tuple[int, int], float] = {}
    r_map: dict[tuple[int, int], float] = {}
    for i in range(len(rays["orientation"])):
        key = (int(rays["orientation"][i]), int(rays["direction"][i]))
        e_map[key] = float(rays["e_min_kcal"][i])
        r_map[key] = float(rays["r_at_min"][i])
    return e_map, r_map


def _path_energy_profile(
    *,
    tag0: str,
    tag1: str,
    quats: np.ndarray,
    dirs: np.ndarray,
    rays: dict[str, np.ndarray],
    n: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Well depth along a morph path via nearest-ray lookup.

    Returns ``(t, e_min_kcal, r_at_min, keys)`` with ``keys[k] = (ori, dir)``.
    """
    samples = _path_samples(
        tag0=tag0, tag1=tag1, quats=quats, dirs=dirs, rays=rays, n=n,
    )
    e_map, r_map = _ray_lookup_maps(rays)
    ts = np.empty(n)
    es = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    keys: list[tuple[int, int]] = []
    for k, (t, q, u, _r, _R) in enumerate(samples):
        ori = _nearest_ori(quats, q)
        di = _nearest_dir(dirs, u)
        key = (ori, di)
        keys.append(key)
        ts[k] = t
        es[k] = e_map.get(key, np.nan)
        rs[k] = r_map.get(key, np.nan)
    return ts, es, rs, keys


def _ref_ticks_on_path(
    *,
    tag0: str,
    tag1: str,
    quats: np.ndarray,
    dirs: np.ndarray,
    rays_path: dict[str, np.ndarray],
    ref_rays: dict[str, np.ndarray],
    n: int = 32,
) -> list[tuple[float, float]]:
    """(t, e_ref) for physical ref minima nearest to a path sample."""
    samples = _path_samples(
        tag0=tag0, tag1=tag1, quats=quats, dirs=dirs, rays=rays_path, n=n,
    )
    sample_keys = [
        (_nearest_ori(quats, q), _nearest_dir(dirs, u)) for _t, q, u, _r, _R in samples
    ]
    phys = _physical_minima(ref_rays)
    out: list[tuple[float, float]] = []
    seen: set[int] = set()
    for i in np.flatnonzero(phys):
        ori = int(ref_rays["orientation"][i])
        di = int(ref_rays["direction"][i])
        # nearest path sample in (ori, dir) index space
        best_k = min(
            range(n),
            key=lambda k: abs(sample_keys[k][0] - ori) + abs(sample_keys[k][1] - di),
        )
        # only mark if reasonably close to the morph path
        if abs(sample_keys[best_k][0] - ori) > 2 or abs(sample_keys[best_k][1] - di) > 1:
            continue
        if best_k in seen:
            continue
        seen.add(best_k)
        out.append((float(samples[best_k][0]), float(ref_rays["e_min_kcal"][i])))
    return out


def _path_samples(
    *,
    tag0: str,
    tag1: str,
    quats: np.ndarray,
    dirs: np.ndarray,
    rays: dict[str, np.ndarray],
    n: int,
    tags: list[tuple[str, int, str]] | None = None,
    tag_dirs: dict[str, int] | None = None,
) -> list[tuple[float, np.ndarray, np.ndarray, float, np.ndarray]]:
    """SLERP/lerp samples along a tag→tag path: (t, quat, û, r, R)."""
    lookup = _tag_lookup(tags, tag_dirs)
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
    tags: list[tuple[str, int, str]] | None = None,
    tag_dirs: dict[str, int] | None = None,
) -> None:
    tags = tags if tags is not None else REF_TAGS
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

    lookup = _tag_lookup(tags, tag_dirs)
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

    for tag, ori, _ in tags:
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


def _draw_symmetry_guides(ax) -> None:
    """Mark lab-frame equator / meridians on an equirectangular S2 map."""
    ax.axhline(0.0, color="#1C2833", ls="--", lw=0.7, alpha=0.55, zorder=2)
    for lon in (-90.0, 0.0, 90.0):
        ax.axvline(lon, color="#1C2833", ls=":", lw=0.6, alpha=0.45, zorder=2)
    ax.plot([0], [90], "k+", ms=7, zorder=3)
    ax.plot([0], [-90], "k+", ms=7, zorder=3)
    ax.text(4, 4, "eq", fontsize=6, color="#1C2833", alpha=0.7)
    ax.text(4, 82, "+z", fontsize=6, color="#1C2833", alpha=0.7)
    ax.text(4, -88, "-z", fontsize=6, color="#1C2833", alpha=0.7)
    ax.text(92, 4, "+y", fontsize=6, color="#1C2833", alpha=0.7)
    ax.text(-178, 4, "-x/+x", fontsize=6, color="#1C2833", alpha=0.7)


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
    tags: list[tuple[str, int, str]] | None = None,
    tag_dirs: dict[str, int] | None = None,
) -> None:
    """ASE morph frames along one path edge (horizontal or vertical strip)."""
    samples = _path_samples(
        tag0=tag0, tag1=tag1, quats=quats, dirs=dirs, rays=rays, n=n_frames,
        tags=tags, tag_dirs=tag_dirs,
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
        elif abs(t - 0.5) < 0.08 and n_frames <= 3:
            lab = f"{tag0}.{tag1}/2"
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
    mid = axes[n_frames // 2]
    mid.text(
        0.5, -0.10, f"{tag0}-{tag1}",
        transform=mid.transAxes, ha="center", va="top",
        fontsize=9, fontweight="bold",
        color=mcolors.to_hex(0.5 * (c0 + c1)),
        clip_on=False,
    )


def _draw_path_chord(
    ax,
    *,
    tips: np.ndarray,
    tag0: str,
    tag1: str,
    lookup: dict[str, tuple[int, int]],
    tag_color: dict[str, str],
    lw: float = 2.2,
    n: int = 48,
) -> None:
    o0, o1 = lookup[tag0][0], lookup[tag1][0]
    if tips.shape[0] < max(o0, o1) + 1:
        return
    p0, p1 = tips[o0], tips[o1]
    ts = np.linspace(0, 1, n)
    chord = np.stack([(1 - t) * p0 + t * p1 for t in ts], axis=0)
    chord = chord / np.linalg.norm(chord, axis=1, keepdims=True)
    lon_c, lat_c = _xyz_to_lonlat(chord)
    dlon = np.abs(np.diff(lon_c))
    breaks = np.where(dlon > 180)[0]
    start = 0
    mid = mcolors.to_hex(
        0.5 * (np.array(mcolors.to_rgb(tag_color[tag0])) + np.array(mcolors.to_rgb(tag_color[tag1])))
    )
    for b in list(breaks) + [len(lon_c) - 1]:
        ax.plot(
            lon_c[start : b + 1], lat_c[start : b + 1],
            color=mid, lw=lw, alpha=0.95, zorder=4,
        )
        start = b + 1


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
    ref_tip_indices: np.ndarray | None = None,
    tip_index_mode: str = "orientation",
    tags: list[tuple[str, int, str]] | None = None,
    tag_dirs: dict[str, int] | None = None,
    path_edges: tuple[tuple[str, str], ...] | None = None,
    highlight_edge: tuple[str, str] | None = None,
    show_symmetry: bool = False,
    mark_path_samples: list[np.ndarray] | None = None,
) -> object:
    """Equirectangular field. ``ref_tip_indices`` are tip-row indices to star."""
    tags = tags if tags is not None else REF_TAGS
    tag_dirs = tag_dirs if tag_dirs is not None else TAG_DIRS
    path_edges = path_edges if path_edges is not None else PATH_EDGES

    lon_g, lat_g, field = _equirect_field(tips, evals)
    im = ax.pcolormesh(lon_g, lat_g, field, cmap=cmap, norm=norm, shading="auto")
    if show_symmetry:
        _draw_symmetry_guides(ax)
    lon_s, lat_s = _xyz_to_lonlat(tips)
    ax.scatter(
        lon_s, lat_s, c=evals, cmap=cmap, norm=norm,
        s=14, edgecolors="white", linewidths=0.3, zorder=3,
    )
    if ref_tip_indices is not None and ref_tip_indices.size:
        for i in np.unique(ref_tip_indices):
            i = int(i)
            if i < 0 or i >= tips.shape[0]:
                continue
            lo, la = _xyz_to_lonlat(tips[i : i + 1])
            ax.scatter(
                lo, la, s=55, marker="*",
                facecolors="#F7DC6F", edgecolors="black", linewidths=0.5, zorder=6,
            )
    if mark_tags:
        lookup = _tag_lookup(tags, tag_dirs)
        for tag, ori, _ in tags:
            idx = ori if tip_index_mode == "orientation" else tag_dirs[tag]
            if idx >= tips.shape[0]:
                continue
            lo, la = _xyz_to_lonlat(tips[idx : idx + 1])
            ax.scatter(
                lo, la, s=55, facecolors=tag_color[tag],
                edgecolors="white", linewidths=0.8, zorder=5,
            )
            ax.text(lo[0] + 4, la[0] + 4, tag, color=tag_color[tag], fontsize=8, fontweight="bold")
        if tip_index_mode == "orientation":
            for t0, t1 in path_edges:
                bold = highlight_edge == (t0, t1)
                _draw_path_chord(
                    ax, tips=tips, tag0=t0, tag1=t1, lookup=lookup,
                    tag_color=tag_color, lw=2.8 if bold else 1.0,
                )
                if not bold:
                    # dim non-highlighted edges
                    pass
        if mark_path_samples:
            for tip in mark_path_samples:
                lo, la = _xyz_to_lonlat(np.asarray(tip).reshape(1, 3))
                ax.scatter(
                    lo, la, s=22, facecolors="white", edgecolors="#2C3E50",
                    linewidths=0.5, zorder=7,
                )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    if ylabel:
        ax.set_ylabel("lat", fontsize=8)
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
    n_h: int = 3,
    n_v: int = 5,
    shared: SharedScales | None = None,
    ref_rays: dict[str, np.ndarray] | None = None,
    ref_top_n: int = 12,
    validate: Path | None = None,
) -> None:
    """Atlas: all tag–tag links, A–mid–B filmstrips, û↔−û symmetry, E(r_COM)."""
    del n_v
    n_h = 3  # A, midpoint, B only
    plot_utils = _load_plot_utils()
    apply_plot_style("icml")
    quats = super_fibonacci(24)
    dirs = fibonacci_sphere(10)
    Rs = np.stack([quat_to_matrix(q) for q in quats], axis=0)
    axis_tips = [Rs[:, :, k] for k in range(3)]
    energy, _ = _aggregate_by_orientation(rays, how)
    e_dir = _aggregate_by_direction(rays, how)
    sc = scales_for_rays(rays, how=how, shared=shared)
    cmap = sc.energy_cmap()
    norm = sc.energy_norm()
    ymin, ymax = sc.e_ylim
    rmin, rmax = sc.r_xlim

    tag_source = ref_rays if ref_rays is not None else rays
    tags, tag_dirs = _select_surface_minima_tags(
        tag_source, quats=quats, dirs=dirs, n_tags=4,
    )
    # Prefer validate oris (full E(r_COM) curves) when they are physical wells
    val = _validate_tables(Path(validate)) if validate is not None and Path(validate).is_file() else {}
    if val:
        phys_src = _physical_minima(tag_source)
        val_cands: list[tuple[float, int, int]] = []
        for ori, tab in val.items():
            di = int(tab["direction"])
            m = (
                (tag_source["orientation"] == ori)
                & (tag_source["direction"] == di)
                & phys_src
            )
            if m.any():
                val_cands.append((float(tag_source["e_min_kcal"][m][0]), ori, di))
        val_cands.sort()
        # splice deepest validate wells into tag set (keep 4 total, unique oris)
        merged: list[tuple[int, int, float]] = []
        for e, ori, di in val_cands:
            if all(ori != o for o, _d, _e in merged):
                merged.append((ori, di, e))
            if len(merged) >= 2:
                break
        for tag, ori, note in tags:
            if all(ori != o for o, _d, _e in merged):
                # parse e from note "min X.XX kcal/mol" if present
                try:
                    e = float(note.split()[1])
                except (IndexError, ValueError):
                    e = 0.0
                merged.append((ori, tag_dirs[tag], e))
            if len(merged) >= 4:
                break
        labels = "ABCD"
        tips = [Rs[o][:, 0] for o, _d, _e in merged[:4]]
        order = np.argsort([_xyz_to_lonlat(t[None, :])[0][0] for t in tips])
        tags = []
        tag_dirs = {}
        for k, ix in enumerate(order):
            ori, di, e = merged[int(ix)]
            lab = labels[k]
            tags.append((lab, ori, f"min {e:.2f} kcal/mol"))
            tag_dirs[lab] = di

    path_edges = _complete_edges(tags)
    tag_color = _tag_style(tags)
    style = apply_plot_style("icml")
    edge_colors = comparison_colors(style, n=max(len(path_edges), 1))
    e_map, r_map = _ray_lookup_maps(rays)
    e_map_ref, r_map_ref = (
        _ray_lookup_maps(ref_rays) if ref_rays is not None else ({}, {})
    )

    ref_top = (
        _top_physical_indices(ref_rays, top_n=ref_top_n)
        if ref_rays is not None
        else np.asarray([], dtype=int)
    )
    ref_ori_tips = (
        np.asarray(ref_rays["orientation"][ref_top], dtype=int) if ref_top.size else None
    )

    n_edge = len(path_edges)
    n_film_rows = int(np.ceil(n_edge / 3))
    fig = plt.figure(figsize=(20.0, 3.6 + 3.8 + 2.2 * n_film_rows + 3.2))
    outer = GridSpec(
        3 + n_film_rows, 1, figure=fig,
        height_ratios=[0.9, 3.2] + [1.7] * n_film_rows + [2.8],
        hspace=0.35,
    )

    # ── Row 0: endpoint ASE ───────────────────────────────────────────────
    top = outer[0].subgridspec(1, 4, wspace=0.12)
    for j, (tag, ori, note) in enumerate(tags):
        ax = fig.add_subplot(top[0, j])
        _render_tag_overlay(
            ax, plot_utils, mono=mono, rays=rays, quats=quats, dirs=dirs, Rs=Rs,
            tag=tag, ori=ori, tag_color=tag_color[tag],
            rotation=PERSPECTIVES[j % len(PERSPECTIVES)][1],
            title=f"{tag}  {note}",
            tag_dirs=tag_dirs,
        )

    # ── Row 1: surfaces — e0 (all links) | approach+antipodes | SO3 | û/−û ─
    maps = outer[1].subgridspec(1, 4, wspace=0.22)
    ax_e0 = fig.add_subplot(maps[0, 0])
    last_im = _draw_equirect_panel(
        ax_e0,
        tips=axis_tips[0],
        evals=energy,
        title="e0  (all A-D links)",
        cmap=cmap,
        norm=norm,
        tag_color=tag_color,
        mark_tags=True,
        spine_color=AXIS_COLORS[0],
        ylabel=True,
        ref_tip_indices=ref_ori_tips,
        tags=tags,
        tag_dirs=tag_dirs,
        path_edges=path_edges,
        show_symmetry=True,
    )

    ax_ap = fig.add_subplot(maps[0, 1])
    _draw_equirect_panel(
        ax_ap,
        tips=dirs,
        evals=e_dir,
        title="approach  u <-> -u",
        cmap=cmap,
        norm=norm,
        tag_color=tag_color,
        mark_tags=True,
        tip_index_mode="direction",
        tags=tags,
        tag_dirs=tag_dirs,
        path_edges=(),
        show_symmetry=True,
    )
    # antipodal links for each tag (homodimer exchange)
    for tag, _ori, _ in tags:
        di = tag_dirs[tag]
        anti = _antipode_dir(dirs, di)
        lo1, la1 = _xyz_to_lonlat(dirs[di : di + 1])
        lo2, la2 = _xyz_to_lonlat(dirs[anti : anti + 1])
        ax_ap.plot(
            [lo1[0], lo2[0]], [la1[0], la2[0]],
            color=tag_color[tag], ls="--", lw=1.4, alpha=0.85, zorder=4,
        )
        ax_ap.scatter(
            [lo2[0]], [la2[0]], s=40, marker="^",
            facecolors="none", edgecolors=tag_color[tag], linewidths=1.0, zorder=5,
        )

    ax_ball = fig.add_subplot(maps[0, 2], projection="3d")
    _draw_axis_ball(
        ax_ball, quats=quats, energy=energy, tag_color=tag_color,
        cmap=cmap, norm=norm, title="SO(3) all links",
        elev=22, azim=-55, path_edges=path_edges,
        tags=tags, tag_dirs=tag_dirs,
    )

    ax_sym = fig.add_subplot(maps[0, 3])
    xs, ys = _uhat_exchange_pairs(rays, dirs)
    if xs.size:
        ax_sym.scatter(xs, ys, s=10, c="#5D6D7E", alpha=0.45, edgecolors="none")
    if ref_rays is not None:
        xr, yr = _uhat_exchange_pairs(ref_rays, dirs)
        if xr.size:
            ax_sym.scatter(
                xr, yr, s=14, marker="*",
                facecolors="#F7DC6F", edgecolors="black", linewidths=0.3, alpha=0.8,
                label="ref",
            )
    # tag points
    for tag, ori, _ in tags:
        di = tag_dirs[tag]
        anti = _antipode_dir(dirs, di)
        e1 = e_map.get((ori, di))
        e2 = e_map.get((ori, anti))
        if e1 is not None and e2 is not None:
            ax_sym.scatter(
                [e1], [e2], s=70, facecolors=tag_color[tag],
                edgecolors="black", linewidths=0.6, zorder=5,
            )
            ax_sym.text(e1, e2, f" {tag}", color=tag_color[tag], fontsize=8, fontweight="bold")
    lims = [ymin, ymax]
    ax_sym.plot(lims, lims, color=STATUS_COLORS["neutral"], lw=0.8, ls="--")
    ax_sym.set_xlim(lims)
    ax_sym.set_ylim(lims)
    ax_sym.set_aspect("equal")
    ax_sym.set_xlabel("e_min(+u)")
    ax_sym.set_ylabel("e_min(-u)")
    ax_sym.set_title("exchange symmetry")
    ax_sym.tick_params(labelsize=7)

    # ── Filmstrip rows: 3 edges per row, each A | mid | B ─────────────────
    for fr in range(n_film_rows):
        strip = outer[2 + fr].subgridspec(1, 3, wspace=0.18)
        for col in range(3):
            ei = fr * 3 + col
            if ei >= n_edge:
                ax = fig.add_subplot(strip[0, col])
                ax.set_axis_off()
                continue
            t0, t1 = path_edges[ei]
            ecol = edge_colors[ei]
            cell = strip[0, col]
            inner = cell.subgridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.15)
            _draw_filmstrip(
                fig, inner[0, 0], plot_utils=plot_utils, mono=mono, rays=rays,
                quats=quats, dirs=dirs, tag0=t0, tag1=t1, tag_color=tag_color,
                n_frames=n_h, orientation="horizontal",
                tags=tags, tag_dirs=tag_dirs,
            )
            # compact E(r*) stems + validate E(r) if either endpoint matches
            ax_er = fig.add_subplot(inner[0, 1])
            for tag in (t0, t1):
                ori = next(o for t, o, _ in tags if t == tag)
                di = tag_dirs[tag]
                if ori in val:
                    tab = val[ori]
                    ax_er.plot(
                        tab["r"], tab["e_xtb"], color=tag_color[tag], lw=1.6,
                        label=f"{tag}",
                    )
                    ax_er.plot(
                        tab["r"], tab["e_ml"], color=tag_color[tag], lw=1.0, ls="--", alpha=0.75,
                    )
                else:
                    r_star = r_map.get((ori, di), np.nan)
                    e_star = e_map.get((ori, di), np.nan)
                    if np.isfinite(r_star) and np.isfinite(e_star):
                        ax_er.scatter(
                            [r_star], [e_star], s=55, facecolors=tag_color[tag],
                            edgecolors="black", linewidths=0.5, zorder=4,
                        )
                        ax_er.axvline(r_star, color=tag_color[tag], ls=":", lw=0.8, alpha=0.6)
                    if ref_rays is not None:
                        rr = r_map_ref.get((ori, di), np.nan)
                        er = e_map_ref.get((ori, di), np.nan)
                        if np.isfinite(rr) and np.isfinite(er):
                            ax_er.scatter(
                                [rr], [er], s=40, marker="*",
                                facecolors="#F7DC6F", edgecolors="black", linewidths=0.4,
                            )
            ax_er.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.5)
            ax_er.set_xlim(max(3.0, rmin - 0.3), min(8.5, rmax + 0.8))
            ax_er.set_ylim(ymin, ymax)
            ax_er.set_title(f"{t0}-{t1}  r_COM", fontsize=8, color=ecol)
            ax_er.tick_params(labelsize=6)
            ax_er.set_xlabel("r", fontsize=7)

    # ── Bottom: many r_COM views ──────────────────────────────────────────
    bot = outer[2 + n_film_rows].subgridspec(1, 3, wspace=0.28)
    ax_curves = fig.add_subplot(bot[0, 0])
    if val:
        for ori, tab in sorted(val.items()):
            # colour by matching tag if any
            col = "#5D6D7E"
            lab = f"ori{ori}"
            for tag, o, _ in tags:
                if o == ori:
                    col = tag_color[tag]
                    lab = tag
                    break
            ax_curves.plot(tab["r"], tab["e_xtb"], color=col, lw=1.8, label=f"{lab} xtb")
            ax_curves.plot(tab["r"], tab["e_ml"], color=col, lw=1.0, ls="--", alpha=0.75)
    for tag, ori, _ in tags:
        di = tag_dirs[tag]
        r_star = r_map.get((ori, di), np.nan)
        e_star = e_map.get((ori, di), np.nan)
        if np.isfinite(r_star) and np.isfinite(e_star):
            ax_curves.scatter(
                [r_star], [e_star], s=60, facecolors=tag_color[tag],
                edgecolors="black", linewidths=0.6, zorder=5,
            )
    ax_curves.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.5)
    ax_curves.set_xlim(3.0, 8.5)
    ax_curves.set_ylim(ymin, ymax)
    ax_curves.set_xlabel("r_COM (A)")
    ax_curves.set_ylabel("E_int")
    ax_curves.set_title("E(r_COM) validate + tag wells")
    ax_curves.legend(fontsize=6, frameon=False, loc="best")

    # r* of well vs approach dir for each tag ori (more r_COM structure)
    ax_rdir = fig.add_subplot(bot[0, 1])
    for tag, ori, _ in tags:
        ds, rs = [], []
        for di in range(len(dirs)):
            r_star = r_map.get((ori, di), np.nan)
            if np.isfinite(r_star):
                ds.append(di)
                rs.append(r_star)
        if ds:
            ax_rdir.plot(ds, rs, "o-", color=tag_color[tag], lw=1.4, ms=4, label=tag)
    ax_rdir.set_xlabel("approach dir index")
    ax_rdir.set_ylabel("r* (A)")
    ax_rdir.set_title("well r* vs û  (fixed ori)")
    ax_rdir.set_ylim(rmin, rmax)
    ax_rdir.legend(fontsize=7, frameon=False)

    ax_re = fig.add_subplot(bot[0, 2])
    phys = _physical_minima(rays)
    ax_re.scatter(
        rays["r_at_min"][phys], rays["e_min_kcal"][phys],
        c="#AED6F1", s=12, alpha=0.7, edgecolors="none", label="physical",
    )
    spur = ~phys
    ax_re.scatter(
        rays["r_at_min"][spur], rays["e_min_kcal"][spur],
        c=status_color("critical"), s=10, alpha=0.35, marker="x", label="spurious",
    )
    if ref_rays is not None:
        rp = _physical_minima(ref_rays)
        ax_re.scatter(
            ref_rays["r_at_min"][rp], ref_rays["e_min_kcal"][rp],
            s=22, marker="*", facecolors="#F7DC6F", edgecolors="black",
            linewidths=0.3, alpha=0.75, label="ref",
        )
    for tag, ori, _ in tags:
        di = tag_dirs[tag]
        r_star = r_map.get((ori, di), np.nan)
        e_star = e_map.get((ori, di), np.nan)
        if np.isfinite(r_star):
            ax_re.scatter(
                [r_star], [e_star], s=70, facecolors=tag_color[tag],
                edgecolors="black", linewidths=0.7, zorder=5,
            )
            ax_re.text(r_star + 0.04, e_star, tag, color=tag_color[tag], fontsize=8, fontweight="bold")
    ax_re.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.5)
    ax_re.set_xlim(rmin, rmax)
    ax_re.set_ylim(ymin, ymax)
    ax_re.set_xlabel("r* (A)")
    ax_re.set_ylabel("e_min")
    ax_re.set_title("(r*, e_min) all rays")
    ax_re.legend(fontsize=6, frameon=False)

    if last_im is not None:
        cax = fig.add_axes([0.92, 0.55, 0.012, 0.25])
        fig.colorbar(last_im, cax=cax, label=f"{how} well (kcal/mol)")

    tag_line = "  ".join(f"{t}=ori{o}/dir{tag_dirs[t]}" for t, o, _ in tags)
    fig.suptitle(f"Path atlas  all links  |  {tag_line}", fontsize=11, y=0.995)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  tags: {tag_line}")
    print(f"  edges: {', '.join(f'{a}-{b}' for a, b in path_edges)}")


def plot_concentric_atlas(**kwargs) -> None:
    """Backward-compatible name → :func:`plot_path_atlas`."""
    plot_path_atlas(**kwargs)


def _labeled(path: Path, label: str) -> Path:
    """``foo.png`` + label ``xTB`` → ``foo_xTB.png`` (flip-pair friendly)."""
    return path.with_name(f"{path.stem}_{label}{path.suffix}")


def _stamp_label(fig: plt.Figure, label: str) -> None:
    """Large corner badge so flip pages are obvious (ML / GFN2 / gfn2nms / …)."""
    key = label.upper().replace("-", "").replace("_", "")
    if key in ("ML", "HYBRID", "6A", "8A"):
        color = "#1A5276"  # blue
    elif key in ("GFN2", "XTB", "GFN2XTB"):
        color = "#1E8449"  # green — reference GFN2-xTB
    elif "GFN2NMS" in key or key.startswith("NMS"):
        color = "#943126"  # brick — NMS-trained hybrid
    else:
        color = "#5D6D7E"
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
        action="append",
        default=None,
        help="Extra rays.csv used to lock shared colour/axis scales (repeatable)",
    )
    p.add_argument(
        "--ref-rays",
        type=Path,
        default=None,
        help="Reference rays.csv (e.g. GFN2) whose physical minima are starred",
    )
    p.add_argument(
        "--ref-top-n",
        type=int,
        default=12,
        help="How many deepest ref wells to star on maps (default: 12)",
    )
    p.add_argument(
        "--atlas-only",
        action="store_true",
        help="Only write the path-atlas figure",
    )
    args = p.parse_args()
    label = str(args.label).strip()

    args.out.mkdir(parents=True, exist_ok=True)
    rays = _load_rays(args.rays)
    mono = _load_monomer(args.monomer)
    ref_rays = _load_rays(args.ref_rays) if args.ref_rays is not None else None
    if args.match_rays:
        shared = build_shared_scales(
            rays, *[_load_rays(p) for p in args.match_rays], how=args.how
        )
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
            ref_rays=ref_rays,
            ref_top_n=args.ref_top_n,
            validate=args.validate,
        )
        if not args.atlas_only:
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
                "Physical minima (*) from `--ref-rays` (typically GFN2) when provided.",
                "",
                f"- path_atlas_{label}.png",
                f"- equirectangular_maps_{label}.png",
                f"- hemisphere_ase_ring_{label}.png",
                f"- perspectives_gallery_{label}.png",
                f"- hemisphere_annotated_dashboard_{label}.png",
                f"- slice_dir{{0,2,8}}_with_ase_{label}.png",
                "",
                f"Source rays: `{args.rays}`",
                (
                    "Match rays: " + ", ".join(f"`{p}`" for p in args.match_rays)
                    if args.match_rays
                    else "Match rays: (none)"
                ),
                f"Ref rays: `{args.ref_rays}`" if args.ref_rays else "Ref rays: (none)",
                "",
                "Example flip regenerate (atlas only):",
                "",
                "```bash",
                "uv run python scripts/plot_orient_hemisphere_annotated.py \\",
                "  --rays …/orient_6A/rays.csv --label ML --atlas-only \\",
                "  --monomer …/pdb/aco.pdb \\",
                "  --ref-rays …/orient_xtb/rays.csv \\",
                "  --match-rays …/orient_xtb/rays.csv --match-rays …/orient_gfn2nms/rays.csv \\",
                "  --validate …/validate_ACO/rays_ACO.csv \\",
                "  --out …/orient_plots/flip",
                "```",
                "",
            ]
        )
        + "\n"
    )
    print(f"  wrote {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
