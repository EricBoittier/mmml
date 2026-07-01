#!/usr/bin/env python3
"""Generate ML/MM cutoff and heat-staging reference plots for docs/mlpot-settings.md."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "images" / "mlpot-settings"

from mmml.interfaces.pycharmmInterface.cutoffs import (  # noqa: E402
    DEFAULT_ML_SWITCH_WIDTH,
    DEFAULT_MM_SWITCH_ON,
    DEFAULT_MM_SWITCH_WIDTH,
    GAMMA_OFF,
    GAMMA_ON,
    CutoffParameters,
)

CUTOFF_PRESETS: list[tuple[str, str, CutoffParameters]] = [
    (
        "code-default",
        f"Code default ({DEFAULT_MM_SWITCH_ON:g} / {DEFAULT_MM_SWITCH_WIDTH:g} / {DEFAULT_ML_SWITCH_WIDTH:g} Å)",
        CutoffParameters(),
    ),
    (
        "dcm9-stability",
        "Legacy narrow ML (7 / 5 / 0.1 Å)",
        CutoffParameters(ml_switch_width=0.1, mm_switch_on=7.0, mm_switch_width=5.0),
    ),
    (
        "wide-ml-taper",
        "Wide ML taper (7 / 5 / 1.0 Å)",
        CutoffParameters(ml_switch_width=1.0, mm_switch_on=7.0, mm_switch_width=5.0),
    ),
    (
        "extended-handoff",
        "Extended handoff (8 / 3 / 1.5 Å)",
        CutoffParameters(ml_switch_width=1.5, mm_switch_on=8.0, mm_switch_width=3.0),
    ),
]


def _r_grid(cp: CutoffParameters, *, pad: float = 2.0) -> np.ndarray:
    r_max = max(
        float(cp.mm_switch_on) + 2.0 * float(cp.mm_switch_width),
        float(cp.mm_switch_on) - float(cp.ml_switch_width),
    ) * 1.35 + pad
    return np.linspace(0.01, r_max, 600)


def plot_cutoff_preset(name: str, label: str, cp: CutoffParameters) -> Path:
    r = _r_grid(cp)
    s_ml = cp.ml_scale(r, gamma_ml=GAMMA_ON)
    s_mm = cp.mm_scale_complementary(r, gamma_ml=GAMMA_ON, gamma_mm_off=GAMMA_OFF)
    handoff_start = float(cp.mm_switch_on) - float(cp.ml_switch_width)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r, s_ml, lw=2, color="C0", label=r"$s_{\mathrm{ML}}$")
    ax.plot(r, s_mm, lw=2, color="C1", label=r"$s_{\mathrm{MM}}$")
    ax.plot(
        r,
        s_ml + s_mm,
        "k--",
        lw=1.5,
        alpha=0.85,
        label=r"$s_{\mathrm{ML}} + s_{\mathrm{MM}}$",
    )
    ax.axvline(
        handoff_start,
        color="C0",
        ls="--",
        lw=1,
        alpha=0.7,
        label=f"handoff start {handoff_start:.2f} Å",
    )
    ax.axvline(
        cp.mm_switch_on,
        color="k",
        ls="-.",
        lw=1.5,
        label=f"handoff end {cp.mm_switch_on:.2f} Å",
    )
    ax.axvline(
        cp.mm_switch_on + cp.mm_switch_width,
        color="C1",
        ls="--",
        lw=1,
        alpha=0.8,
        label=f"MM taper end {cp.mm_switch_on + cp.mm_switch_width:.2f} Å",
    )
    ax.set_xlabel("Dimer COM distance r (Å)")
    ax.set_ylabel("Energy scale factor")
    ax.set_ylim(-0.05, 1.2)
    ax.set_title(label)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / f"cutoffs_{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_cutoff_comparison() -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 0.45, len(CUTOFF_PRESETS)))

    r_max = 16.0
    r = np.linspace(0.01, r_max, 600)
    for color, (_slug, label, cp) in zip(colors, CUTOFF_PRESETS, strict=True):
        short = label.split("(")[0].strip()
        axes[0].plot(
            r,
            cp.ml_scale(r, gamma_ml=GAMMA_ON),
            lw=2,
            color=color,
            label=short,
        )
        axes[1].plot(
            r,
            cp.mm_scale_complementary(r, gamma_ml=GAMMA_ON, gamma_mm_off=GAMMA_OFF),
            lw=2,
            color=color,
            label=short,
        )

    axes[0].set_ylabel(r"$s_{\mathrm{ML}}$")
    axes[1].set_ylabel(r"$s_{\mathrm{MM}}$")
    axes[1].set_xlabel("Dimer COM distance r (Å)")
    axes[0].set_title("ML/MM handoff presets (complementary)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "cutoffs_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_legacy_vs_complementary() -> Path:
    cp = CutoffParameters()
    r = _r_grid(cp, pad=3.0)
    s_ml = cp.ml_scale(r, gamma_ml=GAMMA_ON)
    mm_comp = cp.mm_scale_complementary(r, gamma_ml=GAMMA_ON, gamma_mm_off=GAMMA_OFF)

    cp_legacy = CutoffParameters(
        ml_switch_width=DEFAULT_ML_SWITCH_WIDTH,
        mm_switch_on=DEFAULT_MM_SWITCH_ON,
        mm_switch_width=DEFAULT_MM_SWITCH_WIDTH,
        complementary_handoff=False,
    )
    mm_legacy = cp_legacy.mm_scale(r, gamma_on=GAMMA_ON, gamma_off=GAMMA_OFF)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r, s_ml, lw=2, color="C0", label=r"$s_{\mathrm{ML}}$ (same)")
    ax.plot(r, mm_comp, lw=2, color="C1", label=r"$s_{\mathrm{MM}}$ complementary")
    ax.plot(r, mm_legacy, lw=2, ls=":", color="C3", label=r"$s_{\mathrm{MM}}$ legacy window")
    ax.set_xlabel("Dimer COM distance r (Å)")
    ax.set_ylabel("Scale factor")
    ax.set_title(
        f"Default cutoffs ({DEFAULT_MM_SWITCH_ON:g} / {DEFAULT_MM_SWITCH_WIDTH:g} / "
        f"{DEFAULT_ML_SWITCH_WIDTH:g} Å): complementary vs legacy MM window"
    )
    ax.set_ylim(-0.05, 1.2)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "cutoffs_complementary_vs_legacy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_heat_segments() -> Path:
    heat_firstt = 0.0
    heat_finalt = 240.0
    ps_heat = 20.0
    segment_counts = (1, 4, 8)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for n_seg in segment_counts:
        seg_ps = ps_heat / n_seg
        times: list[float] = [0.0]
        temps: list[float] = [heat_firstt]
        for seg_i in range(n_seg):
            seg_end = (seg_i + 1) * seg_ps
            seg_t = heat_firstt + (heat_finalt - heat_firstt) * ((seg_i + 1) / n_seg)
            times.extend([seg_i * seg_ps, seg_end])
            t_start = heat_firstt + (heat_finalt - heat_firstt) * (seg_i / n_seg)
            temps.extend([t_start, seg_t])
        axes[0].plot(times, temps, lw=2, marker="o", label=f"{n_seg} segment(s)")

    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("Target bath T (K)")
    axes[0].set_title(f"Staged heat ramp ({heat_firstt:.0f} → {heat_finalt:.0f} K, {ps_heat:.0f} ps)")
    axes[0].set_xlim(0, ps_heat)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    labels = []
    seg_temps = []
    for n_seg in segment_counts:
        labels.append(f"N={n_seg}")
        seg_temps.append(
            [
                heat_firstt + (heat_finalt - heat_firstt) * ((i + 1) / n_seg)
                for i in range(n_seg)
            ]
        )
    x = np.arange(max(segment_counts))
    width = 0.25
    for idx, (n_seg, temps) in enumerate(zip(segment_counts, seg_temps, strict=True)):
        axes[1].bar(
            x[:n_seg] + idx * width,
            temps,
            width=width,
            label=f"{n_seg} segments",
        )
    axes[1].set_xlabel("Segment index (0-based)")
    axes[1].set_ylabel("Segment end target T (K)")
    axes[1].set_title("Per-segment bath target (DCM:9 uses N=4)")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out = OUT_DIR / "heat_staged_ramp.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_cutoff_radius_ladder() -> Path:
    """COM-distance ladder: ML handoff, JAX MM pair reach, CHARMM IMAGE, jax-pme SR."""
    from mmml.interfaces.pycharmmInterface.nbonds_config import PBC_CUTNB

    cp = CutoffParameters()
    handoff_start = float(cp.mm_switch_on) - float(cp.ml_switch_width)
    mm_outer = float(cp.mm_switch_on) + float(cp.mm_switch_width)
    jax_pme_sr = 6.0
    charmm_cut = float(PBC_CUTNB)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_xlim(0, max(charmm_cut, mm_outer) + 4)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Monomer COM–COM distance r (Å)")

    bands = [
        (0, handoff_start, "#3b82f6", "ML dimer fully on (PhysNet)"),
        (handoff_start, cp.mm_switch_on, "#8b5cf6", "Handoff (s_ML + s_MM = 1)"),
        (cp.mm_switch_on, mm_outer, "#f97316", "Switched JAX MM pairs (LJ + MIC Coulomb)"),
        (mm_outer, charmm_cut + 2, "#e2e8f0", "No switched two-body MM/ML"),
    ]
    for x0, x1, color, label in bands:
        ax.axvspan(x0, x1, color=color, alpha=0.55, label=label)

    for x, ls, color, txt in (
        (handoff_start, "--", "#2563eb", f"handoff start {handoff_start:g} Å"),
        (cp.mm_switch_on, "-.", "#0f172a", f"mm_switch_on {cp.mm_switch_on:g} Å"),
        (mm_outer, "--", "#ea580c", f"JAX MM outer {mm_outer:g} Å"),
        (jax_pme_sr, ":", "#059669", f"jax-pme SR {jax_pme_sr:g} Å"),
        (charmm_cut, ":", "#64748b", f"CHARMM IMAGE cutnb {charmm_cut:g} Å (PBC)"),
    ):
        ax.axvline(x, color=color, ls=ls, lw=1.6, alpha=0.9)
        ax.text(x + 0.15, 0.92, txt, rotation=90, va="top", fontsize=7.5, color=color)

    ax.set_title(
        "Default cutoffs (8 / 5 / 1.5 Å): COM-distance regions vs CHARMM list radius",
        fontweight="500",
        pad=10,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "cutoff_radius_ladder.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_system_monomer_regions() -> Path:
    """Top-down cluster: monomers colored by residue; highlight one dimer's COM zones."""
    from matplotlib.patches import Circle
    from mmml.interfaces.crystal_charmm import default_make_res_monomer_pdb

    try:
        from ase.io import read
    except ImportError:
        raise RuntimeError("ASE required for system region figure")

    monomer = read(str(default_make_res_monomer_pdb("DCM")))
    rel = monomer.get_positions() - monomer.get_positions().mean(axis=0)
    com_xy = [
        np.array([0.0, 0.0]),
        np.array([10.5, 2.0]),
        np.array([5.0, 11.0]),
    ]
    colors = ["#2563eb", "#059669", "#d97706"]
    labels = ["Monomer 1 (DCM)", "Monomer 2 (DCM)", "Monomer 3 (DCM)"]

    cp = CutoffParameters()
    handoff_start = float(cp.mm_switch_on) - float(cp.ml_switch_width)
    mm_outer = float(cp.mm_switch_on) + float(cp.mm_switch_width)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.set_aspect("equal")
    ax.set_facecolor("#f8fafc")

    for idx, (com, color, label) in enumerate(zip(com_xy, colors, labels, strict=True)):
        pos = rel[:, :2] + com
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            s=120,
            c=color,
            edgecolors="#1e293b",
            linewidths=0.6,
            zorder=3,
            label=label,
        )
        ax.scatter([com[0]], [com[1]], s=40, c="#0f172a", marker="x", zorder=4)
        ax.text(com[0], com[1] - 1.8, f"COM {idx + 1}", ha="center", fontsize=8, color="#334155")

    # Highlight dimer 1–2 COM distance (~10.7 Å) — in MM tail, ML off
    c1, c2 = com_xy[0], com_xy[1]
    mid = 0.5 * (c1 + c2)
    r12 = float(np.linalg.norm(c2 - c1))
    for r, alpha, color in (
        (handoff_start, 0.12, "#3b82f6"),
        (cp.mm_switch_on, 0.10, "#8b5cf6"),
        (mm_outer, 0.08, "#f97316"),
    ):
        ax.add_patch(
            Circle(mid, r / 2.0, fill=False, ls="--", lw=1.4, ec=color, alpha=alpha, zorder=1)
        )
    ax.annotate(
        "",
        xy=c2,
        xytext=c1,
        arrowprops=dict(arrowstyle="<->", color="#64748b", lw=1.5),
        zorder=2,
    )
    ax.text(
        mid[0],
        mid[1] + 1.2,
        f"dimer COM distance r ≈ {r12:.1f} Å\n(MM tail, ML off)",
        ha="center",
        fontsize=8.5,
        color="#475569",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cbd5e1", alpha=0.95),
    )

    ax.set_xlim(-6, 16)
    ax.set_ylim(-6, 14)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(
        "System split by monomer: atoms inherit residue color; switches use COM–COM distance",
        fontweight="500",
        pad=10,
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
    out = OUT_DIR / "system_monomer_regions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- ASE dimer panels with illustrative hybrid forces at cutoff distances ---

_ILLUSTRATION_STYLE = {
    "figure_facecolor": "#f8fafc",
    "axes_facecolor": "#f8fafc",
    "bond_color": "#64748b",
    "bond_width": 1.2,
    "atom_edge": "#1e293b",
    "monomer_colors": ("#2563eb", "#ea580c"),
}

_K_COULOMB_KCAL = 332.063711
_LJ_EPS_KCAL = {"C": 0.11, "H": 0.03, "O": 0.21, "Cl": 0.25}
_LJ_SIG_A = {"C": 3.5, "H": 2.5, "O": 3.0, "Cl": 3.5}
_PARTIAL_Q = {"C": 0.0, "H": 0.09, "O": -0.45, "Cl": -0.12}


def _element_key(z: int) -> str:
    from ase.data import chemical_symbols

    sym = chemical_symbols[int(z)]
    return "Cl" if sym == "Cl" else sym


def _load_monomer_template(residue: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ase.io import read

    from mmml.interfaces.crystal_charmm import default_make_res_monomer_pdb
    from mmml.paths import default_aco_template_pdb

    res = residue.strip().upper()
    path = (
        default_aco_template_pdb()
        if res == "ACO"
        else default_make_res_monomer_pdb("DCM" if res == "DCM" else res)
    )
    atoms = read(str(path))
    pos = np.asarray(atoms.get_positions(), dtype=float)
    rel = pos - pos.mean(axis=0)
    z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    charges = np.array([_PARTIAL_Q.get(_element_key(zi), 0.0) for zi in z], dtype=float)
    return rel, z, charges


def _dimer_positions(
    mono_rel: np.ndarray, z: np.ndarray, charges: np.ndarray, com_dist: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n = int(mono_rel.shape[0])
    pos1 = mono_rel.copy()
    pos2 = mono_rel + np.array([float(com_dist), 0.0, 0.0], dtype=float)
    pos = np.vstack([pos1, pos2])
    z_full = np.concatenate([z, z])
    q_full = np.concatenate([charges, charges])
    return pos, z_full, q_full, n


def _com_distance(pos: np.ndarray, n_per: int) -> float:
    return float(np.linalg.norm(pos[n_per:].mean(axis=0) - pos[:n_per].mean(axis=0)))


def _lj_sigma_eps(za: int, zb: int) -> tuple[float, float]:
    ea, eb = _element_key(za), _element_key(zb)
    sig = 0.5 * (_LJ_SIG_A.get(ea, 3.4) + _LJ_SIG_A.get(eb, 3.4))
    eps = float(np.sqrt(_LJ_EPS_KCAL.get(ea, 0.1) * _LJ_EPS_KCAL.get(eb, 0.1)))
    return float(sig), eps


def _mm_cross_energy(pos: np.ndarray, z: np.ndarray, q: np.ndarray, n_per: int) -> float:
    energy = 0.0
    for i in range(n_per):
        for j in range(n_per, 2 * n_per):
            r = float(np.linalg.norm(pos[j] - pos[i]))
            r = max(r, 1.0)
            sig, eps = _lj_sigma_eps(int(z[i]), int(z[j]))
            sr = sig / r
            energy += 4.0 * eps * (sr**12 - sr**6)
            energy += _K_COULOMB_KCAL * float(q[i] * q[j]) / r
    return float(energy)


def _ml_com_energy(r_com: float) -> float:
    r0, amplitude = 5.2, 9.0
    return -amplitude * (r0 / max(r_com, 2.5)) ** 6


def _hybrid_energy(
    pos: np.ndarray, z: np.ndarray, q: np.ndarray, n_per: int, cp: CutoffParameters
) -> float:
    r = _com_distance(pos, n_per)
    s_ml = float(cp.ml_scale(r, gamma_ml=GAMMA_ON))
    s_mm = float(cp.mm_scale_complementary(r, gamma_ml=GAMMA_ON, gamma_mm_off=GAMMA_OFF))
    return s_ml * _ml_com_energy(r) + s_mm * _mm_cross_energy(pos, z, q, n_per)


def _hybrid_forces_fd(
    pos: np.ndarray,
    z: np.ndarray,
    q: np.ndarray,
    n_per: int,
    cp: CutoffParameters,
    *,
    delta: float = 1e-4,
) -> np.ndarray:
    forces = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for axis in range(3):
            pos[i, axis] += delta
            e_plus = _hybrid_energy(pos, z, q, n_per, cp)
            pos[i, axis] -= 2.0 * delta
            e_minus = _hybrid_energy(pos, z, q, n_per, cp)
            pos[i, axis] += delta
            forces[i, axis] = -(e_plus - e_minus) / (2.0 * delta)
    return forces


def _bond_segments_2d_panel(atoms, writer) -> np.ndarray:
    from ase.neighborlist import natural_cutoffs, neighbor_list

    cutoffs = natural_cutoffs(atoms, mult=1.08)
    i, j = neighbor_list("ij", atoms, cutoffs, self_interaction=False)
    if len(i) == 0:
        return np.empty((0, 2, 2))
    pos = atoms.get_positions()
    im_i = writer.to_image_plane_positions(pos[i])[:, :2]
    im_j = writer.to_image_plane_positions(pos[j])[:, :2]
    return np.stack([im_i, im_j], axis=1)


def _forces_in_image_plane(pos: np.ndarray, forces: np.ndarray, writer) -> np.ndarray:
    tail = writer.to_image_plane_positions(pos + forces)[:, :2]
    head = writer.to_image_plane_positions(pos)[:, :2]
    return tail - head


def _panel_distance_specs(cp: CutoffParameters) -> list[tuple[str, float, str]]:
    handoff_start = float(cp.mm_switch_on - cp.ml_switch_width)
    mm_outer = float(cp.mm_switch_on + cp.mm_switch_width)
    return [
        (
            "Pure ML",
            handoff_start - 0.7,
            f"r = {handoff_start - 0.7:.1f} Å (below handoff start {handoff_start:.1f} Å)",
        ),
        (
            "Handoff",
            0.5 * (handoff_start + float(cp.mm_switch_on)),
            f"r ≈ {0.5 * (handoff_start + float(cp.mm_switch_on)):.1f} Å — complementary switch",
        ),
        (
            "MM tail",
            0.5 * (float(cp.mm_switch_on) + mm_outer),
            f"r ≈ {0.5 * (float(cp.mm_switch_on) + mm_outer):.1f} Å — ML off, MM tapering",
        ),
        (
            "Beyond switched MM",
            mm_outer + 1.3,
            f"r = {mm_outer + 1.3:.1f} Å (past outer {mm_outer:.0f} Å)",
        ),
    ]


def _draw_dimer_forces_panel(
    ax,
    pos: np.ndarray,
    z: np.ndarray,
    forces: np.ndarray,
    n_per: int,
    *,
    zone: str,
    subtitle: str,
    rotation: str = "25x,18y,0z",
    force_norm=None,
) -> None:
    from ase import Atoms
    from ase.data import chemical_symbols
    from ase.visualize.plot import Matplotlib
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    symbols = [chemical_symbols[int(zi)] for zi in z]
    atoms = Atoms(symbols=symbols, positions=pos)
    writer = Matplotlib(
        atoms,
        ax,
        rotation=rotation,
        radii=0.85,
        scale=40.0,
        show_unit_cell=0,
        auto_bbox_size=1.05,
    )
    segments = _bond_segments_2d_panel(atoms, writer)
    if len(segments):
        ax.add_collection(
            LineCollection(
                segments,
                colors=_ILLUSTRATION_STYLE["bond_color"],
                linewidths=_ILLUSTRATION_STYLE["bond_width"],
                capstyle="round",
                zorder=1,
            )
        )
    im_pos = writer.to_image_plane_positions(pos)
    for i in range(len(pos)):
        color = _ILLUSTRATION_STYLE["monomer_colors"][0 if i < n_per else 1]
        ax.add_patch(
            Circle(
                im_pos[i, :2],
                radius=2.6,
                facecolor=color,
                edgecolor=_ILLUSTRATION_STYLE["atom_edge"],
                linewidth=0.55,
                zorder=3,
                alpha=0.96,
            )
        )
    f2d = _forces_in_image_plane(pos, forces, writer)
    fmag = np.linalg.norm(forces, axis=1)
    fmax = float(force_norm.vmax) if force_norm is not None else float(fmag.max() or 1.0)
    quiver_scale = max(0.25 * fmax, 0.05)
    ax.quiver(
        im_pos[:, 0],
        im_pos[:, 1],
        f2d[:, 0],
        f2d[:, 1],
        fmag,
        cmap="magma",
        norm=force_norm,
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
        width=0.0045,
        zorder=4,
        alpha=0.92,
    )
    ax.set_xlim(0, writer.w)
    ax.set_ylim(0, writer.h)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(f"{zone}\n{subtitle}", fontsize=8.5, fontweight="500", pad=4)


def plot_dimer_forces_cutoff_panels(residue: str) -> Path:
    """ASE orthographic dimer views with force quivers at key COM distances."""
    import matplotlib.colors as mcolors

    cp = CutoffParameters()
    mono_rel, z, charges = _load_monomer_template(residue)
    specs = _panel_distance_specs(cp)
    res_label = residue.strip().upper()

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), facecolor=_ILLUSTRATION_STYLE["figure_facecolor"])
    panel_rows: list[tuple] = []
    global_fmax = 0.0
    for ax, (zone, dist, subtitle) in zip(axes.ravel(), specs, strict=True):
        ax.set_facecolor(_ILLUSTRATION_STYLE["axes_facecolor"])
        pos, z_full, q_full, n_per = _dimer_positions(mono_rel, z, charges, dist)
        forces = _hybrid_forces_fd(pos, z_full, q_full, n_per, cp)
        global_fmax = max(global_fmax, float(np.linalg.norm(forces, axis=1).max()))
        panel_rows.append((ax, zone, dist, subtitle, pos, z_full, forces, n_per))

    force_norm = mcolors.Normalize(vmin=0.0, vmax=max(global_fmax, 1e-6))
    for ax, zone, dist, subtitle, pos, z_full, forces, n_per in panel_rows:
        r_com = _com_distance(pos, n_per)
        s_ml = float(cp.ml_scale(r_com, gamma_ml=GAMMA_ON))
        s_mm = float(cp.mm_scale_complementary(r_com, gamma_ml=GAMMA_ON, gamma_mm_off=GAMMA_OFF))
        ann = f"s_ML={s_ml:.2f}  s_MM={s_mm:.2f}  |F|_max={np.linalg.norm(forces, axis=1).max():.2f} kcal/mol/Å"
        _draw_dimer_forces_panel(
            ax,
            pos,
            z_full,
            forces,
            n_per,
            zone=zone,
            subtitle=f"{subtitle}\n{ann}",
            force_norm=force_norm,
        )

    sm = plt.cm.ScalarMappable(cmap="magma", norm=force_norm)
    sm.set_array([])
    fig.tight_layout(rect=(0, 0.06, 1, 0.78))
    # Dedicated axes at top so the colorbar never steals space from the 2×2 panels.
    cbar_ax = fig.add_axes([0.12, 0.90, 0.76, 0.022])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("|F| (kcal/mol/Å)", fontsize=9)
    fig.suptitle(
        f"{res_label} dimer — illustrative switched hybrid forces at cutoff distances "
        f"(default {DEFAULT_MM_SWITCH_ON:g} / {DEFAULT_MM_SWITCH_WIDTH:g} / {DEFAULT_ML_SWITCH_WIDTH:g} Å)",
        fontsize=10,
        fontweight="600",
        y=0.84,
    )
    fig.text(
        0.5,
        0.01,
        "Illustrative E = s_ML·E_ML(r_COM) + s_MM·E_MM(cross); arrows from −∇E (finite difference). "
        "Not a PhysNet/CHARMM evaluation — shows how force balance shifts across switch regions.",
        ha="center",
        fontsize=7.5,
        color="#475569",
    )
    slug = res_label.lower()
    out = OUT_DIR / f"{slug}_dimer_forces_cutoffs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_dual_stack_responsibilities() -> Path:
    """Which layer owns which physics on ML-tagged atoms."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axis("off")

    layers = [
        ("PhysNet ML", "#3b82f6", "Intra-monomer & dimer ML (sparse COM < mm_switch_on)"),
        ("JAX switched MM", "#f97316", "Cross-monomer LJ r⁻¹² + MIC Coulomb ≤ 13 Å (default)"),
        ("jax-pme LR", "#059669", "Cross-monomer Coulomb + r⁻⁶ tail (full − intra)"),
        ("CHARMM IMAGE", "#64748b", "VDW/ELEC on non-ML atoms; BLOCK zeros ELEC/VDW on ML atoms"),
        ("CHARMM bonded", "#94a3b8", "BOND/ANGL/DIHE (optional scaled internal MM on ML)"),
    ]
    y = 0.85
    for name, color, desc in layers:
        ax.add_patch(plt.Rectangle((0.05, y - 0.11), 0.22, 0.09, color=color, alpha=0.85))
        ax.text(0.16, y - 0.065, name, ha="center", va="center", color="white", fontsize=9, fontweight="600")
        ax.text(0.32, y - 0.065, desc, ha="left", va="center", fontsize=9, color="#334155")
        y -= 0.17

    ax.text(
        0.5,
        0.08,
        "Atoms are tagged ML in the PSF; monomers are never dropped — switching scales pair energies by COM distance.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    ax.set_title("Hybrid stack: who computes what on ML residues", fontweight="500", y=0.98)
    fig.tight_layout()
    out = OUT_DIR / "dual_stack_responsibilities.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_lr_solvers_overview() -> Path:
    """Compare long-range Coulomb backends."""
    solvers = [
        ("mic", "Truncated MIC\n(pair loop only)", "jax_mic", "No k-space", "#94a3b8"),
        ("jax_pme", "jax-pme Ewald/PME/P3M\n(cross-monomer full−intra)", "jax_mic", "JAX k-space", "#059669"),
        ("nvalchemiops_pme", "nvalchemiops PME\n(full-box Coulomb)", "periodic_external", "JAX k-space", "#2563eb"),
        ("scafacos", "ScaFaCoS libfcs\n(PME / P³M / P²NFFT …)", "periodic_external", "Fortran k-space", "#7c3aed"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(solvers) + 1)
    ax.axis("off")
    ax.set_title("Long-range Coulomb solvers (`lr_solver`)", fontweight="500", pad=12)

    for i, (key, desc, mode, kspace, color) in enumerate(solvers):
        y = len(solvers) - i
        ax.add_patch(plt.Rectangle((0.4, y - 0.35), 1.6, 0.7, color=color, alpha=0.9))
        ax.text(1.2, y, key, ha="center", va="center", color="white", fontsize=9, fontweight="600")
        ax.text(2.3, y, desc, ha="left", va="center", fontsize=9, color="#1e293b")
        ax.text(7.2, y, f"mm_nonbond_mode:\n{mode}", ha="left", va="center", fontsize=8, color="#475569")
        ax.text(9.0, y, kspace, ha="right", va="center", fontsize=8, color="#64748b")

    ax.text(
        5.0,
        0.35,
        "Default auto → jax_pme when installed. MIC is pair-loop only beyond 13 Å unless jax-pme is active.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    fig.tight_layout()
    out = OUT_DIR / "lr_solvers_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_lr_energy_split() -> Path:
    """Schematic: short-range vs long-range Coulomb split with jax-pme."""
    r = np.linspace(0.5, 20, 400)
    # Illustrative 1/r envelopes (not a real PME split)
    mic_mask = (r <= 13.0).astype(float)
    sr = mic_mask / np.maximum(r, 0.5)
    lr_tail = (1.0 / np.maximum(r, 0.5)) * (1.0 - mic_mask * 0.85)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(r, 0, sr, alpha=0.35, color="#f97316", label="JAX pair loop (MIC, ≤13 Å)")
    ax.fill_between(r, 0, lr_tail, alpha=0.35, color="#059669", label="jax-pme k-space tail (cross-monomer)")
    ax.axvline(13.0, color="#64748b", ls="--", lw=1.2, label="JAX MM outer radius (default)")
    ax.axvline(6.0, color="#059669", ls=":", lw=1.2, label="jax-pme SR cutoff (6 Å)")
    ax.set_xlabel("Intercharge distance (Å) — schematic")
    ax.set_ylabel("Relative Coulomb weight (illustrative)")
    ax.set_title("Coulomb split: switched pairs + jax-pme correction (not quantitative PME)", fontweight="500")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 20)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "lr_energy_split_schematic.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for slug, label, cp in CUTOFF_PRESETS:
        paths.append(plot_cutoff_preset(slug, label, cp))
    paths.append(plot_cutoff_comparison())
    paths.append(plot_legacy_vs_complementary())
    paths.append(plot_heat_segments())
    paths.append(plot_cutoff_radius_ladder())
    paths.append(plot_system_monomer_regions())
    paths.append(plot_dimer_forces_cutoff_panels("DCM"))
    paths.append(plot_dimer_forces_cutoff_panels("ACO"))
    paths.append(plot_dual_stack_responsibilities())
    paths.append(plot_lr_solvers_overview())
    paths.append(plot_lr_energy_split())
    print(f"Wrote {len(paths)} plots to {OUT_DIR}")
    for p in paths:
        print(f"  {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
