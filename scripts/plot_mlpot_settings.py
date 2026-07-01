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
    paths.append(plot_dual_stack_responsibilities())
    paths.append(plot_lr_solvers_overview())
    paths.append(plot_lr_energy_split())
    print(f"Wrote {len(paths)} plots to {OUT_DIR}")
    for p in paths:
        print(f"  {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
