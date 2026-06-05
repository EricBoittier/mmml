#!/usr/bin/env python3
"""Generate ML/MM cutoff and heat-staging reference plots for docs/mlpot-settings.md."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "images" / "mlpot-settings"

from mmml.interfaces.pycharmmInterface.cutoffs import (  # noqa: E402
    GAMMA_OFF,
    GAMMA_ON,
    CutoffParameters,
)

CUTOFF_PRESETS: list[tuple[str, str, CutoffParameters]] = [
    (
        "code-default",
        "Code default (7 / 5 / 0.1 Å)",
        CutoffParameters(ml_switch_width=0.1, mm_switch_on=7.0, mm_switch_width=5.0),
    ),
    (
        "dcm9-stability",
        "DCM:9 stability script (7 / 5 / 0.1 Å)",
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
    cp = CutoffParameters(ml_switch_width=0.1, mm_switch_on=7.0, mm_switch_width=5.0)
    r = _r_grid(cp, pad=3.0)
    s_ml = cp.ml_scale(r, gamma_ml=GAMMA_ON)
    mm_comp = cp.mm_scale_complementary(r, gamma_ml=GAMMA_ON, gamma_mm_off=GAMMA_OFF)

    cp_legacy = CutoffParameters(
        ml_switch_width=0.1,
        mm_switch_on=7.0,
        mm_switch_width=5.0,
        complementary_handoff=False,
    )
    mm_legacy = cp_legacy.mm_scale(r, gamma_on=GAMMA_ON, gamma_off=GAMMA_OFF)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r, s_ml, lw=2, color="C0", label=r"$s_{\mathrm{ML}}$ (same)")
    ax.plot(r, mm_comp, lw=2, color="C1", label=r"$s_{\mathrm{MM}}$ complementary")
    ax.plot(r, mm_legacy, lw=2, ls=":", color="C3", label=r"$s_{\mathrm{MM}}$ legacy window")
    ax.set_xlabel("Dimer COM distance r (Å)")
    ax.set_ylabel("Scale factor")
    ax.set_title("DCM:9 cutoffs (7 / 5 / 0.1 Å): complementary vs legacy MM window")
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for slug, label, cp in CUTOFF_PRESETS:
        paths.append(plot_cutoff_preset(slug, label, cp))
    paths.append(plot_cutoff_comparison())
    paths.append(plot_legacy_vs_complementary())
    paths.append(plot_heat_segments())
    print(f"Wrote {len(paths)} plots to {OUT_DIR}")
    for p in paths:
        print(f"  {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
