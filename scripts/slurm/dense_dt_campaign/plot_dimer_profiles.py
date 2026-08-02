#!/usr/bin/env python3
"""ICML dimer interaction-profile figures for dense_dt_campaign."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mmml.utils.plotting.styles import (  # noqa: E402
    apply_plot_style,
    comparison_colors,
    legend_outside,
)

SCAN = ROOT / "artifacts/lj_scales/dense_dt_campaign/dimer_scans"
SCAN_UNIT = ROOT / "artifacts/lj_scales/dense_dt_campaign/dimer_scans_unit"
OUT = ROOT / "docs/images/dense-dt-campaign/dimer_scans"


def load_mean_curve(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = (
        df.groupby("r_A")[["E_int_kcal", "E_MM_kcal", "E_ML_kcal", "ml_scale"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    # flatten
    flat = pd.DataFrame({"r_A": out["r_A"]})
    for col in ("E_int_kcal", "E_MM_kcal", "E_ML_kcal", "ml_scale"):
        flat[f"{col}_mean"] = out[(col, "mean")].to_numpy()
        flat[f"{col}_std"] = out[(col, "std")].to_numpy()
        flat[f"{col}_min"] = out[(col, "min")].to_numpy()
        flat[f"{col}_max"] = out[(col, "max")].to_numpy()
    return flat


def main() -> int:
    apply_plot_style("icml")
    OUT.mkdir(parents=True, exist_ok=True)

    learned_csv = SCAN / "orient_components.csv"
    unit_csv = SCAN_UNIT / "orient_components.csv"
    learned = load_mean_curve(learned_csv)
    unit = load_mean_curve(unit_csv) if unit_csv.exists() else None
    colors = comparison_colors(apply_plot_style("icml"), n=5)

    r = learned["r_A"].to_numpy()
    e = learned["E_int_kcal_mean"].to_numpy()
    emin = learned["E_int_kcal_min"].to_numpy()
    emax = learned["E_int_kcal_max"].to_numpy()

    # Fig 1 — full profile
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.fill_between(r, emin, emax, color=colors[0], alpha=0.14, label="learned orientation envelope")
    ax.plot(r, e, color=colors[0], lw=2.0, label="learned LJ scales (mean)")
    if unit is not None:
        ax.plot(
            unit["r_A"],
            unit["E_int_kcal_mean"],
            color=colors[1],
            lw=1.7,
            ls="--",
            label="unit LJ scales (mean)",
        )
    ax.axhline(0.0, color="#666666", lw=0.8, ls=":")
    ax.set_xlabel(r"COM–COM distance $r$ (Å)")
    ax.set_ylabel(r"$E_\mathrm{int}$ (kcal mol$^{-1}$)")
    ax.set_title("DCM–DCM hybrid 1D interaction profile\n(96 orientations × 48 $r$; epoch222)")
    ax.set_xlim(2.5, 12.0)
    ax.grid(alpha=0.18)
    legend_outside(ax, side="right", fontsize=8)
    fig.savefig(OUT / "dcm_dimer_Eint_profile.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "dcm_dimer_Eint_profile.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig 2 — zoom
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    m = r <= 7.0
    ax.fill_between(r[m], emin[m], emax[m], color=colors[0], alpha=0.15)
    ax.plot(r[m], e[m], color=colors[0], lw=2.0, label="learned (mean)")
    if unit is not None:
        ru = unit["r_A"].to_numpy()
        mu = ru <= 7.0
        ax.plot(
            ru[mu],
            unit["E_int_kcal_mean"].to_numpy()[mu],
            color=colors[1],
            lw=1.7,
            ls="--",
            label="unit (mean)",
        )
    ax.axhline(0.0, color="#666666", lw=0.8, ls=":")
    ax.set_xlabel(r"COM–COM distance $r$ (Å)")
    ax.set_ylabel(r"$E_\mathrm{int}$ (kcal mol$^{-1}$)")
    ax.set_title("DCM–DCM well region")
    ax.set_xlim(2.5, 7.0)
    ax.grid(alpha=0.18)
    legend_outside(ax, side="right", fontsize=8)
    fig.savefig(OUT / "dcm_dimer_Eint_zoom.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Fig 3 — mean ML / MM / total decomposition
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(r, learned["E_int_kcal_mean"], color=colors[0], lw=2.0, label=r"$E_\mathrm{int}$")
    ax.plot(r, learned["E_ML_kcal_mean"], color=colors[2], lw=1.6, label=r"$E_\mathrm{ML}$ (scaled)")
    ax.plot(r, learned["E_MM_kcal_mean"], color=colors[3], lw=1.6, label=r"$E_\mathrm{MM}$")
    ax.plot(r, learned["ml_scale_mean"], color=colors[4], lw=1.2, ls=":", label=r"$s_\mathrm{ML}$ (right)")
    ax.axhline(0.0, color="#666666", lw=0.8, ls=":")
    ax.set_xlabel(r"COM–COM distance $r$ (Å)")
    ax.set_ylabel(r"energy (kcal mol$^{-1}$)")
    ax.set_title("Mean hybrid decomposition (DCM–DCM)")
    ax.set_xlim(2.5, 12.0)
    ax.grid(alpha=0.18)
    ax2 = ax.twinx()
    ax2.set_ylabel(r"$s_\mathrm{ML}$")
    ax2.set_ylim(-0.05, 1.05)
    # hide twin line (already plotted on ax); just for scale label clarity
    ax2.grid(False)
    legend_outside(ax, side="right", fontsize=8)
    fig.savefig(OUT / "dcm_dimer_components_mean.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Copy multi-ray panels from the scan
    for name in (
        "hybrid_orient_DCM_epoch222_Eint.png",
        "hybrid_orient_DCM_epoch222_Eint_zoom.png",
        "hybrid_orient_DCM_epoch222_components.png",
        "hybrid_orient_DCM_epoch222_components_mean.png",
        "hybrid_orient_DCM_epoch222_meanF.png",
    ):
        src = SCAN / name
        if src.exists():
            shutil.copy2(src, OUT / name)

    learned_sum = json.loads((SCAN / "summary.json").read_text()) if (SCAN / "summary.json").exists() else {}
    unit_sum = (
        json.loads((SCAN_UNIT / "summary.json").read_text()) if (SCAN_UNIT / "summary.json").exists() else {}
    )
    summary = {
        "learned": learned_sum,
        "unit": unit_sum,
        "mean_well_kcal_learned": float(np.nanmin(e)),
        "r_at_mean_well_A": float(r[int(np.nanargmin(e))]),
        "note": (
            "DCM–DCM hybrid wells are far deeper than literature (~−3 to −5 kcal/mol). "
            "Learned vs unit LJ scales are nearly identical here — the overbinding is "
            "dominated by the ML/electrostatic handoff, which rationalizes the dense "
            "NVT droplet collapse."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# DCM–DCM hybrid 1D dimer interaction profiles",
                "",
                "Rigid COM–COM scans with the dense_dt_campaign hybrid checkpoint",
                "(epoch222 + LJ-scale sidecar). 96 orientations × 48 distances.",
                "",
                "| Figure | Content |",
                "|---|---|",
                "| `dcm_dimer_Eint_profile.png` | Mean + orientation envelope; learned vs unit scales |",
                "| `dcm_dimer_Eint_zoom.png` | Well-region zoom |",
                "| `dcm_dimer_components_mean.png` | Mean ML / MM / total decomposition |",
                "| `hybrid_orient_DCM_epoch222_*.png` | Full multi-ray panels from the scan script |",
                "",
                f"- Mean well (learned): **{summary['mean_well_kcal_learned']:.1f} kcal/mol** "
                f"at r ≈ {summary['r_at_mean_well_A']:.2f} Å",
                f"- Deepest soft well (scan summary): **{learned_sum.get('deepest_soft_kcal', 'n/a')} kcal/mol**",
                "",
                summary["note"],
                "",
            ]
        )
        + "\n"
    )
    print("DONE →", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
