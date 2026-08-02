#!/usr/bin/env python3
"""ICML dimer interaction-profile figures for dense_dt_campaign.

Mean curves / envelopes exclude intermolecular clashes
(``dmin_A < DEFAULT_ORIENT_MIN_CONTACT_A``). COM–COM ``r`` alone is not a
steric coordinate for DCM — unfiltered wells look spuriously deep.
"""
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

from mmml.analysis.dimer_scans import DEFAULT_ORIENT_MIN_CONTACT_A  # noqa: E402
from mmml.utils.plotting.styles import (  # noqa: E402
    apply_plot_style,
    comparison_colors,
    legend_outside,
)
from scripts.slurm.dense_dt_campaign.dimer_scan_contacts import (  # noqa: E402
    annotate_dmin,
    contact_filtered_metrics,
    load_monomer,
)

SCAN = ROOT / "artifacts/lj_scales/dense_dt_campaign/dimer_scans"
SCAN_UNIT = ROOT / "artifacts/lj_scales/dense_dt_campaign/dimer_scans_unit"
OUT = ROOT / "docs/images/dense-dt-campaign/dimer_scans"
DATA = ROOT / "artifacts/lj_scales/dataset_cgenff.npz"


def load_annotated(csv_path: Path, R1: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "dmin_A" not in df.columns:
        df = annotate_dmin(df, R1=R1)
        df.to_csv(csv_path, index=False)
    elif "contact_ok" not in df.columns:
        df["contact_ok"] = df["dmin_A"] >= DEFAULT_ORIENT_MIN_CONTACT_A
    return df


def load_mean_curve(
    df: pd.DataFrame, *, min_contact: float, min_rays: int
) -> pd.DataFrame:
    ok = df[df["dmin_A"] >= min_contact]
    if ok.empty:
        raise SystemExit(f"no points with dmin_A >= {min_contact} Å in scan CSV")
    out = (
        ok.groupby("r_A")[["E_int_kcal", "E_MM_kcal", "E_ML_kcal", "ml_scale"]]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    flat = pd.DataFrame({"r_A": out["r_A"]})
    for col in ("E_int_kcal", "E_MM_kcal", "E_ML_kcal", "ml_scale"):
        flat[f"{col}_mean"] = out[(col, "mean")].to_numpy()
        flat[f"{col}_std"] = out[(col, "std")].to_numpy()
        flat[f"{col}_min"] = out[(col, "min")].to_numpy()
        flat[f"{col}_max"] = out[(col, "max")].to_numpy()
    flat["n_rays"] = out[("E_int_kcal", "count")].to_numpy().astype(int)
    # Drop sparsely covered short-r bins (a handful of barely-clearing rays).
    flat = flat[flat["n_rays"] >= min_rays].reset_index(drop=True)
    if flat.empty:
        raise SystemExit(
            f"no r-bins with ≥{min_rays} contact-ok rays; relax min_contact/min_rays"
        )
    return flat


def main() -> int:
    apply_plot_style("icml")
    OUT.mkdir(parents=True, exist_ok=True)
    min_contact = DEFAULT_ORIENT_MIN_CONTACT_A
    R1, _ = load_monomer(DATA)

    learned_csv = SCAN / "orient_components.csv"
    unit_csv = SCAN_UNIT / "orient_components.csv"
    learned_df = load_annotated(learned_csv, R1)
    unit_df = load_annotated(unit_csv, R1) if unit_csv.exists() else None

    metrics = contact_filtered_metrics(learned_df, min_contact=min_contact)
    min_rays = int(metrics["min_rays_for_mean"])
    learned = load_mean_curve(learned_df, min_contact=min_contact, min_rays=min_rays)
    unit = (
        load_mean_curve(unit_df, min_contact=min_contact, min_rays=min_rays)
        if unit_df is not None
        else None
    )
    colors = comparison_colors(apply_plot_style("icml"), n=5)

    r = learned["r_A"].to_numpy()
    e = learned["E_int_kcal_mean"].to_numpy()
    emin = learned["E_int_kcal_min"].to_numpy()
    emax = learned["E_int_kcal_max"].to_numpy()
    n_rays = learned["n_rays"].to_numpy()

    # Fig 1 — full profile (contact-ok only)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.fill_between(
        r, emin, emax, color=colors[0], alpha=0.14, label="learned orientation envelope"
    )
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
    ax.axhspan(-5, -3, color="0.85", alpha=0.45, label="lit. DCM dimer ~−3…−5")
    ax.set_xlabel(r"COM–COM distance $r$ (Å)")
    ax.set_ylabel(r"$E_\mathrm{int}$ (kcal mol$^{-1}$)")
    ax.set_title(
        "DCM–DCM hybrid 1D interaction profile\n"
        f"(contact-ok only: $d_\\mathrm{{min}}\\geq{min_contact:g}$ Å; epoch222)"
    )
    ax.set_xlim(float(r.min()), 12.0)
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
    ax.axhspan(-5, -3, color="0.85", alpha=0.45, label="lit. ~−3…−5")
    ax.set_xlabel(r"COM–COM distance $r$ (Å)")
    ax.set_ylabel(r"$E_\mathrm{int}$ (kcal mol$^{-1}$)")
    ax.set_title(f"DCM–DCM well region ($d_\\mathrm{{min}}\\geq{min_contact:g}$ Å)")
    ax.set_xlim(float(r.min()), 7.0)
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
    ax.set_title(f"Mean hybrid decomposition (contact-ok, $d_\\mathrm{{min}}\\geq{min_contact:g}$ Å)")
    ax.set_xlim(float(r.min()), 12.0)
    ax.grid(alpha=0.18)
    ax2 = ax.twinx()
    ax2.set_ylabel(r"$s_\mathrm{ML}$")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(False)
    legend_outside(ax, side="right", fontsize=8)
    fig.savefig(OUT / "dcm_dimer_components_mean.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Coverage: how many rays survive the contact cut at each r
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.plot(r, n_rays, color=colors[0], lw=1.8)
    ax.set_xlabel(r"COM–COM distance $r$ (Å)")
    ax.set_ylabel("n rays (contact-ok)")
    ax.set_title(f"Orientation coverage after $d_\\mathrm{{min}}\\geq{min_contact:g}$ Å cut")
    ax.set_xlim(2.5, 12.0)
    ax.grid(alpha=0.18)
    fig.savefig(OUT / "dcm_dimer_contact_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Copy multi-ray panels from the scan (raw; contact-filtered summary is authoritative)
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

    raw_sum = json.loads((SCAN / "summary.json").read_text()) if (SCAN / "summary.json").exists() else {}
    unit_sum = (
        json.loads((SCAN_UNIT / "summary.json").read_text()) if (SCAN_UNIT / "summary.json").exists() else {}
    )
    slim_metrics = {
        k: v
        for k, v in metrics.items()
        if k not in ("soft_wells", "ray_wells", "mean_curve")
    }
    slim_metrics["deepest_soft"] = metrics.get("deepest_soft_ray")
    (SCAN / "summary_contact_filtered.json").write_text(json.dumps(slim_metrics, indent=2) + "\n")

    summary = {
        "learned_raw": raw_sum,
        "unit_raw": unit_sum,
        "contact_filter": slim_metrics,
        "mean_well_kcal_learned": metrics["mean_curve_min_kcal"],
        "r_at_mean_well_A": metrics["r_at_mean_curve_min_A"],
        "median_soft_well_kcal": metrics["median_soft_well_kcal"],
        "deepest_soft_kcal": metrics["deepest_soft_kcal"],
        "note": (
            f"Metrics exclude intermolecular clashes (dmin < {min_contact:g} Å). "
            "Unfiltered COM scans mix steric overlaps into the well statistics "
            "(raw deepest ≈ −55 kcal/mol). Contact-ok soft wells sit near the "
            "literature DCM dimer band (~−3 to −5 kcal/mol)."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# DCM–DCM hybrid 1D dimer interaction profiles",
                "",
                "Rigid COM–COM scans with the dense_dt_campaign hybrid checkpoint",
                "(epoch222 + LJ-scale sidecar). 96 orientations × 48 distances.",
                "",
                f"**Contact policy:** metrics and mean curves keep only points with",
                f"intermolecular atom–atom $d_\\mathrm{{min}} \\geq {min_contact:g}$ Å",
                f"(`DEFAULT_ORIENT_MIN_CONTACT_A`). COM distance alone is not steric —",
                "unfiltered wells are dominated by Cl/H clashes.",
                "",
                "| Figure | Content |",
                "|---|---|",
                "| `dcm_dimer_Eint_profile.png` | Contact-ok mean + envelope; learned vs unit |",
                "| `dcm_dimer_Eint_zoom.png` | Well-region zoom |",
                "| `dcm_dimer_components_mean.png` | Mean ML / MM / total decomposition |",
                "| `dcm_dimer_contact_coverage.png` | n rays surviving the dmin cut vs r |",
                "| `hybrid_orient_DCM_epoch222_*.png` | Raw multi-ray panels (include clashes) |",
                "| `povray/` | Clash-filtered POV stills (forces / dipoles / charge) |",
                "",
                f"- Contact-ok mean well: **{metrics['mean_curve_min_kcal']:.1f} kcal/mol** "
                f"at r ≈ {metrics['r_at_mean_curve_min_A']:.2f} Å "
                f"({metrics['n_rays_at_mean_curve_min']} rays)",
                f"- Contact-ok soft-well median / deepest: "
                f"**{metrics['median_soft_well_kcal']:.1f}** / "
                f"**{metrics['deepest_soft_kcal']:.1f}** kcal/mol",
                f"- Raw (unfiltered) deepest soft well was "
                f"**{raw_sum.get('deepest_soft_kcal', 'n/a')}** kcal/mol — clash-dominated",
                "",
                summary["note"],
                "",
                "Regenerate profiles:",
                "```bash",
                "uv run python scripts/slurm/dense_dt_campaign/plot_dimer_profiles.py",
                "```",
                "",
            ]
        )
        + "\n"
    )
    print("DONE →", OUT)
    print(
        f"  contact-ok mean well {metrics['mean_curve_min_kcal']:.2f} @ "
        f"{metrics['r_at_mean_curve_min_A']:.2f} Å; "
        f"soft median {metrics['median_soft_well_kcal']:.2f}; "
        f"soft deepest {metrics['deepest_soft_kcal']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
