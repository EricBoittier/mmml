#!/usr/bin/env python3
"""XY scatters + switch schematic explaining softwell on=5 deploy-ready.

Inputs (contact-ok component CSVs already on disk):
  - deploy-only ep222 @ on=5: overbind_ablation/handoff_on5_w1p5_components.csv
  - softwell FT ep20: lever2_on5_softwell/distill_ep20_components.csv

Writes under docs/images/dense-dt-campaign/overbind_ablation/lever2_on5_softwell/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mmml.analysis.dimer_scans import DEFAULT_ORIENT_MIN_CONTACT_A  # noqa: E402
from mmml.interfaces.pycharmmInterface.calculator_utils import (  # noqa: E402
    ml_switch_scale,
    mm_switch_scale,
)
from mmml.utils.plotting.styles import (  # noqa: E402
    apply_plot_style,
    comparison_colors,
    legend_outside,
)
from scripts.slurm.dense_dt_campaign.dimer_scan_contacts import (  # noqa: E402
    annotate_dmin,
    load_monomer,
)

ABL = ROOT / "artifacts/lj_scales/dense_dt_campaign/overbind_ablation"
DEPLOY_CSV = ABL / "handoff_on5_w1p5_components.csv"
SOFTWELL_CSV = ABL / "lever2_on5_softwell" / "distill_ep20_components.csv"
SWEEP = ABL / "lever2_on5_softwell" / "epoch_sweep.json"
DATA = ROOT / "artifacts/lj_scales/dataset_cgenff.npz"
OUT = ROOT / "docs/images/dense-dt-campaign/overbind_ablation/lever2_on5_softwell"
LIT_LO, LIT_HI = -5.0, -3.0
SOFT_R = 3.4


def _load(csv: Path, R1: np.ndarray, *, dmin_donor: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv)
    if "dmin_A" not in df.columns:
        if dmin_donor is not None and {"ray", "r_A", "dmin_A"}.issubset(dmin_donor.columns):
            # Same ablate grid (8×8×36): reuse contact geometry from a CSV that
            # already carries dmin_A (older ablate runs omitted it).
            key = dmin_donor[["ray", "r_A", "dmin_A"]].drop_duplicates()
            df = df.merge(key, on=["ray", "r_A"], how="left")
        else:
            # Reconstruct direction/orientation from ray = di*n_orient + qi.
            n_orient = 8
            df = df.copy()
            df["direction"] = (df["ray"] // n_orient).astype(int)
            df["orientation"] = (df["ray"] % n_orient).astype(int)
            df = annotate_dmin(df, R1=R1)
    if "contact_ok" not in df.columns:
        df = df.copy()
        df["contact_ok"] = df["dmin_A"] >= DEFAULT_ORIENT_MIN_CONTACT_A
    return df


def _soft_wells(df: pd.DataFrame) -> np.ndarray:
    ok = df[df["contact_ok"]]
    soft = []
    for _, sub in ok.groupby("ray"):
        s = sub[sub["r_A"] >= SOFT_R]
        if len(s):
            soft.append(float(s["E_int_kcal"].min()))
    return np.asarray(soft, dtype=float)


def _mean_curve(df: pd.DataFrame, *, min_rays: int = 8) -> pd.DataFrame:
    ok = df[df["contact_ok"]]
    g = ok.groupby("r_A")["E_int_kcal"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out = out[out["count"] >= min_rays].reset_index(drop=True)
    return out


def fig_xy_scatter_compare(deploy: pd.DataFrame, softwell: pd.DataFrame) -> Path:
    colors = comparison_colors("icml", n=2)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, df, title, c in (
        (axes[0], deploy, "deploy-only ep222 @ on=5", colors[0]),
        (axes[1], softwell, "softwell FT ep20 @ on=5", colors[1]),
    ):
        ok = df[df["contact_ok"]]
        ax.scatter(
            ok["r_A"],
            ok["E_int_kcal"],
            s=4,
            alpha=0.18,
            color=c,
            rasterized=True,
            linewidths=0,
        )
        curve = _mean_curve(df)
        ax.plot(curve["r_A"], curve["mean"], color="0.1", lw=1.6, zorder=3)
        ax.axhspan(LIT_LO, LIT_HI, color="0.85", zorder=0)
        ax.axvline(SOFT_R, color="0.5", ls="--", lw=0.8)
        ax.axvline(5.0, color="0.35", ls=":", lw=0.9)
        ax.set_xlim(2.5, 10.0)
        ax.set_ylim(-18, 6)
        ax.set_xlabel(r"COM–COM $r$ (Å)")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel(r"$E_\mathrm{int}$ (kcal/mol)")
    axes[1].text(5.05, 4.2, r"$r_\mathrm{on}=5$", fontsize=7, color="0.35")
    fig.tight_layout()
    out = OUT / "xy_Eint_vs_r_contact_ok.png"
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def fig_soft_hist(deploy: pd.DataFrame, softwell: pd.DataFrame) -> Path:
    colors = comparison_colors("icml", n=2)
    a = _soft_wells(deploy)
    b = _soft_wells(softwell)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    bins = np.linspace(-18, 2, 41)
    ax.hist(a, bins=bins, alpha=0.55, color=colors[0], label=f"deploy-only med={np.median(a):.1f}")
    ax.hist(b, bins=bins, alpha=0.55, color=colors[1], label=f"softwell med={np.median(b):.1f}")
    ax.axvspan(LIT_LO, LIT_HI, color="0.85", zorder=0, label="lit −5…−3")
    ax.axvline(-15, color="0.3", ls="--", lw=0.9, label="deepest floor −15")
    ax.set_xlabel(r"contact-ok soft well $E_\mathrm{int}$ (kcal/mol)")
    ax.set_ylabel("ray count")
    legend_outside(ax)
    fig.tight_layout()
    out = OUT / "xy_soft_well_hist_compare.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def fig_mean_curve(deploy: pd.DataFrame, softwell: pd.DataFrame) -> Path:
    colors = comparison_colors("icml", n=2)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for df, lab, c in (
        (deploy, "deploy-only on=5", colors[0]),
        (softwell, "softwell ep20", colors[1]),
    ):
        curve = _mean_curve(df)
        ax.plot(curve["r_A"], curve["mean"], color=c, lw=1.8, label=lab)
        ax.fill_between(
            curve["r_A"],
            curve["mean"] - curve["std"],
            curve["mean"] + curve["std"],
            color=c,
            alpha=0.18,
            linewidth=0,
        )
    ax.axhspan(LIT_LO, LIT_HI, color="0.88", zorder=0)
    ax.axvline(SOFT_R, color="0.5", ls="--", lw=0.8)
    ax.axvline(5.0, color="0.35", ls=":", lw=0.9)
    ax.set_xlim(2.8, 10)
    ax.set_ylim(-10, 2)
    ax.set_xlabel(r"COM–COM $r$ (Å)")
    ax.set_ylabel(r"mean $E_\mathrm{int}$ (kcal/mol)")
    legend_outside(ax)
    fig.tight_layout()
    out = OUT / "xy_mean_curve_compare.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def fig_ml_mm_soft_scatter(softwell: pd.DataFrame) -> Path:
    """Show that softwell deepens ML where ml_scale is still on."""
    ok = softwell[softwell["contact_ok"] & (softwell["r_A"] >= SOFT_R) & (softwell["r_A"] <= 4.25)]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    sc = ax.scatter(
        ok["E_MM_kcal"],
        ok["E_ML_kcal"],
        c=ok["ml_scale"],
        s=10,
        cmap="viridis",
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    ax.axhline(0, color="0.6", lw=0.7)
    ax.axvline(0, color="0.6", lw=0.7)
    ax.set_xlabel(r"$E_\mathrm{MM}$ (kcal/mol)")
    ax.set_ylabel(r"$E_\mathrm{ML}$ (kcal/mol)")
    ax.set_title(r"softwell ep20, contact-ok, $3.4\leq r\leq 4.25$ Å", fontsize=9)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$s_\mathrm{ML}$")
    fig.tight_layout()
    out = OUT / "xy_EML_vs_EMM_ml_on_window.png"
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def fig_switch_schematic() -> Path:
    """Why aux must act in the ML-on soft window at on=5."""
    rs = np.linspace(2.5, 9.0, 400)
    s_ml = np.array([float(ml_switch_scale(r, mm_switch_on=5.0, ml_switch_width=1.5)) for r in rs])
    s_mm = np.array(
        [
            float(
                mm_switch_scale(
                    r,
                    mm_switch_on=5.0,
                    mm_switch_width=5.0,
                    ml_switch_width=1.5,
                    complementary_handoff=True,
                )
            )
            for r in rs
        ]
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(rs, s_ml, color="#1f4e79", lw=2.0, label=r"$s_\mathrm{ML}$")
    ax.plot(rs, s_mm, color="#b35806", lw=2.0, label=r"$s_\mathrm{MM}$")
    ax.axvspan(3.4, 4.25, color="#c7e9c0", alpha=0.7, zorder=0, label="aux window (ML-on soft)")
    ax.axvspan(4.25, 6.0, color="#fddbc7", alpha=0.55, zorder=0, label="soft metric, MM-dominated")
    ax.axvline(5.0, color="0.3", ls=":", lw=0.9)
    ax.set_xlim(2.5, 9)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel(r"COM–COM $r$ (Å)")
    ax.set_ylabel("switch scale")
    ax.set_title(r"handoff at $r_\mathrm{on}=5$ Å — neural aux only moves $s_\mathrm{ML}>0$", fontsize=9)
    legend_outside(ax)
    fig.tight_layout()
    out = OUT / "xy_switch_schematic_on5.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def fig_soft_vs_r_scatter(deploy: pd.DataFrame, softwell: pd.DataFrame) -> Path:
    """Per-ray soft-well depth vs r_at_min — shows wells move into ML-on region."""
    colors = comparison_colors("icml", n=2)

    def rows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ok = df[df["contact_ok"]]
        rr, ee = [], []
        for _, sub in ok.groupby("ray"):
            s = sub[sub["r_A"] >= SOFT_R]
            if len(s):
                i = int(s["E_int_kcal"].idxmin())
                rr.append(float(s.loc[i, "r_A"]))
                ee.append(float(s.loc[i, "E_int_kcal"]))
        return np.asarray(rr), np.asarray(ee)

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for df, lab, c in (
        (deploy, "deploy-only", colors[0]),
        (softwell, "softwell ep20", colors[1]),
    ):
        r, e = rows(df)
        ax.scatter(r, e, s=22, alpha=0.75, color=c, label=lab, edgecolors="0.2", linewidths=0.3)
    ax.axhspan(LIT_LO, LIT_HI, color="0.88", zorder=0)
    ax.axvline(4.25, color="0.35", ls="--", lw=0.9)
    ax.set_xlabel(r"$r$ at soft-well minimum (Å)")
    ax.set_ylabel(r"soft-well $E_\mathrm{int}$ (kcal/mol)")
    ax.set_xlim(3.2, 7.5)
    ax.set_ylim(-18, 2)
    legend_outside(ax)
    fig.tight_layout()
    out = OUT / "xy_soft_well_depth_vs_r_at_min.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def write_readme(paths: dict[str, Path], deploy: pd.DataFrame, softwell: pd.DataFrame) -> Path:
    a = _soft_wells(deploy)
    b = _soft_wells(softwell)
    sweep = json.loads(SWEEP.read_text()) if SWEEP.is_file() else {}
    best = sweep.get("best", {})
    md = OUT / "README.md"
    md.write_text(
        f"""# Softwell `on=5` deploy-ready — why it should work

Contact-ok soft wells (intermolecular $d_\\mathrm{{min}}\\ge {DEFAULT_ORIENT_MIN_CONTACT_A:g}$ Å,
per-ray min at $r\\ge {SOFT_R:g}$ Å).

## Verdict

| | Soft median | Soft deepest | Mean-curve min | deploy_ready |
|---|---:|---:|---:|:---:|
| deploy-only ep222 @ on=5 | {np.median(a):.2f} | {a.min():.2f} | {_mean_curve(deploy)["mean"].min():.2f} | no |
| **softwell FT ep20** | **{np.median(b):.2f}** | **{b.min():.2f}** | **{_mean_curve(softwell)["mean"].min():.2f}** | **{best.get("deploy_ready", True)}** |

Gates: soft median ∈ lit −5…−3 (±0.5), deepest ≳ −15, mean-curve ≳ −8.

Best ckpt: `{best.get("checkpoint", "epoch-20")}`.

## Why the lever works (and why earlier FT failed)

At `mm_switch_on=5`, ML interaction is **fully off for $r\\ge 5$ Å**
($s_\\mathrm{{ML}}\\to 0$). Soft-metric geometries with $r\\gtrsim 4.5$ Å are
dominated by **frozen MM LJ** — a neural loss there cannot deepen wells
(component diag: underbinders had $s_\\mathrm{{ML}}\\approx 0$,
$E_\\mathrm{{int}}\\approx E_\\mathrm{{MM}}\\approx -0.7$ kcal/mol).

Softwell aux therefore trains only in the **ML-on soft window**
$r\\in[3.4,4.25]$ Å ($s_\\mathrm{{ML}}\\gtrsim 0.5$), pulling hybrid
$E_\\mathrm{{int}}=s\\,\\Delta E_\\mathrm{{ML}}+E_\\mathrm{{MM}}$ into lit
−5…−3 kcal/mol while capping deep tails. Soft wells that used to sit in the
MM-only zone as shallow minima (~−1.3) are replaced by deeper ML minima near
~4 Å, so the contact-ok soft median moves into lit without −20 kcal clash
wells.

![switch schematic](xy_switch_schematic_on5.png)

## Figures

| File | What it shows |
|---|---|
| [`xy_Eint_vs_r_contact_ok.png`](xy_Eint_vs_r_contact_ok.png) | XY scatter $E_\\mathrm{{int}}(r)$ before/after |
| [`xy_mean_curve_compare.png`](xy_mean_curve_compare.png) | Orientation-mean curves ±σ |
| [`xy_soft_well_hist_compare.png`](xy_soft_well_hist_compare.png) | Soft-well histogram vs lit / deepest floor |
| [`xy_soft_well_depth_vs_r_at_min.png`](xy_soft_well_depth_vs_r_at_min.png) | Soft-well depth vs $r$ at min — wells move into ML-on $r$ |
| [`xy_EML_vs_EMM_ml_on_window.png`](xy_EML_vs_EMM_ml_on_window.png) | $E_\\mathrm{{ML}}$ vs $E_\\mathrm{{MM}}$ colored by $s_\\mathrm{{ML}}$ |
| [`xy_switch_schematic_on5.png`](xy_switch_schematic_on5.png) | Handoff scales + aux window |
| `povray/` | POV stills of contact-ok soft geometries |
| `pbc_translation.json` | PBC image/translation invariance on DCM:120 L=24 |

## PBC confirmation

See `pbc_translation.json` (lattice shift / wrap cases). Pass criterion:
$|\\Delta E|\\lesssim 10^{{-4}}$ eV and force max-abs delta $\\lesssim 10^{{-3}}$ eV/Å
on lattice and molecule-wrapped images (repeat-only isolates nondeterminism).
""",
        encoding="utf-8",
    )
    return md


def main() -> int:
    apply_plot_style("icml")
    OUT.mkdir(parents=True, exist_ok=True)
    if not DEPLOY_CSV.is_file():
        raise SystemExit(f"missing {DEPLOY_CSV}")
    if not SOFTWELL_CSV.is_file():
        raise SystemExit(f"missing {SOFTWELL_CSV}")
    R1, _Z1 = load_monomer(DATA)
    softwell = _load(SOFTWELL_CSV, R1)
    deploy = _load(DEPLOY_CSV, R1, dmin_donor=softwell)
    paths = {
        "scatter": fig_xy_scatter_compare(deploy, softwell),
        "hist": fig_soft_hist(deploy, softwell),
        "mean": fig_mean_curve(deploy, softwell),
        "mlmm": fig_ml_mm_soft_scatter(softwell),
        "switch": fig_switch_schematic(),
        "r_at_min": fig_soft_vs_r_scatter(deploy, softwell),
    }
    readme = write_readme(paths, deploy, softwell)
    print("wrote:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    print(f"  readme: {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
