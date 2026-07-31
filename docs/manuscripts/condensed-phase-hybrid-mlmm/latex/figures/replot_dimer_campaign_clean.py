#!/usr/bin/env python3
"""Publication-quality QC interaction curves from dimer-scan campaign CSV.

The archived campaign PNGs mixed raw totals and SpookyNet contact artifacts.
Manuscript panels use cleaned interaction energies for MP2 / PBE0-D3(BJ)
(and GFN2-xTB when it has a soft well), with MP2 lateral-offset families.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = Path(
    "/mmhome/boittier/home/mmml/results/dimer_scan_campaign/"
    "mbd_checkpoint_comparison/scan_results_clean.csv"
)
OUT = Path(__file__).resolve().parent

QC_PRIMARY = "mp2_def2svp_gpu4pyscf_cp"
QC_SECOND = "pbe0_def2svp_gpu4pyscf_d3bj_cp"
XTB = "xtb_gfn2"

COLORS = {
    "family": "#1f4e5f",
    "mp2": "#1f4e5f",
    "pbe0": "#2a9d8f",
    "xtb": "#b85c38",
}

PAIR_CFG = {
    ("DCM", "DCM"): dict(
        ylim=(-4.0, 8.0),
        xlim=(2.8, 11.0),
        bold_offset=0.0,
        show_xtb=False,
    ),
    ("ACE", "ACE"): dict(
        # Short-r ACE QC points were largely rejected; Δ=2.5 retains a clean
        # long-range / soft contact window where xTB shows a shallow well.
        ylim=(-5.0, 6.0),
        xlim=(2.2, 12.0),
        bold_offset=2.5,
        show_xtb=True,
        xtb_offset=2.5,
    ),
    ("DCM", "TIP3"): dict(
        ylim=(-4.0, 6.0),
        xlim=(2.5, 11.0),
        bold_offset=0.0,
        show_xtb=True,
        xtb_offset=0.0,
    ),
}


def _series(
    df: pd.DataFrame, a: str, b: str, backend: str, offset: float
) -> tuple[np.ndarray, np.ndarray]:
    g = df[
        (df.molecule_a == a)
        & (df.molecule_b == b)
        & (df.backend == backend)
        & np.isclose(df.offset_angstrom, offset)
    ].sort_values("distance_angstrom")
    if g.empty:
        return np.asarray([]), np.asarray([])
    r = g.distance_angstrom.to_numpy(float)
    e = g.interaction_kcal_mol_for_cleaning.to_numpy(float)
    m = np.isfinite(r) & np.isfinite(e)
    return r[m], e[m]


def _clip(r: np.ndarray, e: np.ndarray, y_hi: float) -> tuple[np.ndarray, np.ndarray]:
    m = e <= y_hi + 6.0
    return r[m], e[m]


def _mark_min(ax, r, e, color):
    if r.size == 0 or float(np.min(e)) >= -0.05:
        return
    i = int(np.argmin(e))
    ax.scatter(
        [r[i]],
        [e[i]],
        s=28,
        color=color,
        zorder=6,
        edgecolors="white",
        linewidths=0.55,
    )


def plot_pair(df: pd.DataFrame, a: str, b: str) -> dict:
    cfg = PAIR_CFG[(a, b)]
    fig, ax = plt.subplots(figsize=(5.2, 3.55), dpi=180)
    ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)

    offsets = sorted(
        df[(df.molecule_a == a) & (df.molecule_b == b) & (df.backend == QC_PRIMARY)]
        .offset_angstrom.unique()
        .tolist()
    )
    family_labeled = False
    for off in offsets:
        r, e = _series(df, a, b, QC_PRIMARY, off)
        if r.size < 3:
            continue
        r, e = _clip(r, e, cfg["ylim"][1])
        label = "MP2 (lateral offsets)" if not family_labeled else None
        ax.plot(
            r,
            e,
            color=COLORS["family"],
            alpha=0.18,
            lw=1.05,
            zorder=1,
            label=label,
        )
        family_labeled = True

    bold = cfg["bold_offset"]
    metrics = {"pair": f"{a}_{b}", "offset": bold}

    r, e = _series(df, a, b, QC_PRIMARY, bold)
    if r.size >= 3:
        r, e = _clip(r, e, cfg["ylim"][1])
        ax.plot(r, e, color=COLORS["mp2"], lw=2.3, zorder=4, label=f"MP2 (Δ={bold:g} Å)")
        _mark_min(ax, r, e, COLORS["mp2"])
        metrics["mp2_min"] = float(np.min(e))
        metrics["mp2_r"] = float(r[int(np.argmin(e))])

    r, e = _series(df, a, b, QC_SECOND, bold)
    if r.size >= 3:
        r, e = _clip(r, e, cfg["ylim"][1])
        ax.plot(
            r,
            e,
            color=COLORS["pbe0"],
            lw=1.9,
            ls="--",
            zorder=3,
            label=f"PBE0-D3(BJ) (Δ={bold:g} Å)",
        )
        _mark_min(ax, r, e, COLORS["pbe0"])
        metrics["pbe0_min"] = float(np.min(e))
        metrics["pbe0_r"] = float(r[int(np.argmin(e))])

    if cfg.get("show_xtb"):
        xoff = cfg.get("xtb_offset", bold)
        r, e = _series(df, a, b, XTB, xoff)
        if r.size >= 3 and float(np.min(e)) < 0.5:
            r, e = _clip(r, e, cfg["ylim"][1])
            ax.plot(
                r,
                e,
                color=COLORS["xtb"],
                lw=1.7,
                ls=":",
                zorder=2,
                label=f"GFN2-xTB (Δ={xoff:g} Å)",
            )
            _mark_min(ax, r, e, COLORS["xtb"])
            metrics["xtb_min"] = float(np.min(e))
            metrics["xtb_r"] = float(r[int(np.argmin(e))])

    ax.set_xlabel(r"$r_{\mathrm{COM}}$ / Å")
    ax.set_ylabel(r"$E_{\mathrm{int}}$ / kcal mol$^{-1}$")
    ax.set_title(f"{a}–{b} interaction scans")
    ax.set_xlim(*cfg["xlim"])
    ax.set_ylim(*cfg["ylim"])
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    stem = f"{a}_{b}"
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png  {metrics}")
    return metrics


def plot_combined(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=180)
    for ax, pair in zip(axes, [("DCM", "DCM"), ("DCM", "TIP3")]):
        a, b = pair
        cfg = PAIR_CFG[pair]
        ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)
        offsets = sorted(
            df[(df.molecule_a == a) & (df.molecule_b == b) & (df.backend == QC_PRIMARY)]
            .offset_angstrom.unique()
            .tolist()
        )
        for off in offsets:
            r, e = _series(df, a, b, QC_PRIMARY, off)
            if r.size < 3:
                continue
            r, e = _clip(r, e, cfg["ylim"][1])
            ax.plot(r, e, color=COLORS["family"], alpha=0.18, lw=1.0, zorder=1)
        bold = cfg["bold_offset"]
        r, e = _series(df, a, b, QC_PRIMARY, bold)
        if r.size >= 3:
            r, e = _clip(r, e, cfg["ylim"][1])
            ax.plot(r, e, color=COLORS["mp2"], lw=2.3, label=f"MP2 (Δ={bold:g} Å)")
            _mark_min(ax, r, e, COLORS["mp2"])
        r, e = _series(df, a, b, QC_SECOND, bold)
        if r.size >= 3:
            r, e = _clip(r, e, cfg["ylim"][1])
            ax.plot(
                r,
                e,
                color=COLORS["pbe0"],
                lw=1.9,
                ls="--",
                label=f"PBE0-D3(BJ) (Δ={bold:g} Å)",
            )
            _mark_min(ax, r, e, COLORS["pbe0"])
        if cfg.get("show_xtb"):
            xoff = cfg.get("xtb_offset", bold)
            r, e = _series(df, a, b, XTB, xoff)
            if r.size >= 3 and float(np.min(e)) < 0.5:
                r, e = _clip(r, e, cfg["ylim"][1])
                ax.plot(
                    r,
                    e,
                    color=COLORS["xtb"],
                    lw=1.7,
                    ls=":",
                    label=f"GFN2-xTB (Δ={xoff:g} Å)",
                )
                _mark_min(ax, r, e, COLORS["xtb"])
        ax.set_xlabel(r"$r_{\mathrm{COM}}$ / Å")
        ax.set_title(f"{a}–{b}")
        ax.set_xlim(*cfg["xlim"])
        ax.set_ylim(*cfg["ylim"])
        ax.legend(frameon=False, fontsize=7.0, loc="upper right")
    axes[0].set_ylabel(r"$E_{\mathrm{int}}$ / kcal mol$^{-1}$")
    fig.suptitle("Rigid dimer interaction scans (cleaned QC)", fontsize=10.5, y=1.02)
    fig.tight_layout()
    out = OUT / "dimer_scan_DCM_ACE.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")
    return out


def main() -> int:
    import json

    df = pd.read_csv(CSV)
    metrics = [plot_pair(df, *pair) for pair in PAIR_CFG]
    plot_combined(df)
    (OUT / "dimer_scan_qc_metrics.json").write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
