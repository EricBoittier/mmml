#!/usr/bin/env python3
"""Plot DCM/ACO dimer COM scans comparing LR solvers and MM backends.

Reads NPZ files from ``run_dcm_aco_dimer_lr_scans.sh`` (or manual
``scan_mlpot_dimer_2d_pycharmm.py`` runs) and writes comparison PNGs.

Example::

    uv run python scripts/plot_dimer_lr_scan_compare.py \\
      --root artifacts/dimer_lr_scans
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
EV_PER_KCAL = 1.0 / 23.0605


def _load_npz_meta(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    meta = {
        "path": str(path),
        "composition": str(data["composition"].item()) if "composition" in data else "",
        "scan_tag": str(data["scan_tag"].item()) if "scan_tag" in data else path.parent.name,
        "lr_solver_active": str(data["lr_solver_active"].item())
        if "lr_solver_active" in data
        else "",
        "mm_nonbond_mode": str(data["mm_nonbond_mode"].item())
        if "mm_nonbond_mode" in data
        else "",
        "jax_pme_method": str(data["jax_pme_method"].item())
        if "jax_pme_method" in data
        else "",
    }
    d1 = np.asarray(data["scan_2d_d01_A"], dtype=float)
    hybrid = np.asarray(data["scan_2d_hybrid_energy_kcal"], dtype=float)
    ml2b = np.asarray(data["scan_2d_ml_2b_E_kcal"], dtype=float)
    mm = np.asarray(data["scan_2d_mm_E_kcal"], dtype=float) if "scan_2d_mm_E_kcal" in data else None
    # 1D slice: first column for scan_1d, else diagonal for square grids
    if hybrid.ndim == 1 or hybrid.shape[1] == 1:
        y_hybrid = hybrid.reshape(-1)
        y_ml2b = ml2b.reshape(-1)
        y_mm = mm.reshape(-1) if mm is not None else None
        x = d1.reshape(-1)
    else:
        n = min(hybrid.shape)
        idx = np.arange(n)
        y_hybrid = hybrid[idx, idx]
        y_ml2b = ml2b[idx, idx]
        y_mm = mm[idx, idx] if mm is not None else None
        x = d1
    return {
        **meta,
        "d01_A": x,
        "hybrid_kcal": y_hybrid,
        "ml2b_kcal": y_ml2b,
        "mm_kcal": y_mm,
    }


def discover_npz_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/scan_1d.npz")) + sorted(root.glob("**/scan_2d.npz"))


def plot_composition_compare(series: list[dict[str, Any]], *, composition: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_e, ax_c = axes
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(series), 1)))

    for color, rec in zip(colors, series, strict=False):
        label = rec["scan_tag"] or rec["lr_solver_active"]
        ax_e.plot(rec["d01_A"], rec["hybrid_kcal"], lw=2, color=color, label=label)
        ax_c.plot(rec["d01_A"], rec["ml2b_kcal"], lw=1.8, color=color, ls="-", label=f"{label} ML2B")
        if rec["mm_kcal"] is not None:
            ax_c.plot(
                rec["d01_A"],
                rec["mm_kcal"],
                lw=1.2,
                color=color,
                ls="--",
                alpha=0.85,
                label=f"{label} MM",
            )

    cp_on = 8.0
    cp_ml = 6.5
    cp_mm = 13.0
    for ax in axes:
        ax.axvline(cp_ml, color="#8b5cf6", ls=":", lw=1, alpha=0.7)
        ax.axvline(cp_on, color="#0f172a", ls="-.", lw=1, alpha=0.7)
        ax.axvline(cp_mm, color="#f97316", ls=":", lw=1, alpha=0.7)
    ax_e.set_ylabel("Hybrid energy (kcal/mol)")
    ax_c.set_ylabel("Component energy (kcal/mol)")
    ax_c.set_xlabel("Dimer COM distance d₀₁ (Å)")
    ax_e.set_title(f"{composition} dimer scan — LR solver / backend comparison")
    ax_e.legend(loc="best", fontsize=7, ncol=2)
    ax_c.legend(loc="best", fontsize=6, ncol=2)
    ax_e.grid(alpha=0.3)
    ax_c.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cutoff_bands(out: Path) -> None:
    """Small reference panel for default cutoff lines used in scans."""
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.set_xlim(3, 15)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("COM distance (Å)")
    bands = [
        (3, 6.5, "#3b82f6", "ML on"),
        (6.5, 8.0, "#8b5cf6", "handoff"),
        (8.0, 13.0, "#f97316", "MM tail"),
    ]
    for x0, x1, c, _ in bands:
        ax.axvspan(x0, x1, color=c, alpha=0.45)
    ax.set_title("Default scan cutoffs (8 / 5 / 1.5 Å)", fontweight="500")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO / "artifacts" / "dimer_lr_scans",
        help="Root directory containing scan NPZ trees",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot output directory (default: <root>/plots)",
    )
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    out_dir = (args.output_dir or (root / "plots")).expanduser().resolve()
    files = discover_npz_files(root)
    if not files:
        print(f"No scan NPZ files under {root}")
        return 1

    by_comp: dict[str, list[dict[str, Any]]] = {}
    index: list[dict[str, str]] = []
    for path in files:
        rec = _load_npz_meta(path)
        comp = rec["composition"] or "unknown"
        by_comp.setdefault(comp, []).append(rec)
        index.append(
            {
                "composition": comp,
                "scan_tag": rec["scan_tag"],
                "lr_solver_active": rec["lr_solver_active"],
                "mm_nonbond_mode": rec["mm_nonbond_mode"],
                "path": rec["path"],
            }
        )

    plot_cutoff_bands(out_dir / "cutoff_reference.png")
    for comp, series in sorted(by_comp.items()):
        tag = comp.replace(":", "_").lower()
        plot_composition_compare(series, composition=comp, out=out_dir / f"{tag}_lr_compare.png")

    (out_dir / "scan_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote plots to {out_dir} ({len(files)} scans)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
