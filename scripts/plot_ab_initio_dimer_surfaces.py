#!/usr/bin/env python3
"""Merge and plot collected counterpoise-corrected ab-initio dimer scans.

For every molecular pair, all available levels of theory are restricted to the
same finite, non-clashing (distance, offset) geometry intersection.  Two panel
sets are written: a fixed -10..20 kcal/mol comparison and a common symmetric-
log full-range diagnostic.  Duplicate calculations are audited rather than
silently averaged.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmml.utils.plotting.styles import apply_plot_style


METHOD_LABELS = {
    "hf_def2svp_gpu4pyscf_cp": "HF/def2-SVP",
    "mp2_def2svp_gpu4pyscf_cp": "MP2/def2-SVP",
    "pbe0_def2svp_gpu4pyscf_cp": "PBE0/def2-SVP",
    "pbe0_def2svp_gpu4pyscf_d3bj_cp": "PBE0-D3BJ/def2-SVP",
    "ccsd_def2svp_gpu4pyscf_cp": "CCSD/def2-SVP",
    "ccsd_def2svpd_gpu4pyscf_cp": "CCSD/def2-SVPD",
}
KEYS = ["molecule_a", "molecule_b", "distance_angstrom", "offset_angstrom", "backend"]


def _load(inputs: list[Path], min_contact: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frames = []
    for priority, path in enumerate(inputs):
        frame = pd.read_csv(path)
        frame = frame[frame.backend.isin(METHOD_LABELS)].copy()
        frame["source_file"] = str(path)
        frame["source_priority"] = priority
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw["energy_kcal_mol"] = pd.to_numeric(raw.energy_kcal_mol, errors="coerce")
    raw["min_contact_angstrom"] = pd.to_numeric(raw.min_contact_angstrom, errors="coerce")

    counts = raw.groupby(KEYS, dropna=False).size()
    duplicate_keys = counts[counts > 1].index
    duplicate_rows = raw.set_index(KEYS).loc[duplicate_keys].reset_index() if len(duplicate_keys) else raw.iloc[0:0]
    raw = raw.sort_values("source_priority").drop_duplicates(KEYS, keep="last")
    raw = raw[np.isfinite(raw.energy_kcal_mol) & (raw.min_contact_angstrom >= min_contact)].copy()

    kept = []
    coverage: dict[str, dict] = {}
    for pair, pair_frame in raw.groupby(["molecule_a", "molecule_b"], sort=False):
        methods = sorted(pair_frame.backend.unique(), key=lambda x: list(METHOD_LABELS).index(x))
        common = None
        for method in methods:
            idx = pd.MultiIndex.from_frame(
                pair_frame[pair_frame.backend == method][["distance_angstrom", "offset_angstrom"]]
            )
            common = idx if common is None else common.intersection(idx)
        assert common is not None
        indexed = pair_frame.set_index(["distance_angstrom", "offset_angstrom"])
        masked = indexed.loc[indexed.index.isin(common)].reset_index()
        kept.append(masked)
        coverage[f"{pair[0]}-{pair[1]}"] = {
            "methods": [METHOD_LABELS[m] for m in methods],
            "common_geometries": int(len(common)),
            "min_contact_angstrom": float(masked.min_contact_angstrom.min()),
        }
    return pd.concat(kept, ignore_index=True), duplicate_rows, coverage


def _plot_pair(frame: pd.DataFrame, pair: tuple[str, str], output: Path, *, full_range: bool) -> None:
    subset = frame[(frame.molecule_a == pair[0]) & (frame.molecule_b == pair[1])]
    methods = sorted(subset.backend.unique(), key=lambda x: list(METHOD_LABELS).index(x))
    ncols = 2 if len(methods) == 4 else min(3, len(methods))
    nrows = int(np.ceil(len(methods) / ncols))
    values = subset.energy_kcal_mol.to_numpy()
    if full_range:
        magnitude = max(1.0, float(np.nanmax(np.abs(values))))
        norm = mcolors.SymLogNorm(linthresh=1.0, vmin=-magnitude, vmax=magnitude)
        suffix = "full range (shared symmetric log)"
    else:
        norm = mcolors.Normalize(vmin=-10.0, vmax=20.0)
        suffix = "shared scale [-10, 20] kcal mol$^{-1}$"
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_under("#34205c")
    cmap.set_over("#5a160f")
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * ncols + 0.7, 2.5 * nrows + 0.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for ax, method in zip(axes.flat, methods):
        data = subset[subset.backend == method]
        pivot = data.pivot(index="offset_angstrom", columns="distance_angstrom", values="energy_kcal_mol")
        z = pivot.to_numpy()
        image = ax.pcolormesh(pivot.columns, pivot.index, z, cmap=cmap, norm=norm, shading="nearest")
        finite = z[np.isfinite(z)]
        ax.set_title(METHOD_LABELS[method])
        ax.text(
            0.02,
            0.04,
            f"min {finite.min():.3g}\nmax {finite.max():.3g}",
            transform=ax.transAxes,
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        if not full_range:
            saturated = int(np.count_nonzero((finite < -10.0) | (finite > 20.0)))
            if saturated:
                ax.text(0.98, 0.96, f"{saturated} sat.", transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_xlabel("separation (Å)")
        ax.set_ylabel("offset (Å)")
    for ax in axes.flat[len(methods):]:
        ax.set_visible(False)
    if image is not None:
        fig.colorbar(image, ax=axes, label="counterpoise interaction energy (kcal mol$^{-1}$)", extend="both")
    fig.suptitle(f"{pair[0]}–{pair[1]} ab-initio surfaces — {suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-contact", type=float, default=1.5)
    parser.add_argument("--html", type=Path, help="Optional inline-visualization HTML fragment")
    parser.add_argument("--html-pair", nargs=2, action="append", metavar=("A", "B"))
    args = parser.parse_args()
    apply_plot_style("icml")
    frame, duplicates, coverage = _load(args.input, args.min_contact)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "scan_results_ab_initio_identical_mask.csv", index=False)
    duplicates.to_csv(args.output_dir / "duplicate_calculations.csv", index=False)
    (args.output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2))
    for pair in frame[["molecule_a", "molecule_b"]].drop_duplicates().itertuples(index=False, name=None):
        stem = f"{pair[0]}_{pair[1]}"
        _plot_pair(frame, pair, args.output_dir / f"{stem}_ab_initio_shared.png", full_range=False)
        _plot_pair(frame, pair, args.output_dir / f"{stem}_ab_initio_full.png", full_range=True)

    if args.html:
        selected = args.html_pair or [("TIP3", "TIP3"), ("DCM", "DCM"), ("DCM", "TIP3")]
        figures = []
        for pair in selected:
            path = args.output_dir / f"{pair[0]}_{pair[1]}_ab_initio_shared.png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            label = f"{pair[0]}–{pair[1]} ab-initio surfaces on a shared energy scale"
            figures.append(
                f'<figure><img src="data:image/png;base64,{encoded}" alt="{label}">'
                f'<figcaption class="text-small text-muted">{label}</figcaption></figure>'
            )
        fragment = (
            '<div id="mmml-ab-initio-surfaces">\n'
            '<style>\n'
            '#mmml-ab-initio-surfaces{display:grid;gap:16px;color:var(--foreground);}\n'
            '#mmml-ab-initio-surfaces figure{margin:0;}\n'
            '#mmml-ab-initio-surfaces img{display:block;width:100%;height:auto;}\n'
            '#mmml-ab-initio-surfaces figcaption{margin-top:4px;}\n'
            '</style>\n'
            + "\n".join(figures)
            + '\n</div>\n'
        )
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(fragment)


if __name__ == "__main__":
    main()
