#!/usr/bin/env python3
"""Download and visualize the ``aaa.ama`` ``dataset_aaa.npz`` training set.

Writes plots for MkDocs and a JSON summary::

    uv run python scripts/analyze_aaa_ama_dataset.py
    uv run python scripts/analyze_aaa_ama_dataset.py --npz /path/to/dataset_aaa.npz

Figures land in ``docs/images/examples/aaa-ama/``.
Summary JSON: ``mmml/data/external/aaa_ama_dataset_summary.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
IMG = REPO / "docs" / "images" / "examples" / "aaa-ama"
SUMMARY = REPO / "mmml" / "data" / "external" / "aaa_ama_dataset_summary.json"
DEFAULT_NPZ = REPO / "mmml" / "data" / "external" / "dataset_aaa.npz"


def _use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _structure_figure(data: dict[str, np.ndarray], out: Path) -> None:
    from ase import Atoms
    from ase.io import write as ase_write

    z = np.asarray(data["Z"][0], dtype=int)
    r = np.asarray(data["R"][0], dtype=float)
    symbols = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    sym = [symbols[i - 1] if 1 <= i <= len(symbols) else "X" for i in z]
    atoms = Atoms(symbols=sym, positions=r)
    atoms.center()
    out.parent.mkdir(parents=True, exist_ok=True)
    ase_write(out.with_suffix(".png"), atoms, rotation="10x,10y,10z")
    # Also emit matplotlib 2D projection for docs consistency
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    colors = {1: "#ffffff", 6: "#909090", 7: "#3050f8", 8: "#ff0d0d"}
    for zi, pos in zip(z, r):
        ax.scatter(pos[0], pos[1], s=80, c=colors.get(int(zi), "#333"), edgecolors="k", linewidths=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title("aaa.ama frame 0 (peptide)")
    ax.set_facecolor("#f8fafc")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _histogram_figures(data: dict[str, np.ndarray], img_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from mmml.data.external.aaa_ama import inspect_dataset_aaa, per_element_force_magnitudes

    report = inspect_dataset_aaa(data)
    e = np.asarray(data["E"], dtype=float).ravel()
    f_mag = np.linalg.norm(np.asarray(data["F"]), axis=-1).ravel()
    by_elem = per_element_force_magnitudes(data)

    img_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=140)
    ax.hist(e, bins=60, color="#2563eb", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Frames")
    ax.set_title(f"Total energy — {report.molecule_label}")
    ax.set_facecolor("#f8fafc")
    fig.tight_layout()
    fig.savefig(img_dir / "energy_histogram.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=140)
    ax.hist(f_mag, bins=80, color="#059669", alpha=0.85, edgecolor="white", range=(0, 15))
    ax.set_xlabel("|F| (eV/Å)")
    ax.set_ylabel("Atom×frame samples")
    ax.set_title("Force magnitude (all atoms)")
    ax.set_facecolor("#f8fafc")
    fig.tight_layout()
    fig.savefig(img_dir / "force_histogram.png", bbox_inches="tight")
    plt.close(fig)

    colors = {"H": "#94a3b8", "C": "#475569", "N": "#3b82f6", "O": "#ef4444"}
    n_by_sym = {sp.symbol: sp.n_atoms for sp in report.element_species}
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=140, sharex=True)
    for ax, sym in zip(axes.ravel(), ("H", "C", "N", "O")):
        vals = by_elem.get(sym)
        if vals is None:
            ax.set_visible(False)
            continue
        ax.hist(vals, bins=50, color=colors.get(sym, "#64748b"), alpha=0.9, edgecolor="white")
        ax.set_title(f"{sym}  (n={n_by_sym.get(sym, 0)} atoms/frame)")
        ax.set_ylabel("Samples")
        ax.set_facecolor("#f8fafc")
    for ax in axes[-1, :]:
        ax.set_xlabel("|F| (eV/Å)")
    fig.suptitle("Force magnitude by element", y=1.02)
    fig.tight_layout()
    fig.savefig(img_dir / "force_by_element.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=140)
    ax.plot(e, lw=0.6, color="#7c3aed", alpha=0.8)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Energy trace (MD samples)")
    ax.set_facecolor("#f8fafc")
    fig.tight_layout()
    fig.savefig(img_dir / "energy_trace.png", bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ, help="NPZ path")
    parser.add_argument("--download", action="store_true", help="Fetch NPZ from GitHub")
    parser.add_argument("--img-dir", type=Path, default=IMG)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args(argv)

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = REPO / summary_path
    img_dir = Path(args.img_dir)
    if not img_dir.is_absolute():
        img_dir = REPO / img_dir
    npz_path = Path(args.npz)
    if not npz_path.is_absolute():
        npz_path = REPO / npz_path

    if args.download or not npz_path.is_file():
        from mmml.data.external.aaa_ama import download_dataset_aaa

        print(f"Downloading {npz_path} …")
        download_dataset_aaa(npz_path)

    from mmml.data.external.aaa_ama import (
        inspect_dataset_aaa,
        load_dataset_aaa,
        write_report_json,
    )

    data = load_dataset_aaa(npz_path)
    report = inspect_dataset_aaa(data)
    write_report_json(report, summary_path)
    print(f"wrote {summary_path.relative_to(REPO)}")
    print(f"  frames={report.n_frames}  atoms={report.n_atoms}  Q={report.net_charge:+.0f}")
    print(f"  molecule: {report.molecule_label}")

    _use_agg()
    _structure_figure(data, img_dir / "peptide_frame0")
    _histogram_figures(data, img_dir)
    print(f"wrote figures under {img_dir.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
