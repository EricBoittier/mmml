#!/usr/bin/env python3
"""ICML figures for dense_dt_campaign arms that completed successfully.

Writes under ``artifacts/lj_scales/dense_dt_campaign/plots/``:
  - per-tag: thermo, RDFs, bond health, box snapshots (first/mid/last)
  - comparison overlays across passed NVT arms
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data.colors import jmol_colors
from ase.io import read as ase_read
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mmml.analysis.lattice_energy import unwrap_molecules  # noqa: E402
from mmml.utils.ase_structure_plot import (  # noqa: E402
    DOCS_STRUCTURE_STYLE,
    SCALE_BOX,
    draw_orthographic_structure,
    use_matplotlib_agg,
)
from mmml.utils.domdec_psf_order import read_psf_atoms_and_bonds  # noqa: E402
from mmml.utils.plotting.styles import (  # noqa: E402
    apply_plot_style,
    comparison_colors,
    legend_outside,
)
from mmml.utils.plotting.trajectory_structure import (  # noqa: E402
    element_pair_rdfs,
)

CAMPAIGN = ROOT / "artifacts/lj_scales/dense_dt_campaign"
PLOT_ROOT = CAMPAIGN / "plots"

PASSED = [
    ("L24_nvt_dt1_f32_50ps", 24, "artifacts/lj_scales/liquid_dense_L24/model.psf"),
    ("L24_nvt_dt05_x64_50ps", 24, "artifacts/lj_scales/liquid_dense_L24/model.psf"),
    ("L26_nvt_dt1_f32_50ps", 26, "artifacts/lj_scales/liquid_dense_L26/model.psf"),
    ("L30_nvt_dt05_x64_20ps", 30, "artifacts/lj_scales/liquid_nvt/mini.psf"),
]

_BOND_SOFT_MAX_A = {"C-H": 1.40, "H-C": 1.40, "C-Cl": 2.15, "Cl-C": 2.15}


def _find_h5(tag_dir: Path) -> Path | None:
    hs = sorted(tag_dir.glob("*.h5"))
    return hs[0] if hs else None


def _find_pdb(tag_dir: Path) -> Path | None:
    for name in (
        "pbc_nvt_jaxmd_pbc_minimized.pdb",
        "pbc_nvt_jaxmd_minimized.pdb",
        "pbc_npt_jaxmd_pbc_minimized.pdb",
        "pbc_nve_jaxmd_pbc_minimized.pdb",
    ):
        p = tag_dir / name
        if p.exists():
            return p
    pdbs = sorted(tag_dir.glob("*.pdb"))
    return pdbs[0] if pdbs else None


def psf_bond_pairs(psf_path: Path) -> list[tuple[int, int]]:
    _atoms, bonds = read_psf_atoms_and_bonds(psf_path)
    return [(int(i), int(j)) if i < j else (int(j), int(i)) for i, j in bonds]


def load_frames(h5_path: Path, numbers: np.ndarray, box: float) -> list[Atoms]:
    with h5py.File(h5_path, "r") as f:
        positions = np.asarray(f["positions"], dtype=float)
    frames: list[Atoms] = []
    for pos in positions:
        atoms = Atoms(numbers=numbers, positions=pos, cell=[box, box, box], pbc=True)
        atoms.wrap()
        frames.append(atoms)
    return frames


def load_thermo(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        out = {
            "time_ps": np.asarray(f["time_ps"], dtype=float),
            "temperature": np.asarray(f["temperature"], dtype=float),
            "potential_energy": np.asarray(f["potential_energy"], dtype=float),
            "total_energy": np.asarray(f["total_energy"], dtype=float),
            "kinetic_energy": np.asarray(f["kinetic_energy"], dtype=float),
        }
        if "invariant" in f:
            out["invariant"] = np.asarray(f["invariant"], dtype=float)
        if "density_g_cm3" in f:
            out["density_g_cm3"] = np.asarray(f["density_g_cm3"], dtype=float)
    return out


def mol_wrap(atoms: Atoms) -> Atoms:
    out = atoms.copy()
    _mol_id, unwrapped = unwrap_molecules(
        out.get_positions(), out.get_atomic_numbers(), np.asarray(out.cell)
    )
    out.set_positions(unwrapped)
    return out


def draw_psf_box(atoms: Atoms, bonds: list[tuple[int, int]], ax, *, rotation: str, scale: float):
    """Orthographic box view with PSF topological bonds (MIC-aware segments)."""
    from ase.geometry import find_mic
    from ase.visualize.plot import Matplotlib

    wrapped = mol_wrap(atoms)
    writer = Matplotlib(
        wrapped,
        ax,
        rotation=rotation,
        radii=0.55,
        scale=scale,
        show_unit_cell=1,
        auto_bbox_size=1.05,
    )
    pos = wrapped.get_positions()
    segs = []
    for i, j in bonds:
        vi = pos[i]
        vj = pos[j]
        vec, _ = find_mic(vj - vi, wrapped.cell, wrapped.pbc)
        im_i = writer.to_image_plane_positions(vi.reshape(1, 3))[0, :2]
        im_j = writer.to_image_plane_positions((vi + vec).reshape(1, 3))[0, :2]
        segs.append([im_i, im_j])
    if segs:
        ax.add_collection(
            LineCollection(
                segs,
                colors=DOCS_STRUCTURE_STYLE["bond_color"],
                linewidths=0.55,
                alpha=0.55,
                zorder=1,
            )
        )
    draw_orthographic_structure(
        wrapped, ax, rotation=rotation, scale=scale, show_unit_cell=1, radii=0.55, writer=writer
    )


def save_box_snapshots(
    frames: list[Atoms],
    bonds: list[tuple[int, int]],
    out_dir: Path,
    tag: str,
    times: np.ndarray,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    idxs = sorted({0, n // 2, n - 1})
    paths: list[Path] = []
    for idx in idxs:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
        for ax, rot, title in zip(
            axes,
            ("20x,25y,0z", "90x,0y,0z"),
            ("perspective", "top"),
        ):
            draw_psf_box(frames[idx], bonds, ax, rotation=rot, scale=SCALE_BOX)
            ax.set_title(title, fontsize=10)
        t = float(times[idx]) if idx < len(times) else float(idx)
        fig.suptitle(f"{tag}  ·  t = {t:.1f} ps  ·  PSF bonds", fontsize=11)
        fig.tight_layout()
        path = out_dir / f"box_t{t:05.1f}ps.png".replace(" ", "")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def bond_health(frames: list[Atoms], bonds: list[tuple[int, int]], times: np.ndarray):
    from ase.geometry import find_mic

    # Infer element labels from first frame
    syms = frames[0].get_chemical_symbols()
    max_len = []
    mean_len = []
    n_soft = []
    for fr in frames:
        pos = fr.get_positions()
        lengths = []
        soft = 0
        for i, j in bonds:
            vec, _ = find_mic(pos[j] - pos[i], fr.cell, fr.pbc)
            L = float(np.linalg.norm(vec))
            lengths.append(L)
            key = f"{syms[i]}-{syms[j]}"
            lim = _BOND_SOFT_MAX_A.get(key)
            if lim is not None and L > lim:
                soft += 1
        arr = np.asarray(lengths)
        max_len.append(float(arr.max()))
        mean_len.append(float(arr.mean()))
        n_soft.append(soft)
    return {
        "time_ps": times,
        "max_bond_A": np.asarray(max_len),
        "mean_bond_A": np.asarray(mean_len),
        "n_soft_outliers": np.asarray(n_soft),
    }


def plot_thermo(thermo: dict[str, np.ndarray], out: Path, tag: str) -> None:
    t = thermo["time_ps"]
    series = [
        ("total_energy", r"$E_\mathrm{tot}$ (eV)"),
        ("potential_energy", r"$E_\mathrm{pot}$ (eV)"),
        ("temperature", r"$T$ (K)"),
    ]
    if "invariant" in thermo:
        series.append(("invariant", r"$H_\mathrm{NHC}$ (eV)"))
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 2.1 * n), sharex=True)
    if n == 1:
        axes = [axes]
    colors = comparison_colors(apply_plot_style("icml"), n=n)
    for ax, (key, ylab), c in zip(axes, series, colors):
        y = thermo[key]
        ax.plot(t, y, color=c, lw=1.6)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.18)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        d = float(y[-1] - y[0])
        ax.set_title(f"Δ = {d:+.3g}", loc="right", fontsize=9)
    axes[-1].set_xlabel(r"$t$ (ps)")
    fig.suptitle(tag, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rdfs(radii, rdfs, out: Path, tag: str) -> None:
    preferred = ["C-C", "C-Cl", "Cl-Cl", "C-H", "Cl-H", "H-H"]
    keys = [k for k in preferred if k in rdfs] or sorted(rdfs)
    colors = comparison_colors(apply_plot_style("icml"), n=len(keys))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for key, c in zip(keys, colors):
        ax.plot(radii, rdfs[key], label=key, color=c, lw=1.7)
    ax.set_xlabel(r"$r$ (Å)")
    ax.set_ylabel(r"$g(r)$")
    ax.set_xlim(0.0, min(12.0, float(radii.max())))
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"Element-pair RDFs · {tag}", fontsize=11)
    ax.grid(alpha=0.18)
    legend_outside(ax, side="right", fontsize=8)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bond_health(bh: dict, out: Path, tag: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    t = bh["time_ps"]
    axes[0].plot(t, bh["max_bond_A"], color="#B9770E", lw=1.6, label="max")
    axes[0].plot(t, bh["mean_bond_A"], color="#1A5276", lw=1.4, label="mean")
    axes[0].axhline(2.15, color="#922B21", ls=":", lw=1.0, label="C–Cl soft max")
    axes[0].axhline(1.40, color="#922B21", ls="--", lw=0.9, label="C–H soft max")
    axes[0].set_ylabel(r"bond length (Å)")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(alpha=0.18)
    axes[1].plot(t, bh["n_soft_outliers"], color="#6C3483", lw=1.6)
    axes[1].set_ylabel("# soft outliers")
    axes[1].set_xlabel(r"$t$ (ps)")
    axes[1].grid(alpha=0.18)
    fig.suptitle(f"PSF bond health · {tag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(all_thermo: dict[str, dict[str, np.ndarray]], out: Path) -> None:
    tags = list(all_thermo)
    colors = comparison_colors(apply_plot_style("icml"), n=len(tags))
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 7.2), sharex=False)
    for tag, c in zip(tags, colors):
        th = all_thermo[tag]
        t = th["time_ps"]
        e = th["total_energy"]
        axes[0].plot(t, e - e[0], label=tag, color=c, lw=1.6)
        axes[1].plot(t, th["temperature"], label=tag, color=c, lw=1.4)
        if "invariant" in th:
            h = th["invariant"]
            axes[2].plot(t, h - h[0], label=tag, color=c, lw=1.6)
    axes[0].set_ylabel(r"$\Delta E_\mathrm{tot}$ (eV)")
    axes[1].set_ylabel(r"$T$ (K)")
    axes[2].set_ylabel(r"$\Delta H_\mathrm{NHC}$ (eV)")
    axes[2].set_xlabel(r"$t$ (ps)")
    for ax in axes:
        ax.grid(alpha=0.18)
    legend_outside(axes[0], side="right", fontsize=7)
    fig.suptitle("dense_dt_campaign · passed NVT arms", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rdf_overlay(all_rdfs: dict[str, tuple[np.ndarray, dict]], out: Path, pair: str = "Cl-Cl") -> None:
    tags = list(all_rdfs)
    colors = comparison_colors(apply_plot_style("icml"), n=len(tags))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for tag, c in zip(tags, colors):
        radii, rdfs = all_rdfs[tag]
        if pair not in rdfs:
            continue
        ax.plot(radii, rdfs[pair], label=tag, color=c, lw=1.7)
    ax.set_xlabel(r"$r$ (Å)")
    ax.set_ylabel(rf"$g_{{{pair}}}(r)$")
    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"{pair} RDF overlay · passed NVT", fontsize=11)
    ax.grid(alpha=0.18)
    legend_outside(ax, side="right", fontsize=7)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    use_matplotlib_agg()
    apply_plot_style("icml")
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    summary: dict = {"arms": [], "outputs": []}
    all_thermo: dict[str, dict[str, np.ndarray]] = {}
    all_rdfs: dict[str, tuple[np.ndarray, dict]] = {}

    for tag, box, psf_rel in PASSED:
        tag_dir = CAMPAIGN / tag
        h5 = _find_h5(tag_dir)
        pdb = _find_pdb(tag_dir)
        psf = ROOT / psf_rel
        if h5 is None or pdb is None or not psf.exists():
            print(f"SKIP {tag}: missing h5/pdb/psf")
            continue
        if not (tag_dir / "SUCCESS.flag").exists():
            # Still plot if RESULT rc=0
            bench = tag_dir / "bench.log"
            ok = bench.exists() and "RESULT" in bench.read_text() and "rc=0" in bench.read_text()
            if not ok:
                print(f"SKIP {tag}: not marked successful")
                continue

        out = PLOT_ROOT / tag
        out.mkdir(parents=True, exist_ok=True)
        print(f"=== {tag} ===")
        numbers = ase_read(str(pdb)).get_atomic_numbers()
        bonds = psf_bond_pairs(psf)
        frames = load_frames(h5, numbers, float(box))
        thermo = load_thermo(h5)
        print(f"  frames={len(frames)} bonds={len(bonds)}")

        plot_thermo(thermo, out / "thermo.png", tag)
        print("  wrote thermo.png")

        # RDF on later half when long; else all frames
        start = len(frames) // 2 if len(frames) >= 20 else 0
        radii, rdfs = element_pair_rdfs(frames[start:], r_max=12.0, bins=200)
        plot_rdfs(radii, rdfs, out / "element_pair_rdfs.png", tag)
        np.savez_compressed(out / "element_pair_rdfs.npz", radii=radii, **rdfs)
        print("  wrote element_pair_rdfs.png")

        bh = bond_health(frames, bonds, thermo["time_ps"])
        plot_bond_health(bh, out / "bond_health.png", tag)
        print("  wrote bond_health.png")

        snaps = save_box_snapshots(frames, bonds, out / "box_snapshots", tag, thermo["time_ps"])
        print(f"  wrote {len(snaps)} box snapshots")

        # Compact summary panel
        fig = plt.figure(figsize=(11.5, 8.5))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        t = thermo["time_ps"]
        ax0.plot(t, thermo["total_energy"] - thermo["total_energy"][0], color="#1A5276", lw=1.5)
        ax0.set_title(r"$\Delta E_\mathrm{tot}$")
        ax0.set_xlabel("t (ps)")
        ax0.grid(alpha=0.18)
        if "invariant" in thermo:
            ax1.plot(t, thermo["invariant"] - thermo["invariant"][0], color="#B9770E", lw=1.5)
        ax1.set_title(r"$\Delta H_\mathrm{NHC}$")
        ax1.set_xlabel("t (ps)")
        ax1.grid(alpha=0.18)
        for key, c in zip(["Cl-Cl", "C-Cl", "C-C"], comparison_colors(apply_plot_style("icml"), 3)):
            if key in rdfs:
                ax2.plot(radii, rdfs[key], label=key, color=c, lw=1.5)
        ax2.set_xlim(0, 12)
        ax2.set_ylim(bottom=0)
        ax2.set_title("RDFs")
        ax2.legend(fontsize=8, frameon=False)
        ax2.grid(alpha=0.18)
        draw_psf_box(frames[len(frames) // 2], bonds, ax3, rotation="20x,25y,0z", scale=SCALE_BOX)
        ax3.set_title(f"box @ t={float(t[len(frames)//2]):.1f} ps")
        fig.suptitle(tag, fontsize=12)
        panel = out / "summary_panel.png"
        fig.savefig(panel, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  wrote summary_panel.png")

        meta = {
            "tag": tag,
            "box_A": box,
            "n_frames": len(frames),
            "dE_tot": float(thermo["total_energy"][-1] - thermo["total_energy"][0]),
            "dH_NHC": float(thermo["invariant"][-1] - thermo["invariant"][0])
            if "invariant" in thermo
            else None,
            "max_bond_end_A": float(bh["max_bond_A"][-1]),
            "outputs": sorted(p.name for p in out.glob("**/*") if p.is_file()),
        }
        (out / "metrics.json").write_text(json.dumps(meta, indent=2))
        summary["arms"].append(meta)
        all_thermo[tag] = thermo
        all_rdfs[tag] = (radii, rdfs)

    if all_thermo:
        plot_comparison(all_thermo, PLOT_ROOT / "compare_thermo.png")
        plot_rdf_overlay(all_rdfs, PLOT_ROOT / "compare_rdf_ClCl.png", "Cl-Cl")
        plot_rdf_overlay(all_rdfs, PLOT_ROOT / "compare_rdf_CCl.png", "C-Cl")
        summary["outputs"].extend(
            ["compare_thermo.png", "compare_rdf_ClCl.png", "compare_rdf_CCl.png"]
        )
        print("wrote comparison overlays")

    (PLOT_ROOT / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# dense_dt_campaign — passed NVT plots",
        "",
        f"Generated under `{PLOT_ROOT}`",
        "",
    ]
    for arm in summary["arms"]:
        lines.append(
            f"- **{arm['tag']}**: ΔE={arm['dE_tot']:+.2f} eV, "
            f"ΔH_NHC={arm['dH_NHC']}, max_bond_end={arm['max_bond_end_A']:.2f} Å"
        )
    lines += [
        "",
        "Key files per tag: `thermo.png`, `element_pair_rdfs.png`, "
        "`bond_health.png`, `box_snapshots/`, `summary_panel.png`",
        "",
        "Overlays: `compare_thermo.png`, `compare_rdf_ClCl.png`, `compare_rdf_CCl.png`",
    ]
    (PLOT_ROOT / "README.md").write_text("\n".join(lines) + "\n")
    print("DONE →", PLOT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
