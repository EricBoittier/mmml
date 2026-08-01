#!/usr/bin/env python3
"""Combine quantitative plots with transparent glossy POV-Ray structures."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from PIL import Image

from mmml.utils.plotting.styles import apply_plot_style
from scripts.render_povray_style_catalog import scene

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/images/povray-overlays"
FONT = "/System/Library/Fonts/Supplemental/Verdana.ttf"


def _render(atoms: Atoms, stem: Path, *, forces: bool = True,
            geometry: bool = False, width: int = 720, height: int = 540) -> Path:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pov = stem.with_suffix(".pov")
    ini = stem.with_suffix(".ini")
    png = stem.with_suffix(".png")
    pov.write_text(scene(atoms, "glossy", "", vectors=forces,
                         geometry=geometry, font=FONT, width=width, height=height))
    ini.write_text(f'Input_File_Name="{pov.name}"\nOutput_File_Name="{png.name}"\n'
                   f'Width={width}\nHeight={height}\nAntialias=On\nOutput_Alpha=On\nDisplay=Off\n')
    subprocess.run([shutil.which("povray") or "povray", ini.name], cwd=stem.parent,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    image = Image.open(png).convert("RGBA")
    alpha = image.getchannel("A"); baseline = alpha.getextrema()[0]
    if 0 < baseline < 255:
        image.putalpha(alpha.point(lambda x: max(0, round((x-baseline)*255/(255-baseline)))))
    image.save(png)
    return png


def _overlay(ax, png: Path, xy, *, zoom: float, xybox=None):
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    box = AnnotationBbox(OffsetImage(np.asarray(Image.open(png)), zoom=zoom), xy,
                         xybox=xybox, xycoords="data",
                         boxcoords="offset points" if xybox else "data",
                         frameon=False, pad=0,
                         arrowprops=(dict(arrowstyle="-", color="0.45", lw=.8)
                                     if xybox else None), zorder=6)
    ax.add_artist(box)


def neb_profile() -> Path:
    import matplotlib.pyplot as plt
    apply_plot_style("icml")
    base = ROOT / "artifacts/nh3_ch3cl/neb"
    data = np.loadtxt(base / "neb_profile.dat")
    frames = read(base / "neb.xyz", index=":")
    picks = np.linspace(0, len(frames)-1, 5, dtype=int)
    thumbs = {i: _render(frames[i], OUT / "neb_frames" / f"frame_{i:02d}",
                         forces=True, geometry=True) for i in picks}
    x, energy = data[:, 0], data[:, 1]
    fig, ax = plt.subplots(figsize=(12, 6.8), constrained_layout=True)
    color = apply_plot_style("icml").colors["train"]
    ax.plot(x, energy, color=color, marker="o", label="NEB energy")
    ax.fill_between(x, energy, energy.min()-12, color=color, alpha=.10)
    ax.set(xlabel="cumulative path length (Å)", ylabel="relative energy (kcal mol$^{-1}$)",
           title="Reaction geometry laid out along the NEB time/path axis")
    ax.set_ylim(energy.min()-12, energy.max()+20)
    ythumb = energy.max()+10
    for i in picks:
        _overlay(ax, thumbs[i], (x[i], ythumb), zoom=.105)
        ax.plot([x[i], x[i]], [energy[i], ythumb-4], color="0.65", lw=.7, zorder=1)
    ax.annotate("reaction progress", xy=(x[-1], energy.min()-9), xytext=(x[0], energy.min()-9),
                arrowprops=dict(arrowstyle="->", color="0.25"), va="center")
    out = OUT / "neb_profile_with_povray.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True); plt.close(fig)
    return out


def trialanine_pes() -> Path:
    import matplotlib.pyplot as plt
    apply_plot_style("icml")
    base = ROOT / "artifacts/trialanine_phi_psi_mm_then_ml_fixed_3x3"
    d = np.load(base / "phi_psi_pes.npz", allow_pickle=True)
    frames = read(base / "phi_psi_pes.traj", index=":")
    phi, psi = d["phi_grid_deg"], d["psi_grid_deg"]
    E = np.asarray(d["ml_energy_eV"]); E = E - np.nanmin(E)
    # Offset toward the plot interior so transparent conformers annotate the
    # surface without colliding with titles, labels, or the colorbar.
    chosen = [(0, 0, (55, 55)), (0, 2, (55, -55)),
              (2, 0, (-55, 55)), (2, 2, (-55, -55))]
    thumbs = {}
    for i, j, _ in chosen:
        idx = i * len(psi) + j
        thumbs[i, j] = _render(frames[idx], OUT / "trialanine_frames" / f"phi{i}_psi{j}", forces=True)
    fig, ax = plt.subplots(figsize=(9, 7.2), constrained_layout=True)
    levels = np.linspace(np.nanmin(E), np.nanmax(E), 12)
    cf = ax.contourf(phi, psi, E.T, levels=levels, cmap="viridis")
    ax.contour(phi, psi, E.T, levels=levels, colors="white", linewidths=.45, alpha=.55)
    fig.colorbar(cf, ax=ax, label="relative ML energy (eV)")
    ax.set(xlabel="$\\phi$ (degrees)", ylabel="$\\psi$ (degrees)",
           title="Trialanine PES with force-annotated conformers")
    for i, j, offset in chosen:
        _overlay(ax, thumbs[i, j], (phi[i], psi[j]), zoom=.11, xybox=offset)
        ax.scatter(phi[i], psi[j], s=25, color="white", edgecolor="0.2", zorder=7)
    out = OUT / "trialanine_pes_with_povray.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True); plt.close(fig)
    return out


def water_dynamics() -> Path:
    import matplotlib.pyplot as plt
    style = apply_plot_style("icml")
    d = np.load(ROOT / "artifacts/robustness_report/water_cluster_nve/trajectory.npz")
    t, energy = d["time_fs"], d["energy_eV"]
    picks = [0, len(t)//2, len(t)-1]
    thumbs = {}
    for i in picks:
        atoms = Atoms(numbers=d["Z"], positions=d["positions"][i])
        atoms.calc = SinglePointCalculator(atoms, forces=d["forces_eV_A"][i])
        thumbs[i] = _render(atoms, OUT / "water_frames" / f"frame_{i:04d}", forces=True)
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    rel = (energy-energy[0])*1000
    ax.plot(t, rel, color=style.colors["train"])
    ax.axhline(0, color=style.colors["muted"], lw=1, linestyle="--")
    ax.set(xlabel="time (fs)", ylabel="energy drift (meV)",
           title="Water-cluster dynamics with force snapshots")
    y = np.nanmax(rel) + .20 * max(np.ptp(rel), 1e-6)
    ax.set_ylim(np.nanmin(rel)-.15*np.ptp(rel), y+.25*np.ptp(rel))
    for i in picks:
        _overlay(ax, thumbs[i], (t[i], y), zoom=.09)
        ax.plot([t[i], t[i]], [rel[i], y], color="0.65", lw=.7)
    out = OUT / "water_nve_with_povray.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True); plt.close(fig)
    return out


def main() -> int:
    for path in (neb_profile(), trialanine_pes(), water_dynamics()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
