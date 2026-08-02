#!/usr/bin/env python3
"""Render POV-Ray and quantitative proof for a full-box Q0 charge evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from PIL import Image

from mmml.utils.plotting.styles import apply_plot_style, default_cmap
from render_povray_style_catalog import scene, vec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q0", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    q0 = np.load(args.q0)
    fixed = np.load(args.fixed)
    z = np.asarray(q0["Z"])
    r = np.asarray(q0["R_base"])
    force = np.asarray(q0["F_base"])
    charge = np.asarray(q0["Qmm_base"])
    fixed_force = np.asarray(fixed["F_base"])

    molecule_r = r.reshape(-1, 3, 3)
    centers = molecule_r.mean(axis=1)
    center = np.median(centers, axis=0)
    selected_molecules = np.argsort(np.linalg.norm(centers - center, axis=1))[:14]
    selected = np.concatenate(
        [np.arange(3 * index, 3 * index + 3) for index in selected_molecules]
    )
    atoms = Atoms(numbers=z[selected], positions=r[selected])
    atoms.arrays["forces"] = force[selected]

    width, height = 1120, 820
    pov = scene(
        atoms,
        "glossy",
        "",
        vectors=True,
        geometry=False,
        font="",
        width=width,
        height=height,
        frame_scale=1.92,
    )
    xyz = atoms.positions - atoms.positions.mean(axis=0)
    overlays: list[str] = []
    for position, value in zip(xyz, charge[selected]):
        color = (0.16, 0.38, 0.92) if value > 0 else (0.88, 0.12, 0.20)
        radius = 0.38 + 2.2 * abs(float(value))
        overlays.append(
            f"sphere {{ {vec(position)}, {radius:.4f} "
            f"pigment {{ color rgbt <{color[0]},{color[1]},{color[2]},0.76> }} "
            "finish { emission 0.04 phong 0.55 } no_shadow }"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    render = args.output.with_name(args.output.stem + "_povray")
    render.with_suffix(".pov").write_text(pov + "\n".join(overlays) + "\n")
    render.with_suffix(".ini").write_text(
        f'Input_File_Name="{render.name}.pov"\n'
        f'Output_File_Name="{render.name}.png"\n'
        f"Width={width}\nHeight={height}\nAntialias=On\nOutput_Alpha=On\nDisplay=Off\n"
    )
    subprocess.run(
        [shutil.which("povray") or "povray", render.with_suffix(".ini").name],
        cwd=render.parent,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    molecular = Image.open(render.with_suffix(".png")).convert("RGBA")

    apply_plot_style("icml")
    colors = apply_plot_style("icml").colors
    fig = plt.figure(figsize=(12.2, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=(1.45, 1.0, 1.0))
    ax_image = fig.add_subplot(grid[:, 0])
    ax_q = fig.add_subplot(grid[0, 1])
    ax_f = fig.add_subplot(grid[0, 2])
    ax_delta = fig.add_subplot(grid[1, 1:])

    ax_image.imshow(molecular)
    ax_image.axis("off")
    ax_image.set_title("Q⁰ MM charges and exact forces")
    ax_image.text(
        0.02,
        0.02,
        "blue +q   red −q   arrows: force\n14 central waters; arrows share one scale",
        transform=ax_image.transAxes,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none"},
    )

    oxygen = z == 8
    hydrogen = z == 1
    bins = np.linspace(charge.min() - 0.005, charge.max() + 0.005, 32)
    ax_q.hist(charge[oxygen], bins=bins, label="O", color=colors["valid"], alpha=0.82)
    ax_q.hist(charge[hydrogen], bins=bins, label="H", color=colors["train"], alpha=0.82)
    ax_q.axvline(0, color="0.25", linewidth=0.8)
    ax_q.set(xlabel="Q⁰ charge (e)", ylabel="atom count", title="Learned MM charges")
    ax_q.legend()

    force_norm = np.linalg.norm(force, axis=1)
    fixed_norm = np.linalg.norm(fixed_force, axis=1)
    force_bins = np.linspace(0, max(force_norm.max(), fixed_norm.max()), 45)
    ax_f.hist(fixed_norm, bins=force_bins, histtype="step", linewidth=1.4, label="fixed TIP3")
    ax_f.hist(force_norm, bins=force_bins, histtype="step", linewidth=1.4, label="Q⁰")
    ax_f.set(xlabel="|F| (eV/Å)", ylabel="atom count", title="Force distribution")
    ax_f.legend()

    delta_force = np.linalg.norm(force - fixed_force, axis=1)
    scatter = ax_delta.scatter(
        charge,
        delta_force,
        c=force_norm,
        s=8,
        alpha=0.55,
        cmap=default_cmap("sequential"),
        rasterized=True,
    )
    ax_delta.set(
        xlabel="Q⁰ charge used by MM Coulomb (e)",
        ylabel="|F(Q⁰) − F(fixed)| (eV/Å)",
        title="Hamiltonian change relative to fixed TIP3 electrostatics",
    )
    fig.colorbar(scatter, ax=ax_delta, label="|F(Q⁰)| (eV/Å)")
    fig.suptitle("Full 732-water charge-aware static validation")
    fig.savefig(args.output, dpi=300, transparent=True, bbox_inches="tight")

    summary = {
        "n_waters": int(len(z) // 3),
        "q0_total_charge_e": float(charge.sum()),
        "q0_max_abs_water_charge_e": float(
            np.max(np.abs(charge.reshape(-1, 3).sum(axis=1)))
        ),
        "q0_charge_range_e": [float(charge.min()), float(charge.max())],
        "q0_force_max_eV_A": float(force_norm.max()),
        "fixed_force_max_eV_A": float(fixed_norm.max()),
        "q0_vs_fixed_force_rms_eV_A": float(
            np.sqrt(np.mean((force - fixed_force) ** 2))
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
