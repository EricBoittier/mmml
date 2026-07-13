#!/usr/bin/env python3
"""Render every figure for the simulation-robustness report
(docs/simulation-robustness-report.md) under the house style, from REAL
data only:

- fluctuating charges/multipoles + energy conservation: a fresh real NVE
  trajectory (scripts/run_robustness_report_md.py), real charge-predicting
  PhysNet checkpoint, real Velocity Verlet integration.
- bond/angle scans: scripts/run_robustness_report_scans.py (same real
  checkpoint).
- dihedral scan: the existing real trialanine phi/psi PES scan
  (artifacts/trialanine_phi_psi_mm_then_ml_64x64/phi_psi_pes.csv).
- dimer-separation scan: the existing real xTB/CHARMM/ML dimer campaign
  (results/dimer_scan_campaign/scan_results.csv).
- structural analysis: internal_coordinate_distributions
  (mmml.utils.plotting.trajectory_structure) run on the new NVE trajectory,
  plus the existing real element-pair RDFs from the large-scale sweep's
  periodic bulk trajectories (see the note above `structural_analysis`).
- large-scale energy conservation: the existing real 12-setting NVE sweep
  (workflows/mixed_calculator_sweep/results/summary.csv).
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors

from mmml.utils.plotting.styles import (
    STATUS_HATCHES,
    apply_plot_style,
    comparison_colors,
    latex_available,
    latex_table_image,
    legend_outside,
    status_color,
    timeseries_with_distribution,
)
from mmml.utils.plotting.trajectory_structure import internal_coordinate_distributions

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "robustness-report-assets"
STYLE_NAME = "icml"

EV_TO_KCAL_MOL = 23.060548867

NVE_DIR = REPO_ROOT / "artifacts" / "robustness_report" / "ethanol_nve"
SCAN_DIR = REPO_ROOT / "artifacts" / "robustness_report" / "scans"
DIHEDRAL_CSV = REPO_ROOT / "artifacts" / "trialanine_phi_psi_mm_then_ml_64x64" / "phi_psi_pes.csv"
DIMER_CSV = REPO_ROOT / "results" / "dimer_scan_campaign" / "scan_results.csv"
SWEEP_SUMMARY_CSV = REPO_ROOT / "workflows" / "mixed_calculator_sweep" / "results" / "summary.csv"
SWEEP_RESULTS_DIR = REPO_ROOT / "workflows" / "mixed_calculator_sweep" / "results"


def _load_plot_utils():
    """Import scripts/plot_utils.py::render_dimer_atoms by path (not a package) --
    same convention as scripts/render_chart_type_gallery.py."""
    spec = importlib.util.spec_from_file_location("plot_utils", REPO_ROOT / "scripts" / "plot_utils.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _atoms_from_traj(traj: dict, frame: int = 0) -> Atoms:
    return Atoms(numbers=traj["Z"], positions=traj["positions"][frame])


# --- 1. Fluctuating charges and multipoles ----------------------------------


def charge_and_dipole_fluctuation(traj: dict, out: Path) -> None:
    Z = traj["Z"]
    charges = traj["charges_e"]           # (n_frames, n_atoms)
    dipole = traj["dipole_eA"]             # (n_frames, 3)
    t = traj["time_fs"]
    element_symbol = {6: "C", 8: "O", 1: "H"}
    symbols = [element_symbol[z] for z in Z]
    present_elements = sorted(set(symbols), key=lambda s: -Z[symbols.index(s)])

    plot_utils = _load_plot_utils()

    fig = plt.figure(figsize=(16, 6.3))
    # top=0.75: leaves headroom for the figure suptitle above the panel
    # titles -- without an explicit top margin the suptitle and the panel
    # titles (icml's bold 17pt) collide.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.0, 2.0], wspace=0.32, top=0.75)

    ax_mol = fig.add_subplot(gs[0, 0])
    plot_utils.render_dimer_atoms(ax_mol, _atoms_from_traj(traj), title="Simulated molecule\n(ethanol, relaxed geometry)")

    ax_q = fig.add_subplot(gs[0, 1])
    # jmol's H color is literal white (1,1,1) -- invisible on a white figure
    # background, so substitute a visible dark gray for H while keeping
    # jmol's colors for the other (visible) elements.
    element_colors = {s: ("#444444" if s == "H" else jmol_colors[atomic_numbers[s]]) for s in present_elements}
    for atom_idx in range(len(Z)):
        sym = symbols[atom_idx]
        ax_q.plot(t, charges[:, atom_idx], color=element_colors[sym], alpha=0.7, linewidth=1.1)
    handles = [plt.Line2D([0], [0], color=element_colors[s], linewidth=2.5, label=f"{s} atoms")
               for s in present_elements]
    ax_q.legend(handles=handles, loc="center right")
    ax_q.set_xlabel("time (fs)")
    ax_q.set_ylabel("predicted partial charge (e)")
    counts = ", ".join(f"{symbols.count(s)} {s}" for s in present_elements)
    ax_q.set_title(f"Per-atom charge fluctuation\n({counts}, real NVE trajectory)")

    dipole_mag = np.linalg.norm(dipole, axis=1)
    colors = comparison_colors(STYLE_NAME, n=3)
    ax_d = fig.add_subplot(gs[0, 2])
    ax_d.plot(t, dipole[:, 0], color=colors[0], linewidth=1.2, alpha=0.85, label="$\\mu_x$")
    ax_d.plot(t, dipole[:, 1], color=colors[1], linewidth=1.2, alpha=0.85, label="$\\mu_y$")
    ax_d.plot(t, dipole[:, 2], color=colors[2], linewidth=1.2, alpha=0.85, label="$\\mu_z$")
    ax_d.plot(t, dipole_mag, color="#222222", linewidth=2.2, label="$|\\mu|$")
    ax_d.set_xlabel("time (fs)")
    ax_d.set_ylabel("molecular dipole (e·Å)")
    ax_d.set_title("Total dipole moment fluctuation")
    legend_outside(ax_d, side="right")

    fig.suptitle("Fluctuating charges and multipoles -- real ethanol molecule, "
                 "real charge-predicting checkpoint")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- 2. Bond / angle / dihedral / dimer scans --------------------------------


def bond_angle_scans(out: Path) -> None:
    bond = np.load(SCAN_DIR / "bond_scan.npz")
    angle = np.load(SCAN_DIR / "angle_scan.npz")
    colors = comparison_colors(STYLE_NAME, n=2)

    fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(12, 5))
    ax_b.plot(bond["bond_A"], bond["energy_eV"], color=colors[0], linewidth=2.2, marker="o", markersize=3)
    ax_b.axvline(float(bond["eq_bond_A"]), color="#222222", linestyle="--", linewidth=1.0,
                 label=f"expt. eq. O-H ({float(bond['eq_bond_A']):.3f} Å)")
    ax_b.set_xlabel("O-H bond length (Å)")
    ax_b.set_ylabel("energy (eV)")
    ax_b.set_title("Real bond-stretch PES scan\n(single water molecule)")
    ax_b.legend(fontsize=9)

    ax_a.plot(angle["angle_deg"], angle["energy_eV"], color=colors[1], linewidth=2.2, marker="o", markersize=3)
    ax_a.axvline(float(angle["eq_angle_deg"]), color="#222222", linestyle="--", linewidth=1.0,
                 label=f"expt. eq. H-O-H ({float(angle['eq_angle_deg']):.1f}°)")
    ax_a.set_xlabel("H-O-H angle (degrees)")
    ax_a.set_ylabel("energy (eV)")
    ax_a.set_title("Real angle-bend PES scan\n(single water molecule)")
    ax_a.legend(fontsize=9)

    fig.suptitle("Bond and angle scans -- real checkpoint evaluations, not fitted curves")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # The wide scan range also reveals a real model limitation worth
    # documenting honestly: a spurious deeper "minimum" appears well outside
    # the chemically-relevant region (compressed/over-stretched geometries
    # the model rarely saw in training) -- separate panel, clearly labeled,
    # rather than silently cropping the scan range to hide it.
    fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, xkey, xlabel, eq_key, eq_label in [
        (ax_b, bond, "bond_A", "O-H bond length (Å)", "eq_bond_A", "Å"),
        (ax_a, angle, "angle_deg", "H-O-H angle (degrees)", "eq_angle_deg", "°"),
    ]:
        x, e = data[xkey], data["energy_eV"]
        real_min_mask = np.abs(x - float(data[eq_key])) < (0.15 if "bond" in xkey else 15)
        ax.plot(x, e, color="#999999", linewidth=1.6, zorder=1)
        ax.plot(x[real_min_mask], e[real_min_mask], color=status_color("good"), linewidth=2.4,
                label="near-equilibrium region\n(physically reliable)", zorder=2)
        global_min_idx = np.argmin(e)
        if not real_min_mask[global_min_idx]:
            ax.scatter([x[global_min_idx]], [e[global_min_idx]], color=status_color("warning"),
                       s=70, zorder=3, edgecolor="#222222",
                       label="spurious extrapolation\nartifact (out-of-distribution)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("energy (eV)")
        ax.legend(fontsize=8, loc="upper center")
    fig.suptitle("Known limitation: this compact demo checkpoint extrapolates poorly\n"
                 "far outside chemically-relevant geometries -- flagged, not hidden", fontsize=12)
    fig.tight_layout()
    fig.savefig(out.with_name(out.stem + "_caveat.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def dihedral_and_dimer_scans(out: Path) -> None:
    dihedral_df = pd.read_csv(DIHEDRAL_CSV)
    dimer_df = pd.read_csv(DIMER_CSV)
    colors = comparison_colors(STYLE_NAME, n=2)

    fig, (ax_dih, ax_dimer) = plt.subplots(1, 2, figsize=(13, 5))

    # 1D slice through the real 2D phi/psi PES at psi closest to its median,
    # so the dihedral (phi) scan reads as a standard 1D torsion profile.
    psi_target = dihedral_df["psi_deg"].median()
    slice_df = dihedral_df.loc[(dihedral_df["psi_deg"] - psi_target).abs().idxmin() == dihedral_df.index]
    nearest_psi = dihedral_df.iloc[(dihedral_df["psi_deg"] - psi_target).abs().argsort()[:1]]["psi_deg"].iloc[0]
    slice_df = dihedral_df[np.isclose(dihedral_df["psi_deg"], nearest_psi)].sort_values("phi_deg")
    e_mm = slice_df["charmm_mm_min_energy_kcal_mol"] - slice_df["charmm_mm_min_energy_kcal_mol"].min()
    ax_dih.plot(slice_df["phi_deg"], e_mm, color=colors[0], linewidth=2.2, marker="o", markersize=3)
    ax_dih.set_xlabel(r"backbone dihedral $\phi$ (degrees)")
    ax_dih.set_ylabel("MM energy above minimum (kcal/mol)")
    ax_dih.set_title(f"Real dihedral (torsion) PES scan\n(trialanine backbone, $\\psi$={nearest_psi:.0f}° slice)")

    # The campaign samples MANY relative orientations per separation (15 per
    # distance here), not a single rigid-scan geometry -- take the
    # lowest-energy (most favorable) orientation at each distance for a
    # clean, physically-meaningful 1D binding curve, same convention as a
    # relaxed/adiabatic PES scan.
    xtb = dimer_df[dimer_df["backend"] == "xtb_gfn2"]
    xtb_min = xtb.groupby("distance_angstrom")["energy_kcal_mol"].min().reset_index().sort_values("distance_angstrom")
    ax_dimer.plot(xtb_min["distance_angstrom"], xtb_min["energy_kcal_mol"], color=colors[1],
                  linewidth=2.2, marker="o", markersize=4)
    ax_dimer.set_xlabel("intermolecular separation (Å)")
    ax_dimer.set_ylabel("min. energy over sampled\norientations (kcal/mol)")
    ax_dimer.set_title(f"Real non-bonded (dimer-separation) PES scan\n"
                        f"(xTB-GFN2, best of {xtb['distance_angstrom'].value_counts().iloc[0]} "
                        f"orientations per distance)")

    fig.suptitle("Dihedral and non-bonded-distance scans -- existing real campaign data")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- 2b. Full 2D potential energy surfaces (trialanine backbone) ------------
#
# Adapted from /Users/ericboittier/Untitled.ipynb's MM/ML landscape figure
# (scatter + interpolated 3D surface + torus inner/outer views), on the same
# real phi/psi PES scan used above -- ported to the house style (default
# sequential colormap instead of an ad hoc pick, apply_plot_style fonts)
# rather than copied verbatim.


def _pes_panel_label(ax, label: str, x: float = 0.03, y: float = 0.94, fontsize: float = 13) -> None:
    target = ax.text2D if hasattr(ax, "text2D") else ax.text
    target(x, y, label, transform=ax.transAxes, fontsize=fontsize, fontweight="bold", ha="left", va="top")


def _pes_interpolate_grid(phi_deg, psi_deg, e, n=200, sigma=2):
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    xi = np.linspace(-180, 180, n)
    yi = np.linspace(-180, 180, n)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    zi = griddata((phi_deg, psi_deg), e, (xi_grid, yi_grid), method="cubic")
    return xi_grid, yi_grid, gaussian_filter(zi, sigma=sigma)


def _pes_interpolate_periodic(phi_rad, psi_rad, e, n=200, smoothing=0.1, sigma=1.5):
    from scipy.interpolate import RBFInterpolator
    from scipy.ndimage import gaussian_filter

    features = np.column_stack([np.cos(phi_rad), np.sin(phi_rad), np.cos(psi_rad), np.sin(psi_rad)])
    rbf = RBFInterpolator(features, e, kernel="thin_plate_spline", smoothing=smoothing)
    phi_g = np.linspace(-np.pi, np.pi, n)
    psi_g = np.linspace(-np.pi, np.pi, n)
    phi_grid, psi_grid = np.meshgrid(phi_g, psi_g)
    grid_features = np.column_stack([np.cos(phi_grid).ravel(), np.sin(phi_grid).ravel(),
                                      np.cos(psi_grid).ravel(), np.sin(psi_grid).ravel()])
    z = rbf(grid_features).reshape(phi_grid.shape)
    return phi_grid, psi_grid, gaussian_filter(z, sigma=sigma)


def _pes_torus_coordinates(phi_grid, psi_grid, major_r=1.15, minor_r=0.55):
    x = (major_r + minor_r * np.cos(psi_grid)) * np.cos(phi_grid)
    y = (major_r + minor_r * np.cos(psi_grid)) * np.sin(phi_grid)
    z = minor_r * np.sin(psi_grid)
    return x, y, z


def _pes_plot_energy_row(axes, phi_deg, psi_deg, e, cmap, norm, row_name, labels) -> None:
    axes[0].scatter(phi_deg, psi_deg, s=14, c=e, cmap=cmap, norm=norm, linewidths=0)
    axes[0].set_xlim(-180, 180)
    axes[0].set_ylim(-180, 180)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel(r"$\phi$ (deg)")
    axes[0].set_ylabel(row_name + "\n" + r"$\psi$ (deg)")
    _pes_panel_label(axes[0], labels[0])

    xi, yi, zi = _pes_interpolate_grid(phi_deg, psi_deg, e)
    axes[1].plot_surface(xi, yi, zi, cmap=cmap, norm=norm, edgecolor="none", antialiased=True)
    axes[1].view_init(elev=33, azim=45, roll=1)
    axes[1].set_proj_type("ortho")
    axes[1].set_xlabel(r"$\phi$", labelpad=-6)
    axes[1].set_ylabel(r"$\psi$", labelpad=-6)
    axes[1].set_zlim(norm.vmin, norm.vmax)
    axes[1].tick_params(labelsize=7, pad=0)
    _pes_panel_label(axes[1], labels[1])

    phi_grid, psi_grid, z_tor = _pes_interpolate_periodic(np.deg2rad(phi_deg), np.deg2rad(psi_deg), e)
    for ax, view, label in ((axes[2], "outer", labels[2]), (axes[3], "inner", labels[3])):
        xt, yt, zt = _pes_torus_coordinates(phi_grid, psi_grid)
        ax.plot_surface(xt, yt, zt, facecolors=cmap(norm(z_tor)), rstride=1, cstride=1,
                         linewidth=0, antialiased=False, shade=False)
        ax.view_init(elev=25 if view == "outer" else 20, azim=45 if view == "outer" else 225, roll=1)
        ax.set_proj_type("ortho")
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 0.55])
        _pes_panel_label(ax, label)


def trialanine_pes_landscape(out: Path) -> None:
    """MM and ML trialanine backbone energy landscapes, four representations
    each (scatter / interpolated 3D surface / torus outer / torus inner) --
    the torus views make phi=-180/+180 and psi=-180/+180 correctly adjacent
    (real periodic topology) rather than an artificial seam at a flat plot's
    edge. Real phi/psi PES scan data, real MM (CHARMM) and ML energies."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mmml.utils.plotting.styles import default_cmap

    df = pd.read_csv(DIHEDRAL_CSV)
    phi_deg = df["phi_deg"].to_numpy()
    psi_deg = df["psi_deg"].to_numpy()
    e_mm = (df["charmm_mm_min_energy_kcal_mol"] - df["charmm_mm_min_energy_kcal_mol"].min()).to_numpy()
    e_ml = (EV_TO_KCAL_MOL * (df["ml_energy_eV"] - df["ml_energy_eV"].min())).to_numpy()

    vmax = 100.0
    cmap = default_cmap("sequential")
    norm = Normalize(vmin=0, vmax=vmax, clip=True)

    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], height_ratios=[1, 1])
    axes = np.empty((2, 4), dtype=object)
    for i in range(2):
        axes[i, 0] = fig.add_subplot(gs[i, 0])
        for j in (1, 2, 3):
            axes[i, j] = fig.add_subplot(gs[i, j], projection="3d")
    cax = fig.add_subplot(gs[:, 4])

    _pes_plot_energy_row(axes[0], phi_deg, psi_deg, e_mm, cmap, norm, "MM", ("(a)", "(b)", "(c)", "(d)"))
    _pes_plot_energy_row(axes[1], phi_deg, psi_deg, e_ml, cmap, norm, "ML", ("(e)", "(f)", "(g)", "(h)"))

    for j, title in enumerate(("Scatter", "Interpolated surface", "Torus outer face", "Torus inner face")):
        axes[0, j].set_title(title, fontsize=13)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="energy above minimum (kcal/mol)")
    fig.suptitle("Trialanine backbone PES: MM (CHARMM) vs. ML, real 64x64 scan", fontsize=16)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


# --- 3. Energy conservation ---------------------------------------------------


def energy_conservation_small_system(stable: dict, unstable: dict, out: Path) -> None:
    """PE and total energy each get their OWN panel (never overlaid --
    they live on different absolute scales and overlaying them either
    crushes one or forces a distracting second y-axis), each as a real
    time-series + marginal-distribution pair (`timeseries_with_distribution`)
    so "does it drift" and "how wide is the fluctuation" both read at a
    glance. Both series are mean-subtracted and reported in kcal/mol (the
    house energy unit), with a molecular render of the actual simulated
    system alongside so the numbers aren't disembodied from what was run.
    """
    plot_utils = _load_plot_utils()
    colors = comparison_colors(STYLE_NAME, n=2)  # [PE color, total-energy color]

    fig = plt.figure(figsize=(15, 10.5))
    # top=0.80: same headroom fix as charge_and_dipole_fluctuation -- the
    # figure suptitle needs real room above the top row's panel titles.
    outer = fig.add_gridspec(2, 3, width_ratios=[0.85, 2.2, 2.2], hspace=0.55, wspace=0.4, top=0.80)

    for row, (traj, label, dt) in enumerate([(stable, "stable", 0.1), (unstable, "unstable", 0.5)]):
        status = "good" if row == 0 else "critical"
        t = traj["time_fs"]
        pe_kcal = traj["energy_eV"] * EV_TO_KCAL_MOL
        etot_kcal = (traj["energy_eV"] + traj["kinetic_eV"]) * EV_TO_KCAL_MOL

        ax_mol = fig.add_subplot(outer[row, 0])
        plot_utils.render_dimer_atoms(
            ax_mol, _atoms_from_traj(traj),
            title=f"dt={dt} fs ({label.upper()})",
        )
        ax_mol.title.set_color(status_color(status))
        ax_mol.title.set_fontweight("bold")

        ax_pe, _ = timeseries_with_distribution(
            fig, outer[row, 1], t, pe_kcal, color=colors[0],
            ylabel="PE $-\\ \\overline{PE}$ (kcal/mol)", xlabel="time (fs)",
        )
        ax_pe.set_title(f"Potential energy (mean-subtracted)\nstd = {pe_kcal.std():.4f} kcal/mol", fontsize=11)

        ax_etot, _ = timeseries_with_distribution(
            fig, outer[row, 2], t, etot_kcal, color=colors[1],
            ylabel="$E_{tot} - \\overline{E_{tot}}$ (kcal/mol)", xlabel="time (fs)",
        )
        ax_etot.set_title(f"Total energy (mean-subtracted)\nstd = {etot_kcal.std():.4f} kcal/mol", fontsize=11)

    fig.suptitle("Energy conservation: correct integration timestep vs. an under-resolved one\n"
                 "(real NVE trajectories, same system/model/initial condition, only dt differs)")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def energy_fluctuation_comparison(stable: dict, unstable: dict, out: Path) -> None:
    """Direct comparison of fluctuation magnitude: both runs' mean-subtracted
    total-energy distributions on the SAME axis (kcal/mol) -- the width
    difference (not just the drift-over-time plot) is the actual claim
    "an under-resolved timestep fluctuates more, not just drifts more."
    """
    colors = [status_color("good"), status_color("critical")]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    stds = []
    for (traj, label, dt), color in zip(
        [(stable, "stable", 0.1), (unstable, "unstable", 0.5)], colors,
    ):
        etot_kcal = (traj["energy_eV"] + traj["kinetic_eV"]) * EV_TO_KCAL_MOL
        centered = etot_kcal - etot_kcal.mean()
        stds.append(float(centered.std()))
        ax.hist(centered, bins=40, density=True, alpha=0.55, color=color,
                edgecolor="#222222", linewidth=0.4,
                label=f"dt={dt} fs ({label}), std={centered.std():.4f} kcal/mol")
    ax.axvline(0, color="#999999", linewidth=1.0, linestyle="--")
    ax.set_xlabel(r"$E_{tot} - \overline{E_{tot}}$ (kcal/mol)")
    ax.set_ylabel("probability density")
    ratio = stds[1] / stds[0] if stds[0] else float("nan")
    ax.set_title(f"Fluctuation comparison: {ratio:.1f}x wider at 5x the timestep\n"
                 f"(Verlet integration error scales as O(dt²): expected ~25x)")
    legend_outside(ax)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def energy_conservation_sweep_summary(out: Path) -> None:
    df = pd.read_csv(SWEEP_SUMMARY_CSV)
    df = df[df["completed"]].copy()
    df["status"] = np.where(
        df["energy_max_abs_deviation_ev"].abs() < 1.0, "good",
        np.where(df["energy_max_abs_deviation_ev"].abs() < 100.0, "warning", "critical"),
    )
    df["label"] = df["setting"] + " (seed " + df["seed"].astype(str) + ")"
    df = df.sort_values("energy_max_abs_deviation_ev", key=np.abs)

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(df) + 1.2))
    y = np.arange(len(df))
    ax.barh(y, np.abs(df["energy_max_abs_deviation_ev"]).clip(lower=1e-6),
            color=[status_color(s) for s in df["status"]],
            hatch=[STATUS_HATCHES[s] for s in df["status"]],
            edgecolor="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("max |energy deviation| over the run (eV, log scale)")
    ax.set_title("Large-scale NVE sweep: 12/12 completed settings\n"
                 "(workflows/mixed_calculator_sweep, real 10000-step CHARMM/ML trajectories)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- 4. Structural analysis ---------------------------------------------------
#
# element_pair_rdfs() normalizes against a uniform bulk (ideal-gas) density
# over the periodic cell -- correct for a real periodic bulk trajectory, but
# meaningless for a small non-periodic cluster stuffed into an artificially
# large box (the local density near the actual cluster is enormously higher
# than the box average, so g(r) blows up at short range regardless of the
# real structure). Rather than force it onto a system it isn't built for,
# reuse the REAL RDFs already computed on real periodic bulk NVE
# trajectories from the large-scale sweep -- see
# workflows/mixed_calculator_sweep/results/*/seed_1/figures/element_pair_rdfs.png.


def structural_analysis(traj: dict, out_internal: Path) -> None:
    positions_all, Z = traj["positions"], traj["Z"]
    frames_bare = [Atoms(numbers=Z, positions=p) for p in positions_all[::4]]
    internal = internal_coordinate_distributions(frames_bare, range(len(Z)))
    _COORDINATE_TYPE_COLORS = {"Bond lengths": "#1A5276", "Angles": "#B9770E", "Dihedrals": "#1E8449"}
    all_groups = ((internal.bonds, "Bond lengths", "Å", 30), (internal.angles, "Angles", "degrees", 30),
                  (internal.dihedrals, "Dihedrals", "degrees", 30))
    # Only build a panel for coordinate types that actually exist for this
    # system (e.g. a system with no 4-atom chain has zero dihedrals) --
    # a "none found" placeholder panel is dead space, not information.
    groups = [g for g in all_groups if g[0]]

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.5))
    if len(groups) == 1:
        axes = [axes]
    for axis, (coordinates, title, unit, bins) in zip(axes, groups):
        color = _COORDINATE_TYPE_COLORS[title]
        all_values = np.concatenate(list(coordinates.values()))
        axis.hist(all_values, bins=bins, density=True, alpha=0.85, color=color)
        axis.set_title(f"{title} ({len(coordinates)} coordinates)")
        axis.set_xlabel(unit)
        axis.set_ylabel("probability density")
        axis.grid(alpha=0.2)
    fig.suptitle("Internal-coordinate distributions -- real NVE trajectory (ethanol)")
    fig.tight_layout()
    fig.savefig(out_internal, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- Summary table -------------------------------------------------------------


def summary_table(stable: dict, unstable: dict, out: Path) -> None:
    stable_etot = (stable["energy_eV"] + stable["kinetic_eV"]) * EV_TO_KCAL_MOL
    unstable_etot = (unstable["energy_eV"] + unstable["kinetic_eV"]) * EV_TO_KCAL_MOL
    cell_text = [
        ["Ethanol NVE, max drift (dt=0.1 fs)", f"{np.abs(stable_etot - stable_etot[0]).max():.4f}", "kcal/mol"],
        ["Ethanol NVE, max drift (dt=0.5 fs)", f"{np.abs(unstable_etot - unstable_etot[0]).max():.4f}", "kcal/mol"],
        ["Ethanol NVE, fluctuation std (dt=0.1 fs)", f"{stable_etot.std():.4f}", "kcal/mol"],
        ["Ethanol NVE, fluctuation std (dt=0.5 fs)", f"{unstable_etot.std():.4f}", "kcal/mol"],
        ["Charge range (all frames)", f"{stable['charges_e'].min():.3f} to {stable['charges_e'].max():.3f}", "e"],
        ["Dipole magnitude range", f"{np.linalg.norm(stable['dipole_eA'], axis=1).min():.3f} to "
                                    f"{np.linalg.norm(stable['dipole_eA'], axis=1).max():.3f}", "e\\textperiodcentered\\r{A}"],
        ["Sweep settings completed", "12 / 12", "--"],
    ]
    if latex_available():
        fig, ax = plt.subplots(figsize=(7, 2.2))
        latex_table_image(ax, cell_text, col_labels=["quantity", "value", "units"], fontsize_pt=12)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style(STYLE_NAME)

    stable_dict = dict(np.load(NVE_DIR / "trajectory.npz"))
    unstable_dict = dict(np.load(NVE_DIR / "trajectory_unstable_dt.npz"))

    charge_and_dipole_fluctuation(stable_dict, OUT_DIR / "chart_charge_dipole_fluctuation.png")
    print("wrote chart_charge_dipole_fluctuation.png")

    bond_angle_scans(OUT_DIR / "chart_bond_angle_scans.png")
    print("wrote chart_bond_angle_scans.png (+ _caveat)")

    dihedral_and_dimer_scans(OUT_DIR / "chart_dihedral_dimer_scans.png")
    print("wrote chart_dihedral_dimer_scans.png")

    trialanine_pes_landscape(OUT_DIR / "chart_pes_landscape.png")
    print("wrote chart_pes_landscape.png")

    energy_conservation_small_system(stable_dict, unstable_dict, OUT_DIR / "chart_energy_conservation_small.png")
    print("wrote chart_energy_conservation_small.png")

    energy_fluctuation_comparison(stable_dict, unstable_dict, OUT_DIR / "chart_energy_fluctuation_comparison.png")
    print("wrote chart_energy_fluctuation_comparison.png")

    energy_conservation_sweep_summary(OUT_DIR / "chart_energy_conservation_sweep.png")
    print("wrote chart_energy_conservation_sweep.png")

    structural_analysis(stable_dict, OUT_DIR / "chart_structural_internal.png")
    print("wrote chart_structural_internal.png")

    for setting, dest_name in [
        ("water_baseline", "chart_structural_rdfs_water_baseline.png"),
        ("mixed_baseline", "chart_structural_rdfs_mixed_baseline.png"),
    ]:
        src = SWEEP_RESULTS_DIR / setting / "seed_1" / "figures" / "element_pair_rdfs.png"
        if src.is_file():
            shutil.copy(src, OUT_DIR / dest_name)
            print(f"copied real {setting} RDF -> {dest_name}")
        else:
            print(f"WARNING: {src} not found, skipping")

    summary_table(stable_dict, unstable_dict, OUT_DIR / "chart_summary_table.png")
    print("wrote chart_summary_table.png")


if __name__ == "__main__":
    main()
