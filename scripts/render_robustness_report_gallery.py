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
)
from mmml.utils.plotting.trajectory_structure import internal_coordinate_distributions

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "robustness-report-assets"
STYLE_NAME = "icml"

NVE_DIR = REPO_ROOT / "artifacts" / "robustness_report" / "water_cluster_nve"
SCAN_DIR = REPO_ROOT / "artifacts" / "robustness_report" / "scans"
DIHEDRAL_CSV = REPO_ROOT / "artifacts" / "trialanine_phi_psi_mm_then_ml_64x64" / "phi_psi_pes.csv"
DIMER_CSV = REPO_ROOT / "results" / "dimer_scan_campaign" / "scan_results.csv"
SWEEP_SUMMARY_CSV = REPO_ROOT / "workflows" / "mixed_calculator_sweep" / "results" / "summary.csv"
SWEEP_RESULTS_DIR = REPO_ROOT / "workflows" / "mixed_calculator_sweep" / "results"


# --- 1. Fluctuating charges and multipoles ----------------------------------


def charge_and_dipole_fluctuation(traj: dict, out: Path) -> None:
    Z = traj["Z"]
    charges = traj["charges_e"]           # (n_frames, n_atoms)
    dipole = traj["dipole_eA"]             # (n_frames, 3)
    t = traj["time_fs"]
    symbols = [{8: "O", 1: "H"}[z] for z in Z]

    fig, (ax_q, ax_d) = plt.subplots(1, 2, figsize=(13, 5))

    # jmol's H color is literal white (1,1,1) -- invisible on a white figure
    # background, so substitute a visible dark gray for H while keeping
    # jmol's red for O (still a semantically-recognizable element color).
    element_colors = {"O": jmol_colors[atomic_numbers["O"]], "H": "#444444"}
    for atom_idx in range(len(Z)):
        sym = symbols[atom_idx]
        ax_q.plot(t, charges[:, atom_idx], color=element_colors[sym], alpha=0.7, linewidth=1.1)
    # Legend by element only (not per-atom -- 12 near-identical lines would be unreadable).
    handles = [plt.Line2D([0], [0], color=element_colors[s], linewidth=2.5, label=f"{s} atoms")
               for s in ("O", "H")]
    ax_q.legend(handles=handles, loc="center right")
    ax_q.set_xlabel("time (fs)")
    ax_q.set_ylabel("predicted partial charge (e)")
    ax_q.set_title(f"Per-atom charge fluctuation\n({(Z == 8).sum()} O + {(Z == 1).sum()} H, real NVE trajectory)")

    dipole_mag = np.linalg.norm(dipole, axis=1)
    colors = comparison_colors(STYLE_NAME, n=3)
    ax_d.plot(t, dipole[:, 0], color=colors[0], linewidth=1.2, alpha=0.85, label="$\\mu_x$")
    ax_d.plot(t, dipole[:, 1], color=colors[1], linewidth=1.2, alpha=0.85, label="$\\mu_y$")
    ax_d.plot(t, dipole[:, 2], color=colors[2], linewidth=1.2, alpha=0.85, label="$\\mu_z$")
    ax_d.plot(t, dipole_mag, color="#222222", linewidth=2.2, label="$|\\mu|$")
    ax_d.set_xlabel("time (fs)")
    ax_d.set_ylabel("molecular dipole (e·Å)")
    ax_d.set_title("Total dipole moment fluctuation")
    legend_outside(ax_d, side="right")

    fig.suptitle(f"Fluctuating charges and multipoles -- real {len(Z)}-atom water cluster, "
                 f"real charge-predicting checkpoint")
    fig.tight_layout()
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


# --- 3. Energy conservation ---------------------------------------------------


def energy_conservation_small_system(stable: dict, unstable: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = comparison_colors(STYLE_NAME, n=3)

    for row, (traj, label, dt) in enumerate([(stable, "stable", 0.1), (unstable, "unstable", 0.5)]):
        ax = axes[row, 0]
        t = traj["time_fs"]
        pe, ke = traj["energy_eV"], traj["kinetic_eV"]
        etot = pe + ke
        ax.plot(t, pe, color=colors[0], linewidth=1.4, label="PE", alpha=0.85)
        ax.plot(t, ke, color=colors[1], linewidth=1.4, label="KE", alpha=0.85)
        ax.plot(t, etot, color="#222222", linewidth=2.2, label="total")
        status = "good" if row == 0 else "critical"
        ax.set_title(f"dt={dt} fs ({label.upper()})", color=status_color(status), fontweight="bold")
        ax.set_xlabel("time (fs)")
        ax.set_ylabel("energy (eV)")
        if row == 0:
            ax.legend(fontsize=9, loc="center right")

        ax2 = axes[row, 1]
        drift = etot - etot[0]
        ax2.plot(t, drift, color=status_color(status), linewidth=2.0)
        ax2.axhline(0, color="#999999", linewidth=1.0, linestyle="--")
        ax2.set_xlabel("time (fs)")
        ax2.set_ylabel(r"$E_{tot}(t) - E_{tot}(0)$ (eV)")
        max_drift = float(np.abs(drift).max())
        ax2.set_title(f"max |drift| = {max_drift:.4f} eV over {t[-1]:.0f} fs")

    fig.suptitle("Energy conservation: correct integration timestep vs. an under-resolved one\n"
                 "(real NVE trajectories, same system/model/initial condition, only dt differs)")
    fig.tight_layout()
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    groups = ((internal.bonds, "Bond lengths", "Å", 30), (internal.angles, "Angles", "degrees", 30),
              (internal.dihedrals, "Dihedrals", "degrees", 30))
    for axis, (coordinates, title, unit, bins) in zip(axes, groups):
        color = _COORDINATE_TYPE_COLORS[title]
        if not coordinates:
            axis.set_title(f"{title} (0 coordinates)")
            axis.text(0.5, 0.5, "none found", ha="center", va="center", transform=axis.transAxes)
            continue
        all_values = np.concatenate(list(coordinates.values()))
        axis.hist(all_values, bins=bins, density=True, alpha=0.85, color=color)
        axis.set_title(f"{title} ({len(coordinates)} coordinates)")
        axis.set_xlabel(unit)
        axis.set_ylabel("probability density")
        axis.grid(alpha=0.2)
    fig.suptitle("Internal-coordinate distributions -- real NVE trajectory (water cluster)")
    fig.tight_layout()
    fig.savefig(out_internal, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- Summary table -------------------------------------------------------------


def summary_table(stable: dict, unstable: dict, out: Path) -> None:
    stable_etot = stable["energy_eV"] + stable["kinetic_eV"]
    unstable_etot = unstable["energy_eV"] + unstable["kinetic_eV"]
    cell_text = [
        ["Water cluster NVE (dt=0.1 fs)", f"{np.abs(stable_etot - stable_etot[0]).max():.4f}", "eV"],
        ["Water cluster NVE (dt=0.5 fs)", f"{np.abs(unstable_etot - unstable_etot[0]).max():.4f}", "eV"],
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

    energy_conservation_small_system(stable_dict, unstable_dict, OUT_DIR / "chart_energy_conservation_small.png")
    print("wrote chart_energy_conservation_small.png")

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
