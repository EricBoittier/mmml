#!/usr/bin/env python3
"""Generate the validation figures for docs/prepare-mm-dataset.md.

Everything here is produced from the *real* CGenFF assignment pipeline
(:mod:`mmml.data.cgenff_dataset`) on real monomer geometries
(:mod:`mmml.analysis.dimer_molecules`), so the page proves the workflow rather
than illustrating it.  Outputs -> docs/images/prepare-mm-dataset/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ase.data import covalent_radii
from ase.data.colors import jmol_colors

from mmml.analysis.dimer_molecules import MOLECULES, make_oriented_scan_geometries
from mmml.data.cgenff_dataset import (
    assign_frame_cgenff,
    compute_inter_monomer_mm,
    load_reference,
    match_cgenff_template,
)
from mmml.utils.plotting.styles import apply_plot_style, comparison_colors

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "prepare-mm-dataset"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = apply_plot_style("icml")
COLORS = comparison_colors(STYLE, n=6)

REF = load_reference()
IDX_TO_NAME = {v: k for k, v in REF.nb_map.items()}

# ACE (ASE) uses the CGenFF residue name ACO; map for display.
_DISPLAY_NAME = {"ACE": "ACO (acetone)", "DCM": "DCM", "BENZ": "BENZ", "TIP3": "TIP3", "MEOH": "MEOH"}


def _pca_2d(pos: np.ndarray) -> np.ndarray:
    """Project 3D coords onto their two principal axes (max in-plane spread)."""
    c = pos - pos.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[:2].T


def _draw_molecule(ax, z, pos2d, labels, *, title=None):
    """Ball-and-stick with CGenFF type labels; pos2d already projected."""
    # Bonds
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            r = np.linalg.norm(pos2d[i] - pos2d[j])
            if r < 1.35 * (covalent_radii[z[i]] + covalent_radii[z[j]]):
                ax.plot(*zip(pos2d[i], pos2d[j]), color="0.4", lw=2.0, zorder=1)
    for i, zi in enumerate(z):
        ax.scatter(*pos2d[i], s=380 * covalent_radii[zi], color=jmol_colors[zi],
                   edgecolors="0.2", linewidths=0.8, zorder=2)
        ax.annotate(labels[i], pos2d[i], textcoords="offset points", xytext=(7, 6),
                    fontsize=8, fontweight="bold", color="black", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)


def fig_monomer_atom_types():
    """Montage: each supported monomer with its assigned CGenFF type per atom."""
    order = ["DCM", "ACE", "BENZ", "TIP3", "MEOH"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, name in zip(axes, order):
        atoms = MOLECULES[name]
        z = atoms.get_atomic_numbers()
        r = atoms.get_positions()
        resi, tidx, q = match_cgenff_template(REF, z, r)
        labels = [IDX_TO_NAME[int(t)] for t in tidx]
        _draw_molecule(ax, z, _pca_2d(r), labels, title=_DISPLAY_NAME[name])
        ax.text(0.5, -0.02, f"RESI {resi} · Σq = {q.sum():+.2f} e",
                transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.35")
    fig.suptitle("CGenFF atom types assigned by composition → RESI template match",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "monomer_atom_types.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _acodcm_frame(separation=4.0):
    g = next(make_oriented_scan_geometries("ACE", "DCM", [separation], offsets_angstrom=[0.0]))
    atoms = g.atoms
    return atoms.get_atomic_numbers(), atoms.get_positions()


def fig_acodcm_assignment():
    """The tutorial ACO–DCM dimer: segmentation + per-atom assignment."""
    z, r = _acodcm_frame(4.0)
    a, reason = assign_frame_cgenff(z, r, REF)
    assert a is not None, reason
    labels = [IDX_TO_NAME[int(t)] for t in a.cgenff_type_idx]
    pos2d = _pca_2d(r)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    mol_colors = {0: COLORS[0], 1: COLORS[1]}
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            rr = np.linalg.norm(pos2d[i] - pos2d[j])
            if rr < 1.35 * (covalent_radii[z[i]] + covalent_radii[z[j]]):
                ax.plot(*zip(pos2d[i], pos2d[j]), color="0.5", lw=2.0, zorder=1)
    for i, zi in enumerate(z):
        ax.scatter(*pos2d[i], s=420 * covalent_radii[zi], color=jmol_colors[zi],
                   edgecolors=mol_colors[int(a.mol_id[i])], linewidths=2.4, zorder=2)
        ax.annotate(f"{labels[i]}\n{a.cgenff_charge[i]:+.2f}", pos2d[i],
                    textcoords="offset points", xytext=(8, 6), fontsize=7.5,
                    fontweight="bold", zorder=3)
    ax.scatter([], [], color="w", edgecolors=mol_colors[0], linewidths=2.4, s=120,
               label=f"mol_id 0 · ACO (Σq={a.cgenff_charge[a.mol_id == 0].sum():+.2f} e)")
    ax.scatter([], [], color="w", edgecolors=mol_colors[1], linewidths=2.4, s=120,
               label=f"mol_id 1 · DCM (Σq={a.cgenff_charge[a.mol_id == 1].sum():+.2f} e)")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    ax.set_title("acodcm frame: 2 covalent components → CGenFF type + conserved charge\n"
                 f"(ring outline = mol_id · label = type / charge · E_MM = {a.e_cgenff_mm*1000:.1f} meV)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "acodcm_assignment.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_mm_baseline():
    """Inter-monomer CGenFF MM energy vs COM separation, LJ/Coulomb decomposed."""
    seps = np.linspace(2.6, 8.0, 40)
    tot, lj, coul = [], [], []
    for s in seps:
        z, r = _acodcm_frame(float(s))
        a, _ = assign_frame_cgenff(z, r, REF, compute_mm=False)
        ca = np.flatnonzero(a.mol_id == 0)
        cb = np.flatnonzero(a.mol_id == 1)
        ta, tb = a.cgenff_type_idx[ca], a.cgenff_type_idx[cb]
        qa, qb = a.cgenff_charge[ca], a.cgenff_charge[cb]
        e_tot, _ = compute_inter_monomer_mm(REF, r, list(ca), ta, qa, list(cb), tb, qb)
        e_lj, _ = compute_inter_monomer_mm(REF, r, list(ca), ta, qa * 0, list(cb), tb, qb * 0)
        tot.append(e_tot * 1000)
        lj.append(e_lj * 1000)
        coul.append((e_tot - e_lj) * 1000)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.plot(seps, tot, color=COLORS[0], lw=2.4, label="total $E_{MM}$")
    ax.plot(seps, lj, color=COLORS[1], lw=1.8, ls="--", label="Lennard-Jones")
    ax.plot(seps, coul, color=COLORS[2], lw=1.8, ls=":", label="Coulomb")
    imin = int(np.argmin(tot))
    ax.scatter([seps[imin]], [tot[imin]], color=COLORS[0], zorder=5)
    ax.annotate(f"min {tot[imin]:.0f} meV\n@ {seps[imin]:.2f} Å",
                (seps[imin], tot[imin]), textcoords="offset points", xytext=(12, -6), fontsize=8)
    ax.set_xlabel("centre-of-mass separation  (Å)")
    ax.set_ylabel("inter-monomer energy  (meV)")
    ax.set_title("ACO–DCM CGenFF MM baseline (C–H···O=C approach)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mm_baseline_decomposition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_force_validation():
    """Analytic F_cgenff_mm vs central finite-difference of E_cgenff_mm."""
    z, r = _acodcm_frame(3.8)
    a, _ = assign_frame_cgenff(z, r, REF)
    ca = np.flatnonzero(a.mol_id == 0)
    cb = np.flatnonzero(a.mol_id == 1)
    ta, tb = a.cgenff_type_idx[ca], a.cgenff_type_idx[cb]
    qa, qb = a.cgenff_charge[ca], a.cgenff_charge[cb]

    def energy(pos):
        e, _ = compute_inter_monomer_mm(REF, pos, list(ca), ta, qa, list(cb), tb, qb)
        return e

    h = 1e-4
    fd = np.zeros_like(r)
    for i in range(len(z)):
        for d in range(3):
            rp = r.copy(); rp[i, d] += h
            rm = r.copy(); rm[i, d] -= h
            fd[i, d] = -(energy(rp) - energy(rm)) / (2 * h)

    ana = a.f_cgenff_mm
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    lim = 1.05 * max(np.abs(ana).max(), np.abs(fd).max())
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=1.0, zorder=1)
    ax.scatter(fd.ravel(), ana.ravel(), s=36, color=COLORS[0], edgecolors="0.2",
               linewidths=0.5, zorder=2)
    max_err = np.abs(ana - fd).max()
    ax.set_xlabel("finite-difference force  (eV/Å)")
    ax.set_ylabel("analytic $F_{cgenff\\_mm}$  (eV/Å)")
    ax.set_title(f"Force baseline is a true gradient\nmax |Δ| = {max_err:.2e} eV/Å")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "force_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return max_err


def fig_charge_conservation():
    """Per-monomer net charge across a jittered ensemble -> machine-precision zero."""
    rng = np.random.default_rng(0)
    net_a, net_b = [], []
    for _ in range(300):
        z, r = _acodcm_frame(float(rng.uniform(3.0, 6.0)))
        r = r + rng.normal(scale=0.05, size=r.shape)
        a, reason = assign_frame_cgenff(z, r, REF)
        if a is None:
            continue
        net_a.append(a.cgenff_charge[a.mol_id == 0].sum())
        net_b.append(a.cgenff_charge[a.mol_id == 1].sum())
    net_a = np.array(net_a); net_b = np.array(net_b)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.plot(net_a / 1e-16, ".", color=COLORS[0], ms=4, label="ACO net charge")
    ax.plot(net_b / 1e-16, ".", color=COLORS[1], ms=4, label="DCM net charge")
    ax.set_xlabel("frame")
    ax.set_ylabel("net monomer charge  ($10^{-16}$ e)")
    ax.set_title(f"Strict per-monomer charge conservation\n"
                 f"max |Σq| = {max(np.abs(net_a).max(), np.abs(net_b).max()):.1e} e  "
                 f"over {len(net_a)} jittered frames")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "charge_conservation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_monomer_atom_types()
    print("  monomer_atom_types.png")
    fig_acodcm_assignment()
    print("  acodcm_assignment.png")
    fig_mm_baseline()
    print("  mm_baseline_decomposition.png")
    err = fig_force_validation()
    print(f"  force_validation.png (max |Δ| = {err:.2e} eV/Å)")
    fig_charge_conservation()
    print("  charge_conservation.png")
    print(f"[+] figures written to {OUT}")


if __name__ == "__main__":
    main()
