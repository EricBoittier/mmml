#!/usr/bin/env python3
"""Summarize finite-difference (analytic-vs-numerical force) validation
results as figures, under the house style.

Two sources, deliberately contrasted:

1. **SMD bias term** (`mmml.md.energy.terms.SMDBiasTerm`) -- the analytic
   `ase_contribution` force checked against a central finite difference,
   recomputed here the same way `tests/unit/test_md_energy_terms.py::
   test_ase_forces_match_finite_difference` does. This one PASSES
   (atol=1e-4): a clean small-molecule autodiff-vs-FD check.
2. **CHARMM/mlpot ML-only calculator**, from the saved result at
   `artifacts/pycharmm_mlpot/mlpot_force_fd.json` -- a real run from
   `mmml/interfaces/pycharmmInterface/mlpot/derivative_test.py`. This one
   currently FAILS all 60 checked force components (tol=0.005 kcal/mol/A) --
   included as-is (not cherry-picked) since it's the actual state of that
   integration, and directly relevant to the open "real neighbor-list
   support" item in docs/md-cg-unification-design.md SS11.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmml.utils.plotting.styles import apply_plot_style, comparison_colors, legend_outside

STYLE_NAME = "icml"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "plot-style-gallery-assets"
MLPOT_FD_JSON = REPO_ROOT / "artifacts" / "pycharmm_mlpot" / "mlpot_force_fd.json"


def _smd_fd_check():
    """Recompute the SMD-bias-term analytic-vs-FD force check (same
    procedure as test_ase_forces_match_finite_difference) and return
    (analytic, fd) force arrays, shape (n_atoms, 3)."""

    from mmml.md.energy.registry import EnergyContext
    from mmml.md.energy.terms import SMDBiasTerm
    from mmml.md.system import MolecularSystem

    rng = np.random.default_rng(5)
    R = rng.uniform(-5, 5, size=(6, 3))
    system = MolecularSystem(R=R, Z=np.ones(6, int), box=None, mol_id=np.arange(6))
    term = SMDBiasTerm(atom_i=0, atom_j=5, k_ev_per_A2=2.0, target=3.0)
    contribution = term.make(system, EnergyContext()).ase_contribution

    class _Stub:
        def __init__(self, pos):
            self._pos = pos

        def get_positions(self):
            return self._pos

        def __len__(self):
            return len(self._pos)

    pos = np.asarray(R)
    _, analytic = contribution(_Stub(pos))

    h = 1e-5
    fd = np.zeros_like(pos)
    for a in range(pos.shape[0]):
        for c in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[a, c] += h
            pm[a, c] -= h
            ep, _ = contribution(_Stub(pp))
            em, _ = contribution(_Stub(pm))
            fd[a, c] = -(ep - em) / (2 * h)
    return np.asarray(analytic), fd


def _load_mlpot_fd():
    data = json.loads(MLPOT_FD_JSON.read_text())
    analytic = np.array([c["analytic_force_kcalmol_A"] for c in data["components"]])
    fd = np.array([c["fd_force_kcalmol_A"] for c in data["components"]])
    return data, analytic, fd


def parity_and_residuals(out: Path) -> None:
    """Analytic-vs-FD parity scatter + residual histogram, one row per
    check -- the SMD term (passes) on top, the mlpot ML calculator (fails)
    on the bottom, same axis scale within each row so the eye reads
    "on the diagonal = correct" directly.
    """
    smd_analytic, smd_fd = _smd_fd_check()
    data, mlpot_analytic, mlpot_fd = _load_mlpot_fd()
    colors = comparison_colors(STYLE_NAME, n=2)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Row 1: SMD bias term (passes)
    ax = axes[0, 0]
    lim = max(np.abs(smd_analytic).max(), np.abs(smd_fd).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    ax.scatter(smd_analytic, smd_fd, s=40, color=colors[0], edgecolor="#222222", linewidth=0.5, zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("analytic force (eV/Å)")
    ax.set_ylabel("finite-difference force (eV/Å)")
    ax.set_title("SMD bias term: PASSES (atol=1e-4)", color="#2E7D32", fontweight="bold")

    ax = axes[0, 1]
    smd_resid = (smd_fd - smd_analytic).ravel()
    ax.hist(smd_resid, bins=14, color=colors[0], edgecolor="#222222", linewidth=0.6)
    ax.axvline(0, color="#222222", linewidth=1.0)
    ax.set_xlabel("FD - analytic (eV/Å)")
    ax.set_ylabel("count (of 18 components)")
    ax.set_title(f"max |residual| = {np.abs(smd_resid).max():.2e} eV/Å")

    # Row 2: mlpot ML-only calculator (fails)
    ax = axes[1, 0]
    lim = max(np.abs(mlpot_analytic).max(), np.abs(mlpot_fd).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    ax.scatter(mlpot_analytic, mlpot_fd, s=30, color=colors[1], edgecolor="#222222", linewidth=0.4,
               alpha=0.85, zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("analytic force (kcal/mol/Å)")
    ax.set_ylabel("finite-difference force (kcal/mol/Å)")
    ax.set_title(f"CHARMM/mlpot ML calc: FAILS ({data['n_fail']}/{data['n_components_checked']})",
                 color="#C62828", fontweight="bold")

    ax = axes[1, 1]
    mlpot_resid = mlpot_fd - mlpot_analytic
    ax.hist(mlpot_resid, bins=16, color=colors[1], edgecolor="#222222", linewidth=0.6)
    ax.axvline(0, color="#222222", linewidth=1.0)
    ax.set_xlabel("FD - analytic (kcal/mol/Å)")
    ax.set_ylabel(f"count (of {data['n_components_checked']} components)")
    ax.set_title(f"max |residual| = {data['max_abs_diff_kcalmol_A']:.3f}, "
                 f"rms = {data['rms_abs_diff_kcalmol_A']:.3f} kcal/mol/Å")

    fig.suptitle("Finite-difference force checks: analytic vs. numerical derivative")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def per_atom_mlpot_diff(out: Path) -> None:
    """Per-atom max |analytic - FD| for the failing mlpot check -- reads
    whether the discrepancy is spread evenly (systematic, e.g. a
    missing/extra term) or concentrated on a few atoms (e.g. a boundary or
    cutoff-related bug) at a glance.
    """
    data, _, _ = _load_mlpot_fd()
    n_atoms = data["n_atoms_selected"]
    per_atom_max = np.zeros(n_atoms)
    for c in data["components"]:
        a = c["atom"] - 1
        per_atom_max[a] = max(per_atom_max[a], c["abs_diff_kcalmol_A"])

    fig, ax = plt.subplots(figsize=(9, 5))
    color = comparison_colors(STYLE_NAME, n=1)[0]
    ax.bar(np.arange(1, n_atoms + 1), per_atom_max, color=color, edgecolor="#222222", linewidth=0.6)
    ax.axhline(data["tol_kcalmol_A"], color="#C62828", linewidth=1.4, linestyle="--",
               label=f"tolerance ({data['tol_kcalmol_A']} kcal/mol/Å)")
    ax.set_xlabel("atom index")
    ax.set_ylabel("max |analytic - FD| (kcal/mol/Å)")
    ax.set_title("CHARMM/mlpot FD check: discrepancy spread evenly across all 20 atoms\n"
                 "(not localized -- consistent with a systematic force-term issue, not a boundary bug)")
    ax.set_xticks(range(1, n_atoms + 1, 2))
    legend_outside(ax)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style(STYLE_NAME)

    parity_and_residuals(OUT_DIR / "chart_fd_parity_residuals.png")
    print(f"wrote {OUT_DIR / 'chart_fd_parity_residuals.png'}")
    per_atom_mlpot_diff(OUT_DIR / "chart_fd_mlpot_per_atom.png")
    print(f"wrote {OUT_DIR / 'chart_fd_mlpot_per_atom.png'}")


if __name__ == "__main__":
    main()
