#!/usr/bin/env python3
"""Figure: which DES species have a condensed-phase reference, and at what density.

Answers two questions that decide what can actually be validated:

1. For how many of the 94 DES residues does a liquid-phase reference exist *at
   all*? (Not many. Ten are bare ions, for which a pure-liquid reference is not
   merely unavailable but meaningless.)
2. Where are the low-density state points? The noble gases have NIST reference
   equations of state over their whole saturation curve, and near the critical
   point they are ~10x sparser than water in atoms/A^3 while remaining pure
   Lennard-Jones -- no charges, no intramolecular terms.

Mass density is deliberately *not* the comparison axis. Krypton at 120 K is
2.41 g/cm3, denser than water by mass, while being 6x sparser in atoms per A^3.
Number density is what sets the neighbour-list cost and what "dense" means for a
pair potential; g/cm3 across different molar masses just compares molar masses.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmml.data.reference_state_points import SPECIES, Phase
from mmml.utils.plotting.styles import apply_plot_style

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "images" / "des-so3lr-dimers" / "reference_state_points.png"

C = ["#0072B2", "#D55E00", "#009E73"]
NA = 6.02214076e23
# (molar mass g/mol, atoms per molecule) for the species drawn here.
MOLAR = {"AR1": (39.948, 1), "KR1": (83.798, 1), "TIP3": (18.015, 3), "MEOH": (32.04, 6)}


def number_density(rho_g_cm3: float, resname: str) -> float:
    """atoms per A^3 -- the density that matters for a pair potential."""
    M, n_at = MOLAR[resname]
    return rho_g_cm3 / M * NA * 1e-24 * n_at


LABELS = {
    Phase.NIST_EOS: "NIST reference EOS\n(density at any T, P)",
    Phase.LIQUID: "liquid at 298 K\n(single-point lookup)",
    Phase.GAS: "gas at 298 K\n(run below the NBP)",
    Phase.SOLID: "solid at 298 K\n(run above the MP)",
    Phase.ION: "bare ion\n(no pure-liquid reference exists)",
}
ORDER = [Phase.NIST_EOS, Phase.LIQUID, Phase.GAS, Phase.SOLID, Phase.ION]


def main() -> int:
    apply_plot_style("icml")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14.6, 5.6))

    # -- (a) how much reference data exists at all -------------------------
    # Single series, so no legend: the categories are on the axis and the bars
    # carry direct labels. Counts of species, and of the frames behind them.
    counts = Counter(s.phase_298 for s in SPECIES)
    frames = Counter()
    for s in SPECIES:
        frames[s.phase_298] += s.frames
    y = np.arange(len(ORDER))[::-1]
    vals = [counts[p] for p in ORDER]
    ax_a.barh(y, vals, height=0.62, color=C[0], zorder=3)
    for yy, p, v in zip(y, ORDER, vals):
        ax_a.text(v + 0.6, yy, f"{v} species · {frames[p]:,} frames",
                  va="center", fontsize=9, color="0.25")
    ax_a.set_yticks(y, [LABELS[p] for p in ORDER], fontsize=9.5)
    ax_a.set_xlabel("species classified")
    ax_a.set_xlim(0, max(vals) * 1.75)
    ax_a.set_title("(a) What reference exists, by species", loc="left",
                   fontweight="bold")
    ax_a.grid(axis="x", alpha=0.25)
    ax_a.set_axisbelow(True)
    n_unclassified = 94 - len(SPECIES)
    ax_a.annotate(
        f"{len(SPECIES)} of 94 DES residues classified;\n"
        f"the remaining {n_unclassified} are thin (<60 frames)",
        (0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=8.5,
        color="0.35")

    # -- (b) the low-density opportunity -----------------------------------
    for (res, label, ci) in (("AR1", "argon (pure LJ)", 0), ("KR1", "krypton (pure LJ)", 1)):
        sp = next(s for s in SPECIES if s.resname == res)
        T = np.array([st.T_K for st in sp.states])
        n = np.array([number_density(st.density_g_cm3, res) for st in sp.states])
        ax_b.plot(T, n, "-o", color=C[ci], lw=2.2, ms=8, label=label,
                  markeredgecolor="white", markeredgewidth=1.2, zorder=3)

    # Water and methanol as the reference for "dense", at 298 K.
    for res, rho, lab in (("TIP3", 0.99705, "water, 298 K"), ("MEOH", 0.7866, "methanol, 298 K")):
        nd = number_density(rho, res)
        ax_b.axhline(nd, color="0.55", lw=1.3, ls=(0, (5, 3)), zorder=2)
        ax_b.annotate(f"{lab}  ({nd:.3f})", (88, nd), fontsize=8.5, color="0.35",
                      va="bottom", ha="left")

    ar = next(s for s in SPECIES if s.resname == "AR1")
    lo = number_density(min(st.density_g_cm3 for st in ar.states), "AR1")
    hi = number_density(0.99705, "TIP3")
    ax_b.annotate(
        f"argon near Tc is {hi / lo:.0f}x sparser\nthan water — and has no charges",
        (150, lo), textcoords="offset points", xytext=(-10, 34), ha="right",
        fontsize=9, color="0.20",
        arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2))

    ax_b.set_xlabel("temperature (K)  — along the saturation curve")
    ax_b.set_ylabel("number density (atoms / Å³)")
    ax_b.set_title("(b) Where the sparse state points are", loc="left",
                   fontweight="bold")
    ax_b.legend(loc="center right", fontsize=9, framealpha=0.95)
    ax_b.set_ylim(0, 0.115)
    ax_b.grid(alpha=0.25)
    ax_b.set_axisbelow(True)

    fig.suptitle(
        "Condensed-phase references for the DES species — availability and accessible density",
        fontsize=13, fontweight="bold")
    fig.text(0.5, 0.012,
             "Saturated-liquid densities from the NIST Chemistry WebBook (argon C7440371, "
             "krypton C7439909), converted to atoms/Å³. Mass density would mislead here: "
             "krypton at 120 K is 2.41 g/cm³, denser than water, yet 6× sparser in atoms.",
             ha="center", fontsize=8.5, color="0.40")
    fig.tight_layout(rect=(0, 0.045, 1, 0.93), w_pad=3.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
