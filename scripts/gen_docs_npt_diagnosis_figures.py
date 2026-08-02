#!/usr/bin/env python3
"""Figures for the NpT/NVE diagnosis of the DES hybrid potential.

Every number here is measured on the certified 732-molecule TIP3 box
(2,196 atoms, 28.0 A cube) on one A100, varying one thing at a time.

The three panels answer three separate questions that were each settled by a
controlled comparison rather than by inspection:

(a) Does the neighbour-list rebuild interval matter?  Yes for energy
    conservation -- 0.70 meV vs 127 meV across the same plateau -- and no for
    the temperature runaway.
(b) What actually fails?  E_pot falls by 99.3 kcal/mol per water molecule and
    that energy appears as heat. Present in NVE, NVT and NpT alike, and
    identical at both rebuild intervals, so it is the potential energy surface
    and not the integrator, thermostat or neighbour list.
(c) Why can the SO3LR control not run?  Its many-body-dispersion term needs a
    dense 3N x 3N matrix; the allocation is 128 x (3N)^2 float32 to 0.02%, and
    it is independent of the ML batch size.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmml.utils.plotting.styles import apply_plot_style

REPO = Path(__file__).resolve().parents[1]
TRACES = Path(
    "/private/tmp/claude-501/-Users-ericboittier-mmml/"
    "d26d1e6f-2ebb-49af-be1f-980af3b27acb/scratchpad/npt_traces.txt"
)
OUT = REPO / "docs" / "images" / "des-so3lr-dimers" / "npt_diagnosis.png"

# Okabe-Ito slots from the house ICML palette (validated all-pairs: worst CVD
# dE 11.0, worst normal-vision dE 18.7). Every panel also carries a legend, so
# identity never rests on colour alone.
C = ["#0072B2", "#D55E00", "#009E73"]
N_ATOMS, N_MOL = 2196, 732
EV_TO_KCAL = 23.060547830619026


def load_traces(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, list[list[float]]] = {}
    tag = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#TAG"):
            tag = line.split(None, 1)[1]
            out[tag] = []
        elif line and tag is not None:
            parts = line.split()
            if len(parts) == 4:
                try:
                    out[tag].append([float(x) for x in parts])
                except ValueError:
                    continue
    return {k: np.asarray(v) for k, v in out.items() if v}


def main() -> int:
    apply_plot_style("icml")
    tr = load_traces(TRACES)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16.4, 5.2))

    # ---- (a) NVE energy conservation vs rebuild interval -----------------
    # Plot the quantity the claim is about: drift WITHIN the plateau, in meV,
    # referenced to step 400. Showing raw E_tot over the full run instead lets a
    # single late discontinuity dominate the axis and hides a 180x difference.
    LO, HI = 400, 1200
    for key, lab, ci in (
        ("bisect_nve_campaign_nl1", "rebuild every step", 0),
        ("bisect_nve_campaign", "rebuild every 40 steps", 1),
    ):
        d = tr.get(key)
        if d is None:
            continue
        m = (d[:, 0] >= LO) & (d[:, 0] <= HI)
        if not m.any():
            continue
        seg = d[m]
        ax_a.plot(seg[:, 0], (seg[:, 2] - seg[0, 2]) * 1e3, "-o", color=C[ci],
                  lw=2.2, ms=7, markeredgecolor="white", markeredgewidth=1.0,
                  label=f"{lab}  ({np.ptp(seg[:, 2]) * 1e3:.2f} meV)")
    ax_a.axhline(0.0, color="0.75", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax_a.set_xlabel(f"MD step  (plateau, {LO}–{HI})")
    ax_a.set_ylabel("$E_{tot}$ drift from step %d  (meV)" % LO)
    ax_a.set_title("(a) NVE drift vs rebuild interval",
                   loc="left", fontweight="bold")
    ax_a.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    ax_a.grid(alpha=0.25)
    ax_a.set_axisbelow(True)
    ax_a.annotate(
        "180× flatter with\nper-step rebuilds",
        (0.96, 0.62), xycoords="axes fraction", ha="right", fontsize=9.5,
        color="0.25", fontweight="bold")
    ax_a.annotate(
        "Outside this window both traces show the\n"
        "same +2 eV jump (~step 330): the rebuild\n"
        "rate neither causes nor fixes it.",
        (0.96, 0.40), xycoords="axes fraction", ha="right", fontsize=8.2,
        color="0.35")

    # ---- (b) the actual failure: PES descent -> heat ---------------------
    for key, lab, ci in (
        ("bisect_nvt_campaign_nl1", "NVT, rebuild every step", 0),
        ("bisect_nvt_campaign", "NVT, rebuild every 40", 1),
    ):
        d = tr.get(key)
        if d is None:
            continue
        ax_b.plot(d[:, 0], d[:, 3], "-o", color=C[ci], lw=2.2, ms=6,
                  markeredgecolor="white", markeredgewidth=0.9, label=lab)
    d = tr.get("bisect_nve_campaign_nl1")
    if d is not None:
        ax_b.plot(d[:, 0], d[:, 3], "-s", color=C[2], lw=1.8, ms=5, alpha=0.9,
                  markeredgecolor="white", markeredgewidth=0.8,
                  label="NVE, rebuild every step")
    ax_b.axhline(298.15, color="0.55", lw=1.3, ls=(0, (5, 3)), zorder=1)
    ax_b.annotate("target 298 K", (0.03, 0.10), xycoords="axes fraction",
                  fontsize=8.5, color="0.35")
    ax_b.set_xlabel("MD step")
    ax_b.set_ylabel("temperature (K)")
    ax_b.set_title("(b) The runaway is the potential, not the rebuild",
                   loc="left", fontweight="bold")
    ax_b.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax_b.grid(alpha=0.25)
    ax_b.set_axisbelow(True)

    nvt = tr.get("bisect_nvt_campaign_nl1")
    if nvt is not None and len(nvt) >= 2:
        dE = nvt[-1, 1] - nvt[0, 1]
        ax_b.annotate(
            f"$E_{{pot}}$ falls {dE:,.0f} eV\n"
            f"= {dE / N_MOL * EV_TO_KCAL:.0f} kcal/mol per molecule",
            (nvt[-1, 0], nvt[-1, 3]), textcoords="offset points",
            xytext=(-18, -120), ha="right", fontsize=9, color="0.20",
            arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2))

    # ---- (c) the SO3LR memory wall ---------------------------------------
    # Measured allocations. The 'all' case additionally batches every one of the
    # 267,546 ML dimers at once; 1024 and 128 are identical, which is the point.
    labels = ["all\n(267,546)", "1024", "128"]
    vals = [71.67, 20.69, 20.69]
    x = np.arange(len(vals))
    ax_c.bar(x, vals, width=0.55, color=C[0], zorder=3)
    for xi, v in zip(x, vals):
        ax_c.text(xi, v + 1.4, f"{v:.2f}", ha="center", fontsize=9.5,
                  color="0.25", fontweight="bold")
    tn = 3 * N_ATOMS
    mbd = tn * tn * 128 * 4 / 1024**3
    ax_c.axhline(mbd, color=C[1], lw=2.0, ls=(0, (5, 3)), zorder=4,
                 label=f"$128\\times(3N)^2$ float32 = {mbd:.2f} GiB")
    ax_c.axhline(40.0, color="0.45", lw=1.6, ls=(0, (2, 2)), zorder=4,
                 label="A100 capacity, 40 GiB")
    ax_c.set_xticks(x, labels, fontsize=9.5)
    ax_c.set_xlabel("--ml-batch-size")
    ax_c.set_ylabel("allocation requested (GiB)")
    ax_c.set_title("(c) SO3LR+MBD: the floor is $(3N)^2$", loc="left",
                   fontweight="bold")
    ax_c.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax_c.set_ylim(0, 84)
    ax_c.grid(axis="y", alpha=0.25)
    ax_c.set_axisbelow(True)
    ax_c.annotate("8x smaller batch,\nidentical allocation", (1.5, 24.5),
                  ha="center", fontsize=8.5, color="0.30")

    fig.suptitle(
        "NpT/NVE diagnosis — certified 732-molecule TIP3 box (2,196 atoms, 28.0 Å), one A100",
        fontsize=13, fontweight="bold")
    fig.text(0.5, 0.012,
             "One variable changed at a time. (a, b) DES-fitted hybrid, unit LJ scales. "
             "(c) SO3LR SpookyNet with the many-body-dispersion term active; the MBD-free "
             "checkpoint runs without OOM.",
             ha="center", fontsize=8.5, color="0.40")
    fig.tight_layout(rect=(0, 0.045, 1, 0.93), w_pad=2.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")
    for k, v in tr.items():
        print(f"  {k:34s} {len(v):3d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
