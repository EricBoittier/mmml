#!/usr/bin/env python3
"""Figures for the DES LJ-scale prior sweep and seed reproducibility.

Reads the scale vectors collected from the completed fits
(``artifacts/des_chemspace/lj_scale_bounds_sweep.json``) and writes

* ``lj_scale_prior_sweep.png``  -- how the fitted scales respond to the prior
* ``lj_scale_seed_spread.png``  -- seed-to-seed reproducibility

The bounds are a Bayesian prior on a component of the energy and force that may
generalise at long range, so a scale sitting near its bound is the prior doing
its job, not a failure. What the sweep tests is whether the likelihood ever
takes over: it does, and epsilon converges at 5.53 once the ceiling is lifted
above 4.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmml.utils.plotting.styles import apply_plot_style

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "artifacts" / "des_chemspace" / "lj_scale_bounds_sweep.json"
OUT = REPO / "docs" / "images" / "des-so3lr-dimers"

# Okabe-Ito slots from the house ICML palette, validated all-pairs
# (worst CVD dE 11.0, worst normal-vision dE 18.7) so the three priors stay
# distinguishable under colour-vision deficiency. Every panel also carries a
# legend or direct labels, so identity never rests on colour alone.
C = ["#0072B2", "#D55E00", "#009E73"]
GREY = "#6E6E6E"

PRIORS = [
    ("tight", "σ ±5%", 0),
    ("prod", "σ ±20%", 1),
    ("wide", "σ ±40%", 2),
]
NEAR_TOL = 1e-4


def _near_bound(x: np.ndarray, lo: float, hi: float, tol: float = NEAR_TOL) -> np.ndarray:
    return (np.abs(x - lo) <= tol) | (np.abs(x - hi) <= tol)


def prior_sweep(d: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    ax_s, ax_e, ax_f = axes

    rng = np.random.default_rng(0)

    # (a) sigma scales. Strip plot, not a box plot: n is ~90 and the interesting
    # structure is the pile-up at the bounds, which a box plot hides.
    for key, label, ci in PRIORS:
        v = d[key]
        s = np.asarray(v["sigma"])
        lo, hi = v["bounds"][0], v["bounds"][1]
        y = ci + rng.uniform(-0.16, 0.16, s.size)
        near = _near_bound(s, lo, hi)
        ax_s.scatter(s[~near], y[~near], s=22, color=C[ci], alpha=0.75,
                     edgecolors="white", linewidths=0.4, zorder=3)
        ax_s.scatter(s[near], y[near], s=48, facecolors="none",
                     edgecolors=C[ci], linewidths=1.5, zorder=4)
        ax_s.plot([lo, lo], [ci - 0.3, ci + 0.3], color=GREY, lw=1.4, zorder=2)
        ax_s.plot([hi, hi], [ci - 0.3, ci + 0.3], color=GREY, lw=1.4, zorder=2)
        ax_s.annotate(f"{int(near.sum())}/{s.size} near a bound",
                      (hi, ci + 0.30), fontsize=8.5, color="0.30",
                      ha="right", va="bottom")
    ax_s.axvline(1.0, color="0.75", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax_s.set_yticks(range(3), [p[1] for p in PRIORS], fontsize=10)
    ax_s.set_xlabel("fitted σ scale")
    ax_s.set_title("(a) σ scales vs the prior width", loc="left", fontweight="bold")
    ax_s.set_ylim(-0.55, 2.6)
    ax_s.grid(axis="x", alpha=0.25)
    ax_s.set_axisbelow(True)
    ax_s.annotate("open marker = within 1e-4 of a bound;\ngrey bars = the prior",
                  (0.02, 0.03), xycoords="axes fraction", fontsize=8.5, color="0.35")

    # (b) epsilon scales, log axis: the fitted values span two decades.
    for key, label, ci in PRIORS:
        v = d[key]
        e = np.asarray(v["eps"])
        elo, ehi = v["bounds"][2], v["bounds"][3]
        y = ci + rng.uniform(-0.16, 0.16, e.size)
        near = _near_bound(e, elo, ehi)
        ax_e.scatter(e[~near], y[~near], s=22, color=C[ci], alpha=0.75,
                     edgecolors="white", linewidths=0.4, zorder=3)
        ax_e.scatter(e[near], y[near], s=48, facecolors="none",
                     edgecolors=C[ci], linewidths=1.5, zorder=4)
        for b in (elo, ehi):
            ax_e.plot([b, b], [ci - 0.3, ci + 0.3], color=GREY, lw=1.4, zorder=2)
    ax_e.axvline(1.0, color="0.75", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax_e.set_xscale("log")
    ax_e.set_yticks(range(3), ["ε [0.25, 4]", "ε [0.25, 4]", "ε [0.05, 20]"], fontsize=10)
    ax_e.set_xlabel("fitted ε scale (log)")
    ax_e.set_title("(b) ε scales — the ceiling was binding", loc="left", fontweight="bold")
    ax_e.set_ylim(-0.55, 2.6)
    ax_e.grid(axis="x", alpha=0.25)
    ax_e.set_axisbelow(True)
    e_wide = np.asarray(d["wide"]["eps"])
    ax_e.annotate(f"lifting the ceiling to 20\nlets ε settle at {e_wide.max():.2f}",
                  (e_wide.max(), 2.0), textcoords="offset points", xytext=(-6, 26),
                  ha="right", fontsize=9, color="0.20",
                  arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2))

    # (c) the headline: widening the prior releases the scales.
    widths, counts, labels = [], [], []
    for key, label, ci in PRIORS:
        v = d[key]
        s = np.asarray(v["sigma"])
        lo, hi = v["bounds"][0], v["bounds"][1]
        widths.append(hi - lo)
        counts.append(100.0 * _near_bound(s, lo, hi).mean())
        labels.append(label)
    ax_f.plot(widths, counts, "-o", color=C[0], lw=2.0, ms=9,
              markeredgecolor="white", markeredgewidth=1.2)
    for w, c, lab in zip(widths, counts, labels):
        ax_f.annotate(f"{lab}\n{c:.0f}%", (w, c), textcoords="offset points",
                      xytext=(8, 8), fontsize=9, color="0.25")
    ax_f.set_xlabel("prior width on σ (max − min)")
    ax_f.set_ylabel("σ scales within 1e-4 of a bound (%)")
    ax_f.set_title("(c) A wider prior releases the fit", loc="left", fontweight="bold")
    ax_f.set_xlim(0.0, 1.15)
    ax_f.set_ylim(0, max(counts) * 1.35)
    ax_f.grid(alpha=0.25)
    ax_f.set_axisbelow(True)

    fig.suptitle(
        "DES LJ scales — response to the prior  "
        "(94 reachable CGenFF types, warm-started, 25 epochs)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def seed_spread(d: dict, out: Path) -> None:
    keys = ["prod", "prod_seed7", "prod_seed2026"]
    names = ["seed 42", "seed 7", "seed 2026"]
    if not all(k in d for k in keys):
        print("  seed runs missing; skipping seed figure")
        return

    S = np.array([d[k]["sigma"] for k in keys])
    E = np.array([d[k]["eps"] for k in keys])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))

    order = np.argsort(S[0])
    x = np.arange(S.shape[1])
    for i, (nm, c) in enumerate(zip(names, C)):
        ax1.plot(x, S[i][order], ".", ms=7, color=c, alpha=0.8, label=nm)
    ax1.axhline(1.0, color="0.75", lw=1.0, ls=(0, (4, 3)))
    ax1.set_xlabel("CGenFF type (sorted by seed-42 σ scale)")
    ax1.set_ylabel("fitted σ scale")
    ax1.set_title("(a) Per-type σ across three seeds", loc="left", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax1.grid(alpha=0.25)
    ax1.set_axisbelow(True)

    # Spread per type: max - min across seeds. This is the honest measure of
    # how much of a fitted scale is optimiser noise rather than signal.
    spread_s = S.max(0) - S.min(0)
    spread_e = np.abs(E.max(0) - E.min(0)) / np.maximum(E.mean(0), 1e-12)
    ax2.hist(spread_s, bins=24, color=C[0], alpha=0.85, label="σ  (absolute)")
    ax2.set_xlabel("seed-to-seed spread in σ (max − min)")
    ax2.set_ylabel("CGenFF types")
    ax2.set_title("(b) How much is optimiser noise", loc="left", fontweight="bold")
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_axisbelow(True)
    ax2.annotate(
        f"median σ spread {np.median(spread_s):.4f}\n"
        f"median ε spread {100 * np.median(spread_e):.1f}% of mean\n"
        f"σ means: " + ", ".join(f"{s.mean():.4f}" for s in S),
        (0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=9, color="0.25",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.85"),
    )

    fig.suptitle(
        "DES LJ scales — seed reproducibility (σ ±20% prior, disjoint 108k/12k split)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    apply_plot_style("icml")
    d = json.loads(DATA.read_text())
    prior_sweep(d, OUT / "lj_scale_prior_sweep.png")
    seed_spread(d, OUT / "lj_scale_seed_spread.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
