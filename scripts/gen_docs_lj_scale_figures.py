#!/usr/bin/env python3
"""Figures for the DES LJ-scale prior sweep and seed reproducibility.

Reads the scale vectors collected from the completed fits
(``artifacts/des_chemspace/lj_scale_bounds_sweep.json``) and writes

* ``lj_scale_prior_sweep.png``  -- how the fitted scales respond to the prior
* ``lj_scale_seed_spread.png``  -- seed-to-seed reproducibility

The bounds are a Bayesian prior on a component of the energy and force that may
generalise at long range, so a scale sitting near its bound is the prior doing
its job, not a failure. What the sweep tests is whether the likelihood ever
takes over. It does for sigma and for the epsilon ceiling -- widening sigma to
+/-40% drops the fraction sitting within 1% of the prior width of a bound from
60% to 10%, and epsilon settles at 5.53 once the ceiling moves off 4.0 -- but
*not* for the epsilon floor: dropping it from
0.25 to 0.05 leaves the same five types pinned, now at 0.05. Those types want
their MM dispersion switched off, and a wider prior will not resolve them.
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
# "Near a bound" has to be measured as a fraction of the prior width, not as an
# absolute distance. An absolute 1e-4 is 0.1% of the +/-5% prior but only 0.01%
# of the +/-40% one, so it is strictest exactly where a lenient test would be
# least flattering -- it manufactures the conclusion it is used to support.
NEAR_FRAC = 0.01


def _rel_dist(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Distance to the nearest bound, as a fraction of the prior width."""
    return np.minimum(x - lo, hi - x) / (hi - lo)


def _near_bound(x: np.ndarray, lo: float, hi: float, frac: float = NEAR_FRAC) -> np.ndarray:
    return _rel_dist(x, lo, hi) < frac


def prior_sweep(d: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2))
    ax_s, ax_e, ax_f = axes
    fig.subplots_adjust(wspace=0.34)

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

    # (b) epsilon scales, log axis: the fitted values span two decades.
    for key, label, ci in PRIORS:
        v = d[key]
        e = np.asarray(v["eps"])
        elo, ehi = v["bounds"][2], v["bounds"][3]
        y = ci + rng.uniform(-0.16, 0.16, e.size)
        # eps is plotted on a log axis and its prior spans 1-3 decades, so the
        # width is measured in logs too -- a linear fraction of [0.05, 20] would
        # call eps=1 "10% from the floor", which is nonsense on a log scale.
        near = _near_bound(np.log(e), np.log(elo), np.log(ehi))
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
    ax_e.set_title("(b) ε — ceiling releases, floor does not", loc="left",
                   fontweight="bold")
    ax_e.set_ylim(-0.55, 2.9)
    ax_e.grid(axis="x", alpha=0.25)
    ax_e.set_axisbelow(True)
    e_wide = np.asarray(d["wide"]["eps"])
    # Two findings, and they point opposite ways. Reporting only the first would
    # be the flattering half of the result.
    ax_e.annotate(f"ceiling 4 to 20:\nε settles at {e_wide.max():.2f}, not 20",
                  (e_wide.max(), 2.15), textcoords="offset points", xytext=(4, 30),
                  ha="left", fontsize=8.5, color="0.20",
                  arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2))
    n_floor = int((np.abs(e_wide - 0.05) <= 1e-4).sum())
    ax_e.annotate(f"floor 0.25 to 0.05:\n{n_floor} types just follow it down",
                  (0.05, 2.15), textcoords="offset points", xytext=(6, 30),
                  ha="left", fontsize=8.5, color="0.20",
                  arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2))

    # (c) The headline, stated without a magic tolerance: the full curve of
    # "how many types are within t of a bound" against t. A single cut would be
    # a choice; the curve is the data. The +/-40% trace is flat, which is the
    # real result -- its pinned types are hard against the bound and everything
    # else is far from it, so the count does not depend on where the cut falls.
    thresh = np.logspace(-3.2, -0.7, 120)  # 0.06% .. 20% of the prior width
    for key, label, ci in PRIORS:
        v = d[key]
        s = np.asarray(v["sigma"])
        rel = _rel_dist(s, v["bounds"][0], v["bounds"][1])
        frac = [100.0 * (rel < t).mean() for t in thresh]
        ax_f.plot(100 * thresh, frac, lw=2.2, color=C[ci], label=label)
        ax_f.annotate(f"{100 * (rel < NEAR_FRAC).mean():.0f}%",
                      (100 * NEAR_FRAC, 100 * (rel < NEAR_FRAC).mean()),
                      textcoords="offset points", xytext=(7, 5),
                      fontsize=9, color=C[ci], fontweight="bold")
    ax_f.axvline(100 * NEAR_FRAC, color="0.75", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax_f.set_xscale("log")
    ax_f.set_xlabel("distance to a bound (% of prior width)")
    ax_f.set_ylabel("σ scales within that distance (%)")
    ax_f.set_title("(c) A wider σ prior releases the fit", loc="left", fontweight="bold")
    ax_f.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax_f.set_ylim(0, 100)
    ax_f.grid(alpha=0.25)
    ax_f.set_axisbelow(True)

    fig.suptitle(
        "DES LJ scales — response to the prior  (warm-started PhysNet EF, 25 epochs)",
        fontsize=13, fontweight="bold",
    )
    # The ±20% run trained 88 types against 94 for the other two, so the three
    # priors do not share a type set. The comparison below is distributional,
    # not matched per type -- stated here rather than left for a reader to hit.
    fig.text(
        0.5, 0.006,
        "(a, b) grey bars mark the prior; open markers sit within 1% of its width "
        "of a bound (in logs for ε). "
        "±5% and ±40% fit 94 types, ±20% fits 88 — the trainable masks differ, "
        "so this is a distributional comparison, not a matched per-type one.",
        ha="center", fontsize=8.5, color="0.40",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93), w_pad=3.0)
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

    # The stored vectors carry no type names, so a per-type comparison rests on
    # the three runs indexing the LJ table identically. That is testable rather
    # than assumable: if the order were scrambled, the per-type spread would
    # look like three random draws from the population. It does not -- see the
    # scramble null in panel (b) -- so the alignment is established, not assumed.
    per_type = S.max(0) - S.min(0)
    rng = np.random.default_rng(0)
    null = np.array([
        np.median(np.ptp(np.array([rng.permutation(s) for s in S]), axis=0))
        for _ in range(2000)
    ])

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

    # Panel (b): is the fitted per-type pattern signal or optimiser noise?
    # Measured against the scramble null, which is what "noise" would look like.
    ax2.hist(per_type, bins=24, color=C[0], alpha=0.85,
             label="observed, per type")
    ax2.axvline(np.median(per_type), color=C[0], lw=2.0,
                label=f"observed median {np.median(per_type):.3f}")
    ax2.axvline(null.mean(), color=C[1], lw=2.0, ls=(0, (5, 3)),
                label=f"index-scramble null {null.mean():.3f}")
    ax2.set_xlabel("seed-to-seed spread in σ  (max − min)")
    ax2.set_ylabel("CGenFF types")
    ax2.set_title("(b) Signal, not optimiser noise", loc="left", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_axisbelow(True)
    spread_e = np.abs(E.max(0) - E.min(0)) / np.maximum(E.mean(0), 1e-12)
    ax2.annotate(
        f"seed noise is {100 * np.median(per_type) / S.std(1).mean():.0f}% of the\n"
        f"between-type spread (sd {S.std(1).mean():.3f})\n"
        f"median ε spread {100 * np.median(spread_e):.0f}% of mean",
        (0.97, 0.60), xycoords="axes fraction", ha="right", va="top",
        fontsize=8.5, color="0.25",
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
