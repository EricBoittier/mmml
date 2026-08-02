#!/usr/bin/env python3
"""Plot the 2D water-dimer surfaces: hybrid, its decomposition, and a reference.

The gate on further condensed-phase work. Summary statistics said the hybrid's
dimer interaction spans 1.35 kcal/mol against 28.81 for classical TIP3-TIP3 on
the identical frames, but a span does not say *where* -- and the location of a
spurious well in (R, theta) is what would tie it to the bulk-water behaviour.

Colour is diverging because interaction energy has a real zero (the separated
dimer): blue attractive, orange repulsive, neutral grey at 0. Built from the
house Okabe-Ito slots rather than a rainbow, and blue/orange is the safest
diverging pair under colour-vision deficiency.

Panels (a) and (b) deliberately share one colour scale -- that is the finding,
and putting the hybrid on its own scale first would flatter it. (c) then rescales
the hybrid so its actual structure is visible, and (d) isolates the ML dimer
term, which is what produces essentially all of it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from mmml.utils.plotting.styles import apply_plot_style

REPO = Path(__file__).resolve().parents[1]
GRID = REPO / "artifacts" / "dimer_2d" / "water_dimer_2d.npz"
ARMS = REPO / "artifacts" / "dimer_2d"
OUT = REPO / "docs" / "images" / "des-so3lr-dimers" / "dimer_2d_surfaces.png"

EV2KCAL = 23.060547830619026
COUL = 332.0716  # kcal/mol A / e^2
# CHARMM TIP3
Q = np.array([-0.834, 0.417, 0.417] * 2)
EPS = np.array([0.1521, 0.0, 0.0, 0.1521, 0.0, 0.0])
RMIN2 = np.array([1.7682, 0.0, 0.0, 1.7682, 0.0, 0.0])

# Diverging: Okabe-Ito blue -> neutral grey -> Okabe-Ito vermillion.
DIVERGING = LinearSegmentedColormap.from_list(
    "okabe_div", ["#0072B2", "#7FB4D3", "#E8E8E6", "#EBA07A", "#D55E00"]
)


def classical_tip3_interaction(frames: np.ndarray) -> np.ndarray:
    """Independent reference: CHARMM TIP3 Coulomb + Lennard-Jones, kcal/mol."""
    out = np.empty(frames.shape[0])
    for k, fr in enumerate(frames):
        d = np.linalg.norm(fr[:3, None, :] - fr[None, 3:, :], axis=-1)
        e = 0.0
        for a in range(3):
            for b in range(3):
                r = d[a, b]
                e += COUL * Q[a] * Q[3 + b] / r
                if EPS[a] > 0 and EPS[3 + b] > 0:
                    epsij = np.sqrt(EPS[a] * EPS[3 + b])
                    rm = RMIN2[a] + RMIN2[3 + b]
                    x = (rm / r) ** 6
                    e += epsij * (x * x - 2.0 * x)
        out[k] = e
    return out


def load_arm(name: str, nr: int, nt: int) -> np.ndarray | None:
    p = ARMS / name / "evaluate.npz"
    if not p.is_file():
        return None
    e = np.asarray(np.load(p)["E"]).reshape(-1) * EV2KCAL
    s = e.reshape(nr, nt)
    return s - s[-1].mean()  # interaction: reference the largest separation


def draw(ax, surf, r, th, *, vmax, title, mask=None, cbar_label):
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.pcolormesh(th, r, surf, cmap=DIVERGING, norm=norm, shading="nearest")
    if mask is not None and mask.any():
        # Hatch the frames compressed past physical reach rather than deleting
        # them: a reader should see that the region was evaluated, not omitted.
        ax.contourf(th, r, mask.astype(float), levels=[0.5, 1.5],
                    colors="none", hatches=["///"])
        ax.contour(th, r, mask.astype(float), levels=[0.5], colors="0.35",
                   linewidths=1.0)
    ax.set_xlabel("θ  (deg)")
    ax.set_ylabel("R  (Å)")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    cb = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return im


def main() -> int:
    apply_plot_style("icml")
    g = np.load(GRID)
    r, th = g["grid_r"], g["grid_theta"]
    nr, nt = (int(x) for x in g["grid_shape"])
    contact = g["min_contact_A"].reshape(nr, nt)
    mask = contact < 1.4

    ref_flat = classical_tip3_interaction(np.asarray(g["R"], dtype=np.float64))
    ref = ref_flat.reshape(nr, nt)
    ref = ref - ref[-1].mean()

    full = load_arm("full", nr, nt)
    no_dimer = load_arm("no_dimer", nr, nt)
    if full is None:
        raise SystemExit("missing artifacts/dimer_2d/full/evaluate.npz")
    dimer_term = full - no_dimer if no_dimer is not None else None

    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.9))
    shared = float(np.percentile(np.abs(ref), 99))

    draw(axes[0], ref, r, th, vmax=shared,
         title="(a) classical TIP3–TIP3 (reference)", mask=mask,
         cbar_label="interaction (kcal/mol)")
    draw(axes[1], full, r, th, vmax=shared,
         title="(b) hybrid, SAME scale as (a)", mask=mask,
         cbar_label="interaction (kcal/mol)")
    own = max(float(np.percentile(np.abs(full), 99)), 1e-6)
    draw(axes[2], full, r, th, vmax=own,
         title="(c) hybrid, own scale", mask=mask,
         cbar_label="interaction (kcal/mol)")
    if dimer_term is not None:
        dv = max(float(np.percentile(np.abs(dimer_term), 99)), 1e-6)
        draw(axes[3], dimer_term, r, th, vmax=dv,
             title="(d) ML dimer term = full − no_dimer", mask=mask,
             cbar_label="contribution (kcal/mol)")
    else:
        axes[3].set_axis_off()

    axes[0].annotate(f"min {ref.min():.2f}\nspan {np.ptp(ref):.2f}",
                     (0.03, 0.94), xycoords="axes fraction", va="top",
                     fontsize=8.5, color="0.15", family="monospace",
                     bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8"))
    axes[1].annotate(f"min {full.min():.2f}\nspan {np.ptp(full):.2f}",
                     (0.03, 0.94), xycoords="axes fraction", va="top",
                     fontsize=8.5, color="0.15", family="monospace",
                     bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8"))

    fig.suptitle(
        "Water-dimer 2D surfaces — 25 R × 24 θ, rigid TIP3 monomers, "
        "interaction referenced to the largest separation",
        fontsize=13, fontweight="bold")
    fig.text(0.5, 0.015,
             "Hatched: frames with a closest intermolecular contact < 1.4 Å, compressed past physical reach. "
             "(a) is computed independently from CHARMM TIP3 charges and LJ on the identical frames. "
             "MM contributes ~0 below 6 Å by design (--mm-switch-on 6.0), so the hybrid surface here is the ML dimer term.",
             ha="center", fontsize=8.5, color="0.40")
    fig.tight_layout(rect=(0, 0.055, 1, 0.92), w_pad=2.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")
    print(f"  reference : min {ref.min():8.3f}  span {np.ptp(ref):8.3f} kcal/mol")
    print(f"  hybrid    : min {full.min():8.3f}  span {np.ptp(full):8.3f} kcal/mol")
    if dimer_term is not None:
        print(f"  ML dimer  : min {dimer_term.min():8.3f}  span {np.ptp(dimer_term):8.3f}")
    i, j = np.unravel_index(np.argmin(ref), ref.shape)
    print(f"  reference minimum at R={r[i]:.2f} A, theta={th[j]:.0f} deg")
    i, j = np.unravel_index(np.argmin(full), full.shape)
    print(f"  hybrid    minimum at R={r[i]:.2f} A, theta={th[j]:.0f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
