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
    cb = ax.figure.colorbar(im, ax=ax, pad=0.04, fraction=0.046)
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

    # Four verification frames: the reference minimum, the hybrid minimum, the
    # most repulsive reference cell, and a separated pair. POV-Ray renders of
    # these exact frames sit in the bottom row so the geometry behind any claim
    # can be checked rather than taken on trust.
    picks = []
    i, j = np.unravel_index(np.argmin(ref), ref.shape)
    picks.append(("reference minimum", i, j))
    i, j = np.unravel_index(np.argmin(full), full.shape)
    picks.append(("hybrid minimum", i, j))
    i, j = np.unravel_index(np.argmax(ref), ref.shape)
    picks.append(("most repulsive (ref)", i, j))
    picks.append(("separated", nr - 1, 0))

    fig = plt.figure(figsize=(19.0, 10.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.42)
    axes = [fig.add_subplot(gs[0, k]) for k in range(4)]
    rax = [fig.add_subplot(gs[1, k]) for k in range(4)]
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

    # Mark the verification frames on every surface, same symbols throughout.
    marks = ["o", "s", "^", "D"]
    glyphs = ["circle", "square", "triangle", "diamond"]
    for ax in axes:
        for (lab, i, j), mk in zip(picks, marks):
            ax.plot(th[j], r[i], mk, ms=8, mfc="none", mec="0.10", mew=1.8, zorder=6)

    # Bottom row: the rendered frames, labelled with values measured from the
    # coordinates themselves, not copied from the grid metadata.
    import matplotlib.image as mpimg
    import sys as _sys
    if str(Path(__file__).parent) not in _sys.path:
        _sys.path.insert(0, str(Path(__file__).parent))
    from render_dimer_2d_frames import frame_metrics

    frames_xyz = np.asarray(g["R"], dtype=np.float64)
    zz = np.asarray(g["Z"])
    for ax, (lab, i, j), mk, gl in zip(rax, picks, marks, glyphs):
        idx = i * nt + j
        png = ARMS / "renders" / f"frame_{idx:04d}.png"
        if png.is_file():
            img = mpimg.imread(png)
            h, w = img.shape[:2]
            ax.imshow(img[int(0.04 * h):int(0.96 * h), int(0.04 * w):int(0.96 * w)])
        else:
            ax.text(0.5, 0.5, f"missing\n{png.name}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="0.4")
        m = frame_metrics(frames_xyz[idx], zz[idx])
        ax.set_title(f"[{gl}]  {lab}   frame {idx}", loc="left",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel(
            f"R={m['R_A']:.2f} Å   θ={th[j]:.0f}°   closest contact={m['min_contact_A']:.2f} Å\n"
            f"E_ref={ref[i, j]:+.2f}   E_hybrid={full[i, j]:+.3f} kcal/mol",
            fontsize=8.6, color="0.25")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

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
    fig.text(0.5, 0.008,
             "Hatched: frames with a closest intermolecular contact < 1.4 Å, compressed past physical reach. "
             "(a) is computed independently from CHARMM TIP3 charges and LJ on the identical frames. "
             "MM contributes ~0 below 6 Å by design (--mm-switch-on 6.0), so the hybrid surface here is the ML dimer term.",
             ha="center", fontsize=8.5, color="0.40")

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
