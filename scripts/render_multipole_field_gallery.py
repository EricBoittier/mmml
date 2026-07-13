#!/usr/bin/env python3
"""Gallery renders for the multipole / electrostatic-field / MBD-dispersion
plotters in :mod:`mmml.utils.plotting.multipoles`.

Same house style as ``render_ramachandran_gallery.py`` (``icml`` +
``default_cmap``): a physical quantity is wrapped onto a parametric surface
(the torus idea, here a sphere around each source) or drawn as its 2D field.

Colormap choice follows the data, per ``docs/plotting-style-guide.md``:
- multipole potential is signed (dipole has +/- lobes) -> diverging.
- polarizability / C6 are strictly positive -> sequential.

The electrostatic panels use analytic point multipoles (a monopole, a dipole,
a quadrupole) so the textbook lobe structure is unambiguous. The MBD panels use
the **real trained** checkpoint committed at the repo root
(``mbd_20260711-100037_epoch-0100.json``) on a small water cluster; if that file
or its optional deps (jax/e3x/ase) are unavailable, those two panels are skipped
with a note rather than failing the whole gallery.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmml.utils.plotting.multipoles import (
    plot_dispersion_field_slice,
    plot_field_slice,
    plot_mbd_surfaces,
    plot_multipole_surfaces,
)
from mmml.utils.plotting.styles import apply_plot_style

STYLE_NAME = "icml"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "plot-style-gallery-assets"
MBD_CHECKPOINT = ROOT / "mbd_20260711-100037_epoch-0100.json"
MULTIPOLE_CHECKPOINT = ROOT / "multipoles_20260711-100037_epoch-0100.json"

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

# Diverging palettes worth comparing for signed multipole potential. The house
# default (contrib:pampa) is muted; crameri:vik is the classic red/blue read.
DIVERGING_PALETTES = ("contrib:pampa", "crameri:vik", "cmocean:balance")

# A water trimer reused across panels so electrostatic + dispersion views line up.
_WATER_TRIMER = np.array([
    [0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0],
    [3.0, 0.2, 0.0], [3.76, 0.79, 0.0], [2.24, 0.79, 0.0],
    [1.4, 2.6, 0.3], [2.16, 3.19, 0.3], [0.64, 3.19, 0.3],
])
_WATER_TRIMER_Z = np.array([8, 1, 1, 8, 1, 1, 8, 1, 1])
_WATER_FRAGMENTS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def _analytic_multipole_scene():
    """A 3-source scene: monopole, dipole, quadrupole (moments in a.u.)."""
    origins_bohr = np.array([[-1.3, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 1.6, 0.0]]) * ANGSTROM_TO_BOHR
    charges = np.array([1.0, -0.3, 0.0])
    dipoles_bohr = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.0, 0.0]]) * ANGSTROM_TO_BOHR
    quadrupoles_bohr = np.zeros((3, 3, 3))
    quadrupoles_bohr[2] = np.array([[2.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]) * ANGSTROM_TO_BOHR**2
    return origins_bohr, charges, dipoles_bohr, quadrupoles_bohr, [8, 7, 6]


def multipole_surfaces(out: Path) -> None:
    o, q, d, quad, z = _analytic_multipole_scene()
    plot_multipole_surfaces(o, q, d, quad, atomic_numbers=z, out=out, style=STYLE_NAME)


def field_slice(out: Path) -> None:
    o, q, d, quad, _ = _analytic_multipole_scene()
    plot_field_slice(o, q, d, quad, plane="xy", out=out, style=STYLE_NAME)


def multipole_colormap_variants(out: Path) -> None:
    """The same analytic scene under three diverging palettes, small-multiples.

    Colour choice is a real decision for signed data (see the multipole-triangle
    colormap discussion): this shows the muted house default beside the classic
    red/blue reads so the trade-off is visible, not asserted.
    """
    import matplotlib.pyplot as plt

    o, q, d, quad, z = _analytic_multipole_scene()
    fig = plt.figure(figsize=(15, 5.2))
    for i, name in enumerate(DIVERGING_PALETTES, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        plot_multipole_surfaces(o, q, d, quad, atomic_numbers=z, cmap=name,
                                title=name, style=STYLE_NAME, ax=ax)
    fig.suptitle("Signed multipole potential: diverging palette comparison")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _multipole_prediction():
    """Predict per-molecule multipoles for a water trimer with the real trained
    checkpoint. Returns the arrays plot_multipole_surfaces/plot_field_slice
    expect, or None if the checkpoint or its deps are unavailable."""
    if not MULTIPOLE_CHECKPOINT.exists():
        return None
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        from ase import Atoms

        from mmml.models.multipoles.electrostatics import (
            LearnedMolecularMultipoleElectrostatics,
        )
    except Exception:
        return None
    atoms = Atoms(numbers=_WATER_TRIMER_Z, positions=_WATER_TRIMER)
    n_frag = len(_WATER_FRAGMENTS)
    calc = LearnedMolecularMultipoleElectrostatics(
        MULTIPOLE_CHECKPOINT, fragments=_WATER_FRAGMENTS,
        charges=[0.0] * n_frag, multiplicities=[1.0] * n_frag,
    )
    pred = calc.predict_fragment_multipoles(atoms)
    return (
        pred["origins_bohr"], pred["charges"], pred["dipoles_bohr"],
        pred["quadrupoles_bohr"], pred["octupoles_bohr"],
    )


def multipole_surfaces_learned(out: Path) -> bool:
    """One sphere per water, from the committed multipole weights. The learned
    model emits one molecular multipole per fragment, so each sphere is a whole
    molecule's charge+dipole+quadrupole+octupole, not a single atom."""
    pred = _multipole_prediction()
    if pred is None:
        return False
    origins, charges, dipoles, quads, octs = pred
    plot_multipole_surfaces(
        origins, charges, dipoles, quads, octs,
        probe_radius_angstrom=1.6, radius_gain=0.8, cmap="crameri:vik",
        title="Learned molecular multipoles per water (committed weights)",
        out=out, style=STYLE_NAME,
    )
    return True


def field_slice_learned(out: Path) -> bool:
    """The electrostatic field of the learned water-trimer multipoles."""
    pred = _multipole_prediction()
    if pred is None:
        return False
    origins, charges, dipoles, quads, octs = pred
    plot_field_slice(
        origins, charges, dipoles, quads, octs, plane="xy", span_angstrom=7.0,
        title="Electrostatic field of learned water-trimer multipoles",
        out=out, style=STYLE_NAME,
    )
    return True


def _mbd_prediction():
    """Predict per-atom polarizability / C6 for a water cluster with the real
    trained MBD checkpoint. Returns (positions, z, alpha, c6) or None."""
    if not MBD_CHECKPOINT.exists():
        return None
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        from ase import Atoms

        from mmml.models.mbd.calculator import load_mbd_model, predict_mbd_from_atoms
    except Exception:
        return None
    positions = np.array([
        [0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0],
        [3.0, 0.2, 0.0], [3.76, 0.79, 0.0], [2.24, 0.79, 0.0],
        [1.4, 2.6, 0.3], [2.16, 3.19, 0.3], [0.64, 3.19, 0.3],
    ])
    z = np.array([8, 1, 1, 8, 1, 1, 8, 1, 1])
    model, params = load_mbd_model(MBD_CHECKPOINT)
    pred = predict_mbd_from_atoms(model, params, Atoms(numbers=z, positions=positions))
    return positions, z, pred["polarizabilities_bohr3"], pred["c6_native"]


def main() -> None:
    import warnings

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style(STYLE_NAME)

    multipole_surfaces(OUT_DIR / "chart_multipole_surfaces.png")
    print(f"wrote {OUT_DIR / 'chart_multipole_surfaces.png'}")
    field_slice(OUT_DIR / "chart_multipole_field.png")
    print(f"wrote {OUT_DIR / 'chart_multipole_field.png'}")
    multipole_colormap_variants(OUT_DIR / "chart_multipole_colormap_variants.png")
    print(f"wrote {OUT_DIR / 'chart_multipole_colormap_variants.png'}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learned_ok = multipole_surfaces_learned(OUT_DIR / "chart_multipole_surfaces_learned.png")
        if learned_ok:
            field_slice_learned(OUT_DIR / "chart_multipole_field_learned.png")
    if learned_ok:
        print(f"wrote {OUT_DIR / 'chart_multipole_surfaces_learned.png'}")
        print(f"wrote {OUT_DIR / 'chart_multipole_field_learned.png'}")
    else:
        print(f"SKIP learned multipole panels: {MULTIPOLE_CHECKPOINT.name} or deps unavailable")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mbd = _mbd_prediction()
    if mbd is None:
        print(f"SKIP MBD panels: checkpoint {MBD_CHECKPOINT.name} or jax/e3x/ase unavailable")
        return
    positions, z, alpha, c6 = mbd
    plot_mbd_surfaces(positions, alpha, c6, atomic_numbers=z,
                      out=OUT_DIR / "chart_mbd_polarizability_surfaces.png", style=STYLE_NAME)
    print(f"wrote {OUT_DIR / 'chart_mbd_polarizability_surfaces.png'}")
    plot_dispersion_field_slice(positions, c6, plane="xy",
                                out=OUT_DIR / "chart_mbd_dispersion_field.png", style=STYLE_NAME)
    print(f"wrote {OUT_DIR / 'chart_mbd_dispersion_field.png'}")


if __name__ == "__main__":
    main()
