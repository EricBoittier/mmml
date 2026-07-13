#!/usr/bin/env python3
"""Real learned multipole / field / MBD plots on optimized small molecules.

Unlike ``render_multipole_field_gallery.py`` (analytic moments for teaching the
method), this drives the *actual* trained models end to end:

1. Build small molecules with ASE, then **geometry-optimize** each on the
   trained SpookyNet potential (``examples/spooky_so3lr_muon_mbd_zbl-epoch-0002.json``).
2. Predict the learned molecular multipoles on the optimized geometry with the
   committed ``multipoles_20260711-100037_epoch-0100.json`` weights.
3. Predict the learned MBD per-atom polarizability / C6 with the committed
   ``mbd_20260711-100037_epoch-0100.json`` weights.
4. Render one row of electrostatic-field slices, one of multipole surfaces, and
   one of MBD polarizability spheres.

This needs the SpookyNet/multipole/MBD checkpoints and jax/e3x/ase, and runs a
short CPU optimization per molecule (a few minutes total). If anything required
is missing it prints a note and exits 0 rather than failing a docs build. The
analytic gallery covers the fast, always-available path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "plot-style-gallery-assets"
SPOOKY = ROOT / "examples/spooky_so3lr_muon_mbd_zbl-epoch-0002.json"
MBD = ROOT / "mbd_20260711-100037_epoch-0100.json"
MULTI = ROOT / "multipoles_20260711-100037_epoch-0100.json"

MOLECULES = ["H2O", "NH3", "CH3OH", "H2CO"]
PRETTY = {"H2O": "water", "NH3": "ammonia", "CH3OH": "methanol", "H2CO": "formaldehyde"}
AU_DIPOLE_TO_DEBYE = 2.541746


def _optimize_and_predict():
    """Optimize each molecule on SpookyNet, then predict multipoles + MBD.

    Returns a list of per-molecule dicts, or None if deps/checkpoints missing.
    """
    if not (SPOOKY.exists() and MBD.exists() and MULTI.exists()):
        return None
    try:
        import warnings

        import jax

        jax.config.update("jax_enable_x64", True)
        from ase.build import molecule
        from ase.optimize import LBFGS

        from mmml.models.mbd.calculator import load_mbd_model, predict_mbd_from_atoms
        from mmml.models.multipoles.electrostatics import (
            LearnedMolecularMultipoleElectrostatics,
        )
        from mmml.models.spookynet_calc import SpookyNetCalculator
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"SKIP real-molecule gallery: {exc}")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Spooky-only optimization: dispersion barely shifts small-molecule
        # equilibrium geometry and this halves the per-step JIT/eval cost.
        calc = SpookyNetCalculator(SPOOKY, mbd_checkpoint=False)
        mbd_model, mbd_params = load_mbd_model(MBD)

        records = []
        for name in MOLECULES:
            atoms = molecule(name)
            atoms.calc = calc
            print(f"[{name}] optimizing ({len(atoms)} atoms)...", flush=True)
            LBFGS(atoms, logfile=None).run(fmax=0.05, steps=200)
            fmax = float(np.abs(atoms.get_forces()).max())

            mp_calc = LearnedMolecularMultipoleElectrostatics(
                MULTI, fragments=[list(range(len(atoms)))],
                charges=[0.0], multiplicities=[1.0],
            )
            pred = mp_calc.predict_fragment_multipoles(atoms)
            mbd = predict_mbd_from_atoms(mbd_model, mbd_params, atoms)
            mu_debye = float(np.linalg.norm(pred["dipoles_bohr"][0]) * AU_DIPOLE_TO_DEBYE)
            print(f"[{name}] fmax={fmax:.4f}  q={pred['charges'][0]:+.3f}  "
                  f"|mu|={mu_debye:.2f} D", flush=True)
            records.append({
                "name": name,
                "z": atoms.get_atomic_numbers(),
                "pos": atoms.get_positions(),
                "origins_bohr": pred["origins_bohr"],
                "charges": pred["charges"],
                "dipoles_bohr": pred["dipoles_bohr"],
                "quadrupoles_bohr": pred["quadrupoles_bohr"],
                "octupoles_bohr": pred["octupoles_bohr"],
                "alpha": mbd["polarizabilities_bohr3"],
                "c6": mbd["c6_native"],
                "mu_debye": mu_debye,
            })
    return records


def _render(records):
    import matplotlib.pyplot as plt

    from mmml.utils.plotting.multipoles import (
        plot_field_slice,
        plot_mbd_surfaces,
        plot_multipole_surfaces,
    )
    from mmml.utils.plotting.styles import apply_plot_style

    apply_plot_style("icml")
    n = len(records)

    # 1. Electrostatic field slices (each molecule's learned molecular multipole).
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.0))
    for ax, r in zip(np.atleast_1d(axes), records):
        plot_field_slice(
            r["origins_bohr"], r["charges"], r["dipoles_bohr"],
            r["quadrupoles_bohr"], r["octupoles_bohr"], plane="xy",
            span_angstrom=5.0, ax=ax, colorbar=True,
            title=f"{PRETTY[r['name']]}  (|μ|={r['mu_debye']:.2f} D)",
        )
    fig.suptitle("Learned electrostatic field of SpookyNet-optimized molecules "
                 "(committed multipole weights)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart_real_multipole_fields.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote chart_real_multipole_fields.png")

    # 2. Multipole surfaces (small probe radius exposes the angular structure).
    fig = plt.figure(figsize=(5.0 * n, 5.0))
    for i, r in enumerate(records, start=1):
        ax = fig.add_subplot(1, n, i, projection="3d")
        plot_multipole_surfaces(
            r["origins_bohr"], r["charges"], r["dipoles_bohr"],
            r["quadrupoles_bohr"], r["octupoles_bohr"],
            probe_radius_angstrom=1.4, radius_gain=0.8, title=PRETTY[r["name"]], ax=ax,
        )
    fig.suptitle("Learned molecular multipole surfaces (optimized geometries)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart_real_multipole_surfaces.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote chart_real_multipole_surfaces.png")

    # 3. MBD per-atom polarizability spheres on the optimized geometries.
    fig = plt.figure(figsize=(5.0 * n, 5.0))
    for i, r in enumerate(records, start=1):
        ax = fig.add_subplot(1, n, i, projection="3d")
        plot_mbd_surfaces(r["pos"], r["alpha"], r["c6"],
                          atomic_numbers=r["z"], title=PRETTY[r["name"]], ax=ax)
    fig.suptitle("Learned MBD response on optimized molecules "
                 "(sphere radius ~ polarizability, colour = C₆)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart_real_mbd_surfaces.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote chart_real_mbd_surfaces.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = _optimize_and_predict()
    if records is None:
        print("SKIP real-molecule gallery: checkpoints or jax/e3x/ase unavailable")
        return
    _render(records)


if __name__ == "__main__":
    main()
