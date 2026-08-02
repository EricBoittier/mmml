#!/usr/bin/env python3
"""Does a real bonded model own the intramolecular coordinate the ML gets wrong?

The hybrid computes the dimer term as E_AB - (E_A + E_B), all three from the same
ML model (mmml_calculator.calculate_dimer_contributions). Along an O-H scan the
monomer sum rises, as it must, but the dimer total *falls* -- so the interaction
comes out with a -73 kcal/mol well at O-H = 0.77 A that is pure cancellation
error between two badly extrapolated totals.

If instead a bonded model owns the intramolecular energy and the ML supplies only
the interaction, the ML never has to represent the steep intramolecular scale.
This script measures the premise of that architecture: how the CGenFF bonded
energy behaves along the same scan, next to what the ML produced.

Also reports the ML/MM switching value at the scan geometry, which the arm
comparison assumes is 1.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KCAL = 23.060548  # kcal/mol per eV


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan-npz", type=Path, required=True)
    p.add_argument("--psf", type=Path, required=True, help="PSF supplying CGenFF bonded terms")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ml-switch-width", type=float, default=1.5)
    p.add_argument("--mm-switch-on", type=float, default=6.0)
    args = p.parse_args()

    from mmml.interfaces.pycharmmInterface import import_pycharmm as ipy
    from mmml.interfaces.pycharmmInterface.mlpot.jax_mm_spoof import (
        resolve_monomer_bonded_evaluator,
    )

    if not ipy.ensure_pycharmm_loaded():
        raise RuntimeError("PyCHARMM is unavailable")

    data = np.load(args.scan_npz, allow_pickle=True)
    R = np.asarray(data["R"], dtype=np.float64)
    meta = json.loads(str(data["metadata"]))
    oh = np.asarray(meta["oh_lengths_A"], dtype=np.float64)

    # Monomer A is atoms 0..2; the scan moves atom 1 only.
    bonded = resolve_monomer_bonded_evaluator(
        atoms_per_monomer=3,
        monomer_psf=args.psf.resolve(),
        atom_offset=0,
        energy_unit="eV",
    )

    # BondedEvalFn returns (energy, forces).
    e_bonded = np.array([float(bonded(np.asarray(frame[:3]))[0]) for frame in R]) * KCAL

    # Switching at the scan geometry. The dimer term is scaled by this; the arm
    # differencing assumes it is 1.0 at the fixed O-O of the scan.
    from mmml.interfaces.pycharmmInterface.calculator_utils import ml_switch_simple

    com_a = R[:, :3].mean(axis=1)
    com_b = R[:, 3:].mean(axis=1)
    com_d = np.linalg.norm(com_a - com_b, axis=1)
    s = np.asarray(
        ml_switch_simple(com_d, args.ml_switch_width, args.mm_switch_on), dtype=np.float64
    )

    i0 = int(np.argmin(np.abs(oh - 0.9840)))
    rel = e_bonded - e_bonded[i0]
    imin = int(np.argmin(e_bonded))

    print(f"switching s(R) over the scan: {s.min():.6f} .. {s.max():.6f}")
    print(f"  COM separation {com_d.min():.3f} .. {com_d.max():.3f} A")
    print(f"  -> arm differencing assumes s = 1.0: {'OK' if np.allclose(s, 1.0) else 'NOT SATISFIED'}")

    print(f"\nCGenFF bonded energy of monomer A, relative to O-H = 0.9840 A (kcal/mol)")
    print(f"{'O-H(A)':>8} {'E_bonded':>10}")
    for i in range(0, len(oh), 4):
        print(f"{oh[i]:>8.3f} {rel[i]:>10.2f}")
    print(f"\nbonded minimum at O-H = {oh[imin]:.4f} A   (CHARMM TIP3 b0, physical 0.958)")
    print(f"monotone rising below the minimum: {bool(np.all(np.diff(rel[: imin + 1]) <= 1e-9))}")
    print(f"E_bonded at 0.771 A = {rel[int(np.argmin(np.abs(oh - 0.771)))]:+.2f} kcal/mol")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        oh_A=oh,
        e_bonded_kcal=e_bonded,
        e_bonded_rel_kcal=rel,
        switch=s,
        com_dist_A=com_d,
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
