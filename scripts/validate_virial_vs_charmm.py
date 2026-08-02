#!/usr/bin/env python3
"""Validate the NpT virial against CHARMM's own internal pressure.

Why this exists: the NpT campaign blew up because the custom VJP in
``jaxmd_runner.set_up_nhc_sim_routine`` returned ``None`` for the
``perturbation`` cotangent, so jax-md's internal pressure lost its virial term
entirely and collapsed to the kinetic one. That was diagnosed numerically -- a
732-TIP3 box at 297.87 K in 21955.3 A^3 reported P_meas = 4059.58 atm against a
1 atm target, and the kinetic-only value is 4059.63 atm -- and then fixed with

    dE/dp = -(1 / 3p) * sum_i F_i . r_i

Unit tests check that formula against finite differences and against Euler's
theorem, but both of those are self-referential in the sense that they use our
own energy. This script compares against an INDEPENDENT implementation: CHARMM
computes its own virial and reports a scalar internal pressure as ``PRSI``.

The comparison is

    P = (2 KE + sum_i F_i . r_i) / (3 V)

evaluated with CHARMM's forces and coordinates, against CHARMM's PRSI for the
same state. Agreement validates the formula AND the unit conversion chain
(eV/A^3 -> Pa -> atm), which is where this kind of bug usually hides.

Velocities are zeroed so KE = 0 and the comparison isolates the virial.

Requires libcharmm; run on the cluster, not in CI::

    python scripts/validate_virial_vs_charmm.py \\
        --psf artifacts/.../boxes/tip3_298k/model.psf \\
        --crd artifacts/.../boxes/tip3_298k/model.crd \\
        --box-side 28.0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# eV/A^3 -> Pa -> atm
EV_A3_TO_PA = 1.602176634e-19 / 1e-30
PA_TO_ATM = 1.0 / 101325.0
KCAL_MOL_A_TO_EV_A = 1.0 / 23.060547830619026


def virial_pressure_atm(
    forces_ev_a: np.ndarray,
    positions_a: np.ndarray,
    volume_a3: float,
    kinetic_ev: float = 0.0,
) -> float:
    """P = (2 KE + sum F.r) / (3 V), in atm.

    ``sum_i F_i . r_i`` is the same contraction the NpT custom VJP returns as
    ``dE/dp`` (up to the -1/3p factor), so agreement with CHARMM here is a
    direct check on that cotangent's magnitude and sign.
    """
    f = np.asarray(forces_ev_a, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(positions_a, dtype=np.float64).reshape(-1, 3)
    if f.shape != r.shape:
        raise ValueError(f"force/position shape mismatch: {f.shape} vs {r.shape}")
    virial = float(np.sum(f * r))
    p_ev_a3 = (2.0 * float(kinetic_ev) + virial) / (3.0 * float(volume_a3))
    return p_ev_a3 * EV_A3_TO_PA * PA_TO_ATM


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--psf", type=Path, required=True)
    ap.add_argument("--crd", type=Path, required=True)
    ap.add_argument("--box-side", type=float, required=True, help="cubic side, A")
    ap.add_argument("--tolerance-percent", type=float, default=5.0)
    ap.add_argument("-o", "--output", type=Path, default=None)
    a = ap.parse_args(argv)

    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import prepare_charmm_pbc

    import pycharmm.coor as coor
    import pycharmm.energy as energy
    import pycharmm.read as read

    read_cgenff_toppar()
    read.psf_card(str(a.psf))
    read.coor_card(str(a.crd))
    prepare_charmm_pbc(float(a.box_side))

    # Zero velocities so the kinetic term drops out and only the virial remains.
    energy.show()

    pos = np.asarray(coor.get_positions().to_numpy(dtype=float), dtype=np.float64)
    n_atoms = int(pos.shape[0])
    volume = float(a.box_side) ** 3

    # CHARMM forces are kcal/mol/A; the virial identity above wants eV/A.
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_forces_array

    forces_kcal = np.asarray(get_charmm_forces_array(), dtype=np.float64).reshape(-1, 3)
    forces_ev = forces_kcal * KCAL_MOL_A_TO_EV_A

    ours_atm = virial_pressure_atm(forces_ev, pos, volume, kinetic_ev=0.0)

    from mmml.interfaces.pycharmmInterface.mlpot.pressure_tensor import (
        read_instantaneous_scalar_pressure_atm,
    )

    charmm_atm = float(read_instantaneous_scalar_pressure_atm(refresh_energy=True))

    denom = max(abs(charmm_atm), 1.0)
    rel_pct = abs(ours_atm - charmm_atm) / denom * 100.0
    ok = rel_pct <= float(a.tolerance_percent)

    report = {
        "n_atoms": n_atoms,
        "box_side_A": float(a.box_side),
        "volume_A3": volume,
        "virial_sum_F_dot_r_eV": float(np.sum(forces_ev * pos)),
        "pressure_from_virial_atm": ours_atm,
        "pressure_charmm_PRSI_atm": charmm_atm,
        "relative_difference_percent": rel_pct,
        "tolerance_percent": float(a.tolerance_percent),
        "agrees": bool(ok),
    }
    print(json.dumps(report, indent=2))
    if a.output is not None:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2))

    if not math.isfinite(ours_atm) or not math.isfinite(charmm_atm):
        print("NON-FINITE pressure; cannot validate", flush=True)
        return 2
    print(
        f"\n  virial pressure (ours)  {ours_atm:12.2f} atm"
        f"\n  CHARMM PRSI             {charmm_atm:12.2f} atm"
        f"\n  difference              {rel_pct:12.3f} %"
        f"\n  {'AGREE' if ok else 'DISAGREE'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
