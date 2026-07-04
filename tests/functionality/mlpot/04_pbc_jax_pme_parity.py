#!/usr/bin/env python3
"""PBC jax_mic parity smoke: LR defaults, image dimer distance, MM eterm decomposition.

Requires PyCHARMM for full CHARMM term comparison; core checks run without CHARMM.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    decompose_mlpot_mm_nb_eterms_kcalmol,
)
from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    resolve_jax_pme_sr_cutoff_for_mlpot,
    resolve_lr_solver_for_mlpot,
)
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
    image_aware_dimer_com_distance_numpy,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box-size", type=float, default=24.0)
    p.add_argument("--rtol", type=float, default=0.05)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ns = argparse.Namespace(
        setup="pbc_nve",
        lr_solver=None,
        free_space=False,
        mlpot_pbc=True,
        mm_nonbond_mode="jax_mic",
    )
    lr = resolve_lr_solver_for_mlpot(ns, mlpot_pbc=True, mm_nonbond_mode="jax_mic")
    if lr != "jax_pme":
        print(f"FAIL: expected default lr_solver=jax_pme for PBC, got {lr!r}", file=sys.stderr)
        return 1
    cp = CutoffParameters()
    sr = resolve_jax_pme_sr_cutoff_for_mlpot(ns, cp)
    if abs(sr - (cp.mm_switch_on + cp.mm_switch_width)) > 1e-6:
        print(
            f"FAIL: jax_pme sr cutoff {sr} != switched MM outer {cp.mm_switch_on + cp.mm_switch_width}",
            file=sys.stderr,
        )
        return 1

    side = float(args.box_size)
    cell = np.diag([side, side, side])
    pos = np.zeros((6, 3), dtype=np.float64)
    pos[:3, 0] = side * 0.5 - 1.0
    pos[3:, 0] = -side * 0.5 + 1.0
    d = image_aware_dimer_com_distance_numpy(pos, np.arange(6), 3, 3, cell)
    if d > cp.mm_switch_on:
        print(f"FAIL: cross-face dimer COM distance {d:.3f} Å exceeds mm_switch_on", file=sys.stderr)
        return 1

    pair_idx = np.array([[0, 3]], dtype=np.int32)
    pair_mask = np.array([True], dtype=bool)
    comp = decompose_mlpot_mm_nb_eterms_kcalmol(
        pos,
        pair_idx,
        pair_mask,
        cell,
        charges_e=np.array([0.4, -0.4, 0.4, -0.4, 0.4, -0.4]),
        rmins_A=np.full(6, 1.8),
        epsilons_kcal=np.full(6, 0.05),
        monomer_id=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        mm_switch_on=cp.mm_switch_on,
        mm_switch_width=cp.mm_switch_width,
    )
    if not np.isfinite(comp["mm_total"]):
        print("FAIL: MM decomposition returned non-finite energy", file=sys.stderr)
        return 1

    print(
        f"OK: PBC parity smoke (lr={lr}, sr_cutoff={sr:.1f} Å, "
        f"cross-face d={d:.3f} Å, mm_total={comp['mm_total']:.4f} kcal/mol)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
