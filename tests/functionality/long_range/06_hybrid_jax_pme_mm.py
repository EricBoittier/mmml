#!/usr/bin/env python3
"""Step 6: hybrid MM (switched LJ + jax-pme Coulomb) via build_mm_energy_forces_fn."""

from __future__ import annotations

import os
import sys

import numpy as np

from _common import have_jax_pme_package, print_fail, print_header, print_pass


def main() -> int:
    print_header("Hybrid MM with jax-pme Coulomb (LJ pairs + Ewald/PME/P3M elec)")
    if not have_jax_pme_package():
        print("SKIP: jax-pme not installed")
        return 0

    try:
        from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM
    except Exception:
        CGENFF_PRM = None
    if CGENFF_PRM is None:
        print("SKIP: PyCHARMM/CGENFF not available")
        return 0

    from tests.functionality.neighbor_lists._common import setup_charmm_composition_cluster
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import build_mm_energy_forces_fn

    positions, cell, offsets, _mid, _z = setup_charmm_composition_cluster(
        "ACO:2",
        box_side=40.0,
        spacing=10.0,
    )
    n_monomers = len(offsets) - 1
    atoms_per = int(offsets[1] - offsets[0])
    atoms_list = [atoms_per] * n_monomers
    box_L = float(np.diag(cell)[0])

    common_kw = dict(
        total_atoms=positions.shape[0],
        n_monomers=n_monomers,
        monomer_offsets=offsets,
        atoms_per_monomer_list=atoms_list,
        lambda_monomer=np.ones(n_monomers, dtype=np.float64),
        ml_switch_width=1.0,
        mm_switch_on=12.0,
        mm_switch_width=1.0,
        pbc_cell=box_L,
        defer_xla_gpu_warmup=True,
        mm_nl_backend="cell_list",
        use_jax_md_neighbor_list=False,
    )

    mic_result = build_mm_energy_forces_fn(positions, lr_solver="mic", **common_kw)
    if isinstance(mic_result, tuple):
        mic_fn, _ = mic_result
        e_mic, _ = mic_fn(positions)
    else:
        e_mic, _ = mic_result(positions)

    ok = True
    for method in ("ewald", "pme", "p3m"):
        os.environ["MMML_LR_SOLVER"] = "jax_pme"
        os.environ["JAX_PME_METHOD"] = method
        try:
            pme_result = build_mm_energy_forces_fn(
                positions,
                lr_solver="jax_pme",
                jax_pme_method=method,
                **common_kw,
            )
            if isinstance(pme_result, tuple):
                pme_fn, _ = pme_result
                e_pme, f_pme = pme_fn(positions)
            else:
                e_pme, f_pme = pme_result(positions)
            print_pass(
                f"{method}: E={float(e_pme):.4f} kcal/mol "
                f"(mic truncated elec baseline E={float(e_mic):.4f})"
            )
            if abs(float(e_pme)) <= abs(float(e_mic)) * 1.01:
                print_fail(f"{method}: jax-pme total not larger than truncated MIC")
                ok = False
        except Exception as exc:
            print_fail(f"{method}: {exc}")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
