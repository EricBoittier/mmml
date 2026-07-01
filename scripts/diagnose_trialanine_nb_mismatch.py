#!/usr/bin/env python3
"""Diagnose JAX vs PyCHARMM nonbonded mismatch for TRIA water box."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _perturb(pos: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return pos + rng.normal(scale=0.02, size=pos.shape)


def _charmm_nb_components() -> dict[str, float]:
    import pycharmm.energy as energy

    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        charmm_bonded_energy_components_kcalmol,
        charmm_nonbonded_energy_components_kcalmol,
        run_charmm_bonded_ener_force,
    )

    run_charmm_bonded_ener_force(silent=True)
    bonded = charmm_bonded_energy_components_kcalmol()
    nb = charmm_nonbonded_energy_components_kcalmol()
    return {
        "bonded_total": bonded["total"],
        "urey": bonded.get("urey", 0.0),
        "vdw": nb["vdw"],
        "elec": nb["elec"],
        "nb_total": nb["total"],
        "ener_total": float(energy.get_total()),
    }


def _jax_components(pos, box) -> dict[str, float]:
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        CharmmNbondSettings,
        bonded_energy_and_forces,
        load_bonded_system_from_psf,
        load_nonbonded_system_from_charmm,
        nonbonded_energy_and_forces,
    )

    bonded = load_bonded_system_from_psf(
        box.psf_path, pos, prm_file=box.cgenff_prm
    )
    nb = load_nonbonded_system_from_charmm(box.psf_path, box.cgenff_prm)
    cuts = box.nbond_cutoffs
    settings = CharmmNbondSettings(
        cutnb=float(cuts.cutnb),
        ctonnb=float(cuts.ctonnb),
        ctofnb=float(cuts.ctofnb),
    )
    bc, _ = bonded_energy_and_forces(
        jnp.asarray(pos), bonded.topology, bonded.bonded, energy_unit="kcal/mol"
    )
    nc, _ = nonbonded_energy_and_forces(pos, nb, box.cell, settings)
    return {
        "bonded_total": float(bc["total"]),
        "vdw": float(nc["vdw"]),
        "elec": float(nc["elec"]),
        "nb_total": float(nc["total"]),
        "n_excl": len(nb.excluded_pairs),
        "n_e14": len(nb.e14_pairs),
        "n_pairs": len(
            __import__(
                "mmml.interfaces.pycharmmInterface.mm_system_energy",
                fromlist=["_build_pair_indices"],
            )._build_pair_indices(pos, box.cell, nb.excluded_pairs, settings.cutnb)[0]
        ),
    }


def _reapply_nbonds(box, *, vfswitch: bool) -> None:
    from mmml.interfaces.pycharmmInterface.nbonds_config import apply_nbonds_kwargs

    cuts = box.nbond_cutoffs
    kw = cuts.as_pbc_nbond_kwargs(nbxmod=5)
    kw["vfswitch"] = vfswitch
    kw["fswitch"] = True
    apply_nbonds_kwargs(kw, rebuild=True)


def main() -> int:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded

    ensure_pycharmm_loaded()
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import set_charmm_positions
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        build_trialanine_water_box_in_charmm,
    )

    os.environ.setdefault("MMML_LR_SOLVER", "mic")
    workdir = Path("/tmp/tria_nb_diag")
    box = build_trialanine_water_box_in_charmm(
        n_waters=10, box_side_A=28.0, seed=11, workdir=workdir
    )
    pos = _perturb(box.positions, seed=31)
    set_charmm_positions(pos)
    apply_charmm_mm_block()

    jax = _jax_components(pos, box)
    print("=== JAX (fswitch-style switch on VDW + elec) ===")
    for k, v in jax.items():
        print(f"  {k}: {v}")

    for vfswitch in (True, False):
        _reapply_nbonds(box, vfswitch=vfswitch)
        ch = _charmm_nb_components()
        label = "vfswitch ON" if vfswitch else "vfswitch OFF"
        print(f"\n=== CHARMM ({label}) ===")
        for k, v in ch.items():
            print(f"  {k}: {v}")
        print(
            f"  delta nb vs JAX: vdw {ch['vdw']-jax['vdw']:+.3f}, "
            f"elec {ch['elec']-jax['elec']:+.3f}, "
            f"nb {ch['nb_total']-jax['nb_total']:+.3f}"
        )
        print(
            f"  delta total (bonded+nb): "
            f"{(ch['bonded_total']+ch['nb_total'])-(jax['bonded_total']+jax['nb_total']):+.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
