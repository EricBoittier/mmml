"""Route MLpot MM nonbond components into CHARMM VDW/ELEC/IMNB/IMEL eterm slots."""

from __future__ import annotations

import os
from typing import Any


def mlpot_route_mm_to_charmm_eterms_enabled() -> bool:
    raw = (os.environ.get("MMML_MLPOT_ROUTE_MM_ETERMS") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def push_mlpot_nb_components_to_charmm(
    *,
    vdw_primary_kcal: float,
    vdw_image_kcal: float,
    elec_primary_kcal: float,
    elec_image_kcal: float,
    route: bool = True,
) -> None:
    """Stage MM nonbond buckets for the next ``mlpot_call`` (Fortran ``api_func.F90``)."""
    if not route or not mlpot_route_mm_to_charmm_eterms_enabled():
        return
    try:
        import ctypes

        import pycharmm.lib as lib
    except (ImportError, OSError):
        return
    setter = getattr(lib.charmm, "mlpot_set_nb_components", None)
    if setter is None:
        return
    setter(
        ctypes.c_double(float(vdw_primary_kcal)),
        ctypes.c_double(float(elec_primary_kcal)),
        ctypes.c_double(float(vdw_image_kcal)),
        ctypes.c_double(float(elec_image_kcal)),
        ctypes.c_int(1),
    )


def route_mlpot_callback_energy_kcalmol(
    energy_kcal: float,
    components: dict[str, float],
    *,
    route: bool = True,
) -> float:
    """Push MM buckets to CHARMM eterm slots; return USER energy (ML + LR not routed)."""
    mm_total = float(components.get("mm_total", 0.0))
    energy_kcal = float(energy_kcal)
    do_route = bool(route and mlpot_route_mm_to_charmm_eterms_enabled())
    user_kcal = energy_kcal - mm_total if do_route else energy_kcal
    # When decomposition captures ~all hybrid energy as MM but CHARMM VDW/ELEC are
    # blocked (all-ML BLOCK), routing would leave USER≈0 and discard ML from ENER.
    if (
        do_route
        and abs(user_kcal) <= 1.0e-6
        and abs(energy_kcal) > 1.0
        and abs(mm_total) >= abs(energy_kcal) * 0.99
    ):
        do_route = False
        user_kcal = energy_kcal
    if do_route:
        push_mlpot_nb_components_to_charmm(
            vdw_primary_kcal=float(components.get("vdw_primary", 0.0)),
            vdw_image_kcal=float(components.get("vdw_image", 0.0)),
            elec_primary_kcal=float(components.get("elec_primary", 0.0)),
            elec_image_kcal=float(components.get("elec_image", 0.0)),
            route=True,
        )
    return float(user_kcal)


def decompose_and_route_mlpot_mm_from_callback(
    calculator: Any,
    positions_A: Any,
    mm_pair_idx: Any,
    mm_pair_mask: Any,
    box: Any | None,
    energy_kcal: float,
    *,
    use_mm_pairs: bool,
) -> float:
    """Compute primary/image MM buckets from callback pair lists and adjust USER energy."""
    if not use_mm_pairs or not getattr(calculator, "_do_mm", True):
        return float(energy_kcal)
    if not mlpot_route_mm_to_charmm_eterms_enabled():
        return float(energy_kcal)
    try:
        import numpy as np

        from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
            _get_actual_psf_charges,
            decompose_mlpot_mm_nb_eterms_kcalmol,
        )
    except ImportError:
        return float(energy_kcal)

    pos = np.asarray(positions_A, dtype=np.float64)
    n = int(pos.shape[0])
    pair_idx = np.asarray(mm_pair_idx, dtype=np.int32)
    pair_mask = np.asarray(mm_pair_mask, dtype=bool)
    cp = getattr(calculator, "cutoff_params", None)
    if cp is None:
        return float(energy_kcal)
    try:
        import pycharmm.param as param
    except (ImportError, OSError):
        return float(energy_kcal)

    atc = list(param.get_atc())[:n]
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import CGENFF_PRM, CGENFF_RTF

    cgenff_params_dict: dict[str, tuple[float, float]] = {}
    for line in open(CGENFF_PRM).readlines():
        parts = line.split()
        if len(parts) > 4 and parts[1] == "0.0" and line[0] != "!":
            cgenff_params_dict[parts[0]] = (float(parts[2]), float(parts[3]))
    rmins = np.array([cgenff_params_dict.get(at, (0.0, 0.0))[1] for at in atc], dtype=np.float64)
    eps = np.array(
        [-abs(cgenff_params_dict.get(at, (0.0, 0.0))[0]) for at in atc],
        dtype=np.float64,
    )

    offsets = np.zeros(len(calculator._atoms_per_monomer) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(np.asarray(calculator._atoms_per_monomer, dtype=np.int32))
    monomer_id = np.zeros(n, dtype=np.int32)
    for m in range(len(calculator._atoms_per_monomer)):
        monomer_id[offsets[m] : offsets[m + 1]] = m

    charges = np.asarray(_get_actual_psf_charges(n), dtype=np.float64)[:n]
    cell_np = None
    if box is not None:
        cell_np = np.asarray(box, dtype=np.float64)

    components = decompose_mlpot_mm_nb_eterms_kcalmol(
        pos,
        pair_idx,
        pair_mask,
        cell_np,
        charges_e=charges,
        rmins_A=np.asarray(rmins)[:n],
        epsilons_kcal=np.asarray(eps)[:n],
        monomer_id=monomer_id,
        mm_switch_on=float(cp.mm_switch_on),
        mm_switch_width=float(cp.mm_switch_width),
        ml_switch_width=float(cp.ml_switch_width),
        complementary_handoff=bool(cp.complementary_handoff),
    )
    calculator._last_mm_nb_components_kcalmol = components
    return route_mlpot_callback_energy_kcalmol(float(energy_kcal), components)
