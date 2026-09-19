"""Route MLpot MM nonbond components into CHARMM VDW/ELEC/IMNB/IMEL eterm slots.

``MMML_MLPOT_ETERM_SPLIT_SOURCE`` selects the split (reporting only; forces and the
total energy are unaffected):

* ``charmm`` (default): CHARMM's live q/ε with the CHARMM LJ form. In all-ML runs
  those are zeroed, so the pair pass is skipped (#226) and VDW/ELEC report 0 with
  all MM energy in USER. Costs nothing per step there.
* ``hybrid`` (opt-in): the hybrid JAX MM's own split (``update_mm_pairs.mm_eterm_split``),
  so VDW/ELEC/IMNB/IMEL show the true MM contribution, at the cost of one extra
  MM forward (no grad) per force call.
"""

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
        and abs(user_kcal) <= max(1.0e-6, abs(energy_kcal) * 1.0e-9)
        and abs(energy_kcal) > 1.0e-12
        and abs(mm_total) >= max(abs(energy_kcal) * 0.99, 1.0e-12)
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


def _zero_nb_components() -> dict[str, float]:
    return {
        "vdw_primary": 0.0,
        "vdw_image": 0.0,
        "elec_primary": 0.0,
        "elec_image": 0.0,
        "mm_total": 0.0,
    }


def _hybrid_mm_eterm_split(
    calculator: Any, positions_A: Any, mm_pair_idx: Any, mm_pair_mask: Any, box: Any | None
) -> dict[str, float] | None:
    """Split from the hybrid's own JAX MM (its q/ε/Rmin, λ, COM switch), or None.

    CHARMM's live charges/ε are zeroed for ML atoms in all-ML runs, so a split
    built from them always reports VDW = ELEC = 0 and leaves the MM in USER.
    Opt-in (``MMML_MLPOT_ETERM_SPLIT_SOURCE=hybrid``): one MM forward per callback.
    """
    if (os.environ.get("MMML_MLPOT_ETERM_SPLIT_SOURCE") or "charmm").strip().lower() != "hybrid":
        return None
    update_fn = getattr(calculator, "_cached_update_fn", None)
    get_update_fn = getattr(calculator, "_get_update_fn", None)
    if update_fn is None and get_update_fn is not None:
        update_fn = get_update_fn(positions_A, calculator.cutoff_params, box=box)
        calculator._cached_update_fn = update_fn
    split_fn = getattr(update_fn, "mm_eterm_split", None)
    if split_fn is None:
        return None
    import jax
    import jax.numpy as jnp
    import numpy as np

    v = np.asarray(
        jax.device_get(split_fn(jnp.asarray(positions_A), mm_pair_idx, mm_pair_mask, box)),
        dtype=np.float64,
    )
    v = np.where(np.isfinite(v), v, 0.0)
    keys = ("vdw_primary", "vdw_image", "elec_primary", "elec_image")
    out = {k: float(x) for k, x in zip(keys, v)}
    out["mm_total"] = float(v.sum())
    return out


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
            decompose_mlpot_mm_nb_eterms_kcalmol,
        )
    except ImportError:
        return float(energy_kcal)

    n = int(np.shape(positions_A)[0])
    cp = getattr(calculator, "cutoff_params", None)
    if cp is None:
        return float(energy_kcal)

    try:
        split = _hybrid_mm_eterm_split(calculator, positions_A, mm_pair_idx, mm_pair_mask, box)
    except Exception as exc:
        import sys

        print(f"WARN: hybrid MM eterm split failed ({exc}); using CHARMM params", file=sys.stderr)
        split = None
    if split is not None:
        calculator._last_mm_nb_components_kcalmol = split
        return route_mlpot_callback_energy_kcalmol(float(energy_kcal), split)

    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        _live_charmm_nonbonded_arrays,
    )

    live = _live_charmm_nonbonded_arrays(n)
    if live is not None:
        charges, eps, rmins = live
        charges = np.asarray(charges, dtype=np.float64)[:n]
        eps = np.asarray(eps, dtype=np.float64)[:n]
        rmins = np.asarray(rmins, dtype=np.float64)[:n]
    else:
        try:
            import pycharmm.atom_info as atom_info
        except (ImportError, OSError):
            return float(energy_kcal)
        from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
            CGENFF_PRM,
            _get_actual_psf_charges,
        )

        cgenff_params_dict: dict[str, tuple[float, float]] = {}
        for line in open(CGENFF_PRM).readlines():
            parts = line.split()
            if len(parts) > 4 and parts[1] == "0.0" and line[0] != "!":
                cgenff_params_dict[parts[0]] = (float(parts[2]), float(parts[3]))
        chem_types = atom_info.get_chem_types(list(range(n)))
        rmins = np.array(
            [cgenff_params_dict.get(at, (0.0, 0.0))[1] for at in chem_types],
            dtype=np.float64,
        )
        eps = np.array(
            [-abs(cgenff_params_dict.get(at, (0.0, 0.0))[0]) for at in chem_types],
            dtype=np.float64,
        )
        charges = np.asarray(_get_actual_psf_charges(n), dtype=np.float64)[:n]

    if not (np.any(charges) or np.any(eps)):
        # Every per-pair term carries q_i*q_j or sqrt(eps_i*eps_j), so the split is
        # exactly zero. This is the all-ML case: the energy policy zeroes CHARMM's
        # live charges/eps (the JAX MM term keeps its own copy), and the full pair
        # pass (~6.6e5 pairs, ETOH:181) plus the device->host pair-list copy cost
        # ~1/3 of every MD step for nothing.
        components = _zero_nb_components()
        calculator._last_mm_nb_components_kcalmol = components
        return route_mlpot_callback_energy_kcalmol(float(energy_kcal), components)

    pos = np.asarray(positions_A, dtype=np.float64)
    pair_idx = np.asarray(mm_pair_idx, dtype=np.int32)
    pair_mask = np.asarray(mm_pair_mask, dtype=bool)

    offsets = np.zeros(len(calculator._atoms_per_monomer) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(np.asarray(calculator._atoms_per_monomer, dtype=np.int32))
    monomer_id = np.zeros(n, dtype=np.int32)
    for m in range(len(calculator._atoms_per_monomer)):
        monomer_id[offsets[m] : offsets[m + 1]] = m

    cell_np = None
    if box is not None:
        cell_np = np.asarray(box, dtype=np.float64)

    try:
        components = decompose_mlpot_mm_nb_eterms_kcalmol(
            pos,
            pair_idx,
            pair_mask,
            cell_np,
            charges_e=charges,
            rmins_A=rmins,
            epsilons_kcal=eps,
            monomer_id=monomer_id,
            mm_switch_on=float(cp.mm_switch_on),
            mm_switch_width=float(cp.mm_switch_width),
            ml_switch_width=float(cp.ml_switch_width),
            complementary_handoff=bool(cp.complementary_handoff),
        )
    except Exception as exc:
        import sys

        print(
            f"WARN: MLpot MM eterm decomposition failed ({exc}); "
            "returning full hybrid energy as USER",
            file=sys.stderr,
            flush=True,
        )
        return float(energy_kcal)
    calculator._last_mm_nb_components_kcalmol = components
    return route_mlpot_callback_energy_kcalmol(float(energy_kcal), components)
