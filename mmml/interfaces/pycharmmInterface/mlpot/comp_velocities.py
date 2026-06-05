"""CHARMM comparison (COMP) scalars — force copy only; not used in default heat.

See ``COMP_AND_HEATING.md`` for how COMP relates to ``IASVEL`` / ``IASORS`` and
what to grep in logs. Default staged heat keeps ``--heat-comp-damp`` off, so
these helpers do not affect ``dyna`` unless explicitly enabled.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_COMP_FORCE_MIN_KCALMOL_A = 1.0
DEFAULT_COMP_FORCE_SCALE = 0.01
DEFAULT_HIGHF_STORE_NAME = "highf"

_COMP_COMPONENTS = ("xcomp", "ycomp", "zcomp", "wcomp")
_FORCE_COPY_PAIRS = (("xcomp", "dx"), ("ycomp", "dy"), ("zcomp", "dz"))


def _import_pycharmm():
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    return pycharmm


def _comparison_dataframe(arr: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (3, 4):
        raise ValueError(f"comparison array must be (N, 3) or (N, 4), got {arr.shape}")
    if arr.shape[1] == 3:
        return pd.DataFrame(
            {
                "x": arr[:, 0],
                "y": arr[:, 1],
                "z": arr[:, 2],
                "w": np.zeros(arr.shape[0], dtype=float),
            }
        )
    return pd.DataFrame(
        {
            "x": arr[:, 0],
            "y": arr[:, 1],
            "z": arr[:, 2],
            "w": arr[:, 3],
        }
    )


def set_comparison_array(arr: np.ndarray) -> None:
    """Write COMP via ``coor.set_comparison`` (bulk array path)."""
    pycharmm = _import_pycharmm()
    pycharmm.coor.set_comparison(_comparison_dataframe(arr))


def get_comparison_array() -> np.ndarray:
    """Read COMP as ``(N, 4)`` with columns x, y, z, w."""
    pycharmm = _import_pycharmm()
    df = pycharmm.coor.get_comparison()
    return df[["x", "y", "z", "w"]].to_numpy(dtype=float)


def run_charmm_script(script: str) -> None:
    """Run a single CHARMM script line."""
    _import_pycharmm().lingo.charmm_script(script)


def zero_comparison_scalars(sele: str = "all") -> None:
    """Zero COMP scalar components via ``scalar xcomp/ycomp/zcomp/wcomp set 0``."""
    for comp in _COMP_COMPONENTS:
        run_charmm_script(f"scalar {comp} set 0 select {sele} end")


def clear_comparison_coordinates() -> None:
    """Zero the comparison **coordinate** set (``coor set comp``), not just scalars."""
    pycharmm = _import_pycharmm()
    n_atoms = int(pycharmm.psf.get_natom())
    if n_atoms <= 0:
        return
    zeros = np.zeros((n_atoms, 4), dtype=float)
    set_comparison_array(zeros)


def force_magnitudes_kcalmol_A() -> np.ndarray:
    """Per-atom force magnitude from last ``ENER`` (kcal/mol/Å)."""
    pycharmm = _import_pycharmm()
    forces = pycharmm.coor.get_forces()
    dx = forces["dx"].to_numpy(dtype=float)
    dy = forces["dy"].to_numpy(dtype=float)
    dz = forces["dz"].to_numpy(dtype=float)
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def _selection_count(sel) -> int:
    return int(sum(sel.get_selection()))


def build_high_force_selection(
    min_force_kcalmol_A: float,
    *,
    store_name: str = DEFAULT_HIGHF_STORE_NAME,
    hydrogen_only: bool = False,
    exclude_hydrogen: bool = False,
) -> tuple[str, int]:
    """Store atoms with ``|F| >= min_force``; return ``(store_name, n_selected)``.

    ``hydrogen_only``: keep only hydrogens in the stored set (heat X–H stabilization).
    ``exclude_hydrogen``: drop hydrogens (legacy; prefer ``hydrogen_only`` for heat).
    """
    if hydrogen_only and exclude_hydrogen:
        raise ValueError("hydrogen_only and exclude_hydrogen are mutually exclusive")
    pycharmm = _import_pycharmm()
    select = pycharmm.select

    if select.find(store_name) > 0:
        select.delete_stored_selection(store_name.upper())

    mags = force_magnitudes_kcalmol_A()
    indices = np.flatnonzero(mags >= float(min_force_kcalmol_A)).tolist()
    n_atoms = pycharmm.psf.get_natom()
    if indices:
        sel = pycharmm.SelectAtoms(atom_nums=indices, update=False)
        if hydrogen_only:
            sel = sel & pycharmm.SelectAtoms(hydrogens=True, update=False)
        elif exclude_hydrogen:
            sel = sel & ~pycharmm.SelectAtoms(hydrogens=True, update=False)
    else:
        sel = pycharmm.SelectAtoms(update=False)
        sel.set_selection(select.none_selection(n_atoms))

    stored = sel.store(name=store_name)
    return stored, _selection_count(sel)


def unstore_selection(store_name: str) -> None:
    pycharmm = _import_pycharmm()
    select = pycharmm.select

    if select.find(store_name) > 0:
        select.delete_stored_selection(store_name.upper())


def apply_selective_force_damp_recipe(
    *,
    min_force_kcalmol_A: float = DEFAULT_COMP_FORCE_MIN_KCALMOL_A,
    force_scale: float = DEFAULT_COMP_FORCE_SCALE,
    store_name: str = DEFAULT_HIGHF_STORE_NAME,
    hydrogen_only: bool = False,
    exclude_hydrogen: bool = False,
) -> int:
    """Zero all COMP, then COPY/MULT **forces** (dx,dy,dz) into xcomp/ycomp/zcomp.

    This does **not** set velocities. CHARMM only uses COMP for dynamics when
    ``iasvel=0`` and comparison values are actual velocity components (see
    ``COMP_AND_HEATING.md``). Safe to call for inspection/tests only unless a
    full COMP-velocity workflow is wired.
    """
    zero_comparison_scalars("all")
    stored_name, n_selected = build_high_force_selection(
        min_force_kcalmol_A,
        store_name=store_name,
        hydrogen_only=hydrogen_only,
        exclude_hydrogen=exclude_hydrogen,
    )
    try:
        if n_selected > 0:
            scale = float(force_scale)
            for comp, force_key in _FORCE_COPY_PAIRS:
                run_charmm_script(
                    f"scalar {comp} copy {force_key} select {stored_name} end"
                )
                run_charmm_script(
                    f"scalar {comp} mult {scale} select {stored_name} end"
                )
        run_charmm_script("scalar wcomp set 0 select all end")
    finally:
        unstore_selection(stored_name)
    return n_selected


def prepare_comp_for_iasvel0(
    *,
    min_force_kcalmol_A: float = DEFAULT_COMP_FORCE_MIN_KCALMOL_A,
    force_scale: float = DEFAULT_COMP_FORCE_SCALE,
    zero_only: bool = False,
    store_name: str = DEFAULT_HIGHF_STORE_NAME,
    hydrogen_only: bool = False,
    exclude_hydrogen: bool = False,
) -> int:
    """``ENER`` then zero COMP; optionally selective force-damp into COMP."""
    run_charmm_script("ENER")
    if zero_only:
        zero_comparison_scalars("all")
        run_charmm_script("scalar wcomp set 0 select all end")
        return 0
    return apply_selective_force_damp_recipe(
        min_force_kcalmol_A=min_force_kcalmol_A,
        force_scale=force_scale,
        store_name=store_name,
        hydrogen_only=hydrogen_only,
        exclude_hydrogen=exclude_hydrogen,
    )


def prepare_comp_for_heat(
    *,
    min_force_kcalmol_A: float = DEFAULT_COMP_FORCE_MIN_KCALMOL_A,
    force_scale: float = DEFAULT_COMP_FORCE_SCALE,
    hydrogen_only: bool = True,
) -> int:
    """Selective COMP force-damp for heating (default: high-|F| hydrogens only)."""
    return prepare_comp_for_iasvel0(
        min_force_kcalmol_A=min_force_kcalmol_A,
        force_scale=force_scale,
        hydrogen_only=hydrogen_only,
        exclude_hydrogen=False,
    )


def clear_comp_for_production() -> None:
    """Clear comparison coords + scalars before dynamics (no COMP-velocity path)."""
    clear_comparison_coordinates()
    zero_comparison_scalars("all")
    run_charmm_script("scalar wcomp set 0 select all end")


_COMP_CLEARED_STAGES = frozenset({"nve", "equi", "prod"})


def apply_comp_velocity_policy(
    stage: str,
    kw: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Heat: optional COMP prep + gentler ``iasors=0`` scaling; later stages: clear COMP."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_heat_comp_damp,
        resolve_heat_comp_damp_kwargs,
    )

    if stage == "heat":
        if resolve_heat_comp_damp(args):
            damp_kw = resolve_heat_comp_damp_kwargs(args)
            n = prepare_comp_for_heat(**damp_kw)
            if not getattr(args, "quiet", False):
                target = "H" if damp_kw.get("hydrogen_only", True) else "all"
                print(
                    f"HEAT COMP (experimental): copied scaled forces into COMP "
                    f"for {n} {target} atoms — does NOT change dyna iasvel/iasors; "
                    "see COMP_AND_HEATING.md",
                    flush=True,
                )
        else:
            clear_comp_for_production()
            if not getattr(args, "quiet", False):
                print(
                    "HEAT COMP: cleared (default; no --heat-comp-damp); "
                    "never use iasvel=0 + start for COMP velocities",
                    flush=True,
                )
    elif stage in _COMP_CLEARED_STAGES:
        clear_comp_for_production()
        if not getattr(args, "quiet", False):
            print(f"{stage.upper()}: COMP cleared (no force-damp)", flush=True)
