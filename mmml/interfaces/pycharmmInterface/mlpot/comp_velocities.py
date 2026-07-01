"""CHARMM comparison (COMP) scalars — force copy only; not used in default heat.

See ``COMP_AND_HEATING.md`` for how COMP relates to ``IASVEL`` / ``IASORS`` and
what to grep in logs. Default staged heat keeps ``--heat-comp-damp`` off, so
these helpers do not affect ``dyna`` unless explicitly enabled.
"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
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


def _charmm_lib():
    import pycharmm.lib as lib

    return lib


def coor_set_comparison_capi(arr: np.ndarray) -> int:
    """Write COMP via ``coor_set_comparison`` (ctypes C API; no pandas)."""
    lib = _charmm_lib()

    if not hasattr(lib.charmm, "coor_set_comparison"):
        raise RuntimeError("libcharmm missing coor_set_comparison C API")

    data = np.asarray(arr, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] not in (3, 4):
        raise ValueError(
            f"comparison array must be (N, 3) or (N, 4), got {data.shape}"
        )
    n_atoms = int(data.shape[0])
    if n_atoms <= 0:
        return 0

    x = np.ascontiguousarray(data[:, 0], dtype=np.float64)
    y = np.ascontiguousarray(data[:, 1], dtype=np.float64)
    z = np.ascontiguousarray(data[:, 2], dtype=np.float64)
    if data.shape[1] == 3:
        w = np.zeros(n_atoms, dtype=np.float64)
    else:
        w = np.ascontiguousarray(data[:, 3], dtype=np.float64)

    cx = (ctypes.c_double * n_atoms).from_buffer(x)
    cy = (ctypes.c_double * n_atoms).from_buffer(y)
    cz = (ctypes.c_double * n_atoms).from_buffer(z)
    cw = (ctypes.c_double * n_atoms).from_buffer(w)
    return int(lib.charmm.coor_set_comparison(cx, cy, cz, cw))


def coor_get_comparison_capi(n_atoms: int | None = None) -> np.ndarray:
    """Read COMP as ``(N, 4)`` via ``coor_get_comparison`` (ctypes C API)."""
    lib = _charmm_lib()

    if not hasattr(lib.charmm, "coor_get_comparison"):
        raise RuntimeError("libcharmm missing coor_get_comparison C API")

    if n_atoms is None:
        n_atoms = int(_import_pycharmm().psf.get_natom())
    n_atoms = int(n_atoms)
    if n_atoms <= 0:
        return np.zeros((0, 4), dtype=np.float64)

    x = (ctypes.c_double * n_atoms)()
    y = (ctypes.c_double * n_atoms)()
    z = (ctypes.c_double * n_atoms)()
    w = (ctypes.c_double * n_atoms)()
    lib.charmm.coor_get_comparison(x, y, z, w)
    return np.column_stack(
        [
            np.fromiter(x, dtype=np.float64, count=n_atoms),
            np.fromiter(y, dtype=np.float64, count=n_atoms),
            np.fromiter(z, dtype=np.float64, count=n_atoms),
            np.fromiter(w, dtype=np.float64, count=n_atoms),
        ]
    )


def set_comparison_array(arr: np.ndarray) -> None:
    """Write COMP via :func:`coor_set_comparison_capi`."""
    coor_set_comparison_capi(arr)


def get_comparison_array() -> np.ndarray:
    """Read COMP as ``(N, 4)`` with columns x, y, z, w."""
    try:
        return coor_get_comparison_capi()
    except RuntimeError:
        pycharmm = _import_pycharmm()
        df = pycharmm.coor.get_comparison()
        return df[["x", "y", "z", "w"]].to_numpy(dtype=float)


def run_charmm_script(script: str, *, quiet: bool = False) -> None:
    """Run a single CHARMM script line."""
    if quiet:
        from mmml.interfaces.pycharmmInterface.charmm_levels import (
            run_charmm_script_quiet,
        )

        run_charmm_script_quiet(script)
        return
    _import_pycharmm().lingo.charmm_script(script)


def zero_comparison_scalars(sele: str = "all", *, quiet: bool = False) -> None:
    """Zero COMP scalar components via ``scalar xcomp/ycomp/zcomp/wcomp set 0``."""
    for comp in _COMP_COMPONENTS:
        run_charmm_script(f"scalar {comp} set 0 select {sele} end", quiet=quiet)


def clear_comparison_coordinates() -> None:
    """Zero the comparison **coordinate** set (``coor set comp``), not just scalars."""
    pycharmm = _import_pycharmm()
    n_atoms = int(pycharmm.psf.get_natom())
    if n_atoms <= 0:
        return
    zeros = np.zeros((n_atoms, 4), dtype=float)
    set_comparison_array(zeros)


def sync_comparison_velocities_akma(velocities_akma: np.ndarray) -> None:
    """Mirror AKMA velocity components into the COMP coordinate set (C API)."""
    v = np.asarray(velocities_akma, dtype=np.float64).reshape(-1, 3)
    w = np.zeros(v.shape[0], dtype=np.float64)
    coor_set_comparison_capi(np.column_stack([v, w]))


def comparison_velocities_akma() -> np.ndarray | None:
    """COMP x/y/z as AKMA velocity components when CHARMM uses the COMP path."""
    try:
        comp = coor_get_comparison_capi()
    except Exception:
        return None
    if comp.shape[0] == 0:
        return None
    return np.asarray(comp[:, :3], dtype=np.float64)


def sync_comparison_velocities_from_main() -> bool:
    """Copy readable main-set velocities into COMP; return False when unavailable."""
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        charmm_velocities_akma,
        velocities_are_cold,
    )

    vel = charmm_velocities_akma()
    if vel is None or velocities_are_cold(vel):
        return False
    sync_comparison_velocities_akma(vel)
    return True


def sync_comparison_velocities_from_comparison() -> bool:
    """True when COMP already holds warm AKMA velocities (post-dyna / handoff)."""
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        velocities_are_cold,
    )

    vel = comparison_velocities_akma()
    if vel is None or velocities_are_cold(vel):
        return False
    return True


def sync_comparison_velocities_from_restart(path: Path | str | None) -> bool:
    """Load ``!VELOCITIES`` from a restart file into COMP when present."""
    if path is None:
        return False
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        velocities_are_cold,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_velocities,
    )

    p = Path(path)
    if not p.is_file():
        return False
    vel = read_restart_velocities(p)
    if vel is None or velocities_are_cold(vel):
        return False
    sync_comparison_velocities_akma(vel)
    return True


def mirror_comparison_velocities_for_dynamics(
    kw: dict[str, Any],
    *,
    restart_read_path: Path | str | None = None,
) -> None:
    """When ``iasvel=0``, ensure COMP holds warm velocities for CHARMM ``dyna``.

    PyCHARMM may omit ``start=False`` so CHARMM keeps START and reads COMP as
    velocities. Zeroing COMP at overlap chunk boundaries yields T≈0; prefer
    main-set, in-memory COMP, or restart-file velocities instead.
    """
    if bool(kw.get("start")) or int(kw.get("iasvel", 1) or 0) != 0:
        return
    if restart_read_path is None:
        restart_read_path = kw.pop("_restart_read_path", None)
    if sync_comparison_velocities_from_main():
        return
    if sync_comparison_velocities_from_comparison():
        return
    if sync_comparison_velocities_from_restart(restart_read_path):
        return


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


def clear_comp_for_production(*, quiet: bool = False) -> None:
    """Clear comparison coords + scalars before dynamics (no COMP-velocity path)."""
    clear_comparison_coordinates()
    zero_comparison_scalars("all", quiet=quiet)
    run_charmm_script("scalar wcomp set 0 select all end", quiet=quiet)


_COMP_CLEARED_STAGES = frozenset({"nve", "equi", "prod"})


def apply_comp_velocity_policy(
    stage: str,
    kw: dict[str, Any],
    args: argparse.Namespace,
    *,
    quiet: bool | None = None,
) -> None:
    """Heat: optional COMP prep + gentler ``iasors=0`` scaling; later stages: clear COMP."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_heat_comp_damp,
        resolve_heat_comp_damp_kwargs,
    )

    silent = bool(quiet if quiet is not None else getattr(args, "quiet", False))

    if stage == "heat":
        if resolve_heat_comp_damp(args):
            damp_kw = resolve_heat_comp_damp_kwargs(args)
            n = prepare_comp_for_heat(**damp_kw)
            if not silent:
                from mmml.utils.rich_report import emit_tagged

                target = "H" if damp_kw.get("hydrogen_only", True) else "all"
                emit_tagged(
                    "HEAT COMP",
                    f"(experimental) copied scaled forces into COMP for {n} {target} "
                    "atoms — does NOT change dyna iasvel/iasors; see COMP_AND_HEATING.md",
                    tag_style="bold yellow",
                )
        else:
            clear_comp_for_production(quiet=silent)
            if not silent:
                from mmml.utils.rich_report import emit_tagged

                emit_tagged(
                    "HEAT COMP",
                    "cleared (default; no --heat-comp-damp); "
                    "never use iasvel=0 + start for COMP velocities",
                    tag_style="dim",
                )
    elif stage in _COMP_CLEARED_STAGES:
        clear_comp_for_production(quiet=silent)
        if not silent:
            from mmml.utils.rich_report import emit_tagged

            emit_tagged(
                stage.upper(),
                "COMP cleared (no force-damp)",
                tag_style="dim",
            )
