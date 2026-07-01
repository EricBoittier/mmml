"""Register PhysNet (or other) models with ``pycharmm.MLpot``."""

from __future__ import annotations

import ctypes
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def _positions_xyzw_dataframe(arr: np.ndarray) -> pd.DataFrame:
    """Build a CHARMM coor dataframe with ``x,y,z,w`` (``w`` = 1)."""
    n = arr.shape[0]
    return pd.DataFrame(
        {
            "x": arr[:, 0],
            "y": arr[:, 1],
            "z": arr[:, 2],
            "w": np.ones(n, dtype=float),
        }
    )


def sync_charmm_positions(positions: np.ndarray) -> None:
    """Push ``(N, 3)`` into CHARMM main and auxiliary coordinate sets."""
    import pycharmm.coor as coor

    arr = np.asarray(positions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {arr.shape}")
    n = coor.get_natom()
    if arr.shape[0] != n:
        raise ValueError(f"positions rows {arr.shape[0]} != CHARMM natom {n}")

    xyz = pd.DataFrame(arr, columns=["x", "y", "z"])
    xyzw = _positions_xyzw_dataframe(arr)
    coor.set_positions(xyz)
    coor.set_main(xyzw)
    # Never mirror main coords into COMP. Zero comparison coordinates so a lingering
    # START + iasvel=0 cannot treat positions as velocities (COMP_AND_HEATING.md).
    comp_zero = pd.DataFrame(
        {
            "x": np.zeros(n, dtype=float),
            "y": np.zeros(n, dtype=float),
            "z": np.zeros(n, dtype=float),
            "w": np.zeros(n, dtype=float),
        }
    )
    coor.set_comparison(comp_zero)

    check = get_charmm_positions_array()
    if np.allclose(check, 0.0) and not np.allclose(arr, 0.0):
        raise RuntimeError(
            "sync_charmm_positions: CHARMM coordinates still zero after set_main/set_positions"
        )


def get_charmm_positions_array() -> np.ndarray:
    """Read CHARMM coordinates as ``(N, 3)`` (main set, then positions, then comparison).

    Returns all-zero array when no coordinates are loaded yet (e.g., after a freshly
    built PSF with no subsequent coor read).  Callers that use these positions as a
    restart seed should check ``np.allclose(pos, 0.0)`` and raise if they do not
    expect a fresh start.
    """
    import warnings

    import pycharmm.coor as coor

    for getter in (coor.get_main, coor.get_positions, coor.get_comparison):
        df = getter()
        pos = df[["x", "y", "z"]].to_numpy(dtype=float)
        if pos.shape[0] and not np.allclose(pos, 0.0):
            n = int(coor.get_natom())
            if n > 0 and pos.shape[0] > n:
                pos = np.asarray(pos[:n], dtype=float)
            return pos
    n = coor.get_natom()
    if n > 0:
        warnings.warn(
            f"get_charmm_positions_array: all {n} CHARMM atom coordinates are zero "
            "(PSF loaded but no coordinates set). Returning zero array — this will "
            "produce an invalid restart if used as initial positions.",
            stacklevel=2,
        )
    return np.zeros((n, 3), dtype=float)


def resolve_export_positions(
    *,
    pyCModel: Any = None,
    reference_positions: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Best-effort positions for file export after minimization."""
    charmm = get_charmm_positions_array()
    if charmm.size and not np.allclose(charmm, 0.0):
        # CHARMM SD is authoritative; calculator cache can lag or include image atoms.
        return charmm

    if pyCModel is not None:
        calc = pyCModel.get_pycharmm_calculator()
        cached = getattr(calc, "last_full_positions", None)
        if cached is not None:
            cached = np.asarray(cached, dtype=float)
            if cached.size and not np.allclose(cached, 0.0):
                if cached.shape[0] == charmm.shape[0] or charmm.size == 0:
                    return cached

    if reference_positions is not None:
        ref = np.asarray(reference_positions, dtype=float)
        if ref.size and not np.allclose(ref, 0.0):
            return ref
    return None


@dataclass
class MlpotContext:
    """Active MLpot registration (call :meth:`unset` when finished)."""

    mlpot: Any
    pyCModel: Any
    params: Any
    model: Any
    ml_selection: Any = None
    block_tag: str = "all"
    ml_Z: np.ndarray | None = None
    use_pbc: bool = False
    cubic_box_side_A: float | None = None
    charmm_cubic_box_side_A: float | None = None
    ml_charge: float = 0.0
    ml_fq: bool = True
    mm_internal_scale: float = 0.0
    registration_uses_block: bool = False
    periodic_external: bool = False
    periodic_charmm_vdw: bool = True
    topology_psf_path: Path | None = None
    topology_fingerprint: Any = None
    pre_mlpot_iblo: list[int] | None = None
    pre_mlpot_inb: list[int] | None = None
    sd_watchdog_baseline_grms: float | None = None

    def unset(self) -> None:
        self.mlpot.unset_mlpot()
        from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
            apply_charmm_mm_block,
            clear_mlpot_energy_block,
        )

        if self.ml_selection is not None and self.registration_uses_block:
            clear_mlpot_energy_block(self.ml_selection, block_tag=self.block_tag)
        apply_charmm_mm_block()

    def reregister_mlpot(
        self,
        *,
        verbose: bool = False,
        force: bool = False,
        reregister_params: bool = False,
    ) -> None:
        """Re-attach MLpot after temporary MM-only work or coordinate updates.

        Default ``reregister_params=False``: only re-enable the MLpot callback.
        Set ``reregister_params=True`` after ``unset()`` / full CGENFF restore
        (``READ PARAM APPEND`` clears PBC lists and needs a rebuild).
        """
        if self.ml_selection is None or self.ml_Z is None:
            raise RuntimeError("MlpotContext missing ml_selection or ml_Z for reregister")
        if reregister_params:
            self.block_tag = _apply_mlpot_psf_mm_off_and_pbc(self, verbose=verbose)
        reattach = getattr(self.mlpot, "reattach_mlpot", None)
        if callable(reattach):
            # Do not construct a new MLpot(): __init__ rebuilds iblo/inb via update_bnbnd
            # (upinb), which segfaults after long MD. Re-enable the existing callback.
            if force:
                unset = getattr(self.mlpot, "unset_mlpot", None)
                if callable(unset):
                    unset()
                elif hasattr(self.mlpot, "is_set"):
                    self.mlpot.is_set = False
            reattach()
            rebind_mlpot_calculator_from_pycmodel(self, verbose=verbose)
            return

        self._reattach_mlpot_compat()
        rebind_mlpot_calculator_from_pycmodel(self, verbose=verbose)

    def _reattach_mlpot_compat(self) -> None:
        """Compatibility path for PyCHARMM MLpot builds without ``reattach_mlpot``."""
        required = ("energy_func", "ml_indices", "ml_Z", "ml_Natoms")
        missing = [name for name in required if not hasattr(self.mlpot, name)]
        if missing:
            raise RuntimeError(
                "PyCHARMM MLpot cannot be reattached; missing attributes: "
                + ", ".join(missing)
            )

        pycharmm = _import_pycharmm()
        pycharmm.lib.charmm.mlpot_set_func(self.mlpot.energy_func)
        ml_indices = np.asarray(self.mlpot.ml_indices, dtype=int)
        ml_z = np.asarray(self.mlpot.ml_Z, dtype=int)
        n_ml = int(self.mlpot.ml_Natoms)
        mlidx = (ctypes.c_int * n_ml)()
        mlidx[:] = ml_indices + 1
        mlidz = (ctypes.c_int * n_ml)()
        mlidz[:] = ml_z
        nml = (ctypes.c_int * 1)(n_ml)
        pycharmm.lib.charmm.mlpot_set_properties(nml, mlidx, mlidz)
        if hasattr(self.mlpot, "is_set"):
            self.mlpot.is_set = True


def _read_mlpot_user_energy_kcal(*, force: bool = True) -> float | None:
    """Read CHARMM USER energy (kcal/mol) after ``ENER`` or ``ENER FORCE``."""
    import math

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm
    import pycharmm.energy as energy

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command

    script = "ENER FORCE" if force else "ENER"
    with charmm_silent_command():
        pycharmm.lingo.charmm_script(script)
    try:
        value = float(energy.get_term_by_name("USER"))
    except (ValueError, IndexError, TypeError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _mlpot_user_missing(user: float | None, *, zero_tol_kcalmol: float) -> bool:
    import math

    return user is None or not math.isfinite(user) or abs(float(user)) <= zero_tol_kcalmol


def rebind_mlpot_calculator_from_pycmodel(
    ctx: MlpotContext,
    *,
    verbose: bool = False,
) -> bool:
    """Point CHARMM MLpot at the current ``pyCModel`` calculator (post-JAX finalize).

    Deferred registration binds a stub calculator at ``MLpot.__init__``; after
    ``_finalize_jax_factory`` / warmup, ``get_pycharmm_calculator()`` returns a
    fresh ``DecomposedMlpotCalculator`` while Fortran may still call the stale stub.
    """
    import ctypes

    pyCModel = ctx.pyCModel
    mlpot = ctx.mlpot
    if pyCModel is None or mlpot is None:
        return False
    get_calc = getattr(pyCModel, "get_pycharmm_calculator", None)
    if not callable(get_calc):
        return False

    pycharmm = _import_pycharmm()
    calc = get_calc()
    mlpot.calculator = calc
    mlpot.energy_func = mlpot.func_type(calc.calculate_charmm)
    pycharmm.lib.charmm.mlpot_set_func(mlpot.energy_func)
    mlidx = (ctypes.c_int * mlpot.ml_Natoms)()
    mlidx[:] = mlpot.ml_indices + 1
    mlidz = (ctypes.c_int * mlpot.ml_Natoms)()
    mlidz[:] = mlpot.ml_Z
    nml = (ctypes.c_int * 1)(mlpot.ml_Natoms)
    pycharmm.lib.charmm.mlpot_set_properties(nml, mlidx, mlidz)
    mlpot.is_set = True
    if verbose:
        print(
            "MLpot: rebound CHARMM callback to current pyCModel calculator",
            flush=True,
        )
    return True


def mlpot_skip_charmm_ener_force_before_first_sd(mlpot_ctx: Any) -> bool:
    """Skip CHARMM ``ENER FORCE`` between MLpot registration and the first SD step.

    ``MLpot.__init__`` already runs ``upinb`` once for PBC exclusions. A second
    ``ENER FORCE`` → ``update`` → ``upinb`` on MPI-linked ``libcharmm.so`` (deferred
    JAX path) can segfault before the first MLpot SD step materializes the callback.
    """
    if not bool(getattr(mlpot_ctx, "use_pbc", False)):
        return False
    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        charmm_lib_links_mpi,
        defer_jax_warmup_until_after_mlpot_sd,
    )

    if not charmm_lib_links_mpi():
        return False
    if defer_jax_warmup_until_after_mlpot_sd():
        return True
    pyCModel = getattr(mlpot_ctx, "pyCModel", None)
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import DecomposedMlpotModel

    if isinstance(pyCModel, DecomposedMlpotModel):
        if getattr(pyCModel, "_defer_jax_until_after_sd", False) and not getattr(
            pyCModel, "_jax_on_gpu", False
        ):
            return True
    return False


def assert_mlpot_user_active(
    ctx: MlpotContext,
    *,
    context: str = "dynamics",
    zero_tol_kcalmol: float = 1.0e-12,
    quiet: bool = False,
) -> float:
    """Ensure the CHARMM USER term is active before MLpot dynamics.

    In all-ML workflows CHARMM bonded/nonbonded terms are intentionally zeroed by
    BLOCK, so a missing USER term leaves dynamics integrating a free gas.
    """
    rebind_mlpot_calculator_from_pycmodel(ctx, verbose=not quiet)
    user = _read_mlpot_user_energy_kcal(force=True)
    missing = _mlpot_user_missing(user, zero_tol_kcalmol=zero_tol_kcalmol)
    is_set = getattr(ctx.mlpot, "is_set", True)
    if missing or is_set is False:
        if not quiet:
            print(
                f"WARN: MLpot USER term missing before {context}; attempting reattach",
                flush=True,
            )
        ctx.reregister_mlpot(force=True, reregister_params=True)
        user = _read_mlpot_user_energy_kcal(force=True)
        missing = _mlpot_user_missing(user, zero_tol_kcalmol=zero_tol_kcalmol)
    if missing:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            light_resync_mlpot_state,
        )

        if not quiet:
            print(
                f"WARN: MLpot USER still missing before {context}; "
                "running light resync (ENER FORCE + UPDATE)",
                flush=True,
            )
        light_resync_mlpot_state(
            ctx,
            context=f"{context} USER recovery",
            silent_charmm=True,
            verbose=not quiet,
        )
        rebind_mlpot_calculator_from_pycmodel(ctx, verbose=not quiet)
        user = _read_mlpot_user_energy_kcal(force=True)
        missing = _mlpot_user_missing(user, zero_tol_kcalmol=zero_tol_kcalmol)

    if missing:
        raise RuntimeError(
            f"MLpot USER term is zero/missing before {context}; refusing to run dynamics"
        )
    if not quiet:
        from mmml.data.units import format_energy_kcal_ev
        from mmml.utils.rich_report import emit_tagged

        emit_tagged(
            "MLpot",
            f"USER active before {context}: {format_energy_kcal_ev(float(user))}",
            tag_style="bold green",
        )
    return float(user)


def _z_from_psf_masses(masses: np.ndarray) -> np.ndarray:
    """Element Z from PSF masses (same recipe as ``get_Z_from_psf``)."""
    import ase.data

    ase_m = ase.data.atomic_masses_common
    return np.asarray(
        [int(np.argmin((ase_m - float(m)) ** 2)) for m in masses],
        dtype=int,
    )


def _masses_consistent_with_z(
    masses: np.ndarray,
    z: np.ndarray,
) -> list[str]:
    """Return issues when assigned Z is not the best ASE nearest-mass match.

    Uses the same rule as ``get_Z_from_psf``. CGenFF masses may differ from ASE
    tabulated weights (e.g. Cl); we only require that the registered ``Z`` matches
    the mass-based assignment, not exact ASE mass equality.
    """
    import ase.data

    ase_m = ase.data.atomic_masses_common
    z_from_mass = _z_from_psf_masses(masses)
    issues: list[str] = []
    for i, (mass, zi, z_best) in enumerate(zip(masses, z, z_from_mass, strict=True)):
        zi = int(zi)
        z_best = int(z_best)
        if zi < 0 or zi >= len(ase_m):
            issues.append(f"atom {i}: invalid Z={zi}")
            continue
        if z_best != zi:
            issues.append(
                f"atom {i}: assigned Z={zi} but PSF mass={float(mass):.4f} amu "
                f"best matches Z={z_best} (ASE={float(ase_m[z_best]):.4f} amu)"
            )
    return issues


def _calculator_atomic_numbers(ctx: MlpotContext) -> np.ndarray | None:
    """Best-effort Z array stored on the MLpot-linked PyCHARMM calculator."""
    calc = getattr(ctx.mlpot, "calculator", None)
    if calc is None:
        return None

    def _z_from_calc(target: Any) -> np.ndarray | None:
        for attr in ("ml_atomic_numbers", "atomic_numbers"):
            raw = getattr(target, attr, None)
            if raw is not None:
                return np.asarray(raw, dtype=int)
        return None

    z = _z_from_calc(calc)
    if z is not None:
        return z
    real = getattr(calc, "_real", None)
    if real is not None:
        z = _z_from_calc(real)
        if z is not None:
            return z
    return None


def _model_atomic_numbers(pyCModel: Any) -> np.ndarray | None:
    raw = getattr(pyCModel, "_atomic_numbers", None)
    if raw is None:
        return None
    return np.asarray(raw, dtype=int)


def verify_mlpot_charmm_atom_consistency(
    ctx: MlpotContext,
    *,
    expected_z: Sequence[int] | np.ndarray | None = None,
    context: str = "dynamics",
    quiet: bool = False,
) -> None:
    """Verify PSF masses/Z, MLpot registration, and calculator agree before MD.

    Raises ``RuntimeError`` on mismatch (wrong atom order or element mapping breaks
    PhysNet and CHARMM integration).
    """
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.coor as coor
    import pycharmm.psf as psf

    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

    if ctx.ml_Z is None:
        raise RuntimeError(
            f"Atom consistency check before {context}: MlpotContext.ml_Z is missing"
        )

    n_psf = int(coor.get_natom())
    masses = np.asarray(psf.get_amass(), dtype=float)
    atypes = [str(x) for x in np.asarray(psf.get_atype(), dtype=str)]
    z_psf = np.asarray(get_Z_from_psf(), dtype=int)
    z_ctx = np.asarray(ctx.ml_Z, dtype=int)
    z_mlpot = np.asarray(ctx.mlpot.ml_Z, dtype=int)
    ml_idx = np.asarray(ctx.mlpot.ml_indices, dtype=int)
    n_ml = int(ctx.mlpot.ml_Natoms)

    issues: list[str] = []
    if masses.shape[0] != n_psf:
        issues.append(f"PSF mass count {masses.shape[0]} != natom {n_psf}")
    if z_psf.shape[0] != n_psf:
        issues.append(f"PSF-derived Z length {z_psf.shape[0]} != natom {n_psf}")
    if z_ctx.shape[0] != n_ml:
        issues.append(f"context ml_Z length {z_ctx.shape[0]} != ml_Natoms {n_ml}")
    if z_mlpot.shape[0] != n_ml:
        issues.append(f"mlpot.ml_Z length {z_mlpot.shape[0]} != ml_Natoms {n_ml}")
    if n_ml != n_psf:
        issues.append(f"ml_Natoms {n_ml} != CHARMM natom {n_psf}")

    if expected_z is not None:
        z_exp = np.asarray(expected_z, dtype=int)
        if z_exp.shape != z_ctx.shape or not np.array_equal(z_exp, z_ctx):
            issues.append(
                "cluster build Z != MlpotContext.ml_Z "
                f"(build {z_exp.tolist()[:8]}... vs ctx {z_ctx.tolist()[:8]}...)"
            )

    if not np.array_equal(z_psf, z_ctx):
        mismatch = np.where(z_psf != z_ctx)[0]
        issues.append(
            "PSF mass-derived Z != MlpotContext.ml_Z at indices "
            + ", ".join(str(int(i)) for i in mismatch[:12])
            + ("..." if mismatch.size > 12 else "")
        )
    if not np.array_equal(z_ctx, z_mlpot):
        issues.append("MlpotContext.ml_Z != mlpot.ml_Z (registration vs MLpot object)")

    expected_idx = np.arange(n_ml, dtype=int)
    if ml_idx.shape != expected_idx.shape or not np.array_equal(ml_idx, expected_idx):
        issues.append(
            f"mlpot.ml_indices not 0..{n_ml - 1} (got {ml_idx.tolist()[:16]}...)"
        )

    z_calc = _calculator_atomic_numbers(ctx)
    if z_calc is None:
        issues.append("MLpot calculator has no ml_atomic_numbers / atomic_numbers")
    elif z_calc.shape != z_mlpot.shape or not np.array_equal(z_calc, z_mlpot):
        issues.append("calculator atomic numbers != mlpot.ml_Z")

    z_model = _model_atomic_numbers(ctx.pyCModel)
    if z_model is not None:
        if z_model.shape != z_ctx.shape or not np.array_equal(z_model, z_ctx):
            issues.append("pyCModel._atomic_numbers != MlpotContext.ml_Z")

    issues.extend(_masses_consistent_with_z(masses, z_psf))

    if issues:
        raise RuntimeError(
            f"CHARMM/MLpot atom identity mismatch before {context}:\n  - "
            + "\n  - ".join(issues)
        )

    if not quiet:
        sample = min(3, n_psf)
        lines = [
            f"Atom consistency OK before {context}: N={n_psf} "
            f"(PSF ↔ MLpot ↔ calculator Z and masses)"
        ]
        for i in range(sample):
            lines.append(
                f"  atom {i}: type={atypes[i]} Z={int(z_psf[i])} "
                f"mass={float(masses[i]):.4f} amu"
            )
        print("\n".join(lines), flush=True)


def _import_pycharmm():
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401 — CHARMM env
    import pycharmm

    return pycharmm


def select_all_atoms():
    """CHARMM selection of all atoms."""
    return _import_pycharmm().SelectAtoms().all_atoms()


def select_by_seg_id(seg_id: str):
    """CHARMM selection by segment ID (e.g. ``'AMM1'`` for an ML region)."""
    return _import_pycharmm().SelectAtoms(seg_id=seg_id)


def select_by_resid(resid: int | str):
    """CHARMM selection by residue ID (e.g. ``1`` for the first residue)."""
    return _import_pycharmm().SelectAtoms(res_id=str(resid))


def select_by_resids(resids: Sequence[int | str]) -> Any:
    """Union selection over multiple residue IDs (one CGenFF residue = one monomer)."""
    ids = [str(r).strip() for r in resids if str(r).strip()]
    if not ids:
        raise ValueError("select_by_resids: empty resid list")
    sel = select_by_resid(ids[0])
    for rid in ids[1:]:
        sel = sel | select_by_resid(rid)
    return sel


def apply_charmm_verbosity(
    *,
    prnlev: int = 5,
    warnlev: int = 5,
    bomlev: int = -2,
) -> dict[str, int]:
    """Raise CHARMM console output (``PRNLev``, ``WRNLev``, ``BOMBlev``).

    Returns the previous levels as ``{"prnlev", "warnlev", "bomlev"}``.
    Higher ``prnlev`` / ``warnlev`` (up to ~5) print more from the Fortran core.
    """
    import pycharmm.settings as settings

    old = {
        "prnlev": int(settings.set_verbosity(int(prnlev))),
        "warnlev": int(settings.set_warn_level(int(warnlev))),
        "bomlev": int(settings.set_bomb_level(int(bomlev))),
    }
    return old


def write_charmm_psf(path: PathLike) -> Path:
    """Write the current in-memory PSF (connectivity as in CHARMM).

    Uses the Fortran ``write_psf_card`` C API with a lowercase staging path when
    needed.  ``WRITE PSF`` via ``lingo.charmm_script`` can abort in ``parse.F90``
    on MPI-linked builds (gfortrantmp EOF on unit 90) and cannot open paths with
    uppercase letters on many cluster CHARMM installs.
    """
    import ctypes

    from mmml.interfaces.pycharmmInterface.charmm_paths import charmm_fortran_path
    from pycharmm.charmm_file import c_api_path_buffer

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    fortran_path, alias = charmm_fortran_path(p, for_write=True)
    try:
        import pycharmm.lib as lib

        buf, fn_len = c_api_path_buffer(fortran_path)
        status = int(lib.charmm.write_psf_card(buf, ctypes.byref(fn_len)))
        if status != 1:
            raise RuntimeError(
                f"write_psf_card failed for {p} (staging={fortran_path!r}, status={status})"
            )
    finally:
        if alias is not None:
            alias.finalize()
    return p


def _write_vmd_pdb_from_positions(
    path: PathLike,
    positions: np.ndarray,
    *,
    title: str = "cluster",
) -> Path:
    """Write a VMD-compatible PDB without CHARMM ``WRITE COOR`` (MPI-safe).

    MPI-linked ``libcharmm.so`` under ``mpirun`` can abort in Fortran ``parse.F90``
    on ``WRITE COOR PDB`` immediately after ``WRITE PSF`` (gfortrantmp EOF on unit 90).
    ASE writes coordinates in PSF atom order using ``get_Z_from_psf()``.
    """
    import ase
    import ase.io

    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    numbers = np.asarray(get_Z_from_psf(), dtype=int)
    coords = np.asarray(positions, dtype=float)
    if coords.shape != (len(numbers), 3):
        raise ValueError(
            f"PDB write: positions shape {coords.shape} != ({len(numbers)}, 3)"
        )
    atoms = ase.Atoms(numbers=numbers, positions=coords)
    if title:
        atoms.info["title"] = str(title)
    ase.io.write(str(p), atoms, format="proteindatabank")
    return p


def write_charmm_crd_from_charmm(
    path: PathLike,
    *,
    title: str = "COORD",
    positions: np.ndarray | None = None,
) -> Path:
    """Write a CHARMM EXT CRD without ``WRITE COOR`` (MPI-safe).

    MPI-linked ``libcharmm.so`` under ``mpirun`` can abort in Fortran ``parse.F90``
    on ``WRITE COOR CARD`` (same gfortrantmp EOF failure as ``WRITE COOR PDB``).
    """
    import pycharmm.atom_info as atom_info
    import pycharmm.psf as psf

    if positions is not None:
        sync_charmm_positions(positions)

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    natom = int(psf.get_natom())
    if natom <= 0:
        raise RuntimeError(f"CRD write: PSF has no atoms ({p})")

    coords = (
        np.asarray(positions, dtype=float)
        if positions is not None
        else get_charmm_positions_array()
    )
    if coords.shape != (natom, 3):
        raise ValueError(
            f"CRD write: positions shape {coords.shape} != ({natom}, 3)"
        )

    atom_indices = list(range(natom))
    res_names = atom_info.get_res_names(atom_indices)
    res_ids = atom_info.get_res_ids(atom_indices)
    atypes = psf.get_atype()
    seg_ids = atom_info.get_seg_ids(atom_indices)

    lines: list[str] = []
    if title:
        lines.append(f"* {title}")
    lines.append("*")
    lines.append(f"        {natom}  EXT")
    for i in range(natom):
        iatom = i + 1
        try:
            ires = int(str(res_ids[i]).strip())
        except ValueError:
            ires = iatom
        x, y, z = (float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]))
        lines.append(
            f"{iatom:10d}{ires:10d}  {res_names[i]:<4}  {atypes[i]:<4}  "
            f"{x:20.10f}{y:20.10f}{z:20.10f}  {seg_ids[i]:<4}  {ires:10d}  "
            f"{1.0:20.10f}"
        )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def resolve_topology_psf_for_mlpot_reload(
    psf_path: PathLike,
    *,
    tag: str | None = None,
) -> Path:
    """Return a PSF safe for ``read.psf_card`` before MLpot re-registration.

    ``mini.psf`` / ``mini_full_mlpot_*.psf`` embed large ML–ML exclusion lists;
    CHARMM then aborts with "Maximum number of nonbond exclusions exceeded".
    Prefer ``model.psf`` (saved pre-MLpot).
    """
    from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import (
        resolve_topology_psf_candidates,
    )

    psf = Path(psf_path).expanduser().resolve()
    mini_like = psf.name == "mini.psf" or "mini_full_mlpot_" in psf.name
    if not mini_like:
        return psf

    for candidate in resolve_topology_psf_candidates(psf, tag=tag):
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot reload topology from {psf.name}: it contains MLpot nonbond "
        f"exclusions that exceed CHARMM PSF read limits. Provide "
        f"{psf.parent / 'model.psf'} "
        f"(from the initial build, pre-MLpot) via --from-psf and keep using the "
        f"mini CRD for coordinates."
    )


def save_cluster_topology_for_vmd(
    out_dir: PathLike,
    positions: np.ndarray,
    *,
    stem: str = "model",
    title: str = "cluster",
) -> dict[str, Path]:
    """Save PSF + PDB for VMD (connectivity preserved; MLpot uses zeroed CGENFF params).

    Load in VMD with: ``vmd model.psf model.pdb`` (or a trajectory).
    Also writes a composition fingerprint sidecar for safe inplace recovery.
    """
    import pycharmm.psf as psf

    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size
    from mmml.interfaces.pycharmmInterface.mlpot.topology_recovery import (
        capture_topology_fingerprint_from_charmm,
        save_topology_sidecar,
        topology_fingerprint_path,
    )
    from mmml.interfaces.pycharmmInterface.mpi_rank_io import is_mpi_rank_zero

    sync_charmm_positions(positions)
    out = Path(out_dir).expanduser().resolve()
    _rank, size = mpi_rank_size()
    rank0 = is_mpi_rank_zero() or size <= 1
    if rank0:
        out.mkdir(parents=True, exist_ok=True)
    psf_path = write_charmm_psf(out / f"{stem}.psf")
    pdb_path = out / f"{stem}.pdb"
    if rank0:
        _write_vmd_pdb_from_positions(pdb_path, positions, title=title)
    fingerprint = capture_topology_fingerprint_from_charmm()
    pre_iblo, pre_inb = psf.get_iblo_inb()
    if rank0:
        save_topology_sidecar(
            topology_fingerprint_path(psf_path),
            fingerprint,
            pre_mlpot_iblo=pre_iblo if pre_inb else None,
            pre_mlpot_inb=pre_inb if pre_inb else None,
        )
    return {"psf": psf_path, "pdb": pdb_path.resolve()}


def disable_charmm_domdec() -> None:
    """Turn off domdec once (``domdec dlb off`` would leave domdec on)."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import disable_charmm_domdec as _disable

    _disable()


def ensure_domdec_off_for_mlpot_energy(*, context: str = "MLpot energy") -> bool:
    """Single ``domdec off`` after JAX warmup, before MLpot SD / dynamics."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        ensure_domdec_off_for_mlpot_energy as _ensure,
    )

    return _ensure(context=context)


def prepare_charmm_vacuum() -> None:
    """Vacuum: crystal free (domdec off is deferred until MLpot SD/dynamics)."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import crystal_free_charmm

    crystal_free_charmm()


def setup_default_nbonds(*, nbxmod: int = 5) -> None:
    """Vacuum nonbonds (same kwargs as ``md_pbc_suite/ase._run_charmm_minimize``)."""
    from mmml.interfaces.pycharmmInterface.nbonds_config import apply_vacuum_nbonds

    apply_vacuum_nbonds(nbxmod=nbxmod)


def refresh_nbonds_after_mlpot(*, nbxmod: int = 5) -> None:
    """Rebuild nonbond lists after :class:`pycharmm.MLpot` changes exclusions."""
    from mmml.interfaces.pycharmmInterface.nbonds_config import vacuum_nbond_kwargs

    prepare_charmm_vacuum()
    pycharmm = _import_pycharmm()
    pycharmm.nbonds.update_bnbnd()
    pycharmm.UpdateNonBondedScript(**vacuum_nbond_kwargs(nbxmod=nbxmod)).run()


DEFAULT_WORKFLOW_NBXMOD = 5
RECOVERY_NBXMOD = 2
# NBXMod controls VDW/ELEC exclusion lists (CHARMM nbonds):
#   2 = exclude only 1-2 (bonded) pairs — milder exclusions during rescue SD
#   5 = exclude 1-2, 1-3, and special 1-4 (normal production MD)


def _is_all_ml_pbc_context(ctx: MlpotContext) -> bool:
    if not bool(getattr(ctx, "use_pbc", False)):
        return False
    ml_selection = getattr(ctx, "ml_selection", None)
    if ml_selection is None:
        return False
    try:
        n_ml = len(ml_selection.get_atom_indexes())
        n_total = int(_import_pycharmm().coor.get_natom())
    except Exception:
        return False
    return n_total > 0 and n_ml >= n_total


def apply_recovery_nbonds(ctx: MlpotContext, *, nbxmod: int = RECOVERY_NBXMOD) -> None:
    """Temporary nonbond settings for bonded rescue SD (``NBXMOD 2``, VDW on in BLOCK)."""
    from mmml.interfaces.pycharmmInterface.nbonds_config import vacuum_nbond_kwargs

    pycharmm = _import_pycharmm()
    if _is_all_ml_pbc_context(ctx):
        return
    pycharmm.nbonds.update_bnbnd()
    if ctx.use_pbc and ctx.cubic_box_side_A is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import apply_pbc_nbonds

        cuts = apply_pbc_nbonds(
            nbxmod=nbxmod,
            cubic_box_side_A=float(ctx.cubic_box_side_A),
        )
        pycharmm.UpdateNonBondedScript(**cuts.as_pbc_nbond_kwargs(nbxmod=nbxmod)).run()
    else:
        # Dynamics may leave imgfrq>0; clear before rescue SD (inbfrq=0 is invalid then).
        pycharmm.nbonds.set_imgfrq(-1)
        pycharmm.UpdateNonBondedScript(**vacuum_nbond_kwargs(nbxmod=nbxmod)).run()


def restore_workflow_nbonds(
    ctx: MlpotContext,
    *,
    nbxmod: int = DEFAULT_WORKFLOW_NBXMOD,
) -> None:
    """No-op after overlap rescue when MLpot is (re-)registered.

    Rescue minimization temporarily uses ``NBXMOD 2`` via
    :func:`apply_recovery_nbonds`. Switching back to production ``NBXMOD`` rebuilds
    exclusion lists through ``update_bnbnd`` / ``UpdateNonBondedScript`` → ``upinb``,
    which segfaults once MLpot has established ML exclusions (even after
    ``unset_mlpot``). Hybrid MLpot MD keeps CHARMM VDW/ELEC off on ML atoms via
    BLOCK, so staying on ``NBXMOD 2`` after rescue is safe.
    """
    del ctx, nbxmod  # API kept for callers; intentionally no CHARMM nbond rebuild.


def refresh_nbonds_after_mlpot_pbc(
    *,
    cubic_box_side_A: float,
    nbxmod: int = 5,
    cutnb: float = 18.0,
    force: bool = False,
) -> None:
    """Rebuild PBC nonbond lists after MLpot registration.

    With ``force=False`` (default), skip when the live CHARMM box already matches
    ``cubic_box_side_A``, or when box lengths are unavailable — rebuilding
    crystal/nbonds with MLpot active mid-workflow can segfault in ``upinb``.
    Pass ``force=True`` once immediately after initial MLpot registration.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        _is_cubic_box_sides,
        _read_charmm_box_sides_A,
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )

    side = float(cubic_box_side_A)
    if not force:
        try:
            lx, ly, lz = _read_charmm_box_sides_A()
            if _is_cubic_box_sides(lx, ly, lz):
                mean = (lx + ly + lz) / 3.0
                tol = max(1e-3, 1e-4 * side)
                if abs(mean - side) <= tol:
                    return
            if min(lx, ly, lz) <= 0.0:
                return
        except Exception:
            return

    pycharmm = _import_pycharmm()
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    with charmm_relaxed_bomlev():
        prepare_charmm_pbc(side)
        if force:
            pycharmm.nbonds.update_bnbnd()
        apply_pbc_nbonds(nbxmod=nbxmod, cubic_box_side_A=side)


def report_charmm_topology_summary(*, quiet: bool = False) -> bool:
    """Emit Rich PSF summary when CHARMM topology is loaded in memory."""
    from mmml.utils.rich_report import emit_charmm_topology_summary

    return emit_charmm_topology_summary(quiet=quiet)


def reconcile_n_monomers_with_psf(
    args: Any,
    z: np.ndarray,
    n_mol: int,
) -> tuple[int, list[int] | None]:
    """Use CHARMM PSF residue boundaries when CLI monomer count disagrees with topology."""
    n_atoms = int(len(z))
    if n_atoms <= 0:
        return int(n_mol), getattr(args, "_cluster_atoms_per_list", None)

    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        from mmml.interfaces.pycharmmInterface.mlpot.trimer_scan import (
            atoms_per_monomer_from_psf,
        )

        atoms_per = atoms_per_monomer_from_psf()
    except Exception:
        return int(n_mol), getattr(args, "_cluster_atoms_per_list", None)

    if sum(atoms_per) != n_atoms:
        return int(n_mol), getattr(args, "_cluster_atoms_per_list", None)

    n_psf = len(atoms_per)
    n_mol = int(n_mol)
    existing: list[int] | None = getattr(args, "_cluster_atoms_per_list", None)

    if n_psf == n_mol and n_atoms % max(n_mol, 1) == 0:
        apm = list(existing) if existing is not None and len(existing) == n_mol else atoms_per
        if existing is None:
            setattr(args, "_cluster_atoms_per_list", list(apm))
        return n_mol, apm

    if n_psf != n_mol or n_atoms % max(n_mol, 1) != 0:
        if not getattr(args, "quiet", False):
            per = atoms_per[0] if atoms_per else 0
            print(
                f"Cluster monomer count: CLI n_monomers={n_mol} -> PSF nres={n_psf} "
                f"({n_atoms} atoms, {per} atoms/monomer typical)",
                flush=True,
            )
        setattr(args, "_cluster_atoms_per_list", list(atoms_per))
        comp = getattr(args, "_cluster_composition_summary", None)
        if not comp and n_psf > 0:
            residue = str(getattr(args, "residue", "MOL")).upper()
            setattr(args, "_cluster_composition_summary", {residue: n_psf})
        return n_psf, atoms_per

    return n_mol, existing


def load_cluster_from_artifacts(
    args: Any,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Load PSF + CRD (and optional coordinates from ``--restart-from``)."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import load_minimized_coordinates
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar
    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

    psf = getattr(args, "from_psf", None)
    crd = getattr(args, "from_crd", None)
    if (psf is None or crd is None) and getattr(args, "output_dir", None):
        out = Path(args.output_dir).expanduser().resolve()
        tag_guess = getattr(args, "tag", None)
        if not tag_guess and getattr(args, "composition", None):
            from mmml.cli.run.md_pbc_suite.ase import _parse_composition
            from mmml.interfaces.pycharmmInterface.mlpot.cli_common import composition_tag

            comp = _parse_composition(args.composition)
            n_from_comp = sum(c for _, c in comp)
            tag_guess = composition_tag(
                comp,
                str(getattr(args, "residue", "ACO")).upper(),
                n_from_comp,
            )
        if tag_guess:
            from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import mini_paths

            mini = mini_paths(out)
            psf = psf or mini["mini_psf"]
            crd = crd or mini["mini_crd"]
        elif psf is None and crd is None:
            mini_psf = out / "mini.psf"
            if mini_psf.is_file():
                psf = mini_psf
                crd = out / "mini.crd"
            else:
                psf_candidates = sorted(out.glob("mini_full_mlpot_*.psf"))
                if len(psf_candidates) == 1:
                    psf = psf_candidates[0]
                    crd = out / psf.name.replace(".psf", ".crd")
    if psf is None or crd is None:
        raise ValueError(
            "skip-cluster-build requires --from-psf and --from-crd "
            "(or mini artifacts under --output-dir with --tag)"
        )
    psf_path = Path(psf).expanduser().resolve()
    crd_path = Path(crd).expanduser().resolve()
    if not psf_path.is_file():
        raise FileNotFoundError(f"PSF not found: {psf_path}")
    if not crd_path.is_file():
        raise FileNotFoundError(f"CRD not found: {crd_path}")

    pycharmm = _import_pycharmm()
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    tag_guess = str(
        getattr(args, "tag", None)
        or (psf_path.stem.replace("mini_full_mlpot_", "") if "mini_full" in psf_path.name else "")
    )
    topology_psf = resolve_topology_psf_for_mlpot_reload(psf_path, tag=tag_guess)
    if topology_psf != psf_path and not getattr(args, "quiet", False):
        print(
            f"Reload topology from {topology_psf.name} "
            f"(not {psf_path.name}; mini PSF embeds ML exclusions)",
            flush=True,
        )

    read_cgenff_toppar()
    with charmm_relaxed_bomlev():
        read.psf_card(str(topology_psf))
        load_minimized_coordinates(crd_path)
    z = np.asarray(get_Z_from_psf(), dtype=int)
    r = get_charmm_positions_array()

    n_mol = int(getattr(args, "n_molecules", 0) or 0)
    if getattr(args, "composition", None):
        from mmml.cli.run.md_pbc_suite.ase import _parse_composition

        n_mol = sum(c for _, c in _parse_composition(args.composition))
    if n_mol <= 0:
        n_mol = max(1, int(getattr(args, "n_molecules", 1) or 1))

    tag = str(getattr(args, "tag", None) or psf_path.stem.replace("mini_full_mlpot_", ""))
    if not getattr(args, "quiet", False):
        report_charmm_topology_summary()
    n_mol, _ = reconcile_n_monomers_with_psf(args, z, n_mol)
    return z, r, n_mol, tag


def physnet_ml_atomic_numbers(z: Sequence[int]) -> list[int]:
    """PSF/ASE atomic numbers for MLpot (must match ``setup_calculator`` inputs)."""
    return [int(x) for x in z]


def load_physnet_mlpot_bundle(
    checkpoint: PathLike,
    n_atoms: int,
    ase_atoms: Any,
    *,
    n_monomers: int = 1,
    atoms_per_monomer: Sequence[int] | None = None,
    ml_batch_size: Optional[int] = None,
    ml_gpu_count: Optional[int] = None,
    ml_max_active_dimers: Optional[int] = None,
    cell: float | None = None,
    verbose: bool = False,
    args: Any | None = None,
    defer_jax_until_after_sd: bool = False,
) -> tuple[Any, Any, Any]:
    """Load PhysNet for MLpot. Multi-monomer clusters use monomer/dimer batches."""
    ckpt = Path(checkpoint).expanduser().resolve()
    z = np.asarray(ase_atoms.get_atomic_numbers(), dtype=int)

    if int(n_monomers) > 1:
        from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
            build_decomposed_mlpot_model,
        )

        if atoms_per_monomer is None:
            if int(n_atoms) % int(n_monomers) != 0:
                nres_hint = ""
                try:
                    import pycharmm.psf as psf

                    if hasattr(psf, "get_nres"):
                        nres_hint = f"; PSF has {int(psf.get_nres())} residues"
                except Exception:
                    pass
                raise ValueError(
                    f"atom count {n_atoms} not divisible by n_monomers={n_monomers}{nres_hint}. "
                    "Align --composition or --n-molecules with the loaded PSF/box."
                )
            per = int(n_atoms) // int(n_monomers)
            atoms_per_monomer = [per] * int(n_monomers)
        pyCModel = build_decomposed_mlpot_model(
            ckpt,
            z,
            atoms_per_monomer,
            int(n_monomers),
            ml_batch_size=ml_batch_size,
            ml_gpu_count=ml_gpu_count,
            ml_max_active_dimers=ml_max_active_dimers,
            cell=float(cell) if cell is not None else False,
            verbose=verbose,
            args=args,
            defer_jax_until_mlpot_registered=True,
            defer_jax_until_after_sd=defer_jax_until_after_sd,
        )
        return None, None, pyCModel

    from mmml.cli.base import load_physnet_params_and_ef_model
    from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_pyc

    params, model = load_physnet_params_and_ef_model(ckpt, natoms=n_atoms)
    model.natoms = n_atoms
    pyCModel = get_pyc(params, model, ase_atoms)
    return params, model, pyCModel


def _build_ml_exclusion_lists(
    ml_indices: Sequence[int],
    *,
    natom: int,
) -> tuple[np.ndarray, list[int]]:
    """PSF ``iblo``/``inb`` arrays for all ML–ML pair exclusions (1-based ``inb``)."""
    ml_idx = np.asarray(ml_indices, dtype=int)
    n_ml = int(ml_idx.size)
    ml_iblo = np.zeros(int(natom), dtype=int)
    ml_inb: list[int] = []
    for ii, idx in enumerate(ml_idx):
        ml_iblo[idx:] += n_ml - ii - 1
        for jdx in ml_idx[(ii + 1) :]:
            ml_inb.append(int(jdx) + 1)
    return ml_iblo, ml_inb


def _install_ml_exclusions(ml_selection: Any, *, update: bool = True) -> None:
    """Add ML–ML exclusions; optionally rebuild CHARMM nonbond lists (``upinb``)."""
    pycharmm = _import_pycharmm()
    natom = int(pycharmm.coor.get_natom())
    ml_indices = ml_selection.get_atom_indexes()
    ml_iblo, ml_inb = _build_ml_exclusion_lists(ml_indices, natom=natom)
    if update:
        pycharmm.psf.set_iblo_inb(ml_iblo, ml_inb)
    else:
        pycharmm.psf.set_iblo_inb_no_update(ml_iblo, ml_inb)


def _registration_pbc_box_side_A(
    cubic_box_side_A: float | None,
    budget_box: float | None,
) -> float:
    if cubic_box_side_A is not None:
        return float(cubic_box_side_A)
    if budget_box is not None:
        return float(budget_box)
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        _is_cubic_box_sides,
        _read_charmm_box_sides_A,
    )

    lx, ly, lz = _read_charmm_box_sides_A()
    if _is_cubic_box_sides(lx, ly, lz):
        return float((lx + ly + lz) / 3.0)
    raise RuntimeError(
        "PBC MLpot registration needs a cubic box side "
        "(pass cubic_box_side_A or ensure CHARMM crystal is set)"
    )


def _resolve_mlpot_ctx_pbc_box_side(ctx: MlpotContext) -> float | None:
    """Best-effort cubic box side (Å) from context fields or live CHARMM crystal."""
    if not bool(getattr(ctx, "use_pbc", False)):
        return None
    for attr in ("cubic_box_side_A", "charmm_cubic_box_side_A"):
        val = getattr(ctx, attr, None)
        if val is not None:
            return float(val)
    try:
        return _registration_pbc_box_side_A(None, None)
    except RuntimeError:
        return None


def _apply_mlpot_psf_mm_off_and_pbc(ctx: MlpotContext, *, verbose: bool = False) -> str:
    """Zero CHARMM MM on ML atoms; rebuild PBC lists after CGENFF param read."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
        apply_mlpot_registration_mm_off,
    )

    if bool(getattr(ctx, "use_pbc", False)) and not bool(
        getattr(ctx, "registration_uses_block", False)
    ):
        _suspend_pbc_for_cgenff_param_read(verbose=verbose)
    block_tag = apply_mlpot_registration_mm_off(
        ctx.ml_selection,
        mm_internal_scale=float(ctx.mm_internal_scale),
        verbose=verbose,
        periodic_external=bool(getattr(ctx, "periodic_external", False)),
        use_block=ctx.registration_uses_block,
    )
    if bool(getattr(ctx, "use_pbc", False)) and not bool(
        getattr(ctx, "registration_uses_block", False)
    ):
        box_side = _resolve_mlpot_ctx_pbc_box_side(ctx)
        if box_side is not None:
            _finalize_pbc_mlpot_exclusions_after_param_read(
                ctx.ml_selection,
                cubic_box_side_A=float(box_side),
                verbose=verbose,
            )
    return block_tag


def _suspend_pbc_for_cgenff_param_read(*, verbose: bool = False) -> None:
    """Drop IMAGE transforms before ``READ PARAM APPEND`` during PBC MLpot registration.

    ``read_param_file`` calls ``climag`` (clears image PSF tables) then ``gtnbct INIT``,
    which runs ``upinb`` → ``UPIMNB`` when ``NTRANS > 0``. With images cleared but
    ``NTRANS`` still set (e.g. DCM:100 after lattice prep), ``UPIMNB`` segfaults.
    ``crystal free`` sets ``NTRANS=0`` so the param-read ``upinb`` pass is vacuum-safe;
    :func:`_finalize_pbc_mlpot_exclusions_after_param_read` restores crystal/nb lists.

    Also invoked from :func:`read_cgenff_prm` (append); this wrapper keeps the
    registration-time log line when ``verbose=True``.
    """
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        suspend_pbc_before_cgenff_param_append,
    )

    if not suspend_pbc_before_cgenff_param_append():
        return
    if verbose:
        print(
            "MLpot PBC: crystal free before zeroed CGENFF read "
            "(READ PARAM APPEND clears IMAGE; avoids UPIMNB segfault)",
            flush=True,
        )


def _finalize_pbc_mlpot_exclusions_after_param_read(
    ml_selection: Any,
    *,
    cubic_box_side_A: float,
    verbose: bool = False,
) -> None:
    """Rebuild crystal/nb lists after READ PARAM, then apply ML exclusions once.

    ``read_param_file`` (append or not) always clears NONBOND/HBOND lists and IMAGE
    atoms (``api_read.F90``). Installing ML exclusions via ``set_iblo_inb`` before
    rebuilding PBC leaves ``upinb`` operating on a cleared image table → segfault.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        restore_charmm_cubic_crystal_lattice,
    )

    restore_charmm_cubic_crystal_lattice(
        float(cubic_box_side_A),
        quiet=not verbose,
    )
    _install_ml_exclusions(ml_selection, update=False)
    pycharmm = _import_pycharmm()
    pycharmm.nbonds.update_bnbnd()
    pycharmm.image.update_bimag()
    if verbose:
        print(
            "MLpot PBC: rebuilt crystal/nb lists after CGENFF param read "
            f"(L={float(cubic_box_side_A):.3f} Å)",
            flush=True,
        )


def _require_mlpot_skip_iblo_support(pycharmm: Any) -> None:
    """PBC all-ML registration needs ``skip_iblo_inb_update`` on vendored PyCHARMM."""
    try:
        params = inspect.signature(pycharmm.MLpot.__init__).parameters
    except (TypeError, ValueError):
        params = {}
    if "skip_iblo_inb_update" in params:
        return
    ml_module = getattr(pycharmm.MLpot, "__module__", "unknown")
    raise RuntimeError(
        "PyCHARMM MLpot lacks skip_iblo_inb_update "
        f"(loaded from {ml_module}). Reinstall mmml from this repo (`uv sync`) so "
        "import_pycharmm prefers the vendored pycharmm package over "
        "$CHARMM_HOME/tool/pycharmm. PBC all-ML registration runs upinb after BLOCK "
        "DELTIC without it and segfaults in __nbexcl_MOD_upinb. Also launch under "
        "scripts/mmml-charmm-mpirun.sh for MPI-linked libcharmm.so."
    )


def register_mlpot(
    pyCModel: Any,
    ml_Z: Sequence[int],
    ml_selection: Any,
    *,
    ml_charge: float = 0,
    ml_fq: bool = True,
    mlmm_ctonnb: Optional[float] = None,
    mlmm_ctofnb: Optional[float] = None,
    preserve_psf_internals: bool = True,
    use_pbc: bool = False,
    mm_internal_scale: float = 0.0,
    cubic_box_side_A: float | None = None,
    mm_nonbond_mode: str = "jax_mic",
    periodic_charmm_vdw: bool = True,
    verbose: bool = False,
    use_block_registration: bool | None = None,
    **kwargs: Any,
) -> MlpotContext:
    """Register ``pycharmm.MLpot`` and return a context manager-like handle."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
        apply_mlpot_registration_mm_off,
        mlpot_use_block_registration,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import validate_mlpot_system_size
    from mmml.interfaces.pycharmmInterface.mlpot.periodic_mm import resolve_mm_nonbond_mode

    periodic_external = (
        resolve_mm_nonbond_mode(type("_Args", (), {"mm_nonbond_mode": mm_nonbond_mode})())
        == "periodic_external"
    )

    pycharmm = _import_pycharmm()
    z_ml = physnet_ml_atomic_numbers(ml_Z)
    n_ml = len(ml_selection.get_atom_indexes())
    budget_box = None
    if use_pbc:
        from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import (
            mlpot_limits_status,
            pbc_image_copies_per_atom,
            pbc_pair_budget_box_side_A,
            required_max_npr,
        )

        budget_box = pbc_pair_budget_box_side_A(n_ml, cubic_box_side_A)
        if n_ml >= 500:
            status = mlpot_limits_status()
            need = required_max_npr(n_ml, pbc=True, box_side_A=budget_box)
            copies = pbc_image_copies_per_atom(n_ml, budget_box)
            box_note = ""
            if cubic_box_side_A is not None:
                box_note = f" L={float(cubic_box_side_A):g} Å"
                if budget_box is None:
                    box_note += (
                        " (pair budget uses n_ml baseline; box-aware tier too large)"
                    )
                else:
                    box_note += f" ~{copies:.1f}× image copies"
            print(
                f"MLpot registration: n_ml={n_ml} PBC needs max_Npr>={need};"
                f"{box_note}; loaded lib max_Npr={status.max_npr} ({status.source})",
                flush=True,
            )
    validate_mlpot_system_size(
        n_ml, pbc=bool(use_pbc), box_side_A=budget_box
    )
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.nbonds_config import CGENFF_PRM_BOMLEV

    with charmm_relaxed_bomlev(CGENFF_PRM_BOMLEV):
        skip_iblo_inb_update = False
        uses_block = mlpot_use_block_registration(explicit=use_block_registration)
        if use_pbc:
            _require_mlpot_skip_iblo_support(pycharmm)
            if not uses_block:
                _suspend_pbc_for_cgenff_param_read(verbose=verbose)
        if periodic_external and periodic_charmm_vdw:
            block_tag = apply_mlpot_registration_mm_off(
                ml_selection,
                mm_internal_scale=float(mm_internal_scale),
                verbose=verbose,
                periodic_external=True,
                use_block=use_block_registration,
            )
        else:
            block_tag = apply_mlpot_registration_mm_off(
                ml_selection,
                mm_internal_scale=float(mm_internal_scale),
                verbose=verbose,
                periodic_external=periodic_external,
                use_block=use_block_registration,
            )
        if use_pbc:
            box_side = _registration_pbc_box_side_A(cubic_box_side_A, budget_box)
            if uses_block:
                _install_ml_exclusions(ml_selection)
            else:
                _finalize_pbc_mlpot_exclusions_after_param_read(
                    ml_selection,
                    cubic_box_side_A=box_side,
                    verbose=verbose,
                )
            skip_iblo_inb_update = True
        mlpot = pycharmm.MLpot(
            ml_model=pyCModel,
            ml_Z=z_ml,
            ml_selection=ml_selection,
            ml_charge=ml_charge,
            ml_fq=ml_fq,
            mlmm_ctonnb=mlmm_ctonnb,
            mlmm_ctofnb=mlmm_ctofnb,
            preserve_psf_internals=preserve_psf_internals,
            skip_iblo_inb_update=skip_iblo_inb_update,
            **kwargs,
        )
        if not use_pbc:
            # MLpot.__init__ already set iblo/inb and ran update_bnbnd (upinb).
            # Re-running prepare_charmm_vacuum + update_bnbnd here segfaults in upinb
            # for large clusters (e.g. DCM:90) after JAX GPU warmup.
            from mmml.interfaces.pycharmmInterface.nbonds_config import (
                vacuum_nbond_kwargs,
            )

            pycharmm.UpdateNonBondedScript(**vacuum_nbond_kwargs(nbxmod=5)).run()
    ml_z = np.asarray(ml_Z, dtype=int)
    reg_box = (
        float(cubic_box_side_A)
        if use_pbc and cubic_box_side_A is not None
        else (float(budget_box) if use_pbc and budget_box is not None else None)
    )
    return MlpotContext(
        mlpot=mlpot,
        pyCModel=pyCModel,
        params=None,
        model=None,
        ml_selection=ml_selection,
        block_tag=block_tag,
        ml_Z=ml_z,
        use_pbc=bool(use_pbc),
        cubic_box_side_A=reg_box,
        ml_charge=float(ml_charge),
        ml_fq=bool(ml_fq),
        mm_internal_scale=float(mm_internal_scale),
        registration_uses_block=uses_block,
        periodic_external=bool(periodic_external),
        periodic_charmm_vdw=bool(periodic_charmm_vdw),
    )
