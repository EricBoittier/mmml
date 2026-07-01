"""CHARMM periodic box setup for MLpot workflows (matches ``md_pbc_suite/ase.py``)."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np

PathLike = Union[str, Path]
BoxSideSource = Literal["pbound", "xucell", "restart", "fallback"]


def _parse_fortran_float(token: str) -> float:
    """Parse CHARMM restart floats such as ``0.310000000000000D+02``."""
    return float(token.strip().upper().replace("D", "E"))


def parse_cubic_box_side_from_charmm_restart(path: PathLike) -> float | None:
    """Read cubic box side (Å) from a CHARMM dynamics restart (``.res``).

    CPT/NPT restarts store ``!CRYSTAL PARAMETERS`` after the title block.
    Returns ``None`` when the file has no crystal section (e.g. vacuum NVE).
    """
    p = Path(path)
    if not p.is_file():
        return None
    lines = p.read_text(encoding="ascii", errors="ignore").splitlines()
    if not lines:
        return None

    header = lines[0].upper()
    has_cubi = "CUBI" in header
    crystal_idx = next(
        (i for i, ln in enumerate(lines[:40]) if "CRYSTAL PARAMETERS" in ln.upper()),
        None,
    )
    if crystal_idx is None and not has_cubi:
        return None

    if crystal_idx is None:
        return None

    values: list[float] = []
    for j in range(crystal_idx + 1, min(crystal_idx + 4, len(lines))):
        for token in lines[j].split():
            try:
                values.append(_parse_fortran_float(token))
            except ValueError:
                continue

    positives = [v for v in values if v > 1.0]
    if not positives:
        return None

    if has_cubi:
        return float(positives[0])

    if len(values) >= 9:
        mat = np.array(values[:9], dtype=float).reshape(3, 3)
        lengths = [
            float(np.linalg.norm(mat[k]))
            for k in range(3)
            if float(np.linalg.norm(mat[k])) > 1.0
        ]
        if lengths:
            if _is_cubic_box_sides(
                lengths[0],
                lengths[1] if len(lengths) > 1 else lengths[0],
                lengths[2] if len(lengths) > 2 else lengths[0],
            ):
                return sum(lengths) / len(lengths)

    rounded = {round(v, 4) for v in positives}
    if len(rounded) == 1:
        return float(next(iter(rounded)))
    return float(positives[0])


def _charmm_ctypes_scalar(value: Any) -> float:
    """Normalize CHARMM ctypes outputs (may be bare int/float or ctypes wrappers)."""
    if hasattr(value, "value"):
        return float(value.value)
    return float(value)


def cubic_box_length_from_geometry(
    positions: np.ndarray,
    *,
    ml_cutoff: float = 12.0,
    pad: float = 10.0,
) -> float:
    """Auto cubic box side (Å) from cluster span + padding."""
    r = np.asarray(positions, dtype=float)
    span = float(np.max(r.max(axis=0) - r.min(axis=0)))
    return max(span + 2.0 * float(pad), 2.0 * float(ml_cutoff) + float(pad))


def _read_charmm_box_sides_A() -> tuple[float, float, float]:
    """Read CHARMM periodic box lengths (Å) via ``pbound_get_size``."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401 — CHARMM env
    import pycharmm.lib as lib

    size_x = ctypes.c_double(0.0)
    size_y = ctypes.c_double(0.0)
    size_z = ctypes.c_double(0.0)
    lib.charmm.pbound_get_size(
        ctypes.byref(size_x),
        ctypes.byref(size_y),
        ctypes.byref(size_z),
    )
    return (
        _charmm_ctypes_scalar(size_x),
        _charmm_ctypes_scalar(size_y),
        _charmm_ctypes_scalar(size_z),
    )


def _read_charmm_ucell_lengths_A() -> tuple[float, float, float]:
    """Read unit-cell edge lengths (Å) from ``image_get_ucell`` (``XUCELL``)."""
    import pycharmm.image as image

    ucell = image.get_ucell()
    return float(ucell[0]), float(ucell[1]), float(ucell[2])


def _is_cubic_box_sides(
    lx: float,
    ly: float,
    lz: float,
    *,
    rel_tol: float = 1e-3,
) -> bool:
    if min(lx, ly, lz) <= 0.0:
        return False
    mean = (lx + ly + lz) / 3.0
    return max(abs(lx - mean), abs(ly - mean), abs(lz - mean)) <= rel_tol * mean


def resolve_charmm_cubic_box_side_A(
    *,
    fallback_side_A: float | None = None,
    restart_path: PathLike | None = None,
    rel_tol: float = 1e-3,
) -> tuple[float, BoxSideSource]:
    """Return ``(side, source)`` for the current CHARMM cubic cell.

    Resolution order:

    1. Live ``pbound_get_size`` (during/after CPT when crystal is active)
    2. ``image_get_ucell`` edge lengths (``XUCELL``; often valid when pbound is zero)
    3. ``!CRYSTAL PARAMETERS`` in a CHARMM restart file (before CPT read restores PBC)
    4. ``fallback_side_A`` (last known ML MIC cell or CLI ``--box-size``)
    """
    lx, ly, lz = _read_charmm_box_sides_A()
    if _is_cubic_box_sides(lx, ly, lz, rel_tol=rel_tol):
        return (lx + ly + lz) / 3.0, "pbound"

    try:
        ux, uy, uz = _read_charmm_ucell_lengths_A()
        if _is_cubic_box_sides(ux, uy, uz, rel_tol=rel_tol):
            return (ux + uy + uz) / 3.0, "xucell"
    except Exception:
        pass

    if restart_path is not None:
        from_restart = parse_cubic_box_side_from_charmm_restart(restart_path)
        if from_restart is not None and from_restart > 0.0:
            return float(from_restart), "restart"

    if fallback_side_A is not None and float(fallback_side_A) > 0.0:
        return float(fallback_side_A), "fallback"

    if min(lx, ly, lz) > 0.0:
        raise RuntimeError(
            "CHARMM box is not cubic; MLpot MIC sync expects a cubic cell "
            f"(got {lx:.4f}, {ly:.4f}, {lz:.4f} Å)"
        )
    raise RuntimeError(
        "CHARMM cubic box side unavailable (pbound_get_size returned non-positive "
        "lengths and no restart/fallback box side was provided)"
    )


def probe_charmm_cubic_box_side_A(
    *,
    fallback_side_A: float | None = None,
    restart_path: PathLike | None = None,
    rel_tol: float = 1e-3,
) -> tuple[float | None, BoxSideSource | None]:
    """Non-raising box probe; returns ``(None, None)`` when resolution fails."""
    try:
        side, source = resolve_charmm_cubic_box_side_A(
            fallback_side_A=fallback_side_A,
            restart_path=restart_path,
            rel_tol=rel_tol,
        )
        return float(side), source
    except RuntimeError:
        return None, None


def resolve_mlpot_mic_box_side_A(
    *,
    fallback_side_A: float | None = None,
    restart_path: PathLike | None = None,
    rel_tol: float = 1e-3,
) -> tuple[float, BoxSideSource]:
    """Resolve MIC cell side: live pbound when crystal is active, else restart/fallback."""
    restart_for_resolve = restart_path
    try:
        if charmm_crystal_is_active(rel_tol=rel_tol):
            restart_for_resolve = None
    except Exception:
        pass
    return resolve_charmm_cubic_box_side_A(
        fallback_side_A=fallback_side_A,
        restart_path=restart_for_resolve,
        rel_tol=rel_tol,
    )


def get_charmm_cubic_box_side_A(
    *,
    restart_path: PathLike | None = None,
    fallback_side_A: float | None = None,
) -> float:
    """Read the current cubic CHARMM cell side length (Å)."""
    side, _ = resolve_charmm_cubic_box_side_A(
        fallback_side_A=fallback_side_A,
        restart_path=restart_path,
    )
    return side


def push_charmm_cubic_box_side_A(
    target_side_A: float,
    *,
    nbxmod: int = 5,
    quiet: bool = False,
) -> tuple[float, BoxSideSource]:
    """Set CHARMM cubic box via C API (``crystal_define_cubic`` + ``build``).

    Safe before MLpot registration. Rebuilds IMAGE centering and PBC nbonds.
    Skips work when the live CHARMM box already matches ``target_side_A`` and
    the crystal metric is lattice-ready (``pbound`` active, not ``xucell``-only).
    """
    target = float(target_side_A)
    if target <= 0.0:
        raise ValueError(f"target box side must be > 0, got {target}")
    tol = max(1e-3, 1e-4 * target)
    live, source = probe_charmm_cubic_box_side_A(fallback_side_A=target)
    if live is not None and abs(live - target) <= tol:
        if charmm_crystal_lattice_ready():
            return float(live), source or "pbound"
        restore_charmm_cubic_crystal_lattice(target, nbxmod=nbxmod, quiet=quiet)
        side, out_source = resolve_charmm_cubic_box_side_A(fallback_side_A=target)
        return float(side), out_source
    prepare_charmm_pbc(target)
    apply_pbc_nbonds(nbxmod=int(nbxmod), cubic_box_side_A=target)
    side, out_source = resolve_charmm_cubic_box_side_A(fallback_side_A=target)
    if not quiet:
        print(
            f"CHARMM box set via crystal C API: L={side:.3f} Å (source={out_source})",
            flush=True,
        )
    return float(side), out_source


def sync_charmm_box_from_workflow_side(
    workflow_side_A: float,
    *,
    nbxmod: int = 5,
    quiet: bool = False,
) -> tuple[float, BoxSideSource]:
    """Push workflow box side into CHARMM when it differs from the live cell."""
    return push_charmm_cubic_box_side_A(
        float(workflow_side_A),
        nbxmod=nbxmod,
        quiet=quiet,
    )


def cubic_box_matrix_from_side(side_A: float) -> np.ndarray:
    """Return orthorhombic 3×3 cell matrix for a cubic box side (Å)."""
    L = float(side_A)
    return np.array([[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]], dtype=np.float64)


def _charmm_image_ntrans() -> int:
    """IMAGE transform count (0/1 = vacuum)."""
    import pycharmm.image as image

    return int(image.get_ntrans())


def charmm_crystal_is_active(*, rel_tol: float = 1e-3) -> bool:
    """True when CHARMM reports a positive cubic periodic box (crystal + IMAGE)."""
    try:
        lx, ly, lz = _read_charmm_box_sides_A()
        if _is_cubic_box_sides(lx, ly, lz, rel_tol=rel_tol) and min(lx, ly, lz) > 1.0:
            return True
        ux, uy, uz = _read_charmm_ucell_lengths_A()
        return _is_cubic_box_sides(ux, uy, uz, rel_tol=rel_tol) and min(ux, uy, uz) > 1.0
    except Exception:
        return False


def charmm_crystal_lattice_ready(*, rel_tol: float = 1e-3) -> bool:
    """True when CRYSTAL/IMAGE can run lattice minimization or CPT.

    Requires live ``pbound_get_size`` and IMAGE transforms. ``image_get_ucell``
    alone is insufficient: restart/MC handoffs can leave ``XUCELL`` populated
    while ``XTLABC`` is zero (singular metric matrix in ``MBUILD``).
    """
    try:
        if _charmm_image_ntrans() <= 1:
            return False
        lx, ly, lz = _read_charmm_box_sides_A()
        return _is_cubic_box_sides(lx, ly, lz, rel_tol=rel_tol) and min(lx, ly, lz) > 1.0
    except Exception:
        return False


def _ucell_side_matches_expected(
    expected_side_A: float | None,
    *,
    rel_tol: float = 1e-3,
) -> bool:
    """True when ``image_get_ucell`` reports a cubic cell at ``expected_side_A``."""
    if expected_side_A is None:
        return True
    try:
        ux, uy, uz = _read_charmm_ucell_lengths_A()
        if not _is_cubic_box_sides(ux, uy, uz, rel_tol=rel_tol) or min(ux, uy, uz) <= 1.0:
            return False
        mean = (ux + uy + uz) / 3.0
        tol = max(1e-3, rel_tol * float(expected_side_A))
        return abs(mean - float(expected_side_A)) <= tol
    except Exception:
        return False


def charmm_crystal_abnr_ready(
    expected_side_A: float | None = None,
    *,
    rel_tol: float = 1e-3,
) -> bool:
    """True when lattice ABNR can run after a fresh crystal reinstall.

    Same strict ``pbound`` gate as :func:`charmm_crystal_lattice_ready` when the
    C API reports a cubic cell.  After ``prepare_charmm_pbc`` on KEY_LIBRARY /
    MPI builds, ``pbound_get_size`` can stay zero in Python while ``XUCELL`` and
    IMAGE transforms are valid — accept that post-reinstall state when
    ``expected_side_A`` matches ``image_get_ucell``.

    ``image_get_ntrans`` can fail or report vacuum (``NTRANS+1 == 1``) even after
    a successful ``crystal build``; do not treat those probe failures as hard
    vacuum when ``XUCELL`` matches the workflow box.
    """
    if charmm_crystal_lattice_ready(rel_tol=rel_tol):
        if expected_side_A is None:
            return True
        try:
            live, _ = resolve_charmm_cubic_box_side_A(
                fallback_side_A=float(expected_side_A),
                rel_tol=rel_tol,
            )
            tol = max(1e-3, rel_tol * float(expected_side_A))
            return abs(float(live) - float(expected_side_A)) <= tol
        except Exception:
            return True

    if not _ucell_side_matches_expected(expected_side_A, rel_tol=rel_tol):
        return False

    ntrans: int | None = None
    try:
        ntrans = _charmm_image_ntrans()
    except Exception:
        ntrans = None

    if ntrans is None or ntrans > 1:
        return True
    # ``get_ntrans()`` returns 1 for vacuum; after prepare_charmm_pbc the Fortran
    # log can show many transforms while the Python probe still reads vacuum.
    return charmm_crystal_is_active(rel_tol=rel_tol)


def restore_charmm_cubic_crystal_lattice(
    cubic_box_side_A: float,
    *,
    nbxmod: int = 5,
    quiet: bool = False,
    apply_nbonds: bool = True,
) -> float:
    """Reinstall CUBI crystal after ``crystal free`` or ``READ PARAM`` IMAGE clear.

    ``read_param_file`` (append) and ``crystal free`` drop IMAGE atom tables while
    lattice transforms may still be active.  After ``crystal build``, run
    ``image byres`` (via :func:`_image_setup_byres_all`) before ``upinb`` /
    ``UPIMNB`` — otherwise ``MAKGRP`` can segfault with NTRANS>0 and NATIM=0.

    Set ``apply_nbonds=False`` when ML exclusions must be installed before the first
    ``upinb`` (PBC registration after CGENFF param read).
    """
    import pycharmm.crystal as crystal

    from mmml.interfaces.pycharmmInterface.charmm_mpi import mpi_charmm_script
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import pbcset

    side = float(cubic_box_side_A)
    if side <= 0.0:
        raise ValueError(f"cubic box side must be > 0, got {side}")
    mpi_charmm_script(pbcset.format(SIDELENGTH=side), quiet=True)
    if not crystal.set_cubic_side(side):
        raise RuntimeError(f"crystal.set_cubic_side failed for L={side} Å")
    _image_setup_byres_all(0.0, 0.0, 0.0)
    if apply_nbonds:
        apply_pbc_nbonds(nbxmod=nbxmod, cubic_box_side_A=side)
    if not quiet:
        print(
            f"CHARMM crystal lattice restored: L={side:.3f} Å",
            flush=True,
        )
    return side


def _free_charmm_crystal_if_available() -> bool:
    """``CRYSTAL FREE`` when the KEY_LIBRARY export exists."""
    import pycharmm.crystal as crystal

    if crystal.crystal_free_available():
        return bool(crystal.free_crystal())
    return False


def reinstall_charmm_crystal_for_lattice_abnr(
    cubic_box_side_A: float,
    *,
    nbxmod: int = 5,
    quiet: bool = False,
    allow_prepare_pbc: bool = True,
) -> float:
    """Reinstall CRYSTAL/IMAGE before LATT ABNR (avoids singular ``XTLABC`` in ``MBUILD``).

    After coords+box lattice minimization or MC-density handoffs, ``XUCELL`` can
    remain populated while ``pbound_get_size`` reads zero and the crystal metric
    matrix is not usable for box-only (``NOCO``) lattice work.  A light
    :func:`restore_charmm_cubic_crystal_lattice` (no ``pbcset``) can still leave
    ``charmm_crystal_lattice_ready()`` true while ``MBUILD`` dies on a singular
    metric — lattice ABNR therefore uses full :func:`prepare_charmm_pbc` when safe.
    """
    side = float(cubic_box_side_A)
    if side <= 0.0:
        raise ValueError(f"cubic box side must be > 0, got {side}")

    if charmm_crystal_abnr_ready(side):
        return side

    if allow_prepare_pbc:
        _free_charmm_crystal_if_available()
        prepare_charmm_pbc(side)
        apply_pbc_nbonds(nbxmod=nbxmod, cubic_box_side_A=side)
    else:
        def _restore(*, free_first: bool) -> None:
            if free_first:
                _free_charmm_crystal_if_available()
            restore_charmm_cubic_crystal_lattice(
                side,
                nbxmod=nbxmod,
                quiet=quiet,
            )

        for free_first in (False, True):
            _restore(free_first=free_first)
            if charmm_crystal_abnr_ready(side):
                return side

    if not charmm_crystal_abnr_ready(side):
        raise RuntimeError(
            "CHARMM crystal metric not lattice-ready after reinstall; "
            f"cannot run lattice ABNR safely (L={side:.3f} Å)"
        )

    if not quiet:
        print(
            f"CHARMM crystal lattice restored: L={side:.3f} Å",
            flush=True,
        )
    return side


def assert_charmm_pbc_lattice_ready_for_mlpot(
    *,
    context: str = "MLpot",
    cubic_box_side_A: float | None = None,
) -> None:
    """Fail fast before ``mlpot_update`` when PBC crystal/IMAGE lists are unusable.

    ``mlpot_update`` (first MLpot ``ENER`` during SD) dereferences CHARMM neighbor
    and IMAGE arrays. Prefer live ``pbound_get_size`` when available; after
    registration-time crystal rebuild on KEY_LIBRARY/MPI builds, ``XUCELL`` +
    ``NTRANS`` can be valid while ``pbound_get_size`` stays zero in Python —
    :func:`charmm_crystal_abnr_ready` accepts that post-reinstall state.
    """
    if charmm_crystal_lattice_ready():
        return
    if charmm_crystal_abnr_ready(cubic_box_side_A):
        return
    side_hint = ""
    if cubic_box_side_A is not None and float(cubic_box_side_A) > 0.0:
        side_hint = f" (workflow L={float(cubic_box_side_A):.3f} Å)"
    raise RuntimeError(
        f"{context}: CHARMM PBC crystal is not lattice-ready{side_hint}. "
        "IMAGE/neighbor tables are unsafe for MLpot SD (mlpot_update may segfault). "
        "Re-run liquid-box / pre-MLpot prep or rebuild libcharmm with matching PBC "
        "limits."
    )


def ensure_charmm_crystal_for_cpt(
    cubic_box_side_A: float,
    *,
    quiet: bool = False,
) -> None:
    """Install or refresh CHARMM crystal before CPT / Hoover dynamics.

    CGENFF pre-minimize via :func:`minimize_charmm_mm_only` suspends PBC with
    ``crystal free`` during ``READ PARAM APPEND``. Restore the lattice when
    ``xtltyp``/IMAGE are gone but pbound still looks active.
    """
    side = float(cubic_box_side_A)
    if side <= 0.0:
        raise ValueError(f"cubic box side must be > 0, got {side}")
    if charmm_crystal_lattice_ready():
        try:
            live, _ = resolve_charmm_cubic_box_side_A(fallback_side_A=side)
            if abs(live - side) <= max(1e-3, 1e-4 * side):
                return
        except Exception:
            pass
    restore_charmm_cubic_crystal_lattice(side, quiet=quiet)


def _ensure_crystal_image_str() -> None:
    """Copy bundled ``crystal_image.str`` into cwd when missing (no setupBox import)."""
    import shutil

    from mmml.paths import crystal_image_str_source

    dst = Path("crystal_image.str")
    if dst.exists():
        return
    src = crystal_image_str_source()
    if src.is_file():
        shutil.copy2(src, dst)
        return
    raise FileNotFoundError(
        f"crystal_image.str not found in cwd and source {src} does not exist. "
        "CHARMM requires this file for periodic box setup."
    )


def _image_setup_byres_all(
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
) -> None:
    """``image byres ... sele all end`` via the KEY_LIBRARY C API."""
    import pycharmm.image as image
    import pycharmm.psf as psf

    seen: set[str] = set()
    for resname in psf.get_res():
        name = (resname or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        if not image.setup_residue(center_x, center_y, center_z, name):
            raise RuntimeError(f"CHARMM image centering failed for residue {name!r}")
    image.update_bimag()


def prepare_charmm_pbc(cubic_box_side_A: float) -> None:
    """Install CHARMM crystal + IMAGE for a cubic cell."""
    import pycharmm.crystal as crystal

    from mmml.interfaces.pycharmmInterface.charmm_mpi import mpi_charmm_script
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        PBC_NBOND_BOX_MARGIN_A,
        VACUUM_CUTNB,
    )
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import pbcset

    L = float(cubic_box_side_A)
    if L <= 0.0:
        raise ValueError(f"cubic box side must be > 0, got {L}")

    # SET BOXTYPE/FFTX/... still parse on KEY_LIBRARY builds; open/crystal do not.
    mpi_charmm_script(pbcset.format(SIDELENGTH=L), quiet=True)
    if not crystal.define_cubic(L):
        raise RuntimeError(f"crystal.define_cubic failed for L={L} Å")
    build_cut = min(float(VACUUM_CUTNB), max(L / 2.0 - PBC_NBOND_BOX_MARGIN_A, 6.0))
    if not crystal.build(build_cut):
        raise RuntimeError(f"crystal.build failed for cutoff={build_cut} Å (L={L})")
    _image_setup_byres_all(0.0, 0.0, 0.0)


def apply_pbc_nbonds(
    *,
    nbxmod: int = 5,
    cutnb: float = 18.0,
    cubic_box_side_A: float | None = None,
    rebuild: bool = True,
) -> PbcNbondCutoffs:
    """Nonbond list for periodic CHARMM (``cutim >= cutnb``).

    When ``cubic_box_side_A`` is set, clamp all cutoffs to stay below half the box.
    Returns the cutoffs actually applied.
    """
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        PbcNbondCutoffs,
        apply_nbonds_kwargs,
        pbc_nbond_cutoffs,
        scale_vacuum_switch_cutoffs,
    )

    if cubic_box_side_A is not None:
        cuts = pbc_nbond_cutoffs(cubic_box_side_A, cutnb_max=cutnb)
    else:
        ctonnb, ctofnb = scale_vacuum_switch_cutoffs(float(cutnb))
        cutim = float(cutnb) + 4.0
        cuts = PbcNbondCutoffs(
            cubic_box_side_A=float("nan"),
            cutnb=float(cutnb),
            cutim=cutim,
            ctonnb=ctonnb,
            ctofnb=ctofnb,
            ctexnb=float(cutnb),
        )
    from mmml.interfaces.pycharmmInterface.nbonds_config import apply_nbonds_kwargs

    apply_nbonds_kwargs(cuts.as_pbc_nbond_kwargs(nbxmod=nbxmod), rebuild=rebuild)
    return cuts


def setup_charmm_environment(
    *,
    use_pbc: bool,
    cubic_box_side_A: float | None,
    nbxmod: int = 5,
) -> dict[str, Any]:
    """Vacuum or PBC CHARMM environment before MLpot registration."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        prepare_charmm_vacuum,
        setup_default_nbonds,
    )

    if use_pbc:
        if cubic_box_side_A is None or float(cubic_box_side_A) <= 0.0:
            raise ValueError("PBC requires a positive cubic box side (Å)")
        prepare_charmm_pbc(float(cubic_box_side_A))
        cuts = apply_pbc_nbonds(
            nbxmod=nbxmod,
            cubic_box_side_A=float(cubic_box_side_A),
        )
        if cuts.was_capped:
            print(cuts.summary_line(), flush=True)
        return {"pbc": True, "box_A": float(cubic_box_side_A)}
    prepare_charmm_vacuum()
    setup_default_nbonds(nbxmod=nbxmod)
    return {"pbc": False, "box_A": None}


def sync_workflow_pbc_box_side_after_mm_pretreat(
    box_side: float | None,
    *,
    pretreat_restart: PathLike | None = None,
    args: Any | None = None,
    quiet: bool = False,
) -> float | None:
    """Align workflow ML MIC cell with the live CHARMM box after MM pretreat.

    NPT pretreat equi/prod can resize the crystal away from the Packmol density
    estimate passed into ``setup_charmm_environment``.  MLpot registration must
    use the post-pretreat box or MIC dimer distances disagree with CHARMM coords.

    When ``pretreat_restart`` is set, the restart crystal is cross-checked against
    live ``pbound_get_size`` and mismatches are logged.
    """
    if box_side is None:
        return None
    from_restart: float | None = None
    if pretreat_restart is not None:
        from_restart = parse_cubic_box_side_from_charmm_restart(pretreat_restart)
    crystal_active = False
    try:
        crystal_active = charmm_crystal_is_active()
    except Exception:
        crystal_active = False
    restart_for_resolve = None if crystal_active else pretreat_restart
    live, source = resolve_charmm_cubic_box_side_A(
        fallback_side_A=float(box_side),
        restart_path=restart_for_resolve,
    )
    old = float(box_side)
    if (
        from_restart is not None
        and from_restart > 0.0
        and abs(from_restart - live) > max(1e-3, 1e-4 * live)
        and not quiet
    ):
        print(
            "PBC box cross-check after CHARMM MM pretreat: "
            f"pbound/live={live:.3f} Å ({source}) vs restart={from_restart:.3f} Å "
            f"({Path(pretreat_restart).name}); using live CHARMM box for MLpot MIC",
            flush=True,
        )
    if abs(live - old) > 1e-3:
        if not quiet:
            print(
                f"PBC box sync after CHARMM MM pretreat: {old:.3f} -> {live:.3f} Å "
                f"(source={source})",
                flush=True,
            )
    elif not quiet:
        print(
            f"PBC workflow box: L={live:.3f} Å (source={source})",
            flush=True,
        )
    if args is not None and getattr(args, "box_size", None) is not None:
        try:
            from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
                _pretreat_use_fixed_box_nvt,
            )

            fixed_nvt = _pretreat_use_fixed_box_nvt(args, use_pbc=True)
        except Exception:
            fixed_nvt = True
        if not fixed_nvt and abs(live - old) > 1e-3:
            args.box_size = float(live)
    return float(live)


def sync_charmm_crystal_after_mm_pretreat(
    box_side_A: float,
    *,
    quiet: bool = False,
) -> bool:
    """Refresh IMAGE lists after MM pretreat without redefining the crystal.

    CPT/NVT pretreat already installed crystal + IMAGE during cluster setup and
    dynamics.  ``pbound_get_size`` can read zero in Python while Fortran IMAGE
    lists remain valid.  Re-running :func:`prepare_charmm_pbc` here re-folds
    residues and can desync MLpot MIC from in-memory coords (classical GRMS still
    looks fine; MLpot USER/GRMS explode at registration).

    Safe only **before** MLpot registration (no restart READ).
    """
    if box_side_A <= 0.0:
        return False
    if charmm_crystal_is_active():
        return False
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    with charmm_relaxed_bomlev():
        pycharmm.lingo.charmm_script("UPDATE\n")
    if not quiet:
        live, source = probe_charmm_cubic_box_side_A()
        if live is not None and source == "pbound":
            print(
                f"PBC IMAGE refreshed after MM pretreat: pbound L={live:.3f} Å",
                flush=True,
            )
        else:
            print(
                "PBC IMAGE refreshed after MM pretreat (pbound still inactive in "
                f"Python; MLpot MIC uses L={float(box_side_A):.3f} Å from pretreat sync)",
                flush=True,
            )
    return True


def find_latest_pretreat_mm_restart(paths: dict[str, Path]) -> Path | None:
    """Return the latest pretreat MM restart (prod, equi, or heat)."""
    for key in ("charmm_mm_prod_res", "charmm_mm_equi_res", "charmm_mm_heat_res"):
        candidate = paths.get(key)
        if candidate is not None and Path(candidate).is_file():
            return Path(candidate)
    return None


from mmml.interfaces.pycharmmInterface.nbonds_config import (  # noqa: E402
    PbcNbondCutoffs,
    pbc_nbond_cutoffs,
)

__all__ = [
    "PbcNbondCutoffs",
    "apply_pbc_nbonds",
    "assert_charmm_pbc_lattice_ready_for_mlpot",
    "charmm_crystal_abnr_ready",
    "charmm_crystal_is_active",
    "charmm_crystal_lattice_ready",
    "cubic_box_length_from_geometry",
    "cubic_box_matrix_from_side",
    "ensure_charmm_crystal_for_cpt",
    "get_charmm_cubic_box_side_A",
    "push_charmm_cubic_box_side_A",
    "pbc_nbond_cutoffs",
    "prepare_charmm_pbc",
    "reinstall_charmm_crystal_for_lattice_abnr",
    "restore_charmm_cubic_crystal_lattice",
    "setup_charmm_environment",
    "sync_charmm_box_from_workflow_side",
    "sync_workflow_pbc_box_side_after_mm_pretreat",
    "sync_charmm_crystal_after_mm_pretreat",
    "find_latest_pretreat_mm_restart",
]
