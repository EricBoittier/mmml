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
    Skips work when the live CHARMM box already matches ``target_side_A``.
    """
    target = float(target_side_A)
    if target <= 0.0:
        raise ValueError(f"target box side must be > 0, got {target}")
    tol = max(1e-3, 1e-4 * target)
    live, source = probe_charmm_cubic_box_side_A(fallback_side_A=target)
    if live is not None and abs(live - target) <= tol:
        return float(live), source or "pbound"
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


def ensure_charmm_crystal_for_cpt(
    cubic_box_side_A: float,
    *,
    quiet: bool = False,
) -> None:
    """Install or refresh CHARMM crystal before CPT / Hoover dynamics.

    CGENFF pre-minimize via :func:`minimize_charmm_mm_only` used to call vacuum
    nbonds (``crystal free``). Loose-PBC Hoover heat needs the box restored.
    """
    side = float(cubic_box_side_A)
    if side <= 0.0:
        raise ValueError(f"cubic box side must be > 0, got {side}")
    if charmm_crystal_is_active():
        try:
            live, _ = resolve_charmm_cubic_box_side_A(fallback_side_A=side)
            if abs(live - side) <= max(1e-3, 1e-4 * side):
                return
        except Exception:
            pass
    push_charmm_cubic_box_side_A(side, quiet=quiet)


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

    apply_nbonds_kwargs(cuts.as_pbc_nbond_kwargs(nbxmod=nbxmod))
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
    "cubic_box_length_from_geometry",
    "cubic_box_matrix_from_side",
    "ensure_charmm_crystal_for_cpt",
    "get_charmm_cubic_box_side_A",
    "push_charmm_cubic_box_side_A",
    "pbc_nbond_cutoffs",
    "prepare_charmm_pbc",
    "setup_charmm_environment",
    "sync_charmm_box_from_workflow_side",
    "sync_workflow_pbc_box_side_after_mm_pretreat",
    "sync_charmm_crystal_after_mm_pretreat",
    "find_latest_pretreat_mm_restart",
]
