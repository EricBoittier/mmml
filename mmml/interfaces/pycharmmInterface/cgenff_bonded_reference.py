"""PyCHARMM reference energies/forces for CGENFF bonded cross-checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import charmm_bonded_term_kcalmol


def charmm_positions_xyz_array() -> np.ndarray:
    """Read active PyCHARMM coordinates as ``(N, 3)`` with explicit ``x,y,z`` columns."""
    import pycharmm.coor as coor

    return coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=np.float64)


def set_charmm_positions(positions: np.ndarray) -> None:
    """Load ``(N, 3)`` coordinates into the active PyCHARMM session."""
    import pycharmm.coor as coor

    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {arr.shape}")
    coor.set_positions(pd.DataFrame(arr, columns=["x", "y", "z"]))


def _psf_needs_xplor_reader(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    except OSError:
        return False
    return bool(head) and "XPLOR" in head[0].upper()


def read_psf_card_file(
    path: str | Path,
    *,
    append: bool = False,
    xplor: bool | None = None,
) -> None:
    """Read a PSF via the Fortran C API (EXT/XPLOR-safe).

    ``pycharmm.read.psf_card`` (CommandScript) can leave ``nbond=0`` for committed
  EXT PSF fixtures under MPI-linked CHARMM; use this helper in live tests.
    """
    import ctypes

    import pycharmm.lib as lib
    from pycharmm.charmm_file import c_api_path_buffer

    from mmml.interfaces.pycharmmInterface.charmm_paths import charmm_fortran_path

    p = Path(path)
    use_xplor = _psf_needs_xplor_reader(p) if xplor is None else bool(xplor)
    fortran_path, _alias = charmm_fortran_path(p)
    buf, fn_len = c_api_path_buffer(fortran_path)
    c_append = ctypes.c_int(int(append))
    c_xplor = ctypes.c_int(int(use_xplor))
    status = lib.charmm.read_psf_card(
        buf,
        ctypes.byref(fn_len),
        ctypes.byref(c_append),
        ctypes.byref(c_xplor),
    )
    if int(status) != 1:
        raise RuntimeError(f"read_psf_card failed for {p} (status={status})")


def read_pdb_file(path: str | Path, **kwargs: Any) -> None:
    """Read PDB coordinates with lowercase Fortran-safe staging when needed."""
    from mmml.interfaces.pycharmmInterface.charmm_paths import charmm_fortran_path

    import pycharmm.read as read

    fortran_path, _alias = charmm_fortran_path(Path(path))
    read.pdb(fortran_path, **kwargs)


def charmm_cmap_is_active(
    charmm_bonded: dict[str, float] | None = None,
    *,
    atol: float = 1e-8,
) -> bool:
    """True when PyCHARMM reports a non-zero CMAP term after ``ENER`` / ``ENER FORCE``.

    JAX CMAP should be gated off when this returns False so parity checks match
    live CHARMM (e.g. TRIA CGENFF backbone grids not active in PyCHARMM ``ENER``).
    """
    bonded = (
        charmm_bonded
        if charmm_bonded is not None
        else charmm_bonded_energy_components_kcalmol()
    )
    return abs(float(bonded.get("cmap", 0.0))) > float(atol)


def setup_bonded_only_charmm() -> None:
    """Zero nonbonded terms so ``ENER FORCE`` reports bonded MM only."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_bonded_mm_only_block

    apply_bonded_mm_only_block()


def charmm_bonded_energy_components_kcalmol() -> dict[str, float]:
    """Read CHARMM bonded term energies (kcal/mol) after ``ENER`` / ``ENER FORCE``."""
    keys = ("BOND", "ANGL", "DIHE", "IMPR", "UREY", "UB", "CMAP")
    out: dict[str, float] = {}
    total = 0.0
    for key in keys:
        val = charmm_bonded_term_kcalmol(key)
        if val is None:
            continue
        out[key.lower()] = float(val)
        total += float(val)
    out["total"] = total
    return out


def charmm_bonded_forces_kcalmol_A() -> np.ndarray:
    """Per-atom bonded-only forces (kcal/mol/Å) from the last ``ENER FORCE``."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_total_forces_kcalmol_A

    return np.asarray(charmm_total_forces_kcalmol_A(), dtype=np.float64)


def run_charmm_bonded_ener_force(*, silent: bool = True) -> None:
    """Evaluate bonded-only CHARMM energy and forces."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    if silent:
        from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command

        with charmm_silent_command():
            pycharmm.lingo.charmm_script("ENER FORCE")
    else:
        pycharmm.lingo.charmm_script("ENER FORCE")


def compare_bonded_to_charmm(
    jax_components: dict[str, Any],
    jax_forces: np.ndarray,
    *,
    energy_rtol: float = 1e-4,
    energy_atol: float = 1e-4,
    force_rtol: float = 1e-3,
    force_atol: float = 1e-3,
    ignore_charmm_terms: tuple[str, ...] = (),
    include_cmap: bool | None = None,
) -> None:
    """Assert JAX bonded E/F match PyCHARMM bonded-only reference."""
    charmm = charmm_bonded_energy_components_kcalmol()
    charmm_forces = charmm_bonded_forces_kcalmol_A()
    if include_cmap is None:
        include_cmap = charmm_cmap_is_active(charmm)

    ignored = sum(float(charmm.get(term, 0.0)) for term in ignore_charmm_terms)

    mapping = {
        "bond": "bond",
        "angle": "angl",
        "urey": ("urey", "ub"),
        "torsion": "dihe",
        "improper": "impr",
        "cmap": "cmap",
    }
    for jax_key, charmm_key in mapping.items():
        if jax_key not in jax_components:
            continue
        if jax_key == "cmap" and not include_cmap:
            continue
        jax_val = float(jax_components[jax_key])
        if isinstance(charmm_key, tuple):
            charmm_val = sum(float(charmm.get(k, 0.0)) for k in charmm_key)
            if not any(k in charmm for k in charmm_key):
                continue
        else:
            if charmm_key not in charmm:
                continue
            charmm_val = float(charmm[charmm_key])
        np.testing.assert_allclose(
            jax_val,
            charmm_val,
            rtol=energy_rtol,
            atol=energy_atol,
            err_msg=f"bonded energy mismatch for {jax_key}",
        )

    if "total" in jax_components:
        mapped_charmm_total = 0.0
        for k, charmm_key in mapping.items():
            if k not in jax_components:
                continue
            if k == "cmap" and not include_cmap:
                continue
            if isinstance(charmm_key, tuple):
                mapped_charmm_total += sum(
                    float(charmm[m]) for m in charmm_key if m in charmm
                )
            elif charmm_key in charmm:
                mapped_charmm_total += float(charmm[charmm_key])
        mapped_charmm_total -= ignored
        np.testing.assert_allclose(
            float(jax_components["total"]),
            mapped_charmm_total,
            rtol=energy_rtol,
            atol=energy_atol,
            err_msg="bonded energy mismatch for total",
        )

    np.testing.assert_allclose(
        np.asarray(jax_forces, dtype=np.float64),
        charmm_forces,
        rtol=force_rtol,
        atol=force_atol,
        err_msg="bonded forces mismatch vs PyCHARMM",
    )


def setup_nonbonded_only_charmm() -> None:
    """Zero bonded terms so ``ENER FORCE`` reports VDW/ELEC only."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
        _assert_selective_block_safe,
    )

    _assert_selective_block_safe(context="setup_nonbonded_only_charmm")
    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet

    block = """BLOCK
CALL 1 SELE ALL END
COEFF 1 1 0.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0 ELEC 1.0 VDW 1.0
END
"""
    run_charmm_script_quiet(block)


def _charmm_active_energy_terms() -> dict[str, float]:
    """Active CHARMM ``ETERM`` values (kcal/mol) after the last ``ENER``."""
    try:
        import pycharmm.energy as energy

        statuses = energy.get_term_statuses()
        tnames = energy.get_term_names()
        terms = energy.get_terms()
        return {
            str(name).strip().upper(): float(val)
            for active, name, val in zip(statuses, tnames, terms, strict=False)
            if active
        }
    except Exception:
        return {}


def _charmm_nb_term_sum(*names: str) -> float:
    """Sum CHARMM ENER terms (kcal/mol); missing names contribute 0."""
    active = _charmm_active_energy_terms()
    if active:
        return sum(float(active.get(str(name).strip().upper(), 0.0)) for name in names)

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import charmm_bonded_term_kcalmol

    total = 0.0
    for name in names:
        val = charmm_bonded_term_kcalmol(name)
        if val is not None:
            total += float(val)
    return total


def charmm_nonbonded_energy_components_kcalmol() -> dict[str, float]:
    """VDW/ELEC components (kcal/mol) after the last ``ENER``.

    Under PBC, CHARMM splits pair and image contributions (``VDW`` + ``IMNB``,
    ``ELEC`` + ``IMEL``). JAX MIC reports the combined switched pair totals, so
    map CHARMM keys to match JAX ``vdw`` / ``elec`` decomposition.
    """
    vdw_f = _charmm_nb_term_sum("VDW", "IMNB")
    elec_f = _charmm_nb_term_sum("ELEC", "IMEL", "EXTE")
    return {"vdw": vdw_f, "elec": elec_f, "total": vdw_f + elec_f}


def segment_category_block_script(
    category: str,
    *,
    pep_seg: str = "PEPT",
    solv_seg: str = "SOLV",
) -> str:
    """CHARMM BLOCK script enabling nonbonded terms for one segment-pair class only.

    Block 1 = peptide (``PEPT``), block 2 = solvent (``SOLV``).  Bonded terms are
    off; only the requested ``pep_pep`` / ``pep_water`` / ``water_water`` class has
    ``ELEC`` and ``VDW`` enabled.
    """
    flags = {
        "pep_pep": (1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        "pep_water": (0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
        "water_water": (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
    }
    if category not in flags:
        raise ValueError(f"unknown segment category {category!r}")
    e11, v11, e12, v12, e22, v22 = flags[category]
    return f"""BLOCK
CALL 1 SELE SEGID {pep_seg} END
CALL 2 SELE SEGID {solv_seg} END
COEFF 1 1 0.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0 ELEC {e11} VDW {v11}
COEFF 1 2 0.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0 ELEC {e12} VDW {v12}
COEFF 2 2 0.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0 ELEC {e22} VDW {v22}
END
"""


def charmm_nonbonded_by_segment_category(
    categories: tuple[str, ...] = ("pep_pep", "pep_water", "water_water"),
    *,
    pep_seg: str = "PEPT",
    solv_seg: str = "SOLV",
    restore_full_mm_block: bool = True,
) -> dict[str, dict[str, float | np.ndarray]]:
    """Per-class VDW/ELEC energies and forces via selective BLOCK + ``ENER FORCE``.

    Requires an active PBC solvated PSF with ``SEGID`` ``PEPT`` and ``SOLV``.
    Under MPI-linked libcharmm + ``mpirun``, selective BLOCK may hang unless
    ``MMML_ALLOW_SELECTIVE_BONDED_BLOCK=1``.
    """
    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
        _assert_selective_block_safe,
        apply_charmm_mm_block,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_total_forces_kcalmol_A

    _assert_selective_block_safe(context="charmm_nonbonded_by_segment_category")
    out: dict[str, dict[str, float | np.ndarray]] = {}
    for category in categories:
        block = segment_category_block_script(
            category,
            pep_seg=pep_seg,
            solv_seg=solv_seg,
        )
        run_charmm_script_quiet(block)
        run_charmm_bonded_ener_force(silent=True)
        nb = charmm_nonbonded_energy_components_kcalmol()
        forces = np.asarray(charmm_total_forces_kcalmol_A(), dtype=np.float64)
        out[category] = {
            "vdw": float(nb["vdw"]),
            "elec": float(nb["elec"]),
            "total": float(nb["total"]),
            "forces": forces,
        }
    if restore_full_mm_block:
        apply_charmm_mm_block()
    return out


def run_charmm_nonbonded_ener_force(*, silent: bool = True) -> None:
    """Evaluate nonbonded-only CHARMM energy and forces."""
    run_charmm_bonded_ener_force(silent=silent)


def compare_nonbonded_to_charmm(
    jax_components: dict[str, Any],
    jax_forces: np.ndarray,
    *,
    energy_rtol: float = 5e-4,
    energy_atol: float = 1e-3,
    force_rtol: float = 5e-3,
    force_atol: float = 5e-3,
) -> None:
    """Assert JAX switched nonbonded E/F match PyCHARMM (nonbonded-only BLOCK)."""
    charmm = charmm_nonbonded_energy_components_kcalmol()
    charmm_forces = charmm_bonded_forces_kcalmol_A()

    for key in ("vdw", "elec", "total"):
        np.testing.assert_allclose(
            float(jax_components[key]),
            float(charmm[key]),
            rtol=energy_rtol,
            atol=energy_atol,
            err_msg=f"nonbonded energy mismatch for {key}",
        )
    np.testing.assert_allclose(
        np.asarray(jax_forces, dtype=np.float64),
        charmm_forces,
        rtol=force_rtol,
        atol=force_atol,
        err_msg="nonbonded forces mismatch vs PyCHARMM",
    )


def compare_mm_system_to_charmm(
    result: Any,
    *,
    energy_rtol: float = 5e-4,
    energy_atol: float = 2e-2,
    force_rtol: float = 5e-3,
    force_atol: float = 5e-2,
    ignore_charmm_bonded_terms: tuple[str, ...] = (),
) -> None:
    """Assert full JAX MM (bonded + nonbonded) matches PyCHARMM ``ENER FORCE``."""
    import pycharmm.energy as energy

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_total_forces_kcalmol_A

    charmm_bonded = charmm_bonded_energy_components_kcalmol()
    charmm_nb = charmm_nonbonded_energy_components_kcalmol()
    ignored = sum(float(charmm_bonded.get(t, 0.0)) for t in ignore_charmm_bonded_terms)
    charmm_total = float(energy.get_total()) - ignored

    jax_bonded = float(result.bonded.get("total", sum(result.bonded.values())))
    jax_nb = float(result.nonbonded.get("total", 0.0))

    np.testing.assert_allclose(
        jax_bonded,
        float(charmm_bonded.get("total", 0.0)) - ignored,
        rtol=energy_rtol,
        atol=energy_atol,
        err_msg=(
            "bonded MM energy mismatch vs PyCHARMM "
            f"(jax={jax_bonded:.4f}, charmm={charmm_bonded.get('total', 0.0):.4f})"
        ),
    )
    for key in ("vdw", "elec", "total"):
        np.testing.assert_allclose(
            float(result.nonbonded[key]),
            float(charmm_nb[key]),
            rtol=energy_rtol,
            atol=energy_atol,
            err_msg=(
                f"nonbonded energy mismatch for {key} "
                f"(jax={float(result.nonbonded[key]):.4f}, "
                f"charmm={float(charmm_nb[key]):.4f})"
            ),
        )

    np.testing.assert_allclose(
        result.total_energy,
        charmm_total,
        rtol=energy_rtol,
        atol=energy_atol,
        err_msg=(
            "total MM energy mismatch vs PyCHARMM "
            f"(jax={result.total_energy:.4f}, charmm={charmm_total:.4f}, "
            f"jax_bonded={jax_bonded:.4f}, jax_nb={jax_nb:.4f})"
        ),
    )
    charmm_forces = charmm_total_forces_kcalmol_A()
    np.testing.assert_allclose(
        result.forces,
        charmm_forces,
        rtol=force_rtol,
        atol=force_atol,
        err_msg="total MM forces mismatch vs PyCHARMM",
    )
