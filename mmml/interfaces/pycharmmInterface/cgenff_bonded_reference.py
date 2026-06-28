"""PyCHARMM reference energies/forces for CGENFF bonded cross-checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import charmm_bonded_term_kcalmol


def set_charmm_positions(positions: np.ndarray) -> None:
    """Load ``(N, 3)`` coordinates into the active PyCHARMM session."""
    import pycharmm.coor as coor

    arr = np.asarray(positions, dtype=np.float64)
    coor.set_positions(pd.DataFrame(arr, columns=["x", "y", "z"]))


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
    energy_rtol: float = 1e-3,
    energy_atol: float = 5e-3,
    force_rtol: float = 1e-3,
    force_atol: float = 1e-3,
    ignore_charmm_terms: tuple[str, ...] = (),
) -> None:
    """Assert JAX bonded E/F match PyCHARMM bonded-only reference."""
    charmm = charmm_bonded_energy_components_kcalmol()
    charmm_forces = charmm_bonded_forces_kcalmol_A()

    ignored = sum(float(charmm.get(term, 0.0)) for term in ignore_charmm_terms)

    mapping = {
        "bond": "bond",
        "angle": "angl",
        "torsion": "dihe",
        "improper": "impr",
        "cmap": "cmap",
        "total": "total",
    }
    for jax_key, charmm_key in mapping.items():
        if jax_key not in jax_components:
            continue
        jax_val = float(jax_components[jax_key])
        charmm_val = float(charmm.get(charmm_key, 0.0))
        if jax_key == "total":
            charmm_val -= ignored
        if charmm_key not in charmm and jax_key != "total":
            continue
        np.testing.assert_allclose(
            jax_val,
            charmm_val,
            rtol=energy_rtol,
            atol=energy_atol,
            err_msg=f"bonded energy mismatch for {jax_key}",
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
    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet

    block = """BLOCK
CALL 1 SELE ALL END
COEFF 1 1 0.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0 ELEC 1.0 VDW 1.0
END
"""
    run_charmm_script_quiet(block)


def charmm_nonbonded_energy_components_kcalmol() -> dict[str, float]:
    """VDW/ELEC components (kcal/mol) after the last ``ENER``."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import charmm_bonded_term_kcalmol

    vdw = charmm_bonded_term_kcalmol("VDW")
    elec = charmm_bonded_term_kcalmol("ELEC")
    vdw_f = 0.0 if vdw is None else float(vdw)
    elec_f = 0.0 if elec is None else float(elec)
    return {"vdw": vdw_f, "elec": elec_f, "total": vdw_f + elec_f}


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
    ignore_charmm_bonded_terms: tuple[str, ...] = ("cmap",),
) -> None:
    """Assert full JAX MM (bonded + nonbonded) matches PyCHARMM ``ENER FORCE``."""
    import pycharmm.energy as energy

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_total_forces_kcalmol_A

    charmm_bonded = charmm_bonded_energy_components_kcalmol()
    charmm_nb = charmm_nonbonded_energy_components_kcalmol()
    ignored = sum(float(charmm_bonded.get(t, 0.0)) for t in ignore_charmm_bonded_terms)
    charmm_total = float(energy.get_total()) - ignored

    np.testing.assert_allclose(
        result.total_energy,
        charmm_total,
        rtol=energy_rtol,
        atol=energy_atol,
        err_msg="total MM energy mismatch vs PyCHARMM",
    )
    charmm_forces = charmm_total_forces_kcalmol_A()
    np.testing.assert_allclose(
        result.forces,
        charmm_forces,
        rtol=force_rtol,
        atol=force_atol,
        err_msg="total MM forces mismatch vs PyCHARMM",
    )
