"""PyCHARMM reference energies/forces for CGENFF bonded cross-checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import charmm_bonded_term_kcalmol


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
    energy_atol: float = 1e-5,
    force_rtol: float = 1e-3,
    force_atol: float = 1e-3,
) -> None:
    """Assert JAX bonded E/F match PyCHARMM bonded-only reference."""
    charmm = charmm_bonded_energy_components_kcalmol()
    charmm_forces = charmm_bonded_forces_kcalmol_A()

    mapping = {
        "bond": "bond",
        "angle": "angl",
        "torsion": "dihe",
        "improper": "impr",
        "total": "total",
    }
    for jax_key, charmm_key in mapping.items():
        if jax_key not in jax_components:
            continue
        jax_val = float(jax_components[jax_key])
        charmm_val = float(charmm.get(charmm_key, 0.0))
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
