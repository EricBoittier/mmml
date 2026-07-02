"""DCM 27 Å liquid boxes at multiple densities — calculator smoke (jax-mm-spoof)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters


def _box27_case_names() -> list[str]:
    return [
        "synthetic_dcm_liquid_box27",
        "synthetic_dcm_liquid_box27_rho050",
        "synthetic_dcm_liquid_box27_rho075",
        "synthetic_dcm_liquid_box27_rho125",
        "synthetic_dcm_liquid_box27_rho150",
    ]


@pytest.mark.parametrize("case_name", _box27_case_names())
def test_dcm_box27_synthetic_case_geometry(case_name: str) -> None:
    from tests.functionality.neighbor_lists._common import (
        _composition_dict_from_liquid_case,
        build_liquid_density_synthetic_case,
        liquid_density_synthetic_cases,
    )

    cases = {str(c["name"]): c for c in liquid_density_synthetic_cases()}
    case = cases[case_name]
    comp = _composition_dict_from_liquid_case(case)
    assert list(comp.keys()) == ["DCM"]
    assert comp["DCM"] <= 32
    pos, cell, offsets, monomer_id, cutoff, _desc, side, rho = build_liquid_density_synthetic_case(
        case
    )
    assert abs(side - 27.0) < 1e-6
    assert rho > 0.0
    assert cutoff == 13.0
    assert pos.shape[0] == comp["DCM"] * 5
    assert offsets.shape[0] == comp["DCM"] + 1
    assert monomer_id.shape[0] == pos.shape[0]


@pytest.mark.parametrize("case_name", _box27_case_names())
def test_dcm_box27_hybrid_calculator_jax_mm_spoof(case_name: str) -> None:
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator
    from tests.functionality.neighbor_lists._common import (
        build_liquid_density_synthetic_case,
        liquid_density_synthetic_cases,
    )

    cases = {str(c["name"]): c for c in liquid_density_synthetic_cases()}
    pos, cell, offsets, monomer_id, _cutoff, _desc, side, _rho = build_liquid_density_synthetic_case(
        cases[case_name]
    )
    n_monomers = int(offsets.shape[0] - 1)
    n_atoms = int(pos.shape[0])
    atoms_per_monomer = n_atoms // n_monomers
    z = jnp.full((n_atoms,), 6, dtype=jnp.int32)
    fake_mm_fn = MagicMock(return_value=(jnp.array(0.0), jnp.zeros((n_atoms, 3))))

    with patch(
        "mmml.interfaces.pycharmmInterface.mmml_calculator.build_mm_energy_forces_fn",
        return_value=fake_mm_fn,
    ):
        factory = setup_calculator(
            ATOMS_PER_MONOMER=atoms_per_monomer,
            N_MONOMERS=n_monomers,
            model_restart_path=None,
            ml_potential_mode="jax_mm_clone",
            doML=True,
            doMM=True,
            doML_dimer=True,
            MAX_ATOMS_PER_SYSTEM=atoms_per_monomer * 2,
            cell=float(side),
            defer_xla_gpu_warmup=True,
            verbose=False,
            ml_sparse_dimers=False,
        )
        spherical_fn, _, _ = factory(
            atomic_numbers=z,
            atomic_positions=jnp.asarray(pos, dtype=jnp.float64),
            n_monomers=n_monomers,
            cutoff_params=CutoffParameters(),
            doML=True,
            doMM=True,
            doML_dimer=True,
            backprop=False,
            create_ase_calculator=False,
        )
        out = spherical_fn(
            atomic_numbers=z,
            positions=jnp.asarray(pos, dtype=jnp.float64),
            n_monomers=n_monomers,
            cutoff_params=CutoffParameters(),
            doML=True,
            doMM=True,
            doML_dimer=True,
            box=jnp.asarray(cell, dtype=jnp.float64),
        )
    assert bool(jnp.isfinite(out.energy))
    assert out.forces.shape == (n_atoms, 3)
    assert bool(jnp.all(jnp.isfinite(out.forces)))
