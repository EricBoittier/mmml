from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms

from mmml.analysis.dimer_cgenff import (
    CGENFF_ATOM_TYPES,
    attach_cgenff_dimer_metadata,
    load_cgenff_sigma_epsilon,
)
from mmml.interfaces.pycharmmInterface.long_range_backend import per_atom_jax_pme_c6_sqrt
from mmml.models.spookynet_calc import SpookyNetCalculator
from scripts.run_dimer_scan_campaign import _charmm_component_rows


def test_spookynet_report_flags_missing_training_lj_inputs(tmp_path):
    calc = object.__new__(SpookyNetCalculator)
    calc.checkpoint_path = tmp_path / "checkpoint.json"
    calc.raw_config = {
        "no_cgenff_vdw": False,
        "electrostatics_damping_sigma": 4.0,
        "mbd_checkpoint": None,
    }
    calc.normalized_config = {"cutoff": 6.0}
    calc.compute_dtype = jnp.float64
    calc.charge = 0.0
    calc.model = SimpleNamespace(
        charges=True,
        zbl=True,
        predict_atomic_vdw_scale=False,
        learn_cgenff_vdw_scale=False,
    )
    calc.cgenff_lj_inputs_supplied = False
    calc.mbd_calc = None
    calc.mbd_weight = 0.0

    report = calc.energy_function_report()

    assert report["cgenff_lennard_jones"]["parameter_file_radius_field"] == "Rmin/2 (angstrom)"
    assert report["cgenff_lennard_jones"]["annotated_atoms_supported"] is True
    assert report["cgenff_lennard_jones"]["inputs_supplied_at_inference"] is False
    assert report["short_range"]["zbl_repulsion"] is True
    assert any("omit the fixed LJ contribution" in item for item in report["warnings"])


def test_charmm_components_are_materialized_as_independent_backends():
    rows = _charmm_component_rows(
        {"molecule_a": "DCM", "molecule_b": "TIP3"},
        total_kcal=-3.0,
        elec_kcal=-5.0,
        vdw_kcal=2.0,
    )
    by_backend = {row["backend"]: row for row in rows}
    assert set(by_backend) == {"charmm", "charmm_electrostatics", "charmm_lj"}
    assert by_backend["charmm_electrostatics"]["energy_kcal_mol"] == -5.0
    assert by_backend["charmm_lj"]["energy_kcal_mol"] == 2.0


def test_pme_dispersion_converts_charmm_rmin_half_to_full_rmin():
    epsilon = 0.2
    rmin_half = 2.0
    coefficient = per_atom_jax_pme_c6_sqrt(
        np.array([epsilon]), np.array([rmin_half])
    )[0]
    expected_c6 = 2.0 * epsilon * (2.0 * rmin_half) ** 6
    old_wrong_c6 = 2.0 * epsilon * rmin_half**6
    assert coefficient**2 == pytest.approx(expected_c6)
    assert expected_c6 / old_wrong_c6 == pytest.approx(64.0)


def test_dimer_metadata_converts_rmin_half_and_preserves_ace_order(tmp_path):
    prm = tmp_path / "test.prm"
    prm.write_text(
        "NONBONDED\n"
        "OG2D3  0.0 -0.12 1.7\n"
        "CG2O5  0.0 -0.10 2.0\n"
        "CG331  0.0 -0.08 2.1\n"
        "HGA3   0.0 -0.02 1.2\n"
        "NBFIX\n"
    )
    numbers = [8, 6, 6, 6, 1, 1, 1, 1, 1, 1] * 2
    atoms = Atoms(numbers=numbers, positions=np.zeros((20, 3)))
    fragments = (np.arange(10, dtype=int), np.arange(10, 20, dtype=int))
    attach_cgenff_dimer_metadata(
        atoms, ("ACE", "ACE"), fragments, prm_path=prm
    )
    mapping, sigmas, _ = load_cgenff_sigma_epsilon(str(prm.resolve()))
    indices = atoms.arrays["cgenff_type_idx"]
    assert indices[0] == mapping["OG2D3"]  # ASE acetone atom 0 is oxygen.
    assert indices[1] == mapping["CG2O5"]  # Atom 1 is the carbonyl carbon.
    assert sigmas[mapping["CG331"]] == pytest.approx(
        2.0 * 2.1 / 2.0 ** (1.0 / 6.0)
    )
    assert CGENFF_ATOM_TYPES["ACE"][:4] == (
        "OG2D3", "CG2O5", "CG331", "CG331"
    )
