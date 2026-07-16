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
from mmml.analysis.dimer_molecules import MOLECULES
from mmml.interfaces.pycharmmInterface.long_range_backend import per_atom_jax_pme_c6_sqrt
from mmml.models.spookynet_calc import (
    SpookyNetCalculator,
    _infer_vdw_architecture_config,
    _is_spooky_checkpoint,
)
from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet
from scripts.run_dimer_scan_campaign import _charmm_component_rows


def test_spooky_checkpoint_markers_can_be_nested_under_params():
    # JSON checkpoint trees are Flax variable dicts: module names live under
    # the outer ``params`` collection, not at the root.
    tree = {"params": {"charge_feature_projection": {"kernel": np.ones((1, 2))}}}
    assert _is_spooky_checkpoint("", tree) is True
    assert _is_spooky_checkpoint("spooky", {}) is True
    assert _is_spooky_checkpoint("physnet", tree) is False


def test_legacy_checkpoint_disables_absent_optional_vdw_heads():
    tree = {"params": {"Dense_12": {"kernel": np.ones((1, 1))}}}
    config = _infer_vdw_architecture_config({}, tree)
    assert config["predict_atomic_vdw_scale"] is False
    assert config["learn_cgenff_vdw_scale"] is False


def test_recent_checkpoint_infers_present_optional_vdw_heads():
    tree = {
        "params": {
            "Dense_13": {"kernel": np.ones((1, 1))},
            "global_vdw_scale": np.ones(1),
            "element_vdw_scale": np.ones(18),
        }
    }
    config = _infer_vdw_architecture_config({}, tree)
    assert config["predict_atomic_vdw_scale"] is True
    assert config["learn_cgenff_vdw_scale"] is True


def test_explicit_vdw_architecture_flags_override_tree_inference():
    tree = {"params": {"Dense_13": {"kernel": np.ones((1, 1))}}}
    config = _infer_vdw_architecture_config(
        {"predict_atomic_vdw_scale": False, "learn_cgenff_vdw_scale": True}, tree
    )
    assert config["predict_atomic_vdw_scale"] is False
    assert config["learn_cgenff_vdw_scale"] is True


def test_legacy_checkpoint_infers_trainable_zbl_from_parameter_tree():
    tree = {"params": {"repulsion": {"a_coefficient": np.asarray(0.5)}}}
    config = _infer_vdw_architecture_config({}, tree)
    assert config["trainable_zbl"] is True
    # Must not reuse the ML neighbor cutoff as the ZBL window.
    assert config["zbl_cuton"] == pytest.approx(0.1)
    assert config["zbl_cutoff"] == pytest.approx(0.6)


def test_legacy_zbl_clamps_wide_cutoff_copied_from_model_cutoff():
    from mmml.utils.model_checkpoint import infer_trainable_zbl_config

    tree = {"params": {"repulsion": {"a_coefficient": np.asarray(0.5)}}}
    config = infer_trainable_zbl_config(
        {"trainable_zbl": True, "cutoff": 6.0, "zbl_cuton": None, "zbl_cutoff": 6.0},
        tree,
    )
    assert config["zbl_cuton"] == pytest.approx(0.1)
    assert config["zbl_cutoff"] == pytest.approx(0.6)


def test_new_checkpoint_keeps_explicit_fixed_zbl():
    tree = {"params": {"repulsion": {"a_coefficient": np.asarray(0.5)}}}
    config = _infer_vdw_architecture_config({"trainable_zbl": False}, tree)
    assert config["trainable_zbl"] is False
    assert config["zbl_cuton"] == pytest.approx(0.1)
    assert config["zbl_cutoff"] == pytest.approx(0.6)


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
    assert report["short_range"]["zbl_trainable"] is False
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


def test_dcm_geometry_order_matches_cgenff_types_and_topology(tmp_path):
    prm = tmp_path / "dcm.prm"
    prm.write_text(
        "NONBONDED\n"
        "CG321 0.0 -0.0560 2.0100\n"
        "CLGA1 0.0 -0.3430 1.9100\n"
        "HGA2 0.0 -0.0350 1.3400\n"
        "NBFIX\n"
    )
    monomer = MOLECULES["DCM"]
    atoms = monomer + monomer.copy()
    fragments = (np.arange(5), np.arange(5, 10))
    attach_cgenff_dimer_metadata(
        atoms, ("DCM", "DCM"), fragments, prm_path=prm
    )
    mapping, sigmas, epsilons = load_cgenff_sigma_epsilon(str(prm.resolve()))
    expected_names = ("CG321", "CLGA1", "CLGA1", "HGA2", "HGA2")
    assert monomer.get_chemical_symbols() == ["C", "Cl", "Cl", "H", "H"]
    assert CGENFF_ATOM_TYPES["DCM"] == expected_names
    np.testing.assert_array_equal(
        atoms.arrays["cgenff_type_idx"][:5],
        [mapping[name] for name in expected_names],
    )
    assert epsilons[mapping["CLGA1"]] == pytest.approx(0.3430)
    assert sigmas[mapping["CLGA1"]] == pytest.approx(
        2.0 * 1.9100 / 2.0 ** (1.0 / 6.0)
    )


def test_cgenff_lj_uses_pair_distance_not_coulomb_kernel():
    sigma = 3.0
    epsilon = 0.2
    r_min = 2.0 ** (1.0 / 6.0) * sigma
    model = SpookyPhysNet(
        learn_cgenff_vdw_scale=False,
        predict_atomic_vdw_scale=False,
    )
    _, batch_vdw = model._calculate_cgenff_vdw(
        jnp.asarray([[r_min, 0.0, 0.0], [-r_min, 0.0, 0.0]]),
        jnp.ones(2),
        jnp.zeros(2, dtype=jnp.int32),
        jnp.asarray([sigma]),
        jnp.asarray([epsilon]),
        jnp.asarray([6, 6]),
        None,
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([1, 0], dtype=jnp.int32),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.ones(2),
        jnp.zeros(2, dtype=jnp.int32),
        1,
    )
    assert float(batch_vdw.squeeze()) == pytest.approx(
        -epsilon * 0.0433641153, rel=1e-6
    )
