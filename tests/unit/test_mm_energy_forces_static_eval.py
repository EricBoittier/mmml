"""Static MM eval with cell-list pairs (force_static_mm_eval)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

jax = pytest.importorskip("jax")


def test_build_mm_energy_forces_fn_force_static_cell_list_sets_pair_lambda():
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import build_mm_energy_forces_fn

    n_atoms = 8
    n_mono = 2
    offsets = np.array([0, 4, 8], dtype=np.int32)
    atoms_per = [4, 4]
    lambda_m = np.ones(n_mono, dtype=np.float64)
    R = np.random.default_rng(0).uniform(0.0, 10.0, size=(n_atoms, 3))
    box = np.diag([20.0, 20.0, 20.0])

    pair_i = np.array([0, 1, 4, 5], dtype=np.int32)
    pair_j = np.array([4, 5, 0, 1], dtype=np.int32)
    pair_mask = np.ones(len(pair_i), dtype=np.float64)

    fake_psf = MagicMock()
    fake_psf.get_charges.return_value = np.zeros(n_atoms, dtype=np.float64)
    fake_psf.get_iac.return_value = np.ones(n_atoms, dtype=np.int32)
    fake_param = MagicMock()
    fake_param.get_atc.return_value = ["CG321", "HGA2"]

    rtf_mock = MagicMock()
    rtf_mock.readlines.return_value = ["ATOM C1 CG321 -0.1\n"]
    prm_mock = MagicMock()
    prm_mock.readlines.return_value = ["CG321 0.0 -0.1 3.5\n"]

    with patch(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces.have_jax_md",
        return_value=False,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces.have_vesin",
        return_value=False,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces._cell_list_pairs",
        return_value=object(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces.build_mm_pairs_with_backend",
        return_value=(pair_i, pair_j, pair_mask, len(pair_i), len(pair_i), "cell_list"),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces.resolve_mm_nl_backend",
        return_value="cell_list",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces.pick_static_rebuild_backend",
        return_value="cell_list",
    ), patch("pycharmm.psf", fake_psf), patch("pycharmm.param", fake_param), patch(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces.open",
        side_effect=[rtf_mock, prm_mock],
    ), patch(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces._get_actual_psf_charges",
        return_value=np.zeros(n_atoms, dtype=np.float64),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces.CGENFF_PRM",
        "/dev/null",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces.CGENFF_RTF",
        "/dev/null",
    ):
        mm_fn = build_mm_energy_forces_fn(
            R,
            total_atoms=n_atoms,
            n_monomers=n_mono,
            monomer_offsets=offsets,
            atoms_per_monomer_list=atoms_per,
            lambda_monomer=lambda_m,
            ml_switch_width=1.0,
            mm_switch_on=6.0,
            mm_switch_width=4.0,
            pbc_cell=box,
            use_jax_md_neighbor_list=False,
            mm_nl_backend="cell_list",
            force_static_mm_eval=True,
            lr_solver="mic",
            defer_xla_gpu_warmup=True,
            debug=False,
        )

    assert callable(mm_fn)
    energy, forces = jax.device_get(mm_fn(jnp.asarray(R)))
    assert np.isfinite(float(energy))
    assert forces.shape == (n_atoms, 3)
    assert np.all(np.isfinite(forces))
