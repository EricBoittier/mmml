"""Finite-difference Jacobian checks and linear-readout equivalence."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.acquisition.linear_student import init_linear_student, init_linear_teacher
from mmml.acquisition.physnet_readout import architecture_equivalence_note, flatten_params
from mmml.acquisition.representations import linear_readout_pooled_equals_energy_grad
from mmml.acquisition.splits import StructureRecord


def _water():
    z = np.array([8, 1, 1], dtype=np.int32)
    r = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=np.float64)
    return r, z


def _record():
    r, z = _water()
    return StructureRecord(
        index=0,
        structure_id="w",
        geometry_fingerprint="w",
        composition="H2O1",
        stratum="H2O1|T300",
        group="g",
        n_atoms=3,
        atomic_numbers=z,
        positions=r,
    )


def test_energy_jacobian_matches_finite_difference():
    student = init_linear_student(seed=0)
    r, z = _water()
    g = student.energy_jacobian(r, z)
    packed = student.pack_readout()
    eps = 1e-5
    numeric = np.zeros_like(packed)
    e0, _ = student.energy_forces(r, z)
    for i in range(len(packed)):
        step = packed.copy()
        step[i] += eps
        e1, _ = student.with_readout(step).energy_forces(r, z)
        numeric[i] = (e1 - e0) / eps
    np.testing.assert_allclose(g, numeric, rtol=2e-3, atol=2e-4)


def test_force_jacobian_matches_finite_difference():
    student = init_linear_student(seed=1)
    r, z = _water()
    j = student.force_jacobian(r, z)
    packed = student.pack_readout()
    eps = 1e-5
    _, f0 = student.energy_forces(r, z)
    numeric = np.zeros_like(j)
    for i in range(len(packed)):
        step = packed.copy()
        step[i] += eps
        _, f1 = student.with_readout(step).energy_forces(r, z)
        numeric[:, i] = (f1 - f0).reshape(-1) / eps
    np.testing.assert_allclose(j, numeric, rtol=3e-3, atol=3e-4)


def test_sum_pooled_activations_equal_energy_gradient_for_linear_readout():
    student = init_linear_student(seed=2)
    recs = [_record()]
    stats = linear_readout_pooled_equals_energy_grad(student, recs)
    assert stats["dims_equal"]
    assert stats["mean_cosine"] == pytest.approx(1.0, abs=1e-8)
    assert stats["mean_relative_l2"] == pytest.approx(0.0, abs=1e-8)


def test_loss_gradient_vanishes_if_teacher_equals_student():
    student = init_linear_student(seed=3)
    r, z = _water()
    e, f = student.energy_forces(r, z)
    g = student.loss_gradient(
        r, z, energy_target=e, forces_target=f, energy_weight=1.0, forces_weight=52.91
    )
    np.testing.assert_allclose(g, 0.0, atol=1e-10)


def test_loss_gradient_is_nonzero_against_a_different_teacher():
    student = init_linear_student(seed=3)
    teacher = init_linear_teacher(student, seed=9)
    r, z = _water()
    e_t, f_t = teacher.energy_forces(r, z)
    g = student.loss_gradient(
        r, z, energy_target=e_t, forces_target=f_t, energy_weight=1.0, forces_weight=52.91
    )
    assert np.linalg.norm(g) > 1e-6


def test_information_block_row_count_matches_energy_plus_forces():
    student = init_linear_student(seed=0)
    r, z = _water()
    J = student.information_block(r, z, energy_weight=1.0, forces_weight=52.91)
    assert J.shape[0] == 1 + 3 * 3
    assert J.shape[1] == student.n_readout_params()


def test_physnet_architecture_note_does_not_claim_equivalence():
    note = architecture_equivalence_note()
    assert "not" in note.lower() or "only" in note.lower()
    # flatten_params is a tree walker used by the adapter
    flat = flatten_params({"params": {"energy_bias": np.zeros(4), "Dense_0": {"kernel": np.ones((2, 1))}}})
    assert any("energy_bias" in k for k in flat)
