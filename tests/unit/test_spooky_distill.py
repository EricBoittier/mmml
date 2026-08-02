"""Unit tests for SpookyPhysNet teacher distillation.

Two groups: the pure helpers (target parsing, teacher-architecture recovery,
energy-zero alignment), and end-to-end properties of the trainer's actual loss
-- the alpha endpoints and the fact that the teacher is gradient-blocked.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet
from mmml.models.physnetjax.physnetjax.training.spooky_distill import (
    EnergyAlignment,
    fit_energy_alignment,
    element_counts_from_atomic_numbers,
    parse_spooky_distill_targets,
    teacher_architecture_from_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_trainer():
    """Import the trainer script by path (it lives in scripts/, not a package)."""
    module_name = "_train_so3lr_spooky_extxyz_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = REPO_ROOT / "scripts" / "train_so3lr_spooky_extxyz.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Target parsing
# ---------------------------------------------------------------------------


def test_parse_targets_defaults_to_energy_and_forces():
    assert parse_spooky_distill_targets(None) == (True, True)
    assert parse_spooky_distill_targets(["energy", "forces"]) == (True, True)


def test_parse_targets_supports_each_channel_independently():
    assert parse_spooky_distill_targets(["energy"]) == (True, False)
    assert parse_spooky_distill_targets(["FORCES"]) == (False, True)


@pytest.mark.parametrize("target", ["dipole", "charges", "charge", "dipoles"])
def test_parse_targets_rejects_charge_and_dipole(target):
    """Charge/dipole must be refused, not silently dropped.

    Silently ignoring them would let a run report charge distillation it never
    performed, while the student's charge head is exactly what the campaign is
    trying to supervise from reference data.
    """
    with pytest.raises(ValueError, match="supervised by reference data"):
        parse_spooky_distill_targets([target])


def test_parse_targets_rejects_unknown_and_empty():
    with pytest.raises(ValueError, match="Unknown distillation target"):
        parse_spooky_distill_targets(["entropy"])
    with pytest.raises(ValueError, match="no usable target names"):
        parse_spooky_distill_targets([" "])


# ---------------------------------------------------------------------------
# Teacher architecture recovery
# ---------------------------------------------------------------------------


def test_architecture_comes_from_model_attributes_not_the_student():
    restored = {
        "model_attributes": {
            "features": 128,
            "max_degree": 2,
            "efa": False,
            "use_energy_bias": False,
            "charges": True,
            "n_refinement_blocks": 2,
            "zbl": True,
        }
    }
    arch = teacher_architecture_from_checkpoint(restored, max_padded_atoms=48)
    assert arch.source == "model_attributes"
    assert arch.kwargs["features"] == 128
    assert arch.kwargs["max_degree"] == 2
    assert arch.kwargs["efa"] is False
    assert arch.kwargs["use_energy_bias"] is False
    assert arch.kwargs["max_padded_atoms"] == 48


def test_architecture_falls_back_to_config_with_cli_aliases():
    """A params-only JSON export carries argparse flags, not model fields."""
    restored = {
        "config": {
            "features": 128,
            "max_degree": 2,
            "predict_charges": True,
            "n_res": 3,
            "no_zbl": False,
            "efa": False,
            "use_energy_bias": False,
            "fixed_cgenff_vdw": True,
        }
    }
    arch = teacher_architecture_from_checkpoint(restored, max_padded_atoms=16)
    assert arch.source == "config"
    assert arch.kwargs["charges"] is True
    assert arch.kwargs["n_refinement_blocks"] == 3
    assert arch.kwargs["zbl"] is True
    assert arch.kwargs["learn_cgenff_vdw_scale"] is False
    assert arch.kwargs["predict_atomic_vdw_scale"] is False


def test_architecture_uses_model_defaults_for_fields_the_checkpoint_predates():
    """Fields a checkpoint never recorded must fall back to the model default.

    They must NOT pick up the current CLI default or the student's value, or an
    older teacher would be silently rebuilt as a different model.
    """
    restored = {"config": {"features": 128}}
    arch = teacher_architecture_from_checkpoint(restored, max_padded_atoms=16)
    assert arch.kwargs["zbl_cutoff"] == 0.6
    assert arch.kwargs["switch_end"] == 10.0
    assert "zbl_cutoff" in arch.missing_fields
    assert "features" not in arch.missing_fields


def test_legacy_teacher_without_trainable_zbl_is_rebuilt_as_trainable():
    """Pre-trainable_zbl checkpoints used trainable ZBL parameters.

    Taking the field's current default (False) instead would build a module with
    no repulsion parameters, so the checkpoint's four would have nowhere to go.
    Mirrors the warm-start path's legacy inference.
    """
    arch = teacher_architecture_from_checkpoint(
        {"config": {"features": 128, "no_zbl": False}}, max_padded_atoms=16
    )
    assert arch.kwargs["zbl"] is True
    assert arch.kwargs["trainable_zbl"] is True
    assert "trainable_zbl" in arch.missing_fields


def test_legacy_inference_does_not_apply_when_zbl_is_off():
    arch = teacher_architecture_from_checkpoint(
        {"config": {"features": 128, "no_zbl": True}}, max_padded_atoms=16
    )
    assert arch.kwargs["zbl"] is False
    assert arch.kwargs["trainable_zbl"] is False


def test_recorded_trainable_zbl_is_never_overridden_by_the_legacy_guess():
    arch = teacher_architecture_from_checkpoint(
        {"model_attributes": {"zbl": True, "trainable_zbl": False}}, max_padded_atoms=16
    )
    assert arch.kwargs["trainable_zbl"] is False
    assert "trainable_zbl" not in arch.missing_fields


def test_architecture_coerces_restored_scalar_types():
    """Orbax gives back 0-d arrays and JSON turns bools into ints."""
    restored = {
        "model_attributes": {
            "features": np.int64(64),
            "efa": np.array(0),
            "charges": 1,
            "cutoff": np.array(5.5),
        }
    }
    arch = teacher_architecture_from_checkpoint(restored, max_padded_atoms=8)
    assert isinstance(arch.kwargs["features"], int)
    assert arch.kwargs["efa"] is False
    assert arch.kwargs["charges"] is True
    assert arch.kwargs["cutoff"] == pytest.approx(5.5)


def test_architecture_refuses_a_checkpoint_with_no_recorded_architecture():
    with pytest.raises(ValueError, match="cannot be confirmed"):
        teacher_architecture_from_checkpoint({"params": {}}, max_padded_atoms=8)


def test_differing_fields_reports_teacher_vs_student():
    arch = teacher_architecture_from_checkpoint(
        {"model_attributes": {"features": 128, "max_degree": 2}}, max_padded_atoms=8
    )
    diff = arch.differing_fields({"features": 16, "max_degree": 2})
    assert diff == {"features": (128, 16)}


# ---------------------------------------------------------------------------
# Energy-zero alignment
# ---------------------------------------------------------------------------


def _counts(rows, n_z=10):
    out = np.zeros((len(rows), n_z))
    for i, row in enumerate(rows):
        for z, n in row.items():
            out[i, z] = n
    return out


def test_atomic_alignment_recovers_a_known_per_element_offset():
    """The exact case this exists for: two caches whose atomic references differ."""
    rng = np.random.default_rng(0)
    rows = [{1: int(h), 8: int(o)} for h, o in rng.integers(1, 9, size=(40, 2))]
    counts = _counts(rows)
    true_offsets = np.zeros(10)
    true_offsets[1] = -0.37
    true_offsets[8] = 2.15
    teacher = rng.normal(size=len(rows)) * 3.0
    reference = teacher + counts @ true_offsets

    alignment = fit_energy_alignment(teacher, reference, counts, mode="atomic")

    assert alignment.mode == "atomic"
    assert alignment.fallback_reason is None
    assert alignment.element_offsets[1] == pytest.approx(-0.37, abs=1e-8)
    assert alignment.element_offsets[8] == pytest.approx(2.15, abs=1e-8)
    assert alignment.rms_after_eV == pytest.approx(0.0, abs=1e-8)
    assert alignment.rms_after_eV < alignment.rms_before_eV
    np.testing.assert_allclose(alignment.apply(teacher, counts), reference, atol=1e-8)


def test_scalar_alignment_is_the_mean_residual():
    counts = _counts([{1: 2, 8: 1}] * 5)
    teacher = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    reference = teacher + 7.0
    alignment = fit_energy_alignment(teacher, reference, counts, mode="scalar")
    assert alignment.mode == "scalar"
    assert alignment.scalar_offset == pytest.approx(7.0)
    assert np.count_nonzero(alignment.element_offsets) == 0
    assert alignment.rms_after_eV == pytest.approx(0.0, abs=1e-12)


def test_none_mode_applies_nothing_but_still_records_the_residual():
    counts = _counts([{1: 2, 8: 1}] * 4)
    teacher = np.zeros(4)
    reference = np.full(4, 5.0)
    alignment = fit_energy_alignment(teacher, reference, counts, mode="none")
    assert alignment.mode == "none"
    assert alignment.scalar_offset == 0.0
    assert alignment.mean_abs_shift_eV == 0.0
    # The offset was not applied, but the disagreement is still measured.
    assert alignment.rms_before_eV == pytest.approx(5.0)
    assert alignment.rms_after_eV == pytest.approx(5.0)
    np.testing.assert_allclose(alignment.apply(teacher, counts), teacher)


def test_atomic_alignment_falls_back_to_scalar_when_undersampled():
    counts = _counts([{1: 2, 8: 1}, {1: 4, 8: 2}])
    teacher = np.array([0.0, 1.0])
    reference = np.array([3.0, 4.5])
    alignment = fit_energy_alignment(teacher, reference, counts, mode="atomic")
    assert alignment.mode == "scalar"
    assert alignment.requested_mode == "atomic"
    assert "calibration structures" in alignment.fallback_reason


def test_alignment_metadata_is_json_serializable():
    import json

    counts = _counts([{1: 2, 8: 1}] * 6)
    alignment = fit_energy_alignment(np.zeros(6), np.ones(6), counts, mode="scalar")
    payload = json.dumps(alignment.to_metadata())
    assert "element_offsets" in payload
    assert json.loads(payload)["mode"] == "scalar"


def test_alignment_rejects_mismatched_shapes_and_unknown_modes():
    counts = _counts([{1: 2}] * 3)
    with pytest.raises(ValueError, match="agree on the number"):
        fit_energy_alignment(np.zeros(3), np.zeros(2), counts, mode="scalar")
    with pytest.raises(ValueError, match="Unknown energy-alignment mode"):
        fit_energy_alignment(np.zeros(3), np.zeros(3), counts, mode="quadratic")


def test_element_counts_ignore_padded_atoms():
    z = np.array([[1, 1, 8, 0], [6, 8, 0, 0]])
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]])
    counts = element_counts_from_atomic_numbers(z, mask, n_z=10)
    assert counts[0, 1] == 2
    assert counts[0, 8] == 1
    assert counts[0, 0] == 0  # the padded slot must not be counted as element 0
    assert counts[1, 6] == 1
    assert counts[1, 8] == 1
    assert counts.sum() == 5


# ---------------------------------------------------------------------------
# Trainer loss: alpha endpoints and teacher gradient blocking
# ---------------------------------------------------------------------------

N_ATOMS = 6
BATCH = 2


def _tiny_args(**overrides):
    args = argparse.Namespace(
        batch_size_per_device=BATCH,
        energy_weight=1.0,
        forces_weight=2.0,
        dipole_weight=0.0,
        charges_weight=0.0,
        mbd_weight=0.0,
        mbd_ramp_steps=0,
        multipole_consistency_weight=0.0,
        neural_interaction_l2=0.0,
        far_field_charge_weight=0.0,
        far_field_max_k=4,
        far_field_augment_fraction=0.0,
        predict_charges=False,
        no_cgenff_vdw=True,
        interaction_trust_map=False,
        trust_map_evidence=1.0,
        trust_map_hyperprior=1.0,
        cutoff=5.0,
        distill_alpha=0.75,
        distill_targets=("energy", "forces"),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _tiny_model(**overrides):
    kwargs = dict(
        features=8,
        max_degree=1,
        num_iterations=1,
        num_basis_functions=8,
        cutoff=5.0,
        max_atomic_number=10,
        charges=False,
        max_padded_atoms=N_ATOMS,
        n_refinement_blocks=1,
        zbl=False,
    )
    kwargs.update(overrides)
    return SpookyPhysNet(**kwargs)


def _tiny_batch(seed=0):
    rng = np.random.default_rng(seed)
    n = BATCH * N_ATOMS
    dst, src = zip(*[(i, j) for i in range(n) for j in range(n) if i != j], strict=True)
    batch_segments = np.repeat(np.arange(BATCH), N_ATOMS)
    edge_mask = (batch_segments[list(dst)] == batch_segments[list(src)]).astype(np.float32)
    return {
        "Z": jnp.asarray(np.tile([1, 1, 8, 6, 1, 1], BATCH), dtype=jnp.int32),
        "R": jnp.asarray(rng.normal(size=(n, 3)) * 1.5, dtype=jnp.float32),
        "F": jnp.asarray(rng.normal(size=(n, 3)) * 0.1, dtype=jnp.float32),
        "E": jnp.asarray(rng.normal(size=(BATCH, 1)), dtype=jnp.float32),
        "D": jnp.zeros((BATCH, 3), dtype=jnp.float32),
        "Q_atoms": jnp.zeros(n, dtype=jnp.float32),
        "S_atoms": jnp.zeros(n, dtype=jnp.float32),
        "Q_total": jnp.zeros((BATCH, 1), dtype=jnp.float32),
        "S_total": jnp.ones((BATCH, 1), dtype=jnp.float32),
        "dst_idx": jnp.asarray(dst, dtype=jnp.int32),
        "src_idx": jnp.asarray(src, dtype=jnp.int32),
        "batch_segments": jnp.asarray(batch_segments, dtype=jnp.int32),
        "batch_mask": jnp.asarray(edge_mask),
        "atom_mask": jnp.ones(n, dtype=jnp.float32),
    }


def _init_params(model, batch):
    return model.init(
        jax.random.PRNGKey(0),
        atomic_numbers=batch["Z"],
        charges=batch["Q_atoms"],
        spins=batch["S_atoms"],
        positions=batch["R"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=BATCH,
        batch_mask=batch["batch_mask"],
        atom_mask=batch["atom_mask"],
        compute_forces=False,
    )


@pytest.fixture(scope="module")
def distill_fixture():
    trainer = _load_trainer()
    from mmml.models.physnetjax.physnetjax.training.spooky_distill import (
        fit_energy_alignment as _fit,
    )

    batch = _tiny_batch()
    student = _tiny_model()
    # Deliberately a *different* architecture from the student, matching the
    # real campaign (a wider, higher-degree teacher distilled into a small
    # charge-aware student).
    teacher = _tiny_model(features=16, num_iterations=2)
    student_params = _init_params(student, batch)
    # Spooky zero-initializes its energy head, so a freshly-initialized teacher
    # predicts exactly 0 eV and would agree with the student by construction --
    # every distillation assertion below would then pass vacuously. Jitter every
    # leaf (including the zeroed head) so the teacher is a genuinely different
    # function, which is what the endpoints are supposed to be tested against.
    jitter_keys = jax.random.split(jax.random.PRNGKey(7), len(jax.tree_util.tree_leaves(
        _init_params(teacher, batch)
    )))
    key_iter = iter(jitter_keys)
    teacher_params = jax.tree.map(
        lambda leaf: leaf + 0.1 * jax.random.normal(next(key_iter), leaf.shape, leaf.dtype),
        _init_params(teacher, batch),
    )
    identity_alignment = _fit(
        np.zeros(1), np.zeros(1), np.zeros((1, 11)), mode="none"
    )
    return {
        "trainer": trainer,
        "batch": batch,
        "student": student,
        "teacher": teacher,
        "student_params": student_params,
        "teacher_params": teacher_params,
        "alignment": identity_alignment,
    }


def _loss_fn(fixture, *, with_teacher, **arg_overrides):
    trainer = fixture["trainer"]
    args = _tiny_args(**arg_overrides)
    kwargs = {}
    if with_teacher:
        kwargs = {
            "teacher_model": fixture["teacher"],
            "teacher_params": fixture["teacher_params"],
            "teacher_alignment": fixture["alignment"],
            "teacher_no_cgenff_vdw": True,
        }
    *_, loss_fn = trainer.make_steps(
        fixture["student"],
        args,
        jax.devices()[:1],
        return_loss_fn=True,
        **kwargs,
    )
    return loss_fn


def test_alpha_one_reproduces_undistilled_training_exactly(distill_fixture):
    """alpha=1.0 is the ground-truth endpoint: the teacher must not move the loss."""
    params, batch = distill_fixture["student_params"], distill_fixture["batch"]
    plain = _loss_fn(distill_fixture, with_teacher=False)
    distilled = _loss_fn(distill_fixture, with_teacher=True, distill_alpha=1.0)

    plain_loss, plain_metrics = plain(params, batch, jnp.asarray(1.0))
    distilled_loss, distilled_metrics = distilled(params, batch, jnp.asarray(1.0))

    assert float(distilled_loss) == pytest.approx(float(plain_loss), rel=1e-6)
    for key in ("energy_mae", "forces_mae", "energy_mse", "forces_mse"):
        assert float(distilled_metrics[key]) == pytest.approx(
            float(plain_metrics[key]), rel=1e-6
        )
    # ...and the gradients too, which is what actually drives training.
    plain_grads = jax.grad(lambda p: plain(p, batch, jnp.asarray(1.0))[0])(params)
    distilled_grads = jax.grad(lambda p: distilled(p, batch, jnp.asarray(1.0))[0])(params)
    for a, b in zip(
        jax.tree_util.tree_leaves(plain_grads),
        jax.tree_util.tree_leaves(distilled_grads),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7)


def test_alpha_zero_is_the_pure_teacher_endpoint(distill_fixture):
    """alpha=0.0 must drop the reference labels entirely."""
    params, batch = distill_fixture["student_params"], distill_fixture["batch"]
    distilled = _loss_fn(distill_fixture, with_teacher=True, distill_alpha=0.0)
    loss, metrics = distilled(params, batch, jnp.asarray(1.0))

    args = _tiny_args()
    expected = (
        args.energy_weight * float(metrics["distill_energy_mse"])
        + args.forces_weight * float(metrics["distill_forces_mse"])
    )
    assert float(loss) == pytest.approx(expected, rel=1e-6)
    # The teacher is a different model, so it genuinely disagrees with both the
    # student and the reference -- otherwise this endpoint would be vacuous.
    assert float(metrics["distill_energy_mse"]) > 0.0
    assert float(metrics["distill_forces_mse"]) > 0.0


def test_intermediate_alpha_is_the_recorded_convex_blend(distill_fixture):
    params, batch = distill_fixture["student_params"], distill_fixture["batch"]
    alpha = 0.75
    distilled = _loss_fn(distill_fixture, with_teacher=True, distill_alpha=alpha)
    loss, metrics = distilled(params, batch, jnp.asarray(1.0))

    args = _tiny_args()
    expected_energy = alpha * float(metrics["energy_mse"]) + (1 - alpha) * float(
        metrics["distill_energy_mse"]
    )
    expected_forces = alpha * float(metrics["forces_mse"]) + (1 - alpha) * float(
        metrics["distill_forces_mse"]
    )
    expected = args.energy_weight * expected_energy + args.forces_weight * expected_forces
    assert float(loss) == pytest.approx(expected, rel=1e-6)
    assert float(metrics["distill_alpha"]) == pytest.approx(alpha)


def test_teacher_parameters_receive_no_gradient(distill_fixture):
    """The teacher is frozen reference physics: d(loss)/d(teacher_params) == 0.

    Differentiating through make_steps itself is what makes this a real proof --
    the teacher parameters enter as a traced value, so any missing
    stop_gradient would show up as a non-zero leaf here.
    """
    trainer = distill_fixture["trainer"]
    params, batch = distill_fixture["student_params"], distill_fixture["batch"]
    args = _tiny_args(distill_alpha=0.0)

    def loss_of_teacher_params(teacher_params):
        *_, loss_fn = trainer.make_steps(
            distill_fixture["student"],
            args,
            jax.devices()[:1],
            teacher_model=distill_fixture["teacher"],
            teacher_params=teacher_params,
            teacher_alignment=distill_fixture["alignment"],
            teacher_no_cgenff_vdw=True,
            return_loss_fn=True,
        )
        return loss_fn(params, batch, jnp.asarray(1.0))[0]

    grads = jax.grad(loss_of_teacher_params)(distill_fixture["teacher_params"])
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves, "teacher parameter tree was empty -- the test proves nothing"
    for leaf in leaves:
        np.testing.assert_array_equal(np.asarray(leaf), np.zeros_like(np.asarray(leaf)))

    # Control: the same loss IS sensitive to the student parameters, so the
    # zeros above are gradient blocking rather than a dead loss.
    student_grads = jax.grad(lambda p: _loss_fn(
        distill_fixture, with_teacher=True, distill_alpha=0.0
    )(p, batch, jnp.asarray(1.0))[0])(params)
    assert max(
        float(np.abs(np.asarray(leaf)).max())
        for leaf in jax.tree_util.tree_leaves(student_grads)
    ) > 0.0


def test_energy_alignment_shifts_the_teacher_energy_target(distill_fixture):
    """A recorded per-element offset must actually move the distilled target."""
    trainer = distill_fixture["trainer"]
    params, batch = distill_fixture["student_params"], distill_fixture["batch"]

    offsets = np.zeros(11)
    offsets[8] = 3.0
    alignment = EnergyAlignment(
        mode="atomic",
        scalar_offset=0.0,
        element_offsets=offsets,
        n_samples=BATCH,
        rms_before_eV=3.0,
        rms_after_eV=0.0,
        mean_abs_shift_eV=3.0,
    )

    args = _tiny_args(distill_alpha=0.0)
    shifted = trainer.make_steps(
        distill_fixture["student"],
        args,
        jax.devices()[:1],
        teacher_model=distill_fixture["teacher"],
        teacher_params=distill_fixture["teacher_params"],
        teacher_alignment=alignment,
        teacher_no_cgenff_vdw=True,
        return_loss_fn=True,
    )[-1]
    unshifted = _loss_fn(distill_fixture, with_teacher=True, distill_alpha=0.0)

    _, shifted_metrics = shifted(params, batch, jnp.asarray(1.0))
    _, unshifted_metrics = unshifted(params, batch, jnp.asarray(1.0))

    assert float(shifted_metrics["distill_energy_mse"]) != pytest.approx(
        float(unshifted_metrics["distill_energy_mse"]), rel=1e-6
    )
    # Forces are invariant to an energy-zero shift, by construction.
    assert float(shifted_metrics["distill_forces_mse"]) == pytest.approx(
        float(unshifted_metrics["distill_forces_mse"]), rel=1e-6
    )


def test_forces_only_distillation_leaves_the_energy_term_on_ground_truth(distill_fixture):
    params, batch = distill_fixture["student_params"], distill_fixture["batch"]
    distilled = _loss_fn(
        distill_fixture, with_teacher=True, distill_alpha=0.0, distill_targets=("forces",)
    )
    loss, metrics = distilled(params, batch, jnp.asarray(1.0))
    args = _tiny_args()
    expected = (
        args.energy_weight * float(metrics["energy_mse"])
        + args.forces_weight * float(metrics["distill_forces_mse"])
    )
    assert float(loss) == pytest.approx(expected, rel=1e-6)
    assert float(metrics["distill_energy_mse"]) == 0.0


def test_make_steps_rejects_an_out_of_range_alpha(distill_fixture):
    with pytest.raises(ValueError, match="must be in"):
        _loss_fn(distill_fixture, with_teacher=True, distill_alpha=1.5)


def test_make_steps_requires_teacher_model_and_params_together(distill_fixture):
    trainer = distill_fixture["trainer"]
    with pytest.raises(ValueError, match="must be provided together"):
        trainer.make_steps(
            distill_fixture["student"],
            _tiny_args(),
            jax.devices()[:1],
            teacher_model=distill_fixture["teacher"],
        )


def test_make_steps_requires_an_alignment_when_a_teacher_is_given(distill_fixture):
    trainer = distill_fixture["trainer"]
    with pytest.raises(ValueError, match="teacher_alignment is required"):
        trainer.make_steps(
            distill_fixture["student"],
            _tiny_args(),
            jax.devices()[:1],
            teacher_model=distill_fixture["teacher"],
            teacher_params=distill_fixture["teacher_params"],
        )
