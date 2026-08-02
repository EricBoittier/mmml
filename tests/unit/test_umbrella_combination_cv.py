"""Combination (antisymmetric-stretch) CVs through the umbrella stack.

The Menshutkin reaction coordinate ``xi = r(C-Cl) - r(C-N)`` is a two-term CV;
these tests pin the pieces the packed sampler and MBAR rely on and check that
the plain-distance path still behaves identically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmml.md.restraints import LinearDistanceCV
from mmml.umbrella.config import UmbrellaConfig, WindowSchedule
from mmml.umbrella.energy import (
    numpy_bias_matrix,
    numpy_bias_matrix_nd,
    packed_bias_energies,
    packed_bias_energies_nd,
    packed_bias_forces,
    packed_bias_forces_nd,
    packed_cv_values,
)

# Dataset atom order for this system is Cl(0), N(1), C(2).
_MENSHUTKIN_CV = LinearDistanceCV.difference(minuend=(2, 0), subtrahend=(2, 1))
_FRAME = np.array(
    [
        [0.0, 0.0, 0.0],  # Cl
        [4.8, 0.0, 0.0],  # N
        [1.8, 0.0, 0.0],  # C
    ],
    dtype=np.float64,
)


def _cfg(tmp_path: Path, **kwargs) -> UmbrellaConfig:
    base = dict(
        checkpoint=tmp_path / "ckpt.json",
        structure=tmp_path / "mol.xyz",
        output_dir=tmp_path / "out",
    )
    base.update(kwargs)
    return UmbrellaConfig(**base)


# --- config / schedule ------------------------------------------------------


def test_config_accepts_a_difference_cv_without_atom_indices(tmp_path):
    cfg = _cfg(tmp_path, cv_x=_MENSHUTKIN_CV, xi_min=-1.3, xi_max=1.6, n_windows=30)
    sched = cfg.resolve_schedule()
    assert sched.ndim == 1
    assert sched.n_windows == 30
    assert sched.cvs == (_MENSHUTKIN_CV,)
    assert sched.xi0[0] == pytest.approx(-1.3)
    assert sched.xi0[-1] == pytest.approx(1.6)


def test_config_without_cv_or_atoms_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="either cv_x or both atom_i and atom_j"):
        _cfg(tmp_path, targets_A=(1.0,))


def test_config_rejects_a_malformed_cv_spec_at_construction(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        _cfg(tmp_path, cv_x={"pairs": [[1, 1]], "coefficients": [1.0]}, targets_A=(1.0,))


def test_legacy_atom_pairs_still_build_distance_cvs(tmp_path):
    cfg = _cfg(tmp_path, atom_i=0, atom_j=2, targets_A=(1.8, 2.0))
    sched = cfg.resolve_schedule()
    assert sched.cvs == (LinearDistanceCV.distance(0, 2),)
    assert sched.atom_pairs == ((0, 2),)


def test_2d_mixes_a_difference_cv_with_a_plain_distance(tmp_path):
    cfg = _cfg(
        tmp_path,
        cv_x=_MENSHUTKIN_CV,
        atom_k=2,
        atom_l=1,
        cv_y=LinearDistanceCV.distance(2, 1),
        xi_min=-1.0,
        xi_max=1.0,
        n_windows=3,
        yi_min=1.5,
        yi_max=2.5,
        n_windows_y=2,
    )
    sched = cfg.resolve_schedule()
    assert sched.ndim == 2
    assert sched.n_windows == 6
    assert sched.cvs[0] == _MENSHUTKIN_CV
    assert sched.cvs[1].is_plain_distance


def test_schedule_backfills_cvs_from_atom_pairs():
    """Legacy WindowSchedule construction (no cvs) still yields distance CVs."""
    sched = WindowSchedule(
        ndim=1,
        atom_pairs=((0, 2),),
        xi0=(1.8,),
        yi0=None,
        k_x=(10.0,),
        k_y=None,
        grid_shape=(1,),
    )
    assert sched.cvs == (LinearDistanceCV.distance(0, 2),)
    assert sched.targets_per_cv == ((1.8,),)
    assert sched.k_per_cv == ((10.0,),)


def test_cv_specs_round_trip_through_from_spec(tmp_path):
    cfg = _cfg(tmp_path, cv_x=_MENSHUTKIN_CV, targets_A=(0.0,))
    specs = cfg.resolve_schedule().cv_specs()
    assert [LinearDistanceCV.from_spec(s) for s in specs] == [_MENSHUTKIN_CV]


def test_config_dict_round_trip_preserves_the_cv(tmp_path):
    cfg = _cfg(tmp_path, cv_x=_MENSHUTKIN_CV, targets_A=(0.0, 0.5))
    restored = UmbrellaConfig.from_dict(cfg.to_dict())
    assert restored.resolve_cvs() == (_MENSHUTKIN_CV,)


# --- packed energies / forces ----------------------------------------------


def test_packed_cv_values_track_the_difference():
    import jax.numpy as jnp

    frames = np.stack([_FRAME, _FRAME + 0.0])
    frames[1, 2, 0] = 3.2  # move C toward N: xi becomes positive
    packed = jnp.asarray(frames.reshape(-1, 3))
    values = np.asarray(packed_cv_values(packed, 3, _MENSHUTKIN_CV, 2))
    assert values[0] == pytest.approx(1.8 - 3.0)
    assert values[1] == pytest.approx(3.2 - 1.6)


def test_bias_is_minimised_at_the_window_center():
    import jax.numpy as jnp

    xi = _MENSHUTKIN_CV.value_numpy(_FRAME)
    packed = jnp.asarray(np.tile(_FRAME, (3, 1)))
    targets = (xi, xi + 0.5, xi - 0.5)
    ks = (10.0, 10.0, 10.0)
    bias = np.asarray(packed_bias_energies_nd(packed, 3, (_MENSHUTKIN_CV,), (targets,), (ks,)))
    assert bias[0] == pytest.approx(0.0)
    assert bias[1] == pytest.approx(0.5 * 10.0 * 0.25)
    assert bias[2] == pytest.approx(0.5 * 10.0 * 0.25)


def test_packed_bias_forces_match_autodiff_of_the_bias():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(3)
    frames = _FRAME[None] + 0.3 * rng.standard_normal((4, 3, 3))
    packed = jnp.asarray(frames.reshape(-1, 3))
    targets = (0.0, 0.2, -0.4, 1.0)
    ks = (10.0, 12.0, 8.0, 10.0)

    analytic = np.asarray(
        packed_bias_forces_nd(packed, 3, (_MENSHUTKIN_CV,), (targets,), (ks,))
    )

    def total_bias(r):
        return jnp.sum(
            packed_bias_energies_nd(r, 3, (_MENSHUTKIN_CV,), (targets,), (ks,))
        )

    autodiff = -np.asarray(jax.grad(total_bias)(packed))
    np.testing.assert_allclose(analytic, autodiff, rtol=1e-9, atol=1e-9)


def test_bias_forces_are_translation_invariant():
    """A pure CV bias must exert no net force on the system."""
    import jax.numpy as jnp

    packed = jnp.asarray(_FRAME)
    forces = np.asarray(
        packed_bias_forces_nd(packed, 3, (_MENSHUTKIN_CV,), ((0.0,),), ((10.0,),))
    )
    np.testing.assert_allclose(forces.sum(axis=0), np.zeros(3), atol=1e-9)


# --- backward compatibility of the plain-distance path ----------------------


def test_distance_cv_reproduces_the_legacy_packed_energy():
    import jax.numpy as jnp

    packed = jnp.asarray(np.tile(_FRAME, (2, 1)))
    targets = (1.5, 2.5)
    ks = (4.0, 4.0)
    legacy = np.asarray(packed_bias_energies(packed, 3, 2, 0, targets, ks))
    viacv = np.asarray(
        packed_bias_energies_nd(
            packed, 3, (LinearDistanceCV.distance(2, 0),), (targets,), (ks,)
        )
    )
    np.testing.assert_allclose(viacv, legacy, rtol=1e-12, atol=1e-12)


def test_distance_cv_reproduces_the_legacy_packed_forces():
    import jax.numpy as jnp

    packed = jnp.asarray(np.tile(_FRAME, (2, 1)))
    targets = (1.5, 2.5)
    ks = (4.0, 4.0)
    legacy = np.asarray(packed_bias_forces(packed, 3, 2, 0, targets, ks))
    viacv = np.asarray(
        packed_bias_forces_nd(
            packed, 3, (LinearDistanceCV.distance(2, 0),), (targets,), (ks,)
        )
    )
    np.testing.assert_allclose(viacv, legacy, rtol=1e-12, atol=1e-12)


def test_numpy_bias_matrix_nd_matches_the_legacy_distance_helper():
    targets = (1.5, 1.8, 2.5)
    ks = (4.0, 4.0, 4.0)
    legacy = numpy_bias_matrix(_FRAME, 2, 0, targets, ks)
    viacv = numpy_bias_matrix_nd(
        _FRAME, (LinearDistanceCV.distance(2, 0),), (targets,), (ks,)
    )
    np.testing.assert_allclose(viacv, legacy, rtol=1e-12, atol=1e-12)


def test_numpy_bias_matrix_nd_uses_the_full_difference_cv():
    """Guards the MBAR path: only the first pair would give a different answer."""
    targets = (0.0,)
    ks = (10.0,)
    full = numpy_bias_matrix_nd(_FRAME, (_MENSHUTKIN_CV,), (targets,), (ks,))
    first_pair_only = numpy_bias_matrix_nd(
        _FRAME, (LinearDistanceCV.distance(2, 0),), (targets,), (ks,)
    )
    assert full[0] == pytest.approx(0.5 * 10.0 * (1.8 - 3.0) ** 2)
    assert not np.isclose(full[0], first_pair_only[0])


# --- snapshot / MBAR CV recovery -------------------------------------------


def test_mbar_recovers_a_combination_cv_from_cv_spec():
    from mmml.umbrella.mbar import _snap_cvs

    snap = {
        "atom_i": 2,
        "atom_j": 0,
        "cv_spec": [
            {"pairs": [[2, 0], [2, 1]], "coefficients": [1.0, -1.0]},
        ],
    }
    assert _snap_cvs(snap) == [_MENSHUTKIN_CV]


def test_mbar_falls_back_to_atom_indices_for_legacy_snapshots():
    from mmml.umbrella.mbar import _snap_cvs

    assert _snap_cvs({"atom_i": 2, "atom_j": 0}) == [LinearDistanceCV.distance(2, 0)]
    assert _snap_cvs(
        {"atom_i": 2, "atom_j": 0, "atom_k": np.int32(2), "atom_l": np.int32(1)}
    ) == [LinearDistanceCV.distance(2, 0), LinearDistanceCV.distance(2, 1)]


def test_snapshot_cv_spec_survives_a_save_load_round_trip(tmp_path):
    import json

    from mmml.umbrella.io import load_snapshots, save_snapshots
    from mmml.umbrella.mbar import _snap_cvs

    cv_specs = [{"pairs": [[2, 0], [2, 1]], "coefficients": [1.0, -1.0]}]
    path = save_snapshots(
        tmp_path / "snap.npz",
        positions=np.zeros((2, 3, 3, 3)),
        Z=np.array([17, 7, 6], dtype=np.int32),
        atom_i=2,
        atom_j=0,
        xi0=np.array([-1.0, 1.0]),
        k_ev_A2=np.array([10.0, 10.0]),
        temperature_K=300.0,
        dt_fs=0.5,
        extra={"cv_spec": np.asarray(json.dumps(cv_specs))},
    )
    loaded = load_snapshots(path)
    assert _snap_cvs(loaded) == [_MENSHUTKIN_CV]


# --- window seeding ---------------------------------------------------------


def test_antisymmetric_seeding_hits_each_window_center():
    from mmml.umbrella.structure import pack_window_seeds

    targets = (-1.5, -0.5, 0.0, 0.5, 1.5)
    packed = pack_window_seeds(
        positions=_FRAME,
        atom_pairs=((2, 0),),
        targets_per_cv=(targets,),
        seed_mode="stretch",
        cvs=(_MENSHUTKIN_CV,),
    )
    seeds = packed.reshape(len(targets), 3, 3)
    achieved = _MENSHUTKIN_CV.values_numpy(seeds)
    np.testing.assert_allclose(achieved, targets, atol=1e-6)


def test_antisymmetric_seeding_keeps_both_distances_bonded_length():
    """xi alone is degenerate; seeds must stay compact, not dissociated."""
    from mmml.umbrella.structure import pack_window_seeds

    targets = (-1.5, 0.0, 1.5)
    packed = pack_window_seeds(
        positions=_FRAME,
        atom_pairs=((2, 0),),
        targets_per_cv=(targets,),
        seed_mode="stretch",
        cvs=(_MENSHUTKIN_CV,),
    )
    seeds = packed.reshape(len(targets), 3, 3)
    r_ccl = np.linalg.norm(seeds[:, 0] - seeds[:, 2], axis=-1)
    r_cn = np.linalg.norm(seeds[:, 1] - seeds[:, 2], axis=-1)
    assert np.all(r_ccl > 1.3)
    assert np.all(r_cn > 1.3)
    # Reference sum is preserved, so nothing drifts apart.
    np.testing.assert_allclose(r_ccl + r_cn, 1.8 + 3.0, atol=1e-6)


def test_stretch_seeding_refuses_a_cv_it_cannot_invert():
    from mmml.umbrella.structure import pack_window_seeds

    same_sign = LinearDistanceCV(pairs=((2, 0), (2, 1)), coefficients=(1.0, 1.0))
    with pytest.raises(ValueError, match="opposite-sign"):
        pack_window_seeds(
            positions=_FRAME,
            atom_pairs=((2, 0),),
            targets_per_cv=((4.0,),),
            seed_mode="stretch",
            cvs=(same_sign,),
        )
