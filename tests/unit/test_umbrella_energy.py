"""Unit tests for umbrella packed-graph and bias energies."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.umbrella.config import UmbrellaConfig
from mmml.umbrella.energy import (
    bias_energy,
    build_packed_graph,
    cv_distance,
    make_packed_energy_fn,
    numpy_bias_matrix,
    pack_positions,
    packed_bias_energies,
    packed_cv_distances,
)


def test_cv_and_bias_match_distance_restraint():
    import jax.numpy as jnp

    from mmml.md.restraints import DistanceRestraint

    pos = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64)
    d = float(cv_distance(pos, 0, 1))
    assert d == pytest.approx(2.0)
    e = float(bias_energy(d, target=1.0, k_ev_A2=2.0))
    assert e == pytest.approx(1.0)
    assert e == pytest.approx(
        float(DistanceRestraint((0, 1), target_A=1.0, k_ev_A2=2.0).energy(pos))
    )


def test_packed_bias_increases_away_from_target():
    import jax.numpy as jnp

    n_atoms = 2
    k = 3
    r0 = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
    packed = pack_positions(r0, k)
    # Stretch window 1 and 2 along x
    packed = packed.reshape(k, n_atoms, 3).copy()
    packed[1, 1, 0] = 2.0
    packed[2, 1, 0] = 3.0
    packed = packed.reshape(k * n_atoms, 3)

    targets = (1.5, 1.5, 1.5)
    ks = (4.0, 4.0, 4.0)
    biases = np.asarray(
        packed_bias_energies(jnp.asarray(packed), n_atoms, 0, 1, targets, ks)
    )
    assert biases[0] == pytest.approx(0.0)
    assert biases[1] > biases[0]
    assert biases[2] > biases[1]


def test_make_packed_energy_fn_bias_only_with_zero_ml():
    import jax
    import jax.numpy as jnp

    n_atoms = 2
    n_windows = 2
    graph = build_packed_graph(n_atoms, n_windows)
    z = np.array([1, 1], dtype=np.int32)
    targets = (1.0, 2.0)
    ks = (2.0, 2.0)

    def fake_apply(params, **kwargs):
        del params
        b = int(kwargs["batch_size"])
        return {"energy": jnp.zeros((b,), dtype=jnp.float64)}

    energy_fn = make_packed_energy_fn(
        model_apply=fake_apply,
        params={},
        atomic_numbers=z,
        graph=graph,
        atom_pairs=((0, 1),),
        targets_per_cv=(targets,),
        k_per_cv=(ks,),
    )

    # Window 0 at r=1, window 1 at r=2 → both biases zero
    pos = np.zeros((n_windows, n_atoms, 3), dtype=np.float64)
    pos[0, 1, 0] = 1.0
    pos[1, 1, 0] = 2.0
    e0 = float(energy_fn(jnp.asarray(pos.reshape(-1, 3))))
    assert e0 == pytest.approx(0.0)

    # Move window 0 away from its target
    pos[0, 1, 0] = 2.0
    e1 = float(energy_fn(jnp.asarray(pos.reshape(-1, 3))))
    assert e1 == pytest.approx(1.0)  # 0.5*2*(1)^2


def test_numpy_bias_matrix():
    r = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    w = numpy_bias_matrix(r, 0, 1, targets_A=(1.0, 2.0, 3.0), k_ev_A2=(2.0, 2.0, 2.0))
    assert w.shape == (3,)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.0)
    assert w[2] == pytest.approx(1.0)


def test_umbrella_config_grid_and_validation(tmp_path):
    cfg = UmbrellaConfig(
        checkpoint=tmp_path / "ckpt",
        structure=tmp_path / "mol.xyz",
        output_dir=tmp_path / "out",
        atom_i=0,
        atom_j=1,
        xi_min=1.0,
        xi_max=3.0,
        n_windows=5,
        k_ev_A2=10.0,
    )
    targets = cfg.resolve_targets()
    assert len(targets) == 5
    assert targets[0] == pytest.approx(1.0)
    assert targets[-1] == pytest.approx(3.0)
    assert len(cfg.resolve_force_constants()) == 5

    with pytest.raises(ValueError, match="distinct"):
        UmbrellaConfig(
            checkpoint=tmp_path / "ckpt",
            structure=tmp_path / "mol.xyz",
            output_dir=tmp_path / "out",
            atom_i=0,
            atom_j=0,
            targets_A=(1.0,),
        )


def test_umbrella_config_2d_product_grid(tmp_path):
    cfg = UmbrellaConfig(
        checkpoint=tmp_path / "ckpt",
        structure=tmp_path / "mol.xyz",
        output_dir=tmp_path / "out",
        atom_i=0,
        atom_j=2,
        atom_k=1,
        atom_l=2,
        xi_min=1.5,
        xi_max=2.5,
        n_windows=3,
        yi_min=1.8,
        yi_max=2.8,
        n_windows_y=2,
        k_ev_A2=10.0,
        k_y_ev_A2=12.0,
    )
    sched = cfg.resolve_schedule()
    assert sched.ndim == 2
    assert sched.n_windows == 6
    assert sched.grid_shape == (3, 2)
    assert sched.yi0 is not None
    assert len(sched.yi0) == 6
    assert sched.k_y is not None
    assert all(k == 12.0 for k in sched.k_y)


def test_packed_2d_bias_sum():
    import jax.numpy as jnp

    from mmml.umbrella.energy import packed_bias_energies_nd

    # 3 atoms: hub at 0, others at 1 and 2
    n_atoms, k = 3, 1
    pos = np.zeros((k * n_atoms, 3))
    pos[1, 0] = 2.0
    pos[2, 0] = 3.0
    # targets for CV (0,1)=2 and (0,2)=3 → zero bias
    w0 = packed_bias_energies_nd(
        jnp.asarray(pos),
        n_atoms,
        ((0, 1), (0, 2)),
        ((2.0,), (3.0,)),
        ((4.0,), (4.0,)),
    )
    assert float(w0[0]) == pytest.approx(0.0)
    # wrong targets → nonzero
    w1 = packed_bias_energies_nd(
        jnp.asarray(pos),
        n_atoms,
        ((0, 1), (0, 2)),
        ((1.0,), (3.0,)),
        ((2.0,), (0.0,)),
    )
    assert float(w1[0]) == pytest.approx(1.0)  # 0.5*2*(1)^2


def test_packed_cv_distances_shape():
    import jax.numpy as jnp

    n_atoms, k = 3, 4
    pos = jnp.zeros((k * n_atoms, 3))
    # set atom 2 of each window
    for i in range(k):
        pos = pos.at[i * n_atoms + 2, 0].set(float(i + 1))
    d = packed_cv_distances(pos, n_atoms, 0, 2, k)
    assert d.shape == (k,)
    np.testing.assert_allclose(np.asarray(d), [1.0, 2.0, 3.0, 4.0])
