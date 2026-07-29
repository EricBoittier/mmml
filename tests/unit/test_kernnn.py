"""Unit tests for mmml.models.kernnn (JAX KerNN)."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms

from mmml.models.kernnn import (
    DualFFNet,
    FFNet,
    KerNNApplyAdapter,
    KerNNCalculator,
    KerNNConfig,
    KerNNStats,
    build_kernnn_batch_apply,
    energy_and_forces,
    get_1d_kernels_k33,
    get_bond_length_abcc,
    get_bond_length_abcc_sym,
    get_bond_length_acem,
    get_bond_length_form,
    init_params,
    is_kernnn_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from mmml.models.kernnn.checkpoint import torch_state_dict_to_flax_params
from mmml.models.kernnn.evaluate import build_parser as build_eval_parser
from mmml.models.kernnn.training import build_parser as build_train_parser
from mmml.neb.run import _default_calculator_factory


def _h2co_geometry() -> np.ndarray:
    """Simple planar formaldehyde-like geometry (Å), C O H H order."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.21, 0.0, 0.0],
            [-0.55, 0.94, 0.0],
            [-0.55, -0.94, 0.0],
        ],
        dtype=np.float32,
    )


def _stats_for(pos, n_feat: int = 6) -> KerNNStats:
    if n_feat == 6:
        r0 = get_bond_length_abcc(jnp.asarray(pos))
    else:
        r0 = get_bond_length_abcc_sym(jnp.asarray(pos))
    return KerNNStats(
        mean_e=0.0,
        std_e=1.0,
        min_r=np.asarray(r0),
        mean_k=np.zeros(n_feat, dtype=np.float32),
        std_k=np.ones(n_feat, dtype=np.float32),
    )


def test_abcc_distances_single_and_batch():
    pos = _h2co_geometry()
    r = get_bond_length_abcc(jnp.asarray(pos))
    assert r.shape == (6,)
    np.testing.assert_allclose(r[0], 1.21, rtol=1e-5)
    np.testing.assert_allclose(r[5], 1.88, rtol=1e-3)

    batch = jnp.stack([jnp.asarray(pos), jnp.asarray(pos) * 1.01], axis=0)
    r_b = get_bond_length_abcc(batch)
    assert r_b.shape == (2, 6)
    np.testing.assert_allclose(r_b[0], r, rtol=1e-5)


def test_abcc_sym_feature_count_and_permutation():
    pos = jnp.asarray(_h2co_geometry())
    r = get_bond_length_abcc_sym(pos)
    assert r.shape == (7,)
    # Swap H atoms → features invariant
    swapped = pos.at[2].set(pos[3]).at[3].set(pos[2])
    r2 = get_bond_length_abcc_sym(swapped)
    np.testing.assert_allclose(np.asarray(r), np.asarray(r2), rtol=1e-5)


def test_k33_matches_hand_formula():
    x = jnp.array([1.5, 2.0], dtype=jnp.float32)
    xi = jnp.array([1.2, 2.5], dtype=jnp.float32)
    k = get_1d_kernels_k33(x, xi)
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    expected = (
        3.0 / (20.0 * xl**4)
        - 6.0 / 35.0 * xs / xl**5
        + 3.0 / 56.0 * xs**2 / xl**6
    )
    np.testing.assert_allclose(np.asarray(k), np.asarray(expected), rtol=1e-6)


def test_ffnet_forward_shape():
    key = jax.random.key(0)
    config = KerNNConfig(n_input=6, n_hidden=8, n_out=1)
    params = init_params(key, config=config)
    model = FFNet(n_input=6, n_hidden=8, n_out=1)
    x = jnp.zeros((4, 6), dtype=jnp.float32)
    y = model.apply({"params": params}, x)
    assert y.shape == (4, 1)


def test_dual_ffnet_and_energy():
    key = jax.random.key(7)
    config = KerNNConfig(
        n_input=6, n_hidden=8, architecture="dual", distance_scheme="abcc"
    )
    params = init_params(key, config=config)
    assert "kernel_0" in params
    assert "dihedral_0" in params
    pos = jnp.asarray(_h2co_geometry())
    # Break planarity so dihedral is nonzero
    pos = pos.at[2, 2].set(0.2)
    stats = _stats_for(pos)
    e, f = energy_and_forces(params, pos, stats, config=config)
    assert np.isfinite(float(e))
    assert f.shape == (4, 3)


def test_energy_and_forces_fd_consistency():
    key = jax.random.key(1)
    config = KerNNConfig(n_input=6, n_hidden=8, n_out=1)
    params = init_params(key, config=config)
    pos = jnp.asarray(_h2co_geometry())
    stats = _stats_for(pos)
    energy, forces = energy_and_forces(params, pos, stats, config=config)
    assert np.asarray(energy).ndim == 0
    assert forces.shape == (4, 3)

    eps = 1e-3
    fd = np.zeros_like(np.asarray(forces))
    pos_np = np.asarray(pos)
    for i in range(4):
        for j in range(3):
            plus = pos_np.copy()
            minus = pos_np.copy()
            plus[i, j] += eps
            minus[i, j] -= eps
            e_p, _ = energy_and_forces(params, jnp.asarray(plus), stats, config=config)
            e_m, _ = energy_and_forces(params, jnp.asarray(minus), stats, config=config)
            fd[i, j] = -(float(e_p) - float(e_m)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(forces), fd, rtol=2e-2, atol=2e-3)


def test_checkpoint_roundtrip(tmp_path: Path):
    key = jax.random.key(2)
    config = KerNNConfig(n_hidden=8)
    params = init_params(key, config=config)
    stats = KerNNStats(
        mean_e=-1.0,
        std_e=0.5,
        min_r=np.ones(6),
        mean_k=np.zeros(6),
        std_k=np.ones(6),
    )
    path = tmp_path / "best.json"
    save_checkpoint(path, params=params, config=config, stats=stats, metadata={"epoch": 1})
    loaded_p, loaded_c, loaded_s, meta = load_checkpoint(path)
    assert loaded_c.n_hidden == 8
    assert loaded_s.mean_e == -1.0
    assert meta["epoch"] == 1
    assert is_kernnn_checkpoint(path)
    for a, b in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(loaded_p),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5)


def test_calculator_smoke(tmp_path: Path):
    key = jax.random.key(3)
    config = KerNNConfig(n_hidden=8)
    params = init_params(key, config=config)
    pos = _h2co_geometry()
    stats = _stats_for(pos)
    path = tmp_path / "model.json"
    save_checkpoint(path, params=params, config=config, stats=stats)

    atoms = Atoms(symbols=["C", "O", "H", "H"], positions=pos)
    atoms.calc = KerNNCalculator(path)
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    assert np.isfinite(e)
    assert f.shape == (4, 3)

    calc2 = KerNNCalculator.from_components(params, stats, config)
    atoms2 = Atoms(symbols=["C", "O", "H", "H"], positions=pos)
    atoms2.calc = calc2
    np.testing.assert_allclose(atoms2.get_potential_energy(), e, rtol=1e-5)


def test_apply_adapter_and_batch_apply(tmp_path: Path):
    key = jax.random.key(4)
    config = KerNNConfig(n_hidden=8)
    params = init_params(key, config=config)
    pos = _h2co_geometry()
    stats = _stats_for(pos)
    adapter = KerNNApplyAdapter(stats=stats, config=config, n_atoms=4)
    out = adapter.apply(params, positions=jnp.asarray(pos), compute_forces=True)
    assert "energy" in out and "forces" in out

    packed = jnp.concatenate([jnp.asarray(pos), jnp.asarray(pos) * 1.01], axis=0)
    out2 = adapter.apply(
        params, positions=packed, batch_size=2, compute_forces=True
    )
    assert out2["energy"].shape == (2,)
    assert out2["forces"].shape == (8, 3)

    apply_fn = build_kernnn_batch_apply(
        params=params, stats=stats, config=config, max_atoms=4, atoms_per_monomer=4
    )
    z = jnp.zeros(4, dtype=jnp.int32)
    r = jnp.asarray(pos).reshape(4, 3)
    result = apply_fn(z, r, jnp.asarray([4]), None)
    assert result["energy"].shape == (1,)
    assert result["forces"].shape == (4, 3)


def test_neb_factory_selects_kernnn(tmp_path: Path):
    key = jax.random.key(5)
    config = KerNNConfig(n_hidden=4)
    params = init_params(key, config=config)
    path = tmp_path / "best.json"
    save_checkpoint(path, params=params, config=config, stats=_stats_for(_h2co_geometry()))
    make = _default_calculator_factory(path, calculator="kernnn")
    calc = make()
    assert isinstance(calc, KerNNCalculator)


def test_torch_state_dict_transpose():
    state = {
        "layers.0.weight": np.ones((20, 6), dtype=np.float32),
        "layers.0.bias": np.zeros(20, dtype=np.float32),
        "layers.2.weight": np.ones((20, 20), dtype=np.float32),
        "layers.2.bias": np.zeros(20, dtype=np.float32),
        "layers.4.weight": np.ones((20, 20), dtype=np.float32),
        "layers.4.bias": np.zeros(20, dtype=np.float32),
        "layers.6.weight": np.ones((1, 20), dtype=np.float32),
        "layers.6.bias": np.zeros(1, dtype=np.float32),
    }
    params = torch_state_dict_to_flax_params(state)
    assert params["dense_0"]["kernel"].shape == (6, 20)
    assert params["dense_3"]["kernel"].shape == (20, 1)


def test_form_acem_feature_counts():
    from mmml.models.kernnn.distances import n_features_for_scheme

    form = jnp.zeros((6, 3), dtype=jnp.float32)
    form = form.at[1, 0].set(1.0)
    assert get_bond_length_form(form).shape == (15,)
    assert n_features_for_scheme("form") == 15

    acem = jnp.zeros((9, 3), dtype=jnp.float32)
    acem = acem.at[1, 0].set(1.2)
    assert get_bond_length_acem(acem).shape == (36,)
    assert n_features_for_scheme("acem") == 36


def test_cli_parsers():
    tp = build_train_parser()
    ep = build_eval_parser()
    targs = tp.parse_args(
        [
            "--ntrain",
            "100",
            "--architecture",
            "ffnet",
            "--distance-scheme",
            "acem",
            "--distill-alpha",
            "0.5",
            "--teacher-checkpoint",
            "teacher.json",
        ]
    )
    assert targs.distance_scheme == "acem"
    assert targs.distill_alpha == 0.5
    eargs = ep.parse_args(["--split", "all"])
    assert eargs.split == "all"


def test_dimer_scan_factory_requires_checkpoint():
    from mmml.dimer_scan.calculators import calculator_factory
    from mmml.dimer_scan.config import DimerScanConfig

    cfg = DimerScanConfig(
        residues=("H2CO", "H2CO"),
        calculator="kernnn",
        distances_angstrom=(3.0, 4.0),
        checkpoint=None,
    )
    try:
        calculator_factory(cfg)
        raised = False
    except ValueError as exc:
        raised = True
        assert "kernnn" in str(exc).lower()
    assert raised
