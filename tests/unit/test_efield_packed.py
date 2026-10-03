"""Packed variable-size batches must reproduce the fixed-size padded path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mmml.data.spice_alpha_ragged import offsets_from_counts, split_ragged
from mmml.models.efield.model_functions import predicted_polarizability_bohr3
from mmml.models.efield.packed import PackSpec, iter_packed_batches, make_steps, SubsetMetrics
from mmml.models.efield.training import EFieldPhysNet, energy_and_forces, load_ef_npz, prepare_batches

from test_spice_alpha_efield_train import _init_params, _spice_efield_splits, _tiny_model


@pytest.fixture(scope="module", autouse=True)
def _float32():
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


def _packed_model() -> EFieldPhysNet:
    return _tiny_model().clone(packed=True)


def _ragged_from_padded(d: dict) -> dict:
    Z = np.asarray(d["atomic_numbers"])
    keep = Z > 0
    N = keep.sum(axis=1).astype(np.int16)
    n = len(N)
    return {
        "Z": Z[keep].astype(np.int8),
        "R": np.asarray(d["positions"])[keep],
        "F": np.asarray(d["forces"])[keep],
        "N": N,
        "E": np.asarray(d["energies"], np.float64),
        "D": np.asarray(d["D"]),
        "Q": np.zeros((n,), np.float32),
        "polar": np.asarray(d["polar"]),
        "mol": np.arange(n),
        "subset": np.zeros((n,), np.int8),
        "offsets": offsets_from_counts(N),
    }


def test_packed_matches_padded(tmp_path):
    written = _spice_efield_splits(tmp_path, n_confs=8)
    train = load_ef_npz(written["train"])
    padded_model, packed_model = _tiny_model(), _packed_model()
    params = _init_params(padded_model, train, jax.random.PRNGKey(5))
    B = 2
    pb = prepare_batches(jax.random.PRNGKey(0), train, B, shuffle=False, rot_augment=False)[0]
    n_atoms = pb["positions"].shape[0] // B

    e_pad, f_pad, d_pad = energy_and_forces(
        padded_model.apply, params, pb["atomic_numbers"], pb["positions"], pb["electric_field"],
        pb["dst_idx_flat"], pb["src_idx_flat"], pb["batch_segments"], B)
    a_pad = predicted_polarizability_bohr3(
        padded_model.apply, params, pb["atomic_numbers"].reshape(B, n_atoms),
        pb["positions"].reshape(B, n_atoms, 3), pb["dst_idx_flat"], pb["src_idx_flat"],
        pb["batch_segments"], B)

    ragged = _ragged_from_padded(train)
    spec = PackSpec(max_molecules=B + 1, max_atoms=3 * n_atoms + 1, max_edges=4096, cutoff=5.0)
    batch = next(iter_packed_batches(ragged, np.arange(B), spec))
    M = spec.max_molecules
    bj = {k: jnp.asarray(v) for k, v in batch.items()}
    e_pk, f_pk, d_pk = energy_and_forces(
        packed_model.apply, params, bj["atomic_numbers"], bj["positions"], bj["electric_field"],
        bj["dst_idx_flat"], bj["src_idx_flat"], bj["batch_segments"], M)
    a_pk = predicted_polarizability_bohr3(
        packed_model.apply, params, bj["atomic_numbers"], bj["positions"], bj["dst_idx_flat"],
        bj["src_idx_flat"], bj["batch_segments"], M)

    np.testing.assert_allclose(np.asarray(e_pk)[:B], np.asarray(e_pad), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(np.asarray(d_pk)[:B], np.asarray(d_pad), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(a_pk)[:B], np.asarray(a_pad), rtol=1e-4, atol=1e-4)
    real_pad = np.asarray(pb["atomic_numbers"]) > 0
    real_pk = np.asarray(bj["atomic_numbers"]) > 0
    np.testing.assert_allclose(np.asarray(f_pk)[real_pk], np.asarray(f_pad)[real_pad], rtol=1e-4, atol=1e-5)
    # The padding molecule slot gets nothing.
    assert float(np.abs(np.asarray(d_pk)[M - 1]).max()) == 0.0


def test_packed_train_step_moves_params_and_masks_padding(tmp_path):
    written = _spice_efield_splits(tmp_path, n_confs=8)
    train = load_ef_npz(written["train"])
    model = _packed_model()
    params = _init_params(_tiny_model(), train, jax.random.PRNGKey(6))
    ragged = _ragged_from_padded(train)
    spec = PackSpec(max_molecules=4, max_atoms=16, max_edges=512, cutoff=5.0)
    batches = list(iter_packed_batches(ragged, np.arange(len(ragged["N"])), spec))
    assert len(batches) >= 2  # budget forces several batches
    weights = {"energy": 0.0, "forces": 1.0, "dipole": 1.0, "charge": 1.0, "polar": 1.0}
    opt = optax.adam(1e-2)
    train_step, eval_step = make_steps(model.apply, opt, spec.max_molecules, weights, 0.001, False, 0.5)
    b = {k: jnp.asarray(v) for k, v in batches[0].items()}
    new_params, ema, _, loss, terms, finite = train_step(params, params, opt.init(params), b)
    assert bool(finite) and np.isfinite(float(loss))
    delta = max(float(jnp.max(jnp.abs(x - y))) for x, y in
                zip(jax.tree_util.tree_leaves(new_params), jax.tree_util.tree_leaves(params)))
    assert delta > 0.0
    loss_e, terms_e, preds = eval_step(new_params, b)
    metrics = SubsetMetrics({0: "water"})
    metrics.update(batches[0], jax.device_get(preds))
    res = metrics.result()
    assert res["water"]["n_frames"] == int(batches[0]["mol_mask"].sum())
    assert all(np.isfinite(v) for v in res["all"].values())


def test_split_ragged_holds_out_whole_molecules():
    n_mol, per = 200, 5
    mol = np.repeat(np.arange(n_mol), per)
    data = {"subset": np.zeros(len(mol), np.int8), "mol": mol}
    idx = split_ragged(data, valid_frac=0.1, test_frac=0.1, seed=1)
    sets = {k: set(mol[v]) for k, v in idx.items()}
    assert not (sets["train"] & sets["valid"]) and not (sets["train"] & sets["test"])
    assert sum(len(v) for v in idx.values()) == len(mol)


def test_packed_long_range_coulomb_matches_padded_all_pairs(tmp_path):
    """With a message cutoff shorter than the molecule, the padded path still sums
    Coulomb over all pairs; packed batches reproduce that only with a separate
    all-pairs Coulomb list, and differ when Coulomb reuses the short edges."""
    import math

    written = _spice_efield_splits(tmp_path, n_confs=8)
    train = load_ef_npz(written["train"])
    short = dict(cutoff=1.2)  # excludes the ~1.5 Å H–H pair in water from messages
    padded_model = _tiny_model().clone(**short)
    packed_model = _tiny_model().clone(packed=True, **short)
    params = _init_params(padded_model, train, jax.random.PRNGKey(7))
    # make charges non-trivial so Coulomb matters
    params = jax.tree_util.tree_map(lambda x: x + 0.05, params)
    B = 2
    pb = prepare_batches(jax.random.PRNGKey(0), train, B, shuffle=False, rot_augment=False)[0]
    e_pad, f_pad, _ = energy_and_forces(
        padded_model.apply, params, pb["atomic_numbers"], pb["positions"], pb["electric_field"],
        pb["dst_idx_flat"], pb["src_idx_flat"], pb["batch_segments"], B)

    ragged = _ragged_from_padded(train)
    n_atoms = pb["positions"].shape[0] // B

    def packed_energy(spec):
        batch = {k: jnp.asarray(v) for k, v in next(iter_packed_batches(ragged, np.arange(B), spec)).items()}

        def efn(pos):
            e, _ = packed_model.apply(
                params, atomic_numbers=batch["atomic_numbers"], positions=pos, Ef=batch["electric_field"],
                dst_idx_flat=batch["dst_idx_flat"], src_idx_flat=batch["src_idx_flat"],
                batch_segments=batch["batch_segments"], batch_size=spec.max_molecules,
                coulomb_dst_idx_flat=batch.get("coulomb_dst_idx_flat"),
                coulomb_src_idx_flat=batch.get("coulomb_src_idx_flat"))
            return -jnp.sum(e[:B]), e

        (_, e), f = jax.value_and_grad(efn, has_aux=True)(batch["positions"])
        real = np.asarray(batch["atomic_numbers"]) > 0
        return np.asarray(e)[:B], np.asarray(f)[real]

    base = dict(max_molecules=B + 1, max_atoms=3 * n_atoms + 1, max_edges=4096, cutoff=1.2)
    e_all, f_all = packed_energy(PackSpec(**base, coulomb_cutoff=math.inf, max_coulomb_edges=4096))
    e_short, _ = packed_energy(PackSpec(**base))
    real_pad = np.asarray(pb["atomic_numbers"]) > 0
    np.testing.assert_allclose(e_all, np.asarray(e_pad), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(f_all, np.asarray(f_pad)[real_pad], rtol=1e-4, atol=1e-5)
    assert np.abs(e_short - np.asarray(e_pad)).max() > 1e-4


def test_linear_charge_head_allows_charges_below_silu_floor(tmp_path):
    """silu bounds atomic charges at -0.278 e; the linear head does not."""
    written = _spice_efield_splits(tmp_path, n_confs=4)
    train = load_ef_npz(written["train"])
    for act, floor_ok in (("silu", True), ("linear", False)):
        model = _tiny_model().clone(charge_activation=act)
        params = _init_params(model, train, jax.random.PRNGKey(8))
        # push the charge-head bias strongly negative
        params = jax.tree_util.tree_map_with_path(
            lambda path, x: x - 5.0 if "Dense_0" in jax.tree_util.keystr(path) and "bias" in jax.tree_util.keystr(path) else x,
            params)
        pb = prepare_batches(jax.random.PRNGKey(0), train, 2, shuffle=False, rot_augment=False)[0]
        _, state = model.apply(params, atomic_numbers=pb["atomic_numbers"], positions=pb["positions"],
                               Ef=pb["electric_field"], dst_idx_flat=pb["dst_idx_flat"],
                               src_idx_flat=pb["src_idx_flat"], batch_segments=pb["batch_segments"],
                               batch_size=2, mutable=["intermediates"])
        q = np.asarray(state["intermediates"]["atomic_charges"][-1])
        assert (q.min() >= -0.2785) == floor_ok
