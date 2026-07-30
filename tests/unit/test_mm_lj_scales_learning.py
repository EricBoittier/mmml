"""Validation that learnable CGenFF LJ scales actually train and deploy.

``test_mm_lj_scales.py`` covers the mechanics (attach/split/apply, JSON I/O, ATC
remap, nonzero gradients).  What it does not answer is whether the scales
*converge* under a real optimizer and whether the number that trained is the
number MD deploys.  These tests close that gap, and pin the documented
Ewald limitation so it cannot go stale silently.

See ``docs/hybrid-mm-lj-scales.md``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.mm_lj_scales import (
    attach_mm_lj_scales,
    find_learnable_lj_scales_sidecar,
    resolve_md_lj_scales,
    scales_to_atc,
    split_mm_lj_scale_params,
    write_mm_lj_scales_into_hybrid_mm_json,
)

# Two CGenFF-like types; values in the range of real CG331 / HGA3 entries.
MASTER_SIGMAS = jnp.array([3.6527, 2.3876])
MASTER_EPSILONS = jnp.array([0.0780, 0.0240])

SWITCH_KW = dict(
    mm_switch_on=3.0,
    mm_switch_width=2.0,
    ml_switch_width=1.0,
    complementary_handoff=False,
)


def _zero_ml_apply(params, *, atomic_numbers, positions, dst_idx, src_idx,
                   batch_segments, batch_size, batch_mask, atom_mask):
    """ML head contributing a constant, so the loss is driven purely by E_MM."""
    del params, atomic_numbers, dst_idx, src_idx, batch_segments, batch_mask
    e = jnp.sum(atom_mask) * jnp.asarray(-1.0)
    return {"energy": e.reshape(batch_size, 1), "forces": jnp.zeros_like(positions)}


def _dimer_batch(
    separation_A: float = 3.5,
    type_idx: tuple[int, int, int, int] = (0, 1, 0, 1),
) -> dict:
    """Two 2-atom monomers along x, separated so intermolecular LJ is active.

    ``type_idx`` assigns each atom a master-table row.  Using a single type
    throughout makes a scalar energy target identify that type's scale exactly.
    """
    n = 4
    pos = jnp.array(
        [[0.0, 0, 0], [1.0, 0, 0], [separation_A, 0, 0], [separation_A + 1.0, 0, 0]],
        dtype=jnp.float32,
    )
    idx = jnp.arange(n)
    dst, src = jnp.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    return {
        "R": pos,
        "Z": jnp.array([6, 1, 6, 1]),
        "mol_id": jnp.array([0, 0, 1, 1]).reshape(1, n),
        "cgenff_type_idx": jnp.array(type_idx).reshape(1, n),
        # Zero charges: E_MM is pure LJ, so the fit is unambiguous.
        "cgenff_charge": jnp.zeros(n).reshape(1, n),
        "atom_mask": jnp.ones(n, dtype=jnp.float32),
        "batch_mask": (dst != src).astype(jnp.float32),
        "dst_idx": dst,
        "src_idx": src,
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
    }


def _fit_loss(targets, batches, *, freeze: str | None = None):
    """Mean squared error of E_MM over ``batches`` against ``targets``.

    ``freeze`` stops the gradient on one leaf.  This matters: σ and ε are
    *mutually degenerate* against an energy-only target — a larger well depth and
    a slightly larger radius produce the same E_MM — so a test that plants one and
    leaves the other free recovers neither.  Real training breaks the degeneracy
    with forces and many geometries; here we simply hold one fixed.
    """

    def loss_fn(p):
        _, sig, eps = split_mm_lj_scale_params(p)
        if freeze == "sigma":
            sig = jax.lax.stop_gradient(sig)
        elif freeze == "epsilon":
            eps = jax.lax.stop_gradient(eps)
        return sum(
            (_e_mm(sig, eps, b) - t) ** 2 for b, t in zip(batches, targets)
        ) / len(batches)

    return loss_fn


def _fit_scales(loss_fn, params, *, lr: float, steps: int):
    """Adam loop over the LJ-scale leaves; returns (params, loss0, loss1)."""
    import optax

    opt = optax.adam(lr)
    state = opt.init(params)
    loss0 = float(loss_fn(params))

    @jax.jit
    def step(p, s):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, s = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    for _ in range(steps):
        params, state, _ = step(params, state)
    return params, loss0, float(loss_fn(params))


def _e_mm(sigma_scale, epsilon_scale, batch) -> jnp.ndarray:
    from mmml.models.hybrid_energy import hybrid_forward

    out = hybrid_forward(
        _zero_ml_apply,
        {"params": {}},
        batch,
        1,
        MASTER_SIGMAS,
        MASTER_EPSILONS,
        learn_mm_lj_scales=True,
        mm_lj_sigma_scale=sigma_scale,
        mm_lj_epsilon_scale=epsilon_scale,
        **SWITCH_KW,
    )
    return jnp.asarray(out["e_mm"]).reshape(())


def test_optimizer_recovers_planted_epsilon_scale():
    """Adam on the params-pytree leaves recovers a known ε scale.

    The real question behind issue #133: not "is there a gradient" but "does the
    optimizer move these leaves to the right place".  A single-type system with σ
    held fixed makes the scalar energy target identify ``s^ε[0]`` uniquely.
    """
    batch = _dimer_batch(type_idx=(0, 0, 0, 0))
    truth_eps = 1.6
    target = _e_mm(jnp.ones(2), jnp.array([truth_eps, 1.0]), batch)

    params, loss0, loss1 = _fit_scales(
        _fit_loss([target], [batch], freeze="sigma"),
        attach_mm_lj_scales({"params": {}}, 2),
        lr=3e-2, steps=400,
    )
    assert loss1 < loss0 * 1e-3, f"loss did not converge: {loss0:g} -> {loss1:g}"

    _, sig_out, eps_out = split_mm_lj_scale_params(params)
    assert float(np.asarray(eps_out)[0]) == pytest.approx(truth_eps, rel=2e-2)
    # Type 1 is absent from the system, so it gets no gradient and must stay at
    # its 1.0 init — the same property the ATC remap relies on for solvent types.
    assert float(np.asarray(eps_out)[1]) == pytest.approx(1.0, abs=1e-6)
    # Frozen leaf untouched.
    np.testing.assert_allclose(np.asarray(sig_out), np.ones(2), atol=1e-6)


def test_optimizer_recovers_planted_sigma_scale():
    """Same, for σ — which enters through r^-12/r^-6, not linearly."""
    batch = _dimer_batch(type_idx=(0, 0, 0, 0))
    truth_sig = 1.08
    target = _e_mm(jnp.array([truth_sig, 1.0]), jnp.ones(2), batch)

    params, loss0, loss1 = _fit_scales(
        _fit_loss([target], [batch], freeze="epsilon"),
        attach_mm_lj_scales({"params": {}}, 2),
        lr=1e-2, steps=600,
    )
    assert loss1 < loss0 * 1e-3, f"loss did not converge: {loss0:g} -> {loss1:g}"

    _, sig_out, eps_out = split_mm_lj_scale_params(params)
    assert float(np.asarray(sig_out)[0]) == pytest.approx(truth_sig, rel=2e-2)
    assert float(np.asarray(sig_out)[1]) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(np.asarray(eps_out), np.ones(2), atol=1e-6)


def test_two_types_identified_from_multiple_geometries():
    """Both per-type ε scales recover once the fit sees several separations.

    This is the shape of a real training set: one geometry leaves the per-type
    split degenerate, a distance scan does not, because each type pair carries a
    different r-dependence.
    """
    separations = (3.2, 3.6, 4.1, 4.8, 5.6)
    batches = [_dimer_batch(d) for d in separations]
    truth_eps = jnp.array([1.5, 0.6])
    targets = [_e_mm(jnp.ones(2), truth_eps, b) for b in batches]

    def loss_fn(p):
        _, sig, eps = split_mm_lj_scale_params(p)
        return sum(
            (_e_mm(sig, eps, b) - t) ** 2 for b, t in zip(batches, targets)
        ) / len(batches)

    params, loss0, loss1 = _fit_scales(
        loss_fn, attach_mm_lj_scales({"params": {}}, 2), lr=2e-2, steps=1500
    )
    assert loss1 < loss0 * 1e-4, f"loss did not converge: {loss0:g} -> {loss1:g}"

    _, _, eps_out = split_mm_lj_scale_params(params)
    np.testing.assert_allclose(
        np.asarray(eps_out), np.asarray(truth_eps), rtol=5e-2
    )


def test_scales_stay_finite_and_positive_under_training():
    """A σ scale driven to <= 0 would make r^-12 explode; guard the trajectory."""
    import optax

    batch = _dimer_batch()
    target = _e_mm(jnp.array([1.2, 0.85]), jnp.array([1.4, 0.7]), batch)
    params = attach_mm_lj_scales({"params": {}}, 2)

    def loss_fn(p):
        _, sig, eps = split_mm_lj_scale_params(p)
        return (_e_mm(sig, eps, batch) - target) ** 2

    opt = optax.adam(5e-2)
    state = opt.init(params)
    for _ in range(400):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        assert np.isfinite(float(loss)), "loss went non-finite during training"
        for key in ("mm_lj_sigma_scale", "mm_lj_epsilon_scale"):
            assert np.all(np.isfinite(np.asarray(grads[key]))), f"{key} grad non-finite"
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        assert np.all(np.asarray(params["mm_lj_sigma_scale"]) > 0.0)


def test_trained_scales_survive_round_trip_to_md_atc(tmp_path: Path):
    """The number that trained is the number MD deploys.

    Covers the seam that unit tests skipped: trained vector -> hybrid_mm.json ->
    resolve_md_lj_scales -> the exact ``at_ep``/``at_rm`` arithmetic from
    ``mm_energy_forces.build_mm_energy_forces_fn``.
    """
    type_names = ["CG331", "HGA3"]
    trained_sig = np.array([1.0731, 0.9422])
    trained_eps = np.array([1.6044, 0.5981])

    run = tmp_path / "run-xyz"
    run.mkdir()
    write_mm_lj_scales_into_hybrid_mm_json(
        run / "hybrid_mm.json",
        type_names=type_names,
        sigma_scale=trained_sig,
        epsilon_scale=trained_eps,
    )

    # CHARMM ATC order differs from the master table and carries extra types.
    atc = ["OG2D1", "HGA3", "CG331", "CLGA1"]
    ep_scale, sig_scale = resolve_md_lj_scales(
        checkpoint=run / "params.json", atc_names=atc
    )
    assert ep_scale is not None and sig_scale is not None

    # Reordered onto ATC, with 1.0 for types absent from training.
    np.testing.assert_allclose(sig_scale, [1.0, trained_sig[1], trained_sig[0], 1.0])
    np.testing.assert_allclose(ep_scale, [1.0, trained_eps[1], trained_eps[0], 1.0])

    # Mirror mm_energy_forces: at_ep = -|eps| * ep_scale, at_rm = rmin * sig_scale.
    atc_epsilons = np.array([-0.1200, -0.0240, -0.0780, -0.3430])
    atc_rmins = np.array([1.7000, 1.3400, 2.0600, 1.9100])
    at_ep = -1 * np.abs(atc_epsilons) * ep_scale
    at_rm = atc_rmins * sig_scale

    # Trained types scaled; untouched types bit-identical to stock CGenFF.
    assert at_ep[1] == pytest.approx(-0.0240 * trained_eps[1])
    assert at_ep[2] == pytest.approx(-0.0780 * trained_eps[0])
    assert at_rm[1] == pytest.approx(1.3400 * trained_sig[1])
    assert at_rm[2] == pytest.approx(2.0600 * trained_sig[0])
    assert at_ep[0] == pytest.approx(-0.1200)
    assert at_rm[3] == pytest.approx(1.9100)


def test_atc_types_missing_from_training_are_left_at_unit_scale():
    """A box with solvent types the model never saw must keep stock CGenFF LJ."""
    ep, sig = scales_to_atc(
        ["CG331"], [1.5], [2.0], ["CG331", "OT", "HT"]
    )
    np.testing.assert_allclose(sig, [1.5, 1.0, 1.0])
    np.testing.assert_allclose(ep, [2.0, 1.0, 1.0])


def test_lj_scales_are_inert_under_ewald_solver():
    """Pins the documented limitation: ewald forces LJ off, so scales do nothing.

    If someone implements LJ under Ewald, this test fails on purpose — update
    ``docs/hybrid-mm-lj-scales.md`` § "What is still unsupported" at the same time.
    """
    from mmml.models.hybrid_energy import hybrid_forward

    batch = _dimer_batch()
    kw = dict(
        SWITCH_KW,
        lr_solver="ewald",
        pme_box_length=25.0,
        learn_mm_lj_scales=True,
    )

    base = hybrid_forward(
        _zero_ml_apply, {"params": {}}, batch, 1, MASTER_SIGMAS, MASTER_EPSILONS,
        mm_lj_sigma_scale=jnp.ones(2), mm_lj_epsilon_scale=jnp.ones(2), **kw
    )
    scaled = hybrid_forward(
        _zero_ml_apply, {"params": {}}, batch, 1, MASTER_SIGMAS, MASTER_EPSILONS,
        mm_lj_sigma_scale=jnp.array([1.4, 0.7]),
        mm_lj_epsilon_scale=jnp.array([3.0, 0.2]),
        **kw
    )
    np.testing.assert_allclose(
        np.asarray(base["e_mm"]), np.asarray(scaled["e_mm"]), rtol=1e-6
    )


def test_find_learnable_sidecar_discovery(tmp_path: Path):
    """Backs the doMM-off guard in hybrid_mlpot: detection needs no CHARMM."""
    run = tmp_path / "run"
    run.mkdir()
    assert find_learnable_lj_scales_sidecar(checkpoint=run / "params.json") is None

    path = run / "hybrid_mm.json"
    write_mm_lj_scales_into_hybrid_mm_json(
        path, type_names=["CG331"], sigma_scale=[1.1], epsilon_scale=[1.2]
    )
    assert find_learnable_lj_scales_sidecar(checkpoint=run / "params.json") == path
    assert find_learnable_lj_scales_sidecar(scales_file=path) == path


def test_find_learnable_sidecar_ignores_non_learnable(tmp_path: Path):
    import json

    path = tmp_path / "hybrid_mm.json"
    path.write_text(json.dumps({"learn_mm_lj_scales": False}), encoding="utf-8")
    assert find_learnable_lj_scales_sidecar(scales_file=path) is None
