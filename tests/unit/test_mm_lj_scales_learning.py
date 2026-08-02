"""Validation that learnable CGenFF LJ scales actually train and deploy.

``test_mm_lj_scales.py`` covers the mechanics (attach/split/apply, JSON I/O, ATC
remap, nonzero gradients).  What it does not answer is whether the scales
*converge* under a real optimizer and whether the number that trained is the
number MD deploys.  These tests close that gap, and cover Ewald + switched LJ
(#139 Phase 1): scales move ``e_mm`` when ``include_lj=True``, stay inert when
``False``.

See ``docs/hybrid-mm-lj-scales.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.mm_lj_scales import (
    MM_LJ_EPSILON_SCALE_BOUNDS,
    MM_LJ_EPSILON_SCALE_KEY,
    MM_LJ_SIGMA_SCALE_BOUNDS,
    MM_LJ_SIGMA_SCALE_KEY,
    attach_mm_lj_scales,
    clip_mm_lj_scale_params,
    find_learnable_lj_scales_sidecar,
    out_of_bounds_mm_lj_scales,
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


def _e_mm(
    sigma_scale,
    epsilon_scale,
    batch,
    *,
    master_epsilons=MASTER_EPSILONS,
    lr_solver: str = "mic",
    pme_box_length: float | None = None,
    include_lj: bool = True,
) -> jnp.ndarray:
    from mmml.models.hybrid_energy import hybrid_forward

    kw = dict(SWITCH_KW)
    if lr_solver != "mic":
        kw.update(
            lr_solver=lr_solver,
            pme_box_length=float(pme_box_length if pme_box_length is not None else 25.0),
            include_lj=include_lj,
            short_range_wall=False,
        )
    out = hybrid_forward(
        _zero_ml_apply,
        {"params": {}},
        batch,
        1,
        MASTER_SIGMAS,
        master_epsilons,
        learn_mm_lj_scales=True,
        mm_lj_sigma_scale=sigma_scale,
        mm_lj_epsilon_scale=epsilon_scale,
        **kw,
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


def test_optimizer_recovers_planted_epsilon_scale_under_ewald():
    """#139: learn_mm_lj_scales under lr_solver=ewald (q=0 → pure switched LJ)."""
    batch = _dimer_batch(type_idx=(0, 0, 0, 0))
    truth_eps = 1.6
    ewald_kw = dict(lr_solver="ewald", pme_box_length=25.0, include_lj=True)
    target = _e_mm(
        jnp.ones(2), jnp.array([truth_eps, 1.0]), batch, **ewald_kw
    )

    def loss_fn(p):
        _, sig, eps = split_mm_lj_scale_params(p)
        sig = jax.lax.stop_gradient(sig)
        return (_e_mm(sig, eps, batch, **ewald_kw) - target) ** 2

    params, loss0, loss1 = _fit_scales(
        loss_fn, attach_mm_lj_scales({"params": {}}, 2), lr=3e-2, steps=400
    )
    assert loss1 < loss0 * 1e-3, f"loss did not converge: {loss0:g} -> {loss1:g}"
    _, sig_out, eps_out = split_mm_lj_scale_params(params)
    assert float(np.asarray(eps_out)[0]) == pytest.approx(truth_eps, rel=2e-2)
    assert float(np.asarray(eps_out)[1]) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(np.asarray(sig_out), np.ones(2), atol=1e-6)


def test_optimizer_recovers_planted_sigma_scale():
    """Same, for σ — which enters through r^-12/r^-6, not linearly."""
    batch = _dimer_batch(type_idx=(0, 0, 0, 0))
    truth_sig = 1.04
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

    params, loss0, loss1 = _fit_scales(
        _fit_loss(targets, batches, freeze="sigma"),
        attach_mm_lj_scales({"params": {}}, 2),
        lr=2e-2, steps=1200,
    )
    assert loss1 < loss0 * 1e-3, f"loss did not converge: {loss0:g} -> {loss1:g}"

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


def test_scale_bounds_are_the_documented_physical_ranges():
    """σ is pinned to ±5%; ε may move by a factor of four either way."""
    assert MM_LJ_SIGMA_SCALE_BOUNDS == (0.95, 1.05)
    assert MM_LJ_EPSILON_SCALE_BOUNDS == (0.25, 4.0)


def test_clip_projects_scales_and_leaves_the_model_weights_alone():
    params = {
        "params": {"w": jnp.array([3.0, -7.0])},
        MM_LJ_SIGMA_SCALE_KEY: jnp.array([0.5, 1.0, 2.0]),
        MM_LJ_EPSILON_SCALE_KEY: jnp.array([-3.0, 1.0, 12.0]),
    }
    out = clip_mm_lj_scale_params(params)

    np.testing.assert_allclose(np.asarray(out[MM_LJ_SIGMA_SCALE_KEY]), [0.95, 1.0, 1.05])
    np.testing.assert_allclose(np.asarray(out[MM_LJ_EPSILON_SCALE_KEY]), [0.25, 1.0, 4.0])
    # Network weights are free parameters and must pass through untouched.
    np.testing.assert_allclose(np.asarray(out["params"]["w"]), [3.0, -7.0])


def test_clip_is_traceable_and_a_noop_without_scale_leaves():
    """It is called unconditionally inside the jitted step."""
    plain = {"params": {"w": jnp.ones(2)}}
    assert clip_mm_lj_scale_params(plain) is plain

    out = jax.jit(clip_mm_lj_scale_params)(attach_mm_lj_scales({"params": {}}, 2))
    np.testing.assert_allclose(np.asarray(out[MM_LJ_SIGMA_SCALE_KEY]), np.ones(2))


def _drift_loop(*, clip: bool, steps: int = 300, lr: float = 5e-2):
    """Adam chasing an E_MM the bounds cannot deliver.

    This is the long-run failure mode in miniature. ``E_MM`` is linear in the
    geometric-mean ε, so a target far outside reach gives a gradient that keeps
    pointing the same way for the whole run and the scales simply travel. Adam
    makes that travel roughly ``lr`` per step no matter how small the gradient
    gets, which is why a real fit crosses into unphysical σ/ε after enough
    epochs rather than settling.
    """
    import optax

    batch = _dimer_batch()
    target = 20.0 * _e_mm(jnp.ones(2), jnp.ones(2), batch)

    def loss_fn(p):
        _, sig, eps = split_mm_lj_scale_params(p)
        return (_e_mm(sig, eps, batch) - target) ** 2

    params = attach_mm_lj_scales({"params": {}}, 2)
    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(p, s):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, s = opt.update(grads, s, p)
        p = optax.apply_updates(p, updates)
        if clip:
            p = clip_mm_lj_scale_params(p)
        return p, s, loss

    losses = []
    for _ in range(steps):
        params, state, loss = step(params, state)
        losses.append(float(loss))
    return params, losses


def test_unclipped_scales_leave_the_physical_range():
    """The regression this guards: without projection the scales run away.

    ε reaching zero is the fatal one -- the CHARMM combining rule takes
    ``sqrt(eps_i * eps_j)``, so one type crossing zero NaNs every pair that mixes
    it with a positive type.
    """
    params, _ = _drift_loop(clip=False)
    sig = np.asarray(params[MM_LJ_SIGMA_SCALE_KEY])
    eps = np.asarray(params[MM_LJ_EPSILON_SCALE_KEY])

    escaped = (
        np.any(sig < MM_LJ_SIGMA_SCALE_BOUNDS[0])
        or np.any(sig > MM_LJ_SIGMA_SCALE_BOUNDS[1])
        or np.any(eps < MM_LJ_EPSILON_SCALE_BOUNDS[0])
        or np.any(eps > MM_LJ_EPSILON_SCALE_BOUNDS[1])
    )
    assert escaped, f"expected runaway scales, got sigma={sig}, epsilon={eps}"


def test_clipped_training_stays_bounded_and_finite():
    """Same optimizer, same unreachable target, with the projection in place."""
    params, losses = _drift_loop(clip=True)
    sig = np.asarray(params[MM_LJ_SIGMA_SCALE_KEY])
    eps = np.asarray(params[MM_LJ_EPSILON_SCALE_KEY])

    assert np.all(np.isfinite(losses)), "loss went non-finite under bounded scales"
    assert np.all(sig >= MM_LJ_SIGMA_SCALE_BOUNDS[0])
    assert np.all(sig <= MM_LJ_SIGMA_SCALE_BOUNDS[1])
    assert np.all(eps >= MM_LJ_EPSILON_SCALE_BOUNDS[0])
    assert np.all(eps <= MM_LJ_EPSILON_SCALE_BOUNDS[1])
    # The bound is doing work, not merely agreeing with where the fit stopped.
    assert np.any(np.isclose(eps, MM_LJ_EPSILON_SCALE_BOUNDS[1]))


def test_zero_epsilon_type_keeps_energy_and_gradients_finite():
    """CGenFF lone pairs carry ε = 0 by design; padding borrows master row 0.

    ``d sqrt(x)/dx`` is infinite at the origin, so the geometric mean needs the
    same masked-gradient treatment as the pair distance or a single zero-ε type
    turns every force into NaN.
    """
    tables = jnp.array([MASTER_EPSILONS[0], 0.0])
    batch = _dimer_batch(type_idx=(0, 1, 0, 1))

    def e(scales):
        return _e_mm(scales[0], scales[1], batch, master_epsilons=tables)

    scales = (jnp.ones(2), jnp.ones(2))
    assert np.isfinite(float(e(scales)))
    grads = jax.grad(e)(scales)
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite scale gradient: {g}"


def test_out_of_bounds_report_names_the_offending_type():
    problems = out_of_bounds_mm_lj_scales(
        ["CG331", "HGA3"], [1.0, 1.4], [2.0, 0.01]
    )
    assert any("HGA3" in p and "sigma" in p for p in problems)
    assert any("HGA3" in p and "epsilon" in p for p in problems)
    assert not out_of_bounds_mm_lj_scales(["CG331"], [1.02], [3.0])


def test_sidecar_outside_bounds_still_loads_but_warns(tmp_path: Path):
    """Runs predating the bounds must not be stranded, only flagged."""
    from mmml.models.mm_lj_scales import load_mm_lj_scales_sidecar

    path = tmp_path / "hybrid_mm.json"
    write_mm_lj_scales_into_hybrid_mm_json(
        path, type_names=["CG331"], sigma_scale=[1.4], epsilon_scale=[12.0]
    )
    with pytest.warns(RuntimeWarning, match="outside the trainable bounds"):
        payload = load_mm_lj_scales_sidecar(path)
    np.testing.assert_allclose(payload["mm_lj_sigma_scale"], [1.4])


def test_train_step_projects_lj_scales_into_bounds():
    """The projection has to live in the real update path, not just a test loop."""
    import optax

    from mmml.models.hybrid_energy import HybridMMConfig
    from mmml.models.physnetjax.physnetjax.training.trainstep import train_step

    class _Transform(NamedTuple):
        scale: jnp.ndarray

    batch = dict(_dimer_batch())
    # E_ML is a constant -4 eV here, so this asks E_MM for +1 eV -- orders of
    # magnitude past what bounded LJ can produce, pinning the scales at a bound.
    batch["E"] = jnp.array([[-3.0]])
    batch["F"] = jnp.zeros((4, 3))

    cfg = HybridMMConfig.coerce(
        dict(
            SWITCH_KW,
            master_sigmas=tuple(float(x) for x in MASTER_SIGMAS),
            master_epsilons=tuple(float(x) for x in MASTER_EPSILONS),
            learn_mm_lj_scales=True,
        )
    )

    params = attach_mm_lj_scales({"params": {}}, 2)
    ema_params = params
    opt = optax.adam(0.5)
    opt_state = opt.init(params)
    transform_state = _Transform(scale=jnp.asarray(1.0))

    for _ in range(20):
        params, ema_params, opt_state, transform_state, loss, *_ = train_step(
            model_apply=_zero_ml_apply,
            optimizer_update=opt.update,
            transform_state=transform_state,
            batch=batch,
            batch_size=1,
            doCharges=False,
            energy_weight=1.0,
            forces_weight=1.0,
            dipole_weight=0.0,
            charges_weight=0.0,
            opt_state=opt_state,
            params=params,
            ema_params=ema_params,
            hybrid_mm=cfg,
        )
        assert np.isfinite(float(loss))

    sig = np.asarray(params[MM_LJ_SIGMA_SCALE_KEY])
    eps = np.asarray(params[MM_LJ_EPSILON_SCALE_KEY])
    assert np.all(sig >= MM_LJ_SIGMA_SCALE_BOUNDS[0] - 1e-6)
    assert np.all(sig <= MM_LJ_SIGMA_SCALE_BOUNDS[1] + 1e-6)
    assert np.all(eps >= MM_LJ_EPSILON_SCALE_BOUNDS[0] - 1e-6)
    assert np.all(eps <= MM_LJ_EPSILON_SCALE_BOUNDS[1] + 1e-6)
    assert np.any(np.isclose(eps, MM_LJ_EPSILON_SCALE_BOUNDS[1], atol=1e-6)), (
        "learning rate was too small to prove the projection engaged"
    )
    # The EMA is what gets written to hybrid_mm.json, so it must be bounded too.
    ema_eps = np.asarray(ema_params[MM_LJ_EPSILON_SCALE_KEY])
    assert np.all(ema_eps <= MM_LJ_EPSILON_SCALE_BOUNDS[1] + 1e-6)


def test_trained_scales_survive_round_trip_to_md_atc(tmp_path: Path):
    """The number that trained is the number MD deploys.

    Covers the seam that unit tests skipped: trained vector -> hybrid_mm.json ->
    resolve_md_lj_scales -> the exact ``at_ep``/``at_rm`` arithmetic from
    ``mm_energy_forces.build_mm_energy_forces_fn``.
    """
    type_names = ["CG331", "HGA3"]
    trained_sig = np.array([1.0312, 0.9622])
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


def test_lj_scales_move_emm_under_ewald_when_include_lj():
    """#139 Phase 1: fixed scales affect Ewald E_MM only when include_lj=True."""
    from mmml.data.units import KCAL_MOL_TO_EV
    from mmml.models.cgenff_mm import cgenff_lj_energy, monomer_centroids
    from mmml.interfaces.pycharmmInterface.calculator_utils import mm_switch_scale
    from mmml.models.hybrid_energy import hybrid_forward
    from mmml.models.mm_lj_scales import apply_mm_lj_scales

    batch = _dimer_batch(separation_A=3.5)
    base_kw = dict(
        SWITCH_KW,
        lr_solver="ewald",
        pme_box_length=25.0,
        short_range_wall=False,
        learn_mm_lj_scales=False,
    )
    sig_s = jnp.array([1.4, 0.7])
    eps_s = jnp.array([3.0, 0.2])

    # Coulomb-only: scales must remain inert.
    base_off = hybrid_forward(
        _zero_ml_apply, {"params": {}}, batch, 1, MASTER_SIGMAS, MASTER_EPSILONS,
        include_lj=False,
        mm_lj_sigma_scale=jnp.ones(2), mm_lj_epsilon_scale=jnp.ones(2), **base_kw
    )
    scaled_off = hybrid_forward(
        _zero_ml_apply, {"params": {}}, batch, 1, MASTER_SIGMAS, MASTER_EPSILONS,
        include_lj=False,
        mm_lj_sigma_scale=sig_s, mm_lj_epsilon_scale=eps_s, **base_kw
    )
    np.testing.assert_allclose(
        np.asarray(base_off["e_mm"]), np.asarray(scaled_off["e_mm"]), rtol=1e-6
    )

    # LJ on: scales must move e_mm.
    base_on = hybrid_forward(
        _zero_ml_apply, {"params": {}}, batch, 1, MASTER_SIGMAS, MASTER_EPSILONS,
        include_lj=True,
        mm_lj_sigma_scale=jnp.ones(2), mm_lj_epsilon_scale=jnp.ones(2), **base_kw
    )
    scaled_on = hybrid_forward(
        _zero_ml_apply, {"params": {}}, batch, 1, MASTER_SIGMAS, MASTER_EPSILONS,
        include_lj=True,
        mm_lj_sigma_scale=sig_s, mm_lj_epsilon_scale=eps_s, **base_kw
    )
    assert not np.allclose(
        np.asarray(base_on["e_mm"]), np.asarray(scaled_on["e_mm"]), rtol=1e-5
    )

    # LJ-on − LJ-off == KCAL_MOL_TO_EV * λ_MM * E_LJ (unit scales).
    pos = batch["R"]
    mid = batch["mol_id"].reshape(-1)
    tidx = batch["cgenff_type_idx"].reshape(-1)
    sig_eff, eps_eff = apply_mm_lj_scales(
        MASTER_SIGMAS, MASTER_EPSILONS, jnp.ones(2), jnp.ones(2), include_lj=True
    )
    e_lj = float(cgenff_lj_energy(pos, tidx, mid, sig_eff, eps_eff))
    coms = monomer_centroids(pos, mid, n_monomers=2)
    r_com = float(np.linalg.norm(np.asarray(coms[1] - coms[0])))
    lam = float(
        mm_switch_scale(
            jnp.asarray(r_com),
            mm_switch_on=SWITCH_KW["mm_switch_on"],
            mm_switch_width=SWITCH_KW["mm_switch_width"],
            ml_switch_width=SWITCH_KW["ml_switch_width"],
            complementary_handoff=SWITCH_KW["complementary_handoff"],
        )
    )
    expected_delta = KCAL_MOL_TO_EV * lam * e_lj
    delta = float(
        np.asarray(base_on["e_mm"]).reshape(-1)[0]
        - np.asarray(base_off["e_mm"]).reshape(-1)[0]
    )
    np.testing.assert_allclose(delta, expected_delta, rtol=1e-5, atol=1e-8)


def test_ewald_plus_lj_force_energy_fd_smoke():
    """Finite-difference check for Ewald Coulomb + switched LJ forces."""
    from mmml.models.hybrid_energy import hybrid_forward

    jax.config.update("jax_enable_x64", True)
    # float64 for a stable central-difference check (training path is float32).
    batch = _dimer_batch(separation_A=3.5)
    batch = {
        k: (jnp.asarray(v, dtype=jnp.float64) if hasattr(v, "dtype") and
            jnp.issubdtype(v.dtype, jnp.floating) else v)
        for k, v in batch.items()
    }
    kw = dict(
        SWITCH_KW,
        lr_solver="ewald",
        pme_box_length=25.0,
        short_range_wall=False,
        include_lj=True,
    )
    master_sig = jnp.asarray(MASTER_SIGMAS, dtype=jnp.float64)
    master_eps = jnp.asarray(MASTER_EPSILONS, dtype=jnp.float64)

    def _fwd(b):
        return hybrid_forward(
            _zero_ml_apply, {"params": {}}, b, 1, master_sig, master_eps, **kw
        )

    out = _fwd(batch)
    f0 = np.asarray(out["forces"]).reshape(-1, 3)
    assert np.all(np.isfinite(f0))

    d = np.zeros_like(f0)
    d[0, 0] = 1.0
    d[2, 0] = -1.0
    d = d / np.linalg.norm(d)
    eps = 1e-4
    pos = np.asarray(batch["R"], dtype=np.float64)

    def energy_at(p):
        b = dict(batch)
        b["R"] = jnp.asarray(p, dtype=jnp.float64)
        return float(np.asarray(_fwd(b)["energy"]).reshape(-1)[0])

    fd = (energy_at(pos + eps * d) - energy_at(pos - eps * d)) / (2.0 * eps)
    analytic = float(-np.sum(f0 * d))
    rel = abs(fd - analytic) / max(abs(analytic), 1e-8)
    assert rel < 5e-3, f"fd={fd}, -F.d={analytic}, rel={rel}"


def test_find_learnable_sidecar_discovery(tmp_path: Path):
    """Backs the doMM-off guard in hybrid_mlpot: detection needs no CHARMM."""
    run = tmp_path / "run"
    run.mkdir()
    assert find_learnable_lj_scales_sidecar(checkpoint=run / "params.json") is None

    path = run / "hybrid_mm.json"
    write_mm_lj_scales_into_hybrid_mm_json(
        path, type_names=["CG331"], sigma_scale=[1.03], epsilon_scale=[1.2]
    )
    assert find_learnable_lj_scales_sidecar(checkpoint=run / "params.json") == path
    assert find_learnable_lj_scales_sidecar(scales_file=path) == path


def test_find_learnable_sidecar_ignores_non_learnable(tmp_path: Path):
    import json

    path = tmp_path / "hybrid_mm.json"
    path.write_text(json.dumps({"learn_mm_lj_scales": False}), encoding="utf-8")
    assert find_learnable_lj_scales_sidecar(scales_file=path) is None


def test_md_example_ships_a_condensed_phase_run():
    """Ship both scalable JAX and full-box CHARMM deployment demonstrations."""
    import yaml

    root = Path(__file__).resolve().parents[2]
    md = yaml.safe_load(
        (root / "examples/hybrid_mm_charges/md_fixed_lj_scales.yaml").read_text()
    )
    runs = md["runs"]
    liquid = [
        r for r in runs.values() if str(r.get("setup", "")).startswith("pbc_")
    ]
    assert liquid, "no condensed-phase (pbc_*) run in md_fixed_lj_scales.yaml"
    modes = {run.get("mm_nonbond_mode") for run in liquid}
    assert "jax_mic" in modes, "missing scalable JAX LJ deployment demo"
    assert "periodic_external" in modes, "missing full-box CHARMM LJ deployment demo"
    for run in liquid:
        assert run.get("box_size"), "pbc_* run needs a box_size"
        if run.get("mm_nonbond_mode") == "periodic_external":
            assert run.get("backend") == "pycharmm"
            assert run.get("lr_solver") in {"ewald", "jax_pme", "nvalchemiops_pme"}
    assert md["defaults"]["include_mm"] is True


def test_negative_epsilon_scale_does_not_produce_nan():
    """Regression: an unconstrained scale crossing zero used to NaN the loss.

    ``eps_ij = sqrt(eps_i * eps_j)`` is a geometric mean, so its sign only
    cancels while every per-type epsilon shares a sign. A scale that crossed zero
    flipped one type's sign and every mixed pair became ``sqrt(negative)``. That
    NaN reached the forces and the loss and never recovered — a real DCM training
    run died this way at epoch ~89.

    ``apply_mm_lj_scales`` now floors scales at ``MM_LJ_MIN_SCALE``.
    """
    batch = _dimer_batch()  # two types -> mixed pairs exist

    baseline = float(_e_mm(jnp.ones(2), jnp.ones(2), batch))
    assert np.isfinite(baseline) and abs(baseline) > 0

    for bad in (-1e-3, -0.05, -0.5, -50.0):
        got = float(_e_mm(jnp.ones(2), jnp.array([bad, 1.0]), batch))
        assert np.isfinite(got), f"epsilon scale {bad} produced {got}"
        got = float(_e_mm(jnp.array([bad, 1.0]), jnp.ones(2), batch))
        assert np.isfinite(got), f"sigma scale {bad} produced {got}"


def test_scale_floor_leaves_normal_values_untouched():
    """The floor must not perturb any scale in a plausible range."""
    from mmml.models.mm_lj_scales import MM_LJ_MIN_SCALE, apply_mm_lj_scales

    assert MM_LJ_MIN_SCALE > 0
    sig_in = jnp.array([0.5, 2.0])
    eps_in = jnp.array([0.5, 2.0])
    s, e = apply_mm_lj_scales(MASTER_SIGMAS, MASTER_EPSILONS, sig_in, eps_in)
    np.testing.assert_allclose(np.asarray(s), np.asarray(MASTER_SIGMAS * sig_in))
    np.testing.assert_allclose(np.asarray(e), np.asarray(MASTER_EPSILONS * eps_in))


def test_gradient_survives_across_the_floor():
    """Above the floor the gradient is unchanged; the floor only clamps below."""
    batch = _dimer_batch()
    g = jax.grad(lambda e: _e_mm(jnp.ones(2), e, batch))(jnp.array([0.5, 1.0]))
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.abs(np.asarray(g)) > 0)
