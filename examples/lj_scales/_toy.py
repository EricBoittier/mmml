"""Shared toy system for steps 03 and 04.

A two-monomer, four-atom system with zero charges, so ``E_MM`` is pure Lennard-
Jones and every gradient is attributable to the LJ term alone. The ML head is a
constant, standing in for PhysNet.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Stand-in for the CGenFF master tables. Real ones carry ~1200 types; these two
# rows are in the range of real CG331 / HGA3 entries.
MASTER_SIGMAS = jnp.array([3.6527, 2.3876])
MASTER_EPSILONS = jnp.array([0.0780, 0.0240])
TYPE_NAMES = ["CG331", "HGA3"]

SWITCH_KW = dict(
    mm_switch_on=3.0,
    mm_switch_width=2.0,
    ml_switch_width=1.0,
    complementary_handoff=False,
)


def constant_ml(params, *, atomic_numbers, positions, dst_idx, src_idx,
                batch_segments, batch_size, batch_mask, atom_mask):
    """Placeholder for PhysNet: constant energy, zero forces."""
    del params, atomic_numbers, dst_idx, src_idx, batch_segments, batch_mask
    e = jnp.sum(atom_mask) * jnp.asarray(-1.0)
    return {"energy": e.reshape(batch_size, 1), "forces": jnp.zeros_like(positions)}


def dimer(separation_A: float = 3.5, type_idx=(0, 1, 0, 1)) -> dict:
    """Two 2-atom monomers along x, close enough that intermolecular LJ is on."""
    n = 4
    pos = jnp.array(
        [[0.0, 0, 0], [1.0, 0, 0],
         [separation_A, 0, 0], [separation_A + 1.0, 0, 0]],
        dtype=jnp.float32,
    )
    i = jnp.arange(n)
    dst, src = (a.reshape(-1) for a in jnp.meshgrid(i, i, indexing="ij"))
    return {
        "R": pos,
        "Z": jnp.array([6, 1, 6, 1]),
        "mol_id": jnp.array([0, 0, 1, 1]).reshape(1, n),
        "cgenff_type_idx": jnp.array(type_idx).reshape(1, n),
        "cgenff_charge": jnp.zeros(n).reshape(1, n),
        "atom_mask": jnp.ones(n, dtype=jnp.float32),
        "batch_mask": (dst != src).astype(jnp.float32),
        "dst_idx": dst,
        "src_idx": src,
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
    }


def e_mm(sigma_scale, epsilon_scale, batch):
    """Intermolecular MM energy under the given per-type scales."""
    from mmml.models.hybrid_energy import hybrid_forward

    out = hybrid_forward(
        constant_ml, {"params": {}}, batch, 1,
        MASTER_SIGMAS, MASTER_EPSILONS,
        learn_mm_lj_scales=True,
        mm_lj_sigma_scale=sigma_scale,
        mm_lj_epsilon_scale=epsilon_scale,
        **SWITCH_KW,
    )
    return jnp.asarray(out["e_mm"]).reshape(())


def fit(loss_fn, params, *, lr: float, steps: int):
    """Adam over the LJ-scale leaves. Returns (params, loss_before, loss_after)."""
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
