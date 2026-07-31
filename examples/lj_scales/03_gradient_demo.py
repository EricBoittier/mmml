#!/usr/bin/env python
"""Step 03 — prove the gradient reaches σ and ε.

The whole approach rests on the per-type scales being differentiable *inside* the
hybrid energy. Don't take that on faith: build a two-monomer system and
differentiate. Self-contained — no dataset, no CHARMM, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np

from _toy import MASTER_EPSILONS, MASTER_SIGMAS, TYPE_NAMES, dimer, e_mm

from mmml.models.mm_lj_scales import (
    apply_mm_lj_scales,
    attach_mm_lj_scales,
    split_mm_lj_scale_params,
)

print("=== 03: gradients reach the LJ scales ===")

# --- the parameter tree ----------------------------------------------------
# Scales live as two ordinary leaves next to the network weights, so the same
# optimizer updates both. They start at exactly 1.0, which makes an untrained
# model bit-identical to stock CGenFF.
params = attach_mm_lj_scales({"params": {"...network weights..."}}, n_types=2)
print("\n-- parameter tree --")
print("  leaves      :", sorted(params.keys()))
print("  sigma init  :", np.asarray(params["mm_lj_sigma_scale"]))
print("  epsilon init:", np.asarray(params["mm_lj_epsilon_scale"]))

model_params, sig, eps = split_mm_lj_scale_params(params)
print("  network sees:", sorted(model_params.keys()), "(scales stripped)")

s, e = apply_mm_lj_scales(MASTER_SIGMAS, MASTER_EPSILONS, sig, eps)
identical = bool(np.allclose(s, MASTER_SIGMAS) and np.allclose(e, MASTER_EPSILONS))
print(f"  unit scales reproduce stock CGenFF exactly: {identical}")
assert identical, "unit scales must be a no-op"

# --- differentiate ---------------------------------------------------------
batch = dimer()
g_sig, g_eps = jax.grad(lambda a, b: e_mm(a, b, batch), argnums=(0, 1))(
    jnp.ones(2), jnp.ones(2)
)
print("\n-- dE_MM/dscale at s = 1 --")
print(f"  {'type':10s} {'d/dsigma':>12s} {'d/depsilon':>12s}")
for name, a, b in zip(TYPE_NAMES, np.asarray(g_sig), np.asarray(g_eps)):
    print(f"  {name:10s} {a:12.6f} {b:12.6f}")

finite = bool(np.all(np.isfinite(g_sig)) and np.all(np.isfinite(g_eps)))
nonzero = bool(np.any(np.abs(g_sig) > 0) and np.any(np.abs(g_eps) > 0))
print(f"\n  finite: {finite}   non-zero: {nonzero}")
assert finite and nonzero, "gradients must be finite and non-zero"

# --- separation dependence -------------------------------------------------
# Pull the monomers apart and the gradient dies with the switch: beyond the
# cutoff there is nothing to learn from. This is why the sampling range of your
# training set decides which part of the LJ curve actually gets fitted.
print("\n-- gradient vs separation (the switch turns MM off) --")
print(f"  {'r (A)':>7s} {'E_MM':>12s} {'|dE/deps|':>12s}")
for r in (3.0, 3.5, 4.0, 5.0, 6.0, 8.0):
    b = dimer(separation_A=r)
    energy = float(e_mm(jnp.ones(2), jnp.ones(2), b))
    g = jax.grad(lambda a, c: e_mm(a, c, b), argnums=1)(jnp.ones(2), jnp.ones(2))
    print(f"  {r:7.1f} {energy:12.6f} {float(np.abs(np.asarray(g)).sum()):12.6f}")

print("\n03: OK")
