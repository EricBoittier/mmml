#!/usr/bin/env python
"""Step 04 — plant a known scale, recover it, and meet the degeneracy.

This is the most important step in the ladder to actually read.

σ and ε are **mutually degenerate against an energy-only target**: a deeper well
with a slightly larger radius produces the same energy as a shallower well with a
smaller one. So you can drive the loss to zero and still recover *wrong
parameters*. Part C below demonstrates exactly that failure.

Part E shows the other way a long run goes wrong: with nothing holding them in
range, the scales simply drift until the LJ term is unphysical or NaN.

Self-contained — no dataset, no CHARMM, no GPU. Runs in a few seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np

from _toy import dimer, e_mm, fit

from mmml.models.mm_lj_scales import (
    MM_LJ_EPSILON_SCALE_BOUNDS as EPS_BOUNDS,
    MM_LJ_SIGMA_SCALE_BOUNDS as SIG_BOUNDS,
    attach_mm_lj_scales,
    split_mm_lj_scale_params,
)

RTOL = 5e-2


def loss_against(targets, batches, *, freeze):
    """MSE vs reference energies, optionally holding one leaf fixed."""

    def loss_fn(p):
        _, sig, eps = split_mm_lj_scale_params(p)
        if freeze == "sigma":
            sig = jax.lax.stop_gradient(sig)
        elif freeze == "epsilon":
            eps = jax.lax.stop_gradient(eps)
        return sum(
            (e_mm(sig, eps, b) - t) ** 2 for b, t in zip(batches, targets)
        ) / len(batches)

    return loss_fn


failures: list[str] = []

# --- A. epsilon, single type, sigma frozen ---------------------------------
print("=== 04: miniature fits ===")
print("\n-- A. recover a planted epsilon (one type, sigma frozen) --")
single = dimer(type_idx=(0, 0, 0, 0))
truth = 1.6
target = e_mm(jnp.ones(2), jnp.array([truth, 1.0]), single)

p, l0, l1 = fit(loss_against([target], [single], freeze="sigma"),
                attach_mm_lj_scales({"params": {}}, 2), lr=3e-2, steps=400)
_, sig_out, eps_out = split_mm_lj_scale_params(p)
got = float(np.asarray(eps_out)[0])
print(f"  loss {l0:.3e} -> {l1:.3e}")
print(f"  planted {truth}  recovered {got:.4f}")
print(f"  type 1 (absent from the system) stays at {float(np.asarray(eps_out)[1]):.6f}")
print(f"  frozen sigma untouched: {np.asarray(sig_out)}")
if abs(got - truth) > RTOL * truth:
    failures.append(f"A: epsilon {got:.4f} != {truth}")

# --- B. sigma, single type, epsilon frozen ---------------------------------
print("\n-- B. recover a planted sigma (one type, epsilon frozen) --")
truth_s = 1.04
target = e_mm(jnp.array([truth_s, 1.0]), jnp.ones(2), single)
p, l0, l1 = fit(loss_against([target], [single], freeze="epsilon"),
                attach_mm_lj_scales({"params": {}}, 2), lr=1e-2, steps=600)
_, sig_out, _ = split_mm_lj_scale_params(p)
got_s = float(np.asarray(sig_out)[0])
print(f"  loss {l0:.3e} -> {l1:.3e}")
print(f"  planted {truth_s}  recovered {got_s:.4f}")
if abs(got_s - truth_s) > RTOL * truth_s:
    failures.append(f"B: sigma {got_s:.4f} != {truth_s}")

# --- C. the trap: fit both at once -----------------------------------------
print("\n-- C. THE TRAP: fit sigma and epsilon together on one energy --")
target = e_mm(jnp.ones(2), jnp.array([truth, 1.0]), single)
p, l0, l1 = fit(loss_against([target], [single], freeze=None),
                attach_mm_lj_scales({"params": {}}, 2), lr=3e-2, steps=400)
_, sig_bad, eps_bad = split_mm_lj_scale_params(p)
got_bad = float(np.asarray(eps_bad)[0])
print(f"  loss {l0:.3e} -> {l1:.3e}   <- converged just as well!")
print(f"  planted epsilon {truth}  recovered {got_bad:.4f}   <- WRONG")
print(f"  sigma drifted to {float(np.asarray(sig_bad)[0]):.4f} to compensate")
print("\n  Loss alone does NOT tell you the parameters are right. In real")
print("  training the degeneracy is broken by forces (which constrain the shape")
print("  of the curve, not just its value) and by many geometries — which is")
print("  why a distance scan beats a pile of equilibrium structures.")

# --- D. many geometries break the degeneracy per type ----------------------
print("\n-- D. two types identified from a separation scan --")
seps = (3.2, 3.6, 4.1, 4.8, 5.6)
batches = [dimer(d) for d in seps]
truth_v = jnp.array([1.5, 0.6])
targets = [e_mm(jnp.ones(2), truth_v, b) for b in batches]
p, l0, l1 = fit(loss_against(targets, batches, freeze="sigma"),
                attach_mm_lj_scales({"params": {}}, 2), lr=2e-2, steps=1200)
_, _, eps_v = split_mm_lj_scale_params(p)
print(f"  separations {seps}")
print(f"  loss {l0:.3e} -> {l1:.3e}")
print(f"  planted   {np.asarray(truth_v)}")
print(f"  recovered {np.asarray(eps_v)}")
if not np.allclose(np.asarray(eps_v), np.asarray(truth_v), rtol=RTOL):
    failures.append(f"D: {np.asarray(eps_v)} != {np.asarray(truth_v)}")

# --- E. why real training projects the scales after every step -------------
print("\n-- E. the bound that keeps a long run alive --")
runaway = loss_against([20.0 * e_mm(jnp.ones(2), jnp.ones(2), single)],
                       [single], freeze=None)
free, _, _ = fit(runaway, attach_mm_lj_scales({"params": {}}, 2),
                 lr=5e-2, steps=300, clip=False)
held, _, _ = fit(runaway, attach_mm_lj_scales({"params": {}}, 2),
                 lr=5e-2, steps=300, clip=True)

print("  target: 20x the stock E_MM — unreachable inside the bounds")
print(f"  {'':12s} {'s_sigma':>18s} {'s_epsilon':>18s}")
for label, p in (("unbounded", free), ("projected", held)):
    _, s, e = split_mm_lj_scale_params(p)
    print(f"  {label:12s} {str(np.asarray(s)):>18s} {str(np.asarray(e)):>18s}")

print(f"\n  bounds: sigma {SIG_BOUNDS}   epsilon {EPS_BOUNDS}")
print("  The unbounded fit answers with a 25% change in Rmin, which no CGenFF")
print("  radius tolerates — it is chasing an energy the LJ term cannot produce.")
print("  The projected fit saturates instead, which is a legible symptom.")
print("  Left alone the travel does not stop: Adam moves a parameter by about")
print("  the learning rate per step however small the gradient is, and a real")
print("  run takes 500 steps per epoch. An epsilon that reaches zero is fatal —")
print("  the combining rule takes sqrt(eps_i * eps_j), so one type crossing zero")
print("  NaNs every pair that mixes it with a positive type.")

_, sig_free, eps_free = split_mm_lj_scale_params(free)
_, sig_held, eps_held = split_mm_lj_scale_params(held)
if not (np.any(np.asarray(eps_free) > EPS_BOUNDS[1])
        or np.any(np.asarray(sig_free) > SIG_BOUNDS[1])):
    failures.append("E: unbounded fit did not leave the physical range")
if (np.any(np.asarray(sig_held) < SIG_BOUNDS[0] - 1e-6)
        or np.any(np.asarray(sig_held) > SIG_BOUNDS[1] + 1e-6)
        or np.any(np.asarray(eps_held) < EPS_BOUNDS[0] - 1e-6)
        or np.any(np.asarray(eps_held) > EPS_BOUNDS[1] + 1e-6)):
    failures.append("E: projected fit left the physical range")

print()
if failures:
    for f in failures:
        print(f"ERROR: {f}", file=sys.stderr)
    print("04: FAILED", file=sys.stderr)
    sys.exit(1)
print("04: OK")
