"""Cutoff intermolecular pair list with the ``lambda_t`` / ``elec_scale`` hooks.

Replaces ``gpu_pairs.make_static_pair_fn`` for the solvated runs. Measured on
the 2709-atom water box, gradient of ``mm_nonbonded + ml_mm_elec``:

    static, complete O(N^2)   3 665 250 slots   1.0x    dE = 0
    cutoff 12 A, tight cap    1 117 586 slots   4.0x    dE = 3e-6 eV

The static list was introduced on the belief that avoiding host-side rebuilds
would be faster, and it was honestly reported at the time as showing no
measurable gain on GPU. It is in fact *more* work: 3.3x the pair slots, and the
switching function then multiplies most of them by zero.

Two things that look like optimisations here and are not:

* **Padding headroom.** The neighbour list pads to a fixed capacity so shapes
  stay static for XLA. At the default 2x headroom a 12 A list pads to ~3.9M
  slots -- LARGER than the complete O(N^2) list -- and measured *slower* than
  the thing it replaced. The capacity has to be sized from the actual pair
  count, which is what ``capacity_headroom`` does here.

* **Retuning the Ewald alpha.** Lowering alpha shrinks the reciprocal sum, and
  the reciprocal is genuinely the larger remaining cost. But alpha is shared
  with the self-energy and exclusion corrections, and overriding it in the term
  kwargs alone breaks their cancellation: measured dE = +12.2 eV at alpha 0.29
  and +307 eV at 0.25. Leave it alone.

The 12 A cutoff itself is exact to 3e-6 eV for this system. 10 A is not
(dE = +0.13 eV), so do not shorten it for speed.
"""

from __future__ import annotations

import numpy as np


def make_cutoff_pair_fn(system, cutoff_A: float = 12.0, *,
                        with_lambda: bool = False,
                        capacity_headroom: float = 1.45,
                        verbose: bool = True):
    """Padded cutoff pair list, rebuilt on the host when the driver asks.

    ``capacity_headroom`` multiplies the pair count measured on the *initial*
    geometry, which is a lower bound rather than a typical value: the cached box
    is equilibrated at one state point and the count grows as windows walk along
    the coordinate. At 1.20 a production run overflowed after ~7 minutes
    (1 207 851 required against 1 166 416 allocated), losing the whole job.

    1.45 covers the growth actually observed. Overflow still raises rather than
    truncating -- dropped interactions would be a silent physics error, which is
    far worse than a crash -- and the cost of the extra slots is small next to
    the 3.7x saving over the complete O(N^2) list.
    """
    import jax.numpy as jnp

    from mmml.md.neighbors import make_intermolecular_neighbor_fn

    probe = make_intermolecular_neighbor_fn(system, cutoff_A=cutoff_A)
    n_real = int(np.asarray(probe(np.asarray(system.R), None)["pair_mask"]).sum())
    cap = int(n_real * float(capacity_headroom))
    inner = make_intermolecular_neighbor_fn(system, cutoff_A=cutoff_A,
                                            capacity=cap, on_overflow="raise")
    if verbose:
        n_all = system.n_atoms * (system.n_atoms - 1) // 2
        print(f"  cutoff pair list: {n_real} pairs within {cutoff_A:g} A, "
              f"padded to {cap} ({cap / max(n_all, 1) * 100:.0f}% of the "
              f"complete list)")

    state = {"lambda_t": jnp.asarray(0.0, dtype=jnp.float64),
             "elec_scale": jnp.asarray(1.0, dtype=jnp.float64)}

    def neighbor_fn(positions=None, box=None):
        pos = np.asarray(system.R if positions is None else positions,
                         dtype=np.float64)
        payload = inner(pos, box)
        if with_lambda:
            return {**payload,
                    "lambda_t": state["lambda_t"],
                    "elec_scale": state["elec_scale"]}
        return payload

    def set_lambda(value: float) -> None:
        state["lambda_t"] = jnp.asarray(float(value), dtype=jnp.float64)

    def set_elec_scale(value: float) -> None:
        state["elec_scale"] = jnp.asarray(float(value), dtype=jnp.float64)

    neighbor_fn.n_pairs = cap                       # type: ignore[attr-defined]
    neighbor_fn.set_lambda = set_lambda             # type: ignore[attr-defined]
    neighbor_fn.set_elec_scale = set_elec_scale     # type: ignore[attr-defined]
    return neighbor_fn
