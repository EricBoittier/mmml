"""Static on-device intermolecular pair list.

Complements :func:`make_intermolecular_neighbor_fn`, which rebuilds a padded
pair list on the host at every block boundary. For a solute in a solvent box of
a few thousand atoms that host work dominates: profiling a 1767-atom step gave

    neighbour build (numpy, host)   4.07 s
    energy   (cached, GPU)          0.50 s
    grad     (cached, GPU)          0.02 s

-- 20 ms of GPU work per step behind four seconds of host work, at 107 % CPU and
0 % GPU, with megabytes shipped across the bus each block.

This exploits what the switched force field already guarantees: pairs beyond the
cutoff contribute **exactly** zero, because ``mm_nonbonded`` and ``ml_mm_elec``
both multiply by a switching function that reaches zero at the cutoff. So the
list need not know which pairs are within range. Enumerate every intermolecular
pair once, upload once, and let the switch cull on the GPU.

Measured on this campaign: 8.3 -> 24 steps/s on a 2625-atom water box, with
energies agreeing to 0.000000 eV.

It is O(N^2) in the energy, so beyond roughly ten thousand atoms a real cell
list (``jax_md.partition.neighbor_list``) wins again -- a different regime from
a solute in a solvent box.
"""

from __future__ import annotations


import numpy as np

__all__ = ["make_static_pair_fn", "static_pair_count"]


def _candidate_pairs(system) -> tuple[np.ndarray, np.ndarray]:
    """Every intermolecular pair, excluding anything in ``FFParams.exclusions``."""
    n = int(system.n_atoms)
    mol = np.asarray(system.mol_id)
    i, j = np.triu_indices(n, 1)
    keep = mol[i] != mol[j]
    i, j = i[keep], j[keep]

    ff = getattr(system, "ff_params", None)
    if ff is not None and getattr(ff, "exclusions", None) is not None:
        exc = np.asarray(ff.exclusions).reshape(-1, 2)
        if exc.size:
            # Intermolecular pairs are never in the intramolecular exclusion
            # list, but honour it rather than assume.
            key = i.astype(np.int64) * n + j.astype(np.int64)
            exc_key = (
                np.minimum(exc[:, 0], exc[:, 1]).astype(np.int64) * n
                + np.maximum(exc[:, 0], exc[:, 1]).astype(np.int64)
            )
            drop = np.isin(key, exc_key)
            i, j = i[~drop], j[~drop]
    return i.astype(np.int32), j.astype(np.int32)


def static_pair_count(system) -> int:
    return int(_candidate_pairs(system)[0].shape[0])


def make_static_pair_fn(system, verbose: bool = True, with_lambda: bool = False):
    """Return a ``neighbor_fn`` that hands back the same device arrays every call.

    Marked ``device_native`` so :class:`~mmml.md.drivers.JaxmdDriver` skips the
    host round-trip entirely.

    ``with_lambda`` additionally threads an umbrella window centre through as a
    **traced device scalar** (``lambda_t``), which ``rxncoor`` reads in place of
    its baked-in target. This is what makes a multi-window run affordable:
    building the energy afresh per window, with the target as a Python float in
    the closure, forces a full XLA recompilation each time -- about 25 s for
    this system, i.e. over half an hour of pure compilation across 27 windows
    and three legs each. Updating the *value* of a device scalar keeps the same
    shape and dtype, so the compiled graph is reused.

    Set the centre with ``neighbor_fn.set_lambda(xi0)`` between windows.
    """
    import jax.numpy as jnp

    i, j = _candidate_pairs(system)
    n_pairs = int(i.shape[0])
    payload = {
        "pair_i": jnp.asarray(i),
        "pair_j": jnp.asarray(j),
        "pair_mask": jnp.ones(n_pairs, dtype=jnp.int8),
    }
    state = {
        "lambda_t": jnp.asarray(0.0, dtype=jnp.float64),
        # Scales the ML/MM electrostatics. Threaded here rather than
        # baked into the energy closure so it can be ramped between legs
        # without forcing an XLA recompilation each time.
        "elec_scale": jnp.asarray(1.0, dtype=jnp.float64),
    }

    if verbose:
        mem = n_pairs * (4 + 4 + 1) / 1e6
        print(f"  static pair list: {n_pairs} intermolecular pairs "
              f"({mem:.1f} MB, uploaded once)")

    def neighbor_fn(positions=None, box=None):
        # Deliberately ignores its arguments: the list is complete, and the
        # switching function culls by distance on the GPU.
        del positions, box
        if with_lambda:
            return {
                **payload,
                "lambda_t": state["lambda_t"],
                "elec_scale": state["elec_scale"],
            }
        return payload

    def set_lambda(value: float) -> None:
        state["lambda_t"] = jnp.asarray(float(value), dtype=jnp.float64)

    def set_elec_scale(value: float) -> None:
        state["elec_scale"] = jnp.asarray(float(value), dtype=jnp.float64)

    neighbor_fn.device_native = True  # type: ignore[attr-defined]
    neighbor_fn.n_pairs = n_pairs  # type: ignore[attr-defined]
    neighbor_fn.set_lambda = set_lambda  # type: ignore[attr-defined]
    neighbor_fn.set_elec_scale = set_elec_scale  # type: ignore[attr-defined]
    return neighbor_fn
