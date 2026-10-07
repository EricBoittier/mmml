"""PhysNet energy-readout parameter discovery (optional production adapter).

PhysNet's energy path is not a single linear head on pooled features:

1. Invariant/equivariant atomic features ``x`` (returned as ``output['state']``)
2. ``e3x.nn.Dense(1)`` inside ``_calculate_atomic_energies``
3. ``nn.Dense(1)`` scalar readout
4. Optional per-element ``energy_bias``
5. Optional ZBL repulsion and electrostatics added before the segment sum

Pooled activations of ``x`` and ``∇_w E`` for the last Dense are therefore
**not** assumed equivalent.  Measure cosine similarity on real checkpoints
before treating the two methods as independent information sources.

This module does not run expensive labeling.  It only inspects a frozen
student for acquisition.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def flatten_params(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested Flax param dict to ``path -> array``."""
    out: dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            out.update(flatten_params(v, key))
        return out
    arr = np.asarray(tree)
    if arr.dtype == object:
        return out
    out[prefix] = arr
    return out


def readout_leaf_names(params: Any) -> list[str]:
    """Heuristic: ``energy_bias`` plus Dense kernels whose path mentions energy.

    Callers should record the chosen leaves in the fingerprint JSON rather than
    relying on this heuristic silently.  When in doubt, include every leaf that
    ``jax.grad(energy)`` touches but ``jax.grad(charge_sum)`` does not.
    """
    flat = flatten_params(params)
    names = []
    for path in flat:
        low = path.lower()
        if "energy_bias" in low:
            names.append(path)
            continue
        if "charge" in low or "spin" in low or "zbl" in low:
            continue
        if low.endswith("/kernel") or low.endswith("/bias"):
            # Keep; production configs should subset to the final energy Dense.
            names.append(path)
    # Prefer energy_bias-only + last two Dense kernels if many Dense leaves.
    energy = [n for n in names if "energy_bias" in n.lower()]
    dense = [n for n in names if n not in energy]
    if len(dense) > 4:
        dense = dense[-2:]
    return energy + dense


def architecture_equivalence_note() -> str:
    return (
        "PhysNet energy readout is e3x.Dense(1) then nn.Dense(1) on atomic "
        "features, plus optional energy_bias, ZBL, and electrostatics. "
        "Pooled pre-readout activations and energy Jacobians w.r.t. the last "
        "Dense are equivalent only for a purely linear E = sum_i w·φ_i model "
        "(see LinearStudent tests).  Measure alignment on the loaded checkpoint."
    )
