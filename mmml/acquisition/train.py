"""Fine-tune identical copies of the initial student under matched settings.

Two experimental settings are first-class:

* ``readout`` — only the linear energy-readout parameters (``w``, ``b``)
* ``full`` — the same parameters for the linear student (it *is* the readout);
  for PhysNet this is reserved as the full-model update path

The unmodified student is a baseline (zero gradient steps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from mmml.acquisition.linear_student import LinearStudent, _energy_from_params, _forces_from_params
import jax
import jax.numpy as jnp

TuneMode = Literal["readout", "full", "unmodified"]


@dataclass
class TrainSettings:
    energy_weight: float = 1.0
    forces_weight: float = 52.91
    learning_rate: float = 0.05
    n_steps: int = 40
    seed: int = 0
    batch_size: int = 4


def _loss_one(w, b, R, Z, mask, e_t, f_t, a_e, a_f, max_z, k_dist):
    e = _energy_from_params(w, b, R, Z, mask, max_z, k_dist)
    f = _forces_from_params(w, b, R, Z, mask, max_z, k_dist)
    n_real = jnp.maximum(mask.sum(), 1.0)
    e_loss = 0.5 * (e - e_t) ** 2
    diff = (f - f_t) * mask[:, None]
    f_loss = 0.5 * jnp.sum(diff * diff) / n_real
    return a_e * e_loss + a_f * f_loss


def finetune_linear(
    student: LinearStudent,
    dataset: dict[str, np.ndarray],
    settings: TrainSettings,
    *,
    mode: TuneMode = "readout",
) -> tuple[LinearStudent, dict[str, Any]]:
    """SGD on readout parameters.  ``unmodified`` returns a copy with no steps."""
    if mode == "unmodified":
        return LinearStudent(
            weights=np.array(student.weights, copy=True),
            bias=np.array(student.bias, copy=True),
            max_z=student.max_z,
            k_dist=student.k_dist,
            name=student.name + "+unmodified",
        ), {"n_steps": 0, "mode": mode, "final_loss": None}

    R = jnp.asarray(dataset["R"], dtype=jnp.float64)
    Z = jnp.asarray(dataset["Z"], dtype=jnp.int32)
    N = jnp.asarray(dataset["N"], dtype=jnp.int32)
    E = jnp.asarray(dataset["E"], dtype=jnp.float64).reshape(-1)
    F = jnp.asarray(dataset["F"], dtype=jnp.float64)
    n = len(E)
    pad = R.shape[1]
    w = jnp.asarray(student.weights, dtype=jnp.float64)
    b = jnp.asarray(student.bias, dtype=jnp.float64)
    a_e = jnp.asarray(settings.energy_weight, dtype=jnp.float64)
    a_f = jnp.asarray(settings.forces_weight, dtype=jnp.float64)
    max_z = int(student.max_z)
    k_dist = int(student.k_dist)
    lr = float(settings.learning_rate)
    rng = np.random.default_rng(int(settings.seed))
    last_loss = None

    def batch_loss(ww, bb, idx):
        def body(i):
            j = idx[i]
            r = R[j]
            z = Z[j]
            mask = (jnp.arange(pad) < N[j]).astype(jnp.float64) * (z > 0).astype(jnp.float64)
            return _loss_one(ww, bb, r, z, mask, E[j], F[j], a_e, a_f, max_z, k_dist)
        losses = jax.vmap(body)(jnp.arange(idx.shape[0]))
        return jnp.mean(losses)

    grad_fn = jax.jit(jax.value_and_grad(batch_loss, argnums=(0, 1)))
    bs = min(int(settings.batch_size), max(n, 1))
    for step in range(int(settings.n_steps)):
        idx = rng.integers(0, n, size=(bs,))
        loss, (gw, gb) = grad_fn(w, b, jnp.asarray(idx))
        last_loss = float(np.asarray(loss))
        w = w - lr * gw
        b = b - lr * gb
    trained = LinearStudent(
        weights=np.asarray(w),
        bias=np.asarray(b),
        max_z=student.max_z,
        k_dist=student.k_dist,
        name=f"{student.name}+{mode}",
    )
    return trained, {
        "n_steps": int(settings.n_steps),
        "mode": mode,
        "final_loss": last_loss,
        "learning_rate": lr,
        "energy_weight": float(settings.energy_weight),
        "forces_weight": float(settings.forces_weight),
        "n_train": int(n),
        "seed": int(settings.seed),
    }


def predict_dataset(student: LinearStudent, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    R = np.asarray(data["R"])
    Z = np.asarray(data["Z"])
    N = np.asarray(data["N"], dtype=np.int32)
    n, pad = R.shape[0], R.shape[1]
    E = np.zeros(n, dtype=np.float64)
    F = np.zeros((n, pad, 3), dtype=np.float64)
    for i in range(n):
        ni = int(N[i])
        e, f = student.energy_forces(R[i, :ni], Z[i, :ni])
        E[i] = e
        F[i, :ni] = f
    return {"E": E, "F": F, "R": R, "Z": Z, "N": N}
