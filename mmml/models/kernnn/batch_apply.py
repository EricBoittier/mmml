"""Hybrid MLpot batch apply for KerNN (4-atom ABCC monomers / dimers)."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from mmml.models.kernnn.model import KerNNConfig, KerNNStats, energy_and_forces


def build_kernnn_batch_apply(
    *,
    params: Mapping[str, Any],
    stats: KerNNStats | Mapping[str, Any],
    config: KerNNConfig | Mapping[str, Any] | None = None,
    max_atoms: int,
    atoms_per_monomer: int | Sequence[int] = 4,
) -> Callable[..., dict[str, Array]]:
    """Return ``apply_model(Z, R, N, N_a=None)`` for hybrid ``setup_calculator``.

    Only supports monomer size 4 (H2CO / ABCC). Dimers are evaluated as the sum
    of two independent KerNN monomers (no learned cross terms).
    """
    cfg = config if isinstance(config, KerNNConfig) else KerNNConfig.from_mapping(config)
    if isinstance(atoms_per_monomer, (list, tuple)):
        per_list = [int(x) for x in atoms_per_monomer]
    else:
        per_list = [int(atoms_per_monomer)]
    if any(n != 4 for n in per_list):
        raise ValueError(
            f"KerNN hybrid backend requires 4-atom monomers; got {per_list}"
        )
    if max_atoms < 4:
        raise ValueError(f"max_atoms={max_atoms} too small for KerNN (need >= 4)")

    mono_n = 4
    slice_pad = max_atoms

    def _eval_mono(R: Array) -> tuple[Array, Array]:
        e, f = energy_and_forces(params, R[:mono_n], stats, config=cfg)
        f_pad = jnp.zeros((max_atoms, 3), dtype=R.dtype).at[:mono_n].set(f)
        return e, f_pad

    def _eval_one(R: Array, N: Array, N_a: Array) -> tuple[Array, Array]:
        na = jnp.asarray(N_a, dtype=jnp.int32)
        n_tot = jnp.asarray(N, dtype=jnp.int32)
        is_dimer = n_tot > na

        def _mono(_):
            return _eval_mono(R)

        def _dimer(_):
            r_ext = jnp.concatenate([R, jnp.zeros_like(R[:slice_pad])], axis=0)
            window_b = jax.lax.dynamic_slice(
                r_ext,
                (na, jnp.asarray(0, dtype=na.dtype)),
                (max_atoms, 3),
            )
            e_a, f_a = _eval_mono(R)
            e_b, f_b_window = _eval_mono(window_b)
            idx = jnp.arange(max_atoms, dtype=jnp.int32)
            in_b = (idx >= na) & (idx < n_tot)
            safe_local = jnp.where(in_b, idx - na, 0)
            f_b = jnp.where(in_b[:, None], f_b_window[safe_local], 0.0)
            return e_a + e_b, f_a + f_b

        return jax.lax.cond(is_dimer, _dimer, _mono, operand=None)

    vmapped = jax.vmap(_eval_one, in_axes=(0, 0, 0))

    def apply_model(
        atomic_numbers: Array,
        positions: Array,
        batch_n: Array,
        batch_n_a: Array | None = None,
    ) -> dict[str, Array]:
        _ = atomic_numbers
        batch_size = positions.shape[0] // max_atoms
        R = positions.reshape(batch_size, max_atoms, 3)
        N = jnp.asarray(batch_n, dtype=jnp.int32).reshape(batch_size)
        if batch_n_a is None:
            mono_n_j = jnp.asarray(mono_n, dtype=jnp.int32)
            N_a = jnp.where(N > mono_n_j, mono_n_j, N)
        else:
            N_a = jnp.asarray(batch_n_a, dtype=jnp.int32).reshape(batch_size)
        energies, forces = vmapped(R, N, N_a)
        return {
            "energy": energies.reshape(batch_size),
            "forces": forces.reshape(batch_size * max_atoms, 3),
        }

    return apply_model


def is_kernnn_checkpoint(path) -> bool:
    """True if ``path`` is a KerNN JSON checkpoint (has model_type/stats)."""
    import json
    from pathlib import Path

    p = Path(path).expanduser()
    if p.is_dir():
        cand = p / "best.json"
        if not cand.is_file():
            cand = p / "params.json"
        p = cand
    if not p.is_file() or p.suffix != ".json":
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    cfg = data.get("config") or {}
    if isinstance(cfg, dict) and str(cfg.get("model_type", "")).lower() == "kernnn":
        return True
    return "stats" in data and "params" in data and "mean_e" in (data.get("stats") or {})
