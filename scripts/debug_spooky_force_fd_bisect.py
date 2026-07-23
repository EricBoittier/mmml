#!/usr/bin/env python3
"""Bisect SpookyPhysNet force–energy consistency (bare model + optional checkpoint)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _fd_force(energy_fn, pos, atom: int, axis: int, h: float = 1e-4) -> float:
    plus = pos.at[atom, axis].add(h)
    minus = pos.at[atom, axis].add(-h)
    return -(float(energy_fn(plus)) - float(energy_fn(minus))) / (2 * h)


def _check_model(
    model,
    variables,
    batch: dict,
    *,
    label: str,
    atoms: list[tuple[int, int]],
    h: float = 1e-4,
) -> dict:
    import jax.numpy as jnp

    pos = batch["positions"]

    def energy_at(p):
        b = {**batch, "positions": p}
        out = model.apply(variables, **b, compute_forces=False)
        return jnp.sum(out["energy"])

    out_f = model.apply(variables, **batch, compute_forces=True)
    forces = np.asarray(out_f["forces"])

    rows = []
    for atom, axis in atoms:
        f_an = float(forces[atom, axis])
        f_fd = _fd_force(energy_at, pos, atom, axis, h=h)
        err = abs(f_an - f_fd)
        rows.append(
            {
                "atom": atom,
                "axis": axis,
                "f_analytic": f_an,
                "f_fd": f_fd,
                "abs_err": err,
            }
        )
    worst = max(rows, key=lambda r: r["abs_err"])
    print(f"\n=== {label} ===")
    print(f"worst abs_err={worst['abs_err']:.6e} atom={worst['atom']} axis={worst['axis']}")
    print(f"  analytic={worst['f_analytic']:.6e} fd={worst['f_fd']:.6e}")
    return {"label": label, "worst": worst, "rows": rows}


def _random_batch(*, spooky_kwargs, n_real: int = 4, max_atoms: int = 8, key):
    import jax
    import jax.numpy as jnp

    from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet

    model = SpookyPhysNet(max_padded_atoms=max_atoms, **spooky_kwargs)
    pos = jax.random.normal(key, (max_atoms, 3)) * 0.5 + jnp.array([5.0, 0.0, 0.0])
    z = jnp.zeros(max_atoms, dtype=jnp.int32)
    z = z.at[:n_real].set(jnp.array([6, 6, 8, 1][:n_real], dtype=jnp.int32))
    atom_mask = (jnp.arange(max_atoms) < n_real).astype(jnp.float32)
    # simple complete graph edges among real atoms
    dst, src = [], []
    for i in range(n_real):
        for j in range(n_real):
            if i != j:
                dst.append(i)
                src.append(j)
    batch = {
        "atomic_numbers": z,
        "charges": jnp.zeros((max_atoms, 1)),
        "spins": atom_mask.reshape(-1, 1),
        "positions": pos,
        "dst_idx": jnp.asarray(dst, dtype=jnp.int32),
        "src_idx": jnp.asarray(src, dtype=jnp.int32),
        "batch_segments": jnp.zeros(max_atoms, dtype=jnp.int32),
        "batch_size": 1,
        "batch_mask": jnp.ones((len(dst), 1, 1, 1)),
        "atom_mask": atom_mask,
    }
    variables = model.init(jax.random.PRNGKey(0), **batch, compute_forces=False)
    return model, variables, batch


def _load_ckpt(path: Path):
    from mmml.utils.model_checkpoint import load_model_from_checkpoint

    model, params, _meta = load_model_from_checkpoint(path)
    return model, {"params": params}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--h", type=float, default=1e-4)
    args = parser.parse_args()

    import jax

    configs = [
        ("zbl_only", dict(zbl=True, charges=False, efa=False)),
        ("charges", dict(zbl=True, charges=True, efa=False)),
        ("efa", dict(zbl=True, charges=False, efa=True)),
        ("full", dict(zbl=True, charges=True, efa=True)),
    ]
    atoms = [(0, 0), (1, 1), (2, 2)]

    if args.checkpoint and args.checkpoint.is_file():
        model, variables = _load_ckpt(args.checkpoint)
        cfg = getattr(model, "return_attributes", lambda: {})()
        print("checkpoint config:", json.dumps({k: cfg.get(k) for k in ("efa", "charges", "zbl")}, default=str))
        # minimal batch from model natoms
        max_atoms = int(getattr(model, "max_padded_atoms", 35))
        _, _, batch = _random_batch(
            spooky_kwargs=dict(zbl=True, charges=True, efa=bool(cfg.get("efa", False))),
            n_real=min(4, max_atoms),
            max_atoms=max_atoms,
            key=jax.random.PRNGKey(1),
        )
        _check_model(model, variables, batch, label=f"ckpt {args.checkpoint.name}", atoms=atoms, h=args.h)
        return 0

    for name, kw in configs:
        model, variables, batch = _random_batch(
            spooky_kwargs=kw,
            key=jax.random.PRNGKey(2),
        )
        _check_model(model, variables, batch, label=name, atoms=atoms, h=args.h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
