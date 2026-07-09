#!/usr/bin/env python3
"""Train the E3x QCML MBD surrogate on energy, force, C6, and alpha targets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import e3x
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state

from mmml.models.mbd import E3xMBDModel, mbd_energy_and_forces


@dataclass(frozen=True)
class MBDTrainConfig:
    features: int = 64
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 16
    cutoff: float = 12.0
    max_atomic_number: int = 118


def restore_cache(path: Path) -> dict[str, np.ndarray]:
    cache = {
        key: np.asarray(value)
        for key, value in ocp.PyTreeCheckpointer().restore(path).items()
    }
    required = {
        "R", "Z", "Q", "S", "E_mbd", "F_mbd", "C6_mbd", "alpha_mbd", "atom_mask"
    }
    missing = required.difference(cache)
    if missing:
        raise KeyError(f"Cache is missing required fields: {sorted(missing)}")
    return cache


def limit_cache(cache, max_structures):
    """Limit all structure-aligned arrays before splitting."""
    if max_structures is None:
        return cache
    if max_structures <= 0:
        raise ValueError("max_structures must be positive")
    size = cache["R"].shape[0]
    limit = min(max_structures, size)
    return {
        key: value[:limit] if value.ndim > 0 and value.shape[0] == size else value
        for key, value in cache.items()
    }


def split_indices(size: int, validation_fraction: float, seed: int):
    order = np.random.default_rng(seed).permutation(size)
    num_validation = max(1, round(size * validation_fraction)) if validation_fraction else 0
    return order[num_validation:], order[:num_validation]


def eligible_indices(cache, max_atoms):
    atom_counts = np.asarray(cache["atom_mask"].sum(axis=1), dtype=np.int32)
    if max_atoms is None:
        return np.arange(len(atom_counts))
    if max_atoms <= 0:
        raise ValueError("max_atoms must be positive")
    indices = np.flatnonzero(atom_counts <= max_atoms)
    if not len(indices):
        raise ValueError(f"No structures contain at most {max_atoms} atoms")
    return indices


def bucket_indices(cache, indices, bucket_width):
    if bucket_width <= 0:
        raise ValueError("bucket_width must be positive")
    atom_counts = np.asarray(cache["atom_mask"].sum(axis=1), dtype=np.int32)
    buckets = {}
    for index in indices:
        count = int(atom_counts[index])
        ceiling = min(
            ((count + bucket_width - 1) // bucket_width) * bucket_width,
            cache["R"].shape[1],
        )
        buckets.setdefault(ceiling, []).append(int(index))
    return {
        ceiling: np.asarray(values, dtype=np.int64)
        for ceiling, values in sorted(buckets.items())
    }


def iter_batches(
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    indices = np.array(indices, copy=True)
    if rng is not None:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        example_mask = np.ones(len(selected), dtype=np.float32)
        if len(selected) < batch_size:
            padding = batch_size - len(selected)
            selected = np.pad(selected, (0, padding), mode="edge")
            example_mask = np.pad(example_mask, (0, padding))
        yield selected, example_mask


def iter_bucket_batches(buckets, batch_size, rng=None):
    ceilings = np.asarray(list(buckets), dtype=np.int32)
    if rng is not None:
        rng.shuffle(ceilings)
    for ceiling in ceilings:
        for indices, example_mask in iter_batches(
            buckets[int(ceiling)], batch_size, rng
        ):
            yield indices, example_mask, int(ceiling)


def make_batch(cache, indices, example_mask, max_atoms=None):
    positions = cache["R"][indices, :max_atoms].astype(np.float32)
    atomic_numbers = cache["Z"][indices, :max_atoms].astype(np.int32)
    atom_mask = cache["atom_mask"][indices, :max_atoms].astype(np.float32)
    target_forces = cache["F_mbd"][indices, :max_atoms]
    target_c6 = cache["C6_mbd"][indices, :max_atoms]
    target_alpha = cache["alpha_mbd"][indices, :max_atoms]
    batch_size, max_atoms = atomic_numbers.shape
    template_dst, template_src = map(
        np.asarray,
        e3x.ops.sparse_pairwise_indices(max_atoms),
    )
    offsets = np.arange(batch_size, dtype=np.int32)[:, None] * max_atoms
    dst_idx = (template_dst[None, :] + offsets).reshape(-1)
    src_idx = (template_src[None, :] + offsets).reshape(-1)
    edge_mask = (atom_mask[:, template_dst] * atom_mask[:, template_src]).reshape(-1)
    return {
        "positions": jnp.asarray(positions.reshape(-1, 3)),
        "atomic_numbers": jnp.asarray(atomic_numbers.reshape(-1)),
        "charge": jnp.asarray(cache["Q"][indices].reshape(-1), dtype=jnp.float32),
        "spin": jnp.asarray(cache["S"][indices].reshape(-1), dtype=jnp.float32),
        "dst_idx": jnp.asarray(dst_idx),
        "src_idx": jnp.asarray(src_idx),
        "batch_segments": jnp.repeat(jnp.arange(batch_size), max_atoms),
        "batch_size": batch_size,
        "atom_mask": jnp.asarray(atom_mask.reshape(-1)),
        "edge_mask": jnp.asarray(edge_mask),
        "target_energy": jnp.asarray(cache["E_mbd"][indices].reshape(-1), dtype=jnp.float32),
        "target_forces": jnp.asarray(target_forces.reshape(-1, 3), dtype=jnp.float32),
        "target_c6": jnp.asarray(target_c6.reshape(-1), dtype=jnp.float32),
        "target_alpha": jnp.asarray(target_alpha.reshape(-1), dtype=jnp.float32),
        "example_mask": jnp.asarray(example_mask),
    }


def mbd_loss(
    output,
    forces,
    batch,
    *,
    energy_weight,
    force_weight,
    c6_weight,
    alpha_weight,
):
    example_denominator = jnp.maximum(jnp.sum(batch["example_mask"]), 1.0)
    atom_denominator = jnp.maximum(jnp.sum(batch["atom_mask"]), 1.0)
    energy_error = jnp.square(output["energy"] - batch["target_energy"])
    energy_loss = jnp.sum(energy_error * batch["example_mask"]) / example_denominator
    force_loss = (
        jnp.sum(jnp.square(forces - batch["target_forces"]) * batch["atom_mask"][:, None])
        / (3 * atom_denominator)
    )
    c6_loss = (
        jnp.sum(
            jnp.square(jnp.log1p(output["c6_coefficients"]) - jnp.log1p(batch["target_c6"]))
            * batch["atom_mask"]
        )
        / atom_denominator
    )
    alpha_loss = (
        jnp.sum(
            jnp.square(
                jnp.log1p(output["polarizabilities"])
                - jnp.log1p(batch["target_alpha"])
            )
            * batch["atom_mask"]
        )
        / atom_denominator
    )
    components = {
        "energy": energy_loss,
        "forces": force_loss,
        "c6": c6_loss,
        "alpha": alpha_loss,
    }
    total = (
        energy_weight * energy_loss
        + force_weight * force_loss
        + c6_weight * c6_loss
        + alpha_weight * alpha_loss
    )
    return total, components


def model_inputs(batch):
    keys = (
        "positions", "atomic_numbers", "charge", "spin", "dst_idx", "src_idx",
        "batch_segments", "atom_mask", "edge_mask",
    )
    return {
        **{key: batch[key] for key in keys},
        "batch_size": batch["target_energy"].shape[0],
    }


def build_steps(model, weights):
    def objective(params, batch):
        output, forces = mbd_energy_and_forces(model, params, **model_inputs(batch))
        return mbd_loss(output, forces, batch, **weights)

    @jax.jit
    def train_step(state, batch):
        (loss, components), gradients = jax.value_and_grad(objective, has_aux=True)(
            state.params, batch
        )
        return state.apply_gradients(grads=gradients), loss, components

    @jax.jit
    def validation_step(params, batch):
        return objective(params, batch)

    return train_step, validation_step


def evaluate(params, cache, indices, batch_size, validation_step, bucket_width):
    if not len(indices):
        return float("nan")
    total = 0.0
    count = 0.0
    buckets = bucket_indices(cache, indices, bucket_width)
    for selected, example_mask, max_atoms in iter_bucket_batches(
        buckets, batch_size
    ):
        loss, _ = validation_step(
            params,
            make_batch(cache, selected, example_mask, max_atoms),
        )
        weight = float(example_mask.sum())
        total += float(loss) * weight
        count += weight
    return total / count


def save_checkpoint(path, state, config, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(
        path,
        {"params": state.params, "opt_state": state.opt_state, "step": np.asarray(state.step)},
    )
    (path / "model_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-structures", type=int)
    parser.add_argument("--max-atoms", type=int)
    parser.add_argument("--bucket-width", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--num-basis-functions", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=12.0)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=1.0)
    parser.add_argument("--c6-weight", type=float, default=0.1)
    parser.add_argument("--alpha-weight", type=float, default=0.1)
    args = parser.parse_args()

    cache = limit_cache(restore_cache(args.cache), args.max_structures)
    available_indices = eligible_indices(cache, args.max_atoms)
    training_selection, validation_selection = split_indices(
        len(available_indices), args.validation_fraction, args.seed
    )
    training_indices = available_indices[training_selection]
    validation_indices = available_indices[validation_selection]
    if not len(training_indices):
        raise ValueError("Validation split consumed the full dataset")
    config = MBDTrainConfig(
        features=args.features,
        num_iterations=args.num_iterations,
        num_basis_functions=args.num_basis_functions,
        cutoff=args.cutoff,
    )
    model = E3xMBDModel(**asdict(config))
    training_buckets = bucket_indices(cache, training_indices, args.bucket_width)
    initial_indices, initial_mask, initial_max_atoms = next(
        iter_bucket_batches(training_buckets, args.batch_size)
    )
    initial_batch = make_batch(
        cache, initial_indices, initial_mask, initial_max_atoms
    )
    variables = model.init(jax.random.key(args.seed), **model_inputs(initial_batch))
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.adamw(args.learning_rate, weight_decay=args.weight_decay),
    )
    weights = {
        "energy_weight": args.energy_weight,
        "force_weight": args.force_weight,
        "c6_weight": args.c6_weight,
        "alpha_weight": args.alpha_weight,
    }
    train_step, validation_step = build_steps(model, weights)
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        totals = {"loss": 0.0, "energy": 0.0, "forces": 0.0, "c6": 0.0, "alpha": 0.0}
        count = 0.0
        for indices, example_mask, max_atoms in iter_bucket_batches(
            training_buckets, args.batch_size, rng
        ):
            batch = make_batch(cache, indices, example_mask, max_atoms)
            state, loss, components = train_step(state, batch)
            weight = float(example_mask.sum())
            totals["loss"] += float(loss) * weight
            for key in components:
                totals[key] += float(components[key]) * weight
            count += weight
        metrics = {key: value / count for key, value in totals.items()}
        metrics["epoch"] = epoch
        metrics["validation_loss"] = evaluate(
            state.params,
            cache,
            validation_indices,
            args.batch_size,
            validation_step,
            args.bucket_width,
        )
        print(
            f"epoch={epoch:04d} loss={metrics['loss']:.8g} "
            f"valid={metrics['validation_loss']:.8g} "
            f"E={metrics['energy']:.4g} F={metrics['forces']:.4g} "
            f"C6={metrics['c6']:.4g} alpha={metrics['alpha']:.4g}"
        )
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint = args.workdir / f"epoch-{epoch:04d}"
            save_checkpoint(checkpoint, state, config, metrics)
            print(f"Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
