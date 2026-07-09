#!/usr/bin/env python3
"""Train the E3x molecular multipole model from the QCML Orbax cache."""

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

from mmml.models.multipoles import E3xMultipoleModel


@dataclass(frozen=True)
class TrainConfig:
    features: int = 64
    max_degree: int = 3
    num_iterations: int = 3
    num_basis_functions: int = 16
    cutoff: float = 6.0
    max_atomic_number: int = 118


def restore_cache(path: Path) -> dict[str, np.ndarray]:
    """Restore the cache and validate fields required for training."""
    restored = ocp.PyTreeCheckpointer().restore(path)
    cache = {key: np.asarray(value) for key, value in restored.items()}
    required = {"R", "Z", "Q", "S", "atom_mask", "multipoles"}
    missing = required.difference(cache)
    if missing:
        raise KeyError(f"Cache is missing required fields: {sorted(missing)}")
    if cache["multipoles"].shape[-1] != 16:
        raise ValueError("Expected 16 packed target components through l=3")
    return cache


def split_indices(
    size: int,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    order = np.random.default_rng(seed).permutation(size)
    validation_size = (
        max(1, round(size * validation_fraction)) if validation_fraction else 0
    )
    validation = order[:validation_size]
    training = order[validation_size:]
    if not len(training):
        raise ValueError("Validation split consumed the full dataset")
    return training, validation


def iter_batches(
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield fixed-size index batches and masks for padded final batches."""
    indices = np.array(indices, copy=True)
    if rng is not None:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        example_mask = np.ones(len(batch), dtype=np.float32)
        if len(batch) < batch_size:
            padding = batch_size - len(batch)
            batch = np.pad(batch, (0, padding), mode="edge")
            example_mask = np.pad(example_mask, (0, padding))
        yield batch, example_mask


def make_batch(
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    example_mask: np.ndarray,
) -> dict[str, jax.Array]:
    """Create a flattened, padding-safe E3x graph batch."""
    positions = cache["R"][indices].astype(np.float32)
    atomic_numbers = cache["Z"][indices].astype(np.int32)
    atom_mask = cache["atom_mask"][indices].astype(np.float32)
    batch_size, max_atoms = atomic_numbers.shape
    template_dst, template_src = e3x.ops.sparse_pairwise_indices(max_atoms)
    template_dst = np.asarray(template_dst)
    template_src = np.asarray(template_src)
    offsets = np.arange(batch_size, dtype=np.int32)[:, None] * max_atoms
    dst_idx = (template_dst[None, :] + offsets).reshape(-1)
    src_idx = (template_src[None, :] + offsets).reshape(-1)
    edge_mask = (
        atom_mask[:, template_dst] * atom_mask[:, template_src]
    ).reshape(-1)

    return {
        "positions": jnp.asarray(positions.reshape(-1, 3)),
        "atomic_numbers": jnp.asarray(atomic_numbers.reshape(-1)),
        "charge": jnp.asarray(cache["Q"][indices].reshape(batch_size), dtype=jnp.float32),
        "spin": jnp.asarray(cache["S"][indices].reshape(batch_size), dtype=jnp.float32),
        "dst_idx": jnp.asarray(dst_idx),
        "src_idx": jnp.asarray(src_idx),
        "batch_segments": jnp.repeat(jnp.arange(batch_size), max_atoms),
        "atom_mask": jnp.asarray(atom_mask.reshape(-1)),
        "edge_mask": jnp.asarray(edge_mask),
        "targets": jnp.asarray(cache["multipoles"][indices], dtype=jnp.float32),
        "example_mask": jnp.asarray(example_mask),
    }


def multipole_loss(
    prediction: jax.Array,
    target: jax.Array,
    example_mask: jax.Array,
    max_degree: int = 3,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Average degree-balanced MSE over non-padding examples."""
    losses = {}
    start = 0
    denominator = jnp.maximum(jnp.sum(example_mask), 1.0)
    for degree in range(max_degree + 1):
        width = 2 * degree + 1
        error = prediction[:, start : start + width] - target[:, start : start + width]
        per_example = jnp.mean(jnp.square(error), axis=-1)
        losses[f"l{degree}"] = jnp.sum(per_example * example_mask) / denominator
        start += width
    return jnp.mean(jnp.stack(tuple(losses.values()))), losses


def create_state(
    model: E3xMultipoleModel,
    batch: dict[str, jax.Array],
    seed: int,
    learning_rate: float,
    weight_decay: float,
) -> train_state.TrainState:
    inputs = {key: batch[key] for key in _MODEL_INPUT_KEYS}
    variables = model.init(jax.random.key(seed), **inputs, batch_size=batch["targets"].shape[0])
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
    )


_MODEL_INPUT_KEYS = (
    "positions",
    "atomic_numbers",
    "charge",
    "spin",
    "dst_idx",
    "src_idx",
    "batch_segments",
    "atom_mask",
    "edge_mask",
)


def build_steps(model: E3xMultipoleModel, batch_size: int):
    def loss_fn(params: Any, batch: dict[str, jax.Array]):
        inputs = {key: batch[key] for key in _MODEL_INPUT_KEYS}
        prediction = model.apply(
            {"params": params},
            **inputs,
            batch_size=batch_size,
        )["multipoles"]
        return multipole_loss(prediction, batch["targets"], batch["example_mask"])

    @jax.jit
    def train_step(state: train_state.TrainState, batch: dict[str, jax.Array]):
        (loss, degree_losses), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch
        )
        return state.apply_gradients(grads=gradients), loss, degree_losses

    @jax.jit
    def validation_step(params: Any, batch: dict[str, jax.Array]):
        return loss_fn(params, batch)

    return train_step, validation_step


def evaluate(
    params: Any,
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    validation_step: Any,
) -> float:
    if not len(indices):
        return float("nan")
    weighted_loss = 0.0
    count = 0.0
    for batch_indices, example_mask in iter_batches(indices, batch_size):
        loss, _ = validation_step(
            params,
            make_batch(cache, batch_indices, example_mask),
        )
        weight = float(example_mask.sum())
        weighted_loss += float(loss) * weight
        count += weight
    return weighted_loss / count


def save_checkpoint(
    workdir: Path,
    epoch: int,
    state: train_state.TrainState,
    config: TrainConfig,
    metrics: dict[str, float],
) -> Path:
    checkpoint = workdir / f"epoch-{epoch:04d}"
    payload = {
        "params": state.params,
        "opt_state": state.opt_state,
        "step": np.asarray(state.step),
    }
    ocp.PyTreeCheckpointer().save(checkpoint, payload)
    (checkpoint / "model_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (checkpoint / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--num-basis-functions", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=6.0)
    args = parser.parse_args()

    cache = restore_cache(args.cache)
    training_indices, validation_indices = split_indices(
        cache["R"].shape[0],
        args.validation_fraction,
        args.seed,
    )
    config = TrainConfig(
        features=args.features,
        num_iterations=args.num_iterations,
        num_basis_functions=args.num_basis_functions,
        cutoff=args.cutoff,
    )
    model = E3xMultipoleModel(**asdict(config))
    initial_indices, initial_mask = next(
        iter_batches(training_indices, args.batch_size)
    )
    initial_batch = make_batch(cache, initial_indices, initial_mask)
    state = create_state(
        model,
        initial_batch,
        args.seed,
        args.learning_rate,
        args.weight_decay,
    )
    train_step, validation_step = build_steps(model, args.batch_size)
    rng = np.random.default_rng(args.seed)
    args.workdir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_total = 0.0
        train_count = 0.0
        for batch_indices, example_mask in iter_batches(
            training_indices, args.batch_size, rng
        ):
            batch = make_batch(cache, batch_indices, example_mask)
            state, loss, _ = train_step(state, batch)
            weight = float(example_mask.sum())
            train_total += float(loss) * weight
            train_count += weight
        metrics = {
            "epoch": epoch,
            "train_loss": train_total / train_count,
            "validation_loss": evaluate(
                state.params,
                cache,
                validation_indices,
                args.batch_size,
                validation_step,
            ),
        }
        print(
            f"epoch={epoch:04d} train={metrics['train_loss']:.8g} "
            f"valid={metrics['validation_loss']:.8g}"
        )
        if epoch % args.save_every == 0 or epoch == args.epochs:
            path = save_checkpoint(args.workdir, epoch, state, config, metrics)
            print(f"Saved checkpoint: {path}")


if __name__ == "__main__":
    main()
