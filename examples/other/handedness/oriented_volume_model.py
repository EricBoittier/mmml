"""Synthetic regression: predict oriented volume (scalar triple product) from random3D points.

The target is V = (p1-p0) · ((p2-p0) × (p3-p0)) for the first four particles — a pseudoscalar
(it flips sign under reflections). Message-passing with pseudotensors can represent such outputs;
without them, a parity-even readout often struggles.

Run a small comparison grid (writes CSV with ``valid_mae_last10_mean``, hyperparameters, and ``seed``)::

  python oriented_volume_model.py --sweep --num_epochs 40 --num_train 400 --num_valid 100 \\
      --sweep_csv results/oriented_volume_sweep.csv
"""

import argparse
import csv
import functools
from datetime import datetime, timezone
import warnings

import e3x
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_atoms", type=int, default=4)
  parser.add_argument("--num_samples", type=int, default=1000)
  parser.add_argument("--features", type=int, default=32)
  parser.add_argument("--max_degree", type=int, default=4)
  parser.add_argument(
      "--include_pseudotensors",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument("--num_iterations", type=int, default=2)
  parser.add_argument("--num_basis_functions", type=int, default=8)
  parser.add_argument("--cutoff", type=float, default=10.0)
  parser.add_argument("--num_train", type=int, default=800)
  parser.add_argument("--num_valid", type=int, default=200)
  parser.add_argument("--num_epochs", type=int, default=100)
  parser.add_argument("--learning_rate", type=float, default=1e-3)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
      "--sweep",
      action="store_true",
      help="Train a grid of (max_degree × pseudotensors) and print valid MAE.",
  )
  parser.add_argument(
      "--sweep_max_degrees",
      type=str,
      default="0,1,2,3,4",
      help="Comma-separated max_degree values for --sweep.",
  )
  parser.add_argument(
      "--sweep_csv",
      type=str,
      default=None,
      help="Path to write sweep results CSV (default: oriented_volume_sweep_seed<seed>.csv).",
  )
  parser.add_argument(
      "--sweep_verbose",
      action="store_true",
      help="Print per-epoch metrics for each sweep run (very noisy).",
  )
  return parser.parse_args()


def mean_last_n(values, n=10):
  """Mean of the last n entries; if fewer than n, mean of all."""
  if not values:
    return float("nan")
  return float(np.mean(values[-min(n, len(values)) :]))


def oriented_volume_first_four(positions):
  """Scalar triple product for vertices 0..3: proportional to signed tetrahedron volume."""
  p0, p1, p2, p3 = positions[..., 0, :], positions[..., 1, :], positions[..., 2, :], positions[..., 3, :]
  e1 = p1 - p0
  e2 = p2 - p0
  e3 = p3 - p0
  return jnp.einsum("...i,...i->...", jnp.cross(e1, e2), e3)


def generate_synthetic_dataset(key, num_samples, num_atoms):
  """Uniform points in [-1, 1]^3; identical labels (hydrogen); target = oriented volume."""
  if num_atoms < 4:
    raise ValueError("num_atoms must be at least 4 (need four vertices for triple product).")
  key_r, key_split = jax.random.split(key)
  r = jax.random.uniform(key_r, (num_samples, num_atoms, 3), minval=-1.0, maxval=1.0)
  z = jnp.ones((num_samples, num_atoms), dtype=jnp.int32)
  vol = oriented_volume_first_four(r)
  return {
      "R": np.asarray(r, dtype=np.float32),
      "Z": np.asarray(z, dtype=np.int32),
      "oriented_volume": np.asarray(vol, dtype=np.float32),
  }


class MessagePassingModel(nn.Module):
  features: int = 32
  max_degree: int = 2
  num_iterations: int = 3
  num_basis_functions: int = 8
  cutoff: float = 5.0
  max_atomic_number: int = 2
  include_pseudotensors: bool = True

  def predict_scalar(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
    positions_b = positions.reshape(batch_size, -1, 3)
    num_atoms = positions_b.shape[1]
    positions_flat = positions_b.reshape(-1, 3)
    atomic_numbers_flat = atomic_numbers.reshape(-1)
    offsets = jnp.arange(batch_size) * num_atoms
    dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
    src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)
    displacements = positions_flat[src_idx_flat] - positions_flat[dst_idx_flat]
    basis = e3x.nn.basis(
        displacements,
        num=self.num_basis_functions,
        max_degree=self.max_degree,
        radial_fn=e3x.nn.reciprocal_bernstein,
        cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
    )
    x = e3x.nn.Embed(
        num_embeddings=self.max_atomic_number + 1, features=self.features
    )(atomic_numbers_flat)
    for _ in range(self.num_iterations):
      y = e3x.nn.MessagePass(
          include_pseudotensors=self.include_pseudotensors, max_degree=self.max_degree
      )(x, basis, dst_idx=dst_idx_flat, src_idx=src_idx_flat)
      x = e3x.nn.add(x, y)
      x = e3x.nn.Dense(self.features)(x)
      x = e3x.nn.silu(x)
      x = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(x)
      x = e3x.nn.add(x, y)

    # l=0 scalars per atom (same pattern as EF atomic energy / charge readout).
    x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
    atomic_out = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x)
    atomic_out = jnp.squeeze(atomic_out, axis=(-1, -2, -3))
    return atomic_out.reshape(batch_size, num_atoms).sum(axis=-1)

  @nn.compact
  def __call__(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments=None, batch_size=None):
    if batch_segments is None:
      batch_segments = jnp.zeros(atomic_numbers.shape[:1], dtype=jnp.int32)
      if batch_size is None:
        batch_size = 1
    return self.predict_scalar(
        atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
    )


def prepare_datasets(key, num_train, num_valid, dataset):
  num_data = len(dataset["R"])
  num_draw = num_train + num_valid
  if num_draw > num_data:
    raise RuntimeError(
        f"dataset has {num_data} samples, requested num_train={num_train}, num_valid={num_valid}"
    )
  choice = np.asarray(jax.random.choice(key, num_data, shape=(num_draw,), replace=False))
  train_choice = choice[:num_train]
  valid_choice = choice[num_train:]
  train_data = dict(
      oriented_volume=jnp.asarray(dataset["oriented_volume"], dtype=jnp.float32)[train_choice],
      atomic_numbers=jnp.asarray(dataset["Z"], dtype=jnp.int32)[train_choice],
      positions=jnp.asarray(dataset["R"], dtype=jnp.float32)[train_choice],
  )
  valid_data = dict(
      oriented_volume=jnp.asarray(dataset["oriented_volume"], dtype=jnp.float32)[valid_choice],
      atomic_numbers=jnp.asarray(dataset["Z"], dtype=jnp.int32)[valid_choice],
      positions=jnp.asarray(dataset["R"], dtype=jnp.float32)[valid_choice],
  )
  return train_data, valid_data


def mean_squared_loss(prediction, target):
  return jnp.mean(optax.l2_loss(prediction, target))


def mean_absolute_error(prediction, target):
  return jnp.mean(jnp.abs(prediction - target))


def prepare_batches(key, data, batch_size, num_atoms):
  data_size = len(data["oriented_volume"])
  steps_per_epoch = data_size // batch_size
  perms = jax.random.permutation(key, data_size)
  perms = perms[: steps_per_epoch * batch_size]
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_segments = jnp.zeros(num_atoms, dtype=jnp.int32)
  offsets = jnp.arange(batch_size) * num_atoms
  dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
  dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
  src_idx = (src_idx + offsets[:, None]).reshape(-1)
  return [
      dict(
          oriented_volume=np.asarray(data["oriented_volume"][perm], dtype=np.float32),
          atomic_numbers=jnp.asarray(data["atomic_numbers"][perm].reshape(-1)),
          positions=data["positions"][perm].reshape(-1, 3),
          dst_idx=dst_idx,
          src_idx=src_idx,
          batch_segments=batch_segments,
      )
      for perm in perms
  ]


@functools.partial(jax.jit, static_argnames=("model_apply", "optimizer_update", "batch_size"))
def train_step(model_apply, optimizer_update, batch, batch_size, opt_state, params):
  def loss_fn(params):
    pred = model_apply(
        params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    targets = jnp.asarray(batch["oriented_volume"])
    loss = mean_squared_loss(pred.reshape(-1), targets.reshape(-1))
    return loss, pred

  (loss, pred), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
  updates, opt_state = optimizer_update(grad, opt_state, params)
  params = optax.apply_updates(params, updates)
  targets = jnp.asarray(batch["oriented_volume"])
  mae = mean_absolute_error(pred.reshape(-1), targets.reshape(-1))
  return params, opt_state, loss, mae


@functools.partial(jax.jit, static_argnames=("model_apply", "batch_size"))
def eval_step(model_apply, batch, batch_size, params):
  pred = model_apply(
      params,
      atomic_numbers=batch["atomic_numbers"],
      positions=batch["positions"],
      dst_idx=batch["dst_idx"],
      src_idx=batch["src_idx"],
      batch_segments=batch["batch_segments"],
      batch_size=batch_size,
  )
  targets = jnp.asarray(batch["oriented_volume"])
  loss = mean_squared_loss(pred.reshape(-1), targets.reshape(-1))
  mae = mean_absolute_error(pred.reshape(-1), targets.reshape(-1))
  return loss, mae


def train_model(key, model, train_data, valid_data, num_epochs, learning_rate, batch_size, num_atoms, verbose=True):
  key, init_key = jax.random.split(key)
  optimizer = optax.adam(learning_rate)
  dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
  params = model.init(
      init_key,
      atomic_numbers=train_data["atomic_numbers"][0],
      positions=train_data["positions"][0],
      dst_idx=dst_idx,
      src_idx=src_idx,
      batch_size=1,
  )
  opt_state = optimizer.init(params)
  key, shuffle_key = jax.random.split(key)
  valid_batches = prepare_batches(shuffle_key, valid_data, batch_size, num_atoms)

  valid_mae_per_epoch = []
  last_valid_mae = 0.0
  for epoch in range(1, num_epochs + 1):
    key, shuffle_key = jax.random.split(key)
    train_batches = prepare_batches(shuffle_key, train_data, batch_size, num_atoms)
    train_loss = 0.0
    train_mae = 0.0
    for i, batch in enumerate(train_batches):
      params, opt_state, loss, mae = train_step(
          model_apply=model.apply,
          optimizer_update=optimizer.update,
          batch=batch,
          batch_size=batch_size,
          opt_state=opt_state,
          params=params,
      )
      train_loss += (loss - train_loss) / (i + 1)
      train_mae += (mae - train_mae) / (i + 1)

    valid_loss = 0.0
    valid_mae = 0.0
    for i, batch in enumerate(valid_batches):
      loss, mae = eval_step(
          model_apply=model.apply,
          batch=batch,
          batch_size=batch_size,
          params=params,
      )
      valid_loss += (loss - valid_loss) / (i + 1)
      valid_mae += (mae - valid_mae) / (i + 1)
    last_valid_mae = float(valid_mae)
    valid_mae_per_epoch.append(last_valid_mae)

    if verbose:
      print(f"epoch: {epoch:3d}                    train:   valid:")
      print(f"    mse                     {train_loss:8.5f} {valid_loss: 8.5f}")
      print(f"    oriented volume mae     {train_mae: 8.5f} {valid_mae: 8.5f}")

  metrics = {
      "valid_mae_final": last_valid_mae,
      "valid_mae_last10_mean": mean_last_n(valid_mae_per_epoch, 10),
      "valid_mae_per_epoch": valid_mae_per_epoch,
  }
  return params, model, valid_batches, metrics


def main(args):
  key = jax.random.PRNGKey(args.seed)
  num_atoms = args.num_atoms
  key, data_key = jax.random.split(key)
  dataset = generate_synthetic_dataset(data_key, args.num_samples, num_atoms)
  train_data, valid_data = prepare_datasets(key, args.num_train, args.num_valid, dataset)

  if args.sweep:
    degrees = [int(x.strip()) for x in args.sweep_max_degrees.split(",") if x.strip()]
    csv_path = args.sweep_csv or f"oriented_volume_sweep_seed{args.seed}.csv"
    print("Sweep: reporting mean valid MAE over last 10 epochs (or all epochs if <10).")
    print(f"Writing results to {csv_path}\n")
    fieldnames = [
        "utc_timestamp",
        "seed",
        "max_degree",
        "include_pseudotensors",
        "num_epochs",
        "num_train",
        "num_valid",
        "num_samples",
        "num_atoms",
        "batch_size",
        "learning_rate",
        "features",
        "num_iterations",
        "num_basis_functions",
        "cutoff",
        "valid_mae_final",
        "valid_mae_last10_mean",
    ]
    print(f"{'max_degree':>10} {'pseudotensors':>14} {'mae_last10':>12} {'mae_final':>12}")
    print("-" * 52)
    with open(csv_path, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      for max_degree in degrees:
        for include_pt in (False, True):
          key, run_key = jax.random.split(key)
          model = MessagePassingModel(
              features=args.features,
              max_degree=max_degree,
              num_iterations=args.num_iterations,
              num_basis_functions=args.num_basis_functions,
              cutoff=args.cutoff,
              include_pseudotensors=include_pt,
          )
          _, _, _, metrics = train_model(
              run_key,
              model,
              train_data,
              valid_data,
              args.num_epochs,
              args.learning_rate,
              args.batch_size,
              num_atoms,
              verbose=args.sweep_verbose,
          )
          row = {
              "utc_timestamp": datetime.now(timezone.utc).isoformat(),
              "seed": args.seed,
              "max_degree": max_degree,
              "include_pseudotensors": include_pt,
              "num_epochs": args.num_epochs,
              "num_train": args.num_train,
              "num_valid": args.num_valid,
              "num_samples": args.num_samples,
              "num_atoms": num_atoms,
              "batch_size": args.batch_size,
              "learning_rate": args.learning_rate,
              "features": args.features,
              "num_iterations": args.num_iterations,
              "num_basis_functions": args.num_basis_functions,
              "cutoff": args.cutoff,
              "valid_mae_final": metrics["valid_mae_final"],
              "valid_mae_last10_mean": metrics["valid_mae_last10_mean"],
          }
          writer.writerow(row)
          f.flush()
          print(
              f"{max_degree:10d} {str(include_pt):>14} "
              f"{metrics['valid_mae_last10_mean']:12.5f} {metrics['valid_mae_final']:12.5f}"
          )
    return csv_path

  model = MessagePassingModel(
      features=args.features,
      max_degree=args.max_degree,
      num_iterations=args.num_iterations,
      num_basis_functions=args.num_basis_functions,
      cutoff=args.cutoff,
      include_pseudotensors=args.include_pseudotensors,
  )
  params, model, valid_batches, _ = train_model(
      key,
      model,
      train_data,
      valid_data,
      args.num_epochs,
      args.learning_rate,
      args.batch_size,
      num_atoms,
      verbose=True,
  )
  return params, model, valid_batches


if __name__ == "__main__":
  args = get_args()
  main(args)
