import argparse
import functools
import os
import warnings

import e3x
from flax import linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

warnings.simplefilter(action="ignore", category=FutureWarning)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--max_degree", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--num_basis_functions", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--num_train", type=int, default=2000)
    parser.add_argument("--num_valid", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_pseudotensors", type=str2bool, default=True)
    parser.add_argument("--scan_max_degree", type=str2bool, default=True)
    parser.add_argument("--plot_scan", type=str2bool, default=True)
    parser.add_argument("--save_valid_predictions", type=str2bool, default=True)
    parser.add_argument(
        "--valid_predictions_path",
        type=str,
        default="dihedral_valid_predictions.npz",
    )
    return parser.parse_args()


@jax.jit
def signed_dihedral(xyz):
    """
    xyz: (..., 4, 3)
    returns: (..., 1) signed dihedral angle in radians
    """
    p0 = xyz[..., 1, :]
    p1 = xyz[..., 0, :]
    p2 = xyz[..., 2, :]
    p3 = xyz[..., 3, :]

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = jnp.linalg.norm(b1, axis=-1, keepdims=True)
    b1_hat = b1 / jnp.clip(b1_norm, a_min=1e-8)

    v = b0 - jnp.sum(b0 * b1_hat, axis=-1, keepdims=True) * b1_hat
    w = b2 - jnp.sum(b2 * b1_hat, axis=-1, keepdims=True) * b1_hat

    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1_hat, v) * w, axis=-1)
    angle = jnp.arctan2(y, x)
    return angle[..., None]


def wrap_angle(delta):
    return jnp.arctan2(jnp.sin(delta), jnp.cos(delta))


def angular_mse(prediction, target):
    return jnp.mean(jnp.square(wrap_angle(prediction - target)))


def angular_mae(prediction, target):
    return jnp.mean(jnp.abs(wrap_angle(prediction - target)))


def make_dataset(key, num_samples):
    key_pos, key_noise = jax.random.split(key)
    xyz = jax.random.uniform(key_pos, (num_samples, 4, 3), minval=-1.0, maxval=1.0)
    xyz = xyz + 0.05 * jax.random.normal(key_noise, xyz.shape)
    dihedral = signed_dihedral(xyz)[..., 0]
    atomic_numbers = jnp.full((num_samples, 4), 6, dtype=jnp.int32)
    return {
        "positions": xyz.astype(jnp.float32),
        "atomic_numbers": atomic_numbers,
        "dihedral": dihedral.astype(jnp.float32),
    }


def split_dataset(key, dataset, num_train, num_valid):
    num_total = num_train + num_valid
    if num_total > dataset["positions"].shape[0]:
        raise ValueError("Requested more train/valid samples than available.")
    perm = jax.random.permutation(key, dataset["positions"].shape[0])[:num_total]
    train_idx = perm[:num_train]
    valid_idx = perm[num_train:]
    train = {k: v[train_idx] for k, v in dataset.items()}
    valid = {k: v[valid_idx] for k, v in dataset.items()}
    return train, valid


def prepare_batches(key, data, batch_size):
    n = data["positions"].shape[0]
    steps = n // batch_size
    if steps == 0:
        raise ValueError("Batch size larger than dataset.")
    perm = jax.random.permutation(key, n)[: steps * batch_size]
    perm = perm.reshape(steps, batch_size)
    return [
        {
            "positions": data["positions"][idx],
            "atomic_numbers": data["atomic_numbers"][idx],
            "dihedral": data["dihedral"][idx],
        }
        for idx in perm
    ]


class MessagePassingModel(nn.Module):
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 2
    num_basis_functions: int = 16
    cutoff: float = 4.0
    max_atomic_number: int = 10
    include_pseudotensors: bool = True

    @nn.compact
    def __call__(self, atomic_numbers, positions, dst_idx, src_idx):
        displacements = positions[src_idx] - positions[dst_idx]
        basis = e3x.nn.basis(
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
        )

        x = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1,
            features=self.features,
        )(atomic_numbers)

        for _ in range(self.num_iterations):
            y = e3x.nn.MessagePass(
                include_pseudotensors=self.include_pseudotensors,
                max_degree=self.max_degree,
            )(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            x = e3x.nn.add(x, y)
            x = e3x.nn.Dense(self.features)(x)
            x = e3x.nn.silu(x)
            x = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(x)
            x = e3x.nn.add(x, y)

        atomic = nn.Dense(1, use_bias=False)(x)
        atomic = jnp.squeeze(atomic, axis=-1)
        graph_scalar = jnp.sum(atomic)
        #return jnp.arctan2(jnp.sin(graph_scalar), jnp.cos(graph_scalar))

        return graph_scalar


@functools.partial(jax.jit, static_argnames=("model_apply", "optimizer_update"))
def train_step(model_apply, optimizer_update, params, opt_state, batch, dst_idx, src_idx):
    def loss_fn(p):
        pred = jax.vmap(
            lambda z, r: model_apply(
                p,
                atomic_numbers=z,
                positions=r,
                dst_idx=dst_idx,
                src_idx=src_idx,
            )
        )(batch["atomic_numbers"], batch["positions"])
        loss = angular_mse(pred, batch["dihedral"])
        return loss, pred

    (loss, pred), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    mae = angular_mae(pred, batch["dihedral"])
    return params, opt_state, loss, mae


@functools.partial(jax.jit, static_argnames=("model_apply",))
def eval_step(model_apply, params, batch, dst_idx, src_idx):
    pred = jax.vmap(
        lambda z, r: model_apply(
            params,
            atomic_numbers=z,
            positions=r,
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
    )(batch["atomic_numbers"], batch["positions"])
    loss = angular_mse(pred, batch["dihedral"])
    mae = angular_mae(pred, batch["dihedral"])
    return loss, mae, pred


def train_model(key, model, train_data, valid_data, num_epochs, learning_rate, batch_size):
    key, init_key = jax.random.split(key)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(4)
    params = model.init(
        init_key,
        atomic_numbers=train_data["atomic_numbers"][0],
        positions=train_data["positions"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    key, batch_key = jax.random.split(key)
    valid_batches = prepare_batches(batch_key, valid_data, batch_size)
    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_mae": [],
        "valid_mae": [],
    }
    best_valid_loss = float("inf")
    best_valid_epoch = -1
    best_valid_pred = None
    best_valid_true = None

    for epoch in range(1, num_epochs + 1):
        key, batch_key = jax.random.split(key)
        train_batches = prepare_batches(batch_key, train_data, batch_size)

        train_loss = 0.0
        train_mae = 0.0
        for i, batch in enumerate(train_batches):
            params, opt_state, loss, mae = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                params=params,
                opt_state=opt_state,
                batch=batch,
                dst_idx=dst_idx,
                src_idx=src_idx,
            )
            train_loss += (loss - train_loss) / (i + 1)
            train_mae += (mae - train_mae) / (i + 1)

        valid_loss = 0.0
        valid_mae = 0.0
        valid_pred_chunks = []
        valid_true_chunks = []
        for i, batch in enumerate(valid_batches):
            loss, mae, pred = eval_step(
                model_apply=model.apply,
                params=params,
                batch=batch,
                dst_idx=dst_idx,
                src_idx=src_idx,
            )
            valid_loss += (loss - valid_loss) / (i + 1)
            valid_mae += (mae - valid_mae) / (i + 1)
            valid_pred_chunks.append(pred)
            valid_true_chunks.append(batch["dihedral"])

        history["train_loss"].append(float(train_loss))
        history["valid_loss"].append(float(valid_loss))
        history["train_mae"].append(float(train_mae))
        history["valid_mae"].append(float(valid_mae))
        print(
            f"epoch {epoch:3d} | "
            f"loss(train/valid): {train_loss:8.5f}/{valid_loss:8.5f} | "
            f"mae(train/valid): {train_mae:8.5f}/{valid_mae:8.5f}"
        )

        epoch_valid_loss = float(valid_loss)
        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            best_valid_epoch = epoch
            best_valid_pred = jnp.concatenate(valid_pred_chunks, axis=0)
            best_valid_true = jnp.concatenate(valid_true_chunks, axis=0)

    history["best_valid_loss"] = best_valid_loss
    history["best_valid_epoch"] = best_valid_epoch
    best_valid_payload = {
        "pred": np.asarray(best_valid_pred),
        "target": np.asarray(best_valid_true),
        "best_valid_loss": best_valid_loss,
        "best_valid_epoch": best_valid_epoch,
    }
    return params, history, best_valid_payload


def run_single(args):
    key = jax.random.PRNGKey(args.seed)
    key_data, key_split = jax.random.split(key)
    dataset = make_dataset(key_data, args.num_train + args.num_valid)
    train_data, valid_data = split_dataset(
        key_split, dataset, args.num_train, args.num_valid
    )
    model = MessagePassingModel(
        features=args.features,
        max_degree=args.max_degree,
        num_iterations=args.num_iterations,
        num_basis_functions=args.num_basis_functions,
        cutoff=args.cutoff,
        include_pseudotensors=args.include_pseudotensors,
    )
    params, history, best_valid_payload = train_model(
        key,
        model,
        train_data,
        valid_data,
        args.num_epochs,
        args.learning_rate,
        args.batch_size,
    )
    if args.save_valid_predictions:
        np.savez(
            args.valid_predictions_path,
            pred=best_valid_payload["pred"],
            target=best_valid_payload["target"],
            best_valid_loss=best_valid_payload["best_valid_loss"],
            best_valid_epoch=best_valid_payload["best_valid_epoch"],
            include_pseudotensors=args.include_pseudotensors,
            max_degree=args.max_degree,
        )
    return params, model, history, best_valid_payload


def run_scan(args):
    results = []
    for degree in range(args.max_degree + 1):
        run_args = argparse.Namespace(**vars(args))
        run_args.max_degree = degree
        if run_args.save_valid_predictions:
            root, ext = os.path.splitext(run_args.valid_predictions_path)
            ext = ext if ext else ".npz"
            run_args.valid_predictions_path = (
                f"{root}_L{degree}_pt{int(run_args.include_pseudotensors)}{ext}"
            )
        print(
            f"\n=== max_degree={degree} | include_pseudotensors={run_args.include_pseudotensors} ==="
        )
        _, _, history, best_valid_payload = run_single(run_args)
        results.append(
            {
                "max_degree": degree,
                "include_pseudotensors": run_args.include_pseudotensors,
                "final_train_loss": history["train_loss"][-1],
                "final_valid_loss": history["valid_loss"][-1],
                "final_train_mae": history["train_mae"][-1],
                "final_valid_mae": history["valid_mae"][-1],
                "best_valid_loss": best_valid_payload["best_valid_loss"],
                "best_valid_epoch": best_valid_payload["best_valid_epoch"],
                "valid_predictions_path": (
                    run_args.valid_predictions_path
                    if run_args.save_valid_predictions
                    else None
                ),
            }
        )

    if args.plot_scan:
        degrees = [row["max_degree"] for row in results]
        losses = [row["final_valid_loss"] for row in results]
        plt.figure(figsize=(6, 4))
        plt.plot(degrees, losses, marker="o")
        plt.xlabel("max_degree")
        plt.ylabel("final validation angular MSE")
        plt.title(
            f"Dihedral loss vs max_degree (pseudotensors={args.include_pseudotensors})"
        )
        plt.grid(alpha=0.3)
        plt.tight_layout()

    return results


def main(args):
    if args.scan_max_degree:
        return run_scan(args)
    return run_single(args)


if __name__ == "__main__":
    args = get_args()
    output = main(args)
    print(output)
