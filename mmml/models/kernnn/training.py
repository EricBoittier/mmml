"""Train KerNN (JAX/Flax) on NPZ datasets with R, E, F."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mmml.models.kernnn.checkpoint import init_params, save_checkpoint
from mmml.models.kernnn.distances import get_bond_length_abcc
from mmml.models.kernnn.kernels import get_1d_kernels_k33
from mmml.models.kernnn.model import (
    KerNNConfig,
    KerNNStats,
    energy_and_forces,
)
from mmml.utils.cli_args import exit_if_unknown_long_options

EV_TO_KCAL_MOL = 23.060541945

_TRAIN_DEFAULTS = {
    "data": "data.npz",
    "workdir": "artifacts/kernnn",
    "ntrain": 3200,
    "nvalid": 400,
    "seed": 42,
    "n_hidden": 20,
    "batch_size": 93,
    "learning_rate": 0.005,
    "f_weight": 10.0,
    "epochs": 1000,
    "patience": 200,
    "ema_decay": 0.999,
    "kernel": "k33",
}


def build_parser() -> argparse.ArgumentParser:
    d = _TRAIN_DEFAULTS
    p = argparse.ArgumentParser(
        description="Train KerNN (kernel Softplus MLP) on NPZ (R, E, F)"
    )
    p.add_argument("--data", type=str, default=d["data"], help="NPZ with R, E, F")
    p.add_argument("--workdir", type=str, default=d["workdir"], help="Output directory")
    p.add_argument("--ntrain", type=int, default=d["ntrain"], help="Training set size")
    p.add_argument("--nvalid", type=int, default=d["nvalid"], help="Validation set size")
    p.add_argument("--seed", type=int, default=d["seed"], help="RNG seed for split/init")
    p.add_argument("--n-hidden", type=int, default=d["n_hidden"], help="Hidden layer width")
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--learning-rate", type=float, default=d["learning_rate"])
    p.add_argument("--f-weight", type=float, default=d["f_weight"], help="Force loss weight")
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument(
        "--patience",
        type=int,
        default=d["patience"],
        help="Early-stop after this many non-improving validation epochs",
    )
    p.add_argument("--ema-decay", type=float, default=d["ema_decay"])
    p.add_argument("--kernel", type=str, default=d["kernel"], help="1D kernel name (default k33)")
    return p


def get_args(argv: list[str] | None = None):
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    exit_if_unknown_long_options(unknown, prog="mmml kernnn-train")
    return args


def _compute_stats(
    positions: np.ndarray,
    energies: np.ndarray,
    idx_train: np.ndarray,
    *,
    kernel: str = "k33",
) -> KerNNStats:
    """Match Torch script: min_r / k stats on full set; E stats on train only."""
    from mmml.models.kernnn.kernels import KERNEL_FNS

    pos_j = jnp.asarray(positions, dtype=jnp.float32)
    e_j = jnp.asarray(energies, dtype=jnp.float32)
    nintdist = 6
    min_idx = int(jnp.argmin(e_j))
    min_r = get_bond_length_abcc(pos_j[min_idx], nintdist)
    r_all = get_bond_length_abcc(pos_j, nintdist)
    k_fn = KERNEL_FNS[kernel]
    k_all = k_fn(r_all, min_r, 1.0)
    mean_k = jnp.mean(k_all, axis=0)
    std_k = jnp.std(k_all, axis=0)
    e_train = e_j[idx_train]
    return KerNNStats(
        mean_e=float(jnp.mean(e_train)),
        std_e=float(jnp.std(e_train)),
        min_r=np.asarray(min_r),
        mean_k=np.asarray(mean_k),
        std_k=np.asarray(std_k),
    )


def _batch_indices(n: int, batch_size: int, key: jax.Array, *, shuffle: bool):
    if shuffle:
        idx = jax.random.permutation(key, n)
    else:
        idx = jnp.arange(n)
    n_batches = n // batch_size
    if n_batches < 1:
        raise ValueError(f"need at least one full batch; n={n}, batch_size={batch_size}")
    idx = idx[: n_batches * batch_size]
    return idx.reshape(n_batches, batch_size)


def _mse(a, b):
    return jnp.mean((a - b) ** 2)


def _loss_on_batch(params, pos, e_ref, f_ref, stats, config, f_weight):
    e_pred, f_pred = energy_and_forces(params, pos, stats, config=config)
    eloss = _mse(e_pred, e_ref)
    floss = _mse(f_pred, f_ref)
    return eloss + f_weight * floss, (eloss, floss)


def train(args) -> Path:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data)
    if not all(k in data for k in ("R", "E", "F")):
        raise ValueError(f"{args.data} must contain keys R, E, F")
    positions = np.asarray(data["R"], dtype=np.float32)
    energies = np.asarray(data["E"], dtype=np.float32).reshape(-1)
    forces = np.asarray(data["F"], dtype=np.float32)
    ndata = positions.shape[0]
    if positions.shape[1] != 4:
        raise ValueError(f"KerNN v1 expects 4 atoms (ABCC); got R shape {positions.shape}")

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(ndata)
    ntrain = int(args.ntrain)
    nvalid = int(args.nvalid)
    if ntrain + nvalid > ndata:
        raise ValueError(
            f"ntrain+nvalid={ntrain + nvalid} exceeds dataset size {ndata}"
        )
    idx_train = idx[:ntrain]
    idx_valid = idx[ntrain : ntrain + nvalid]
    idx_test = idx[ntrain + nvalid :]

    config = KerNNConfig(
        n_input=6,
        n_hidden=int(args.n_hidden),
        n_out=1,
        kernel=str(args.kernel),
        distance_scheme="abcc",
    )
    stats = _compute_stats(positions, energies, idx_train, kernel=config.kernel)

    key = jax.random.key(int(args.seed))
    key, init_key = jax.random.split(key)
    params = init_params(init_key, config=config)
    ema_params = jax.tree.map(lambda x: x.copy(), params)

    optimizer = optax.adam(learning_rate=float(args.learning_rate), b1=0.9, b2=0.999, eps=1e-8)
    # Match Torch AMSGrad-ish behavior loosely; Optax adam is fine for this MLP
    opt_state = optimizer.init(params)

    pos_train = jnp.asarray(positions[idx_train])
    e_train = jnp.asarray(energies[idx_train])
    f_train = jnp.asarray(forces[idx_train])
    pos_valid = jnp.asarray(positions[idx_valid])
    e_valid = jnp.asarray(energies[idx_valid])
    f_valid = jnp.asarray(forces[idx_valid])

    f_weight = float(args.f_weight)
    batch_size = int(args.batch_size)
    ema_decay = float(args.ema_decay)

    @jax.jit
    def train_step(params, opt_state, ema_params, pos_b, e_b, f_b):
        def loss_fn(p):
            loss, (el, fl) = _loss_on_batch(p, pos_b, e_b, f_b, stats, config, f_weight)
            return loss, (el, fl)

        (loss, (el, fl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = jax.tree.map(
            lambda e, p: ema_decay * e + (1.0 - ema_decay) * p,
            ema_params,
            params,
        )
        return params, opt_state, ema_params, loss, el, fl

    @jax.jit
    def eval_step(params, pos_b, e_b, f_b):
        loss, (el, fl) = _loss_on_batch(params, pos_b, e_b, f_b, stats, config, f_weight)
        return loss, el, fl

    best_vloss = float("inf")
    stopper = 0
    best_path = workdir / "best.json"
    history: list[dict[str, Any]] = []

    split_path = workdir / "data_split.json"
    split_path.write_text(
        json.dumps(
            {
                "seed": int(args.seed),
                "ntrain": ntrain,
                "nvalid": nvalid,
                "ntest": int(len(idx_test)),
                "idx_train": idx_train.tolist(),
                "idx_valid": idx_valid.tolist(),
                "idx_test": idx_test.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"KerNN train: ntrain={ntrain} nvalid={nvalid} ntest={len(idx_test)}")
    print(f"  mean_e={stats.mean_e:.8f} std_e={stats.std_e:.8f}")
    print(f"  workdir={workdir}")

    for epoch in range(int(args.epochs)):
        t0 = time.time()
        key, shuf_key = jax.random.split(key)
        batches = _batch_indices(ntrain, batch_size, shuf_key, shuffle=True)
        train_losses = []
        for b in range(batches.shape[0]):
            bi = batches[b]
            params, opt_state, ema_params, loss, el, fl = train_step(
                params,
                opt_state,
                ema_params,
                pos_train[bi],
                e_train[bi],
                f_train[bi],
            )
            train_losses.append((float(loss), float(el), float(fl)))

        # Validation with EMA weights
        v_batches = _batch_indices(
            nvalid, batch_size, jax.random.key(0), shuffle=False
        )
        v_losses = []
        for b in range(v_batches.shape[0]):
            bi = v_batches[b]
            loss, el, fl = eval_step(
                ema_params, pos_valid[bi], e_valid[bi], f_valid[bi]
            )
            v_losses.append((float(loss), float(el), float(fl)))

        avg_t = np.mean([x[0] for x in train_losses])
        avg_v = np.mean([x[0] for x in v_losses])
        avg_te = np.mean([x[1] for x in train_losses])
        avg_ve = np.mean([x[1] for x in v_losses])
        avg_tf = np.mean([x[2] for x in train_losses])
        avg_vf = np.mean([x[2] for x in v_losses])
        dt = time.time() - t0
        print(
            f"epoch {epoch + 1:4d}  train {avg_t:.6e}  valid {avg_v:.6e}  "
            f"(E {avg_te:.3e}/{avg_ve:.3e}  F {avg_tf:.3e}/{avg_vf:.3e})  {dt:.1f}s"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss_train": avg_t,
                "loss_valid": avg_v,
                "eloss_train": avg_te,
                "eloss_valid": avg_ve,
                "floss_train": avg_tf,
                "floss_valid": avg_vf,
            }
        )

        if avg_v < best_vloss:
            best_vloss = avg_v
            stopper = 0
            save_checkpoint(
                best_path,
                params=ema_params,
                config=config,
                stats=stats,
                metadata={
                    "epoch": epoch + 1,
                    "loss_valid": avg_v,
                    "seed": int(args.seed),
                    "ntrain": ntrain,
                    "data": str(args.data),
                },
            )
            print(f"  saved {best_path}")
        else:
            stopper += 1
            if stopper >= int(args.patience):
                print(f"early stop: no improvement for {stopper} epochs")
                break

    (workdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    # Final (non-EMA) snapshot
    save_checkpoint(
        workdir / "last.json",
        params=params,
        config=config,
        stats=stats,
        metadata={"epoch": history[-1]["epoch"] if history else 0, "ema": False},
    )
    print(f"best validation loss={best_vloss:.6e} → {best_path}")
    return best_path


def main(args=None) -> Path | None:
    if args is None:
        args = get_args()
    return train(args)


if __name__ == "__main__":
    main()
    sys.exit(0)
