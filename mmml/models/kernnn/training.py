"""Train KerNN (JAX/Flax) on NPZ datasets with R, E, F.

Supports single NPZ + random split, or explicit train/valid(/test) NPZs.
Optional PhysNet teacher distillation via ``--teacher-checkpoint``.
"""

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
from mmml.models.kernnn.dihedrals import h2co_hcoh_dihedral
from mmml.models.kernnn.distances import (
    DISTANCE_FNS,
    n_atoms_for_scheme,
    n_features_for_scheme,
)
from mmml.models.kernnn.kernels import KERNEL_FNS
from mmml.models.kernnn.model import (
    KerNNConfig,
    KerNNStats,
    energy_and_forces,
)
from mmml.models.physnetjax.physnetjax.training.distill import blend_regression_loss
from mmml.utils.cli_args import exit_if_unknown_long_options

EV_TO_KCAL_MOL = 23.060541945

_TRAIN_DEFAULTS = {
    "data": None,
    "train_npz": None,
    "valid_npz": None,
    "test_npz": None,
    "workdir": "artifacts/kernnn",
    "ntrain": 3200,
    "nvalid": 400,
    "seed": 42,
    "n_hidden": 64,
    "batch_size": 64,
    "learning_rate": 0.005,
    "f_weight": 10.0,
    "epochs": 1000,
    "patience": 200,
    "ema_decay": 0.999,
    "kernel": "k33",
    "distance_scheme": "abcc",
    "architecture": "ffnet",
    "teacher_checkpoint": None,
    "distill_alpha": 1.0,
}


def build_parser() -> argparse.ArgumentParser:
    d = _TRAIN_DEFAULTS
    p = argparse.ArgumentParser(
        description="Train KerNN (kernel Softplus MLP) on NPZ (R, E, F)"
    )
    p.add_argument(
        "--data",
        type=str,
        default=d["data"],
        help="Single NPZ with R,E,F (random train/valid/test split)",
    )
    p.add_argument("--train-npz", type=str, default=d["train_npz"], help="Train split NPZ")
    p.add_argument("--valid-npz", type=str, default=d["valid_npz"], help="Valid split NPZ")
    p.add_argument("--test-npz", type=str, default=d["test_npz"], help="Optional test split NPZ")
    p.add_argument("--workdir", type=str, default=d["workdir"], help="Output directory")
    p.add_argument("--ntrain", type=int, default=d["ntrain"], help="Training size when using --data")
    p.add_argument("--nvalid", type=int, default=d["nvalid"], help="Validation size when using --data")
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
    p.add_argument(
        "--distance-scheme",
        type=str,
        default=d["distance_scheme"],
        choices=sorted(DISTANCE_FNS),
        help="Distance descriptor: abcc, abcc_sym, form (6 atoms), acem (9 atoms)",
    )
    p.add_argument(
        "--architecture",
        type=str,
        default=d["architecture"],
        choices=("ffnet", "dual"),
        help="ffnet (default) or dual (ABCC + dihedral only)",
    )
    p.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=d["teacher_checkpoint"],
        help="PhysNet checkpoint (JSON/Orbax) used as distillation teacher",
    )
    p.add_argument(
        "--distill-alpha",
        type=float,
        default=d["distill_alpha"],
        help="Blend GT vs teacher: loss = alpha*GT + (1-alpha)*teacher (1=pure GT)",
    )
    return p


def get_args(argv: list[str] | None = None):
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    exit_if_unknown_long_options(unknown, prog="mmml kernnn-train")
    return args


def _load_ef_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    # Accept PhysNet-style keys or plain R/E/F
    if "R" in data:
        positions = np.asarray(data["R"], dtype=np.float32)
    else:
        raise ValueError(f"{path} missing R")
    if "E" in data:
        energies = np.asarray(data["E"], dtype=np.float32).reshape(-1)
    elif "energy" in data:
        energies = np.asarray(data["energy"], dtype=np.float32).reshape(-1)
    else:
        raise ValueError(f"{path} missing E")
    if "F" in data:
        forces = np.asarray(data["F"], dtype=np.float32)
    elif "forces" in data:
        forces = np.asarray(data["forces"], dtype=np.float32)
    else:
        raise ValueError(f"{path} missing F")
    return positions, energies, forces


def _compute_stats(
    positions: np.ndarray,
    energies: np.ndarray,
    idx_train: np.ndarray,
    *,
    kernel: str = "k33",
    distance_scheme: str = "abcc",
    architecture: str = "ffnet",
) -> KerNNStats:
    pos_j = jnp.asarray(positions, dtype=jnp.float32)
    e_j = jnp.asarray(energies, dtype=jnp.float32)
    dist_fn = DISTANCE_FNS[distance_scheme]
    min_idx = int(jnp.argmin(e_j))
    min_r = dist_fn(pos_j[min_idx])
    r_all = dist_fn(pos_j)
    k_fn = KERNEL_FNS[kernel]
    k_all = k_fn(r_all, min_r, 1.0)
    mean_k = jnp.mean(k_all, axis=0)
    std_k = jnp.std(k_all, axis=0)
    # Avoid divide-by-zero on constant features
    std_k = jnp.where(std_k < 1e-12, 1.0, std_k)
    e_train = e_j[idx_train]
    mean_dih = 0.0
    std_dih = 1.0
    if architecture == "dual":
        if n_atoms_for_scheme(distance_scheme) != 4:
            raise ValueError("architecture=dual requires ABCC (4 atoms)")
        phi = h2co_hcoh_dihedral(pos_j)
        mean_dih = float(jnp.mean(phi))
        std_dih = float(jnp.std(phi))
        if std_dih < 1e-8:
            std_dih = 1.0
    return KerNNStats(
        mean_e=float(jnp.mean(e_train)),
        std_e=float(max(float(jnp.std(e_train)), 1e-8)),
        min_r=np.asarray(min_r),
        mean_k=np.asarray(mean_k),
        std_k=np.asarray(std_k),
        mean_dihedral=mean_dih,
        std_dihedral=std_dih,
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


def _load_physnet_teacher_ef_fn(checkpoint: str | Path, natoms: int):
    """Return ``(R_batch) -> (E, F)`` for a PhysNet/Spooky checkpoint."""
    import e3x

    from mmml.models.kernnn import is_kernnn_checkpoint
    from mmml.umbrella.checkpoint import load_params_and_model

    if is_kernnn_checkpoint(checkpoint):
        raise ValueError(
            "--teacher-checkpoint must be a PhysNet checkpoint, not KerNN"
        )

    params, model = load_params_and_model(
        checkpoint, natoms=natoms, model="physnet"
    )
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(natoms)
    dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
    src_idx = jnp.asarray(src_idx, dtype=jnp.int32)

    def one(pos):
        # Prefer compute_forces when available (PhysNet family).
        try:
            out = model.apply(
                params,
                atomic_numbers=None,  # filled below
                positions=pos,
                dst_idx=dst_idx,
                src_idx=src_idx,
                compute_forces=True,
            )
        except TypeError:
            out = None
        return out

    # Need Z — infer later from first batch via closure set by caller
    state: dict[str, Any] = {"Z": None}

    def set_z(z):
        state["Z"] = jnp.asarray(z, dtype=jnp.int32).reshape(natoms)

    def predict_batch(pos_batch):
        z = state["Z"]
        if z is None:
            raise RuntimeError("teacher Z not set; call set_z first")

        def one_struct(pos):
            try:
                out = model.apply(
                    params,
                    atomic_numbers=z,
                    positions=pos,
                    dst_idx=dst_idx,
                    src_idx=src_idx,
                    compute_forces=True,
                )
            except TypeError:
                out = model.apply(
                    params,
                    atomic_numbers=z,
                    positions=pos,
                    dst_idx=dst_idx,
                    src_idx=src_idx,
                )
            e = jnp.asarray(out["energy"]).reshape(())
            if "forces" in out:
                f = jnp.asarray(out["forces"]).reshape(natoms, 3)
            else:
                # Fallback FD-free: value_and_grad on energy
                def _e(p):
                    o = model.apply(
                        params,
                        atomic_numbers=z,
                        positions=p,
                        dst_idx=dst_idx,
                        src_idx=src_idx,
                    )
                    return jnp.asarray(o["energy"]).reshape(())

                _, neg_f = jax.value_and_grad(_e)(pos)
                f = -neg_f
            return e, f

        return jax.vmap(one_struct)(pos_batch)

    return set_z, jax.jit(predict_batch)


def train(args) -> Path:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    scheme = str(args.distance_scheme)
    n_atoms = n_atoms_for_scheme(scheme)
    if str(args.architecture) == "dual" and n_atoms != 4:
        raise ValueError("architecture=dual is only supported for ABCC (4 atoms)")

    distill_alpha = float(args.distill_alpha)
    if not (0.0 <= distill_alpha <= 1.0):
        raise ValueError("--distill-alpha must be in [0, 1]")
    use_teacher = bool(args.teacher_checkpoint)
    if use_teacher and distill_alpha >= 1.0:
        print(
            "warning: --teacher-checkpoint set but --distill-alpha=1.0 "
            "(teacher loss weight is 0); set e.g. --distill-alpha 0.5"
        )

    # ---- data ----
    if args.train_npz and args.valid_npz:
        pos_train, e_train, f_train = _load_ef_npz(args.train_npz)
        pos_valid, e_valid, f_valid = _load_ef_npz(args.valid_npz)
        if args.test_npz:
            pos_test, e_test, _ = _load_ef_npz(args.test_npz)
        else:
            pos_test = e_test = None
        # Stats over train only, but min_r/k over train+valid (+test) like legacy
        pos_all = np.concatenate(
            [pos_train, pos_valid]
            + ([pos_test] if pos_test is not None else []),
            axis=0,
        )
        e_all = np.concatenate(
            [e_train, e_valid] + ([e_test] if e_test is not None else []),
            axis=0,
        )
        idx_train_stats = np.arange(len(e_train))
        ntrain, nvalid = len(e_train), len(e_valid)
        idx_train = np.arange(ntrain)
        idx_valid = np.arange(nvalid)
        idx_test = np.arange(len(e_test)) if e_test is not None else np.array([], dtype=int)
        data_desc = {
            "train_npz": str(args.train_npz),
            "valid_npz": str(args.valid_npz),
            "test_npz": str(args.test_npz) if args.test_npz else None,
        }
    elif args.data:
        positions, energies, forces = _load_ef_npz(args.data)
        ndata = positions.shape[0]
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
        pos_train = positions[idx_train]
        e_train = energies[idx_train]
        f_train = forces[idx_train]
        pos_valid = positions[idx_valid]
        e_valid = energies[idx_valid]
        f_valid = forces[idx_valid]
        pos_test = positions[idx_test] if len(idx_test) else None
        e_test = energies[idx_test] if len(idx_test) else None
        pos_all, e_all = positions, energies
        idx_train_stats = idx_train
        data_desc = {"data": str(args.data)}
    else:
        raise ValueError("provide --data or both --train-npz and --valid-npz")

    if pos_train.shape[1] != n_atoms:
        raise ValueError(
            f"distance_scheme={scheme} expects {n_atoms} atoms; "
            f"got R shape {pos_train.shape}"
        )

    config = KerNNConfig(
        n_input=n_features_for_scheme(scheme),
        n_hidden=int(args.n_hidden),
        n_out=1,
        n_atoms=n_atoms,
        kernel=str(args.kernel),
        distance_scheme=scheme,
        architecture=str(args.architecture),
    )
    stats = _compute_stats(
        pos_all,
        e_all,
        idx_train_stats,
        kernel=config.kernel,
        distance_scheme=config.distance_scheme,
        architecture=config.architecture,
    )

    # Optional Z for teacher
    z_atoms = None
    for path in (
        args.train_npz,
        args.valid_npz,
        args.data,
        args.test_npz,
    ):
        if not path:
            continue
        raw = np.load(path)
        if "Z" in raw:
            z = np.asarray(raw["Z"])
            z_atoms = z[0] if z.ndim == 2 else z
            break

    teacher_predict = None
    if use_teacher:
        set_z, teacher_predict = _load_physnet_teacher_ef_fn(
            args.teacher_checkpoint, n_atoms
        )
        if z_atoms is None:
            raise ValueError(
                "PhysNet teacher needs atomic numbers Z in the NPZ "
                "(or provide a split that contains Z)"
            )
        set_z(z_atoms[:n_atoms])

    key = jax.random.key(int(args.seed))
    key, init_key = jax.random.split(key)
    params = init_params(init_key, config=config)
    ema_params = jax.tree.map(lambda x: x.copy(), params)

    optimizer = optax.adam(learning_rate=float(args.learning_rate))
    opt_state = optimizer.init(params)

    pos_train_j = jnp.asarray(pos_train)
    e_train_j = jnp.asarray(e_train)
    f_train_j = jnp.asarray(f_train)
    pos_valid_j = jnp.asarray(pos_valid)
    e_valid_j = jnp.asarray(e_valid)
    f_valid_j = jnp.asarray(f_valid)

    f_weight = float(args.f_weight)
    batch_size = int(args.batch_size)
    ema_decay = float(args.ema_decay)

    def _loss_parts(params, pos, e_ref, f_ref, e_tch=None, f_tch=None):
        e_pred, f_pred = energy_and_forces(params, pos, stats, config=config)
        eloss_gt = _mse(e_pred, e_ref)
        floss_gt = _mse(f_pred, f_ref)
        if e_tch is not None and distill_alpha < 1.0:
            eloss_t = _mse(e_pred, e_tch)
            floss_t = _mse(f_pred, f_tch)
            eloss = blend_regression_loss(eloss_gt, eloss_t, distill_alpha)
            floss = blend_regression_loss(floss_gt, floss_t, distill_alpha)
        else:
            eloss, floss = eloss_gt, floss_gt
        return eloss + f_weight * floss, (eloss, floss)

    @jax.jit
    def train_step(params, opt_state, ema_params, pos_b, e_b, f_b, e_t, f_t):
        def loss_fn(p):
            loss, (el, fl) = _loss_parts(p, pos_b, e_b, f_b, e_t, f_t)
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
    def eval_step(params, pos_b, e_b, f_b, e_t, f_t):
        loss, (el, fl) = _loss_parts(params, pos_b, e_b, f_b, e_t, f_t)
        return loss, el, fl

    def _teacher_batch(pos_b):
        if teacher_predict is None or distill_alpha >= 1.0:
            return None, None
        return teacher_predict(pos_b)

    best_vloss = float("inf")
    stopper = 0
    best_path = workdir / "best.json"
    history: list[dict[str, Any]] = []

    split_path = workdir / "data_split.json"
    split_path.write_text(
        json.dumps(
            {
                "seed": int(args.seed),
                "ntrain": int(ntrain),
                "nvalid": int(nvalid),
                "ntest": int(len(idx_test)),
                "distance_scheme": scheme,
                "n_atoms": n_atoms,
                "teacher_checkpoint": args.teacher_checkpoint,
                "distill_alpha": distill_alpha,
                **data_desc,
                "idx_train": np.asarray(idx_train).tolist(),
                "idx_valid": np.asarray(idx_valid).tolist(),
                "idx_test": np.asarray(idx_test).tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"KerNN train: scheme={scheme} n_atoms={n_atoms} "
        f"ntrain={ntrain} nvalid={nvalid} n_input={config.n_input}"
    )
    if use_teacher:
        print(
            f"  teacher={args.teacher_checkpoint}  distill_alpha={distill_alpha}"
        )
    print(f"  mean_e={stats.mean_e:.8f} std_e={stats.std_e:.8f}")
    print(f"  workdir={workdir}")

    for epoch in range(int(args.epochs)):
        t0 = time.time()
        key, shuf_key = jax.random.split(key)
        batches = _batch_indices(ntrain, batch_size, shuf_key, shuffle=True)
        train_losses = []
        for b in range(batches.shape[0]):
            bi = batches[b]
            pos_b = pos_train_j[bi]
            e_t, f_t = _teacher_batch(pos_b)
            # Placeholders when no teacher (ignored by loss)
            if e_t is None:
                e_t = e_train_j[bi]
                f_t = f_train_j[bi]
            params, opt_state, ema_params, loss, el, fl = train_step(
                params,
                opt_state,
                ema_params,
                pos_b,
                e_train_j[bi],
                f_train_j[bi],
                e_t,
                f_t,
            )
            train_losses.append((float(loss), float(el), float(fl)))

        # Validation (allow a single partial batch when nvalid < batch_size)
        if nvalid <= batch_size:
            v_batches = jnp.arange(nvalid)[None, :]
        else:
            v_batches = _batch_indices(
                nvalid, batch_size, jax.random.key(0), shuffle=False
            )
        v_losses = []
        for b in range(v_batches.shape[0]):
            bi = v_batches[b]
            pos_b = pos_valid_j[bi]
            e_t, f_t = _teacher_batch(pos_b)
            if e_t is None:
                e_t = e_valid_j[bi]
                f_t = f_valid_j[bi]
            loss, el, fl = eval_step(
                ema_params,
                pos_b,
                e_valid_j[bi],
                f_valid_j[bi],
                e_t,
                f_t,
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
                    "ntrain": int(ntrain),
                    "distance_scheme": scheme,
                    "teacher_checkpoint": args.teacher_checkpoint,
                    "distill_alpha": distill_alpha,
                    **data_desc,
                },
            )
            print(f"  saved {best_path}")
        else:
            stopper += 1
            if stopper >= int(args.patience):
                print(f"early stop: no improvement for {stopper} epochs")
                break

    (workdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
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
