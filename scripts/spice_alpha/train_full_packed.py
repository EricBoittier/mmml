#!/usr/bin/env python
"""Train EFieldPhysNet on full SPICE-α with packed variable-size batches.

Usage:
  python scripts/spice_alpha/train_full_packed.py RAGGED_DIR OUT_DIR [options]

Writes OUT_DIR/{config.json, split.npz, e_ref.npy, history.jsonl,
params-last.json, params-best.json, ema-last.json}. Both the latest and the
best-validation weights are saved every epoch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ragged_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--subsets", nargs="*", default=None, help="ragged shard stems (default: all *.npz)")
    ap.add_argument("--max-train-frames", type=int, default=0, help="subsample train (smoke/benchmark)")
    ap.add_argument("--max-valid-frames", type=int, default=20000, help="validation subsample per epoch")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many train steps (benchmark)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--clip-norm", type=float, default=10.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restart", type=Path, default=None, help="params JSON to start from")
    # packing
    ap.add_argument("--max-molecules", type=int, default=256)
    ap.add_argument("--max-atoms", type=int, default=4096)
    ap.add_argument("--max-edges", type=int, default=131072)
    ap.add_argument("--coulomb-cutoff", type=float, default=None,
                    help="separate Coulomb pair list radius in Å ('inf' = all intra-frame pairs); "
                         "default reuses the message-passing edges")
    ap.add_argument("--max-coulomb-edges", type=int, default=0)
    # model
    ap.add_argument("--features", type=int, default=64)
    ap.add_argument("--max-degree", type=int, default=2)
    ap.add_argument("--num-iterations", type=int, default=3)
    ap.add_argument("--num-basis-functions", type=int, default=32)
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--gradient-checkpoint", action="store_true")
    ap.add_argument("--charge-activation", choices=("silu", "linear"), default="silu")
    ap.add_argument("--no-pseudotensors", dest="pseudotensors", action="store_false",
                    help="even-parity features only (e3x include_pseudotensors=False)")
    ap.add_argument("--parity-correct", action="store_true",
                    help="parity-correct field input and atomic-dipole head (see docs/spice-alpha.md)")
    ap.add_argument("--strict-pseudotensors", action="store_true",
                    help="pass include_pseudotensors to every e3x layer, not only MessagePass")
    ap.add_argument("--no-atomic-dipoles", dest="atomic_dipoles", action="store_false",
                    help="molecular dipole from atomic charges only (no per-atom dipole head)")
    # loss weights (0.5·squared error per component, as the padded trainer)
    ap.add_argument("--energy-weight", type=float, default=0.0)
    ap.add_argument("--forces-weight", type=float, default=1.0)
    ap.add_argument("--dipole-weight", type=float, default=10.0)
    ap.add_argument("--polar-weight", type=float, default=1.0)
    ap.add_argument("--charge-weight", type=float, default=1000.0)
    ap.add_argument("--log-every", type=int, default=200)
    return ap.parse_args()


def main():
    args = parse_args()
    import jax
    import jax.numpy as jnp
    import optax

    jax.config.update("jax_enable_x64", False)
    from mmml.data.spice_alpha_ragged import SUBSET_IDS, fit_atomic_energy_refs, load_ragged, split_ragged, take_frames
    from mmml.models.efield.packed import PackSpec, SubsetMetrics, iter_packed_batches, make_steps, prefetch
    from mmml.models.efield.training import EFieldPhysNet, load_params, save_params_json

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                                 for k, v in vars(args).items()}, indent=2))
    shards = sorted(args.ragged_dir.glob("*.npz"))
    if args.subsets:
        shards = [p for p in shards if p.stem in args.subsets]
    t0 = time.time()
    data = load_ragged(shards)
    print(f"loaded {len(shards)} shards: {len(data['N'])} frames, {len(data['Z'])} atoms "
          f"({time.time() - t0:.0f}s)", flush=True)

    split = split_ragged(data, seed=args.seed)
    np.savez(out / "split.npz", **split)
    rng = np.random.default_rng(args.seed)
    train_idx = split["train"]
    if args.max_train_frames:
        train_idx = rng.choice(train_idx, size=min(args.max_train_frames, len(train_idx)), replace=False)
    valid_idx = split["valid"]
    if args.max_valid_frames and len(valid_idx) > args.max_valid_frames:
        valid_idx = np.sort(rng.choice(valid_idx, size=args.max_valid_frames, replace=False))
    print({k: len(v) for k, v in split.items()}, f"train used {len(train_idx)}, valid used {len(valid_idx)}", flush=True)

    e_ref = None
    if args.energy_weight > 0:
        e_ref = fit_atomic_energy_refs(take_frames(data, train_idx))
        np.save(out / "e_ref.npy", e_ref)

    spec = PackSpec(args.max_molecules, args.max_atoms, args.max_edges, args.cutoff,
                    coulomb_cutoff=args.coulomb_cutoff, max_coulomb_edges=args.max_coulomb_edges)
    model = EFieldPhysNet(features=args.features, max_degree=args.max_degree,
                          num_iterations=args.num_iterations, num_basis_functions=args.num_basis_functions,
                          cutoff=args.cutoff, max_atomic_number=55, include_pseudotensors=args.pseudotensors,
                          field_scale=0.001, zbl=False, electrostatics_damping_sigma=4.0, packed=True,
                          charge_activation=args.charge_activation, atomic_dipoles=args.atomic_dipoles,
                          parity_correct_field=args.parity_correct, parity_correct_dipole=args.parity_correct,
                          strict_pseudotensors=args.strict_pseudotensors)
    M = spec.max_molecules
    first = {k: jnp.asarray(v) for k, v in next(iter_packed_batches(data, train_idx[:M], spec)).items()}
    params = model.init(jax.random.PRNGKey(args.seed), atomic_numbers=first["atomic_numbers"],
                        positions=first["positions"], Ef=first["electric_field"],
                        dst_idx_flat=first["dst_idx_flat"], src_idx_flat=first["src_idx_flat"],
                        batch_segments=first["batch_segments"], batch_size=M)
    params = {"params": params["params"]}
    if args.restart:
        params = {"params": load_params(args.restart)["params"]}
        print(f"restart from {args.restart}", flush=True)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"model params: {n_params}", flush=True)

    steps_per_epoch_est = max(1, int(np.ceil(np.sum(data["N"][train_idx]) / (0.8 * args.max_atoms))))
    total_steps = args.max_steps or steps_per_epoch_est * args.epochs
    schedule = optax.warmup_cosine_decay_schedule(0.0, args.lr, min(args.warmup_steps, total_steps // 10 + 1),
                                                  total_steps, end_value=args.lr * 0.01)
    optimizer = optax.chain(optax.clip_by_global_norm(args.clip_norm), optax.adam(schedule))
    opt_state = optimizer.init(params)
    ema = params
    weights = {"energy": args.energy_weight, "forces": args.forces_weight, "dipole": args.dipole_weight,
               "charge": args.charge_weight, "polar": args.polar_weight}
    train_step, eval_step = make_steps(model.apply, optimizer, M, weights, 0.001,
                                       args.gradient_checkpoint, args.ema_decay)
    names = {v: k for k, v in SUBSET_IDS.items()}
    best = np.inf
    step = 0
    hist = (out / "history.jsonl").open("a")
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train_idx)
        t_ep, t_log, n_frames, n_atoms = time.time(), time.time(), 0, 0
        for batch in prefetch(iter_packed_batches(data, order, spec, e_ref=e_ref)):
            bj = {k: jnp.asarray(v) for k, v in batch.items()}
            params, ema, opt_state, loss, terms, finite = train_step(params, ema, opt_state, bj)
            step += 1
            n_frames += int(batch["mol_mask"].sum())
            n_atoms += int((batch["atomic_numbers"] > 0).sum())
            if step % args.log_every == 0:
                loss = float(loss)
                dt = time.time() - t_log
                print(f"ep {epoch} step {step} loss {loss:.4g} finite {bool(finite)} "
                      f"{n_frames / (time.time() - t_ep):.0f} frames/s "
                      f"fill {n_atoms / (step * args.max_atoms) if epoch == 1 else 0:.2f} "
                      f"terms {{{', '.join(f'{k}: {float(v):.3g}' for k, v in terms.items())}}} "
                      f"[{dt:.0f}s]", flush=True)
                t_log = time.time()
            if args.max_steps and step >= args.max_steps:
                break
        train_s = time.time() - t_ep
        metrics, vloss, vn = SubsetMetrics(names), 0.0, 0
        for batch in prefetch(iter_packed_batches(data, valid_idx, spec, e_ref=e_ref)):
            bj = {k: jnp.asarray(v) for k, v in batch.items()}
            loss, terms, preds = eval_step(ema, bj)
            vloss += float(loss); vn += 1
            metrics.update(batch, jax.device_get(preds))
        vloss /= max(vn, 1)
        res = metrics.result()
        save_params_json(out / "params-last.json", params)
        save_params_json(out / "ema-last.json", ema)
        improved = vloss < best
        if improved:
            best = vloss
            save_params_json(out / "params-best.json", ema)
        rec = {"epoch": epoch, "step": step, "train_s": train_s, "frames_per_s": n_frames / max(train_s, 1e-9),
               "valid_loss": vloss, "improved": improved, "lr": float(schedule(step)), "valid": res}
        hist.write(json.dumps(rec) + "\n"); hist.flush()
        a = res.get("all", {})
        print(f"== epoch {epoch} train {train_s:.0f}s valid_loss {vloss:.4g} "
              f"dipole {a.get('dipole', np.nan):.4f} eÅ polar diag {a.get('polar_diag', np.nan):.4f} "
              f"offdiag {a.get('polar_offdiag', np.nan):.4f} eÅ²/V forces {a.get('forces', np.nan):.3f} eV/Å "
              f"{'*' if improved else ''}", flush=True)
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
