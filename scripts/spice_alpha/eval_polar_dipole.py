#!/usr/bin/env python
"""Dipole and polarizability test errors in the MACE-MDP (Fig. 2) convention.

Reports RMSE of dipole components (e·Å) and of diagonal / off-diagonal
polarizability elements, in e·Å²/V (MACE-MDP units) and Bohr³. Off-diagonal
uses the 6 off-diagonal entries of the 3x3 tensor.

Usage (GPU node):
  python scripts/spice_alpha/eval_polar_dipole.py CKPT_DIR DATA_NPZ [--params params-best.json] [--batch-size 4]
"""

import argparse
import functools
import json
from pathlib import Path

import numpy as np

BOHR3_PER_EA2V = 97.17  # 1 e·Å²/V = 4πε0-scaled 97.17 Bohr³


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("data_npz", type=Path)
    ap.add_argument("--params", default="params-best.json")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--save-npz", type=Path, default=None, help="write predictions and targets")
    args = ap.parse_args()

    import e3x
    import jax
    import jax.numpy as jnp

    from karml.models.efield.model_functions import predicted_polarizability_bohr3
    from karml.models.efield.training import (
        MessagePassingModel,
        load_ef_npz,
        load_params,
        prepare_batches,
    )

    cfg = json.loads((args.ckpt_dir / "config.json").read_text())
    field_scale = cfg["model"]["field_scale"]
    model = MessagePassingModel(**cfg["model"])
    params = load_params(args.ckpt_dir / args.params)

    data = load_ef_npz(args.data_npz)
    B = args.batch_size
    n_frames = int(np.asarray(data["electric_field"]).shape[0])
    n_atoms = int(data["positions"].shape[1])
    dst, src = e3x.ops.sparse_pairwise_indices(n_atoms)
    offsets = jnp.arange(B, dtype=jnp.int32) * n_atoms
    dst_flat = (jnp.asarray(dst, jnp.int32)[None, :] + offsets[:, None]).reshape(-1)
    src_flat = (jnp.asarray(src, jnp.int32)[None, :] + offsets[:, None]).reshape(-1)
    seg = jnp.repeat(jnp.arange(B, dtype=jnp.int32), n_atoms)
    batches = prepare_batches(jax.random.PRNGKey(0), data, B, dst_idx_flat=dst_flat,
                              src_idx_flat=src_flat, batch_segments=seg, shuffle=False)

    # Training params carry an 'intermediates' collection; rebuild it the same way
    # evaluate.py does so sow() tracing matches training.
    b0 = batches[0]
    _, state = model.apply(params, atomic_numbers=b0["atomic_numbers"], positions=b0["positions"],
                           Ef=b0["electric_field"], dst_idx_flat=b0["dst_idx_flat"],
                           src_idx_flat=b0["src_idx_flat"], batch_segments=b0["batch_segments"],
                           batch_size=B, mutable=["intermediates"])
    params = {**params, "intermediates": state["intermediates"]}

    @functools.partial(jax.jit)
    def predict(batch):
        (_, dipole), _ = model.apply(params, atomic_numbers=batch["atomic_numbers"],
                                     positions=batch["positions"], Ef=batch["electric_field"],
                                     dst_idx_flat=batch["dst_idx_flat"], src_idx_flat=batch["src_idx_flat"],
                                     batch_segments=batch["batch_segments"], batch_size=B,
                                     mutable=["intermediates"])
        polar = predicted_polarizability_bohr3(
            model.apply, params,
            batch["atomic_numbers"].reshape(B, n_atoms), batch["positions"].reshape(B, n_atoms, 3),
            batch["dst_idx_flat"], batch["src_idx_flat"], batch["batch_segments"], B,
            field_scale=field_scale)
        return dipole, polar

    pd, pa, td, ta = [], [], [], []
    for batch in batches:
        dipole, polar = predict(batch)
        pd.append(np.asarray(dipole)); pa.append(np.asarray(polar))
        td.append(np.asarray(batch["dipoles"])); ta.append(np.asarray(batch["polar"]))
    pd, pa, td, ta = (np.concatenate(x) for x in (pd, pa, td, ta))

    diag = np.eye(3, dtype=bool)
    err = pa - ta
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    res = {
        "n_frames_evaluated": int(len(pa)), "n_frames_total": n_frames,
        "dipole_rmse_eA": rmse(pd - td), "dipole_mae_eA": float(np.mean(np.abs(pd - td))),
        "polar_diag_rmse_bohr3": rmse(err[:, diag]), "polar_offdiag_rmse_bohr3": rmse(err[:, ~diag]),
        "polar_diag_mae_bohr3": float(np.mean(np.abs(err[:, diag]))),
        "polar_offdiag_mae_bohr3": float(np.mean(np.abs(err[:, ~diag]))),
    }
    for k in [k for k in res if k.endswith("_bohr3")]:
        res[k.replace("_bohr3", "_eA2V")] = res[k] / BOHR3_PER_EA2V
    print(json.dumps(res, indent=2))
    if args.save_npz:
        np.savez(args.save_npz, dipole_pred=pd, dipole_ref=td, polar_pred_bohr3=pa, polar_ref_bohr3=ta)


if __name__ == "__main__":
    main()
