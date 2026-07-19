#!/usr/bin/env python3
"""Measure (not guess) how SO3LR/Spooky error and per-term energies scale with atom count.

Samples structures across one or more .extxyz test files, runs the actual
trained model, and for each structure records:

  - E_true, E_pred, dE, dE/N (energy error, absolute and per-atom)
  - force RMSE (per atom-component)
  - the model's own internal energy-term decomposition: electrostatics,
    ZBL repulsion, CGenFF vdW (each is exposed separately in
    SpookyPhysNet.__call__'s output dict -- see
    mmml/models/physnetjax/physnetjax/models/spooky_model.py:1270-1281 --
    but the evaluation script's eval_fn only ever reads "energy"/"forces",
    so these were never actually inspected before).

Writes a CSV (one row per sampled structure) and a 2x2 PNG:
  top-left:  |dE|/N_atoms vs N_atoms (energy error density)
  top-right: force RMSE vs N_atoms
  bottom-left: |electrostatics|/N_atoms vs N_atoms
  bottom-right: |repulsion (ZBL)|/N_atoms vs N_atoms
so which term (if any) actually blows up with system size is visible
directly from data, not inferred from architecture flags.

One model instance is built (via SpookyPhysNet's actual training config, not
a separately-constructed calculator) padded to the largest sampled structure,
and every smaller structure is padded up to that same size with atom_mask --
this mirrors exactly how evaluate_so3lr_spooky_extxyz.py sizes its model per
file, just extended across files/sizes in one pass.

Usage:
    python scripts/decompose_so3lr_terms_vs_natoms.py \\
        --checkpoint /mmhome/boittier/home/mmml/artifacts/spooky_so3lr_muon3/epoch-0010 \\
        --extxyz ~/data/so3lr_test/ \\
        --max-per-dataset 30 \\
        --out-csv eval_out/term_decomposition.csv \\
        --out-plot eval_out/error_and_terms_vs_natoms.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (_REPO_ROOT, _SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _resolve_extxyz_files(extxyz_arg: Path) -> list[Path]:
    if extxyz_arg.is_dir():
        files = sorted(extxyz_arg.glob("*.extxyz"))
    else:
        files = [extxyz_arg]
    if not files:
        raise FileNotFoundError(f"No .extxyz files found at {extxyz_arg}")
    return files


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Orbax epoch-N checkpoint dir")
    p.add_argument("--extxyz", required=True, type=Path, help=".extxyz file or directory")
    p.add_argument("--max-per-dataset", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--energy-key", default="energy")
    p.add_argument("--forces-key", default="forces")
    p.add_argument("--charge-key", default="charge")
    p.add_argument("--out-csv", type=Path, default=Path("eval_out/term_decomposition.csv"))
    p.add_argument("--out-plot", type=Path, default=Path("eval_out/error_and_terms_vs_natoms.png"))
    args = p.parse_args()

    import e3x
    import jax
    import jax.numpy as jnp
    from ase.io import iread

    import evaluate_so3lr_spooky_extxyz as ev
    from mmml.utils.model_checkpoint import infer_trainable_zbl_config

    checkpoint_path = Path(args.checkpoint).resolve()
    params, config = ev.restore_checkpoint(checkpoint_path)
    config = infer_trainable_zbl_config(config, params)

    files = _resolve_extxyz_files(args.extxyz)
    print(f"Sampling up to {args.max_per_dataset} structure(s) each from {len(files)} dataset(s): "
          f"{', '.join(f.name for f in files)}")

    samples: list[tuple[str, object]] = []
    rng = np.random.default_rng(args.seed)
    for f in files:
        atoms_list = list(iread(f, index=":"))
        n_pick = min(args.max_per_dataset, len(atoms_list))
        picks = sorted(rng.choice(len(atoms_list), size=n_pick, replace=False).tolist())
        for i in picks:
            samples.append((f.stem, atoms_list[i]))
    if not samples:
        raise ValueError("No structures sampled")

    max_atoms = max(len(a) for _, a in samples)
    print(f"Total sampled structures: {len(samples)}; building model padded to max_atoms={max_atoms}")

    model = ev.create_model_from_config(config, max_atoms=max_atoms)

    dst_idx_np, src_idx_np = e3x.ops.sparse_pairwise_indices(max_atoms)
    dst_idx = jnp.asarray(dst_idx_np)
    src_idx = jnp.asarray(src_idx_np)
    batch_segments = jnp.zeros((max_atoms,), dtype=jnp.int32)

    def _apply(z, pos, q, s, atom_mask, batch_mask):
        return model.apply(
            params,
            atomic_numbers=z,
            charges=q,
            spins=s,
            positions=pos,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=1,
            batch_mask=batch_mask,
            atom_mask=atom_mask,
            compute_forces=True,
        )

    apply_fn = jax.jit(_apply)

    rows: list[dict] = []
    for dataset, atoms in samples:
        n_real = len(atoms)
        z = np.zeros(max_atoms, dtype=np.int32)
        z[:n_real] = atoms.get_atomic_numbers()
        pos = np.zeros((max_atoms, 3), dtype=np.float32)
        pos[:n_real] = atoms.get_positions()
        pad = max_atoms - n_real
        if pad:
            far = 1.0e4 + 100.0 * np.arange(pad, dtype=np.float32)
            pos[n_real:] = np.stack([far, np.zeros(pad, dtype=np.float32), np.zeros(pad, dtype=np.float32)], axis=1)

        atom_mask = (z > 0).astype(np.float32)
        valid_pairs = (atom_mask[dst_idx_np] > 0) & (atom_mask[src_idx_np] > 0)
        batch_mask = valid_pairs.astype(np.float32)

        charge = float(atoms.info.get(args.charge_key, 0.0))
        spin = ev._infer_spin_multiplicity(atoms, charge)
        q = jnp.full((max_atoms, 1), charge, dtype=jnp.float32)
        s = jnp.full((max_atoms, 1), spin, dtype=jnp.float32)

        out = apply_fn(
            jnp.asarray(z), jnp.asarray(pos), q, s, jnp.asarray(atom_mask), jnp.asarray(batch_mask)
        )

        def _batch_scalar(key):
            # "energy", "electrostatics", "cgenff_vdw" are already segment_sum'd
            # to one value per structure (SpookyPhysNet.energy return tuple:
            # (energy, atomic_charges, batch_electrostatics, batch_vdw,
            # repulsion, x) -- note "repulsion" is the ONE exception, see below).
            val = out.get(key)
            if val is None:
                return 0.0
            return float(np.asarray(val).reshape(-1)[0])

        def _sum_per_atom(key):
            # "repulsion" is returned PER-ATOM, not batch-summed (confirmed by
            # inspecting SpookyPhysNet._calculate_repulsion's docstring "per
            # atom" and the raw output shape (max_atoms,1,1,1) vs
            # electrostatics' (1,1,1,1) -- sum over real atoms only.
            val = out.get(key)
            if val is None:
                return 0.0
            arr = np.asarray(val).reshape(max_atoms, -1)[:n_real]
            return float(np.sum(arr))

        e_pred = _batch_scalar("energy")
        elec = _batch_scalar("electrostatics")
        rep = _sum_per_atom("repulsion")
        cgenff = _batch_scalar("cgenff_vdw")
        atomic_e = e_pred - elec - rep - cgenff

        f_pred = np.asarray(out["forces"]).reshape(max_atoms, 3)[:n_real]

        e_true = ev._get_energy(atoms, args.energy_key, 0)
        f_true = ev._get_forces(atoms, args.forces_key, 0, default=np.zeros((n_real, 3)))
        f_rmse = float(np.sqrt(np.mean((f_pred - f_true) ** 2)))

        rows.append(
            {
                "dataset": dataset,
                "n_atoms": n_real,
                "E_true": e_true,
                "E_pred": e_pred,
                "dE": e_pred - e_true,
                "dE_per_atom": (e_pred - e_true) / n_real,
                "F_rmse": f_rmse,
                "electrostatics": elec,
                "repulsion_zbl": rep,
                "cgenff_vdw": cgenff,
                "atomic_energy": atomic_e,
                "electrostatics_per_atom": elec / n_real,
                "repulsion_per_atom": rep / n_real,
            }
        )
        print(
            f"{dataset:<28} n={n_real:>4} dE/N={rows[-1]['dE_per_atom']:>10.4f} "
            f"F_rmse={f_rmse:>8.4f} elec/N={rows[-1]['electrostatics_per_atom']:>10.4f} "
            f"rep/N={rows[-1]['repulsion_per_atom']:>10.4f}"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out_csv}")

    _make_plot(rows, args.out_plot)
    print(f"wrote plot -> {args.out_plot}")
    return 0


def _make_plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in rows})
    cmap = plt.get_cmap("tab10")
    colors = {d: cmap(i % 10) for i, d in enumerate(datasets)}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panels = [
        (axes[0, 0], "dE_per_atom", "|dE| / N_atoms (eV)", True),
        (axes[0, 1], "F_rmse", "Force RMSE (eV/A)", True),
        (axes[1, 0], "electrostatics_per_atom", "|electrostatics| / N_atoms (eV)", True),
        (axes[1, 1], "repulsion_per_atom", "|repulsion (ZBL)| / N_atoms (eV)", True),
    ]
    for ax, key, ylabel, abs_val in panels:
        for d in datasets:
            xs = [r["n_atoms"] for r in rows if r["dataset"] == d]
            ys = [abs(r[key]) if abs_val else r[key] for r in rows if r["dataset"] == d]
            ax.scatter(xs, ys, s=14, alpha=0.7, color=colors[d], label=d)
        ax.set_xlabel("N atoms")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
    axes[0, 1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("SO3LR/Spooky: error and energy-term magnitude vs. system size")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
