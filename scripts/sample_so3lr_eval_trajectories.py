#!/usr/bin/env python3
"""Sample structures from SO3LR/Spooky extxyz test sets as true-vs-predicted ASE trajectories.

For each --extxyz file, samples --num-samples random structures and writes
TWO .traj files per dataset: one with the reference (true) energy/forces
attached, one with the model's predicted energy/forces attached for the SAME
structures in the SAME order -- so they can be diffed frame by frame
(`ase gui foo_true.traj foo_pred.traj`, or loaded in a notebook).

Loads the checkpoint the same way evaluate_so3lr_spooky_extxyz.py does
(restore_checkpoint + create_model_from_config from that script) rather than
mmml.models.spookynet_calc.SpookyNetCalculator -- that calculator's own
loader (mmml.utils.model_checkpoint.load_model_checkpoint) does not handle
this checkpoint family's OCDBT-format orbax layout, while
evaluate_so3lr_spooky_extxyz.py's raw ocp.PyTreeCheckpointer().restore(...)
does (confirmed working against this checkpoint on the GPU node).

One model instance is built padded to the largest sampled structure; smaller
structures are padded up to that same size with atom_mask, matching how
evaluate_so3lr_spooky_extxyz.py sizes its model per file.

Usage:
    python scripts/sample_so3lr_eval_trajectories.py \\
        --checkpoint /mmhome/boittier/home/mmml/artifacts/spooky_so3lr_muon3/epoch-0010 \\
        --extxyz ~/data/so3lr_test/ \\
        --num-samples 5 \\
        --out-dir eval_out/sample_trajectories
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import iread
from ase.io.trajectory import Trajectory

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _true_energy_forces(atoms, energy_key: str, forces_key: str) -> tuple[float, np.ndarray]:
    if energy_key in atoms.info:
        energy = float(np.asarray(atoms.info[energy_key]).reshape(-1)[0])
    elif atoms.calc is not None and energy_key in getattr(atoms.calc, "results", {}):
        energy = float(np.asarray(atoms.calc.results[energy_key]).reshape(-1)[0])
    else:
        energy = float(atoms.get_potential_energy())
    if forces_key in atoms.arrays:
        forces = np.asarray(atoms.arrays[forces_key], dtype=np.float64)
    elif atoms.calc is not None and forces_key in getattr(atoms.calc, "results", {}):
        forces = np.asarray(atoms.calc.results[forces_key], dtype=np.float64)
    else:
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    return energy, forces


def _resolve_extxyz_files(extxyz_arg: Path, exclude_substring: str) -> list[Path]:
    if extxyz_arg.is_dir():
        files = sorted(extxyz_arg.glob("*.extxyz"))
    else:
        files = [extxyz_arg]
    if exclude_substring:
        files = [f for f in files if exclude_substring not in f.name]
    if not files:
        raise FileNotFoundError(f"No .extxyz files found at {extxyz_arg} (after exclusions)")
    return files


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Orbax epoch-N checkpoint dir")
    p.add_argument("--extxyz", required=True, type=Path, help="A .extxyz file or a directory of them")
    p.add_argument("--num-samples", type=int, default=5, help="Structures sampled per dataset (default: 5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("eval_out/sample_trajectories"))
    p.add_argument(
        "--exclude-substring",
        default="",
        help="Skip .extxyz files whose name contains this substring (default: none).",
    )
    p.add_argument("--energy-key", default="energy")
    p.add_argument("--forces-key", default="forces")
    p.add_argument("--charge-key", default="charge")
    args = p.parse_args()

    import e3x
    import jax
    import jax.numpy as jnp

    import evaluate_so3lr_spooky_extxyz as ev
    from mmml.utils.model_checkpoint import infer_trainable_zbl_config

    checkpoint_path = Path(args.checkpoint).resolve()
    params, config = ev.restore_checkpoint(checkpoint_path)
    config = infer_trainable_zbl_config(config, params)

    files = _resolve_extxyz_files(args.extxyz, args.exclude_substring)
    print(f"Sampling {args.num_samples} structure(s) from {len(files)} dataset(s): "
          f"{', '.join(f.name for f in files)}")

    rng = np.random.default_rng(args.seed)
    picks_by_file: dict[Path, tuple[list, list[int]]] = {}
    max_atoms = 0
    for f in files:
        atoms_list = list(iread(f, index=":"))
        n_pick = min(args.num_samples, len(atoms_list))
        picks = sorted(rng.choice(len(atoms_list), size=n_pick, replace=False).tolist())
        picks_by_file[f] = (atoms_list, picks)
        max_atoms = max(max_atoms, max(len(atoms_list[i]) for i in picks))

    print(f"Building model padded to max_atoms={max_atoms}")
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

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for extxyz_file, (atoms_list, picks) in picks_by_file.items():
        true_path = args.out_dir / f"{extxyz_file.stem}_true.traj"
        pred_path = args.out_dir / f"{extxyz_file.stem}_pred.traj"
        true_traj = Trajectory(str(true_path), "w")
        pred_traj = Trajectory(str(pred_path), "w")

        print(f"\n--- {extxyz_file.name}: structures {picks} ---")
        print(f"{'idx':>6} {'n_atoms':>8} {'E_true':>12} {'E_pred':>12} {'dE':>10} {'|F_true|max':>12} {'|F_pred|max':>12}")

        for idx in picks:
            atoms = atoms_list[idx]
            n_real = len(atoms)
            e_true, f_true = _true_energy_forces(atoms, args.energy_key, args.forces_key)

            true_atoms = atoms.copy()
            true_atoms.calc = SinglePointCalculator(true_atoms, energy=e_true, forces=f_true)
            true_traj.write(true_atoms)

            z = np.zeros(max_atoms, dtype=np.int32)
            z[:n_real] = atoms.get_atomic_numbers()
            pos = np.zeros((max_atoms, 3), dtype=np.float32)
            pos[:n_real] = atoms.get_positions()
            pad = max_atoms - n_real
            if pad:
                far = 1.0e4 + 100.0 * np.arange(pad, dtype=np.float32)
                pos[n_real:] = np.stack(
                    [far, np.zeros(pad, dtype=np.float32), np.zeros(pad, dtype=np.float32)], axis=1
                )
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
            e_pred = float(np.asarray(out["energy"]).reshape(-1)[0])
            f_pred = np.asarray(out["forces"]).reshape(max_atoms, 3)[:n_real]

            pred_atoms = atoms.copy()
            pred_atoms.calc = SinglePointCalculator(pred_atoms, energy=e_pred, forces=f_pred)
            pred_traj.write(pred_atoms)

            print(
                f"{idx:>6} {n_real:>8} {e_true:>12.4f} {e_pred:>12.4f} "
                f"{e_pred - e_true:>10.4f} {np.abs(f_true).max():>12.4f} {np.abs(f_pred).max():>12.4f}"
            )

        true_traj.close()
        pred_traj.close()
        print(f"wrote {true_path}")
        print(f"wrote {pred_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
