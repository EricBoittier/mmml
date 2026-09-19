"""Many-seed periodic PET dataset: random packing → FIRE intermediates → NVT.

CHARMM-free. One model load, then each seed builds its own randomly packed
box, keeps FIRE intermediates and NVT frames (labelled E/F, cell) in
``<out-dir>/seed_<k>/traj.extxyz``. Split seeds across GPUs with
``--seeds 0-15`` / ``--seeds 16-31`` and ``CUDA_VISIBLE_DEVICES``.

Example::

    MMML_METATOMIC_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 \\
      mmml pet-box-dataset --checkpoint pet-mad-xs-v1.5.0.pt \\
      --seeds 0-15 --out-dir runs/etoh_box_dataset
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path


def parse_seeds(text: str) -> list[int]:
    """``"0-3,7,10-11"`` → ``[0, 1, 2, 3, 7, 10, 11]``."""
    seeds: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError(f"no seeds in {text!r}")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mmml pet-box-dataset",
        description=(
            "Many-seed periodic PET dataset: random packing, FIRE intermediates "
            "and Langevin NVT frames, labelled with the driving model (extxyz)."
        ),
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="metatomic .pt")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seeds", type=str, default="0-7", help="e.g. 0-15 or 0,3,9-12")
    p.add_argument("--residue", type=str, default="ETOH", help="bulk density lookup")
    p.add_argument("--monomer-xyz", type=Path, default=None, help="default: ethanol")
    p.add_argument("--box-size", type=float, default=32.0)
    p.add_argument("--n-molecules", type=int, default=None, help="default: bulk density")
    p.add_argument("--target-density-g-cm3", type=float, default=None)
    p.add_argument(
        "--temperatures",
        type=str,
        default="300",
        help="NVT targets in K, cycled over seeds (e.g. 300,350,400)",
    )
    p.add_argument("--dt-fs", type=float, default=0.5)
    p.add_argument("--friction", type=float, default=0.01, help="Langevin 1/fs")
    p.add_argument("--fire-steps", type=int, default=200)
    p.add_argument("--fire-every", type=int, default=5, help="keep every Nth FIRE step")
    p.add_argument("--fire-fmax", type=float, default=0.05)
    p.add_argument("--md-steps", type=int, default=2000)
    p.add_argument("--md-every", type=int, default=20)
    p.add_argument(
        "--com-jitter-frac",
        type=float,
        default=0.25,
        help="random COM shift, fraction of lattice spacing",
    )
    p.add_argument(
        "--max-force",
        type=float,
        default=15.0,
        help="drop frames with max |F| above this (eV/Å)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from mmml.distill.box_dataset import BoxDatasetConfig, run_seed
    from mmml.interfaces.calculators.metatomic import load_metatomic_calculator
    from mmml.md.metatomic_pbc import (
        default_etoh_monomer_xyz,
        n_molecules_for_residue_box,
    )

    monomer = args.monomer_xyz or default_etoh_monomer_xyz()
    n_mol = args.n_molecules or n_molecules_for_residue_box(
        args.residue, box_side_A=args.box_size, density_g_cm3=args.target_density_g_cm3
    )
    cfg = BoxDatasetConfig(
        monomer_xyz=Path(monomer),
        n_molecules=int(n_mol),
        box_side_A=float(args.box_size),
        temperature_K=300.0,
        dt_fs=float(args.dt_fs),
        friction_per_fs=float(args.friction),
        fire_steps=int(args.fire_steps),
        fire_every=int(args.fire_every),
        fire_fmax=float(args.fire_fmax),
        md_steps=int(args.md_steps),
        md_every=int(args.md_every),
        com_jitter_frac=float(args.com_jitter_frac),
        max_force_eVA=float(args.max_force),
    )
    calc = load_metatomic_calculator(args.checkpoint, extra_kwargs={"non_conservative": False})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    temps = [float(t) for t in str(args.temperatures).split(",") if t.strip()]
    summaries = []
    for seed in parse_seeds(args.seeds):
        seed_cfg = replace(cfg, temperature_K=temps[seed % len(temps)])
        s = run_seed(calc, seed_cfg, seed=seed, out_dir=out)
        summaries.append(s)
        print(
            f"seed {seed} ({seed_cfg.temperature_K:g} K): fire {s['fire_kept']} (+{s['fire_dropped']} dropped)  "
            f"md {s['md_kept']} (+{s['md_dropped']} dropped)  "
            f"E {s['E_start_eV']:.1f} → {s['E_fire_end_eV']:.1f} eV  "
            f"<T>={s['T_md_mean_K']:.0f} K  {s['fire_s'] + s['md_s']:.0f} s",
            flush=True,
        )
    index = out / f"index_{args.seeds.replace(',', '_')}.json"
    index.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Wrote {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
