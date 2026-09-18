"""PET-MAD teacher → PhysNet student NPZ (acetone dataset + synthetic pool).

CHARMM-free. Labels come from an ASE teacher (metatomic `.pt` or any calculator
passed in tests). Does not run MD or PhysNet training.

Example::

    JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu \\
      mmml pet-physnet-distill \\
      --checkpoint /tmp/mmml-metatomic-models/pet-mad-xs-v1.5.0.pt \\
      --out-dir ./acetone_pet_distill --preset smoke
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from mmml.distill.acetone_pool import (
    POOL_PRESETS,
    PRESET_SMOKE,
    AcetonePoolConfig,
    build_acetone_pool,
    pool_config_for_preset,
)
from mmml.distill.npz_export import write_distill_npz
from mmml.distill.teacher_label import ENERGY_MODE_INTERACTION, ENERGY_MODES, label_geometries

_REPO = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mmml pet-physnet-distill",
        description=(
            "Build an acetone geometry pool (dataset + noise/scans), label it "
            "with a metatomic PET teacher, and write PhysNet-train NPZ in eV."
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Teacher AtomisticModel (.pt). Required unless --geometries-only.",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Train/valid NPZ + report.json")
    p.add_argument("--preset", choices=POOL_PRESETS, default=PRESET_SMOKE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--energy-mode",
        choices=ENERGY_MODES,
        default=ENERGY_MODE_INTERACTION,
        help="interaction: monomer E-E_ref and unswitched dimer E_int (default, hybrid MD)",
    )
    p.add_argument(
        "--geometries-only",
        action="store_true",
        help="Write unlabeled R/Z/N NPZ (no teacher). For pool inspection.",
    )
    p.add_argument(
        "--extra-extxyz",
        type=Path,
        nargs="*",
        default=(),
        help="Additional ASE extxyz frames (10-atom monomers or 20-atom dimers)",
    )
    p.add_argument("--valid-fraction", type=float, default=0.15)
    p.add_argument(
        "--student-yaml",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write physnet-train.yaml next to the NPZ (warm-start DESdimers)",
    )
    return p


def _write_student_yaml(path: Path, *, train: Path, valid: Path, seed: int) -> None:
    ckpt = _REPO / "examples" / "ckpts_json" / "DESdimers_params.json"
    if not ckpt.is_file():
        ckpt = Path("examples/ckpts_json/DESdimers_params.json")
    text = f"""# PhysNet student on PET-MAD acetone labels (eV / eV/Å).
# Warm-start architecture from DESdimers; labels *are* the teacher, so
# --distill is off (do not pass the .pt as --teacher-checkpoint).
data: {train.resolve()}
valid_data: {valid.resolve()}
ckpt_dir: ./ckpts/acetone_pet_student
tag: acetone_pet_student
physnet_checkpoint: {ckpt}
match_checkpoint_architecture: true
distill: false
charges: false
use_energy_bias: true
seed: {int(seed)}
batch_size: 8
num_epochs: 200
learning_rate: 0.0003
early_stop_patience: 40
objective: valid_forces_mae
energy_weight: 1.0
forces_weight: 52.91
# MAE in kcal/mol for logs only; NPZ stays eV.
conversion:
  energy: 23.060549
  forces: 23.060549
"""
    path.write_text(text)


def run(args: argparse.Namespace) -> dict:
    extra = tuple(Path(p) for p in (args.extra_extxyz or ()))
    cfg: AcetonePoolConfig = pool_config_for_preset(args.preset, seed=int(args.seed))
    if extra:
        cfg = replace(cfg, extra_extxyz=extra)
    geos = build_acetone_pool(cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.geometries_only:
        from mmml.distill.teacher_label import LabeledSample

        zeros = [
            LabeledSample(
                geometry=g,
                energy_eV=0.0,
                forces_ev_per_angstrom=g.positions * 0.0,
                energy_total_eV=0.0,
                energy_int_eV=None if g.kind == "monomer" else 0.0,
            )
            for g in geos
        ]
        paths = write_distill_npz(
            zeros,
            out_dir,
            valid_fraction=float(args.valid_fraction),
            seed=int(args.seed),
            metadata={"geometries_only": True, "preset": args.preset, "seed": int(args.seed)},
        )
        return {"n_geometries": len(geos), "paths": {k: str(v) for k, v in paths.items()}}

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --geometries-only")
    from mmml.interfaces.calculators.metatomic import load_metatomic_calculator

    calc = load_metatomic_calculator(args.checkpoint)
    labeled = label_geometries(calc, geos, energy_mode=str(args.energy_mode))
    teacher = Path(args.checkpoint).resolve()
    metadata = {
        "teacher": str(teacher),
        "teacher_size_bytes": int(teacher.stat().st_size),
        "energy_mode": str(args.energy_mode),
        "preset": str(args.preset),
        "seed": int(args.seed),
        "n_geometries": len(geos),
    }
    paths = write_distill_npz(
        labeled,
        out_dir,
        valid_fraction=float(args.valid_fraction),
        seed=int(args.seed),
        metadata=metadata,
    )
    if args.student_yaml:
        _write_student_yaml(
            out_dir / "physnet-train.yaml",
            train=paths["train"],
            valid=paths["valid"],
            seed=int(args.seed),
        )
        paths["student_yaml"] = out_dir / "physnet-train.yaml"
    return {
        "n_geometries": len(geos),
        "n_labeled": len(labeled),
        "paths": {k: str(v) for k, v in paths.items()},
        "metadata": metadata,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
