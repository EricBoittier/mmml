"""PET-MAD teacher → PhysNet student NPZ (acetone dataset + synthetic pool).

CHARMM-free. Labels come from an ASE teacher (metatomic `.pt` or any calculator
passed in tests). Does not run MD or PhysNet training.

Example::

    JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu \\
      mmml pet-physnet-distill \\
      --checkpoint /tmp/mmml-metatomic-models/pet-mad-xs-v1.5.0.pt \\
      --out-dir ./acetone_pet_distill --preset smoke

Labels go through one batched TorchScript forward per ``--max-atoms-per-batch``
chunk (``--teacher-backend torchscript``); ``ase`` is the per-structure path.
Other PETs: ``python -c "from upet import save_upet; save_upet(model='pet-omol',
size='m', version='1.0.0', output='pet-omol-m-v1.0.0.pt')"``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from mmml.distill.acetone_pool import (
    DIMER_ATOMS,
    POOL_PRESETS,
    PRESET_SMOKE,
    AcetonePoolConfig,
    build_acetone_pool,
    pool_config_for_preset,
)
from mmml.distill.npz_export import write_distill_npz
from mmml.distill.teacher_label import (
    ENERGY_MODE_MLMM,
    ENERGY_MODE_TOTAL,
    ENERGY_MODES,
    label_geometries,
)

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
        default=ENERGY_MODE_MLMM,
        help=(
            "mlmm (default): monomer E-E_ref, dimer E_AB-2E_ref, full forces; matches "
            "PhysNet MLpot, which forms E_int=P(AB)-P(A)-P(B) itself. interaction: "
            "dimer E=E_int (not MLpot-consistent). total: raw teacher energies."
        ),
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
    p.add_argument(
        "--from-box-extxyz",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Periodic MD frames (extxyz with cell, e.g. metatomic-pbc-md "
            "--traj-every). Replaces the acetone pool with monomers and COM-close "
            "dimers cut out by minimum image. Needs --atoms-per-monomer."
        ),
    )
    p.add_argument("--atoms-per-monomer", type=int, default=None)
    p.add_argument(
        "--reference-monomer-xyz",
        type=Path,
        default=None,
        help="Gas-phase monomer for E_ref in interaction mode (box pool only)",
    )
    p.add_argument("--frame-stride", type=int, default=1, help="Use every Nth box frame")
    p.add_argument(
        "--dimer-com-cutoff",
        type=float,
        default=7.5,
        help="Å centroid distance for box dimers (MLpot sparse ML range: on + ml width)",
    )
    p.add_argument(
        "--dimer-r-bins",
        type=str,
        default="0,3.5,4.5,5.25,6.0,7.5",
        help="Å bin edges for an even dimer draw per frame",
    )
    p.add_argument("--max-monomers-per-frame", type=int, default=8)
    p.add_argument("--max-dimers-per-frame", type=int, default=24)
    p.add_argument("--valid-fraction", type=float, default=0.15)
    p.add_argument(
        "--teacher-backend",
        choices=("torchscript", "ase"),
        default="torchscript",
        help=(
            "torchscript: batched AtomisticModel forward over many structures "
            "(default); ase: one MetatomicCalculator call per structure"
        ),
    )
    p.add_argument(
        "--max-atoms-per-batch",
        type=int,
        default=4096,
        help="torchscript backend: atom budget per forward (lower for larger PETs)",
    )
    p.add_argument(
        "--max-systems-per-batch",
        type=int,
        default=512,
        help="torchscript backend: structure budget per forward",
    )
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


def _box_pool(args: argparse.Namespace):
    from ase.io import read as ase_read

    from mmml.distill.box_clusters import BoxClusterConfig, box_cluster_pool

    if args.atoms_per_monomer is None:
        raise SystemExit("--from-box-extxyz needs --atoms-per-monomer")
    stride = max(int(args.frame_stride), 1)
    frames = []
    for path in args.from_box_extxyz:
        frames.extend(ase_read(str(path), index=f"::{stride}"))
    ref = None
    if args.reference_monomer_xyz is not None:
        ref = ase_read(str(args.reference_monomer_xyz))
    elif str(args.energy_mode) != ENERGY_MODE_TOTAL:
        raise SystemExit(
            "mlmm/interaction labels need --reference-monomer-xyz (gas-phase monomer "
            "for E_ref), or pass --energy-mode total"
        )
    cfg = BoxClusterConfig(
        atoms_per_monomer=int(args.atoms_per_monomer),
        dimer_com_cutoff_A=float(args.dimer_com_cutoff),
        dimer_r_bins_A=tuple(float(x) for x in str(args.dimer_r_bins).split(",") if x.strip()),
        max_monomers_per_frame=int(args.max_monomers_per_frame),
        max_dimers_per_frame=int(args.max_dimers_per_frame),
        seed=int(args.seed),
    )
    stats: dict = {}
    geos = box_cluster_pool(frames, cfg, reference_monomer=ref, stats=stats)
    stats["n_box_frames"] = len(frames)
    return geos, stats


def run(args: argparse.Namespace) -> dict:
    box_stats = None
    if args.from_box_extxyz:
        geos, box_stats = _box_pool(args)
        pad_atoms = 2 * int(args.atoms_per_monomer)
    else:
        extra = tuple(Path(p) for p in (args.extra_extxyz or ()))
        cfg: AcetonePoolConfig = pool_config_for_preset(args.preset, seed=int(args.seed))
        if extra:
            cfg = replace(cfg, extra_extxyz=extra)
        geos = build_acetone_pool(cfg)
        pad_atoms = DIMER_ATOMS
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
            pad_atoms=pad_atoms,
            valid_fraction=float(args.valid_fraction),
            seed=int(args.seed),
            metadata={"geometries_only": True, "preset": args.preset, "seed": int(args.seed)},
        )
        return {"n_geometries": len(geos), "paths": {k: str(v) for k, v in paths.items()}}

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --geometries-only")
    if args.teacher_backend == "torchscript":
        from mmml.distill.batched_teacher import BatchedMetatomicTeacher

        evaluator = BatchedMetatomicTeacher(
            args.checkpoint,
            max_atoms_per_batch=int(args.max_atoms_per_batch),
            max_systems_per_batch=int(args.max_systems_per_batch),
        )
    else:
        from mmml.interfaces.calculators.metatomic import load_metatomic_calculator

        evaluator = load_metatomic_calculator(args.checkpoint)
    t_label = time.perf_counter()
    labeled = label_geometries(evaluator, geos, energy_mode=str(args.energy_mode))
    label_s = time.perf_counter() - t_label
    teacher = Path(args.checkpoint).resolve()
    metadata = {
        "teacher": str(teacher),
        "teacher_size_bytes": int(teacher.stat().st_size),
        "teacher_backend": str(args.teacher_backend),
        "label_s": float(label_s),
        "energy_mode": str(args.energy_mode),
        "preset": str(args.preset),
        "seed": int(args.seed),
        "n_geometries": len(geos),
    }
    if box_stats is not None:
        metadata["box_extxyz"] = [str(Path(p).resolve()) for p in args.from_box_extxyz]
        metadata.update(box_stats)
    paths = write_distill_npz(
        labeled,
        out_dir,
        pad_atoms=pad_atoms,
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
