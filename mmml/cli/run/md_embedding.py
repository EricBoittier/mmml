"""``mmml md-embedding`` — solvated peptide partial MLpot workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml md-embedding",
        description=(
            "Solvated-peptide MD embedding: train PhysNet on peptide NPZ, "
            "build CHARMM PEPT+TIP3 box, register partial MLpot (n_monomers=1). "
            "See docs/examples/md-embedding-design.md."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="phase", required=True)

    train = sub.add_parser(
        "train",
        help="Download/split aaa.ama NPZ, run PhysNet smoke, export JSON checkpoint.",
    )
    train.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Artifact root (train.npz, checkpoints, train_manifest.json).",
    )
    train.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Existing dataset_aaa.npz (default: output-dir/dataset_aaa.npz).",
    )
    train.add_argument(
        "--dataset-url",
        type=str,
        default=None,
        help="Override NPZ download URL (default: aaa.ama GitHub raw).",
    )
    train.add_argument(
        "--no-download",
        action="store_true",
        help="Require local NPZ; do not fetch from GitHub.",
    )
    train.add_argument("--train-fraction", type=float, default=0.9)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--tag", type=str, default="aaa_smoke")
    train.add_argument(
        "--skip-train",
        action="store_true",
        help="Only download/split/write train_config.yaml (no physnet-train).",
    )
    train.add_argument(
        "--skip-export-json",
        action="store_true",
        help="Skip orbax-to-json after training.",
    )
    train.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML overrides merged into generated train config.",
    )
    train.add_argument(
        "--simple-split",
        action="store_true",
        help="Use shuffle split only (skip mmml fix-and-split manifest).",
    )
    train.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip ASE structure figures under output-dir/figures/.",
    )

    build = sub.add_parser(
        "build",
        help="Build CGENFF TRIA + TIP3 box; MM SD minimize; write model.psf/crd/box.json.",
    )
    build.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory for box artifacts (same root as train is fine).",
    )
    build.add_argument("--n-waters", type=int, default=10)
    build.add_argument("--box-side-A", type=float, default=28.0)
    build.add_argument("--seed", type=int, default=11)
    build.add_argument(
        "--charmm-sd-steps",
        type=int,
        default=200,
        help="CHARMM SD steps during MM-only pre-minimize.",
    )
    build.add_argument(
        "--no-charmm-minimize",
        action="store_true",
        help="Skip CHARMM SD after box build.",
    )
    build.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip ASE box/peptide figures under output-dir/figures/.",
    )

    run = sub.add_parser(
        "run",
        help="Load built box, register partial MLpot on PEPT, optional MLpot SD.",
    )
    run.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory containing box.json from build phase.",
    )
    run.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="PhysNet JSON checkpoint (from train or orbax-to-json).",
    )
    run.add_argument("--ml-seg-id", type=str, default="PEPT")
    run.add_argument("--ml-charge", type=float, default=1.0)
    run.add_argument(
        "--ml-fq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fluctuating ML charges (CGenFF on ML atoms).",
    )
    run.add_argument("--mini-nstep", type=int, default=0, help="MLpot SD steps (0=energy only).")
    run.add_argument("--box-side-A", type=float, default=None, help="Override box.json side.")
    run.add_argument(
        "--mlmm-cutoff",
        type=float,
        default=None,
        metavar="ANG",
        help="ML–MM electrostatic outer cutoff (Å); forwarded to MLpot (Phase 2).",
    )
    run.add_argument(
        "--mlmm-cuton",
        type=float,
        default=None,
        metavar="ANG",
        help="ML–MM electrostatic inner cuton (Å); forwarded to MLpot (Phase 2).",
    )

    return parser


def _load_config_overrides(path: Path | None) -> dict | None:
    if path is None:
        return None
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _cmd_train(args: argparse.Namespace) -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.embedding_workflow import run_train_phase

    overrides = _load_config_overrides(args.config)
    result = run_train_phase(
        args.output_dir,
        npz_path=args.npz,
        download=not args.no_download,
        train_fraction=args.train_fraction,
        seed=args.seed,
        skip_train=args.skip_train,
        skip_export_json=args.skip_export_json,
        tag=args.tag,
        config_overrides=overrides,
        use_fix_and_split=not args.simple_split,
        write_plots=not args.no_plot,
    )
    print(json.dumps(result.report, indent=2))
    print(f"Wrote {result.manifest_path}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        maybe_rerun_mmml_under_mpirun,
        prepare_serial_charmm_mpi_env,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.embedding_workflow import (
        build_embedding_box,
    )

    prepare_serial_charmm_mpi_env()
    rerun = maybe_rerun_mmml_under_mpirun(
        ["build", "-o", str(args.output_dir)],
        subcommand="md-embedding",
    )
    if rerun is not None:
        return int(rerun)

    try:
        result = build_embedding_box(
            args.output_dir,
            n_waters=args.n_waters,
            box_side_A=args.box_side_A,
            seed=args.seed,
            charmm_mm_minimize=not args.no_charmm_minimize,
            charmm_sd_steps=args.charmm_sd_steps,
            write_plots=not args.no_plot,
        )
    except ModuleNotFoundError as exc:
        if "pycharmm" in str(exc).lower() or "charmm" in str(exc).lower():
            print("Error: md-embedding build requires PyCHARMM/CHARMM.", file=sys.stderr)
            return 1
        raise

    print(f"Wrote {result.psf_path}, {result.crd_path}, {result.box_json_path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        maybe_rerun_mmml_under_mpirun,
        prepare_serial_charmm_mpi_env,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.embedding_workflow import (
        run_embedding_phase,
    )

    prepare_serial_charmm_mpi_env()
    rerun = maybe_rerun_mmml_under_mpirun(
        ["run", "-o", str(args.output_dir), "--checkpoint", str(args.checkpoint)],
        subcommand="md-embedding",
    )
    if rerun is not None:
        return int(rerun)

    try:
        result = run_embedding_phase(
            args.output_dir,
            args.checkpoint,
            ml_seg_id=args.ml_seg_id,
            ml_charge=args.ml_charge,
            ml_fq=args.ml_fq,
            mini_nstep=args.mini_nstep,
            box_side_A=args.box_side_A,
            mlmm_cutoff=args.mlmm_cutoff,
            mlmm_cuton=args.mlmm_cuton,
        )
    except ModuleNotFoundError as exc:
        if "pycharmm" in str(exc).lower() or "charmm" in str(exc).lower():
            print("Error: md-embedding run requires PyCHARMM/CHARMM.", file=sys.stderr)
            return 1
        raise

    print(
        f"ML segment {result.ml_seg_id}: {result.n_ml_atoms} ML atoms / "
        f"{result.n_total_atoms} total; ENER={result.charmm_total_energy_kcalmol}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parsed_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(parsed_argv)
    if args.phase == "train":
        return _cmd_train(args)
    if args.phase == "build":
        return _cmd_build(args)
    if args.phase == "run":
        return _cmd_run(args)
    parser.error(f"unknown phase: {args.phase}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
