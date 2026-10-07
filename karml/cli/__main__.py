#!/usr/bin/env python
"""
Main entry point for KARML CLI commands.

Provides a unified interface for all KARML command-line tools.
"""

import os
import sys

# Must run before any ``import karml.*`` / ``import jax`` (umbrella → jax_md).
# Do not import karml here: package init pulls PhysNet and can init JAX too early.
# Stale JAX_PLATFORMS=rocm on NVIDIA nodes aborts backend init.
_plat = (os.environ.get("JAX_PLATFORMS") or "").strip()
if _plat:
    _parts = [p.strip() for p in _plat.split(",") if p.strip()]
    _clean = [p for p in _parts if p.lower() != "rocm"]
    if _clean != _parts:
        if _clean:
            os.environ["JAX_PLATFORMS"] = ",".join(_clean)
        else:
            # Prefer CUDA when a GPU allocation is visible; else leave unset (auto).
            os.environ.pop("JAX_PLATFORMS", None)
            if (
                (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
                or (os.environ.get("SLURM_JOB_GPUS") or "").strip()
                or "gpu" in str(os.environ.get("SLURM_JOB_PARTITION", "")).lower()
            ):
                os.environ["JAX_PLATFORMS"] = "cuda"
if (os.environ.get("JAX_PLATFORM_NAME") or "").strip().lower() == "rocm":
    os.environ.pop("JAX_PLATFORM_NAME", None)

import argparse

from karml.cli.completion import completion_main, try_autocomplete
from karml.cli.help_text import format_top_level_help, validate_command
from karml.cli.help_style import install_colored_argparse
from karml.cli.registry import _DISPATCH_COMMANDS

install_colored_argparse()


def _hard_exit(code: int | None) -> None:
    """Terminate with *code*, preserving non-zero status past CHARMM teardown.

    Importing pycharmm installs a Fortran/MPI finalizer that runs during
    interpreter shutdown and **resets the process exit status to 0**. A command
    that returned 1 therefore reported success to the shell, so every caller
    that trusts exit codes -- Slurm, CI, Make, the validation campaign -- was
    blind to failures.

    For **non-zero** codes we ``os._exit`` so atexit/Fortran shutdown cannot
    mask the failure. For **zero** we take a normal ``SystemExit`` so OpenMPI
    can finalize cleanly: ``os._exit(0)`` skips ``MPI_Finalize`` and PRRTE then
    often returns exit 1 (with empty Sphinx help noise) even though the app
    succeeded.
    """
    code_i = int(code or 0)
    sys.stdout.flush()
    sys.stderr.flush()
    if code_i == 0:
        raise SystemExit(0)
    os._exit(code_i)


def cli() -> None:
    """Console-script entry point. Never returns.

    ``main`` keeps returning an ``int`` so it stays callable from tests; only
    this wrapper forces the process exit status.

    Uncaught exceptions must also go through ``os._exit``: importing PyCHARMM
    installs a Fortran atexit that otherwise resets the process status to 0,
    which made failed ``umbrella-sample`` look successful to Snakemake.
    """
    try:
        code = main()
    except SystemExit as exc:
        code = exc.code
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    _hard_exit(code)


class _KARMLTopLevelParser(argparse.ArgumentParser):
    """Top-level parser with compact ``-h`` (details live in ``commands`` / ``examples``)."""

    def format_help(self) -> str:
        return format_top_level_help(prog=self.prog)


def main():
    """Main CLI dispatcher."""
    if len(sys.argv) > 1 and sys.argv[1] == "completion":
        return completion_main(sys.argv[2:])

    if try_autocomplete():
        return 0

    parser = _KARMLTopLevelParser(
        prog="karml",
        description="KARML: Machine Learning for Molecular Modeling",
        add_help=True,
    )

    parser.add_argument(
        "command",
        nargs="?",
        metavar="command",
        help="Subcommand (see: karml commands)",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments for the command",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    err = validate_command(args.command, allowed=_DISPATCH_COMMANDS)
    if err:
        parser.error(err)

    command = args.command

    # Dispatch to appropriate command
    if command == "commands":
        from .commands_help import commands_main

        return commands_main(args.args)

    elif command == "examples":
        from .commands_help import examples_main

        return examples_main(args.args)

    elif command == "configure":
        from .configure import configure_main

        return configure_main(args.args)

    elif command == "plot-restart-velocities":
        from .plot.plot_restart_velocities import main as plot_restart_velocities_main

        return plot_restart_velocities_main(args.args)

    elif command == "make-res":
        from .misc import make_res_cli
        sys.argv = ["karml make-res"] + args.args
        return make_res_cli.main()

    elif command == "make-box":
        from .misc import make_box_cli
        sys.argv = ["karml make-box"] + args.args
        return make_box_cli.main()

    elif command == "build-crystal":
        from .misc import build_crystal_cli
        sys.argv = ["karml build-crystal"] + args.args
        return build_crystal_cli.main()

    elif command == "run":
        from .run.run_sim import main as run_sim_main
        sys.argv = ["karml run"] + args.args
        return run_sim_main()

    elif command == "md-system":
        from .run import md_system
        sys.argv = ["karml md-system"] + args.args
        return md_system.main()

    elif command == "liquid-box":
        from .run import liquid_box
        sys.argv = ["karml liquid-box"] + args.args
        return liquid_box.main()

    elif command == "md-embedding":
        from .run import md_embedding
        sys.argv = ["karml md-embedding"] + args.args
        return md_embedding.main()

    elif command == "mpi-check":
        from .run import mpi_check
        sys.argv = ["karml mpi-check"] + args.args
        return mpi_check.main()

    elif command == "mpi-launch":
        from .run import mpi_launch
        return mpi_launch.main(args.args)

    elif command == "doctor":
        from . import doctor
        return doctor.main(args.args)

    elif command == "health-check":
        from .run import health_check
        sys.argv = ["karml health-check"] + args.args
        return health_check.main()

    elif command == "env":
        from . import env as env_cli
        sys.argv = ["karml env"] + args.args
        return env_cli.main()

    elif command == "warmup-mlpot-jax":
        from .run import warmup_mlpot_jax
        sys.argv = ["karml warmup-mlpot-jax"] + args.args
        return warmup_mlpot_jax.main()

    elif command == "lambda-mbar":
        from .run import lambda_mbar
        sys.argv = ["karml lambda-mbar"] + args.args
        return lambda_mbar.main()

    elif command == "run-pycharmm":
        from .run.run_pycharmm import main
        sys.argv = ["karml run-pycharmm"] + args.args
        return main()

    elif command == "pycharmm-two-residue-sample":
        from .run.pycharmm_two_residue_sample import main
        sys.argv = ["karml pycharmm-two-residue-sample"] + args.args
        return main()

    elif command == "xml2npz":
        from .misc import xml2npz
        sys.argv = ["karml xml2npz"] + args.args
        return xml2npz.main()

    elif command == "npz2traj":
        from .misc import convert_npz_traj
        return convert_npz_traj.main(args.args)

    elif command == "validate":
        from .misc import validate_cli
        return validate_cli.main(args.args)

    elif command == "train-joint":
        from .misc import train_joint
        sys.argv = ["karml train-joint"] + args.args
        return train_joint.main()

    elif command == "downstream":
        from . import downstream
        sys.argv = ["karml downstream"] + args.args
        return downstream.main()

    elif command == "fix-and-split":
        from .misc import fix_and_split
        sys.argv = ["karml fix-and-split"] + args.args
        return fix_and_split.main()

    elif command == "prepare-mm-dataset":
        from .misc import prepare_mm_dataset
        return prepare_mm_dataset.main(args.args)

    elif command == "pyscf-dft":
        from .misc import pyscf_dft
        sys.argv = ["karml pyscf-dft"] + args.args
        return pyscf_dft.main()

    elif command == "pyscf-mp2":
        from .misc import pyscf_mp2
        sys.argv = ["karml pyscf-mp2"] + args.args
        return pyscf_mp2.main()

    elif command == "pyscf-evaluate":
        from .misc import pyscf_evaluate
        sys.argv = ["karml pyscf-evaluate"] + args.args
        return pyscf_evaluate.main()

    elif command == "pyscf-evaluate-mp2":
        from .misc import pyscf_evaluate_mp2
        sys.argv = ["karml pyscf-evaluate-mp2"] + args.args
        return pyscf_evaluate_mp2.main()

    elif command == "verify-esp-alignment":
        from .misc import verify_esp_alignment
        sys.argv = ["karml verify-esp-alignment"] + args.args
        return verify_esp_alignment.main()

    elif command == "normal-mode-sample":
        from .misc import normal_mode_sample
        sys.argv = ["karml normal-mode-sample"] + args.args
        return normal_mode_sample.main()

    elif command == "dimer-scan":
        from .misc import dimer_scan
        return dimer_scan.main(args.args)
    elif command == "ic-scan":
        from .misc import ic_scan
        return ic_scan.main(args.args)

    elif command == "neb":
        from .misc import neb
        return neb.main(args.args)

    elif command == "umbrella-sample":
        from .misc import umbrella_sample
        return umbrella_sample.main(args.args)

    elif command == "umbrella-mbar":
        from .misc import umbrella_mbar
        return umbrella_mbar.main(args.args)

    elif command == "dmc":
        from karml.generate.dmc.dmc import main as dmc_main
        return dmc_main(args.args)

    elif command == "mode-check":
        from .misc import mode_check
        return mode_check.main(args.args)

    elif command == "physnet-train":
        from .make import make_training
        sys.argv = ["karml physnet-train"] + args.args
        return make_training.main()

    elif command == "label-acquire":
        from .misc import label_acquire
        return label_acquire.main(args.args)

    elif command == "pet-box-dataset":
        from .misc import pet_box_dataset
        return pet_box_dataset.main(args.args)

    elif command == "pet-physnet-distill":
        from .misc import pet_physnet_distill
        return pet_physnet_distill.main(args.args)

    elif command == "tune-mm-nonbonded":
        from .misc import tune_mm_nonbonded
        return tune_mm_nonbonded.main(args.args)

    elif command == "pet-interaction-pes":
        from .misc import pet_interaction_pes
        return pet_interaction_pes.main(args.args)

    elif command == "metatomic-pbc-md":
        from .misc import metatomic_pbc_md
        return metatomic_pbc_md.main(args.args)

    elif command == "physnet-md":
        from .misc import physnet_md
        sys.argv = ["karml physnet-md"] + args.args
        return physnet_md.main()

    elif command == "physnet-evaluate":
        from .misc import physnet_evaluate
        sys.argv = ["karml physnet-evaluate"] + args.args
        return physnet_evaluate.main()

    elif command == "compare-npz":
        from .misc import compare_npz
        sys.argv = ["karml compare-npz"] + args.args
        return compare_npz.main()

    elif command == "diagnose-lc-outliers":
        from .misc.diagnose_learning_curve_outliers import (
            main as diagnose_lc_outliers_main,
        )
        return diagnose_lc_outliers_main(args.args)

    elif command == "compare-charmm-ml":
        from .misc import compare_charmm_ml
        sys.argv = ["karml compare-charmm-ml"] + args.args
        return compare_charmm_ml.main()

    elif command == "cross-check":
        from .misc import cross_check
        sys.argv = ["karml cross-check"] + args.args
        return cross_check.main()

    elif command == "efield-train":
        from .misc import efield_train
        sys.argv = ["karml efield-train"] + args.args
        return efield_train.main()

    elif command == "efield-evaluate":
        from .misc import efield_evaluate
        sys.argv = ["karml efield-evaluate"] + args.args
        return efield_evaluate.main()

    elif command == "efield-md":
        from .misc import efield_md
        sys.argv = ["karml efield-md"] + args.args
        return efield_md.main()

    elif command == "kernnn-train":
        from .misc import kernnn_train
        sys.argv = ["karml kernnn-train"] + args.args
        return kernnn_train.main()

    elif command == "kernnn-evaluate":
        from .misc import kernnn_evaluate
        sys.argv = ["karml kernnn-evaluate"] + args.args
        return kernnn_evaluate.main()

    elif command == "active-learning":
        from .misc import active_learning
        sys.argv = ["karml active-learning"] + args.args
        return active_learning.main()

    elif command == "pes-design":
        from .misc import pes_design
        return pes_design.main(args.args)

    elif command == "kernel-fit":
        from .misc import kernel_fit
        sys.argv = ["karml kernel-fit"] + args.args
        return kernel_fit.main()

    elif command == "interpolate-xyz":
        from .misc import interpolate_xyz
        sys.argv = ["karml interpolate-xyz"] + args.args
        return interpolate_xyz.main()

    elif command == "unwrap-traj":
        from .misc import unwrap_traj
        sys.argv = ["karml unwrap-traj"] + args.args
        return unwrap_traj.main()

    elif command == "analyze-liquid":
        from .misc import analyze_liquid
        return analyze_liquid.main(args.args)

    elif command == "sample-diverse-xyz":
        from karml.generate.sample import sample_diverse_xyz
        sys.argv = ["karml sample-diverse-xyz"] + args.args
        return sample_diverse_xyz.main()

    elif command == "gui":
        from . import gui
        sys.argv = ["karml gui"] + args.args
        return gui.main()

    elif command == "extract-checkpoint-metrics":
        from .misc import extract_checkpoint_metrics
        sys.argv = ["karml extract-checkpoint-metrics"] + args.args
        return extract_checkpoint_metrics.main()

    elif command == "orbax-to-json":
        from .misc import orbax_to_json_cmd
        sys.argv = ["karml orbax-to-json"] + args.args
        return orbax_to_json_cmd.main()

    elif command == "orca-server":
        from karml.interfaces.orca_external.server import main as orca_server_main
        return orca_server_main(args.args)

    elif command == "orca-client":
        from karml.interfaces.orca_external.client import main as orca_client_main
        return orca_client_main(args.args)

    elif command == "orca-external":
        from karml.interfaces.orca_external.runner import main as orca_external_main
        return orca_external_main(args.args)

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    cli()
