#!/usr/bin/env python
"""
Main entry point for MMML CLI commands.

Provides a unified interface for all MMML command-line tools.
"""

import sys
import argparse

from mmml.cli.completion import completion_main, try_autocomplete
from mmml.cli.help_text import format_top_level_help, validate_command
from mmml.cli.registry import _DISPATCH_COMMANDS


class _MMMLTopLevelParser(argparse.ArgumentParser):
    """Top-level parser with compact ``-h`` (details live in ``commands`` / ``examples``)."""

    def format_help(self) -> str:
        return format_top_level_help(prog=self.prog)


def main():
    """Main CLI dispatcher."""
    if len(sys.argv) > 1 and sys.argv[1] == "completion":
        return completion_main(sys.argv[2:])

    if try_autocomplete():
        return 0

    parser = _MMMLTopLevelParser(
        prog="mmml",
        description="MMML: Machine Learning for Molecular Modeling",
        add_help=True,
    )

    parser.add_argument(
        "command",
        nargs="?",
        metavar="command",
        help="Subcommand (see: mmml commands)",
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

    elif command == "make-res":
        from .misc import make_res_cli
        sys.argv = ["mmml make-res"] + args.args
        return make_res_cli.main()

    elif command == "make-box":
        from .misc import make_box_cli
        sys.argv = ["mmml make-box"] + args.args
        return make_box_cli.main()

    elif command == "build-crystal":
        from .misc import build_crystal_cli
        sys.argv = ["mmml build-crystal"] + args.args
        return build_crystal_cli.main()

    elif command == "run":
        from .run.run_sim import main as run_sim_main
        sys.argv = ["mmml run"] + args.args
        return run_sim_main()

    elif command == "md-system":
        from .run import md_system
        sys.argv = ["mmml md-system"] + args.args
        return md_system.main()

    elif command == "liquid-box":
        from .run import liquid_box
        sys.argv = ["mmml liquid-box"] + args.args
        return liquid_box.main()

    elif command == "mpi-check":
        from .run import mpi_check
        sys.argv = ["mmml mpi-check"] + args.args
        return mpi_check.main()

    elif command == "health-check":
        from .run import health_check
        sys.argv = ["mmml health-check"] + args.args
        return health_check.main()

    elif command == "warmup-mlpot-jax":
        from .run import warmup_mlpot_jax
        sys.argv = ["mmml warmup-mlpot-jax"] + args.args
        return warmup_mlpot_jax.main()

    elif command == "lambda-mbar":
        from .run import lambda_mbar
        sys.argv = ["mmml lambda-mbar"] + args.args
        return lambda_mbar.main()

    elif command == "run-pycharmm":
        from .run.run_pycharmm import main
        sys.argv = ["mmml run-pycharmm"] + args.args
        return main()

    elif command == "pycharmm-two-residue-sample":
        from .run.pycharmm_two_residue_sample import main
        sys.argv = ["mmml pycharmm-two-residue-sample"] + args.args
        return main()

    elif command == "xml2npz":
        from .misc import xml2npz
        sys.argv = ["mmml xml2npz"] + args.args
        return xml2npz.main()

    elif command == "validate":
        from .misc import validate_cli
        return validate_cli.main(args.args)

    elif command == "train":
        from . import train
        sys.argv = ["mmml train"] + args.args
        return train.main()

    elif command == "train-joint":
        from .misc import train_joint
        sys.argv = ["mmml train-joint"] + args.args
        return train_joint.main()

    elif command == "evaluate":
        from . import evaluate
        sys.argv = ["mmml evaluate"] + args.args
        return evaluate.main()

    elif command == "downstream":
        from . import downstream
        sys.argv = ["mmml downstream"] + args.args
        return downstream.main()

    elif command == "fix-and-split":
        from .misc import fix_and_split
        sys.argv = ["mmml fix-and-split"] + args.args
        return fix_and_split.main()

    elif command == "pyscf-dft":
        from .misc import pyscf_dft
        sys.argv = ["mmml pyscf-dft"] + args.args
        return pyscf_dft.main()

    elif command == "pyscf-mp2":
        from .misc import pyscf_mp2
        sys.argv = ["mmml pyscf-mp2"] + args.args
        return pyscf_mp2.main()

    elif command == "pyscf-evaluate":
        from .misc import pyscf_evaluate
        sys.argv = ["mmml pyscf-evaluate"] + args.args
        return pyscf_evaluate.main()

    elif command == "pyscf-evaluate-mp2":
        from .misc import pyscf_evaluate_mp2
        sys.argv = ["mmml pyscf-evaluate-mp2"] + args.args
        return pyscf_evaluate_mp2.main()

    elif command == "verify-esp-alignment":
        from .misc import verify_esp_alignment
        sys.argv = ["mmml verify-esp-alignment"] + args.args
        return verify_esp_alignment.main()

    elif command == "normal-mode-sample":
        from .misc import normal_mode_sample
        sys.argv = ["mmml normal-mode-sample"] + args.args
        return normal_mode_sample.main()

    elif command == "physnet-train":
        from .make import make_training
        sys.argv = ["mmml physnet-train"] + args.args
        return make_training.main()

    elif command == "physnet-md":
        from .misc import physnet_md
        sys.argv = ["mmml physnet-md"] + args.args
        return physnet_md.main()

    elif command == "physnet-evaluate":
        from .misc import physnet_evaluate
        sys.argv = ["mmml physnet-evaluate"] + args.args
        return physnet_evaluate.main()

    elif command == "compare-npz":
        from .misc import compare_npz
        sys.argv = ["mmml compare-npz"] + args.args
        return compare_npz.main()

    elif command == "cross-check":
        from .misc import cross_check
        sys.argv = ["mmml cross-check"] + args.args
        return cross_check.main()

    elif command == "efield-train":
        from .misc import efield_train
        sys.argv = ["mmml efield-train"] + args.args
        return efield_train.main()

    elif command == "efield-evaluate":
        from .misc import efield_evaluate
        sys.argv = ["mmml efield-evaluate"] + args.args
        return efield_evaluate.main()

    elif command == "efield-md":
        from .misc import efield_md
        sys.argv = ["mmml efield-md"] + args.args
        return efield_md.main()

    elif command == "ef-train":
        from .misc import ef_train
        sys.argv = ["mmml ef-train"] + args.args
        return ef_train.main()

    elif command == "ef-evaluate":
        from .misc import ef_evaluate
        sys.argv = ["mmml ef-evaluate"] + args.args
        return ef_evaluate.main()

    elif command == "ef-md":
        from .misc import ef_md
        sys.argv = ["mmml ef-md"] + args.args
        return ef_md.main()

    elif command == "active-learning":
        from .misc import active_learning
        sys.argv = ["mmml active-learning"] + args.args
        return active_learning.main()

    elif command == "kernel-fit":
        from .misc import kernel_fit
        sys.argv = ["mmml kernel-fit"] + args.args
        return kernel_fit.main()

    elif command == "interpolate-xyz":
        from .misc import interpolate_xyz
        sys.argv = ["mmml interpolate-xyz"] + args.args
        return interpolate_xyz.main()

    elif command == "unwrap-traj":
        from .misc import unwrap_traj
        sys.argv = ["mmml unwrap-traj"] + args.args
        return unwrap_traj.main()

    elif command == "sample-diverse-xyz":
        from mmml.generate.sample import sample_diverse_xyz
        sys.argv = ["mmml sample-diverse-xyz"] + args.args
        return sample_diverse_xyz.main()

    elif command == "gui":
        from . import gui
        sys.argv = ["mmml gui"] + args.args
        return gui.main()

    elif command == "extract-checkpoint-metrics":
        from .misc import extract_checkpoint_metrics
        sys.argv = ["mmml extract-checkpoint-metrics"] + args.args
        return extract_checkpoint_metrics.main()

    elif command == "orbax-to-json":
        from .misc import orbax_to_json_cmd
        sys.argv = ["mmml orbax-to-json"] + args.args
        return orbax_to_json_cmd.main()

    elif command == "orca-server":
        from mmml.interfaces.orca_external.server import main as orca_server_main
        return orca_server_main(args.args)

    elif command == "orca-client":
        from mmml.interfaces.orca_external.client import main as orca_client_main
        return orca_client_main(args.args)

    elif command == "orca-external":
        from mmml.interfaces.orca_external.runner import main as orca_external_main
        return orca_external_main(args.args)

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
