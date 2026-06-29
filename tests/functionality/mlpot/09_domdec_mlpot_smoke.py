#!/usr/bin/env python3
"""Tier 3 DOMDEC + MLpot smoke: tiny CHARMM ENER with optional active DOMDEC.

This is a manual cluster smoke, not a production workflow and not CI. It answers
one narrow question: can a DOMDEC-enabled CHARMM build run MLpot registration and
one ``ENER`` after an explicit DOMDEC stream command?

Safety:
  - Live mode requires ``MMML_DOMDEC_MLPOT_SMOKE=1``.
  - The default live command is ``domdec on``; pass ``--no-domdec-command`` for a
    no-DOMDEC baseline or ``--domdec-command 'domdec off'`` for an off-control.

Example:
  MMML_MPI_NP=1 MMML_DOMDEC_MLPOT_SMOKE=1 \\
    ./scripts/mmml-charmm-mpirun.sh python \\
    tests/functionality/mlpot/09_domdec_mlpot_smoke.py \\
    --residue ACO --n-molecules 2 --box-side 28
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _common import (
    add_cluster_args,
    all_atom_selection,
    build_ase_cluster,
    charmm_energy_row,
    check_mlpot_symbols,
    load_physnet_for_cluster,
    print_header,
    resolve_checkpoint,
    setup_charmm_nbonds,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cluster_args(parser)
    parser.set_defaults(residue="ACO", n_molecules=2, spacing=5.0)
    parser.add_argument("--box-side", type=float, default=28.0)
    parser.add_argument(
        "--domdec-command",
        default="domdec on",
        help="CHARMM stream command before MLpot registration (default: 'domdec on').",
    )
    parser.add_argument(
        "--no-domdec-command",
        action="store_true",
        help="Do not send a DOMDEC command; useful as a same-script baseline.",
    )
    parser.add_argument(
        "--allow-domdec-off-hook",
        action="store_true",
        help=(
            "Do not set MMML_NO_CHARMM_DOMDEC_OFF=1. Default protects this active-DOMDEC "
            "smoke from the MLpot DOMDEC-off safety hook."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the recommended cluster command and exit without importing PyCHARMM.",
    )
    return parser.parse_args()


def _mpi_info() -> tuple[int, int]:
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

        return mpi_rank_size()
    except Exception:
        return 0, 1


def _print_dry_run(args: argparse.Namespace) -> None:
    cmd = [
        "MMML_MPI_NP=1",
        "MMML_DOMDEC_MLPOT_SMOKE=1",
        "./scripts/mmml-charmm-mpirun.sh",
        "python",
        "tests/functionality/mlpot/09_domdec_mlpot_smoke.py",
        "--residue",
        str(args.residue),
        "--n-molecules",
        str(args.n_molecules),
        "--spacing",
        str(args.spacing),
        "--box-side",
        str(args.box_side),
    ]
    if args.no_domdec_command:
        cmd.append("--no-domdec-command")
    else:
        cmd.extend(["--domdec-command", repr(str(args.domdec_command))])
    if args.checkpoint is not None:
        cmd.extend(["--checkpoint", str(Path(args.checkpoint))])
    print(" ".join(cmd))


def _run_domdec_command(command: str | None) -> None:
    if not command:
        print("DOMDEC command: skipped", flush=True)
        return
    import pycharmm.lingo as lingo

    print(f"DOMDEC command: {command!r}", flush=True)
    lingo.charmm_script(str(command))


def main() -> int:
    args = _parse_args()
    if args.dry_run:
        _print_dry_run(args)
        return 0

    if os.environ.get("MMML_DOMDEC_MLPOT_SMOKE") != "1":
        print(
            "Refusing live DOMDEC+MLpot smoke without MMML_DOMDEC_MLPOT_SMOKE=1.",
            file=sys.stderr,
        )
        print("Use --dry-run for a launch recipe.", file=sys.stderr)
        return 2

    if not args.allow_domdec_off_hook:
        os.environ["MMML_NO_CHARMM_DOMDEC_OFF"] = "1"

    rank, size = _mpi_info()
    print_header("Tier 3 DOMDEC + MLpot ENER smoke")
    if rank == 0:
        print(
            f"MPI size={size}; residue={args.residue}; n_molecules={args.n_molecules}; "
            f"box={float(args.box_side):.3f} Å",
            flush=True,
        )

    missing = check_mlpot_symbols()
    if missing:
        print(f"FAIL: missing libcharmm MLpot symbols: {missing}", file=sys.stderr)
        return 1

    ckpt = resolve_checkpoint(args.checkpoint)
    z, r = build_ase_cluster(args.residue, args.n_molecules, args.spacing)
    n_atoms = len(z)
    if rank == 0:
        print(f"Cluster: {n_atoms} atoms; checkpoint={ckpt}", flush=True)

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import ase
    import pycharmm
    import pycharmm.energy as energy

    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment

    setup_charmm_environment(use_pbc=True, cubic_box_side_A=float(args.box_side))
    setup_charmm_nbonds()
    _run_domdec_command(None if args.no_domdec_command else args.domdec_command)

    params, model = load_physnet_for_cluster(ckpt, n_atoms)
    model.natoms = n_atoms
    atoms = ase.Atoms(numbers=z, positions=r)

    from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_pyc

    pyc_model = get_pyc(params, model, atoms)
    mlpot = None
    try:
        print("Registering MLpot...", flush=True)
        mlpot = pycharmm.MLpot(
            ml_model=pyc_model,
            ml_Z=list(z),
            ml_selection=all_atom_selection(),
            ml_charge=0,
            ml_fq=True,
        )
        print("Running CHARMM ENER...", flush=True)
        energy.show()
        terms = charmm_energy_row()
    except Exception as exc:
        print(f"FAIL: DOMDEC+MLpot smoke raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if mlpot is not None:
            try:
                mlpot.unset_mlpot()
            except Exception as exc:
                print(f"WARN: MLpot unset failed: {exc}", file=sys.stderr)

    if rank == 0:
        print("\nCHARMM energy terms (kcal/mol):")
        for key in sorted(terms):
            if abs(terms[key]) > 1e-8 or key in ("ENER", "USER", "TOTE"):
                print(f"  {key:12s} {terms[key]:14.6f}")
        print("\nPASS: DOMDEC+MLpot ENER smoke completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
