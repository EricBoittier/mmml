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
    --residue OCOH --n-molecules 1 --box-side 32
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residue", default="OCOH")
    parser.add_argument("--n-molecules", type=int, default=1)
    parser.add_argument("--spacing", type=float, default=5.0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="PhysNet checkpoint (.json or Orbax root). Default: MMML_CKPT or repo ckpts.",
    )
    parser.add_argument("--box-side", type=float, default=32.0)
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
        "--allow-domdec-order-risk",
        action="store_true",
        help=(
            "Run even when the selected residue is known to have DOMDEC-incompatible "
            "heavy/H atom ordering."
        ),
    )
    parser.add_argument(
        "--allow-mpi-size-gt1",
        action="store_true",
        help="Allow np>1. Default refuses because this Tier 3 smoke can hang in CHARMM setup.",
    )
    parser.add_argument(
        "--no-skip-vacuum-crystal-free-on-mpi",
        action="store_true",
        help=(
            "For np>1 diagnostics, do not monkeypatch the pre-cluster vacuum "
            "crystal-free helper. This reproduces the known hang."
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

        rank, size = mpi_rank_size()
        if size <= 1 and os.environ.get("MMML_MPI_NP"):
            return rank, max(1, int(os.environ["MMML_MPI_NP"]))
        return rank, size
    except Exception:
        return 0, max(1, int(os.environ.get("MMML_MPI_NP", "1")))


def _print_dry_run(args: argparse.Namespace) -> None:
    cmd = [
        f"MMML_MPI_NP={2 if args.allow_mpi_size_gt1 else 1}",
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


def _rank_log(msg: str) -> None:
    rank, size = _mpi_info()
    print(f"[domdec-smoke rank {rank}/{size}] {msg}", flush=True)


def _run_domdec_command(command: str | None) -> None:
    if not command:
        _rank_log("DOMDEC command: skipped")
        return
    import pycharmm.lingo as lingo

    _rank_log(f"DOMDEC command: {command!r} begin")
    lingo.charmm_script(str(command))
    _rank_log(f"DOMDEC command: {command!r} done")


def _domdec_command_turns_on(command: str | None) -> bool:
    if not command:
        return False
    words = str(command).lower().replace("=", " ").split()
    return "domdec" in words and "on" in words and "off" not in words


def _known_domdec_order_issue(residue: str) -> str | None:
    residue = residue.upper()
    if residue == "ACO":
        return (
            "ACO is ordered as O1,C1,C2,C3,H21,H22,H23,H31,H32,H33 in the generated "
            "PSF. CHARMM DOMDEC groupxfast requires hydrogens to be adjacent to the "
            "heavy atom they are bonded to, so this dies before testing MLpot/JAX."
        )
    if residue == "MEOH":
        return (
            "MEOH is ordered as CB,OG,HG1,HB1,HB2,HB3 in the generated PSF. The CB "
            "hydrogens are separated from CB by OG/HG1, which trips DOMDEC groupxfast."
        )
    return None


def _skip_vacuum_crystal_free_for_mpi_cluster_build() -> None:
    """Avoid the known np>1 hang in the fresh-process cluster-build reset."""
    import mmml.interfaces.pycharmmInterface.mlpot.setup as mlpot_setup

    def _skip() -> None:
        _rank_log("skipping pre-cluster prepare_charmm_vacuum/crystal free for np>1")

    mlpot_setup.prepare_charmm_vacuum = _skip


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
    if size > 1 and not args.allow_mpi_size_gt1:
        print(
            f"Refusing np={size} DOMDEC+MLpot smoke without --allow-mpi-size-gt1. "
            "The current Tier 3 diagnostic may hang during CHARMM setup.",
            file=sys.stderr,
            flush=True,
        )
        return 4
    if size > 1 and not args.no_skip_vacuum_crystal_free_on_mpi:
        os.environ["MMML_SKIP_CHARMM_RESET_BLOCK"] = "1"
        _rank_log("skipping CHARMM reset_block for np>1 diagnostic")
        _skip_vacuum_crystal_free_for_mpi_cluster_build()

    domdec_command = None if args.no_domdec_command else args.domdec_command
    if _domdec_command_turns_on(domdec_command) and not args.allow_domdec_order_risk:
        issue = _known_domdec_order_issue(str(args.residue))
        if issue:
            print("Refusing active-DOMDEC smoke for this residue atom order.", file=sys.stderr)
            print(issue, file=sys.stderr)
            print(
                "Use --residue OCOH --n-molecules 1 for the default DOMDEC-order smoke, "
                "or pass --allow-domdec-order-risk to reproduce the CHARMM failure.",
                file=sys.stderr,
            )
            return 3

    _rank_log("importing functionality helpers")
    from _common import (
        all_atom_selection,
        build_ase_cluster,
        charmm_energy_row,
        check_mlpot_symbols,
        resolve_checkpoint,
    )
    _rank_log("imported functionality helpers")

    if rank == 0:
        print("\n================================")
        print("Tier 3 DOMDEC + MLpot ENER smoke")
        print("================================")
        print(
            f"MPI size={size}; residue={args.residue}; n_molecules={args.n_molecules}; "
            f"box={float(args.box_side):.3f} Å",
            flush=True,
        )

    _rank_log("checking MLpot symbols")
    missing = check_mlpot_symbols()
    if missing:
        print(f"FAIL: missing libcharmm MLpot symbols: {missing}", file=sys.stderr)
        return 1
    _rank_log("MLpot symbols ok")

    _rank_log("resolving checkpoint")
    ckpt = resolve_checkpoint(args.checkpoint)
    _rank_log("building CHARMM/ASE cluster begin")
    z, r = build_ase_cluster(args.residue, args.n_molecules, args.spacing)
    _rank_log("building CHARMM/ASE cluster done")
    n_atoms = len(z)
    if rank == 0:
        print(f"Cluster: {n_atoms} atoms; checkpoint={ckpt}", flush=True)

    _rank_log("importing PyCHARMM modules")
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import ase
    import pycharmm
    import pycharmm.energy as energy
    _rank_log("imported PyCHARMM modules")

    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment

    _rank_log("PBC setup begin")
    setup_charmm_environment(use_pbc=True, cubic_box_side_A=float(args.box_side))
    _rank_log("PBC setup done")
    # Do not call the vacuum nbonds helper here: it runs ``crystal free`` and
    # leaves DOMDEC with a zero/NaN box.
    _run_domdec_command(domdec_command)

    _rank_log("loading checkpoint/model begin")
    from _common import load_physnet_for_cluster

    params, model = load_physnet_for_cluster(ckpt, n_atoms)
    model.natoms = n_atoms
    atoms = ase.Atoms(numbers=z, positions=r)
    _rank_log("loading checkpoint/model done")

    from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_pyc

    _rank_log("building PyCHARMM ML model begin")
    pyc_model = get_pyc(params, model, atoms)
    _rank_log("building PyCHARMM ML model done")
    mlpot = None
    try:
        _rank_log("Registering MLpot begin")
        mlpot = pycharmm.MLpot(
            ml_model=pyc_model,
            ml_Z=list(z),
            ml_selection=all_atom_selection(),
            ml_charge=0,
            ml_fq=True,
        )
        _rank_log("Registering MLpot done")
        _rank_log("CHARMM ENER begin")
        energy.show()
        _rank_log("CHARMM ENER done")
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
