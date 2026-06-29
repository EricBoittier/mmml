#!/usr/bin/env python3
"""Minimal PyCHARMM cooperative READ gate for ``np>1`` bisect (cluster only).

No JAX, ASE, Rich, or MLpot — isolates ``eval_charmm_script`` topology I/O hangs.

Usage (node09)::

    MMML_MPI_NP=1 ./scripts/run_mpi_pycharmm_read_gate.sh
    MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode psf-crd
    MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode stream-inp
    MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode restart

Pass: exit 0 and ``PASS read_gate: n_atoms=100`` on all ranks (DCM:20 prebuilt).
Fail/hang: note the last ``[step rank */N] begin`` line before the stall.
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("psf-crd", "stream-inp", "restart"),
        default="psf-crd",
        help="Bootstrap path to exercise (default: psf-crd).",
    )
    p.add_argument(
        "--psf",
        type=Path,
        default=Path("artifacts/domdec_spatial_smoke/dcm_20mer.psf"),
    )
    p.add_argument(
        "--crd",
        type=Path,
        default=Path("artifacts/domdec_spatial_smoke/dcm_20mer.crd"),
    )
    p.add_argument(
        "--res",
        type=Path,
        default=Path("artifacts/domdec_spatial_smoke/dcm_20mer.res"),
    )
    p.add_argument(
        "--box-side",
        type=float,
        default=40.0,
        help="Optional cubic box for crystal define/build after READ.",
    )
    p.add_argument(
        "--with-crystal",
        action="store_true",
        help="Run crystal define/build after PSF/CRD load.",
    )
    p.add_argument(
        "--expected-n-atoms",
        type=int,
        default=100,
        help="Expected atom count after successful load (DCM:20 default).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print recommended commands and exit.",
    )
    return p.parse_args()


def _mpi_info() -> tuple[int, int]:
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

        return mpi_rank_size()
    except Exception:
        return 0, max(1, int(os.environ.get("MMML_MPI_NP", "1")))


def _log(step: str, msg: str) -> None:
    rank, size = _mpi_info()
    print(f"[{step} rank {rank}/{size}] {msg}", flush=True)


def _print_dry_run(args: argparse.Namespace) -> None:
    psf = args.psf
    for np in (1, 2, 4):
        print(f"# np={np} mode={args.mode}")
        print(
            f"MMML_MPI_NP={np} ./scripts/run_mpi_pycharmm_read_gate.sh "
            f"--mode {args.mode} --psf {psf}"
        )
        print()
    print("# Native CHARMM control (requires CHARMM_EXE):")
    print(
        "PSF=artifacts/domdec_spatial_smoke/dcm_20mer.psf \\"
    )
    print(
        "CRD=artifacts/domdec_spatial_smoke/dcm_20mer.crd \\"
    )
    print(
        "MMML_MPI_NP=4 ./scripts/mmml-charmm-mpirun.sh "
        "$CHARMM_EXE -i /tmp/read_gate.inp -o /tmp/read_gate.out"
    )


def main() -> int:
    args = _parse_args()
    if args.dry_run:
        _print_dry_run(args)
        return 0

    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        bootstrap_topology_mpi,
        configure_mpi_bootstrap_env,
        mpi4py_openmpi_mismatch,
        prepare_serial_charmm_mpi_env,
        sync_import_pycharmm_for_bootstrap,
    )

    prepare_serial_charmm_mpi_env()
    ok, msg = mpi4py_openmpi_mismatch()
    if not ok:
        print(f"FAIL: {msg}", file=sys.stderr)
        return 1

    rank, size = _mpi_info()
    configure_mpi_bootstrap_env()
    _log("gate", f"mode={args.mode} np={size} psf={args.psf.resolve()}")
    sync_import_pycharmm_for_bootstrap(tag="read_gate")

    crystal = float(args.box_side) if args.with_crystal else None
    try:
        n_atoms = bootstrap_topology_mpi(
            args.psf,
            args.crd,
            restart_path=args.res,
            mode=args.mode,
            crystal_side_A=crystal,
            log_fn=_log,
        )
    except Exception as exc:
        print(
            f"FAIL rank {rank}/{size}: bootstrap raised {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    expected = int(args.expected_n_atoms)
    if n_atoms != expected:
        print(
            f"FAIL rank {rank}/{size}: n_atoms={n_atoms} expected {expected}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    if rank == 0:
        print(
            f"PASS read_gate: mode={args.mode} np={size} n_atoms={n_atoms}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
