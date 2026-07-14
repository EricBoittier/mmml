"""``mmml doctor``: is this machine ready to run MMML?

One command that answers the install question. It reuses the existing interface
checks in :mod:`mmml.cli.run.health_check` rather than reimplementing them, and
adds the two things that actually confuse people during setup: which CHARMM
paths were auto-discovered, and whether ``libcharmm`` is newer than the
``api_func.F90`` it was built from.

Exits nonzero when the environment cannot run MM/ML work, so it is usable as a
CI or campaign gate.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path


@contextlib.contextmanager
def _stdout_to_stderr():
    """Send everything written to fd 1 to fd 2 for the duration of the block.

    CHARMM's Fortran layer writes its banner and timer profile straight to file
    descriptor 1, bypassing ``sys.stdout`` entirely, so
    ``contextlib.redirect_stdout`` cannot catch it. Only a ``dup2`` on the
    underlying descriptor keeps ``--json`` parseable.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)

# The install-relevant subset: MPI and checkpoints are runtime concerns, not
# "did the build work" concerns, so they are opt-in via --mpi / --checkpoint.
INSTALL_CHECKS = ("core", "jax", "charmm", "mlpot", "packmol")


def _charmm_section() -> tuple[list[str], bool]:
    """Resolved CHARMM paths and libcharmm freshness."""
    from mmml.interfaces.pycharmmInterface.charmm_paths import resolve_charmm_paths
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import mlpot_limits_status

    lines: list[str] = []
    home, lib_dir = resolve_charmm_paths()

    lines.append("CHARMM paths (auto-discovered; no env vars required)")
    lines.append(f"  CHARMM_HOME    = {home or '(not found)'}")
    lines.append(f"  CHARMM_LIB_DIR = {lib_dir or '(not found)'}")
    lines.append("")

    status = mlpot_limits_status()
    lines.append(status.message())

    ok = bool(home and lib_dir) and status.libcharmm is not None
    if not ok:
        lines.append("")
        lines.append("  -> Build the native side with:  make install-native")
    return lines, ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml doctor",
        description="Check that this machine can run MMML (Python, JAX, CHARMM, Packmol).",
    )
    parser.add_argument(
        "--json", action="store_true", help="machine-readable report on stdout"
    )
    parser.add_argument(
        "--require-gpu", action="store_true", help="fail unless JAX sees a GPU"
    )
    parser.add_argument(
        "--mpi", action="store_true", help="also check OpenMPI / mpi4py wiring"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="also validate an ML checkpoint"
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Populate CHARMM_HOME / CHARMM_LIB_DIR from the discovery chain before the
    # checks read them, so `doctor` reports what a real run would actually use.
    from mmml.interfaces.pycharmmInterface.charmm_paths import bootstrap_charmm_env
    from mmml.cli.run.health_check import (
        HealthReport,
        render_health_report,
        run_health_check,
    )

    bootstrap_charmm_env()

    checks = list(INSTALL_CHECKS)
    if args.mpi:
        checks.append("mpi")
    if args.checkpoint is not None:
        checks.append("checkpoint")

    with _stdout_to_stderr():
        report: HealthReport = run_health_check(
            only=checks,
            checkpoint=args.checkpoint,
            require_gpu=args.require_gpu,
            strict=args.strict,
        )
        charmm_lines, charmm_ok = _charmm_section()

    if args.json:
        payload = report.to_dict()
        from mmml.interfaces.pycharmmInterface.charmm_paths import resolve_charmm_paths
        from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import (
            mlpot_limits_status,
        )

        home, lib_dir = resolve_charmm_paths()
        status = mlpot_limits_status()
        payload["charmm"] = {
            "CHARMM_HOME": home or None,
            "CHARMM_LIB_DIR": lib_dir or None,
            "libcharmm": str(status.libcharmm) if status.libcharmm else None,
            "max_Nml": status.max_nml,
            "max_Npr": status.max_npr,
            "source": status.source,
        }
        payload["ok"] = bool(report.ok and charmm_ok)
        print(json.dumps(payload, indent=2))
        return 0 if payload["ok"] else 1

    print(render_health_report(report))
    print()
    print("\n".join(charmm_lines))
    print()

    ok = report.ok and charmm_ok
    print(f"doctor: {'OK -- this machine can run MMML' if ok else 'FAIL'}")
    if not ok:
        print("Install: make install       (Python deps via uv)")
        print("         make install-native (libcharmm + packmol)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
