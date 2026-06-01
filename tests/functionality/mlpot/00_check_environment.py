#!/usr/bin/env python3
"""Step 0: verify PyCHARMM, MLpot symbols, JAX/ASE, and checkpoint."""

from __future__ import annotations

import sys

from _common import PROJECT_ROOT, check_mlpot_symbols, print_header, resolve_checkpoint


def main() -> int:
    print_header("MLpot environment check")
    print(f"Project root: {PROJECT_ROOT}")

    ok = True

    for name in ("numpy", "jax", "e3x", "ase"):
        try:
            __import__(name)
            print(f"  OK  import {name}")
        except Exception as exc:
            print(f"  FAIL import {name}: {exc}")
            ok = False

    try:
        import pycharmm
        from pycharmm import MLpot  # noqa: F401

        print(f"  OK  import pycharmm (MLpot from {MLpot.__module__})")
    except Exception as exc:
        print(f"  FAIL import pycharmm / MLpot: {exc}")
        ok = False
        return 1

    missing = check_mlpot_symbols()
    if missing:
        print(f"  FAIL CHARMM lib missing symbols: {missing}")
        print("       Your libcharmm build may predate MLpot support.")
        ok = False
    else:
        print("  OK  libcharmm mlpot_set_func / mlpot_set_properties / mlpot_unset")

    try:
        ckpt = resolve_checkpoint()
        print(f"  OK  checkpoint: {ckpt}")
    except FileNotFoundError as exc:
        print(f"  WARN {exc}")
        ok = False

    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401

        print("  OK  mmml import_pycharmm (CHARMM_HOME / CHARMM_LIB_DIR)")
    except Exception as exc:
        print(f"  FAIL import_pycharmm: {exc}")
        ok = False

    if ok:
        print("\nEnvironment looks ready for scripts 01–03.")
        return 0
    print("\nFix failures above before running 01_callback_vs_ase_no_charmm.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
