#!/usr/bin/env python3
"""Runtime sentinel test for CHARMM's raw velocity-buffer C ABI."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np


def probe(library: Path, n: int = 17) -> dict:
    handle = ctypes.CDLL(str(library.resolve()))
    fn = handle.dynamics_velocity_buffer_probe
    fn.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 6
    fn.restype = ctypes.c_int

    base = np.arange(n, dtype=np.float64)
    inputs = [
        np.ascontiguousarray(0.125 + base * 1.25),
        np.ascontiguousarray(-9.5 - base * 0.75),
        np.ascontiguousarray(1000.0 + base * 0.03125),
    ]
    outputs = [np.full(n, np.nan, dtype=np.float64) for _ in range(3)]
    pointers = [array.ctypes.data_as(ctypes.c_void_p) for array in (*inputs, *outputs)]
    status = int(fn(n, *pointers))
    null_status = int(fn(n, *([None] * 6)))
    matches = [bool(np.array_equal(source, target)) for source, target in zip(inputs, outputs)]
    maximum_errors = [
        float(np.max(np.abs(source - target))) for source, target in zip(inputs, outputs)
    ]
    return {
        "schema_version": 1,
        "library": str(library.resolve()),
        "n": n,
        "status": status,
        "null_pointer_status": null_status,
        "component_matches": dict(zip(("x", "y", "z"), matches)),
        "component_max_abs_error": dict(zip(("x", "y", "z"), maximum_errors)),
        "passed": status == 1 and null_status == -1 and all(matches),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--n", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/charmm_dynamics_velocity_abi.json"),
    )
    args = parser.parse_args()
    report = probe(args.library, args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
