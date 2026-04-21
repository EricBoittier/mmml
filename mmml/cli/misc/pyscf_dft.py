#!/usr/bin/env python
"""
CLI for GPU-accelerated DFT calculations via PySCF/gpu4pyscf.

Usage:
    mmml pyscf-dft --mol "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0" --energy
    mmml pyscf-dft --mol water.xyz --energy --gradient --output results.hdf5
    mmml pyscf-dft --mol water.xyz --energy --hessian --harmonic --thermo

Requires: gpu4pyscf, pyscf (GPU/quantum environment)
"""

import sys
import time
from pathlib import Path


def _normalize_output_base(path: Path) -> Path:
    """Return output base path without .npz/.h5/.hdf5 suffix."""
    if path.suffix.lower() in {".npz", ".h5", ".hdf5"}:
        return path.with_suffix("")
    return path


def _ensure_output_does_not_exist(path: Path) -> tuple[bool, list[Path]]:
    """Return whether output base is available and list conflicting files."""
    base = _normalize_output_base(path)
    conflicts = [p for p in (base.with_suffix(".npz"), base.with_suffix(".h5")) if p.exists()]
    return len(conflicts) == 0, conflicts


def main() -> int:
    """Run pyscf-dft CLI."""
    t0 = time.perf_counter()
    try:
        from mmml.interfaces.pyscf4gpuInterface.calcs import (
            parse_args,
            process_calcs,
            compute_dft,
            save_pyscf_results,
        )
    except ModuleNotFoundError as e:
        if "cupy" in str(e).lower() or "gpu4pyscf" in str(e).lower():
            print("Error: pyscf-dft requires cupy and gpu4pyscf.", file=sys.stderr)
            print("Install with: uv sync --extra quantum-gpu", file=sys.stderr)
            print("Or: uv pip install cupy-cuda13x gpu4pyscf-cuda13x", file=sys.stderr)
            return 1
        raise

    args = parse_args()
    calcs, extra = process_calcs(args)

    if not calcs:
        print("Error: At least one calculation flag is required.", file=sys.stderr)
        print("Use --energy, --gradient, --hessian, --optimize, etc.", file=sys.stderr)
        return 1

    output_base = _normalize_output_base(Path(args.output))
    available, conflicts = _ensure_output_does_not_exist(Path(args.output))
    if not available:
        conflicts_str = ", ".join(str(p) for p in conflicts)
        print("Error: Output target already exists.", file=sys.stderr)
        print(f"Conflicting files: {conflicts_str}", file=sys.stderr)
        print(
            "Choose a different --output path or remove existing files.",
            file=sys.stderr,
        )
        return 1

    output = compute_dft(args, calcs, extra)
    save_pyscf_results(str(output_base), output)
    elapsed = time.perf_counter() - t0
    print(f"Results saved to {output_base}.npz (ML keys) and {output_base}.h5 (all arrays)")
    print(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
