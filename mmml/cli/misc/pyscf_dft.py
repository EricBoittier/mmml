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


def _next_available_output_base(path: Path) -> Path:
    """Return base path whose .npz/.h5 outputs do not already exist."""
    base = _normalize_output_base(path)
    candidate = base
    idx = 1
    while candidate.with_suffix(".npz").exists() or candidate.with_suffix(".h5").exists():
        candidate = base.with_name(f"{base.name}_{idx}")
        idx += 1
    return candidate


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

    output = compute_dft(args, calcs, extra)
    output_base = _next_available_output_base(Path(args.output))
    save_pyscf_results(str(output_base), output)
    elapsed = time.perf_counter() - t0
    if output_base != _normalize_output_base(Path(args.output)):
        print(f"Output exists, using {output_base} instead of {args.output}")
    print(f"Results saved to {output_base}.npz (ML keys) and {output_base}.h5 (all arrays)")
    print(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
