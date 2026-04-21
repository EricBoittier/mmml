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
import re


def _normalize_output_base(path: Path) -> Path:
    """Return output base path without .npz/.h5/.hdf5 suffix."""
    if path.suffix.lower() in {".npz", ".h5", ".hdf5"}:
        return path.with_suffix("")
    return path


def _next_available_output_base(path: Path) -> Path:
    """Return base path whose .npz/.h5 outputs do not already exist.

    Uses one directory scan instead of repeated `.exists()` probes to avoid
    slow linear filesystem checks when many prior outputs are present.
    """
    base = _normalize_output_base(path)
    parent = base.parent
    stem = base.name

    try:
        existing_names = {p.name for p in parent.iterdir()}
    except OSError:
        # Fall back to the original direct probe behavior if directory listing
        # is unavailable (permissions, transient IO errors, etc).
        candidate = base
        idx = 1
        while candidate.with_suffix(".npz").exists() or candidate.with_suffix(".h5").exists():
            candidate = base.with_name(f"{stem}_{idx}")
            idx += 1
        return candidate

    base_npz = f"{stem}.npz"
    base_h5 = f"{stem}.h5"
    if base_npz not in existing_names and base_h5 not in existing_names:
        return base

    pat = re.compile(rf"^{re.escape(stem)}_(\d+)\.(?:npz|h5)$")
    max_idx = 0
    for name in existing_names:
        m = pat.match(name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    return base.with_name(f"{stem}_{max_idx + 1}")


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
