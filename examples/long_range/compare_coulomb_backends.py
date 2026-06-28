#!/usr/bin/env python3
"""Example: compare MIC, jax-pme, and ScaFaCoS on simple charge systems.

Run from repo root:

    JAX_PLATFORMS=cpu python examples/long_range/compare_coulomb_backends.py

Requires jax-pme (core MMML dependency). ScaFaCoS is optional (set SCAFACOS_LIB).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "functionality" / "long_range"))

from _common import (  # noqa: E402
    cscl_crystal,
    describe_environment,
    evaluate_backend,
    have_scafacos_library,
    ion_dimer_system,
    print_header,
)


def main() -> None:
    print_header("MMML Coulomb backend comparison")
    print(describe_environment())
    print()

    systems = [
        ion_dimer_system(separation_A=5.0, box_length_A=30.0),
        cscl_crystal(box_length_A=10.0),
    ]
    backends = ["mic", "mic_trunc", "jax_ewald", "jax_pme", "jax_p3m"]
    if have_scafacos_library():
        backends.append("scafacos")

    for system in systems:
        print(f"\n{system.name} (L={system.box_length_A} Å)")
        for backend in backends:
            try:
                result = evaluate_backend(system, backend)  # type: ignore[arg-type]
                print(f"  {backend:12s}  E = {result.energy_kcalmol:12.6f} kcal/mol")
            except Exception as exc:
                print(f"  {backend:12s}  ERROR: {exc}")


if __name__ == "__main__":
    main()
