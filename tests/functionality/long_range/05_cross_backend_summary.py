#!/usr/bin/env python3
"""Step 5: summary table — MIC, jax-pme (Ewald/PME/P3M), ScaFaCoS on shared systems."""

from __future__ import annotations

import sys

from _common import (
    cscl_crystal,
    evaluate_backend,
    have_jax_pme_package,
    scafacos_integration_enabled,
    ion_dimer_system,
    print_header,
    random_neutral_cluster,
)


def main() -> int:
    print_header("Cross-backend Coulomb energy summary (kcal/mol)")
    systems = [
        ion_dimer_system(separation_A=5.0, box_length_A=30.0),
        cscl_crystal(box_length_A=10.0),
        random_neutral_cluster(n_atoms=8, box_length_A=12.0, seed=1),
    ]

    backends = ["mic", "mic_trunc", "jax_ewald", "jax_pme", "jax_p3m"]
    if scafacos_integration_enabled():
        backends.append("scafacos")

    header = f"{'system':<22}" + "".join(f"{b:>14}" for b in backends)
    print(header)
    print("-" * len(header))

    for system in systems:
        row = f"{system.name:<22}"
        for backend in backends:
            try:
                if backend.startswith("jax_") and not have_jax_pme_package():
                    row += f"{'n/a':>14}"
                    continue
                if backend == "scafacos" and not scafacos_integration_enabled():
                    row += f"{'n/a':>14}"
                    continue
                e = evaluate_backend(system, backend).energy_kcalmol  # type: ignore[arg-type]
                row += f"{e:14.4f}"
            except Exception as exc:
                row += f"{'ERR':>14}"
                print(f"  {system.name} / {backend}: {exc}", file=sys.stderr)
        print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
