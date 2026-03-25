"""CLI entry for molecular dynamics with the trained EF (electric-field) model.

Dispatches to :mod:`mmml.models.EF.ase_md` (default) or :mod:`mmml.models.EF.jax_md`.
All arguments after optional ``--backend`` / ``-b`` are forwarded unchanged.

Examples:
    mmml ef-md --params ./ef_run/params.json --data splits/train.npz --steps 5000
    mmml ef-md --backend jax --params ./ef_run/params.json --data splits/train.npz \\
        --thermostat langevin --temperature 300 --steps 10000 --output traj.traj
    mmml ef-md --backend ase --params ./ef_run/params.json --xyz mol.xyz \\
        --n-replicas 4 --output run.traj

For full backend-specific flags (replicas, field ramp, save intervals, etc.):
    python -m mmml.models.EF.ase_md --help
    python -m mmml.models.EF.jax_md --help
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mmml ef-md",
        description=(
            "Run MD with the electric-field equivariant model (trained via mmml ef-train). "
            "Default backend is ASE; use --backend jax for the fully JIT-compiled integrator."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backends:
  ase  ASE Langevin / Velocity-Verlet with AseCalculatorEF. Supports electric-field
       ramps and --n-replicas > 1 (GPU-batched JAX loop, multi-replica HDF5 output).
  jax  Single-replica NVE / Langevin in one compiled loop; writes an ASE .traj after the run.

Common forwarded flags (both backends):
  --params, --config, --data, --xyz, --index, --electric-field,
  --thermostat, --temperature, --friction, --dt, --steps, --output, --seed,
  --optimize, --optimizer, --fmax, --opt-steps, --maxstep, --save-charges

ASE-only examples:
  --n-replicas 4          Independent replicas (batched on GPU when > 1)
  --ramp-field-axis z --ramp-field-peak 0.01   Triangular E-field ramp on one axis
  --log-interval 100 --traj-interval 1

JAX-only examples:
  --save-interval 10 --print-interval 5
  --field-scale 0.001 --dipole-field-coupling
""",
    )
    parser.add_argument(
        "--backend",
        "-b",
        choices=["ase", "jax"],
        default="ase",
        help="ase (default): ASE integrator; jax: JIT jax_md single-replica runner",
    )
    _args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    if _args.backend == "jax":
        from mmml.models.EF.jax_md import main as md_main

        md_main()
    else:
        from mmml.models.EF.ase_md import main as md_main

        md_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
