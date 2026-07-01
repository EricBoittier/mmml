# `mmml efield-md`

MD with external electric-field PhysNet.


## Usage

```bash
mmml efield-md --help
```

## Options

```text
usage: mmml efield-md [-h] [--backend {ase,jax}]

Run MD with the electric-field equivariant model (trained via mmml ef-train).
Default backend is ASE; use --backend jax for the fully JIT-compiled integrator.

options:
  -h, --help            show this help message and exit
  --backend, -b {ase,jax}
                        ase (default): ASE integrator; jax: JIT jax_md single-
                        replica runner

Backends: ase ASE Langevin / Velocity-Verlet with AseCalculatorEF. Supports
electric-field ramps and --n-replicas > 1 (GPU-batched JAX loop, multi-replica
HDF5 output). jax Single-replica NVE / Langevin in one compiled loop; writes an
ASE .traj after the run. Common forwarded flags (both backends): --params,
--config, --data, --xyz, --index, --electric-field, --thermostat, --temperature,
--friction, --dt, --steps, --output, --seed, --optimize, --optimizer, --fmax,
--opt-steps, --maxstep, --save-charges / --no-save-charges (default: do not save
atomic charges) ASE-only examples: --n-replicas 4 Independent replicas (batched
on GPU when > 1) --ramp-field-axis z --ramp-field-peak 0.01 Triangular E-field
ramp on one axis --log-interval 100 --traj-interval 1 JAX-only examples: --save-
interval 10 --print-interval 5 --field-scale 0.001 --dipole-field-coupling
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
