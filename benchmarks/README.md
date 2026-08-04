# mmml speed benchmarks (airspeed velocity)

A performance suite for the code **this repository implements** — the PhysNet /
SpookyNet JAX models, the CHARMM-compatible MM kernels, host neighbour
construction, SHAKE/RATTLE, the `mmml.md` driver, the production ML calculator,
and the training-batch and trajectory-I/O paths. Third-party libraries appear
only underneath an mmml entry point; nothing here benchmarks JAX or ASE for
their own sake.

Runs are **manual** and typically happen on a GPU node. This suite is not wired
into CI: a meaningful timing needs a quiet machine with a known accelerator, and
a shared CI runner is neither.

---

## Quick start

```bash
uv sync --extra dev
bash benchmarks/run_bench.sh
```

That runs everything against the current checkout, then regenerates the HTML
report into `benchmarks/html/`. To view it:

```bash
uv run asv preview
```

Narrow the run by passing an asv `--bench` regex — a module, a class, or a
single method:

```bash
bash benchmarks/run_bench.sh bench_md_driver
bash benchmarks/run_bench.sh MDSystemSize
bash benchmarks/run_bench.sh 'MMNonbonded.time_forces'
```

On a cluster:

```bash
sbatch benchmarks/slurm_bench_gpu.sh
sbatch --export=ALL,BENCH_PATTERN=bench_ml_physnet benchmarks/slurm_bench_gpu.sh
```

The Slurm job refuses to run if JAX comes up on the CPU backend, so a
misconfigured environment fails fast instead of quietly producing CPU numbers
under a GPU machine name.

---

## What is measured

| Module | Covers |
| --- | --- |
| `bench_ml_physnet.py` | PhysNet forward and force backward; scaling in atoms, feature width, message-passing depth, and `max_degree`; the charge/electrostatics head; ZBL repulsion; SpookyPhysNet; XLA compile time |
| `bench_mm_energy.py` | `mm_nonbonded` switched VDW + Coulomb (energy and gradient); MIC vs. native Ewald; the host `nonbonded_energy_and_forces` reference; `hybrid_ewald_coulomb_energy_with_cell` |
| `bench_neighbors.py` | `_build_pair_indices` (Vesin dispatcher) vs. the chunked-NumPy fallback; `get_intermolecular_pairs`; a full driver refresh; the Verlet-skin cache; `pad_indices` |
| `bench_constraints.py` | `shake_positions`, `rattle_velocities`, `constraint_residuals`, and the shipped `constrained_nve` step against plain `jax_md.simulate.nve` |
| `bench_md_driver.py` | End-to-end `JaxmdDriver` throughput in **ns/day**: box size, ensemble (NVE/NVT/NpT), jitted block size, and Verlet skin width |
| `bench_calculator.py` | The real bundled checkpoint through `setup_calculator`: ASE calculator vs. jitted `spherical_fn`, cold-start cost, and vmapped dimer scans |
| `bench_data.py` | `prepare_batches_jit` vs. `prepare_batches_fast` (with and without `pair_cache`), rotation augmentation, `_pair_indices`, and DCD write/read |

Benchmarks named `time_*` are wall-clock timings. Benchmarks named `track_*`
report a domain quantity instead — `ns/day`, pairs in cutoff, k-vector count,
rebuild fraction, compile seconds — so a result is readable without knowing the
step count or the box.

### The headline number

`bench_md_driver.MDSystemSize.track_ns_per_day` is what a simulation actually
cares about. Everything else in the suite exists to explain it.

---

## Conventions that make the numbers trustworthy

**Every timing blocks on the device.** JAX dispatch is asynchronous, so a
benchmark that does not call `jax.block_until_ready` measures Python overhead
rather than the kernel. `_common.block()` wraps this and every `time_*` method
goes through it.

**Compilation is warmed out of the timed region.** Each `setup()` runs the
jitted function once before asv starts sampling; XLA compile time is measured
separately and deliberately, by `track_compile_*` benchmarks that call
`jax.clear_caches()` first.

**Precision is a process-global, fixed from the environment.**
`jax_enable_x64` cannot be flipped per benchmark without leaking into whatever
runs next in the same worker, so `_common` sets it once from `MMML_BENCH_X64`
(default `1`, matching `examples/md_cpu/_env.sh` and the production MD path).
Results are only comparable between runs that agree on it — changing it starts a
new series rather than extending the old one.

**Fixtures are plain NumPy.** Geometry and force-field construction build the
*input* to the code under test, so they must be deterministic and free of
anything that could show up in a timing. `water_box()` places rigid TIP3
geometries on a jittered lattice at liquid density; `synthetic_ff_params()`
supplies per-molecule-neutral charges and plausible CHARMM-scale LJ values,
which keeps the MM benchmarks runnable on machines with no CHARMM build. The MM
kernels' cost depends on pair count and cutoffs, not on the exact epsilon, so
this is an honest timing rather than a convenient one.

**Box sizes respect the minimum-image convention.** The MM and neighbour
classes start at 512 waters because the 12 Å production cutoff needs
`L >= 2 × cutoff`. Below that, `_build_pair_indices` drops off Vesin onto
chunked NumPy and the pair list degenerates to nearly all pairs — a regime no
real run is in, which would make the small end of every scaling curve
meaningless.

**A missing dependency skips, it does not fail.** Heavy imports live inside
`setup()` and raise `_common.skip(...)`, which asv treats as a skip. One
unavailable optional dependency therefore costs one benchmark, not the whole
module.

---

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `MMML_BENCH_X64` | `1` | float64 (production MD default). Set `0` for a float32 series. |
| `MMML_BENCH_CKPT` / `MMML_CKPT` | bundled `examples/ckpts_json/DESdimers_params.json` | Checkpoint for `bench_calculator` |
| `JAX_PLATFORMS` | auto | `cpu` / `cuda`; the Slurm job pins `cuda` and verifies it |
| `OMP_NUM_THREADS` | `1` in the runners | Keeps NumPy and JAX's CPU backend from fighting over cores |

---

## How results accumulate

asv writes one JSON per commit under `benchmarks/results/<machine>/`, so
repeated runs on *different* commits build history rather than replacing it.
Re-running the **same** commit replaces that commit's entry; set
`BENCH_APPEND_SAMPLES=1` to merge new samples into the existing one instead.

`benchmarks/html/` is fully regenerated from `benchmarks/results/` by
`asv publish`, so it is safe to delete and pointless to edit. Commit the
`results/` JSON — `asv publish`'s regression view is only as long as the history
that is checked in.

`asv.conf.json` lists only `main` under `branches`, so a run on a feature branch
records its JSON but `asv publish` reports

```
Couldn't find <hash> in branches (main)
```

and leaves that point off the graphs. This is expected: the report tracks `main`
over time, and a branch's numbers join it when the branch merges. To plot a
branch before then, add it to `branches` locally — don't commit that, or the
report grows a dead series once the branch is gone.

Because `environment_type` is `existing`, asv benchmarks the interpreter that
invokes it rather than building a fresh environment per commit. Rebuilding a
JAX + CUDA + libcharmm environment for every commit is not viable, and the runs
are manual anyway. The practical consequence: **results describe the working
tree at the time of the run**, labelled with its commit hash. Run on a clean
checkout if you intend to keep the numbers.

---

## Adding a benchmark

1. Put heavy imports inside `setup()`, and raise `_common.skip(...)` when a
   dependency is missing.
2. Build fixtures in `setup()`; the timed method should do nothing but call the
   code under test.
3. Wrap any JAX result in `_common.block()`.
4. Warm the JIT at the end of `setup()`.
5. Prefer a `track_*` benchmark in domain units when "seconds" would not be the
   number someone actually wants.
