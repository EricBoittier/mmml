# `mmml umbrella-sample`

Batched distance umbrella NVT sampling with a PhysNet / SpookyNet checkpoint.


## Usage

```bash
mmml umbrella-sample --help
```

## Options

```text
usage: mmml umbrella-sample [-h] [--config CONFIG] [--checkpoint CHECKPOINT]
                            [--structure STRUCTURE] [--output-dir OUTPUT_DIR]
                            [--atoms I,J] [--targets TARGETS]
                            [--xi-min XI_MIN] [--xi-max XI_MAX]
                            [--n-windows N_WINDOWS] [--k K]
                            [--temperature TEMPERATURE] [--timestep TIMESTEP]
                            [--nsteps NSTEPS] [--printfreq PRINTFREQ]
                            [--savefreq SAVEFREQ] [--seed SEED] [--no-ema]
                            [--overwrite]
```

Batched distance umbrella sampling with a PhysNet / SpookyNet checkpoint via
JAX-MD NVT Nose-Hoover.

### Input & configuration

- `--config` — YAML/JSON `UmbrellaConfig`; CLI flags override file values
- `--checkpoint` — PhysNet / SpookyNet checkpoint
- `--structure` — Input XYZ / structure
- `--output-dir` / `-o` — Snapshots, trajectories, summary

### Collective variable

- `--atoms I,J` — 0-based atom indices for the distance CV
- `--targets` — Comma-separated umbrella centers ξ₀ (Å)
- `--xi-min` / `--xi-max` / `--n-windows` — Linear grid if `--targets` omitted
- `--k` — Shared harmonic force constant (eV/Å², default 10)

### Dynamics

- `--temperature` — NVT temperature in K (default 300)
- `--timestep` — Timestep in fs (default 0.5)
- `--nsteps` — Number of NVT steps (default 1000)
- `--printfreq` / `--savefreq` — Print / snapshot intervals
- `--seed` — PRNG seed
- `--no-ema` — Prefer non-EMA checkpoint params
- `--overwrite` — Allow non-empty output directory

## See also

- Overview: [Batched umbrella sampling](../umbrella.md)
- Post-process: [`mmml umbrella-mbar`](umbrella-mbar.md)
