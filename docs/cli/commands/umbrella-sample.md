# `mmml umbrella-sample`

Batched distance umbrella NVT sampling (PhysNet/SpookyNet).


## Usage

```bash
mmml umbrella-sample --help
```

## Options

```text
usage: mmml umbrella-sample [-h] [--config CONFIG] [--checkpoint CHECKPOINT]
                            [--structure STRUCTURE] [--output-dir OUTPUT_DIR]
                            [--atoms ATOMS] [--targets TARGETS]
                            [--xi-min XI_MIN] [--xi-max XI_MAX]
                            [--n-windows N_WINDOWS] [--k K_EV_A2]
                            [--temperature TEMPERATURE_K]
                            [--timestep TIMESTEP_FS] [--nsteps NSTEPS]
                            [--printfreq PRINTFREQ] [--savefreq SAVEFREQ]
                            [--seed SEED] [--no-ema] [--overwrite]

Batched distance umbrella sampling with a PhysNet / SpookyNet checkpoint via
JAX-MD NVT Nose-Hoover.

Input & configuration:
  --config CONFIG       YAML/JSON UmbrellaConfig; CLI flags override file values
                        when set
  --checkpoint CHECKPOINT
                        PhysNet / SpookyNet checkpoint
  --structure STRUCTURE
                        Input XYZ / structure file

Scientific model:
  --temperature TEMPERATURE_K
                        NVT temperature in K (default: 300)

Execution:
  --nsteps NSTEPS       NVT steps (default: 1000)
  --seed SEED           PRNG seed (default: 42)

Output & artifacts:
  --output-dir, -o OUTPUT_DIR
                        Directory for snapshots, trajectories, and summary
  --savefreq SAVEFREQ   Snapshot save interval (default: same as printfreq)
  --overwrite           Allow writing into a non-empty output directory

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --atoms ATOMS         0-based atom indices for the distance CV (I,J)
  --targets TARGETS     Comma-separated umbrella centers ξ₀ (Å)
  --xi-min XI_MIN       Grid start (Å) if --targets omitted
  --xi-max XI_MAX       Grid end (Å) if --targets omitted
  --n-windows N_WINDOWS
                        Number of windows on [xi-min, xi-max]
  --k K_EV_A2           Harmonic force constant (eV/Å²); shared across windows
                        (default: 10)
  --timestep TIMESTEP_FS
                        Timestep in fs (default: 0.5)
  --printfreq PRINTFREQ
                        Print interval in steps (default: 100)
  --no-ema              Prefer non-EMA checkpoint params

CLI for batched umbrella NVT sampling with PhysNet / SpookyNet. Usage: mmml
umbrella-sample \ --checkpoint out/ckpts/model \ --structure molecule.xyz \
--atoms 0,1 \ --xi-min 1.5 --xi-max 3.5 --n-windows 11 \ --k 20 --temperature
300 --nsteps 5000 -o out/umbrella
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
