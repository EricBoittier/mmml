# `mmml umbrella-sample`

Batched distance umbrella NVT sampling (PhysNet/SpookyNet).


## Usage

```bash
mmml umbrella-sample --help
```

## Options

```text
usage: mmml umbrella-sample [-h] [--config CONFIG] [--checkpoint CHECKPOINT]
                            [--structure STRUCTURE]
                            [--structure-index STRUCTURE_INDEX]
                            [--seed-mode {stretch,tile,frames}]
                            [--output-dir OUTPUT_DIR] [--atoms ATOMS]
                            [--atoms2 ATOMS2] [--targets TARGETS]
                            [--targets-y TARGETS_Y] [--xi-min XI_MIN]
                            [--xi-max XI_MAX] [--n-windows N_WINDOWS]
                            [--yi-min YI_MIN] [--yi-max YI_MAX]
                            [--n-windows-y N_WINDOWS_Y] [--k K_EV_A2]
                            [--ky K_Y_EV_A2] [--temperature TEMPERATURE_K]
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
                        Starting geometry: XYZ, PDB, or NPZ with R/Z arrays
  --structure-index STRUCTURE_INDEX
                        Frame index for multi-frame XYZ/PDB/NPZ (default: 0)

Scientific model:
  --temperature TEMPERATURE_K
                        NVT temperature in K (default: 300)

Execution:
  --seed-mode {stretch,tile,frames}
                        Window seeding: stretch CV to each ξ₀ (default), tile
                        reference, or use consecutive frames from --structure
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
  --atoms ATOMS         0-based atom indices for CV1 distance (I,J)
  --atoms2 ATOMS2       0-based atom indices for CV2 distance (K,L); enables 2D
                        umbrella
  --targets TARGETS     Comma-separated CV1 centers ξ₀ (Å)
  --targets-y TARGETS_Y
                        Comma-separated CV2 centers η₀ (Å); product grid with
                        --targets
  --xi-min XI_MIN       CV1 grid start (Å) if --targets omitted
  --xi-max XI_MAX       CV1 grid end (Å) if --targets omitted
  --n-windows N_WINDOWS
                        Number of CV1 windows on [xi-min, xi-max]
  --yi-min YI_MIN       CV2 grid start (Å)
  --yi-max YI_MAX       CV2 grid end (Å)
  --n-windows-y N_WINDOWS_Y
                        Number of CV2 windows on [yi-min, yi-max]
  --k K_EV_A2           CV1 harmonic force constant (eV/Å²); shared across
                        windows (default: 10)
  --ky K_Y_EV_A2        CV2 force constant (eV/Å²); default same as --k
  --timestep TIMESTEP_FS
                        Timestep in fs (default: 0.5)
  --printfreq PRINTFREQ
                        Print interval in steps (default: 100)
  --no-ema              Prefer non-EMA checkpoint params

CLI for batched umbrella NVT sampling with PhysNet / SpookyNet. Usage: mmml
umbrella-sample \ --checkpoint examples/m/kl.json \ --structure
examples/m/neb/reag_0_opt.xyz \ --atoms 1,2 \ --xi-min 1.5 --xi-max 3.5
--n-windows 11 \ --k 20 --temperature 300 --nsteps 5000 -o out/umbrella # 2D
(Cl–C × N–C product grid) mmml umbrella-sample --checkpoint examples/m/kl.json \
--structure examples/m/neb/reag_0_opt.xyz \ --atoms 0,2 --atoms2 1,2 \ --xi-min
1.5 --xi-max 3.0 --n-windows 4 \ --yi-min 1.5 --yi-max 3.0 --n-windows-y 4 \ --k
20 --ky 20 -o out/umbrella2d --overwrite # NPZ (R, Z) or PDB also work; --seed-
mode frames uses consecutive frames as windows mmml umbrella-sample --checkpoint
ckpt.json --structure data.npz \ --atoms 0,1 --targets 1.8,2.0,2.2 --seed-mode
frames -o out/umb
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
