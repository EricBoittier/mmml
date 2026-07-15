# `mmml normal-mode-sample`

Sample along vibrational modes.


## Usage

```bash
mmml normal-mode-sample --help
```

## Options

```text
usage: mmml normal-mode-sample [-h] -i INPUT [-o OUTPUT] [--amplitude AMPLITUDE]
                               [--amplitudes AMPLITUDES [AMPLITUDES ...]]
                               [--freq-min FREQ_MIN] [--include-equilibrium]
                               [--samples-per-mode {1,2}] [--max-samples N]

Sample geometries along vibrational modes from pyscf-dft harmonic output.

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Path to pyscf-dft output .h5 (must contain harmonic
                        group)
  -o, --output OUTPUT   Output NPZ path (default: sampled.npz)
  --amplitude AMPLITUDE
                        Displacement amplitude in Angstrom (default: 0.1)
  --amplitudes AMPLITUDES [AMPLITUDES ...]
                        List of amplitudes (overrides --amplitude)
  --freq-min FREQ_MIN   Minimum frequency (cm^-1) to include (default: 50)
  --include-equilibrium
                        Add equilibrium geometry as first sample
  --samples-per-mode {1,2}
                        2 for +/- amplitude (default), 1 for + only
  --max-samples N       Maximum number of structures to generate (default: no
                        limit)

CLI for normal mode sampling from pyscf-dft harmonic output. Samples geometries
along vibrational modes for downstream QM/ML. Input: .h5 from mmml pyscf-dft
--harmonic Output: NPZ with R (n_samples, n_atoms, 3), Z, N Usage: mmml normal-
mode-sample -i out/04_results.h5 -o out/06_sampled.npz --amplitude 0.1 mmml
normal-mode-sample -i out/04_results.h5 -o out/06_sampled.npz --amplitude 0.1
--include-equilibrium
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
