# `mmml pyscf-evaluate`

Batch E/F/D/ESP evaluation.


## Usage

```bash
mmml pyscf-evaluate --help
```

## Options

```text
usage: mmml pyscf-evaluate [-h] -i INPUT [-o OUTPUT] [--basis BASIS] [--xc XC] [--spin SPIN] [--charge CHARGE] [--no-energy] [--no-gradient] [--no-dipole] [--esp] [--esp-cpu-fallback] [--polarizability] [--EF] [--efield Ex,Ey,Ez] [--efield-sigma EFIELD_SIGMA]
                           [--add-random-noise SIGMA] [--seed SEED] [--no-efield-include-nuclear-energy]

Evaluate geometries with pyscf-dft (energy, forces, dipoles, ESP).

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Input NPZ with R, Z, N (e.g. from normal-mode-sample)
  -o, --output OUTPUT   Output NPZ path (default: evaluated.npz)
  --basis BASIS         Basis set (default: def2-SVP)
  --xc XC               XC functional (default: PBE0)
  --spin SPIN           2*spin (0=singlet, 1=doublet, default: 0)
  --charge CHARGE       Total charge (default: 0)
  --no-energy           Skip energy (not recommended)
  --no-gradient         Skip forces/gradients
  --no-dipole           Skip dipole moments
  --esp                 Compute ESP on density-selected grid (slower)
  --esp-cpu-fallback    Use CPU path for ESP (slower; default: GPU int1e_grids)
  --polarizability      Compute molecular polarizability tensor for each geometry
  --EF                  Include uniform electric field in the Hamiltonian (atomic units). Without --efield, draw a random (Ex,Ey,Ez) per geometry (see --efield-sigma). Giving --efield alone also enables the field (same vector for all frames).
  --efield Ex,Ey,Ez     Fixed field in a.u., same for all geometries (enables E-field even without --EF). If the first component is negative, use equals form, e.g. --efield=-0.01,0,0 (argparse otherwise treats -0.01,... as a separate flag).
  --efield-sigma EFIELD_SIGMA
                        Std dev (a.u.) per component for random fields when --EF is set without --efield (default: 0.01)
  --add-random-noise SIGMA
                        Gaussian noise std dev in Angstrom added to all R components before evaluation
  --seed SEED           RNG seed for --add-random-noise and random --EF draws
  --no-efield-include-nuclear-energy
                        With --EF/--efield: use mf.kernel energy only (omit nuclear-field term after SCF).

CLI to evaluate sampled geometries with pyscf-dft (energy, forces, dipoles, ESP).

Runs all geometries in one process (same GPU context) for speed.
Input: NPZ with R (n_samples, n_atoms, 3), Z, N (e.g. from normal-mode-sample)
Output: NPZ with R, Z, N, E, F, Dxyz, esp, esp_grid (if --esp), Ef (if --EF)

Usage:
    mmml pyscf-evaluate -i out/06_sampled.npz -o out/07_evaluated.npz
    mmml pyscf-evaluate -i out/06_sampled.npz -o out/07_evaluated.npz --esp
    mmml pyscf-evaluate -i traj.npz -o out.npz --EF
    mmml pyscf-evaluate -i traj.npz -o out.npz --EF --efield 0,0,0.01
    mmml pyscf-evaluate -i traj.npz -o out.npz --efield=-0.01,0,0
    mmml pyscf-evaluate -i traj.npz -o out.npz --EF --no-efield-include-nuclear-energy
    mmml pyscf-evaluate -i traj.npz -o out.npz --add-random-noise 0.1
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
