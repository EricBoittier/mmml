# `mmml run`

MM/ML simulation (ASE + JAX-MD hybrid).

!!! warning "legacy"
    Legacy command. Prefer **`mmml md-system`**. Prefer md-system for new MD; run kept for hybrid calculator demos.


## Usage

```bash
mmml run --help
```

## Options

```text
usage: mmml run [-h] --pdbfile PDBFILE --checkpoint CHECKPOINT [--validate]
                [--energy-catch ENERGY_CATCH] [--cell CELL]
                [--flat-bottom-radius Å] [--flat-bottom-k eV/Å²]
                [--flat-bottom-mode {system,monomer}]
                [--min-com-restraint-distance Å] [--min-com-restraint-k eV/Å²]
                --n-monomers N_MONOMERS --n-atoms-monomer N_ATOMS_MONOMER
                [--ml-switch-width ML_SWITCH_WIDTH]
                [--mm-switch-on MM_SWITCH_ON]
                [--mm-switch-width MM_SWITCH_WIDTH]
                [--no-complementary-handoff] [--include-mm] [--skip-ml-dimers]
                [--mm-r-min Å] [--ml-batch-size N] [--ml-gpu-count N]
                [--debug] [--temperature TEMPERATURE] [--timestep TIMESTEP]
                [--nsteps_jaxmd NSTEPS_JAXMD]
                [--steps-per-recording STEPS_PER_RECORDING]
                [--output-prefix OUTPUT_PREFIX] [--nsteps_ase NSTEPS_ASE]
                [--optimize-monomers] [--ensemble ENSEMBLE]
                [--nhc-chain-length NHC_CHAIN_LENGTH]
                [--nhc-chain-steps NHC_CHAIN_STEPS]
                [--nhc-sy-steps NHC_SY_STEPS] [--nhc-tau NHC_TAU]
                [--pressure PRESSURE] [--nhc-barostat-tau NHC_BAROSTAT_TAU]
                [--npt-diagnose] [--nbr-monitor]
                [--heating_interval HEATING_INTERVAL]
                [--write_interval WRITE_INTERVAL] [--view-braille]
                [--charmm_heat] [--charmm_equilibration] [--charmm_production]
                [--pycharmm-minimize/--no-pycharmm-minimize | --no-pycharmm-minimize/--no-pycharmm-minimize]
                [--pycharmm-minimize-steps N]
                [--two-residue-sampling | --no-two-residue-sampling]
                [--two-residue-restraint-force K]
                [--two-residue-restraint-r0 ANGSTROM]
                [--two-residue-sampling-steps N]
                [--two-residue-restraint-resid1 TWO_RESIDUE_RESTRAINT_RESID1]
                [--two-residue-restraint-resid2 TWO_RESIDUE_RESTRAINT_RESID2]
                [--skip-setup-energy-show] [--jaxmd-minimize-steps N]
                [--jaxmd-pbc-minimize-steps N]
                [--use_physnet_calculator_for_full_system]
                [--trajectory-format {traj,dcd}] [--precompile]

PDB file processing and MD simulation demo

options:
  -h, --help            show this help message and exit
  --pdbfile PDBFILE     Path to the PDB file to load for pycharmm [requires
                        correct atom names and types].
  --checkpoint CHECKPOINT
  --validate            Validate the force field (default: False).
  --energy-catch ENERGY_CATCH
                        Energy catch factor for the simulation (default:
                        0.05).
  --cell CELL           Cubic cell side length in Å for periodic boundary
                        conditions (default: None = no PBC).
  --flat-bottom-radius Å
                        Radius of flat bottom potential to constrain system
                        COM to center (default: None = disabled). When set,
                        V=0 for |d|<=R and V=k*(|d|-R)^2 for |d|>R. With
                        --cell, center=box center; else origin.
  --flat-bottom-k eV/Å²
                        Force constant for flat bottom potential when outside
                        radius (default: 1.0).
  --flat-bottom-mode {system,monomer}
                        system: cluster COM; monomer: sum over per-monomer COM
                        restraints (same R, k).
  --min-com-restraint-distance Å
                        Pairwise inter-monomer COM lower wall. Adds
                        0.5*k*(r_min-r)^2 when COM distance r < r_min
                        (default: disabled).
  --min-com-restraint-k eV/Å²
                        Force constant for --min-com-restraint-distance
                        (default: 1.0).
  --n-monomers N_MONOMERS
                        Number of monomers in the system (default: 2).
  --n-atoms-monomer N_ATOMS_MONOMER
                        Number of atoms per monomer. Defaults to
                        total_atoms/n_monomers derived from the dataset.
  --ml-switch-width, --ml-cutoff ML_SWITCH_WIDTH
                        COM-distance width (Å) of the ML→MM handoff; ML tapers
                        to zero at mm_switch_on (default: 0.1).
  --mm-switch-on MM_SWITCH_ON
                        COM distance (Å) where ML→0 and MM→1 in complementary
                        handoff (default: 8).
  --mm-switch-width, --mm-cutoff MM_SWITCH_WIDTH
                        COM-distance width (Å) of the MM outer tail past
                        mm_switch_on (default: 5).
  --no-complementary-handoff
                        Use legacy MM switching: MM starts at mm_switch_on
                        instead of filling the ML taper handoff. When set,
                        mm_r_min defaults to mm_switch_on to exclude close
                        monomers.
  --include-mm          Keep MM contributions enabled when evaluating the
                        hybrid calculator.
  --skip-ml-dimers      If set, skip the ML dimer correction in the hybrid
                        calculator.
  --mm-r-min Å          MM inner cutoff: exclude pairs with dimer COM < this.
                        Defaults: legacy mode -> mm_switch_on*0.9;
                        complementary -> (mm_switch_on-ml_switch_width)*0.9.
  --ml-batch-size N     Max systems per ML forward pass. When set, chunk large
                        batches to reduce memory. Default: None (no chunking).
                        Suggested: 256–512 for 8–16 GB GPU, 512–1024 for 24
                        GB+.
  --ml-gpu-count N      Parallel PhysNet chunks on N local GPUs (default 1; or
                        MMML_MLPOT_N_GPUS).
  --debug               Enable verbose debug output inside the calculator
                        factory.
  --temperature TEMPERATURE
                        Temperature for MD simulation in Kelvin (default:
                        300.0).
  --timestep TIMESTEP   Timestep for MD simulation in fs (default: 0.5).
  --nsteps_jaxmd NSTEPS_JAXMD
                        Number of MD steps to run in JAX-MD (default: 100000).
  --steps-per-recording STEPS_PER_RECORDING
                        Steps between recording blocks (default: 25 for NPT,
                        1000 for NVT/NVE). NPT requires frequent neighbor list
                        updates; use smaller values if unstable.
  --output-prefix OUTPUT_PREFIX
                        Prefix for output files (default: md_simulation).
  --nsteps_ase NSTEPS_ASE
                        Number of steps to run in ASE (default: 10000).
  --optimize-monomers   If set, run monomer-wise optimization with
                        simple_physnet before hybrid BFGS. Default: skip (use
                        hybrid BFGS from CHARMM structure). Monomer
                        optimization ignores inter-monomer interactions and
                        can produce overlapping geometries.
  --ensemble ENSEMBLE   Ensemble to run the simulation in (default: nvt).
  --nhc-chain-length NHC_CHAIN_LENGTH
                        Number of chains in the Nose-Hoover chain thermostat
                        (default: 3).
  --nhc-chain-steps NHC_CHAIN_STEPS
                        Number of steps per chain in the Nose-Hoover chain
                        thermostat (default: 2).
  --nhc-sy-steps NHC_SY_STEPS
                        Number of Suzuki-Yoshida steps in the Nose-Hoover
                        chain thermostat (default: 3).
  --nhc-tau NHC_TAU     Thermostat coupling time multiplier (tau = nhc_tau *
                        dt) (default: 100).
  --pressure PRESSURE   Target pressure in atm for NPT ensemble (default:
                        1.0). Use 0 to preserve initial density (P = N*kT/V
                        for N molecules).
  --nhc-barostat-tau NHC_BAROSTAT_TAU
                        Barostat coupling time multiplier for NPT (tau =
                        nhc_barostat_tau * dt) (default: 10000).
  --npt-diagnose        Run NPT diagnostic tests before simulation (energy,
                        stress, shift, step).
  --nbr-monitor         Monitor neighbor list: log n_valid pairs, capacity,
                        fill ratio to progress and HDF5 (NPT only).
  --heating_interval HEATING_INTERVAL
                        Interval to heat the system in ASE (default: 500).
  --write_interval WRITE_INTERVAL
                        Interval to write the trajectory in ASE (default:
                        100).
  --view-braille        Display braille molecular viewer at each
                        timestep/minimization step (width=height=100).
  --charmm_heat         Run CHARMM heat (default: False).
  --charmm_equilibration
                        Run CHARMM equilibration (default: False).
  --charmm_production   Run CHARMM production (default: False).
  --pycharmm-minimize/--no-pycharmm-minimize, --no-pycharmm-minimize/--no-pycharmm-minimize
                        Run PyCHARMM nbonds/minimize before ASE/JAX-MD
                        (default: True). Use --no-pycharmm-minimize to skip
                        when going straight to JAX-MD (e.g. with --nsteps_ase
                        0) to avoid slow single-threaded CHARMM phase.
  --pycharmm-minimize-steps N
                        Number of ABNR minimization steps when PyCHARMM
                        minimize is enabled (default: 1000). Use fewer (e.g.
                        100) for faster startup when structure is already
                        reasonable.
  --two-residue-sampling, --no-two-residue-sampling
                        Run restrained two-residue PyCHARMM sampling after
                        nbonds/block setup (default: True).
  --two-residue-restraint-force K
                        CHARMM harmonic restraint force constant for two-
                        residue sampling (default: 1.0).
  --two-residue-restraint-r0 ANGSTROM
                        CHARMM harmonic restraint target distance r0 for two-
                        residue sampling (default: 2.5 Angstrom).
  --two-residue-sampling-steps N
                        ABNR steps for restrained two-residue sampling
                        (default: --pycharmm-minimize-steps).
  --two-residue-restraint-resid1 TWO_RESIDUE_RESTRAINT_RESID1
                        First CHARMM residue id for two-residue sampling
                        (default: 1).
  --two-residue-restraint-resid2 TWO_RESIDUE_RESTRAINT_RESID2
                        Second CHARMM residue id for two-residue sampling
                        (default: 2).
  --skip-setup-energy-show
                        Skip energy.show() in setup_box to avoid slow CHARMM
                        energy evaluation (Drude setup). Use for faster
                        startup; less validation of the initial structure.
  --jaxmd-minimize-steps N
                        Number of FIRE minimization steps before JAX-MD
                        (default: 1000). Use 0 to skip.
  --jaxmd-pbc-minimize-steps N
                        Number of PBC FIRE minimization steps when --cell is
                        set (default: 1000). Use 0 to skip.
  --use_physnet_calculator_for_full_system
                        Use the physnet calculator for the full system
                        (default: False).
  --trajectory-format {traj,dcd}
                        Output trajectory format: traj (ASE) or dcd (CHARMM-
                        readable, pure Python) (default: traj).
  --precompile          Compile JAX energy/force once and exit without running
                        simulation. Use to separate slow first-run compilation
                        from production MD.
```


---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
