# `mmml liquid-box`

Build/certify periodic liquid boxes (MM only).


## Usage

```bash
mmml liquid-box --help
```

## Options

```text
usage: mmml liquid-box [-h] --composition COMPOSITION --output-dir OUTPUT_DIR
                       [--profile {standard,dense,conservative}]
                       [--spacing SPACING] [--seed SEED]
                       [--temperature TEMPERATURE] [--dt-fs DT_FS]
                       [--echeck KCAL] [--no-echeck]
                       [--allow-incomplete-dynamics]
                       [--charmm-mm-pretreat-echeck KCAL]
                       [--charmm-sd-steps CHARMM_SD_STEPS]
                       [--charmm-abnr-steps CHARMM_ABNR_STEPS]
                       [--packmol-tolerance PACKMOL_TOLERANCE]
                       [--reuse-packmol-cache | --no-reuse-packmol-cache]
                       [--rebuild-packmol]
                       [--packmol-cache-dir PACKMOL_CACHE_DIR] [--quiet]
                       [--box-size ANG] [--box-auto {geometry,density,count}]
                       [--box-auto-count-min-molecules N]
                       [--box-auto-count-max-molecules N]
                       [--target-density-g-cm3 RHO]
                       [--bulk-density-fraction FRAC]
                       [--mc-density-equalize | --no-mc-density-equalize]
                       [--mc-density-target-g-cm3 RHO] [--mc-density-steps N]
                       [--mc-density-step-scale LOGSCALE]
                       [--mc-density-temperature T] [--mc-density-seed SEED]
                       [--mc-density-min-scale S] [--mc-density-max-scale S]
                       [--mini-box-equil-ps PS]
                       [--mini-box-equil-allow-fixed-box]
                       [--mini-box-equil-fixed-nvt]
                       [--jaxmd-mini-box-equil-ps PS]
                       [--mini-lattice-abnr-steps N]
                       [--mini-lattice-abnr-nocoords]
                       [--mini-lattice-abnr-allow-fixed-box]
                       [--liquid-prep | --no-liquid-prep]
                       [--density-prep-mode {off,resilient}]
                       [--density-prep-ladder | --no-density-prep-ladder]
                       [--density-prep-ladder-max-rounds N]
                       [--density-prep-lattice-abnr-steps N]
                       [--pre-mlpot-overlap-min-distance ANG]
                       [--prep-ladder-dir PREP_LADDER_DIR]
                       [--cleanup-dir CLEANUP_DIR] [--no-recovery-artifacts]
                       [--cleanup | --no-cleanup]

Build and certify a periodic liquid box under CHARMM MM only (Packmol → MC
density → SD/ABNR → optional lattice/NPT → geometry gate). Writes
model.psf/crd, box.json, and REPORT.md. See docs/liquid-box-workflow.md.

options:
  -h, --help            show this help message and exit
  --composition COMPOSITION
                        Composition for Packmol cube build, e.g. DCM:206 or
                        MEOH:100. (default: None)
  --output-dir, -o OUTPUT_DIR
                        Directory for certified box artifacts. (default: None)
  --profile {standard,dense,conservative}
                        standard: MC + CHARMM pre-minimize only; dense:
                        liquid-prep preventive stack; conservative: dense with
                        looser initial density. (default: dense)
  --spacing SPACING     Monomer template spacing for cluster build (Å).
                        (default: 4.0)
  --seed SEED           Random seed for Packmol / MC placement. (default: 123)
  --temperature TEMPERATURE
                        Temperature for mini box equilibration (K). (default:
                        300.0)
  --dt-fs DT_FS         Timestep for mini box equilibration (fs). (default:
                        0.25)
  --charmm-mm-pretreat-echeck KCAL
                        Enable ECHECK during mini box CPT equil (kcal/mol).
                        Default: disabled (NPT prep often exceeds ML echeck
                        floors). (default: None)
  --charmm-sd-steps CHARMM_SD_STEPS
                        CHARMM SD steps during MM pre-minimize (dense profile
                        bumps to ≥1000). (default: 50)
  --charmm-abnr-steps CHARMM_ABNR_STEPS
                        CHARMM ABNR steps during MM pre-minimize (dense
                        profile bumps to ≥1000). (default: 100)
  --packmol-tolerance PACKMOL_TOLERANCE
                        Packmol distance tolerance (Å). (default: 2.0)
  --reuse-packmol-cache, --no-reuse-packmol-cache
                        Reuse on-disk Packmol cache when composition matches.
                        (default: True)
  --rebuild-packmol     Ignore Packmol cache and rebuild placement. (default:
                        False)
  --packmol-cache-dir PACKMOL_CACHE_DIR
                        Packmol cache root (default: output-
                        dir/.packmol_cache). (default: None)
  --quiet               Reduce log output. (default: False)

Dynamics stability (ECHECK):
  --echeck KCAL         Stop dynamics if total energy change exceeds this
                        (kcal/mol); default 100. Auto-loosened for large
                        clusters (see --no-scale-echeck). Use --no-echeck to
                        disable. (default: 100.0)
  --no-echeck           Disable ECHECK (CHARMM -1 = no early stop) (default:
                        False)
  --allow-incomplete-dynamics
                        Do not fail staged MD when CHARMM stops early (echeck)
                        or the stage DCD has too few frames. Default: abort
                        with a clear error. (default: False)

PBC box sizing:
  --box-size ANG        Fixed cubic box side (Å) for Packmol cube and PBC
                        cell. With --box-auto count, scales --composition to
                        target ρ at this side. (default: None)
  --box-auto {geometry,density,count}
                        How to choose the cubic box / molecule count:
                        geometry=span+padding (default); density=box side from
                        --composition counts and target ρ; count=scale
                        --composition stoichiometry to target ρ in fixed
                        --box-size. (default: None)
  --box-auto-count-min-molecules N
                        Minimum total molecules for --box-auto count (default:
                        1). (default: 1)
  --box-auto-count-max-molecules N
                        Optional cap on total molecules for --box-auto count.
                        (default: None)
  --target-density-g-cm3 RHO
                        Target mass density (g/cm³) for --box-auto density
                        (requires --composition). (default: None)
  --bulk-density-fraction FRAC
                        Fraction of experimental bulk ρ for a single-species
                        --composition (e.g. 0.85 for 85% of liquid DCM
                        density). (default: None)
  --mc-density-equalize, --no-mc-density-equalize
                        Run default post-build MC cubic-volume equalization
                        for PBC composition builds when a density target can
                        be resolved (default: on). (default: True)
  --mc-density-target-g-cm3 RHO
                        Target density for MC density equalization. Defaults
                        to --target-density-g-cm3, --bulk-density-fraction, or
                        known single-solvent bulk density. (default: None)
  --mc-density-steps N  MC density equalization proposal count (default: 64).
                        (default: 64)
  --mc-density-step-scale LOGSCALE
                        Log box-side proposal noise scale for MC density
                        equalization (default: 0.04). (default: 0.04)
  --mc-density-temperature T
                        Dimensionless Metropolis temperature for density-error
                        acceptance (default: 0.02). (default: 0.02)
  --mc-density-seed SEED
                        Random seed for MC density equalization (default:
                        --seed). (default: None)
  --mc-density-min-scale S
                        Minimum allowed final box side relative to the initial
                        side (default: 0.35). (default: 0.35)
  --mc-density-max-scale S
                        Maximum allowed final box side relative to the initial
                        side (default: 1.50). (default: 1.5)
  --mini-box-equil-ps PS
                        PyCHARMM mini: short CPT NPT equilibration (ps) after
                        coordinate-only CHARMM MM mini and before MLpot SD.
                        0=off. Skipped when pretreat NPT equi runs. (default:
                        0.0)
  --mini-box-equil-allow-fixed-box
                        Allow --mini-box-equil-ps CPT NPT even when --box-size
                        is set (default: fixed --box-size uses Hoover NVT only
                        during pretreat). (default: False)
  --mini-box-equil-fixed-nvt
                        During mini box equil with a fixed --box-size, use
                        Hoover NVT instead of CPT NPT (liquid-prep dense
                        default). (default: False)
  --jaxmd-mini-box-equil-ps PS
                        JAX-MD: short NPT prelude (ps) after PBC FIRE minimize
                        to relax the cell before the main ensemble. 0=off.
                        (default: 0.0)
  --mini-lattice-abnr-steps N
                        PyCHARMM mini: CHARMM MINI ABNR LATTice steps to
                        optimize the cubic unit cell after coordinate-only
                        CHARMM MM mini and before MLpot SD. 0=off. Requires
                        CRYSTAL/PBC. (default: 0)
  --mini-lattice-abnr-nocoords
                        With --mini-lattice-abnr-steps: optimize only the unit
                        cell (NOCOordinates); default optimizes coordinates
                        and box together. (default: False)
  --mini-lattice-abnr-allow-fixed-box
                        Allow --mini-lattice-abnr-steps even when --box-size
                        is set (default: fixed --box-size skips lattice
                        minimization). (default: False)
  --liquid-prep, --no-liquid-prep
                        Easy dense-liquid setup: same as --density-prep-mode
                        resilient (looser Packmol, MC density equalization,
                        stronger CHARMM/lattice mini, mini box equil, post-
                        mini rescue ladder when GRMS is high). For full prep +
                        dynamics recovery in one flag, prefer --cleanup.
                        (default: False)
  --density-prep-mode {off,resilient}
                        Condensed-phase box prep strategy. resilient: start
                        Packmol slightly below target density, enable MC
                        equalization, stronger CHARMM/lattice mini, and the
                        post-mini density prep ladder when GRMS is high.
                        (default: off)
  --density-prep-ladder, --no-density-prep-ladder
                        After MLpot mini, run a multi-step density/box rescue
                        ladder (repack, MC density, lattice ABNR, bonded MM,
                        ASE BFGS/FIRE, MLpot SD) when GRMS exceeds --max-grms-
                        before-dyn. Default on for --density-prep-mode
                        resilient. (default: None)
  --density-prep-ladder-max-rounds N
                        Maximum density prep ladder rounds (default: 3).
                        (default: 3)
  --density-prep-lattice-abnr-steps N
                        Lattice ABNR steps inside the density prep ladder
                        (0=use --mini-lattice-abnr-steps or 100). (default: 0)
  --pre-mlpot-overlap-min-distance ANG
                        Pre-MLpot geometry gate: minimum inter-monomer atom
                        distance in Å (default: 1.0; independent of
                        --dynamics-overlap-min-distance). Catches true cross-
                        monomer clashes while allowing tight liquid contacts
                        that hybrid mini relaxes. (default: None)

Recovery artifact folders:
  --prep-ladder-dir PREP_LADDER_DIR
                        Subfolder under --output-dir for density / pre-MLpot
                        ladder checkpoints (default: prep_ladder). (default:
                        prep_ladder)
  --cleanup-dir CLEANUP_DIR
                        Subfolder under --output-dir for geometry cleanup /
                        overlap rescue checkpoints (default: cleanup).
                        (default: cleanup)
  --no-recovery-artifacts
                        Do not write prep_ladder/ or cleanup/ checkpoint
                        folders. (default: False)

Geometry cleanup (one-shot recovery):
  --cleanup, --no-cleanup
                        Enable the full geometry cleanup ladder: resilient
                        liquid prep, pre-MLpot repack gate, density prep
                        ladder, hybrid calculator pre-minimize, bonded-MM
                        recovery, and dynamics overlap rescue (selective
                        monomer repack when forces indicate 1–2 hot spots).
                        Use once when a run breaks (ECHECK, overlap, high
                        GRMS) to reach a stable restart, then re-run without
                        --cleanup for production trajectories where time-
                        series correlations matter. Superset of --liquid-prep;
                        individual recovery flags remain overridable.
                        (default: False)
```

## Related docs

- [Liquid box workflow](../../liquid-box-workflow.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
