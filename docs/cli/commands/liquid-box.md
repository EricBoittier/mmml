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
density → SD/ABNR → optional lattice/NPT → geometry gate). Writes model.psf/crd,
box.json, and REPORT.md. See docs/liquid-box-workflow.md.

options:
  -h, --help            show this help message and exit
  --composition COMPOSITION
                        Composition for Packmol cube build, e.g. DCM:206 or
                        MEOH:100.
  --output-dir, -o OUTPUT_DIR
                        Directory for certified box artifacts.
  --profile {standard,dense,conservative}
                        standard: MC + CHARMM pre-minimize only; dense: liquid-
                        prep preventive stack; conservative: dense with looser
                        initial density.
  --spacing SPACING     Monomer template spacing for cluster build (Å).
  --seed SEED           Random seed for Packmol / MC placement.
  --temperature TEMPERATURE
                        Temperature for mini box equilibration (K).
  --dt-fs DT_FS         Timestep for mini box equilibration (fs).
  --charmm-mm-pretreat-echeck KCAL
                        Enable ECHECK during mini box CPT equil (kcal/mol).
                        Default: disabled (NPT prep often exceeds ML echeck
                        floors).
  --charmm-sd-steps CHARMM_SD_STEPS
                        CHARMM SD steps during MM pre-minimize (dense profile
                        bumps to ≥1000).
  --charmm-abnr-steps CHARMM_ABNR_STEPS
                        CHARMM ABNR steps during MM pre-minimize (dense profile
                        bumps to ≥1000).
  --packmol-tolerance PACKMOL_TOLERANCE
                        Packmol distance tolerance (Å).
  --reuse-packmol-cache, --no-reuse-packmol-cache
                        Reuse on-disk Packmol cache when composition matches.
  --rebuild-packmol     Ignore Packmol cache and rebuild placement.
  --packmol-cache-dir PACKMOL_CACHE_DIR
                        Packmol cache root (default: output-dir/.packmol_cache).
  --quiet               Reduce log output.

Dynamics stability (ECHECK):
  --echeck KCAL         Stop dynamics if total energy change exceeds this
                        (kcal/mol); default 100. Auto-loosened for large
                        clusters (see --no-scale-echeck). Use --no-echeck to
                        disable.
  --no-echeck           Disable ECHECK (CHARMM -1 = no early stop)
  --allow-incomplete-dynamics
                        Do not fail staged MD when CHARMM stops early (echeck)
                        or the stage DCD has too few frames. Default: abort with
                        a clear error.

PBC box sizing:
  --box-size ANG        Fixed cubic box side (Å) for Packmol cube and PBC cell.
                        With --box-auto count, scales --composition to target ρ
                        at this side.
  --box-auto {geometry,density,count}
                        How to choose the cubic box / molecule count:
                        geometry=span+padding (default); density=box side from
                        --composition counts and target ρ; count=scale
                        --composition stoichiometry to target ρ in fixed --box-
                        size.
  --box-auto-count-min-molecules N
                        Minimum total molecules for --box-auto count (default:
                        1).
  --box-auto-count-max-molecules N
                        Optional cap on total molecules for --box-auto count.
  --target-density-g-cm3 RHO
                        Target mass density (g/cm³) for --box-auto density
                        (requires --composition).
  --bulk-density-fraction FRAC
                        Fraction of experimental bulk ρ for a single-species
                        --composition (e.g. 0.85 for 85% of liquid DCM density).
  --mc-density-equalize, --no-mc-density-equalize
                        Run default post-build MC cubic-volume equalization for
                        PBC composition builds when a density target can be
                        resolved (default: on).
  --mc-density-target-g-cm3 RHO
                        Target density for MC density equalization. Defaults to
                        --target-density-g-cm3, --bulk-density-fraction, or
                        known single-solvent bulk density.
  --mc-density-steps N  MC density equalization proposal count (default: 64).
  --mc-density-step-scale LOGSCALE
                        Log box-side proposal noise scale for MC density
                        equalization (default: 0.04).
  --mc-density-temperature T
                        Dimensionless Metropolis temperature for density-error
                        acceptance (default: 0.02).
  --mc-density-seed SEED
                        Random seed for MC density equalization (default:
                        --seed).
  --mc-density-min-scale S
                        Minimum allowed final box side relative to the initial
                        side (default: 0.35).
  --mc-density-max-scale S
                        Maximum allowed final box side relative to the initial
                        side (default: 1.50).
  --mini-box-equil-ps PS
                        PyCHARMM mini: short CPT NPT equilibration (ps) after
                        coordinate-only CHARMM MM mini and before MLpot SD.
                        0=off. Skipped when pretreat NPT equi runs.
  --mini-box-equil-allow-fixed-box
                        Allow --mini-box-equil-ps CPT NPT even when --box-size
                        is set (default: fixed --box-size uses Hoover NVT only
                        during pretreat).
  --mini-box-equil-fixed-nvt
                        During mini box equil with a fixed --box-size, use
                        Hoover NVT instead of CPT NPT (liquid-prep dense
                        default).
  --jaxmd-mini-box-equil-ps PS
                        JAX-MD: short NPT prelude (ps) after PBC FIRE minimize
                        to relax the cell before the main ensemble. 0=off.
  --mini-lattice-abnr-steps N
                        PyCHARMM mini: CHARMM MINI ABNR LATTice steps to
                        optimize the cubic unit cell after coordinate-only
                        CHARMM MM mini and before MLpot SD. 0=off. Requires
                        CRYSTAL/PBC.
  --mini-lattice-abnr-nocoords
                        With --mini-lattice-abnr-steps: optimize only the unit
                        cell (NOCOordinates); default optimizes coordinates and
                        box together.
  --mini-lattice-abnr-allow-fixed-box
                        Allow --mini-lattice-abnr-steps even when --box-size is
                        set (default: fixed --box-size skips lattice
                        minimization).
  --liquid-prep, --no-liquid-prep
                        Easy dense-liquid setup: same as --density-prep-mode
                        resilient (looser Packmol, MC density equalization,
                        stronger CHARMM/lattice mini, mini box equil, post-mini
                        rescue ladder when GRMS is high). For full prep +
                        dynamics recovery in one flag, prefer --cleanup.
  --density-prep-mode {off,resilient}
                        Condensed-phase box prep strategy. resilient: start
                        Packmol slightly below target density, enable MC
                        equalization, stronger CHARMM/lattice mini, and the
                        post-mini density prep ladder when GRMS is high.
  --density-prep-ladder, --no-density-prep-ladder
                        After MLpot mini, run a multi-step density/box rescue
                        ladder (repack, MC density, lattice ABNR, bonded MM, ASE
                        BFGS/FIRE, MLpot SD) when GRMS exceeds --max-grms-
                        before-dyn. Default on for --density-prep-mode
                        resilient.
  --density-prep-ladder-max-rounds N
                        Maximum density prep ladder rounds (default: 3).
  --density-prep-lattice-abnr-steps N
                        Lattice ABNR steps inside the density prep ladder (0=use
                        --mini-lattice-abnr-steps or 100).
  --pre-mlpot-overlap-min-distance ANG
                        Pre-MLpot geometry gate: minimum inter-monomer atom
                        distance in Å (default: 1.0; independent of --dynamics-
                        overlap-min-distance). Catches true cross-monomer
                        clashes while allowing tight liquid contacts that hybrid
                        mini relaxes.

Recovery artifact folders:
  --prep-ladder-dir PREP_LADDER_DIR
                        Subfolder under --output-dir for density / pre-MLpot
                        ladder checkpoints (default: prep_ladder).
  --cleanup-dir CLEANUP_DIR
                        Subfolder under --output-dir for geometry cleanup /
                        overlap rescue checkpoints (default: cleanup).
  --no-recovery-artifacts
                        Do not write prep_ladder/ or cleanup/ checkpoint
                        folders.

Geometry cleanup (one-shot recovery):
  --cleanup, --no-cleanup
                        Enable the full geometry cleanup ladder: resilient
                        liquid prep, pre-MLpot repack gate, density prep ladder,
                        hybrid calculator pre-minimize, bonded-MM recovery, and
                        dynamics overlap rescue (selective monomer repack when
                        forces indicate 1–2 hot spots). Use once when a run
                        breaks (ECHECK, overlap, high GRMS) to reach a stable
                        restart, then re-run without --cleanup for production
                        trajectories where time-series correlations matter.
                        Superset of --liquid-prep; individual recovery flags
                        remain overridable.
```

## Example structures

![Density prep ladder (schematic)](../../images/plots/liquid-box-density-ladder.png)

More detail: [Structure building guide](../structure-building.md).

## Related docs

- [Liquid box workflow](../../liquid-box-workflow.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
