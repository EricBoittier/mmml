# `mmml md-system`

Mixed-composition MD (ASE/JAX-MD/PyCHARMM).


ASE and JAX-MD command routing can be imported and inspected without a local
`libcharmm`. PyCHARMM is loaded lazily only when a CHARMM-backed builder,
minimizer, or backend operation is requested. Those operations require
`CHARMM_LIB_DIR` to point to the directory containing `libcharmm.so` on Linux
(`libcharmm.dylib` on macOS). Certified `--from-psf`/`--from-crd` geometry can
therefore be routed and unit-tested on ordinary CI runners without initializing
the native CHARMM runtime.

## Usage

```bash
mmml md-system --help
```

## Options

```text
usage: mmml md-system [-h]
                      [--setup {free_nve,free_nvt,free_thermalize,pbc_nve,pbc_nvt,pbc_thermalize,pbc_npt,lambda_ti,pycharmm_minimize,pycharmm_full,all}]
                      [--backend {auto,ase,jaxmd,pycharmm}]
                      [--checkpoint CHECKPOINT]
                      [--mbd-checkpoint MBD_CHECKPOINT]
                      [--mbd-weight MBD_WEIGHT]
                      [--multipole-checkpoint MULTIPOLE_CHECKPOINT]
                      [--sampler {md,rigid}] [--ff {cgenff,zbl-mbd-multipoles}]
                      [--jaxmd-unified]
                      [--electrostatics-damping-sigma ELECTROSTATICS_DAMPING_SIGMA]
                      [--output-dir OUTPUT_DIR] [--job-name JOB_NAME]
                      [--jobs-dir JOBS_DIR] [--template-pdb TEMPLATE_PDB]
                      [--n-molecules N_MOLECULES] [--composition COMPOSITION]
                      [--spacing SPACING] [--ps PS] [--dt-fs DT_FS]
                      [--traj-chunk-frames TRAJ_CHUNK_FRAMES]
                      [--traj-export-molecular-wrap] [--temperature TEMPERATURE]
                      [--nvt-integrator {auto,nhc,langevin}]
                      [--pressure PRESSURE] [--seed SEED]
                      [--builder {gas,liquid,crystal}]
                      [--min-intermonomer-atom-distance MIN_INTERMONOMER_ATOM_DISTANCE]
                      [--packmol | --no-packmol]
                      [--packmol-placement {cube,sphere}]
                      [--packmol-sphere | --no-packmol-sphere]
                      [--packmol-radius Å] [--packmol-center CX CY CZ]
                      [--packmol-tolerance PACKMOL_TOLERANCE]
                      [--reuse-packmol-cache | --no-reuse-packmol-cache]
                      [--rebuild-packmol]
                      [--packmol-cache-dir PACKMOL_CACHE_DIR]
                      [--pyxtal | --no-pyxtal] [--pyxtal-spg PYXTAL_SPG]
                      [--pyxtal-dim {0,1,2,3}] [--pyxtal-factor PYXTAL_FACTOR]
                      [--pyxtal-stoichiometry Z [Z ...]]
                      [--pyxtal-supercell NX,NY,NZ]
                      [--pyxtal-attempts PYXTAL_ATTEMPTS]
                      [--pyxtal-trim | --no-pyxtal-trim] [--optimize-pyxtal]
                      [--optimize-pyxtal-emt] [--box-size ANG]
                      [--packmol-box-size ANG] [--packmol-box-padding ANG]
                      [--box-auto {geometry,density,count}]
                      [--box-auto-count-min-molecules N]
                      [--box-auto-count-max-molecules N]
                      [--target-density-g-cm3 RHO]
                      [--bulk-density-fraction FRAC]
                      [--mc-density-equalize | --no-mc-density-equalize]
                      [--mc-density-target-g-cm3 RHO] [--mc-density-steps N]
                      [--mc-density-step-scale LOGSCALE]
                      [--mc-density-temperature T] [--mc-density-seed SEED]
                      [--mc-density-min-scale S] [--mc-density-max-scale S]
                      [--mini-box-equil-ps PS] [--mini-box-equil-ps-heat PS]
                      [--mini-box-equil-ps-cool PS]
                      [--mini-box-equil-hot-temp K]
                      [--mini-box-equil-allow-fixed-box]
                      [--mini-box-equil-fixed-nvt]
                      [--jaxmd-mini-box-equil-ps PS]
                      [--mini-lattice-abnr-steps N]
                      [--mini-lattice-abnr-nocoords]
                      [--mini-lattice-abnr-allow-fixed-box]
                      [--mini-lattice-abnr-allow-density-resize]
                      [--density-certify-relative-tolerance FRAC]
                      [--liquid-prep | --no-liquid-prep]
                      [--density-prep-mode {off,resilient}]
                      [--density-prep-ladder | --no-density-prep-ladder]
                      [--density-prep-ladder-max-rounds N]
                      [--density-prep-lattice-abnr-steps N]
                      [--pre-mlpot-overlap-min-distance ANG]
                      [--charmm-image-mlpot-min-distance ANG]
                      [--pre-mlpot-h-heavy-min-distance ANG]
                      [--pre-mlpot-heavy-heavy-min-distance ANG]
                      [--mlpot-registration-max-grms KCAL]
                      [--prep-ladder-dir PREP_LADDER_DIR]
                      [--cleanup-dir CLEANUP_DIR] [--no-recovery-artifacts]
                      [--cleanup | --no-cleanup] [--save-run-state]
                      [--run-state-dir RUN_STATE_DIR]
                      [--overlap-run-state-dir OVERLAP_RUN_STATE_DIR]
                      [--overlap-run-state-every-chunks OVERLAP_RUN_STATE_EVERY_CHUNKS]
                      [--flat-bottom-radius Å] [--flat-bottom-k eV/Å²]
                      [--flat-bottom-selection FLAT_BOTTOM_SELECTION]
                      [--flat-bottom-mode {system,monomer}]
                      [--min-com-restraint-distance Å]
                      [--min-com-restraint-k eV/Å²] [--extra-args ...]
                      [--fix-resids FIX_RESIDS]
                      [--constrain-resids CONSTRAIN_RESIDS] [--no-fix]
                      [--mini-nstep MINI_NSTEP] [--no-pre-minimize]
                      [--echeck ECHECK] [--no-echeck] [--no-echeck-heat]
                      [--allow-incomplete-dynamics] [--nprint NPRINT]
                      [--dyn-nprint DYN_NPRINT] [--dyn-iprfrq DYN_IPRFRQ]
                      [--heat-ihtfrq N] [--heat-bussi-taut PS]
                      [--heat-thermostat {bussi,scale,hoover}] [--heat-firstt K]
                      [--heat-finalt K] [--heat-mode {ramp,hold}]
                      [--heat-hoover-tmass M] [--nve-boltzmann-temp K]
                      [--nve-max-f-start-eVA EV_PER_A]
                      [--nve-etot-drift-abort-eV EV]
                      [--nve-etot-drift-rescue | --no-nve-etot-drift-rescue]
                      [--nve-etot-drift-rescue-attempts NVE_ETOT_DRIFT_RESCUE_ATTEMPTS]
                      [--nve-etot-drift-rescue-fire-steps NVE_ETOT_DRIFT_RESCUE_FIRE_STEPS]
                      [--nve-etot-drift-rescue-grace-eV EV]
                      [--nve-etot-drift-rescue-dt-scale NVE_ETOT_DRIFT_RESCUE_DT_SCALE]
                      [--nve-etot-drift-rescue-min-dt-fs NVE_ETOT_DRIFT_RESCUE_MIN_DT_FS]
                      [--heat-comp-damp | --no-heat-comp-damp]
                      [--heat-comp-hydrogen-only | --no-heat-comp-hydrogen-only]
                      [--heat-comp-force-min KCAL]
                      [--heat-comp-force-scale HEAT_COMP_FORCE_SCALE]
                      [--skip-energy-show] [--show-energy | --no-show-energy]
                      [--quiet] [--verbose] [--dcd-nsavc DCD_NSAVC]
                      [--dcd-interval-ps PS] [--dcd-max-frames N]
                      [--save-forces-npz]
                      [--forces-npz-interval FORCES_NPZ_INTERVAL]
                      [--no-scale-mini-nstep] [--no-scale-echeck]
                      [--allow-high-grms] [--no-scale-max-grms]
                      [--max-grms-before-dyn MAX_GRMS_BEFORE_DYN] [--test-first]
                      [--test-first-tol TEST_FIRST_TOL]
                      [--test-first-step TEST_FIRST_STEP]
                      [--test-first-resids TEST_FIRST_RESIDS]
                      [--test-first-charmm] [--test-first-update-nbonds]
                      [--ml-batch-size N] [--ml-gpu-count N] [--max-pairs N]
                      [--ml-spatial-mpi] [--charmm-omp-threads N]
                      [--ml-compute-dtype {float32,float64}]
                      [--ml-max-active-dimers N] [--md-stages MD_STAGES]
                      [--md-stage {mini,heat,nve,equi,prod}] [--tag TAG]
                      [--ps-heat PS_HEAT] [--charmm-mm-pretreat]
                      [--charmm-mm-pretreat-on-handoff]
                      [--charmm-mm-pretreat-with-liquid-prep]
                      [--charmm-mm-pretreat-ps-heat PS]
                      [--charmm-mm-pretreat-heat-nstep N]
                      [--charmm-mm-pretreat-ps-equi PS]
                      [--charmm-mm-pretreat-ps-prod PS]
                      [--charmm-mm-pretreat-mini-sd N]
                      [--charmm-mm-pretreat-mini-abnr N]
                      [--charmm-mm-pretreat-dt-fs FS]
                      [--charmm-mm-pretreat-temperature K]
                      [--charmm-mm-pretreat-pressure ATM]
                      [--charmm-mm-pretreat-echeck KCAL]
                      [--charmm-mm-pretreat-inbfrq N]
                      [--charmm-mm-pretreat-imgfrq N]
                      [--charmm-mm-pretreat-ixtfrq N] [--ps-nve PS_NVE]
                      [--ps-equi PS_EQUI] [--ps-prod PS_PROD]
                      [--npt-thermostat {hoover,berendsen}]
                      [--npt-pressure NPT_PRESSURE] [--npt-pgamma NPT_PGAMMA]
                      [--n-heat-segments N_HEAT_SEGMENTS]
                      [--n-equi-segments N_EQUI_SEGMENTS]
                      [--n-prod-segments N_PROD_SEGMENTS]
                      [--bonded-mm-mini | --no-bonded-mm-mini]
                      [--bonded-mm-mini-after BONDED_MM_MINI_AFTER]
                      [--bonded-mm-mini-steps BONDED_MM_MINI_STEPS]
                      [--bonded-recovery-backend {auto,jax,charmm,sidecar}]
                      [--bonded-mm-mini-always]
                      [--bonded-mm-internal-margin BONDED_MM_INTERNAL_MARGIN]
                      [--bonded-mm-grms-margin BONDED_MM_GRMS_MARGIN]
                      [--bonded-mm-internal-energy-margin BONDED_MM_INTERNAL_ENERGY_MARGIN]
                      [--bonded-mm-angl-margin BONDED_MM_ANGL_MARGIN]
                      [--bonded-mm-max-angl-kcal BONDED_MM_MAX_ANGL_KCAL]
                      [--bonded-mm-max-internal-kcal BONDED_MM_MAX_INTERNAL_KCAL]
                      [--allow-high-bonded-strain]
                      [--dynamics-overlap-action {error,warn,rescue,off}]
                      [--dynamics-overlap-charmm-sd-steps DYNAMICS_OVERLAP_CHARMM_SD_STEPS]
                      [--dynamics-overlap-charmm-abnr-steps DYNAMICS_OVERLAP_CHARMM_ABNR_STEPS]
                      [--dynamics-overlap-min-distance ANG]
                      [--dynamics-intra-min-distance ANG]
                      [--no-dynamics-intra-exclude-1-3]
                      [--dynamics-intra-rescue-sd-steps DYNAMICS_INTRA_RESCUE_SD_STEPS]
                      [--dynamics-overlap-check-interval DYNAMICS_OVERLAP_CHECK_INTERVAL]
                      [--heat-overlap-segment-boundary-only]
                      [--dynamics-overlap-memory-handoff]
                      [--no-dynamics-overlap-separate]
                      [--dynamics-overlap-separate-margin ANG]
                      [--dynamics-max-monomer-extent ANG]
                      [--no-dynamics-geometry-limits-auto]
                      [--no-dynamics-max-monomer-extent]
                      [--no-dynamics-monomer-health]
                      [--dynamics-monomer-health-debug]
                      [--no-dynamics-monomer-template-restore]
                      [--no-dynamics-monomer-jax-after-restore]
                      [--no-dynamics-monomer-velocity-restore]
                      [--dynamics-monomer-health-max-restore DYNAMICS_MONOMER_HEALTH_MAX_RESTORE]
                      [--dynamics-monomer-velocity-warn-recover-fraction DYNAMICS_MONOMER_VELOCITY_WARN_RECOVER_FRACTION]
                      [--dynamics-monomer-baseline-floor-fraction DYNAMICS_MONOMER_BASELINE_FLOOR_FRACTION]
                      [--dynamics-monomer-ratio-without-abs]
                      [--dynamics-monomer-template-on-force]
                      [--dynamics-monomer-bond-stretch-factor DYNAMICS_MONOMER_BOND_STRETCH_FACTOR]
                      [--dynamics-monomer-bond-stretch-abs DYNAMICS_MONOMER_BOND_STRETCH_ABS]
                      [--dynamics-monomer-com-flyoff DYNAMICS_MONOMER_COM_FLYOFF]
                      [--no-dynamics-monomer-com-flyoff]
                      [--restart-from RESTART_FROM] [--from-psf FROM_PSF]
                      [--from-crd FROM_CRD] [--skip-cluster-build]
                      [--skip-if-crd-exists] [--no-save-vmd-topology]
                      [--free-space] [--mlpot-pbc] [--dyn-inbfrq DYN_INBFRQ]
                      [--dyn-imgfrq N] [--dyn-freq-cadence N]
                      [--pre-nve-charmm-update | --no-pre-nve-charmm-update]
                      [--lambda-md-mode {free_nve,free_nvt,pbc_nve,pbc_nvt}]
                      [--couple-residues COUPLE_RESIDUES]
                      [--lambda-windows LAMBDA_WINDOWS [LAMBDA_WINDOWS ...]]
                      [--pre-min-steps PRE_MIN_STEPS]
                      [--pre-min-fmax PRE_MIN_FMAX] [--min-steps MIN_STEPS]
                      [--min-fmax MIN_FMAX] [--bfgs-maxstep BFGS_MAXSTEP]
                      [--fire-min-steps FIRE_MIN_STEPS]
                      [--fire-min-maxstep FIRE_MIN_MAXSTEP]
                      [--pre-min-ase-order {fire-first,bfgs-first}]
                      [--skip-bfgs | --no-skip-bfgs]
                      [--bfgs-polish-max-fmax BFGS_POLISH_MAX_FMAX]
                      [--rescue-fire-fmax RESCUE_FIRE_FMAX]
                      [--quiet-bfgs | --no-quiet-bfgs] [--verbose-bfgs]
                      [--bfgs-log-every N]
                      [--charmm-pre-minimize | --no-charmm-pre-minimize]
                      [--calculator-pre-minimize | --no-calculator-pre-minimize]
                      [--calculator-safe-grms KCAL] [--pre-min-safe-grms KCAL]
                      [--geometry-packing-safe-grms KCAL]
                      [--geometry-packing-fire-bfgs-crossover-grms KCAL]
                      [--charmm-sd-steps CHARMM_SD_STEPS]
                      [--charmm-abnr-steps CHARMM_ABNR_STEPS]
                      [--charmm-tolenr CHARMM_TOLENR]
                      [--charmm-tolgrd CHARMM_TOLGRD]
                      [--charmm-nbxmod CHARMM_NBXMOD]
                      [--rescue-minimize | --no-rescue-minimize]
                      [--max-fmax-after-min MAX_FMAX_AFTER_MIN]
                      [--n-equil N_EQUIL] [--save-equil-traj]
                      [--equil-traj-interval EQUIL_TRAJ_INTERVAL]
                      [--n-prod N_PROD]
                      [--repeats-per-window REPEATS_PER_WINDOW]
                      [--interval INTERVAL]
                      [--min-com-start-distance MIN_COM_START_DISTANCE]
                      [--no-fix-com] [--no-stationary] [--ml-cutoff ML_CUTOFF]
                      [--ml-switch-width ML_SWITCH_WIDTH]
                      [--mm-switch-on MM_SWITCH_ON]
                      [--mm-cutoff MM_SWITCH_WIDTH]
                      [--mlpot-mm-internal-scale W]
                      [--mm-nonbond-mode {jax_mic,periodic_external}]
                      [--lr-solver {auto,mic,scafacos,jax_pme,nvalchemiops_pme,ewald}]
                      [--jax-pme-method {ewald,pme,p3m}] [--jax-pme-sr-cutoff A]
                      [--jax-pme-dispersion | --no-jax-pme-dispersion]
                      [--scafacos-method SCAFACOS_METHOD]
                      [--periodic-charmm-vdw | --no-periodic-charmm-vdw]
                      [--charmm-zero-energy-terms TERMS]
                      [--include-mm | --no-include-mm]
                      [--mm-charge-mode {fixed,latent,fixed_plus_latent,latent_mean}]
                      [--mm-charge-correction]
                      [--mm-latent-charge-template MM_LATENT_CHARGE_TEMPLATE]
                      [--jax-mm-spoof] [--jax-mm-spoof-psf JAX_MM_SPOOF_PSF]
                      [--residue RESIDUE] [--skip-jit-warmup]
                      [--auto-warmup-mlpot-jax | --no-auto-warmup-mlpot-jax]
                      [--resume] [--config CONFIG] [--job-id JOB_ID] [--run-all]
                      [--resume-campaign]
                      [--campaign-output-dir CAMPAIGN_OUTPUT_DIR]
                      [--continue-from CONTINUE_FROM]
                      [--continue-from-frame CONTINUE_FROM_FRAME]
                      [--continue-velocities | --no-continue-velocities]
                      [--handoff-write-res | --no-handoff-write-res]
                      [--handoff-template-res HANDOFF_TEMPLATE_RES]
                      [--handoff-pre-minimize]
                      [--handoff-quality-gate | --no-handoff-quality-gate]
                      [--handoff-quality-fmax-eVA HANDOFF_QUALITY_FMAX_EVA]
                      [--handoff-quality-action {minimize,warn,error}]
                      [--handoff-velocity-remove-drift | --no-handoff-velocity-remove-drift]
                      [--handoff-require-cell | --no-handoff-require-cell]
                      [--jaxmd-minimize-steps JAXMD_MINIMIZE_STEPS]
                      [--jaxmd-pbc-minimize-steps JAXMD_PBC_MINIMIZE_STEPS]
                      [--jaxmd-fire-skip-max-f-eVA JAXMD_FIRE_SKIP_MAX_F_EVA]
                      [--jax-md-update-interval JAX_MD_UPDATE_INTERVAL]
                      [--jax-md-skin-distance JAX_MD_SKIN_DISTANCE]
                      [--steps-per-recording STEPS_PER_RECORDING]
                      [--evaluate-npz PATH] [--evaluate-output EVALUATE_OUTPUT]
                      [--evaluate-frame EVALUATE_FRAME]
                      [--evaluate-forces-npz PATH] [--evaluate-traj PATH]
                      [--no-evaluate-save-artifacts]
                      [--evaluate-reference-npz PATH]
                      [--evaluate-reference-frame EVALUATE_REFERENCE_FRAME]
                      [--evaluate-reference-energy-unit {hartree,ev,kcal_mol}]
                      [--evaluate-reference-force-unit {hartree_bohr,ev_ang}]
                      [--evaluate-compare-output EVALUATE_COMPARE_OUTPUT]
                      [--dyna-probe] [--dyna-probe-nstep DYNA_PROBE_NSTEP]
                      [--dyna-probe-dt-fs DYNA_PROBE_DT_FS]
                      [--dyna-probe-output DYNA_PROBE_OUTPUT]
                      [--optimize-cutoffs] [--reference-npz PATH]
                      [--optimize-output OPTIMIZE_OUTPUT]
                      [--ml-switch-width-grid ML_SWITCH_WIDTH_GRID]
                      [--mm-switch-on-grid MM_SWITCH_ON_GRID]
                      [--mm-switch-width-grid MM_SWITCH_WIDTH_GRID]
                      [--energy-weight ENERGY_WEIGHT]
                      [--force-weight FORCE_WEIGHT] [--max-frames MAX_FRAMES]
                      [--no-run-advice] [--no-stage-summary] [--mlpot-profile]

Run predefined MD setups (free-space NVE/NVT, periodic NVE/NVT, periodic NPT,
lambda TI for arbitrary compositions) for arbitrary residue compositions. Runs
mmml.cli.run.md_pbc_suite (ASE, JAX-MD, or CHARMM MLpot) and
mmml.cli.run.lambda_dynamics (lambda_ti). MBAR: mmml lambda-mbar --run-dir
<output-dir>.

options:
  -h, --help            show this help message and exit
  --setup {free_nve,free_nvt,free_thermalize,pbc_nve,pbc_nvt,pbc_thermalize,pbc_npt,lambda_ti,pycharmm_minimize,pycharmm_full,all}
                        Simulation setup preset. lambda_ti: alchemical TI with
                        CHARMM+MMML minimization per λ window (--lambda-md-mode,
                        --backend ase|jaxmd); mmml lambda-mbar afterward.
                        pycharmm_minimize: CHARMM MLpot SD only (--backend
                        pycharmm). pycharmm_full: mini → heat → NVE → equi →
                        prod (--backend pycharmm). pbc_* with --backend
                        pycharmm: same staged pipeline with CHARMM
                        crystal/IMAGE. free_nve/free_nvt with --backend
                        pycharmm: mini + NVE or mini + heat.
                        free_thermalize/pbc_thermalize: minimize then heat to
                        --temperature (ramp by default) and optional NVT equi
                        (pbc only); use presets/thermalize.yaml.
  --backend {auto,ase,jaxmd,pycharmm}
                        MD engine: ase runs md_pbc_suite.ase; jaxmd runs
                        md_pbc_suite.jaxmd; pycharmm runs CHARMM MLpot (vacuum,
                        non-PBC: SD + NVE/NVT). auto uses ASE for vacuum
                        (free_*) and fixed-volume PBC, JAX-MD for NPT. Use jaxmd
                        with --setup free_nve or free_nvt for open-boundary JAX-
                        MD.
  --checkpoint CHECKPOINT
                        Model checkpoint path.
  --mbd-checkpoint MBD_CHECKPOINT
                        Optional learned MBD dispersion checkpoint. On ASE/JAX-
                        MD and PyCHARMM hybrid paths, adds E += mbd_weight *
                        E_mbd. When omitted on the hybrid path, auto-loads from
                        the model checkpoint's recorded mbd_checkpoint if that
                        path exists locally. With --jaxmd-unified --ff zbl-mbd-
                        multipoles, used once at build to freeze per-atom
                        C6/C8/C10 for the classical pairwise dispersion term
                        (default: bundled example).
  --mbd-weight MBD_WEIGHT
                        Weight for the --mbd-checkpoint correction (default 1.0;
                        match training).
  --multipole-checkpoint MULTIPOLE_CHECKPOINT
                        Optional learned multipole checkpoint. With --jaxmd-
                        unified --ff zbl-mbd-multipoles, used once at build to
                        freeze fragment multipoles for classical pair
                        electrostatics during rigid MC (default: bundled
                        example).
  --sampler {md,rigid}  Propagator for --jaxmd-unified: md (JaxmdDriver) or
                        rigid (RigidBodySampler Metropolis MC over whole
                        monomers). Default: md.
  --ff {cgenff,zbl-mbd-multipoles}
                        Intermolecular FF preset for --jaxmd-unified. cgenff:
                        CHARMM/CGenFF mm_nonbonded only (default when --sampler
                        rigid and no --checkpoint). zbl-mbd-multipoles:
                        intermolecular ZBL (defaults) + fixed multipoles + fixed
                        C6 dispersion. Omit for hybrid ml_intra+mm_nonbonded
                        when a checkpoint is set.
  --jaxmd-unified       EXPERIMENTAL: run --backend jaxmd through the unified
                        mmml.md pipeline (mmml.cli.run.md_system_unified)
                        instead of the legacy md_pbc_suite.jaxmd inline loop.
                        Only supports the packmol composition builder; see
                        docs/md-cg-unification-design.md for scope and status.
  --electrostatics-damping-sigma ELECTROSTATICS_DAMPING_SIGMA
                        Override learned-charge Coulomb erf damping sigma in
                        Angstrom; set 0 to disable.
  --output-dir OUTPUT_DIR
                        Output directory for artifacts.
  --job-name JOB_NAME   Logical job id for this run. Writes
                        artifacts/md_system/jobs/<job-name>.json with the last
                        command and options (default: basename of --output-dir
                        when set).
  --jobs-dir JOBS_DIR   Directory for per-job last-run manifest JSON files
                        (default: artifacts/md_system/jobs).
  --template-pdb TEMPLATE_PDB
                        Monomer template PDB path.
  --n-molecules N_MOLECULES
                        Number of molecules for single-residue runs.
  --composition COMPOSITION
                        Residue composition: comma-separated RES:N entries, e.g.
                        MEOH:5,TIP3:5. A bare RES (no ':N') implies a single
                        copy (N=1); when this option is set, --n-molecules is
                        not passed to the backend (use DCM:10 for ten DCM).
  --spacing SPACING     Target minimum random COM spacing in Angstrom.
  --ps PS               Simulation length in ps.
  --dt-fs DT_FS         Timestep in fs.
  --traj-chunk-frames TRAJ_CHUNK_FRAMES
                        Split trajectory output into multi-file chunks with at
                        most this many frames (0 = single file).
  --traj-export-molecular-wrap
                        JAX-MD only: molecular COM wrap when writing HDF5/.traj
                        (slower).
  --temperature TEMPERATURE
                        Target temperature in K (NVT/NPT).
  --nvt-integrator {auto,nhc,langevin}
                        Integrator for NVT in ASE route. auto=nhc for
                        homogeneous, langevin for mixed composition.
  --pressure PRESSURE   Target pressure in atm (NPT).
  --seed SEED           Random seed for initial placement and velocities.
  --builder {gas,liquid,crystal}
                        Grid-based builder (skips Packmol): gas=open grid,
                        liquid=cube/sphere grid plus CHARMM refinement,
                        crystal=PyXtal. Default for --composition is Packmol
                        unless --builder or --no-packmol is set.
  --min-intermonomer-atom-distance MIN_INTERMONOMER_ATOM_DISTANCE
                        Abort if atoms from different monomers get closer than
                        this distance in Angstrom (<=0 disables).
  --packmol, --no-packmol
                        Pack --composition with Packmol (default when
                        --composition is set). Use --no-packmol or --builder
                        liquid/gas for grid placement.
  --packmol-placement {cube,sphere}
                        Initial placement constraint: cube (default) or sphere
                        (--packmol-radius).
  --packmol-sphere, --no-packmol-sphere
                        Legacy alias for --packmol-placement sphere.
  --packmol-radius Å    Sphere radius in Angstrom (required for --packmol-
                        placement sphere).
  --packmol-center CX CY CZ
                        Initial placement center in Angstrom (default: 0 0 0).
  --packmol-tolerance PACKMOL_TOLERANCE
                        Packmol distance tolerance (Å) for cluster placement
                        (default: 2.0).
  --reuse-packmol-cache, --no-reuse-packmol-cache
                        pycharmm: reuse disk cache for Packmol cluster builds
                        (default: on).
  --rebuild-packmol     pycharmm: ignore Packmol cache and rebuild placement.
  --packmol-cache-dir PACKMOL_CACHE_DIR
                        pycharmm: Packmol cache root (default: output-
                        dir/.packmol_cache or MMML_PACKMOL_CACHE).
  --save-run-state      pycharmm: after staged MD, save positions/velocities +
                        metadata (Orbax if installed, else NPZ; PhysNet stays in
                        --checkpoint).
  --run-state-dir RUN_STATE_DIR
                        pycharmm: run-state output directory (default: output-
                        dir/run_state).
  --overlap-run-state-dir OVERLAP_RUN_STATE_DIR
                        pycharmm: overlap-chunk geometry sidecar directory
                        (default: output-dir/run_state/overlap).
  --overlap-run-state-every-chunks OVERLAP_RUN_STATE_EVERY_CHUNKS
                        pycharmm: save overlap run-state sidecar every N
                        successful chunks (0=off).
  --flat-bottom-radius Å
                        Harmonic flat-bottom on system COM: V=0 inside radius R,
                        V=k(|d|-R)^2 outside. Independent of --packmol-radius.
                        Vacuum: center at origin; PBC: MIC to box center.
  --flat-bottom-k eV/Å²
                        Flat-bottom force constant when COM is outside --flat-
                        bottom-radius (default: 1.0).
  --flat-bottom-selection FLAT_BOTTOM_SELECTION
                        pycharmm: CHARMM atom selection for MMFP wall (default:
                        all).
  --flat-bottom-mode {system,monomer}
                        Flat-bottom anchor: system = one restraint on mass-
                        weighted cluster COM; monomer = sum of harmonic
                        restraints on each monomer COM (same R and k).
  --min-com-restraint-distance Å
                        Pairwise inter-monomer COM lower wall. Adds
                        0.5*k*(r_min-r)^2 when COM distance r < r_min (default:
                        disabled).
  --min-com-restraint-k eV/Å²
                        Force constant for --min-com-restraint-distance
                        (default: 1.0).
  --extra-args ...      Additional raw args forwarded to the underlying script;
                        put this option last.
  --fix-resids FIX_RESIDS
                        pycharmm: monomers held in SD pass 2 (comma-separated
                        1-based resids; default: none — use e.g. 1 or 1,3 to
                        anchor monomers during minimize)
  --constrain-resids CONSTRAIN_RESIDS
                        pycharmm: freeze these resids during MD (comma-
                        separated)
  --no-fix              pycharmm: skip constrained SD pass 2
  --mini-nstep MINI_NSTEP
                        pycharmm: SD steps per minimization pass before dynamics
  --no-pre-minimize     pycharmm: skip SD minimization before dynamics
  --echeck ECHECK       pycharmm: CHARMM ECHECK tolerance (kcal/mol); use --no-
                        echeck to disable
  --no-echeck           pycharmm: disable CHARMM ECHECK early stop
  --no-echeck-heat      pycharmm: disable CHARMM ECHECK during the heat stage
                        only (equi/prod still use --echeck)
  --allow-incomplete-dynamics
                        pycharmm: do not fail when dynamics stops early or the
                        stage DCD is truncated
  --nprint NPRINT       pycharmm: print SD minimization energy every N steps
                        (default: 50)
  --dyn-nprint DYN_NPRINT
                        pycharmm: print dynamics energy every N steps (default:
                        500)
  --dyn-iprfrq DYN_IPRFRQ
                        pycharmm: detailed dynamics status every N steps
                        (default: 2000)
  --heat-ihtfrq N       pycharmm: heat rescale cadence for bussi (ASE) or scale
                        (CHARMM ihtfrq); 0 = match --dyn-freq-cadence, else
                        --dyn-nprint. Ignored for hoover.
  --heat-bussi-taut PS  pycharmm: Bussi coupling time taut (ps) when --heat-
                        thermostat bussi.
  --heat-thermostat {bussi,scale,hoover}
                        pycharmm heat stage: bussi=ASE Bussi rescaling
                        (default); scale=CHARMM IHTFRQ; hoover=CHARMM Hoover
                        NVT.
  --heat-firstt K       pycharmm: heat start temperature (FIRSTT). Default
                        0.2×--temperature; use 0 for cold start + IHTFRQ
                        scaling.
  --heat-finalt K       pycharmm: heat end temperature (FINALT). Default
                        --temperature; DCM:9 stability often uses 240.
  --heat-mode {ramp,hold}
                        pycharmm: heat protocol — ramp=gradual FIRSTT→FINALT
                        (default); hold=Boltzmann at target T then NVT hold (no
                        ramp).
  --heat-hoover-tmass M
                        pycharmm Hoover heat only: thermostat mass tmass
                        (kcal·mol⁻¹·ps²). Default clamps PSF tmass to 400–1200.
                        Lower = stronger T coupling.
  --nve-boltzmann-temp K
                        pycharmm: Boltzmann velocity temperature before NVE
                        after mini. Default 0.2×--temperature; use 50–100 K for
                        a gentler start than 300 K.
  --nve-max-f-start-eVA EV_PER_A
                        jaxmd: refuse to start NVE when post-FIRE max atomic |F|
                        exceeds this value in eV/Å (default: 1.5; <=0 disables).
                        Hybrid liquid handoffs often land near ~1 eV/Å after
                        FIRE.
  --nve-etot-drift-abort-eV EV
                        jaxmd: abort NVE when |E_tot| drift exceeds this (eV;
                        <=0 disables).
  --nve-etot-drift-rescue, --no-nve-etot-drift-rescue
                        jaxmd: on NVE E_tot drift, rewind and try
                        NL/FIRE/CHARMM/rethermalize before aborting (default:
                        on).
  --nve-etot-drift-rescue-attempts NVE_ETOT_DRIFT_RESCUE_ATTEMPTS
                        jaxmd: max mid-run E_tot drift repair attempts (default:
                        5).
  --nve-etot-drift-rescue-fire-steps NVE_ETOT_DRIFT_RESCUE_FIRE_STEPS
                        jaxmd: FIRE steps per E_tot drift rescue (default: 100).
  --nve-etot-drift-rescue-grace-eV EV
                        jaxmd: base E_tot gate after each drift rescue
                        (progressive g→1.5g→2g…; default: 2.5).
  --nve-etot-drift-rescue-dt-scale NVE_ETOT_DRIFT_RESCUE_DT_SCALE
                        jaxmd: multiply MD dt on drift-rescue dt_halve (default:
                        0.5).
  --nve-etot-drift-rescue-min-dt-fs NVE_ETOT_DRIFT_RESCUE_MIN_DT_FS
                        jaxmd: minimum MD dt (fs) when backing off (default:
                        0.05).
  --heat-comp-damp, --no-heat-comp-damp
                        pycharmm: experimental COMP force copy before heat
                        (default: off).
  --heat-comp-hydrogen-only, --no-heat-comp-hydrogen-only
                        pycharmm: with --heat-comp-damp, select high-|F|
                        hydrogens only (default). --no-heat-comp-hydrogen-only =
                        all atom types.
  --heat-comp-force-min KCAL
                        pycharmm: |F| threshold for heat COMP selection
                        (kcal/mol/Å).
  --heat-comp-force-scale HEAT_COMP_FORCE_SCALE
                        pycharmm: scale for forces copied into COMP during heat.
  --skip-energy-show    pycharmm: skip CHARMM energy.show() (MPI/cluster
                        segfault guard)
  --show-energy, --no-show-energy
                        pycharmm: print CHARMM energy tables (off by default)
  --quiet               pycharmm: reduce CHARMM console output
  --verbose             Print CHARMM BLOCK Rich summaries and extra MLpot setup
                        detail
  --dcd-nsavc DCD_NSAVC
                        pycharmm: DCD frame every N integration/SD steps
  --dcd-interval-ps PS  pycharmm: DCD save interval in ps (overrides --dcd-nsavc
                        when set)
  --dcd-max-frames N    pycharmm: cap DCD output to ~N frames per stage when
                        --dcd-interval-ps is unset (0 = no cap)
  --save-forces-npz     pycharmm: write <output-dir>/forces.npz during dynamics
  --forces-npz-interval FORCES_NPZ_INTERVAL
                        pycharmm: force NPZ save every N steps (default: 1)
  --no-scale-mini-nstep
                        pycharmm: do not auto-increase --mini-nstep for large
                        clusters
  --no-scale-echeck     pycharmm: do not auto-loosen --echeck for large clusters
                        (default scales with N monomers/atoms)
  --allow-high-grms     pycharmm: start dynamics even if post-min GRMS is high
  --no-scale-max-grms   pycharmm: use --max-grms-before-dyn exactly (skip size-
                        aware scaling from per-monomer hybrid GRMS tails)
  --max-grms-before-dyn MAX_GRMS_BEFORE_DYN
                        pycharmm: abort if post-min GRMS exceeds this
                        (kcal/mol/Å)
  --test-first          pycharmm: CHARMM TEST FIRSt after MLpot SD minimization
  --test-first-tol TEST_FIRST_TOL
                        pycharmm: TEST FIRSt tolerance (default: 0.005)
  --test-first-step TEST_FIRST_STEP
                        pycharmm: TEST FIRSt finite-difference step in Å
                        (default: 1e-4)
  --test-first-resids TEST_FIRST_RESIDS
                        pycharmm: limit derivative tests to these resids
                        (default: all atoms)
  --test-first-charmm   pycharmm: also run CHARMM TEST FIRSt (ANALYTIC omits
                        MLpot USER energy)
  --test-first-update-nbonds
                        pycharmm: UPDATE nonbond lists before CHARMM TEST FIRSt
  --ml-batch-size N     pycharmm: chunk PhysNet batches (auto: 256 on GPU / 64
                        on CPU for n>=40; or MMML_MLPOT_ML_BATCH_SIZE). DCM:90
                        try 256-512 on one GPU.
  --ml-gpu-count N      pycharmm: parallel PhysNet chunks on N local GPUs
                        (default 1; or MMML_MLPOT_N_GPUS). Set
                        CUDA_VISIBLE_DEVICES to the GPU ids to use.
  --max-pairs N         PBC: cell-list MM pair buffer size (auto from N and box
                        when unset). Increase if you see 'MM Pair List
                        Truncated' during MLpot mini/MD.
  --ml-spatial-mpi      pycharmm: per-rank spatial ML decomposition when MPI
                        size>1 (PBC only; or MMML_MLPOT_SPATIAL_MPI=1). Use with
                        MMML_MPI_NP>1 and --ml-gpu-count 1.
  --charmm-omp-threads N
                        pycharmm: set MMML_CHARMM_OMP_THREADS before MPI-linked
                        CHARMM bootstrap (default 1; CPU performance experiment
                        knob).
  --ml-compute-dtype {float32,float64}
                        JAX dtype for ML/MM hybrid interior (default: float32,
                        or MMML_ML_DTYPE / JAX_ENABLE_X64=1 → float64). CHARMM
                        I/O stays float64.
  --ml-max-active-dimers N
                        pycharmm: sparse ML dimer slot cap per step (PBC default
                        max(1000, 6*n_monomers); free-space default all unique
                        dimers). Run scripts/validate_mlpot_sparse_dimers.py to
                        check.
  --md-stages MD_STAGES
                        pycharmm: comma-separated mini,heat,nve,equi,prod
                        (default from --setup)
  --md-stage {mini,heat,nve,equi,prod}
                        pycharmm: run one stage only (implies prior artifacts
                        under --output-dir)
  --tag TAG             pycharmm: artifact tag for staged outputs (default: from
                        composition)
  --ps-heat PS_HEAT     pycharmm: heating length in ps (default: 10)
  --charmm-mm-pretreat  pycharmm: CGENFF minimize + CHARMM heat/equi/prod before
                        MLpot (no PhysNet); see --charmm-mm-pretreat-ps-* and
                        --charmm-mm-pretreat-heat-nstep. Skipped with --liquid-
                        prep unless --charmm-mm-pretreat-with-liquid-prep;
                        skipped on handoff unless --charmm-mm-pretreat-on-
                        handoff
  --charmm-mm-pretreat-on-handoff
                        pycharmm: run CHARMM MM pretreat even when
                        jaxmd/PyCHARMM handoff coords are already in memory
                        (default: pretreat only on cold composition starts)
  --charmm-mm-pretreat-with-liquid-prep
                        pycharmm: run CHARMM MM pretreat heat/equi/prod even
                        when --liquid-prep already built the box (default: skip
                        redundant pretreat on dense-liquid prep)
  --charmm-mm-pretreat-ps-heat PS
                        pycharmm: pretreat CHARMM heat length in ps (overrides
                        --charmm-mm-pretreat-heat-nstep when set)
  --charmm-mm-pretreat-heat-nstep N
                        pycharmm: integration steps for pretreat CHARMM heat
                        (default: 2000)
  --charmm-mm-pretreat-ps-equi PS
                        pycharmm: pretreat CHARMM NPT equilibration in ps (0
                        skips; default: 0)
  --charmm-mm-pretreat-ps-prod PS
                        pycharmm: pretreat CHARMM NPT production in ps (0 skips;
                        default: 0)
  --charmm-mm-pretreat-mini-sd N
                        pycharmm: pretreat CHARMM SD steps (default: --charmm-
                        sd-steps)
  --charmm-mm-pretreat-mini-abnr N
                        pycharmm: pretreat CHARMM ABNR steps (default: --charmm-
                        abnr-steps)
  --charmm-mm-pretreat-dt-fs FS
                        Pretreat CHARMM dynamics timestep in fs (default: 1.0).
                        Independent of MLpot --dt-fs.
  --charmm-mm-pretreat-temperature K
                        Pretreat CHARMM heat/equi/prod temperature (default:
                        --temperature).
  --charmm-mm-pretreat-pressure ATM
                        Pretreat CHARMM NPT reference pressure (default: --npt-
                        pressure or --pressure).
  --charmm-mm-pretreat-echeck KCAL
                        ECHECK for pretreat CPT equi/prod and mini box equil
                        (kcal/mol). Default: disabled. Use 0 or a negative value
                        to keep ECHECK off.
  --charmm-mm-pretreat-inbfrq N
                        Pretreat CHARMM nonbond list rebuild cadence (inbfrq).
                        Default scales with --charmm-mm-pretreat-dt-fs (400 at 2
                        fs vs 50 for MLpot).
  --charmm-mm-pretreat-imgfrq N
                        Pretreat PBC image/HB list cadence
                        (imgfrq/ihbfrq/ilbfrq). Default matches pretreat inbfrq.
  --charmm-mm-pretreat-ixtfrq N
                        Pretreat crystal transform cadence (ixtfrq; default
                        scales with pretreat dt).
  --ps-nve PS_NVE       pycharmm: NVE length in ps (default: --ps)
  --ps-equi PS_EQUI     pycharmm: NPT equilibration length in ps (default: 50)
  --ps-prod PS_PROD     pycharmm: production length in ps (default: --ps)
  --npt-thermostat {hoover,berendsen}
                        pycharmm: NPT temperature control for equi/prod
                        (default: hoover)
  --npt-pressure NPT_PRESSURE
                        pycharmm: NPT reference pressure in atm for equi/prod
                        (default: 1.0)
  --npt-pgamma NPT_PGAMMA
                        pycharmm: CPT barostat Langevin collision frequency in
                        1/ps (default: 5; 0 disables barostat coupling)
  --n-heat-segments N_HEAT_SEGMENTS
                        pycharmm: split heating into short chained restart
                        segments
  --n-equi-segments N_EQUI_SEGMENTS
                        pycharmm: split NPT equilibration into chained restart
                        segments
  --n-prod-segments N_PROD_SEGMENTS
                        pycharmm: split production into chained restart segments
  --bonded-mm-mini, --no-bonded-mm-mini
                        pycharmm: bonded-only SD if MM bonded strain exceeds
                        post-MM-pre-min baseline (default: off — opt in; can
                        stall on PBC crystal free / CGENFF APPEND; heat is
                        always checked when enabled)
  --bonded-mm-mini-after BONDED_MM_MINI_AFTER
                        pycharmm: comma-separated stages to check (default:
                        mini,heat; heat always)
  --bonded-mm-mini-steps BONDED_MM_MINI_STEPS
                        pycharmm: bonded recovery mini steps (default: 50)
  --bonded-recovery-backend {auto,jax,charmm,sidecar}
                        pycharmm: bonded recovery minimizer — JAX FIRE without
                        MLpot detach (auto tries JAX first), CHARMM SD, isolated
                        subprocess (sidecar), or auto (default: auto)
  --bonded-mm-mini-always
                        pycharmm: bonded SD after every --bonded-mm-mini-after
                        stage (ignore strain margins)
  --bonded-mm-internal-margin BONDED_MM_INTERNAL_MARGIN
                        pycharmm: deprecated alias for --bonded-mm-grms-margin
                        (default: 0)
  --bonded-mm-grms-margin BONDED_MM_GRMS_MARGIN
                        pycharmm: kcal/mol/Å above baseline GRMS before recovery
  --bonded-mm-internal-energy-margin BONDED_MM_INTERNAL_ENERGY_MARGIN
                        pycharmm: kcal/mol above baseline bonded internal before
                        recovery (0=off)
  --bonded-mm-angl-margin BONDED_MM_ANGL_MARGIN
                        pycharmm: kcal/mol above baseline ANGL before recovery
                        (0=off)
  --bonded-mm-max-angl-kcal BONDED_MM_MAX_ANGL_KCAL
                        pycharmm: abort after MM pre-min if ANGL exceeds this
                        (e.g. 15)
  --bonded-mm-max-internal-kcal BONDED_MM_MAX_INTERNAL_KCAL
                        pycharmm: abort after MM pre-min if bonded internal
                        exceeds this
  --allow-high-bonded-strain
                        pycharmm: continue when max-angl/max-internal pre-min
                        limits exceeded
  --restart-from RESTART_FROM
                        pycharmm: CHARMM .res restart for first dynamics stage
  --from-psf FROM_PSF   Load PSF instead of rebuilding cluster (pycharmm, ase,
                        jaxmd). Requires --from-crd for certified liquid-box /
                        mini handoff.
  --from-crd FROM_CRD   Load CRD with --from-psf (pycharmm, ase, jaxmd). Skips
                        Packmol; sibling box.json overrides --box-size when
                        present.
  --skip-cluster-build  pycharmm: skip Packmol/IC; use --from-psf/--from-crd or
                        prior mini artifacts
  --skip-if-crd-exists  pycharmm: skip MLpot SD when mini CRD already exists in
                        --output-dir
  --no-save-vmd-topology
                        pycharmm: skip cluster_for_vmd PSF/PDB before MLpot
                        registration
  --free-space          pycharmm: force vacuum (no PBC). free_nve/free_nvt
                        setups are vacuum by default; use to override when
                        --box-size is also set.
  --mlpot-pbc           pycharmm: enable ML MIC / periodic dimer lists (default
                        for pbc_* setups). With free_* + --box-size, CHARMM uses
                        loose PBC unless this flag is set.
  --dyn-inbfrq DYN_INBFRQ
                        pycharmm: CHARMM inbfrq for dynamics (-1=heuristic,
                        50=vacuum default)
  --dyn-imgfrq N        pycharmm PBC: image/HB list rebuild every N steps
                        (default 50; larger=faster)
  --dyn-freq-cadence N  pycharmm: align heat/print cadence (ihtfrq, nprint, …)
                        to N steps; decoupled from DCD nsavc (default: 500).
                        Overlap CPT chunks still disable interior inbfrq/imgfrq.
                        Use 0 for legacy behavior.
  --pre-nve-charmm-update, --no-pre-nve-charmm-update
                        pycharmm: ENER+UPDATE after mini before vacuum NVE
                        (default: on)
  --lambda-md-mode {free_nve,free_nvt,pbc_nve,pbc_nvt}
                        lambda_ti: MD ensemble (vacuum/PBC × NVE/NVT); use
                        --backend ase or jaxmd.
  --couple-residues COUPLE_RESIDUES
                        lambda_ti: 1-based residue numbers sharing λ (comma-
                        separated, cluster order).
  --lambda-windows LAMBDA_WINDOWS [LAMBDA_WINDOWS ...]
                        lambda_ti: shared λ values for coupled residues.
  --pre-min-steps PRE_MIN_STEPS
                        lambda_ti: MMML BFGS steps per window.
  --pre-min-fmax PRE_MIN_FMAX
                        lambda_ti: MMML BFGS fmax (eV/Å).
  --min-steps MIN_STEPS
                        lambda_ti: alias for --pre-min-steps.
  --min-fmax MIN_FMAX   lambda_ti: alias for --pre-min-fmax.
  --bfgs-maxstep BFGS_MAXSTEP
  --fire-min-steps FIRE_MIN_STEPS
                        ASE FIRE steps during calculator pre-minimize / rescue
                        (default 200).
  --fire-min-maxstep FIRE_MIN_MAXSTEP
                        ASE FIRE max atomic displacement per step in Å (default
                        0.2).
  --pre-min-ase-order {fire-first,bfgs-first}
                        ASE hybrid pre-min order for jaxmd/ase: fire-first
                        (default) runs FIRE on rough surfaces and BFGS only to
                        polish; bfgs-first is legacy.
  --skip-bfgs, --no-skip-bfgs
                        Skip ASE BFGS during pre-minimization (FIRE only).
                        Default ON: plain ASE BFGS trusts a quadratic model and
                        takes long steps; on a hybrid ML PES it descends into a
                        hole outside the training data. Observed at 0 K on a
                        real acetone box: E -7427.7 -> -7545.8 eV while max|F|
                        rose 0.199 -> 36.3, i.e. reachable from the relaxed
                        structure by following forces, no thermal barrier
                        needed. FIRE converged cleanly on the same system (1.40
                        -> 0.17). Pass --no-skip-bfgs to re-enable once BFGS is
                        fixed (the line search + force-spike abort in
                        mlpot/calculator_minimize.py are the likely fix).
                        Overrides --pre-min-ase-order.
  --bfgs-polish-max-fmax BFGS_POLISH_MAX_FMAX
                        Soft hybrid max|F| gate (eV/Å; default 1.0): BFGS polish
                        only at/below this with fire-first; also skip MM CHARMM
                        rescue when already soft.
  --rescue-fire-fmax RESCUE_FIRE_FMAX
                        FIRE force convergence threshold in eV/Å for calculator
                        rescue (default 0.05).
  --quiet-bfgs, --no-quiet-bfgs
                        Suppress ASE BFGS/FIRE progress lines entirely (default:
                        compact progress).
  --verbose-bfgs        Print the full ASE BFGS/FIRE step table instead of
                        compact progress.
  --bfgs-log-every N    Compact BFGS/FIRE log every N steps (default: ~10 lines
                        per run).
  --charmm-pre-minimize, --no-charmm-pre-minimize
                        lambda_ti: CHARMM SD/ABNR before MMML BFGS (default on).
  --calculator-pre-minimize, --no-calculator-pre-minimize
                        lambda_ti: MMML-calculator BFGS after CHARMM (default
                        on).
  --calculator-safe-grms KCAL
                        Hybrid GRMS (kcal/mol/Å) to stop pre-SD ASE FIRE/BFGS
                        early (default: 30; 0 disables).
  --pre-min-safe-grms KCAL
                        Alias/fallback for --calculator-safe-grms during pre-
                        minimize / pre-dynamics FIRE/BFGS (default: inherit
                        calculator-safe-grms or 30; 0 disables).
  --geometry-packing-safe-grms KCAL
                        Hybrid GRMS (kcal/mol/Å) to stop geometry-packing
                        FIRE/BFGS early (default: inherit calculator-safe-grms
                        or 30; 0 disables).
  --geometry-packing-fire-bfgs-crossover-grms KCAL
                        Run FIRE before BFGS in geometry packing when hybrid
                        GRMS exceeds this threshold (default: 30 kcal/mol/Å).
  --charmm-sd-steps CHARMM_SD_STEPS
  --charmm-abnr-steps CHARMM_ABNR_STEPS
  --charmm-tolenr CHARMM_TOLENR
  --charmm-tolgrd CHARMM_TOLGRD
  --charmm-nbxmod CHARMM_NBXMOD
  --rescue-minimize, --no-rescue-minimize
                        lambda_ti: ASE FIRE if BFGS fmax stays high.
  --max-fmax-after-min MAX_FMAX_AFTER_MIN
  --n-equil N_EQUIL     lambda_ti: equilibration steps per window.
  --save-equil-traj     lambda_ti: write …_eq.traj under trajectories/ during
                        equilibration (debug).
  --equil-traj-interval EQUIL_TRAJ_INTERVAL
                        lambda_ti: equil trajectory frame interval (default:
                        --interval).
  --n-prod N_PROD       lambda_ti: production steps per window.
  --repeats-per-window REPEATS_PER_WINDOW
                        lambda_ti: independent repeats per λ window.
  --interval INTERVAL   lambda_ti: sample dU/dλ every N production steps.
  --min-com-start-distance MIN_COM_START_DISTANCE
                        lambda_ti: minimum inter-monomer COM distance after
                        placement (Å).
  --no-fix-com          lambda_ti: disable ASE FixCom (COM position can drift
                        during MD).
  --no-stationary       lambda_ti: skip Stationary/ZeroRotation on velocity init
                        (with --no-fix-com, COM can translate).
  --ml-cutoff ML_CUTOFF
                        lambda_ti: ML cutoff (Å).
  --ml-switch-width, --ml-cutoff-distance ML_SWITCH_WIDTH
                        COM-distance width (Å) of the ML→MM handoff for
                        pycharmm/MMML; ML is fully on below mm_switch_on - width
                        and reaches zero at mm_switch_on (default: 1.5). Does
                        not affect lambda_ti (see --ml-cutoff).
  --mm-switch-on MM_SWITCH_ON
                        COM handoff distance (Å); ML→0 / MM→1 at this separation
                        (default: 6).
  --mm-cutoff, --mm-switch-width MM_SWITCH_WIDTH
                        COM-distance width (Å) of the MM outer tail past
                        mm_switch_on (default: 5).
  --mlpot-mm-internal-scale W
                        pycharmm: scale CGENFF BOND/ANGL/DIHE on ML atoms during
                        MLpot BLOCK (0=off, 0.1=10% internal). ELEC/VDW stay off
                        in jax_mic mode; periodic_external keeps CHARMM VDW on.
  --mm-nonbond-mode {jax_mic,periodic_external}
                        pycharmm MLpot MM nonbonds: jax_mic (default) or
                        periodic_external (external Coulomb + CHARMM IMAGE VDW;
                        requires pbc_*).
  --lr-solver {auto,mic,scafacos,jax_pme,nvalchemiops_pme,ewald}
                        Long-range Coulomb backend. Default: truncated MIC in
                        the switched-MM pair loop. Opt in: jax_pme, scafacos,
                        nvalchemiops_pme, ewald (periodic_external for full-box
                        Coulomb; ewald is pure JAX, no external PME library or
                        CUDA requirement, and is the same operator --lr-solver
                        ewald trains against). Legacy alias: auto (= mic).
  --jax-pme-method {ewald,pme,p3m}
                        jax-pme method when --lr-solver=jax_pme (default: env
                        JAX_PME_METHOD or ewald).
  --jax-pme-sr-cutoff A
                        jax-pme real-space cutoff in Å (default 6.0).
  --jax-pme-dispersion, --no-jax-pme-dispersion
                        pycharmm jax_mic + jax_pme: include reciprocal r^-6 LJ
                        dispersion (default: env MMML_JAX_PME_DISPERSION or on).
                        Use --no-jax-pme-dispersion for Coulomb-only long range.
  --scafacos-method SCAFACOS_METHOD
                        ScaFaCoS fcs_init method when --lr-solver=scafacos
                        (default: ewald).
  --periodic-charmm-vdw, --no-periodic-charmm-vdw
                        CHARMM IMAGE / primary VDW on (default). --no-periodic-
                        charmm-vdw enforces VDW+IMNB≈0 via PSF/.prm reload after
                        MLpot registration (all MM modes).
  --charmm-zero-energy-terms TERMS
                        Comma-separated CHARMM ENER components to enforce at
                        zero via PSF/.prm reload after MLpot registration: vdw,
                        elec, bonded, hbond. --no-periodic-charmm-vdw implies
                        vdw.
  --include-mm, --no-include-mm
                        Include switched JAX MM pairs (LJ + MIC Coulomb) in the
                        hybrid calculator. --no-include-mm evaluates PhysNet ML
                        only (doMM=False); cutoff keys are ignored for MM pair
                        lists.
  --mm-charge-mode, --mm_charge_mode {fixed,latent,fixed_plus_latent,latent_mean}
                        Hybrid MM Coulomb charges for E_MM: fixed (q_CGenFF,
                        default), latent (neutralize(q_ML)), fixed_plus_latent
                        (q_CGenFF + neutralize(q_ML)), or latent_mean (a
                        precomputed per-monomer latent charge template tiled
                        across the box; see --mm-latent-charge-template). Modes
                        latent / fixed_plus_latent require a charges=True
                        checkpoint and are dimer-only; latent_mean works for any
                        n_monomers (liquids) (see docs/hybrid-mm-charges.md).
  --mm-charge-correction, --mm_charge_correction
                        Alias for --mm-charge-mode fixed_plus_latent on the MD
                        calculator.
  --mm-latent-charge-template, --mm_latent_charge_template MM_LATENT_CHARGE_TEMPLATE
                        Path to a .npz template from
                        scripts/compute_latent_monomer_charges.py. Required with
                        --mm-charge-mode latent_mean.
  --jax-mm-spoof        Use JAX CGenFF bonded clone instead of PhysNet for ML
                        terms (no checkpoint; box / calculator infrastructure
                        testing).
  --jax-mm-spoof-psf JAX_MM_SPOOF_PSF
                        Optional cluster PSF for --jax-mm-spoof bonded
                        parameters.
  --residue RESIDUE     Single-residue name when --composition is not set
                        (ignored when --composition is set; lambda_ti default
                        MEOH).
  --skip-jit-warmup     Skip JIT/XLA warmup. jaxmd/ase: generic XLA GPU compile
                        and pre-MD hybrid MMML eval; lambda_ti: skip first MMML
                        energy eval per window; pycharmm: skip serial auto
                        warmup-mlpot-jax before CHARMM MLpot.
  --auto-warmup-mlpot-jax, --no-auto-warmup-mlpot-jax
                        pycharmm: run serial warmup-mlpot-jax before MPI/CHARMM
                        to populate JAX_COMPILATION_CACHE_DIR (default on). Also
                        disabled by --skip-jit-warmup or
                        MMML_NO_AUTO_WARMUP_MLPOT_JAX=1.
  --resume              Resume existing work instead of starting in new output
                        directories. Campaign (--run-all): reuse output dirs and
                        skip jobs with valid handoffs. PyCHARMM retry: when re-
                        running a failed leg in the same output_dir, continues
                        from the latest .res checkpoint. lambda_ti: skip
                        complete production trajectories; redo partial prod.traj
                        files.
  --config CONFIG       YAML file with md-system options or a campaign (defaults
                        + runs/jobs).
  --job-id JOB_ID       Run one job from a campaign config (--config); honors
                        depends_on chain.
  --run-all             Run all jobs from a campaign config in dependency order
                        (in-process). If the campaign output dir already exists,
                        a new suffixed directory is used unless --resume is set.
  --resume-campaign     Alias for --resume when using --run-all or campaign YAML
                        configs.
  --campaign-output-dir CAMPAIGN_OUTPUT_DIR
                        Directory for campaign_plan.json and
                        campaign_summary.json. With --run-all, an existing path
                        gets a UUID suffix unless --resume is set.
  --continue-from CONTINUE_FROM
                        Resume from handoff path (.res, .h5, .traj, run_state/,
                        state.npz).
  --continue-from-frame CONTINUE_FROM_FRAME
                        Frame index for .h5/.traj continue-from (default: -1
                        last).
  --continue-velocities, --no-continue-velocities
                        Use velocities from handoff when present (else re-
                        thermalize).
  --handoff-write-res, --no-handoff-write-res
                        Write handoff/final.res alongside state.npz after
                        dynamics.
  --handoff-template-res HANDOFF_TEMPLATE_RES
                        Template CHARMM .res for pure-Python handoff writer.
  --handoff-pre-minimize
                        Run pre-minimization even when continuing from a
                        handoff.
  --handoff-quality-gate, --no-handoff-quality-gate
                        When continuing from handoff, evaluate initial MMML |F|
                        and optionally run pre-minimization if above --handoff-
                        quality-fmax-eVA (default: off).
  --handoff-quality-fmax-eVA HANDOFF_QUALITY_FMAX_EVA
                        |F| threshold (eV/Å) for --handoff-quality-gate
                        (default: 1.0).
  --handoff-quality-action {minimize,warn,error}
                        Action when quality gate threshold is exceeded (default:
                        minimize).
  --handoff-velocity-remove-drift, --no-handoff-velocity-remove-drift
                        Remove net momentum and rotation from handoff velocities
                        before MD (default: on).
  --handoff-require-cell, --no-handoff-require-cell
                        Require periodic cell in handoff for PBC continuation
                        (default: off).
  --jaxmd-minimize-steps JAXMD_MINIMIZE_STEPS
                        Pre-dynamics FIRE steps in the JAX-MD runner (default:
                        1000). Free space: COM-centered. PBC: molecular
                        (monomer-COM) wrapping. Best max|F| frame is restored if
                        FIRE wanders.
  --jaxmd-pbc-minimize-steps JAXMD_PBC_MINIMIZE_STEPS
                        Additional PBC FIRE steps with molecular wrapping before
                        dynamics (default: 1000; only when a cell is set).
  --jaxmd-fire-skip-max-f-eVA JAXMD_FIRE_SKIP_MAX_F_EVA
                        Skip jax-md FIRE when start max|F| ≤ this (eV/Å; default
                        0.10). Set 0 to always run FIRE backoff stages.
  --jax-md-update-interval JAX_MD_UPDATE_INTERVAL
                        JAX-MD/ASE PBC MM neighbor-list refresh interval in MD
                        steps or calculator calls (default: 1, conservative).
                        Larger values reduce host/device sync when pair-list
                        stability has been validated.
  --jax-md-skin-distance JAX_MD_SKIN_DISTANCE
                        JAX-MD/ASE PBC MM neighbor-list skin distance in Å
                        (default: 0.25).
  --steps-per-recording STEPS_PER_RECORDING
                        JAX-MD: MD steps between trajectory/HDF5 records
                        (default: 100; must be a multiple of --jax-md-update-
                        interval).
  --evaluate-npz PATH   Single-point evaluation: load positions (and optional
                        charges/LJ types) from an NPZ file, build the selected
                        backend calculator, and write energy/forces to
                        evaluate.json (no dynamics).
  --evaluate-output EVALUATE_OUTPUT
                        JSON path for --evaluate-npz results (default: <output-
                        dir>/evaluate.json).
  --evaluate-frame EVALUATE_FRAME
                        Frame index when --evaluate-npz uses trajectory keys R/Z
                        (default: 0).
  --evaluate-forces-npz PATH
                        Trajectory-style NPZ output with R, F, E, Z, N (default:
                        <output-dir>/evaluate.npz).
  --evaluate-traj PATH  Extended XYZ with attached energy/forces for ASE/Ovito
                        (default: <output-dir>/evaluate.extxyz).
  --no-evaluate-save-artifacts
                        Do not write evaluate.npz / evaluate.extxyz alongside
                        evaluate.json.
  --evaluate-reference-npz PATH
                        MP2/QM reference trajectory NPZ (keys R, E, optional F)
                        for on-the-fly comparison; writes evaluate_compare.json.
                        With --max-frames > 1, geometries are taken from this
                        file and a multi-frame evaluate.extxyz is written.
  --evaluate-reference-frame EVALUATE_REFERENCE_FRAME
                        Reference NPZ frame for comparison (default: same as
                        --evaluate-frame).
  --evaluate-reference-energy-unit {hartree,ev,kcal_mol}
                        Unit of E in --evaluate-reference-npz. Default: infer
                        from NPZ _mmml_units / units_manifest.json / force
                        magnitudes (else hartree).
  --evaluate-reference-force-unit {hartree_bohr,ev_ang}
                        Unit of F in --evaluate-reference-npz. Default: infer
                        from NPZ metadata or force magnitudes (else
                        hartree_bohr).
  --evaluate-compare-output EVALUATE_COMPARE_OUTPUT
                        JSON path for reference comparison (default: <output-
                        dir>/evaluate_compare.json).
  --dyna-probe          PyCHARMM only: one short NVE DYNA step with pre/post
                        snapshots of all force lanes (spherical_fn,
                        mlpot_callback, charmm_total). Requires --evaluate-npz
                        and --composition.
  --dyna-probe-nstep DYNA_PROBE_NSTEP
                        Number of NVE integration steps for --dyna-probe
                        (default: 1).
  --dyna-probe-dt-fs DYNA_PROBE_DT_FS
                        Timestep in fs for --dyna-probe (default: 0.5).
  --dyna-probe-output DYNA_PROBE_OUTPUT
                        JSON path for --dyna-probe results (default: <output-
                        dir>/dyna_probe.json).
  --optimize-cutoffs    Grid-search ML/MM handoff cutoffs against a reference
                        trajectory NPZ (requires --reference-npz and
                        --composition). ASE backend only.
  --reference-npz PATH  Trajectory NPZ with keys R (n_frames, N, 3) and optional
                        E, F, Z, N for --optimize-cutoffs (not the single-frame
                        handoff format used by --evaluate-npz).
  --optimize-output OPTIMIZE_OUTPUT
                        JSON path for --optimize-cutoffs results (default:
                        <output-dir>/optimize_cutoffs.json).
  --ml-switch-width-grid, --ml-cutoff-grid ML_SWITCH_WIDTH_GRID
                        Comma-separated ML handoff width grid (Å).
  --mm-switch-on-grid MM_SWITCH_ON_GRID
                        Comma-separated mm_switch_on grid (Å).
  --mm-switch-width-grid, --mm-cutoff-grid MM_SWITCH_WIDTH_GRID
                        Comma-separated MM outer taper width grid (Å).
  --energy-weight ENERGY_WEIGHT
                        Weight for energy MSE in cutoff objective.
  --force-weight FORCE_WEIGHT
                        Weight for force MSE in cutoff objective.
  --max-frames MAX_FRAMES
                        Max trajectory frames to evaluate (-1 = all). Default:
                        200 for --optimize-cutoffs; 1 for --evaluate-npz with
                        --evaluate-reference-npz unless this flag is set.
  --no-run-advice       Do not print or write next-run guidance (next_run.yaml /
                        next_run.sh) when a job finishes or fails.
  --no-stage-summary    Do not write stage_summary.json (campaigns).
  --mlpot-profile       Enable profiling of MLpot callbacks and JAX/XLA
                        compilation timers

PyXtal crystal placement (requires mmml[chem]):
  --pyxtal, --no-pyxtal
                        Build --composition with PyXtal (space-group crystal)
                        instead of Packmol/grid. Requires uv sync --extra chem.
  --pyxtal-spg PYXTAL_SPG
                        International space-group number for PyXtal from_random
                        (default: 14).
  --pyxtal-dim {0,1,2,3}
                        PyXtal crystal dimensionality (default: 3).
  --pyxtal-factor PYXTAL_FACTOR
                        PyXtal volume factor passed to from_random (default:
                        1.0).
  --pyxtal-stoichiometry Z [Z ...]
                        Formula units per unique species in the PyXtal unit cell
                        (default: 2 for each; one value repeats for all
                        species).
  --pyxtal-supercell NX,NY,NZ
                        Supercell expansion after PyXtal build (e.g. 2,2,2).
                        Default: 1,1,1.
  --pyxtal-attempts PYXTAL_ATTEMPTS
                        Maximum PyXtal from_random retries (default: 20).
  --pyxtal-trim, --no-pyxtal-trim
                        When the PyXtal supercell has more molecules than
                        --composition, keep the first N and warn (default: on).
  --optimize-pyxtal     Optional ASE pre-relax of the PyXtal structure before
                        CHARMM MM cluster minimize.
  --optimize-pyxtal-emt
                        Use ASE EMT for --optimize-pyxtal (smoke tests only).

PBC box sizing:
  --box-size ANG        Fixed cubic simulation cell side (Å) for PBC/CHARMM.
                        Packmol placement uses a smaller inner cube (see
                        --packmol-box-padding). With --box-auto count, scales
                        --composition to target ρ at this side.
  --packmol-box-size ANG
                        Explicit Packmol ``inside cube`` edge (Å). Must be
                        smaller than the simulation cell (--box-size or density-
                        sized L_sim).
  --packmol-box-padding ANG
                        Cold-start margin (Å per side) between Packmol cube and
                        simulation cell (default: 1.0; thin face birth only —
                        pack nearly full cell; raise MM pair capacity if NL fill
                        is high). Capped at 20% of L_sim.
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
                        PyCHARMM mini: MM pretreat hot→cold equilibration (ps
                        total) after lattice ABNR and before MLpot SD. Default
                        200 with --liquid-prep (100 ps heat + 100 ps cool).
                        0=off. Skipped when pretreat NPT equi runs.
  --mini-box-equil-ps-heat PS
                        Pretreat hot-leg length (ps). Default: half of --mini-
                        box-equil-ps. Ramp from --temperature to --mini-box-
                        equil-hot-temp.
  --mini-box-equil-ps-cool PS
                        Pretreat cold-leg length (ps). Default: half of --mini-
                        box-equil-ps. Ramp from hot peak back to --temperature.
  --mini-box-equil-hot-temp K
                        Peak temperature for pretreat hot leg (K). Default:
                        max(1.5×--temperature, --temperature+100).
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
                        CHARMM MM mini and before MLpot SD. 0=off; default with
                        --liquid-prep and no density target: 200. Skipped when a
                        density target sizes the box (see --mini-lattice-abnr-
                        allow-density-resize). Requires CRYSTAL/PBC.
  --mini-lattice-abnr-nocoords
                        With --mini-lattice-abnr-steps: optimize only the unit
                        cell (NOCOordinates); default optimizes coordinates and
                        box together.
  --mini-lattice-abnr-allow-fixed-box
                        Allow --mini-lattice-abnr-steps even when --box-size is
                        set (default: fixed --box-size skips lattice
                        minimization).
  --mini-lattice-abnr-allow-density-resize
                        Allow lattice ABNR to change the cell when the box was
                        sized from --target-density-g-cm3 / --bulk-density-
                        fraction / --box-auto density. Default: hold L fixed so
                        certified density stays meaningful.
  --density-certify-relative-tolerance FRAC
                        liquid-box: fail certification when |ρ_final −
                        ρ_target|/ρ_target exceeds this fraction (default:
                        0.05). <=0 disables the density gate.
  --liquid-prep, --no-liquid-prep
                        Easy dense-liquid setup: same as --density-prep-mode
                        resilient (looser Packmol, MC density equalization,
                        stronger CHARMM mini, fixed-L density hold, post-mini
                        rescue ladder when GRMS is high). For full prep +
                        dynamics recovery in one flag, prefer --cleanup.
  --density-prep-mode {off,resilient}
                        Condensed-phase box prep strategy. resilient: start
                        Packmol slightly below target density, enable MC
                        equalization, stronger CHARMM mini (box held at the
                        density-sized L unless --mini-lattice-abnr-allow-
                        density-resize), and the post-mini density prep ladder
                        when GRMS is high.
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
                        Pre-MLpot geometry gate: minimum inter-monomer MIC
                        distance in Å (default: 2.3; independent of --dynamics-
                        overlap-min-distance). Structures must be ML-safe before
                        USER is enabled.
  --charmm-image-mlpot-min-distance ANG
                        CHARMM <MKIMAT2> / MIC registration floor (Å) before
                        MLpot USER (default: 1.0; atom-pair prep gates remain
                        stricter).
  --pre-mlpot-h-heavy-min-distance ANG
                        Pre-MLpot H–heavy element-pair MIC floor (Å); default
                        2.3.
  --pre-mlpot-heavy-heavy-min-distance ANG
                        Pre-MLpot heavy–heavy element-pair MIC floor (Å);
                        default 2.5.
  --mlpot-registration-max-grms KCAL
                        Abort MLpot registration when hybrid GRMS exceeds this
                        limit (kcal/mol/Å; default 50). Use with --allow-high-
                        grms to warn only.

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

Dynamics overlap guard (PyCHARMM MLpot):
  --dynamics-overlap-action {error,warn,rescue,off}
                        On inter-monomer overlap during MD: rescue=CHARMM
                        bonded+VDW mini, then monomer repack (re-place COMs) if
                        still overlapped (default); error=abort, warn=log only,
                        off=disable. Also controls intra-monomer close-contact
                        checks and max monomer extent (fly-off) recovery.
  --dynamics-overlap-charmm-sd-steps DYNAMICS_OVERLAP_CHARMM_SD_STEPS
                        CHARMM SD steps for overlap rescue (default: 200).
  --dynamics-overlap-charmm-abnr-steps DYNAMICS_OVERLAP_CHARMM_ABNR_STEPS
                        CHARMM ABNR steps for overlap rescue (default: 400).
  --dynamics-overlap-min-distance ANG
                        Minimum allowed inter-monomer atom distance in Å during
                        dynamics (default: 0.45; same scale as pre-MLpot MIC
                        prep floors).
  --dynamics-intra-min-distance ANG
                        Minimum allowed nonbonded atom distance within each
                        monomer (1–2 and 1–3 pairs excluded from PSF bonds).
                        Default: auto from reference geometry (else 0.5 Å). Set
                        0 to disable.
  --no-dynamics-intra-exclude-1-3
                        Intra-monomer checks: only exclude PSF 1–2 bonds, not
                        1–3 pairs.
  --dynamics-intra-rescue-sd-steps DYNAMICS_INTRA_RESCUE_SD_STEPS
                        Bonded+VDW SD steps for intra-monomer close-contact
                        rescue (default: --dynamics-overlap-charmm-sd-steps).
  --dynamics-overlap-check-interval DYNAMICS_OVERLAP_CHECK_INTERVAL
                        Integration steps between overlap/extent checks during
                        dynamics (default: 500). Effective interval is the
                        largest divisor of the stage step count not exceeding
                        this value (and at least dcd-nsavc + 1 when set). Heat
                        uses this mid-segment interval by default; see --heat-
                        overlap-segment-boundary-only for legacy end-only
                        checks.
  --heat-overlap-segment-boundary-only
                        Heat only: run one overlap chunk per heat segment
                        (geometry check at segment end only). Default runs
                        checks every --dynamics-overlap-check-interval inside
                        each segment so extent/T blow-ups fail faster.
  --dynamics-overlap-memory-handoff
                        Continue overlap chunks in-process without READYN on
                        scratch restarts. Default on MPI-linked CHARMM under
                        mpirun (set MMML_NO_OVERLAP_MEMORY_HANDOFF=1 to force
                        scratch .overlap_a/.b.res handoffs).
  --no-dynamics-overlap-separate
                        Do not repack overlapped monomers (re-place COMs with
                        preserved internal geometry) when bonded+VDW rescue
                        minimization fails to restore min inter-monomer
                        distance.
  --dynamics-overlap-separate-margin ANG
                        Extra Å added beyond --dynamics-overlap-min-distance
                        when last-resort monomer repack spacing is derived
                        automatically (default: 0.2).
  --dynamics-max-monomer-extent ANG
                        Maximum allowed axis-aligned monomer extent in Å during
                        dynamics (default: auto from reference bond geometry,
                        else 12.0 Å).
  --no-dynamics-geometry-limits-auto
                        Use fixed --dynamics-max-monomer-extent / --dynamics-
                        intra-min-distance defaults instead of bond-length-
                        derived limits from reference geometry.
  --no-dynamics-max-monomer-extent
                        Disable max monomer extent / fly-off guard.
  --no-dynamics-monomer-health
                        Disable per-monomer velocity/GRMS/geometry bookkeeping
                        and early intervention during dynamics.
  --dynamics-monomer-health-debug
                        Print per-residue monomer health dot matrix
                        (green/yellow/red for velocity, GRMS, geometry) at each
                        dynamics health check.
  --no-dynamics-monomer-template-restore
                        Audit monomer health but do not template-restore
                        geometry-bad monomers (fly-off / collapse).
  --no-dynamics-monomer-jax-after-restore
                        Skip per-monomer JAX bonded mini after template restore.
  --no-dynamics-monomer-velocity-restore
                        Template-restore positions only; do not splice baseline
                        restart velocities (or Maxwell-Boltzmann redraw) onto
                        restored monomers.
  --dynamics-monomer-health-max-restore DYNAMICS_MONOMER_HEALTH_MAX_RESTORE
                        Max geometry-bad monomers to template-restore per health
                        check (default: 4).
  --dynamics-monomer-velocity-warn-recover-fraction DYNAMICS_MONOMER_VELOCITY_WARN_RECOVER_FRACTION
                        Fraction of monomers that must be velocity-warn before
                        velocity-only redraw recovery runs (default: 0.80).
  --dynamics-monomer-baseline-floor-fraction DYNAMICS_MONOMER_BASELINE_FLOOR_FRACTION
                        Floor health baselines at this fraction of the warn
                        absolute cut before ratio math (default: 0.25).
  --dynamics-monomer-ratio-without-abs
                        Allow velocity/GRMS ratios to escalate health without
                        meeting the absolute warn floor (legacy; off by
                        default).
  --dynamics-monomer-template-on-force
                        Allow template+FIRE on GRMS/force-bad monomers without
                        geometry failure (legacy; off by default — restore is
                        for fly-off/collapse).
  --dynamics-monomer-bond-stretch-factor DYNAMICS_MONOMER_BOND_STRETCH_FACTOR
                        Geometry-bad when a PSF 1–2 bond exceeds this ×
                        reference length (default: 1.75; also abs floor via
                        --dynamics-monomer-bond-stretch-abs).
  --dynamics-monomer-bond-stretch-abs DYNAMICS_MONOMER_BOND_STRETCH_ABS
                        Absolute bond-length floor (Å) for geometry-bad stretch
                        (default: 2.50).
  --dynamics-monomer-com-flyoff DYNAMICS_MONOMER_COM_FLYOFF
                        Unwrapped COM drift (Å) from health baseline that marks
                        geometry-bad (default: 0 → max(15, 0.35×box)).
  --no-dynamics-monomer-com-flyoff
                        Disable unwrapped COM-drift fly-off checks in monomer
                        health.
```


## Related docs

- [md-system YAML configs](../../md-system-configs.md)
- [Cross-backend handoff](../../handoff.md)
- [PyCHARMM MPI](../../pycharmm-mpi.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
