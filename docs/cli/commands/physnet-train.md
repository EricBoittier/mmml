# `mmml physnet-train`

Train PhysNet message-passing model (E/F).


## Usage

```bash
mmml physnet-train --help
```

## Options

```text
usage: mmml physnet-train [-h] [--config CONFIG] [--data DATA]
                          [--valid-data VALID_DATA] [--ckpt-dir CKPT_DIR]
                          [--tag TAG] [--model MODEL] [--n-train N_TRAIN]
                          [--n-valid N_VALID] [--seed SEED]
                          [--batch-size BATCH_SIZE] [--num-epochs NUM_EPOCHS]
                          [--learning-rate LEARNING_RATE]
                          [--energy-weight ENERGY_WEIGHT]
                          [--forces-weight FORCES_WEIGHT]
                          [--dipole-weight DIPOLE_WEIGHT]
                          [--charges-weight CHARGES_WEIGHT]
                          [--objective OBJECTIVE]
                          [--mm-charge-mode {fixed,q0,latent,q1,fixed_plus_latent}]
                          [--mm-charge-correction] [--hybrid-mm]
                          [--hybrid-hamiltonian {handoff,shared_cutoff}]
                          [--shared-cutoff SHARED_CUTOFF]
                          [--ml-switch-width ML_SWITCH_WIDTH]
                          [--mm-switch-on MM_SWITCH_ON]
                          [--mm-switch-width MM_SWITCH_WIDTH]
                          [--no-complementary-handoff]
                          [--mm-pair-source {jax,charmm_callback}]
                          [--lr-solver {mic,nvalchemiops_pme,ewald}]
                          [--pme-box-length PME_BOX_LENGTH]
                          [--pme-accuracy PME_ACCURACY]
                          [--mm-include-lj | --no-mm-include-lj | --mm_include_lj | --no-mm_include_lj]
                          [--learn-mm-lj-scales | --no-learn-mm-lj-scales | --learn_mm_lj_scales | --no-learn_mm_lj_scales]
                          [--mm-lj-sigma-scale-min MM_LJ_SIGMA_SCALE_MIN]
                          [--mm-lj-sigma-scale-max MM_LJ_SIGMA_SCALE_MAX]
                          [--mm-lj-epsilon-scale-min MM_LJ_EPSILON_SCALE_MIN]
                          [--mm-lj-epsilon-scale-max MM_LJ_EPSILON_SCALE_MAX]
                          [--mm-lj-min-type-frames MM_LJ_MIN_TYPE_FRAMES]
                          [--ema-decay EMA_DECAY] [--restart RESTART]
                          [--num-atoms NUM_ATOMS] [--features FEATURES]
                          [--max-degree MAX_DEGREE]
                          [--num-basis-functions NUM_BASIS_FUNCTIONS]
                          [--num-iterations NUM_ITERATIONS] [--n-res N_RES]
                          [--cutoff CUTOFF] [--switch-start SWITCH_START]
                          [--switch-end SWITCH_END]
                          [--electrostatics-off-start ELECTROSTATICS_OFF_START]
                          [--electrostatics-off-end ELECTROSTATICS_OFF_END]
                          [--max-atomic-number MAX_ATOMIC_NUMBER] [--zbl]
                          [--no-zbl] [--trainable-zbl] [--use-pbc] [--no-pbc]
                          [--no-energy-bias] [--optimizer OPTIMIZER]
                          [--transform TRANSFORM] [--schedule-fn SCHEDULE_FN]
                          [--early-stop-patience EARLY_STOP_PATIENCE] [--best]
                          [--no-save-every-epoch] [--profile-epoch-timing]
                          [--print-freq PRINT_FREQ]
                          [--batch-method BATCH_METHOD]
                          [--batch-args-dict BATCH_ARGS_DICT]
                          [--data-keys DATA_KEYS [DATA_KEYS ...]]
                          [--conversion CONVERSION] [--init-params INIT_PARAMS]
                          [--physnet-checkpoint PHYSNET_CHECKPOINT]
                          [--physnet-transfer-model PHYSNET_TRANSFER_MODEL]
                          [--list-physnet-transfer-models]
                          [--physnet-transfer-category PHYSNET_TRANSFER_CATEGORY]
                          [--match-checkpoint-architecture]
                          [--no-match-checkpoint-architecture] [--distill]
                          [--distill-alpha DISTILL_ALPHA]
                          [--distill-targets DISTILL_TARGETS [DISTILL_TARGETS ...]]
                          [--teacher-checkpoint TEACHER_CHECKPOINT]
                          [--metrics-plot METRICS_PLOT] [--log-loss]
                          [--rot-augment] [--rot-perturbation ROT_PERTURBATION]
                          [--charges] [--no-charges]
                          [--total-charge TOTAL_CHARGE] [--no-electrostatics]
                          [--efa] [--no-efa] [--debug] [--no-debug]
                          [--save-config SAVE_CONFIG] [--quiet]

Train a PhysNetJAX EF model from NPZ data.

Input & configuration:
  --config CONFIG       YAML file with training options (CLI flags override file
                        values)
  --data DATA           Training NPZ file
  --valid-data, --valid_data VALID_DATA
                        Optional validation NPZ (use full files; no random re-
                        split)
  --data-keys, --data_keys DATA_KEYS [DATA_KEYS ...]
                        Keys to load from NPZ file
  --physnet-checkpoint, --physnet_checkpoint PHYSNET_CHECKPOINT
                        PhysNet checkpoint path (JSON or Orbax) for warm-start
                        transfer learning
  --match-checkpoint-architecture
                        Override EF hyperparameters from transfer checkpoint
                        config (default: on)
  --no-match-checkpoint-architecture
                        Do not override EF hyperparameters from transfer
                        checkpoint config
  --teacher-checkpoint, --teacher_checkpoint TEACHER_CHECKPOINT
                        Teacher checkpoint for distillation (defaults to warm-
                        start checkpoint)
  --save-config, --save_config SAVE_CONFIG
                        Write resolved training options to YAML and exit

Scientific model:
  --model MODEL         Optional model JSON to load instead of creating a new EF
                        model
  --energy-weight, --energy_weight ENERGY_WEIGHT
  --forces-weight, --forces_weight FORCES_WEIGHT
  --charges-weight, --charges_weight CHARGES_WEIGHT
  --mm-charge-mode, --mm_charge_mode {fixed,q0,latent,q1,fixed_plus_latent}
                        Hybrid MM Coulomb charges: fixed (q_CGenFF, default), q0
                        / Q⁰ (neutralize unperturbed monomer q_ML;
                        train+liquid), latent / q1 / Q¹ (neutralize AB-perturbed
                        q_ML; dimer-only), or fixed_plus_latent (q_CGenFF +
                        neutralize(Q¹)). Modes q0/latent/q1/fixed_plus_latent
                        require --charges. latent/q1/fixed_plus_latent are
                        dimer-only. See docs/hybrid-mm-charges.md.
  --mm-charge-correction, --mm_charge_correction
                        Alias for --mm-charge-mode fixed_plus_latent: use the
                        model's predicted charges as a CORRECTION to fixed
                        CGenFF charges in MM electrostatics (q_eff = q_cgenff +
                        dq_ML, projected net-zero per monomer). Requires
                        --charges.
  --shared-cutoff SHARED_CUTOFF
                        Atomic ML/MM cutoff (Å) for --hybrid-hamiltonian
                        shared_cutoff; defaults to the model cutoff.
  --mm-switch-on MM_SWITCH_ON
                        COM distance (Å) where the complementary handoff ends:
                        ML scale reaches 0 and MM scale reaches 1 (default: 6).
  --no-complementary-handoff
                        Legacy MM window: MM starts at mm_switch_on instead of
                        filling the ML taper handoff.
  --pme-box-length, --pme_box_length PME_BOX_LENGTH
                        Cubic box length (Å) for --lr-solver
                        nvalchemiops_pme|ewald (required for those solvers).
  --pme-accuracy, --pme_accuracy PME_ACCURACY
                        nvalchemiops_pme/ewald PME accuracy target (default:
                        1e-6).
  --num-basis-functions, --num_basis_functions NUM_BASIS_FUNCTIONS
  --cutoff CUTOFF       PhysNet radial basis cutoff (Angstrom, atom-pair
                        distance). Must be >= --mm-switch-on: the ML has to be
                        able to see the interaction out to wherever MM takes
                        over, or it is silently truncated inside the handoff.
                        Baked into the checkpoint; MD reads it back
                        automatically.
  --switch-start, --switch_start SWITCH_START
                        Short-range Coulomb switch start (Angstrom)
  --switch-end, --switch_end SWITCH_END
                        Short-range Coulomb switch end (Angstrom)
  --electrostatics-off-start, --electrostatics_off_start ELECTROSTATICS_OFF_START
                        Distance at which the electrostatic term begins
                        switching off (Angstrom). Set this beyond the largest
                        separation the reaction reaches, or the Coulomb tail
                        vanishes there.
  --electrostatics-off-end, --electrostatics_off_end ELECTROSTATICS_OFF_END
                        Distance at which the electrostatic term is fully off
                        (Angstrom)
  --no-energy-bias      Disable per-element energy bias in the model
  --physnet-transfer-model, --physnet_transfer_model PHYSNET_TRANSFER_MODEL
                        Bundled PhysNet transfer model ID, file stem, or
                        category. Defaults to 'joint-training-defaults' when
                        distillation is enabled.
  --charges             Predict atomic charges (useful for dipoles and
                        electrostatics)
  --no-charges          Do not predict atomic charges
  --total-charge, --total_charge TOTAL_CHARGE
                        Total charge constraint of the molecular system

Execution:
  --seed SEED
  --batch-size, --batch_size BATCH_SIZE
  --num-epochs, --num_epochs NUM_EPOCHS
  --ml-switch-width, --ml-cutoff ML_SWITCH_WIDTH
                        COM-distance width (Å) of the ML→MM handoff. ML is fully
                        on below mm_switch_on - width and tapers to zero at
                        mm_switch_on (default: 1.5).
  --mm-switch-width, --mm-cutoff MM_SWITCH_WIDTH
                        COM-distance width (Å) of the MM outer tail after
                        mm_switch_on. Switched MM reaches zero at mm_switch_on +
                        width (default: 5).
  --mm-lj-epsilon-scale-min MM_LJ_EPSILON_SCALE_MIN
  --mm-lj-epsilon-scale-max MM_LJ_EPSILON_SCALE_MAX
  --batch-method, --batch_method BATCH_METHOD
                        Batching method ('default' or 'advanced')
  --batch-args-dict, --batch_args_dict BATCH_ARGS_DICT
                        JSON string or file path for advanced batch arguments

Output & artifacts:
  --no-save-every-epoch
                        Disable saving a checkpoint at every epoch
  --metrics-plot, --metrics_plot METRICS_PLOT
                        After training, write learning-curve plot to this path
                        via Orbax checkpoints
  --log-loss            Use log scale on loss axes when generating --metrics-
                        plot

Diagnostics & safety:
  -h, --help            show this help message and exit
  --profile-epoch-timing
                        Print per-epoch timing breakdown (batch prep / train /
                        valid / checkpoint)
  --list-physnet-transfer-models
                        List bundled PhysNet transfer-learning models and exit
  --debug               Enable debug flags in EF model
  --no-debug            Disable debug flags in EF model
  --quiet, -q           Suppress JAX device summary

Other options:
  --ckpt-dir, --ckpt_dir CKPT_DIR
                        Checkpoint directory (absolute path used for Orbax)
  --tag TAG             Run name for checkpoints
  --n-train, --n_train N_TRAIN
                        Training samples to split from --data (default: 1000).
                        Omit when --valid-data is set: the full files are used.
  --n-valid, --n_valid N_VALID
                        Validation samples to split from --data (default: 100).
                        Omit when --valid-data is set: the full files are used.
  --learning-rate, --learning_rate LEARNING_RATE
  --dipole-weight, --dipole_weight DIPOLE_WEIGHT
  --objective OBJECTIVE
  --hybrid-mm, --hybrid_mm
                        Train on the hybrid ML/MM total the MD calculator
                        evaluates: E = (1-s)*(E_A+E_B) + s*E_AB + E_MM, where
                        the taper s(r_com) applies to the dimer interaction and
                        E_MM is switched CGenFF LJ + electrostatics. Requires a
                        dataset carrying cgenff_type_idx, mol_id, cgenff_charge
                        and the cgenff_master_* LJ tables. The handoff is
                        controlled by --ml-switch-width/--mm-switch-on/--mm-
                        switch-width (same flags and defaults as the MD side).
  --hybrid-hamiltonian {handoff,shared_cutoff}
                        Hybrid assembly: handoff preserves the existing COM-
                        switched Hamiltonian; shared_cutoff uses additive ML+MM
                        with no handoff and force-shifts MM pairs at --shared-
                        cutoff.
  --mm-pair-source {jax,charmm_callback}
                        Decomposed MLpot MM pair provider: Fortran callback
                        idxu/idxv (default) or JAX neighbor rebuild (--mm-pair-
                        source jax). All-ML bulk systems with empty callback
                        lists auto-fall back to JAX. Override with env
                        MMML_MM_PAIR_SOURCE.
  --lr-solver, --lr_solver {mic,nvalchemiops_pme,ewald}
                        Hybrid-MM long-range Coulomb for training (default:
                        mic). mic: switched CGenFF LJ+Coulomb pairs.
                        nvalchemiops_pme: full-box many-to-many PME on fixed
                        CGenFF charges (no exclusions / no intra subtract;
                        requires --pme-box-length and mmml[nvalchemiops-pme]).
                        ewald: same full-box/no-exclusion Coulomb as
                        nvalchemiops_pme, pure JAX (no external PME library, no
                        CUDA requirement); requires --pme-box-length. With --mm-
                        include-lj, COM-switched LJ is added beside the lattice
                        Coulomb (Coulomb itself stays untapered).
  --mm-include-lj, --no-mm-include-lj, --mm_include_lj, --no-mm_include_lj
                        Include CGenFF LJ in hybrid E_MM (default: on). Under
                        --lr-solver ewald|nvalchemiops_pme this is COM-switched
                        intermolecular LJ beside untapered full-box Coulomb;
                        under mic it is the usual switched LJ+Coulomb pair term.
  --learn-mm-lj-scales, --no-learn-mm-lj-scales, --learn_mm_lj_scales, --no-learn_mm_lj_scales
                        Learn per-CGenFF-type multiplicative scales on master σ
                        and ε (separate arrays, init 1.0). Works under --lr-
                        solver mic and under ewald|nvalchemiops_pme when --mm-
                        include-lj is on. Scales are saved in hybrid_mm.json for
                        MD ep_scale/sig_scale.
  --mm-lj-sigma-scale-min MM_LJ_SIGMA_SCALE_MIN
  --mm-lj-sigma-scale-max MM_LJ_SIGMA_SCALE_MAX
  --mm-lj-min-type-frames MM_LJ_MIN_TYPE_FRAMES
                        Freeze LJ scales at 1.0 for CGenFF types seen in fewer
                        training frames.
  --ema-decay, --ema_decay EMA_DECAY
                        Decay for the parameter EMA (default: 0.999).
                        Validation, checkpointing and restart all use the EMA
                        weights. Set 0 to disable EMA (saved weights then track
                        the raw parameters).
  --restart RESTART     Checkpoint path to restart from
  --num-atoms, --num_atoms NUM_ATOMS
                        Atoms per structure (auto-detected from N/R if omitted)
  --features FEATURES
  --max-degree, --max_degree MAX_DEGREE
  --num-iterations, --num_iterations NUM_ITERATIONS
  --n-res, --n_res N_RES
                        Number of refinement residual blocks (not CHARMM
                        residues)
  --max-atomic-number, --max_atomic_number MAX_ATOMIC_NUMBER
  --zbl                 Enable ZBL repulsion in EF model
  --no-zbl              Disable ZBL repulsion in EF model
  --trainable-zbl       Opt in to optimizing ZBL screening parameters; fixed ZBL
                        is the default.
  --use-pbc, --use_pbc  Use periodic boundary conditions
  --no-pbc              Disable periodic boundary conditions
  --optimizer OPTIMIZER
                        Optimizer string (e.g. 'adam', 'adamw', 'amsgrad')
  --transform TRANSFORM
                        Transform string (e.g. 'reduce_on_plateau')
  --schedule-fn, --schedule_fn SCHEDULE_FN
                        Learning rate schedule string (e.g. 'warmup', 'cosine')
  --early-stop-patience, --early_stop_patience EARLY_STOP_PATIENCE
                        Number of epochs to wait for improvement before stopping
                        training
  --best                Only save checkpoint when objective improves
  --print-freq, --print_freq PRINT_FREQ
                        Printing frequency in epochs
  --conversion CONVERSION
                        Display-only MAE scaling for energy/forces (JSON string
                        or .json/.yaml path). Multiplies reported train/valid
                        energy and force MAE after each epoch; does NOT
                        transform NPZ arrays or affect the loss. Default when
                        omitted: {"energy": 1, "forces": 1} (MAE in same units
                        as the NPZ). Example for kcal/mol display when data are
                        eV: '{"energy": 23.060549, "forces": 23.060549}'. Dipole
                        units are not handled here — convert D/Dxyz before
                        training (e.g. mmml fix-and-split --dipole-in debye
                        --dipole-out e-angstrom). See docs/UNITS_SUMMARY.md §
                        physnet-train --conversion.
  --init-params, --init_params INIT_PARAMS
                        JSON string or file path to initialize flax parameters
  --physnet-transfer-category, --physnet_transfer_category PHYSNET_TRANSFER_CATEGORY
                        Filter --list-physnet-transfer-models by manifest
                        category
  --distill             Enable teacher distillation loss during training
  --distill-alpha, --distill_alpha DISTILL_ALPHA
                        Ground-truth loss weight (1.0=GT only, 0.0=teacher only)
  --distill-targets, --distill_targets DISTILL_TARGETS [DISTILL_TARGETS ...]
                        Distillation targets: energy forces dipole (default: all
                        three)
  --rot-augment, --rot_augment
                        Apply random rotation augmentation to inputs
  --rot-perturbation, --rot_perturbation ROT_PERTURBATION
                        Magnitude of rotation perturbation
  --no-electrostatics   Disable electrostatics layer in EF model
  --efa                 Enable Euclidean Fast Attention (EFA) in the model
  --no-efa              Disable Euclidean Fast Attention (EFA)

Examples: mmml physnet-train \ --data output/energies_forces_dipoles_train.npz \
--ckpt-dir ./ckpts/ama_mp2 \ --tag ama_mp2 \ --n-train 24000 --n-valid 3000 \
--batch-size 32 --num-epochs 2000 \ --max-atomic-number 35 mmml physnet-train
--config train.yaml YAML keys match CLI flags (with optional aliases: train,
output, max_epochs). See mmml/cli/misc/physnet_train.example.yaml for a
template. See mmml/cli/misc/physnet_train_transfer.example.yaml for transfer
learning / distillation. See examples/hybrid_mm_charges/ for hybrid-mm +
mm_charge_mode (fixed/latent/fixed_plus_latent).
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
