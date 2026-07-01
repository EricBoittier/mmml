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
                          [--objective OBJECTIVE] [--restart RESTART]
                          [--num-atoms NUM_ATOMS] [--features FEATURES]
                          [--max-degree MAX_DEGREE]
                          [--num-basis-functions NUM_BASIS_FUNCTIONS]
                          [--num-iterations NUM_ITERATIONS] [--n-res N_RES]
                          [--cutoff CUTOFF]
                          [--max-atomic-number MAX_ATOMIC_NUMBER] [--zbl]
                          [--no-zbl] [--use-pbc] [--no-pbc] [--no-energy-bias]
                          [--optimizer OPTIMIZER] [--transform TRANSFORM]
                          [--schedule-fn SCHEDULE_FN]
                          [--early-stop-patience EARLY_STOP_PATIENCE] [--best]
                          [--no-save-every-epoch] [--profile-epoch-timing]
                          [--print-freq PRINT_FREQ]
                          [--batch-method BATCH_METHOD]
                          [--batch-args-dict BATCH_ARGS_DICT]
                          [--data-keys DATA_KEYS [DATA_KEYS ...]]
                          [--conversion CONVERSION]
                          [--init-params INIT_PARAMS]
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
                          [--rot-augment]
                          [--rot-perturbation ROT_PERTURBATION] [--charges]
                          [--no-charges] [--total-charge TOTAL_CHARGE]
                          [--no-electrostatics] [--efa] [--no-efa] [--debug]
                          [--no-debug] [--save-config SAVE_CONFIG] [--quiet]

Train a PhysNetJAX EF model from NPZ data.

options:
  -h, --help            show this help message and exit
  --config CONFIG       YAML file with training options (CLI flags override
                        file values)
  --data DATA           Training NPZ file
  --valid-data, --valid_data VALID_DATA
                        Optional validation NPZ (use full files; no random re-
                        split)
  --ckpt-dir, --ckpt_dir CKPT_DIR
                        Checkpoint directory (absolute path used for Orbax)
  --tag TAG             Run name for checkpoints
  --model MODEL         Optional model JSON to load instead of creating a new
                        EF model
  --n-train, --n_train N_TRAIN
  --n-valid, --n_valid N_VALID
  --seed SEED
  --batch-size, --batch_size BATCH_SIZE
  --num-epochs, --num_epochs NUM_EPOCHS
  --learning-rate, --learning_rate LEARNING_RATE
  --energy-weight, --energy_weight ENERGY_WEIGHT
  --forces-weight, --forces_weight FORCES_WEIGHT
  --dipole-weight, --dipole_weight DIPOLE_WEIGHT
  --charges-weight, --charges_weight CHARGES_WEIGHT
  --objective OBJECTIVE
  --restart RESTART     Checkpoint path to restart from
  --num-atoms, --num_atoms NUM_ATOMS
                        Atoms per structure (auto-detected from N/R if
                        omitted)
  --features FEATURES
  --max-degree, --max_degree MAX_DEGREE
  --num-basis-functions, --num_basis_functions NUM_BASIS_FUNCTIONS
  --num-iterations, --num_iterations NUM_ITERATIONS
  --n-res, --n_res N_RES
                        Number of refinement residual blocks (not CHARMM
                        residues)
  --cutoff CUTOFF
  --max-atomic-number, --max_atomic_number MAX_ATOMIC_NUMBER
  --zbl                 Enable ZBL repulsion in EF model
  --no-zbl              Disable ZBL repulsion in EF model
  --use-pbc, --use_pbc  Use periodic boundary conditions
  --no-pbc              Disable periodic boundary conditions
  --no-energy-bias      Disable per-element energy bias in the model
  --optimizer OPTIMIZER
                        Optimizer string (e.g. 'adam', 'adamw', 'amsgrad')
  --transform TRANSFORM
                        Transform string (e.g. 'reduce_on_plateau')
  --schedule-fn, --schedule_fn SCHEDULE_FN
                        Learning rate schedule string (e.g. 'warmup',
                        'cosine')
  --early-stop-patience, --early_stop_patience EARLY_STOP_PATIENCE
                        Number of epochs to wait for improvement before
                        stopping training
  --best                Only save checkpoint when objective improves
  --no-save-every-epoch
                        Disable saving a checkpoint at every epoch
  --profile-epoch-timing
                        Print per-epoch timing breakdown (batch prep / train /
                        valid / checkpoint)
  --print-freq, --print_freq PRINT_FREQ
                        Printing frequency in epochs
  --batch-method, --batch_method BATCH_METHOD
                        Batching method ('default' or 'advanced')
  --batch-args-dict, --batch_args_dict BATCH_ARGS_DICT
                        JSON string or file path for advanced batch arguments
  --data-keys, --data_keys DATA_KEYS [DATA_KEYS ...]
                        Keys to load from NPZ file
  --conversion CONVERSION
                        Display-only MAE scaling for energy/forces (JSON
                        string or .json/.yaml path). Multiplies reported
                        train/valid energy and force MAE after each epoch;
                        does NOT transform NPZ arrays or affect the loss.
                        Default when omitted: {"energy": 1, "forces": 1} (MAE
                        in same units as the NPZ). Example for kcal/mol
                        display when data are eV: '{"energy": 23.060549,
                        "forces": 23.060549}'. Dipole units are not handled
                        here — convert D/Dxyz before training (e.g. mmml fix-
                        and-split --dipole-in debye --dipole-out e-angstrom).
                        See docs/UNITS_SUMMARY.md § physnet-train
                        --conversion.
  --init-params, --init_params INIT_PARAMS
                        JSON string or file path to initialize flax parameters
  --physnet-checkpoint, --physnet_checkpoint PHYSNET_CHECKPOINT
                        PhysNet checkpoint path (JSON or Orbax) for warm-start
                        transfer learning
  --physnet-transfer-model, --physnet_transfer_model PHYSNET_TRANSFER_MODEL
                        Bundled PhysNet transfer model ID, file stem, or
                        category. Defaults to 'joint-training-defaults' when
                        distillation is enabled.
  --list-physnet-transfer-models
                        List bundled PhysNet transfer-learning models and exit
  --physnet-transfer-category, --physnet_transfer_category PHYSNET_TRANSFER_CATEGORY
                        Filter --list-physnet-transfer-models by manifest
                        category
  --match-checkpoint-architecture
                        Override EF hyperparameters from transfer checkpoint
                        config (default: on)
  --no-match-checkpoint-architecture
                        Do not override EF hyperparameters from transfer
                        checkpoint config
  --distill             Enable teacher distillation loss during training
  --distill-alpha, --distill_alpha DISTILL_ALPHA
                        Ground-truth loss weight (1.0=GT only, 0.0=teacher
                        only)
  --distill-targets, --distill_targets DISTILL_TARGETS [DISTILL_TARGETS ...]
                        Distillation targets: energy forces dipole (default:
                        all three)
  --teacher-checkpoint, --teacher_checkpoint TEACHER_CHECKPOINT
                        Teacher checkpoint for distillation (defaults to warm-
                        start checkpoint)
  --metrics-plot, --metrics_plot METRICS_PLOT
                        After training, write learning-curve plot to this path
                        via Orbax checkpoints
  --log-loss            Use log scale on loss axes when generating --metrics-
                        plot
  --rot-augment, --rot_augment
                        Apply random rotation augmentation to inputs
  --rot-perturbation, --rot_perturbation ROT_PERTURBATION
                        Magnitude of rotation perturbation
  --charges             Predict atomic charges (useful for dipoles and
                        electrostatics)
  --no-charges          Do not predict atomic charges
  --total-charge, --total_charge TOTAL_CHARGE
                        Total charge constraint of the molecular system
  --no-electrostatics   Disable electrostatics layer in EF model
  --efa                 Enable Euclidean Fast Attention (EFA) in the model
  --no-efa              Disable Euclidean Fast Attention (EFA)
  --debug               Enable debug flags in EF model
  --no-debug            Disable debug flags in EF model
  --save-config, --save_config SAVE_CONFIG
                        Write resolved training options to YAML and exit
  --quiet, -q           Suppress JAX device summary

Examples:
  mmml physnet-train \
      --data output/energies_forces_dipoles_train.npz \
      --ckpt-dir ./ckpts/ama_mp2 \
      --tag ama_mp2 \
      --n-train 24000 --n-valid 3000 \
      --batch-size 32 --num-epochs 2000 \
      --max-atomic-number 35

  mmml physnet-train --config train.yaml

YAML keys match CLI flags (with optional aliases: train, output, max_epochs).
See mmml/cli/misc/physnet_train.example.yaml for a template.
See mmml/cli/misc/physnet_train_transfer.example.yaml for transfer learning / distillation.
```


---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
