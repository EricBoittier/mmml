# `mmml efield-train`

Train external electric-field PhysNet.


## Usage

```bash
mmml efield-train --help
```

## Options

```text
usage: mmml efield-train [-h] [--data DATA] [--train-npz TRAIN_NPZ]
                         [--valid-npz VALID_NPZ] [--test-npz TEST_NPZ]
                         [--output-dir OUTPUT_DIR] [--features FEATURES]
                         [--max_degree MAX_DEGREE]
                         [--num_iterations NUM_ITERATIONS]
                         [--num_basis_functions NUM_BASIS_FUNCTIONS]
                         [--cutoff CUTOFF] [--num_train NUM_TRAIN]
                         [--num_valid NUM_VALID] [--num_epochs NUM_EPOCHS]
                         [--learning_rate LEARNING_RATE]
                         [--batch_size BATCH_SIZE] [--clip_norm CLIP_NORM]
                         [--ema_decay EMA_DECAY]
                         [--early_stopping_patience EARLY_STOPPING_PATIENCE]
                         [--early_stopping_min_delta EARLY_STOPPING_MIN_DELTA]
                         [--reduce_on_plateau_patience REDUCE_ON_PLATEAU_PATIENCE]
                         [--reduce_on_plateau_cooldown REDUCE_ON_PLATEAU_COOLDOWN]
                         [--reduce_on_plateau_factor REDUCE_ON_PLATEAU_FACTOR]
                         [--reduce_on_plateau_rtol REDUCE_ON_PLATEAU_RTOL]
                         [--reduce_on_plateau_accumulation_size REDUCE_ON_PLATEAU_ACCUMULATION_SIZE]
                         [--reduce_on_plateau_min_scale REDUCE_ON_PLATEAU_MIN_SCALE]
                         [--restart RESTART] [--energy_weight ENERGY_WEIGHT]
                         [--forces_weight FORCES_WEIGHT]
                         [--dipole_weight DIPOLE_WEIGHT]
                         [--charge_weight CHARGE_WEIGHT]
                         [--dipole_field_coupling] [--field_scale FIELD_SCALE]
                         [--electrostatics_damping_sigma ELECTROSTATICS_DAMPING_SIGMA]
                         [--zbl]
                         [--include-pseudotensors | --no-include-pseudotensors]
                         [--gradient-checkpoint] [--rot-augment]
                         [--rot-perturbation ROT_PERTURBATION] [--verbose]
                         [--save-every N]

Input & configuration:
  --data DATA           Single merged NPZ; random train/valid split via --num-
                        train / --num-valid
  --gradient-checkpoint
                        Use gradient checkpointing to reduce GPU memory (slower
                        training)

Scientific model:
  --num_basis_functions NUM_BASIS_FUNCTIONS
  --cutoff CUTOFF
  --energy_weight ENERGY_WEIGHT
                        Weight for energy loss in total loss
  --forces_weight FORCES_WEIGHT
                        Weight for forces loss in total loss
  --charge_weight CHARGE_WEIGHT
                        Weight for charge neutrality loss (sum of charges per
                        molecule squared)
  --dipole_field_coupling
                        Add explicit E_total = E_nn + mu·Ef coupling
  --field_scale FIELD_SCALE
                        Ef_phys = Ef_input * field_scale (au)

Execution:
  --num_epochs NUM_EPOCHS
  --batch_size BATCH_SIZE
                        Batch size (default 256; use 128 or 64 if OOM)
  --electrostatics_damping_sigma ELECTROSTATICS_DAMPING_SIGMA
                        Apply erf(r/sigma) damping to learned-charge Coulomb;
                        set 0 to disable
  --include-pseudotensors, --no-include-pseudotensors
                        Equivariant parity dimension in e3x MessagePass /
                        tensors (default: on)

Output & artifacts:
  --output-dir OUTPUT_DIR
                        Directory for params-*.json, config-*.json, and symlinks
  --save-every N        Save EMA checkpoint every N epochs to params-epoch-
                        NNNN-<uuid>.json (0 = no periodic saves)

Diagnostics & safety:
  -h, --help            show this help message and exit
  --verbose             Print extra debug output (e.g. [STRUCT] parameter tree
                        dumps)

Other options:
  --train-npz TRAIN_NPZ
                        Training split NPZ (R,Z,N,E,F,Ef[,Dxyz|D]) — use with
                        --valid-npz instead of --data
  --valid-npz VALID_NPZ
                        Validation split NPZ (same keys as train)
  --test-npz TEST_NPZ   Optional test NPZ: only print shapes (not used for
                        training)
  --features FEATURES
  --max_degree MAX_DEGREE
  --num_iterations NUM_ITERATIONS
  --num_train NUM_TRAIN
  --num_valid NUM_VALID
  --learning_rate LEARNING_RATE
  --clip_norm CLIP_NORM
  --ema_decay EMA_DECAY
  --early_stopping_patience EARLY_STOPPING_PATIENCE
  --early_stopping_min_delta EARLY_STOPPING_MIN_DELTA
  --reduce_on_plateau_patience REDUCE_ON_PLATEAU_PATIENCE
  --reduce_on_plateau_cooldown REDUCE_ON_PLATEAU_COOLDOWN
  --reduce_on_plateau_factor REDUCE_ON_PLATEAU_FACTOR
  --reduce_on_plateau_rtol REDUCE_ON_PLATEAU_RTOL
  --reduce_on_plateau_accumulation_size REDUCE_ON_PLATEAU_ACCUMULATION_SIZE
  --reduce_on_plateau_min_scale REDUCE_ON_PLATEAU_MIN_SCALE
  --restart RESTART
  --dipole_weight DIPOLE_WEIGHT
                        Weight for dipole loss in total loss
  --zbl                 Add ZBL nuclear repulsion for short-range stability
  --rot-augment         Apply random SO(3) rotation augmentation to batches (all
                        splits)
  --rot-perturbation ROT_PERTURBATION
                        Rotation perturbation strength in [0, 1] (used with
                        --rot-augment)
```

## Visual examples

![Electrostatic field represented by learned multipoles](../../plot-style-gallery-assets/chart_multipole_field_learned.png)


---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
