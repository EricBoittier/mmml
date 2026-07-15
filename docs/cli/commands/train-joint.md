# `mmml train-joint`

Joint PhysNet+DCMNet training.


## Usage

```bash
mmml train-joint --help
```

## Options

```text
usage: mmml train-joint [-h] --train-efd TRAIN_EFD --train-esp TRAIN_ESP
                        --valid-efd VALID_EFD --valid-esp VALID_ESP
                        [--subtract-atom-energies]
                        [--physnet-features PHYSNET_FEATURES]
                        [--physnet-iterations PHYSNET_ITERATIONS]
                        [--physnet-basis PHYSNET_BASIS]
                        [--physnet-cutoff PHYSNET_CUTOFF]
                        [--physnet-n-res PHYSNET_N_RES] [--zbl] [--no-zbl]
                        [--physnet-max-degree PHYSNET_MAX_DEGREE]
                        [--dcmnet-features DCMNET_FEATURES]
                        [--dcmnet-iterations DCMNET_ITERATIONS]
                        [--dcmnet-basis DCMNET_BASIS]
                        [--dcmnet-cutoff DCMNET_CUTOFF] [--n-dcm N_DCM]
                        [--max-degree MAX_DEGREE] [--use-noneq-model]
                        [--noneq-features NONEQ_FEATURES]
                        [--noneq-layers NONEQ_LAYERS]
                        [--noneq-max-displacement NONEQ_MAX_DISPLACEMENT]
                        [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                        [--optimizer {adam,adamw,rmsprop,muon}]
                        [--learning-rate LEARNING_RATE]
                        [--weight-decay WEIGHT_DECAY]
                        [--use-recommended-hparams] [--seed SEED]
                        [--energy-weight ENERGY_WEIGHT]
                        [--forces-weight FORCES_WEIGHT]
                        [--dipole-weight DIPOLE_WEIGHT]
                        [--esp-weight ESP_WEIGHT]
                        [--esp-min-distance ESP_MIN_DISTANCE]
                        [--esp-max-value ESP_MAX_VALUE]
                        [--mono-weight MONO_WEIGHT]
                        [--charge-reg-weight CHARGE_REG_WEIGHT]
                        [--dipole-source {physnet,dcmnet,mixed}]
                        [--dipole-loss-sources [{physnet,dcmnet,mixed} ...]]
                        [--esp-loss-sources [{physnet,dcmnet,mixed} ...]]
                        [--dipole-metric {l2,mae,rmse}]
                        [--esp-metric {l2,mae,rmse}] [--loss-config LOSS_CONFIG]
                        [--mix-coulomb-energy] [--disable-physnet-point-coulomb]
                        [--mix-warmup-start MIX_WARMUP_START]
                        [--mix-warmup-end MIX_WARMUP_END]
                        [--mix-weight-max MIX_WEIGHT_MAX]
                        [--mix-schedule {linear,cosine}] [--natoms NATOMS]
                        [--max-atomic-number MAX_ATOMIC_NUMBER]
                        [--grad-clip-norm GRAD_CLIP_NORM] [--name NAME]
                        [--ckpt-dir CKPT_DIR]
                        [--write-checkpoint-path WRITE_CHECKPOINT_PATH]
                        [--restart RESTART]
                        [--physnet-checkpoint PHYSNET_CHECKPOINT]
                        [--physnet-transfer-model PHYSNET_TRANSFER_MODEL]
                        [--list-physnet-transfer-models]
                        [--physnet-transfer-category PHYSNET_TRANSFER_CATEGORY]
                        [--use-repo-physnet-params] [--print-freq PRINT_FREQ]
                        [--plot-results] [--plot-freq PLOT_FREQ]
                        [--plot-samples PLOT_SAMPLES]
                        [--plot-esp-examples PLOT_ESP_EXAMPLES] [--verbose]

Joint PhysNet-DCMNet training

options:
  -h, --help            show this help message and exit
  --train-efd TRAIN_EFD
                        Training energies/forces/dipoles NPZ file
  --train-esp TRAIN_ESP
                        Training ESP grids NPZ file
  --valid-efd VALID_EFD
                        Validation energies/forces/dipoles NPZ file
  --valid-esp VALID_ESP
                        Validation ESP grids NPZ file
  --subtract-atom-energies
                        Subtract reference atomic energies from total energies
                        (default: do not subtract)
  --physnet-features PHYSNET_FEATURES
                        PhysNet: number of features
  --physnet-iterations PHYSNET_ITERATIONS
                        PhysNet: message passing iterations
  --physnet-basis PHYSNET_BASIS
                        PhysNet: number of basis functions
  --physnet-cutoff PHYSNET_CUTOFF
                        PhysNet: cutoff distance (Angstroms)
  --physnet-n-res PHYSNET_N_RES
                        PhysNet: number of residual blocks
  --zbl                 Enable PhysNet ZBL short-range repulsion
  --no-zbl              Disable PhysNet ZBL short-range repulsion
  --physnet-max-degree PHYSNET_MAX_DEGREE
                        PhysNet: maximum spherical harmonic degree
  --dcmnet-features DCMNET_FEATURES
                        DCMNet: number of features
  --dcmnet-iterations DCMNET_ITERATIONS
                        DCMNet: message passing iterations
  --dcmnet-basis DCMNET_BASIS
                        DCMNet: number of basis functions
  --dcmnet-cutoff DCMNET_CUTOFF
                        DCMNet: cutoff distance (Angstroms)
  --n-dcm N_DCM         DCMNet: distributed multipoles per atom
  --max-degree MAX_DEGREE
                        DCMNet: maximum spherical harmonic degree
  --use-noneq-model     Use non-equivariant charge model instead of DCMNet
                        (predicts Cartesian displacements)
  --noneq-features NONEQ_FEATURES
                        Non-equivariant model: hidden layer size
  --noneq-layers NONEQ_LAYERS
                        Non-equivariant model: number of MLP layers
  --noneq-max-displacement NONEQ_MAX_DISPLACEMENT
                        Non-equivariant model: maximum displacement distance
                        (Angstroms)
  --batch-size BATCH_SIZE
                        Batch size (start with 1 for debugging)
  --epochs EPOCHS       Number of epochs
  --optimizer {adam,adamw,rmsprop,muon}
                        Optimizer choice (default: adam)
  --learning-rate, --lr LEARNING_RATE
                        Learning rate (default: auto-select based on dataset and
                        optimizer)
  --weight-decay WEIGHT_DECAY
                        Weight decay/L2 regularization (default: auto-select
                        based on optimizer)
  --use-recommended-hparams
                        Use recommended hyperparameters based on dataset
                        properties (overrides manual settings)
  --seed SEED           Random seed
  --energy-weight ENERGY_WEIGHT
                        Energy loss weight
  --forces-weight FORCES_WEIGHT
                        Forces loss weight
  --dipole-weight DIPOLE_WEIGHT
                        Dipole loss weight
  --esp-weight ESP_WEIGHT
                        ESP loss weight
  --esp-min-distance ESP_MIN_DISTANCE
                        Additional minimum distance (Å) from atoms for ESP grid
                        points (default: 0, uses 2×atomic_radius). Set > 0 to
                        add extra distance constraint.
  --esp-max-value ESP_MAX_VALUE
                        Maximum |ESP| value (Hartree/e) to include in loss -
                        filters out high ESP points (default: no limit)
  --mono-weight MONO_WEIGHT
                        Monopole constraint loss weight (enforce distributed
                        charges sum to atomic charges)
  --charge-reg-weight CHARGE_REG_WEIGHT
                        L2 regularization on DCMNet charge magnitudes to prevent
                        blow-up (default: 1.0)
  --dipole-source {physnet,dcmnet,mixed}
                        Source for dipole in loss: physnet (from charges) or
                        dcmnet (from distributed multipoles)
  --dipole-loss-sources [{physnet,dcmnet,mixed} ...]
                        Override dipole supervision sources (e.g. physnet dcmnet
                        mixed). Defaults to --dipole-source when omitted.
  --esp-loss-sources [{physnet,dcmnet,mixed} ...]
                        ESP supervision sources (e.g. dcmnet physnet mixed).
                        Defaults to dcmnet when omitted.
  --dipole-metric {l2,mae,rmse}
                        Error metric for default dipole loss terms (ignored when
                        --loss-config specified)
  --esp-metric {l2,mae,rmse}
                        Error metric for default ESP loss terms (ignored when
                        --loss-config specified)
  --loss-config LOSS_CONFIG
                        Optional JSON or YAML file defining dipole/ESP loss
                        terms (overrides individual loss source flags)
  --mix-coulomb-energy  Mix PhysNet energy with DCMNet Coulomb energy (fixed
                        λ=1; optional warmup schedule)
  --disable-physnet-point-coulomb
                        Disable PhysNet point-charge electrostatics term while
                        still predicting charges
  --mix-warmup-start MIX_WARMUP_START
                        Epoch to start ramping Coulomb mix weight
  --mix-warmup-end MIX_WARMUP_END
                        Epoch to finish ramping Coulomb mix weight
  --mix-weight-max MIX_WEIGHT_MAX
                        Maximum effective Coulomb mix weight (0-1)
  --mix-schedule {linear,cosine}
                        Ramp shape for Coulomb mix warmup
  --natoms NATOMS       Maximum number of atoms (default: auto-detect from data)
  --max-atomic-number MAX_ATOMIC_NUMBER
                        Maximum atomic number
  --grad-clip-norm GRAD_CLIP_NORM
                        Gradient clipping norm (None to disable)
  --name NAME           Experiment name
  --ckpt-dir CKPT_DIR   Checkpoint directory
  --write-checkpoint-path WRITE_CHECKPOINT_PATH
                        After training, write the resolved run directory (ckpt-
                        dir/name) to this file as a single line (for shell
                        scripts and asset tools).
  --restart RESTART     Restart from checkpoint (path to best_params.pkl or
                        checkpoint directory)
  --physnet-checkpoint PHYSNET_CHECKPOINT
                        Load PhysNet params from a pre-trained checkpoint (e.g.
                        from step 09). Path to orbax experiment dir (e.g.
                        <name>-<uuid>) or epoch dir, or JSON.
  --physnet-transfer-model PHYSNET_TRANSFER_MODEL
                        Bundled PhysNet transfer model ID, file stem, or
                        category. Defaults to the 'joint-training-defaults'
                        charged model for fresh joint training. Use --list-
                        physnet-transfer-models to inspect choices.
  --list-physnet-transfer-models
                        List bundled PhysNet transfer-learning models and exit.
  --physnet-transfer-category PHYSNET_TRANSFER_CATEGORY
                        Filter --list-physnet-transfer-models by manifest
                        category.
  --use-repo-physnet-params
                        Initialize the PhysNet part of joint DCMNet training
                        from the bundled repo PhysNet parameters (mmml/models/ph
                        ysnetjax/defaults/meoh_dimer_portable.json).
  --print-freq PRINT_FREQ
                        Print frequency (epochs)
  --plot-results        Create validation plots after training
  --plot-freq PLOT_FREQ
                        Create validation plots every N epochs during training
                        (default: 10, set to 0 to disable)
  --plot-samples PLOT_SAMPLES
                        Number of validation samples to plot
  --plot-esp-examples PLOT_ESP_EXAMPLES
                        Number of ESP examples to visualize
  --verbose             Verbose output
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
