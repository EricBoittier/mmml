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
                        [--esp-metric {l2,mae,rmse}]
                        [--loss-config LOSS_CONFIG] [--mix-coulomb-energy]
                        [--disable-physnet-point-coulomb]
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
                        Training energies/forces/dipoles NPZ file (default:
                        None)
  --train-esp TRAIN_ESP
                        Training ESP grids NPZ file (default: None)
  --valid-efd VALID_EFD
                        Validation energies/forces/dipoles NPZ file (default:
                        None)
  --valid-esp VALID_ESP
                        Validation ESP grids NPZ file (default: None)
  --subtract-atom-energies
                        Subtract reference atomic energies from total energies
                        (default: do not subtract) (default: False)
  --physnet-features PHYSNET_FEATURES
                        PhysNet: number of features (default: 64)
  --physnet-iterations PHYSNET_ITERATIONS
                        PhysNet: message passing iterations (default: 3)
  --physnet-basis PHYSNET_BASIS
                        PhysNet: number of basis functions (default: 64)
  --physnet-cutoff PHYSNET_CUTOFF
                        PhysNet: cutoff distance (Angstroms) (default: 6.0)
  --physnet-n-res PHYSNET_N_RES
                        PhysNet: number of residual blocks (default: 3)
  --zbl                 Enable PhysNet ZBL short-range repulsion (default:
                        True)
  --no-zbl              Disable PhysNet ZBL short-range repulsion (default:
                        True)
  --physnet-max-degree PHYSNET_MAX_DEGREE
                        PhysNet: maximum spherical harmonic degree (default:
                        0)
  --dcmnet-features DCMNET_FEATURES
                        DCMNet: number of features (default: 128)
  --dcmnet-iterations DCMNET_ITERATIONS
                        DCMNet: message passing iterations (default: 2)
  --dcmnet-basis DCMNET_BASIS
                        DCMNet: number of basis functions (default: 64)
  --dcmnet-cutoff DCMNET_CUTOFF
                        DCMNet: cutoff distance (Angstroms) (default: 10.0)
  --n-dcm N_DCM         DCMNet: distributed multipoles per atom (default: 3)
  --max-degree MAX_DEGREE
                        DCMNet: maximum spherical harmonic degree (default: 2)
  --use-noneq-model     Use non-equivariant charge model instead of DCMNet
                        (predicts Cartesian displacements) (default: False)
  --noneq-features NONEQ_FEATURES
                        Non-equivariant model: hidden layer size (default:
                        128)
  --noneq-layers NONEQ_LAYERS
                        Non-equivariant model: number of MLP layers (default:
                        3)
  --noneq-max-displacement NONEQ_MAX_DISPLACEMENT
                        Non-equivariant model: maximum displacement distance
                        (Angstroms) (default: 0.5)
  --batch-size BATCH_SIZE
                        Batch size (start with 1 for debugging) (default: 1)
  --epochs EPOCHS       Number of epochs (default: 100)
  --optimizer {adam,adamw,rmsprop,muon}
                        Optimizer choice (default: adam) (default: adam)
  --learning-rate, --lr LEARNING_RATE
                        Learning rate (default: auto-select based on dataset
                        and optimizer) (default: None)
  --weight-decay WEIGHT_DECAY
                        Weight decay/L2 regularization (default: auto-select
                        based on optimizer) (default: None)
  --use-recommended-hparams
                        Use recommended hyperparameters based on dataset
                        properties (overrides manual settings) (default:
                        False)
  --seed SEED           Random seed (default: 42)
  --energy-weight ENERGY_WEIGHT
                        Energy loss weight (default: 10.0)
  --forces-weight FORCES_WEIGHT
                        Forces loss weight (default: 50.0)
  --dipole-weight DIPOLE_WEIGHT
                        Dipole loss weight (default: 25.0)
  --esp-weight ESP_WEIGHT
                        ESP loss weight (default: 10000.0)
  --esp-min-distance ESP_MIN_DISTANCE
                        Additional minimum distance (Å) from atoms for ESP
                        grid points (default: 0, uses 2×atomic_radius). Set >
                        0 to add extra distance constraint. (default: 0.0)
  --esp-max-value ESP_MAX_VALUE
                        Maximum |ESP| value (Hartree/e) to include in loss -
                        filters out high ESP points (default: no limit)
                        (default: None)
  --mono-weight MONO_WEIGHT
                        Monopole constraint loss weight (enforce distributed
                        charges sum to atomic charges) (default: 100.0)
  --charge-reg-weight CHARGE_REG_WEIGHT
                        L2 regularization on DCMNet charge magnitudes to
                        prevent blow-up (default: 1.0) (default: 1.0)
  --dipole-source {physnet,dcmnet,mixed}
                        Source for dipole in loss: physnet (from charges) or
                        dcmnet (from distributed multipoles) (default:
                        physnet)
  --dipole-loss-sources [{physnet,dcmnet,mixed} ...]
                        Override dipole supervision sources (e.g. physnet
                        dcmnet mixed). Defaults to --dipole-source when
                        omitted. (default: None)
  --esp-loss-sources [{physnet,dcmnet,mixed} ...]
                        ESP supervision sources (e.g. dcmnet physnet mixed).
                        Defaults to dcmnet when omitted. (default: None)
  --dipole-metric {l2,mae,rmse}
                        Error metric for default dipole loss terms (ignored
                        when --loss-config specified) (default: l2)
  --esp-metric {l2,mae,rmse}
                        Error metric for default ESP loss terms (ignored when
                        --loss-config specified) (default: l2)
  --loss-config LOSS_CONFIG
                        Optional JSON or YAML file defining dipole/ESP loss
                        terms (overrides individual loss source flags)
                        (default: None)
  --mix-coulomb-energy  Mix PhysNet energy with DCMNet Coulomb energy (fixed
                        λ=1; optional warmup schedule) (default: False)
  --disable-physnet-point-coulomb
                        Disable PhysNet point-charge electrostatics term while
                        still predicting charges (default: False)
  --mix-warmup-start MIX_WARMUP_START
                        Epoch to start ramping Coulomb mix weight (default: 1)
  --mix-warmup-end MIX_WARMUP_END
                        Epoch to finish ramping Coulomb mix weight (default:
                        1)
  --mix-weight-max MIX_WEIGHT_MAX
                        Maximum effective Coulomb mix weight (0-1) (default:
                        1.0)
  --mix-schedule {linear,cosine}
                        Ramp shape for Coulomb mix warmup (default: linear)
  --natoms NATOMS       Maximum number of atoms (default: auto-detect from
                        data) (default: None)
  --max-atomic-number MAX_ATOMIC_NUMBER
                        Maximum atomic number (default: 28)
  --grad-clip-norm GRAD_CLIP_NORM
                        Gradient clipping norm (None to disable) (default:
                        1.0)
  --name NAME           Experiment name (default: joint_physnet_dcmnet)
  --ckpt-dir CKPT_DIR   Checkpoint directory (default: None)
  --write-checkpoint-path WRITE_CHECKPOINT_PATH
                        After training, write the resolved run directory
                        (ckpt-dir/name) to this file as a single line (for
                        shell scripts and asset tools). (default: None)
  --restart RESTART     Restart from checkpoint (path to best_params.pkl or
                        checkpoint directory) (default: None)
  --physnet-checkpoint PHYSNET_CHECKPOINT
                        Load PhysNet params from a pre-trained checkpoint
                        (e.g. from step 09). Path to orbax experiment dir
                        (e.g. <name>-<uuid>) or epoch dir, or JSON. (default:
                        None)
  --physnet-transfer-model PHYSNET_TRANSFER_MODEL
                        Bundled PhysNet transfer model ID, file stem, or
                        category. Defaults to the 'joint-training-defaults'
                        charged model for fresh joint training. Use --list-
                        physnet-transfer-models to inspect choices. (default:
                        None)
  --list-physnet-transfer-models
                        List bundled PhysNet transfer-learning models and
                        exit. (default: False)
  --physnet-transfer-category PHYSNET_TRANSFER_CATEGORY
                        Filter --list-physnet-transfer-models by manifest
                        category. (default: None)
  --use-repo-physnet-params
                        Initialize the PhysNet part of joint DCMNet training
                        from the bundled repo PhysNet parameters (/Users/ericb
                        oittier/mmml/mmml/models/physnetjax/defaults/meoh_dime
                        r_portable.json). (default: False)
  --print-freq PRINT_FREQ
                        Print frequency (epochs) (default: 1)
  --plot-results        Create validation plots after training (default:
                        False)
  --plot-freq PLOT_FREQ
                        Create validation plots every N epochs during training
                        (default: 10, set to 0 to disable) (default: 10)
  --plot-samples PLOT_SAMPLES
                        Number of validation samples to plot (default: 100)
  --plot-esp-examples PLOT_ESP_EXAMPLES
                        Number of ESP examples to visualize (default: 2)
  --verbose             Verbose output (default: True)
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
