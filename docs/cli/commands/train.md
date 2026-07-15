# `mmml train`

Train DCMNet or legacy unified trainer.

!!! warning "deprecated"
    Deprecated command. Prefer **`mmml physnet-train (PhysNet) or train-joint (PhysNet+DCMNet)`**.


## Usage

```bash
mmml train --help
```

## Options

```text
usage: mmml train [-h] [--config CONFIG] [--save-config SAVE_CONFIG]
                  [--model {dcmnet,physnetjax}]
                  [--physnet-checkpoint PHYSNET_CHECKPOINT]
                  [--physnet-transfer-model PHYSNET_TRANSFER_MODEL]
                  [--list-physnet-transfer-models]
                  [--physnet-transfer-category PHYSNET_TRANSFER_CATEGORY]
                  [--train TRAIN] [--valid VALID]
                  [--train-fraction TRAIN_FRACTION] [--batch-size BATCH_SIZE]
                  [--max-epochs MAX_EPOCHS] [--learning-rate LEARNING_RATE]
                  [--early-stopping EARLY_STOPPING]
                  [--targets TARGETS [TARGETS ...]] [--output OUTPUT]
                  [--log-interval LOG_INTERVAL] [--center-coords]
                  [--normalize-energy] [--rot-augment]
                  [--rot-perturbation ROT_PERTURBATION] [--verbose] [--quiet]
                  [--dry-run]

Train MMML models (DCMNet or PhysNetJAX)

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration YAML file
  --save-config SAVE_CONFIG
                        Save configuration to YAML file and exit
  --model {dcmnet,physnetjax}
                        Model to train (default: dcmnet)
  --physnet-checkpoint PHYSNET_CHECKPOINT
                        PhysNet checkpoint path for transfer learning
  --physnet-transfer-model PHYSNET_TRANSFER_MODEL
                        Bundled PhysNet transfer model ID, file stem, or
                        category. Defaults to the 'joint-training-defaults'
                        charged model for physnetjax training.
  --list-physnet-transfer-models
                        List bundled PhysNet transfer-learning models and exit
  --physnet-transfer-category PHYSNET_TRANSFER_CATEGORY
                        Filter --list-physnet-transfer-models by manifest
                        category
  --train TRAIN         Training NPZ file
  --valid VALID         Validation NPZ file (optional if using --train-fraction)
  --train-fraction TRAIN_FRACTION
                        Fraction for train split if --valid not provided
                        (default: 0.8)
  --batch-size BATCH_SIZE
                        Batch size (default: 32)
  --max-epochs MAX_EPOCHS
                        Maximum number of epochs (default: 1000)
  --learning-rate LEARNING_RATE
                        Learning rate (default: 0.001)
  --early-stopping EARLY_STOPPING
                        Early stopping patience (default: 50)
  --targets TARGETS [TARGETS ...]
                        Training targets (default: energy)
  --output OUTPUT       Output directory for checkpoints (default: checkpoints)
  --log-interval LOG_INTERVAL
                        Log interval in epochs (default: 10)
  --center-coords       Center coordinates at origin
  --normalize-energy    Normalize energies
  --rot-augment         Enable SO(3) rotational augmentation in batch builders
  --rot-perturbation ROT_PERTURBATION
                        Rotation perturbation strength in [0,1] (default: 1.0)
  --verbose, -v         Verbose output
  --quiet, -q           Quiet mode
  --dry-run             Prepare data but do not train

Examples: # Train with config file mmml train --config config.yaml # Train
DCMNet from command line mmml train --model dcmnet \ --train train.npz \ --valid
valid.npz \ --output checkpoints/dcmnet/ # Train with auto train/valid split
mmml train --model dcmnet \ --train dataset.npz \ --train-fraction 0.8 # Train
PhysNetJAX (prefer physnet-train for full options / YAML) mmml physnet-train
--config train.yaml mmml train --model physnetjax --train train.npz --valid
valid.npz
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
