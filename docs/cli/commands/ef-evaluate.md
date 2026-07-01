# `mmml ef-evaluate`

Evaluate EF model (deprecated).

!!! warning "deprecated"
    Deprecated command. Prefer **`mmml efield-evaluate`**.


## Usage

```bash
mmml ef-evaluate --help
```

## Options

```text
usage: mmml ef-evaluate [-h] [--params PARAMS] [--config CONFIG] [--data DATA]
                        [--test-npz TEST_NPZ] [--output-dir OUTPUT_DIR]
                        [--batch-size BATCH_SIZE] [--num-test NUM_TEST]
                        [--model-config MODEL_CONFIG] [--features FEATURES]
                        [--max-degree MAX_DEGREE]
                        [--num-iterations NUM_ITERATIONS]
                        [--num-basis-functions NUM_BASIS_FUNCTIONS]
                        [--cutoff CUTOFF]
                        [--max-atomic-number MAX_ATOMIC_NUMBER]
                        [--save-output-npz] [--output-h5 PATH] [--rot-augment]
                        [--rot-perturbation ROT_PERTURBATION]

Evaluate trained model

options:
  -h, --help            show this help message and exit
  --params PARAMS       Path to parameters JSON file (can be params-UUID.json or
                        params.json)
  --config CONFIG       Path to config JSON file (will be auto-detected from
                        params UUID if not provided)
  --data DATA           Path to dataset NPZ file
  --test-npz TEST_NPZ   Test split NPZ (alias for --data; same as ef-train
                        --test-npz)
  --output-dir OUTPUT_DIR
                        Output directory for plots and metrics
  --batch-size BATCH_SIZE
                        Batch size for evaluation
  --num-test NUM_TEST   Number of test samples to use (None = use all)
  --model-config MODEL_CONFIG
                        Path to model config JSON (deprecated: use --config)
  --features FEATURES   Model features (will be inferred from params/config if
                        not provided)
  --max-degree MAX_DEGREE
                        Max degree (default: 2)
  --num-iterations NUM_ITERATIONS
                        Number of iterations (default: 2)
  --num-basis-functions NUM_BASIS_FUNCTIONS
                        Number of basis functions (default: 64)
  --cutoff CUTOFF       Cutoff radius (default: 10.0)
  --max-atomic-number MAX_ATOMIC_NUMBER
                        Max atomic number (default: 55)
  --save-output-npz     Save evaluation outputs (predictions, targets) to NPZ
                        file
  --output-h5 PATH      Write HDF5 trajectory for mmml gui
                        (R,Z,N,E,E_pred,F,F_pred,Dxyz,Dxyz_pred,Ef). Requires
                        h5py.
  --rot-augment         Apply random SO(3) rotation augmentation when building
                        batches (via prepare_batches)
  --rot-perturbation ROT_PERTURBATION
                        Rotation perturbation strength in [0, 1] (used with
                        --rot-augment)
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
