# `mmml evaluate`

Legacy unified model evaluation.

!!! warning "deprecated"
    Deprecated command. Prefer **`mmml physnet-evaluate or efield-evaluate`**.


## Usage

```bash
mmml evaluate --help
```

## Options

```text
usage: mmml evaluate [-h] --model MODEL [MODEL ...] --data DATA [--properties PROPERTIES [PROPERTIES ...]] [--batch-size BATCH_SIZE] [--output OUTPUT] [--report] [--save-predictions] [--verbose] [--quiet]

Evaluate MMML models

options:
  -h, --help            show this help message and exit
  --model MODEL [MODEL ...]
                        Path(s) to model checkpoint(s)
  --data DATA           Test data NPZ file
  --properties PROPERTIES [PROPERTIES ...]
                        Properties to evaluate (default: energy)
  --batch-size BATCH_SIZE
                        Batch size for prediction (default: 32)
  --output OUTPUT       Output directory (default: results)
  --report              Generate markdown report
  --save-predictions    Save predictions to file
  --verbose, -v         Verbose output
  --quiet, -q           Quiet mode

Examples:
  # Basic evaluation
  mmml evaluate --model checkpoint.pkl --data test.npz
  
  # Evaluate specific properties
  mmml evaluate --model checkpoint.pkl --data test.npz \
           --properties energy forces
  
  # Save results and report
  mmml evaluate --model checkpoint.pkl --data test.npz \
           --output results/ --report
  
  # Compare multiple models
  mmml evaluate --model model1.pkl model2.pkl --data test.npz \
           --output comparison/
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
