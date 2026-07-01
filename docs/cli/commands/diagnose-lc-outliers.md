# `mmml diagnose-lc-outliers`

Inspect learning-curve sweeps for bad seeds and NPZ outliers.


## Usage

```bash
mmml diagnose-lc-outliers --help
```

## Options

```text
usage: mmml diagnose-lc-outliers [-h] --eval-root EVAL_ROOT
                                 [--dataset DATASET] [--train-npz TRAIN_NPZ]
                                 [--json-out JSON_OUT]
                                 [--structure-plot-out STRUCTURE_PLOT_OUT]
                                 [--structure-indices STRUCTURE_INDICES]
                                 [--structure-reference-index STRUCTURE_REFERENCE_INDEX]
                                 [--max-structures MAX_STRUCTURES]
                                 [--plot-out PLOT_OUT]
                                 [--plot-style PLOT_STYLE]
                                 [--spike-relative-factor SPIKE_RELATIVE_FACTOR]
                                 [--spike-jump-factor SPIKE_JUMP_FACTOR]
                                 [--test-z-threshold TEST_Z_THRESHOLD]
                                 [--test-ratio-threshold TEST_RATIO_THRESHOLD]
                                 [--top-samples TOP_SAMPLES]
                                 [--top-spikes TOP_SPIKES] [-q]

Diagnose learning-curve sweep outliers (seeds, spikes, NPZ samples).

options:
  -h, --help            show this help message and exit
  --eval-root EVAL_ROOT
                        Sweep eval root (…/learning_curve/e1000)
  --dataset DATASET     Dataset subdir under eval-root (default: aco)
  --train-npz TRAIN_NPZ
                        Train NPZ for split reproduction + scoring
  --json-out JSON_OUT   Write full JSON report
  --structure-plot-out STRUCTURE_PLOT_OUT
                        Write Kabsch-aligned ASE structure overlay (requires
                        --train-npz)
  --structure-indices STRUCTURE_INDICES
                        Comma-separated NPZ indices to plot (default: top
                        suspects + global outliers)
  --structure-reference-index STRUCTURE_REFERENCE_INDEX
                        Reference NPZ index for alignment (default: first
                        plotted index)
  --max-structures MAX_STRUCTURES
                        Max structures in auto-selected overlay (default: 8)
  --plot-out PLOT_OUT   Write summary dashboard PNG
  --plot-style PLOT_STYLE
                        Matplotlib style preset
  --spike-relative-factor SPIKE_RELATIVE_FACTOR
  --spike-jump-factor SPIKE_JUMP_FACTOR
  --test-z-threshold TEST_Z_THRESHOLD
  --test-ratio-threshold TEST_RATIO_THRESHOLD
  --top-samples TOP_SAMPLES
  --top-spikes TOP_SPIKES
  -q, --quiet           Suppress tables; still writes --json-out

Example:
  mmml diagnose-lc-outliers \
    --eval-root out/eval/learning_curve/e1000 \
    --dataset aco \
    --train-npz out/splits/aco/energies_forces_dipoles_train.npz
```


---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
