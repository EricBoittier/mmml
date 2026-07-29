# `mmml kernnn-train`

Train KerNN kernel Softplus MLP (E/F).


## Usage

```bash
mmml kernnn-train --help
```

## Options

```text
usage: mmml kernnn-train [-h] [--data DATA] [--train-npz TRAIN_NPZ]
                         [--valid-npz VALID_NPZ] [--test-npz TEST_NPZ]
                         [--workdir WORKDIR] [--ntrain NTRAIN] [--nvalid NVALID]
                         [--seed SEED] [--n-hidden N_HIDDEN]
                         [--batch-size BATCH_SIZE]
                         [--learning-rate LEARNING_RATE] [--f-weight F_WEIGHT]
                         [--epochs EPOCHS] [--patience PATIENCE]
                         [--ema-decay EMA_DECAY] [--kernel KERNEL]
                         [--distance-scheme {abcc,abcc_sym,acem,form}]
                         [--architecture {ffnet,dual}]
                         [--teacher-checkpoint TEACHER_CHECKPOINT]
                         [--distill-alpha DISTILL_ALPHA]

Train KerNN (kernel Softplus MLP) on NPZ (R, E, F)

Input & configuration:
  --data DATA           Single NPZ with R,E,F (random train/valid/test split)
  --teacher-checkpoint TEACHER_CHECKPOINT
                        PhysNet checkpoint (JSON/Orbax) used as distillation
                        teacher

Scientific model:
  --distance-scheme {abcc,abcc_sym,acem,form}
                        Distance descriptor: abcc, abcc_sym, form (6 atoms),
                        acem (9 atoms)

Execution:
  --seed SEED           RNG seed for split/init
  --batch-size BATCH_SIZE
  --epochs EPOCHS

Output & artifacts:
  --workdir WORKDIR     Output directory

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --train-npz TRAIN_NPZ
                        Train split NPZ
  --valid-npz VALID_NPZ
                        Valid split NPZ
  --test-npz TEST_NPZ   Optional test split NPZ
  --ntrain NTRAIN       Training size when using --data
  --nvalid NVALID       Validation size when using --data
  --n-hidden N_HIDDEN   Hidden layer width
  --learning-rate LEARNING_RATE
  --f-weight F_WEIGHT   Force loss weight
  --patience PATIENCE   Early-stop after this many non-improving validation
                        epochs
  --ema-decay EMA_DECAY
  --kernel KERNEL       1D kernel name (default k33)
  --architecture {ffnet,dual}
                        ffnet (default) or dual (ABCC + dihedral only)
  --distill-alpha DISTILL_ALPHA
                        Blend GT vs teacher: loss = alpha*GT + (1-alpha)*teacher
                        (1=pure GT)
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
