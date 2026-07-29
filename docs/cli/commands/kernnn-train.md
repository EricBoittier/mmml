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
                         [--ema-decay EMA_DECAY]
                         [--kernel {k20,k21,k22,k23,k24,k25,k26,k30,k31,k32,k33,k34,k35,k36}]
                         [--list-kernels]
                         [--distance-scheme {abcc,abcc_sym,acem,form}]
                         [--architecture {ffnet,dual}]
                         [--teacher-checkpoint TEACHER_CHECKPOINT]
                         [--distill-alpha DISTILL_ALPHA]
                         [--teacher-energy-offset TEACHER_ENERGY_OFFSET]
                         [--no-align-teacher-energy]
                         [--teacher-align-n TEACHER_ALIGN_N]

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
  --teacher-energy-offset TEACHER_ENERGY_OFFSET
                        Add this constant (eV) to teacher energies before
                        distill loss (overrides auto-align). Use when PhysNet
                        atom refs shift the zero.
  --no-align-teacher-energy
                        Do not auto-fit an additive teacher energy offset vs GT

Execution:
  --seed SEED           RNG seed for split/init
  --batch-size BATCH_SIZE
  --epochs EPOCHS

Output & artifacts:
  --workdir WORKDIR     Output directory

Diagnostics & safety:
  -h, --help            show this help message and exit
  --list-kernels        Print the table of available 1D kernel functions and
                        exit

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
  --kernel {k20,k21,k22,k23,k24,k25,k26,k30,k31,k32,k33,k34,k35,k36}
                        1D kernel name (default k33)
  --architecture {ffnet,dual}
                        ffnet (default) or dual (ABCC + dihedral only)
  --distill-alpha DISTILL_ALPHA
                        Blend GT vs teacher: loss = alpha*GT + (1-alpha)*teacher
                        (1=pure GT)
  --teacher-align-n TEACHER_ALIGN_N
                        Number of train structures used to estimate teacher
                        energy offset
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
