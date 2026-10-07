#!/bin/bash
$RES = $1
$FILES = $2
$NUM_TRAIN = $3
$NUM_VALID = $4
$NATOMS = $5

python model_analysis_utils.py \
  --restart $RES \
  --files $FILES \
  --num_train $NTRAIN \
  --num_valid $NVALID \
  --natoms $NATOMS \
  --load_test \
  --do_plot \
  --save_results