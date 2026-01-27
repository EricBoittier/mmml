source settings.source

export CHECKPOINT="$PWD/ACO-b4f39bb9-8ca7-485e-bf51-2e5236e51b56"


python -m mmml.cli.make_training \
  --data $DATA  \
  --ckpt_dir $PWD  \
  --tag $TAG \
  --n_train $NTRAIN \
  --n_valid $NVALID \
  --batch_size 64 \
  --num_epochs 10000 \
  --learning_rate 0.0001 \
  --energy_weight 50.0 \
  --objective "valid_loss" \
  --seed 42 \
  --num_atoms $NATOMSDATA \
  --features 24  --max_degree 1 \
  --num_basis_functions 12 \
  --num_iterations 2 \
  --restart $CHECKPOINT \
  --n_res 2 \
  --cutoff 5.0 \
  --max_atomic_number 28
