#!/bin/bash
cd /mmhome/boittier/home/mmml
source .venv/bin/activate
nohup python scripts/train_so3lr_spooky_extxyz.py \
  --mode train \
  --cache-path /mmhome/boittier/home/data/so3lr_orbax_cache/so3lr_train_flat_2ef9214c00a78127 \
  --workdir artifacts/spooky_so3lr_adam_cw2 \
  --init-checkpoint artifacts/spooky_so3lr_muon3/epoch-0010 \
  --epochs 50 \
  --num-devices 2 \
  --batch-size-per-device 8 \
  --max-pairs-per-device 18000 \
  --atom-bucket-width 4 \
  --auto-batch --force-auto-batch --auto-batch-margin 2 \
  --optimizer adamw \
  --learning-rate 1e-4 \
  --lr-schedule warmup_cosine --lr-warmup-steps 1000 --lr-decay-steps 0 --lr-end-fraction 0.05 \
  --weight-decay 0.0 --clip-global-norm 1.0 \
  --energy-weight 1.0 --forces-weight 52.91 --dipole-weight 1.0 \
  --charges-weight 2.0 \
  --features 128 --max-degree 2 --num-iterations 3 --num-basis-functions 32 \
  --cutoff 6.0 --max-atomic-number 87 --n-res 2 \
  --predict-charges --trainable-zbl --zbl-cuton 0.1 --zbl-cutoff 0.6 \
  --electrostatics-damping-sigma 4.0 \
  --valid-fraction 0.05 --valid-steps 100 \
  --save-every 1 --save-every-steps 20000 \
  --seed 0 \
  --log-every 10000 --log-every-steps 200 \
  > /mmhome/boittier/home/mmml/train_adam_cw2.log 2>&1 &
disown
echo "launched with PID $!"
sleep 3
tail -5 /mmhome/boittier/home/mmml/train_adam_cw2.log 2>&1
