#!/bin/bash
cd /mmhome/boittier/home/mmml
source .venv/bin/activate
python scripts/evaluate_so3lr_spooky_extxyz.py \
  --checkpoint artifacts/spooky_so3lr_muon3/epoch-0013 \
  --extxyz "$HOME/data/so3lr_test/gems_crambin.extxyz" \
  --cache-dir "$HOME/data/so3lr_orbax_cache/muon3_epoch0013" \
  --max-eval-structures 20 \
  --output eval_out/spooky_so3lr_muon3_epoch0013_crambin_sample.json \
  --plots-dir eval_out/spooky_so3lr_muon3_epoch0013_plots
