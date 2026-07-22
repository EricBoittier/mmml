#!/bin/bash
cd /mmhome/boittier/home/mmml
source .venv/bin/activate
python scripts/evaluate_so3lr_spooky_extxyz.py \
  --checkpoint artifacts/spooky_so3lr_muon3/epoch-0013 \
  --extxyz "$HOME/data/so3lr_test/" \
  --cache-dir "$HOME/data/so3lr_orbax_cache/muon3_epoch0013" \
  --output eval_out/spooky_so3lr_muon3_epoch0013.json \
  --plots-dir eval_out/spooky_so3lr_muon3_epoch0013_plots
