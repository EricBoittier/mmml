#!/bin/bash
cd /mmhome/boittier/home/karml
source .venv/bin/activate
for f in TorsionNet500 md22_bb gems_crambin; do
  echo "=== $f ==="
  python scripts/decompose_so3lr_terms_vs_natoms.py \
    --checkpoint /mmhome/boittier/home/karml/artifacts/spooky_so3lr_muon3/epoch-0010 \
    --extxyz "$HOME/data/so3lr_test/${f}.extxyz" \
    --max-per-dataset 10 \
    --out-csv "eval_out/charges_${f}.csv" \
    --out-plot "eval_out/charges_${f}.png"
done
