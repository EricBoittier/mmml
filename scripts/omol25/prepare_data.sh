#!/bin/bash
# OMol25 -> SpookyPhysNet caches (CPU only). Usage: bash scripts/prepare_data.sh [workers]
# 1) stats  2) filter+convert to npz shards  3) Orbax caches: pilot (200k) and full.
set -euo pipefail
U=/mmhome/boittier/home/metatomic-runs/training/omol_universal
P=/mmhome/boittier/home/mmml-latest/.venv-cu12/bin/python
W=${1:-8}
cd /tmp   # never run python from inside the (untrusted) data dirs
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu

$P -I $U/scripts/omol_stats.py --input $U/raw/train_4M --split train_4M --workers $W
$P -I $U/scripts/omol_stats.py --input $U/raw/val --split val --workers $W

$P -I $U/scripts/omol_to_npz.py --input $U/raw/train_4M --out $U/npz/train_4M --workers $W
$P -I $U/scripts/omol_to_npz.py --input $U/raw/val --out $U/npz/val --workers $W

$P -I $U/scripts/npz_to_spooky_cache.py --shards $U/npz/train_4M --n-structures 200000 --seed 0 \
    --linear-ref fit --out $U/caches/omol_train4M_pilot200k
$P -I $U/scripts/npz_to_spooky_cache.py --shards $U/npz/train_4M --linear-ref fit \
    --out $U/caches/omol_train4M_filtered
# held-out OMol25 val with the full-train linear_ref applied (for the pilot, use --linear-ref-from caches/omol_train4M_pilot200k/metadata.json)
$P -I $U/scripts/npz_to_spooky_cache.py --shards $U/npz/val --n-structures 50000 --seed 1 \
    --linear-ref-from $U/caches/omol_train4M_filtered/metadata.json --out $U/caches/omol_val_50k
