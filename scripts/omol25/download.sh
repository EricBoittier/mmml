#!/bin/bash
#SBATCH -p short -c 2 --mem=4G -t 12:00:00 -x node08 -J omol_dl -o /mmhome/boittier/home/metatomic-runs/training/omol_universal/logs/download.%j.out
# OMol25 (CC-BY-4.0) from Meta's public file server: 4M training subset + validation split, ASE-LMDB tarballs.
set -euo pipefail
cd /mmhome/boittier/home/metatomic-runs/training/omol_universal/raw
for f in train_4M.tar.gz val.tar.gz; do
  wget -q -c https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/$f
  ls -la $f
done
for f in train_4M.tar.gz val.tar.gz; do d=${f%.tar.gz}; mkdir -p $d; tar xzf $f -C $d; du -sh $d; done
echo done $(date)
