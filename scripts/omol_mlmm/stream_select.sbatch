#!/bin/bash
#SBATCH -p short -c 20 --mem=24G -t 08:00:00 -x node08 --constraint=E5-2630v4 -J omol_stream_sel -o /mmhome/boittier/home/metatomic-runs/training/omol_universal/mlmm_ft/logs/stream_select.%j.out
# Stream the full OMol25 tarball once; map each new LMDB member on node-local disk (no NFS write-back).
set -euo pipefail; S=$(mktemp -d /tmp/omol_stream.XXXXXX); trap 'rm -rf $S' EXIT
cd /tmp; export PYTHONPATH=/mmhome/boittier/home/metatomic-runs/training/omol_universal/pydeps:/mmhome/boittier/home/metatomic-runs/training/omol_universal/mlmm_ft/scripts; P=/mmhome/boittier/home/metatomic-runs/training/omol_universal/raw/train_full_parts
df -h /tmp | tail -1
cat $P/part0 $P/part1 $P/part2 $P/part3 $P/part4 $P/part5 $P/part6 $P/part7 |   /mmhome/boittier/home/mmml-latest/.venv-cu12/bin/python -P /mmhome/boittier/home/metatomic-runs/training/omol_universal/mlmm_ft/scripts/stream_select.py /mmhome/boittier/home/metatomic-runs/training/omol_universal/mlmm_ft/map_full $S /mmhome/boittier/home/metatomic-runs/training/omol_universal/mlmm_ft/map_full_ready 19
