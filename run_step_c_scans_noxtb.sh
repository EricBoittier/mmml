#!/bin/bash
cd /mmhome/boittier/home/mmml
source .venv/bin/activate
python scripts/run_dimer_scan_campaign.py \
  --spookynet-checkpoint spooky_so3lr_muon3_epoch0013.json \
  --spookynet-tag muon3_ep13 \
  --skip-xtb \
  --output-dir results/dimer_scan_campaign_muon3_ep13
