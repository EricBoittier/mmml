#!/bin/bash
#SBATCH --job-name=dimer_scan_ep10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mmhome/boittier/home/mmml/slurm_logs/dimer_scan_ep10_%j.out

cd /mmhome/boittier/home/mmml
source .venv/bin/activate
python scripts/run_dimer_scan_campaign.py \
  --spookynet-checkpoint spooky_so3lr_muon3_epoch0010.json \
  --spookynet-tag muon3_ep10 \
  --skip-xtb \
  --output-dir results/dimer_scan_campaign_muon3_ep10
