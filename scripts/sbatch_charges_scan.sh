#!/bin/bash
#SBATCH --job-name=charges_scan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mmhome/boittier/home/mmml/slurm_logs/charges_scan_%j.out

cd /mmhome/boittier/home/mmml
source .venv/bin/activate

python scripts/scan_charges_vs_distance.py \
  --checkpoint artifacts/spooky_so3lr_muon3/epoch-0010 \
  --reference-csv results/dimer_scan_campaign_muon3_ep10/scan_results.csv \
  --pairs TIP3:TIP3 MEOH:MEOH TIP3:MEOH DCM:DCM ACE:ACE \
  --out-csv eval_out/charges_vs_distance_ep10.csv

python scripts/scan_charges_vs_distance.py \
  --checkpoint artifacts/spooky_so3lr_muon3/epoch-0013 \
  --reference-csv results/dimer_scan_campaign_muon3_ep13/scan_results.csv \
  --pairs TIP3:TIP3 MEOH:MEOH TIP3:MEOH DCM:DCM ACE:ACE \
  --out-csv eval_out/charges_vs_distance_ep13.csv
