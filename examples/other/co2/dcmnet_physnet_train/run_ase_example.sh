#!/bin/bash
# Example ASE calculations with trained joint model

# Full workflow: optimize, frequencies, and IR spectra
python ase_calculator.py \
  --checkpoint mmml/physnetjax/ckpts/co2_joint_physnet_dcmnet/best_params.pkl \
  --molecule co2 \
  --optimize \
  --frequencies \
  --ir-spectra \
  --output-dir ase_results/co2_full

# Just optimization
python ase_calculator.py \
  --checkpoint mmml/physnetjax/ckpts/co2_joint_physnet_dcmnet/best_params.pkl \
  --molecule co2 \
  --optimize \
  --output-dir ase_results/co2_opt

# Just frequencies
python ase_calculator.py \
  --checkpoint mmml/physnetjax/ckpts/co2_joint_physnet_dcmnet/best_params.pkl \
  --molecule co2 \
  --frequencies \
  --output-dir ase_results/co2_freq

