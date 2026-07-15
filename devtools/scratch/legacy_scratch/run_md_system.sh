#!/bin/bash
export CHARMM_LIB_DIR=/Users/ericboittier/mmml/setup/charmm
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh /Users/ericboittier/mmml/.venv/bin/mmml md-system --composition DCM:4 --no-periodic-charmm-vdw --backend pycharmm --n-prod 1 --n-equil 0 --output-dir scratch/md_out --skip-if-crd-exists --tag test
