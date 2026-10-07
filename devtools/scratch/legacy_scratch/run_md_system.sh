#!/bin/bash
export CHARMM_LIB_DIR=/Users/ericboittier/karml/setup/charmm
KARML_MPI_NP=1 ./scripts/karml-charmm-mpirun.sh /Users/ericboittier/karml/.venv/bin/karml md-system --composition DCM:4 --no-periodic-charmm-vdw --backend pycharmm --n-prod 1 --n-equil 0 --output-dir scratch/md_out --skip-if-crd-exists --tag test
