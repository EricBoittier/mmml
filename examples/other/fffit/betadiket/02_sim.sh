#!/usr/bin/env bash
# Wrapper script to run MD simulation with environment variables
echo $PWD

cat settings.source

source settings.source


# ----------- Dummy environment variables (adjust as needed) -----------
export PDBFILE="$PWD/pdb/init-packmol.pdb"
export CHECKPOINT="$PWD/ACO-b4f39bb9-8ca7-485e-bf51-2e5236e51b56"
export ENERGY_CATCH="0.5"
export N_MONOMERS=$N
export N_ATOMS_MONOMER=$NATOMS
export ML_CUTOFF="1.0"
export MM_SWITCH_ON="7.0"
export MM_CUTOFF="3.0"
export INCLUDE_MM=#"false" #"true"          # set to "true" or leave empty
export SKIP_ML_DIMERS="false"     # set to "true" or leave empty
export DEBUG="false"
export TEMPERATURE="0.0"
export TIMESTEP="0.1" # 0.1 ps (i.e. 100 fs)
export NUM_STEPS="5_0000" # jax md
export OUTPUT_PREFIX="test_run"
export NSTEPS="5_0000" #ase

# ----------- Build the command line -----------
CMD="${PY}/run_sim.py \
  --pdbfile ${PDBFILE} \
  --ensemble nve
  --cell ${L} \
  --checkpoint ${CHECKPOINT} \
  --energy-catch ${ENERGY_CATCH} \
  --n-monomers ${N_MONOMERS} \
  --n-atoms-monomer ${N_ATOMS_MONOMER} \
  --ml-cutoff ${ML_CUTOFF} \
  --mm-switch-on ${MM_SWITCH_ON} \
  --mm-cutoff ${MM_CUTOFF} \
  --temperature ${TEMPERATURE} \
  --heating_interval 50000 \
  --timestep ${TIMESTEP} \
  --nsteps_jaxmd ${NUM_STEPS} \
  --output-prefix ${OUTPUT_PREFIX} \
  --nsteps_ase ${NSTEPS}"

# Boolean flags
if [ "$INCLUDE_MM" = "true" ]; then
  CMD="$CMD --include-mm"
fi

if [ "$SKIP_ML_DIMERS" = "true" ]; then
  CMD="$CMD --skip-ml-dimers"
fi

if [ "$DEBUG" = "true" ]; then
  CMD="$CMD --debug"
fi

# ----------- Run -----------
echo "Running: $CMD"
$CMD

