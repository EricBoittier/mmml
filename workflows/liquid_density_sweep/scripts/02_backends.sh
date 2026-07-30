#!/usr/bin/env bash
# Step 2 — the same physical cell through every MD backend.
#
# Backends (`--backend`):
#   ase       ASE integrators driving the hybrid calculator. Easiest to debug,
#             slowest; good for short NVE drift checks.
#   jaxmd     JAX-MD. Fastest for pure-ML / jax_mic MM; PBC via `pbc_*` setups.
#   pycharmm  CHARMM MLpot (staged mini → heat → equi → prod, CHARMM IMAGE/
#             crystal). The production path for large PBC liquids; needs the
#             MPI-linked libcharmm launcher.
#
# The interesting comparison is that all three should agree on the *energy* of
# the same starting configuration; they differ in integrator and in how PBC
# nonbonds are evaluated.
#
#   ./02_backends.sh                    # one cell, three backends
#   DRY_RUN=1 ./02_backends.sh
#   SOLVENTS=DCM FRACTIONS=1.00 ./02_backends.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_checkpoint

banner "Backend comparison (L=${BOX_SIZE} Å, ${PS_PROD} ps)"

for solvent in $SOLVENTS; do
  for frac in $FRACTIONS; do
    n="$(n_monomers_for "$solvent" "$frac")"
    tag="$(cell_tag "$solvent" "$frac")"

    # --- ASE: hybrid calculator + ASE integrator, NVT ---------------------
    echo
    echo "--- ${tag}: backend=ase (pbc_nvt) ---"
    run_cmd mmml md-system \
      --backend ase --setup pbc_nvt \
      --composition "${solvent}:${n}" --box-size "$BOX_SIZE" \
      --checkpoint "$MMML_CKPT" \
      --temperature "$TEMPERATURE" --dt-fs "$DT_FS" --ps "$PS_PROD" \
      --output-dir "${OUT_ROOT}/backends/${tag}/ase"

    # --- JAX-MD: fastest; same setup preset ------------------------------
    echo
    echo "--- ${tag}: backend=jaxmd (pbc_nvt) ---"
    run_cmd mmml md-system \
      --backend jaxmd --setup pbc_nvt \
      --composition "${solvent}:${n}" --box-size "$BOX_SIZE" \
      --checkpoint "$MMML_CKPT" \
      --temperature "$TEMPERATURE" --dt-fs "$DT_FS" --ps "$PS_PROD" \
      --output-dir "${OUT_ROOT}/backends/${tag}/jaxmd"

    # --- PyCHARMM: staged production pipeline ----------------------------
    # Staged mini → heat → equi → prod with CHARMM crystal/IMAGE. Launched via
    # the MPI wrapper because libcharmm is MPI-linked (see `mmml doctor`).
    echo
    echo "--- ${tag}: backend=pycharmm (pbc_npt, staged) ---"
    run_cmd "$MPIRUN_WRAPPER" md-system \
      --backend pycharmm --setup pbc_npt \
      --composition "${solvent}:${n}" --box-size "$BOX_SIZE" \
      --checkpoint "$MMML_CKPT" \
      --temperature "$TEMPERATURE" --dt-fs "$DT_FS" --ps "$PS_PROD" \
      --output-dir "${OUT_ROOT}/backends/${tag}/pycharmm"
  done
done

cat <<'EOF'

Setup presets worth knowing (`--setup`):
  pbc_nve         energy-conservation check (no thermostat) — the sharpest test
                  that cutoffs / switching are consistent
  pbc_nvt         fixed-volume production
  pbc_npt         constant pressure; use this to check the model *reproduces*
                  the target density rather than being held at it
  pbc_thermalize  minimise → ramp to --temperature → optional NVT equilibration
  pycharmm_full   full staged CHARMM pipeline (mini → heat → NVE → equi → prod)

A density sweep run under pbc_npt is the physically meaningful validation: start
at 0.50/0.75/1.00 × ρ_bulk and see whether each relaxes back toward ρ_bulk.
EOF
