#!/usr/bin/env bash
# Step 3 — long-range electrostatics variants on one liquid cell.
#
# Density sweeps are exactly where the Coulomb treatment shows up: at 1.00×ρ the
# truncated-MIC and full-Ewald energies diverge much more than at 0.50×ρ, so run
# this across all three fractions before trusting any of them.
#
# `--lr-solver` options:
#   mic               truncated minimum-image Coulomb inside the switched MM
#                     pair loop. Default. Cheapest, and the only one that needs
#                     no extra library — but it truncates.
#   ewald             full-box hybrid Ewald on top of jax_mic. Pure JAX, no PME
#                     library. Same operator as `train --lr-solver ewald`, so
#                     use it with Ewald-trained models.
#   jax_pme           jax_mic k-space + switched short range; pick the k-space
#                     method with --jax-pme-method {ewald,pme,p3m}.
#   nvalchemiops_pme  external PME. REQUIRES --mm-nonbond-mode periodic_external.
#   scafacos          ScaFaCoS solvers. REQUIRES --mm-nonbond-mode
#                     periodic_external; method via --scafacos-method.
#
# Note the `periodic_external` requirement: those two solvers replace the JAX MM
# pair loop with an external Coulomb + CHARMM IMAGE VDW, so they only work under
# a `pbc_*` setup.
#
#   ./03_electrostatics.sh
#   DRY_RUN=1 ./03_electrostatics.sh
#   SOLVENTS=DCM FRACTIONS="0.50 1.00" ./03_electrostatics.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_checkpoint

banner "Electrostatics sweep (L=${BOX_SIZE} Å)"

# Common arguments for every variant below.
md_common() {  # solvent n outdir
  printf '%s' "--setup pbc_nvt --composition $1:$2 --box-size $BOX_SIZE \
--checkpoint $MMML_CKPT --temperature $TEMPERATURE --dt-fs $DT_FS \
--ps $PS_PROD --output-dir $3"
}

for solvent in $SOLVENTS; do
  for frac in $FRACTIONS; do
    n="$(n_monomers_for "$solvent" "$frac")"
    tag="$(cell_tag "$solvent" "$frac")"
    base="${OUT_ROOT}/electrostatics/${tag}"

    echo
    echo "########## ${tag} (${solvent}:${n}) ##########"

    # 1. Truncated MIC — the baseline everything else is compared against.
    echo "--- lr-solver=mic (baseline) ---"
    run_cmd mmml md-system --backend jaxmd \
      $(md_common "$solvent" "$n" "${base}/mic") \
      --lr-solver mic

    # 2. Full-box Ewald, pure JAX. Correct choice for Ewald-trained models.
    echo "--- lr-solver=ewald (full-box, JAX) ---"
    run_cmd mmml md-system --backend jaxmd \
      $(md_common "$solvent" "$n" "${base}/ewald") \
      --lr-solver ewald

    # 3. Ewald compatibility operator for models trained *without* Ewald:
    #    cross-monomer Ewald only, omitting intramolecular + Gaussian self.
    #    Use this when the checkpoint was trained under MIC.
    echo "--- lr-solver=ewald --ewald-omit-self (MIC-trained models) ---"
    run_cmd mmml md-system --backend jaxmd \
      $(md_common "$solvent" "$n" "${base}/ewald_omit_self") \
      --lr-solver ewald --ewald-omit-self

    # 4. jax-pme k-space. Three methods; p3m is usually the best accuracy/cost.
    for method in ewald pme p3m; do
      echo "--- lr-solver=jax_pme --jax-pme-method ${method} ---"
      run_cmd mmml md-system --backend jaxmd \
        $(md_common "$solvent" "$n" "${base}/jax_pme_${method}") \
        --lr-solver jax_pme --jax-pme-method "$method" \
        --jax-pme-sr-cutoff 6.0
    done

    # 5. Coulomb-only long range (drop the reciprocal r^-6 LJ dispersion) —
    #    isolates the electrostatic contribution from LJ tail effects.
    echo "--- jax_pme, Coulomb-only (no reciprocal dispersion) ---"
    run_cmd mmml md-system --backend jaxmd \
      $(md_common "$solvent" "$n" "${base}/jax_pme_nodisp") \
      --lr-solver jax_pme --jax-pme-method p3m --no-jax-pme-dispersion

    # 6. External solvers. These need periodic_external, i.e. external Coulomb
    #    plus CHARMM IMAGE VDW, and therefore the pycharmm backend.
    echo "--- lr-solver=nvalchemiops_pme (periodic_external) ---"
    run_cmd "$MPIRUN_WRAPPER" md-system --backend pycharmm \
      $(md_common "$solvent" "$n" "${base}/nvalchemiops_pme") \
      --lr-solver nvalchemiops_pme --mm-nonbond-mode periodic_external

    echo "--- lr-solver=scafacos --scafacos-method ewald (periodic_external) ---"
    run_cmd "$MPIRUN_WRAPPER" md-system --backend pycharmm \
      $(md_common "$solvent" "$n" "${base}/scafacos_ewald") \
      --lr-solver scafacos --scafacos-method ewald \
      --mm-nonbond-mode periodic_external
  done
done

cat <<'EOF'

Reading the results
-------------------
* Compare total / Coulomb energies of the SAME starting configuration across
  solvers first — that isolates the operator from any dynamics differences.
* mic vs ewald should diverge more at 1.00×ρ than at 0.50×ρ. If they agree
  everywhere, the box is probably too dilute to be testing long range at all.
* Match the operator to the model: use plain `--lr-solver ewald` for
  Ewald-trained checkpoints and `--ewald-omit-self` for MIC-trained ones.
  Mixing them silently biases the energies.
* An NVE run (`--setup pbc_nve`) per solver is the sharpest correctness probe:
  a solver/cutoff mismatch shows up as energy drift.
EOF
