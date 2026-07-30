#!/usr/bin/env bash
# Step 4 — ML/MM partitioning and hybrid-charge settings.
#
# Two independent axes decide what the hybrid calculator actually evaluates:
#
#   --do-ml / --no-do-ml              ML monomer (intramolecular) terms
#   --do-ml-dimer / --no-do-ml-dimer  ML pair (intermolecular) terms
#   --include-mm / --no-include-mm    switched JAX MM pairs (LJ + MIC Coulomb)
#
# and then, when MM Coulomb is on, *which charges* it uses:
#
#   --mm-charge-mode fixed             q_CGenFF (default)
#   ...              q0    / Q⁰        neutralised unperturbed monomer q_ML
#   ...              latent / q1 / Q¹  AB-perturbed q_ML   ** DIMER ONLY **
#   ...              fixed_plus_latent fixed + latent correction
#   ...              latent_mean       frozen template (--mm-latent-charge-template)
#   ...              latent_dynamic    live weighted mean of Q¹ over active dimers
#
# IMPORTANT for liquids: `latent` / `q1` is dimer-only and is NOT valid for an
# N-monomer box. For liquid DCM / ACO use fixed, q0, latent_mean or
# latent_dynamic — these are defined for any n_monomers.
# See docs/hybrid-mm-charges.md.
#
#   ./04_ml_mm.sh
#   DRY_RUN=1 ./04_ml_mm.sh
#   SOLVENTS=DCM FRACTIONS=1.00 ./04_ml_mm.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_checkpoint

banner "ML/MM settings sweep (L=${BOX_SIZE} Å)"

md_common() {  # solvent n outdir
  printf '%s' "--setup pbc_nvt --composition $1:$2 --box-size $BOX_SIZE \
--checkpoint $MMML_CKPT --temperature $TEMPERATURE --dt-fs $DT_FS \
--ps $PS_PROD --output-dir $3"
}

for solvent in $SOLVENTS; do
  for frac in $FRACTIONS; do
    n="$(n_monomers_for "$solvent" "$frac")"
    tag="$(cell_tag "$solvent" "$frac")"
    base="${OUT_ROOT}/ml_mm/${tag}"

    echo
    echo "########## ${tag} (${solvent}:${n}) ##########"

    # --- A. Partitioning ------------------------------------------------
    # Pure MM reference: no ML at all. The classical baseline.
    echo "--- pure MM (no ML) ---"
    run_cmd mmml md-system --backend jaxmd \
      $(md_common "$solvent" "$n" "${base}/mm_only") \
      --no-do-ml --no-do-ml-dimer --include-mm

    # Pure ML: PhysNet only, no MM pair terms. Cutoff keys are ignored.
    echo "--- pure ML (no MM pairs) ---"
    run_cmd mmml md-system --backend jaxmd \
      $(md_common "$solvent" "$n" "${base}/ml_only") \
      --do-ml --do-ml-dimer --no-include-mm

    # ML monomers + MM intermolecular: mechanical embedding. Intramolecular
    # physics from the model, intermolecular from CGenFF.
    echo "--- ML monomer + MM pairs (mechanical embedding) ---"
    run_cmd mmml md-system --backend jaxmd \
      $(md_common "$solvent" "$n" "${base}/ml_mono_mm_pairs") \
      --do-ml --no-do-ml-dimer --include-mm

    # Full hybrid: ML monomer + ML dimer + switched MM. The production setting.
    echo "--- full hybrid ML/MM ---"
    run_cmd mmml md-system --backend jaxmd \
      $(md_common "$solvent" "$n" "${base}/hybrid_full") \
      --do-ml --do-ml-dimer --include-mm

    # --- B. Hybrid MM charges (liquid-safe modes only) -------------------
    for mode in fixed q0 latent_dynamic; do
      echo "--- mm-charge-mode=${mode} ---"
      run_cmd mmml md-system --backend jaxmd \
        $(md_common "$solvent" "$n" "${base}/charge_${mode}") \
        --do-ml --do-ml-dimer --include-mm \
        --mm-charge-mode "$mode"
    done

    # fixed + latent correction (alias: --mm-charge-correction)
    echo "--- mm-charge-mode=fixed_plus_latent ---"
    run_cmd mmml md-system --backend jaxmd \
      $(md_common "$solvent" "$n" "${base}/charge_fixed_plus_latent") \
      --do-ml --do-ml-dimer --include-mm \
      --mm-charge-mode fixed_plus_latent

    # --- C. Switching / handover between ML and MM ----------------------
    # Where the ML dimer term hands over to classical MM. Too narrow a switch
    # shows up as energy drift in NVE; too wide double-counts.
    echo "--- switching: ml-switch-width / mm-switch-on / mm-switch-width ---"
    run_cmd mmml md-system --backend jaxmd \
      $(md_common "$solvent" "$n" "${base}/switch_tuned") \
      --do-ml --do-ml-dimer --include-mm \
      --ml-switch-width 1.5 --mm-switch-on 8.0 --mm-switch-width 5.0

    # --- D. MM nonbond provider -----------------------------------------
    # jax_mic (default) evaluates MM pairs in JAX; periodic_external hands
    # Coulomb to an external solver and VDW to CHARMM IMAGE.
    echo "--- mm-nonbond-mode=periodic_external (pycharmm) ---"
    run_cmd "$MPIRUN_WRAPPER" md-system --backend pycharmm \
      $(md_common "$solvent" "$n" "${base}/periodic_external") \
      --do-ml --do-ml-dimer --include-mm \
      --mm-nonbond-mode periodic_external

    # Cross-check the MM pair list itself: JAX-built vs CHARMM's idxu/idxv.
    # Energies must match; a mismatch means the neighbour lists disagree.
    echo "--- mm-pair-source=charmm_callback (parity diagnostic) ---"
    run_cmd "$MPIRUN_WRAPPER" md-system --backend pycharmm \
      $(md_common "$solvent" "$n" "${base}/mm_pair_charmm") \
      --do-ml --do-ml-dimer --include-mm \
      --mm-pair-source charmm_callback
  done
done

cat <<'EOF'

Species-aware ownership (mixed boxes)
-------------------------------------
For a *mixed* system (e.g. DCM + ACO, or solute + solvent) an interaction policy
declares which provider owns each monomer/pair, so nothing is double-counted:

  mmml md-system --backend jaxmd --setup pbc_nvt \
    --composition DCM:100,ACO:100 --box-size 32 \
    --checkpoint "$MMML_CKPT" \
    --interaction-policy ./policy.yaml \
    --output-dir artifacts/liquid_density_sweep/mixed_dcm_aco

Start from examples/interaction_policy_single_provider.yaml (one provider owns
everything) or examples/interaction_policy_peptide_water.yaml (solute/solvent
split), and see docs/md-interaction-policies.md. Policy paths resolve relative
to the config file, and multi-provider / near–far policies fail closed rather
than silently double-counting.

Trainable LJ scales
-------------------
If the checkpoint was trained with --learn-mm-lj-scales, the per-CGenFF-type
sigma/epsilon scales live in hybrid_mm.json next to it and load automatically;
override with --mm-lj-scales-file. See docs/hybrid-mm-lj-scales.md.
EOF
