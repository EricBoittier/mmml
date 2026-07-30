#!/usr/bin/env bash
# Step 1 — MM-only box construction + certification for the whole matrix.
#
# `mmml liquid-box` packs with Packmol, MC-equalizes density, then relaxes with
# CHARMM SD/ABNR. It writes model.psf / model.crd / box.json / REPORT.md and is
# pure MM, so it needs no checkpoint and no GPU. Run it first: if a cell cannot
# be built at MM level it will certainly not run with ML/MM on top.
#
#   ./01_build_boxes.sh                 # DCM+ACO × 0.50/0.75/1.00 at L=28
#   DRY_RUN=1 ./01_build_boxes.sh       # show the commands
#   BOX_SIZE=32 SOLVENTS=DCM ./01_build_boxes.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

banner "Building liquid boxes (L=${BOX_SIZE} Å, MM only)"

for solvent in $SOLVENTS; do
  for frac in $FRACTIONS; do
    n="$(n_monomers_for "$solvent" "$frac")"
    tag="$(cell_tag "$solvent" "$frac")"
    out="${OUT_ROOT}/boxes/${tag}"
    rho="$(awk -v r="${RHO_BULK[$solvent]}" -v f="$frac" 'BEGIN{printf "%.4f", r*f}')"

    echo
    echo "--- ${solvent}:${n}  ρ=${rho} g/cm³ (${frac}×bulk)  L=${BOX_SIZE} Å ---"
    run_cmd mmml liquid-box \
      --composition "${solvent}:${n}" \
      --box-size "$BOX_SIZE" \
      --target-density-g-cm3 "$rho" \
      --temperature "$TEMPERATURE" \
      --dt-fs "$DT_FS" \
      --packmol-tolerance 1.5 \
      --output-dir "$out"
  done
done

cat <<'EOF'

Built boxes land in artifacts/boxes/<tag>/ (model.psf, model.crd, box.json,
REPORT.md). Check REPORT.md for the achieved density and minimisation health
before spending GPU time on the ML/MM runs.

Equivalent parametric form (mmml derives N from the known bulk density instead
of the table in common.sh):

  mmml liquid-box --composition DCM:1 --box-auto count --box-size 28 \
    --bulk-density-fraction 0.75 -o artifacts/boxes/dcm_f075_l28
EOF
