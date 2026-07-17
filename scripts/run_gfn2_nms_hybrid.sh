#!/usr/bin/env bash
# Train the hybrid ML/MM model on the GFN2 normal-mode dimer-scan dataset,
# then GATE it on the orientation scan -- not on validation MAE.
#
# Why the gate: validation MAE cannot see the failure that matters. Model
# 7721fa95 scored a 0.053 eV valid energy MAE while being wrong by 1.4 eV
# (33 kcal/mol vs GFN2) on close approaches, because the old data's dimers were
# overwhelmingly separated (median contact 7.71 A) and the metric never visits
# the region where MD falls over. The orientation scan does.
#
# Run on a GPU node:  bash scripts/run_gfn2_nms_hybrid.sh

set -euo pipefail

ACODCM=/mmhome/boittier/home/mmml_tutorial/acodcm
MMML=/mmhome/boittier/home/mmml
PY="$MMML/.venv/bin/python"
CFG="${1:-$ACODCM/gfn2_nms_hybrid.yaml}"

cd "$ACODCM"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

echo "=== training: $CFG ==="
#"$MMML/.venv/bin/mmml" physnet-train --config "$CFG"

CKPT=$(ls -dt "$ACODCM"/ckpts/gfn2_nms/gfn2nms-*/ 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "no checkpoint under ckpts/gfn2_nms -- training did not produce one" >&2
  exit 1
fi
echo "=== checkpoint: $CKPT ==="

# Freeze it. Pointed at a live run, the parity/scan tooling resolves get_last()
# at different moments and silently compares different epochs -- that alone once
# produced a spurious 0.168 eV "mismatch".
FROZEN="$ACODCM/ckpts/_gate_frozen"
rm -rf "$FROZEN"; mkdir -p "$FROZEN"
EP=$(ls -d "$CKPT"/epoch-* | sed 's#.*epoch-##' | sort -n | tail -1)
cp -r "$CKPT/epoch-$EP" "$FROZEN/epoch-$EP"
echo "frozen at epoch-$EP"

# --- THE GATE ---------------------------------------------------------------
# Spurious minima across S^2 x SO(3), at kT (0.3 kcal/mol at 150 K) -- a
# sub-thermal threshold counts ripple the dynamics cannot resolve and inverts
# the verdict between models.
for RES in DCM ACO; do
  echo "=== orientation scan: $RES ==="
  # PIN the monomer source. The scan takes its rigid monomer from --data, and the
  # reference table below was computed from out_combined_dedup. Passing a
  # different npz silently changes the monomer's ORIENTATION (same molecule, rmsd
  # 0.248 A unaligned), which rotates the whole Fibonacci direction set and makes
  # the numbers incomparable. This bit me: a gate run against gfn2_nms_test gave
  # ACO 8.3%/mean +10.65 vs 4.6%/-1.05 here, for the same checkpoint.
  # --use-ema: live params swing epoch-to-epoch in this scan's unconstrained
  # extrapolation region (measured: DCM spurious fraction 12.5% -> 8.3% between
  # adjacent epochs on live params, flat at 4.2% on ema_params, same checkpoints).
  # Gate on ema_params for a reproducible verdict.
  "$PY" "$MMML/scripts/scan_dimer_orientations.py" \
      --checkpoint "$FROZEN" \
      --data "$ACODCM/out_combined_dedup/energies_forces_dipoles_test.npz" \
      --resid "$RES" --n-directions 10 --n-orientations 24 --n-r 36 \
      --mm-switch-on 6.0 --use-ema --out "$ACODCM/gate_${RES}"
done

echo
echo "=== VERDICT ==============================================================="
echo "Compare against the models trained on the OLD (MD-snapshot) data:"
echo "                  ACO spurious   ACO deepest   DCM spurious   DCM deepest"
echo "  940b8905 (6.0)     14.6%         -7.39         15.0%          -2.44"
echo "  96e58a2d (8.0)     34.6%         -3.89         11.7%          -3.43"
echo "  7721fa95 (new)     23.3%        -14.75         50.4%         -33.64  <- broken"
echo "  GFN2 truth          n/a          ~-5           n/a            ~-1.5"
echo
echo "DCM deepest well is the sharpest discriminator: -2.44 (ok) vs -33.6 (broken)."
echo
echo "DO NOT read the spurious FRACTION as quality: it conflates a harmless"
echo "sub-kcal wiggle with an 8x overbound hole. Measured on the GFN2-dense model:"
echo "  DCM 32.9% 'spurious' -> flagged rays -2.25/-2.16 vs xTB -2.85/-2.83  (fine)"
echo "  ACO  4.6% 'spurious' -> deepest ray -10.87 vs xTB -1.28              (a hole)"
echo "The count said DCM regressed 2x and ACO was best; xTB said the opposite."
echo "Rank by |ML - xTB| at the deepest well instead -- that is what MD feels."
echo "If the spurious fraction collapses, dense coverage was the cause and the"
echo "same recipe at MP2 is the production run. If it does not, coverage was not"
echo "the problem and the architecture/loss is next."
echo
echo "Then confirm the deepest wells against an INDEPENDENT potential:"
echo "  $PY $MMML/scripts/validate_dimer_rays.py --checkpoint $FROZEN \\"
echo "      --data $ACODCM/gfn2_nms_test.npz --resid DCM --rays <deepest> \\"
echo "      --mm-switch-on 6.0 --with-xtb --out validate_gate"
