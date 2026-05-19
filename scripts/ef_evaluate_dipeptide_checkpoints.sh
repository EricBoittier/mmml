#!/usr/bin/env bash
# Evaluate four dipeptide EF checkpoints on the test split (same NPZ as ef-train --test-npz).
#
# Matches checkpoint layout from training logs:
#   0ptFbothNoPert  -> dipeptide_ef2_ptF_run0
#   0ptTbothPert    -> dipetidePert_ef2_ptT_run0
#   1ptFbothNoPert  -> dipeptide_ef2_ptF_run1
#   1ptTbothPert    -> dipetidePert_ef2_ptT_run1
#
# Usage (from repo root or any cwd):
#   ./scripts/ef_evaluate_dipeptide_checkpoints.sh
#   TEST_NPZ=/path/to/test.npz OUT_BASE=./ef_eval ./scripts/ef_evaluate_dipeptide_checkpoints.sh
#   ./scripts/ef_evaluate_dipeptide_checkpoints.sh --rot-augment --rot-perturbation 1.0
#
# Environment:
#   TEST_NPZ     Test split NPZ (default: out/splits01/energies_forces_dipoles_test.npz)
#   OUT_BASE     Output root (default: ef_eval)
#   BATCH_SIZE   Evaluation batch size (default: 64)
#   CKPT_ROOT    Override checkpoint base (default: /mmhome/boittier/home/ckpts)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEST_NPZ="${TEST_NPZ:-${REPO_ROOT}/out/splits01/energies_forces_dipoles_test.npz}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/ef_eval}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CKPT_ROOT="${CKPT_ROOT:-/mmhome/boittier/home/ckpts}"

if [[ ! -f "${TEST_NPZ}" ]]; then
  echo "Error: test NPZ not found: ${TEST_NPZ}" >&2
  echo "Set TEST_NPZ to your splits test file (ef-train --test-npz)." >&2
  exit 1
fi

mkdir -p "${OUT_BASE}"

# name|ckpt_dir|config_basename|params_basename
RUNS=(
  "0ptFbothNoPert|${CKPT_ROOT}/dipeptide_ef2_ptF_run0|config-d0e9ce01-6fcd-4524-aa98-54861c901ec5.json|params-d0e9ce01-6fcd-4524-aa98-54861c901ec5.json"
  "0ptTbothPert|${CKPT_ROOT}/dipetidePert_ef2_ptT_run0|config-85fb458c-b63f-4b72-a102-557683f9f11f.json|params-85fb458c-b63f-4b72-a102-557683f9f11f.json"
  "1ptFbothNoPert|${CKPT_ROOT}/dipeptide_ef2_ptF_run1|config-519bc841-31c3-452c-a2ff-bab82432437e.json|params-519bc841-31c3-452c-a2ff-bab82432437e.json"
  "1ptTbothPert|${CKPT_ROOT}/dipetidePert_ef2_ptT_run1|config-bd8b7a90-13fa-4943-9453-6b3ef7bf4433.json|params-bd8b7a90-13fa-4943-9453-6b3ef7bf4433.json"
)

EXTRA_ARGS=("$@")

echo "Test NPZ:  ${TEST_NPZ}"
echo "Out base:  ${OUT_BASE}"
echo "Batch:     ${BATCH_SIZE}"
echo "Extra args:${EXTRA_ARGS[*]:-<none>}"
echo

for entry in "${RUNS[@]}"; do
  IFS='|' read -r run_name ckpt_dir config_file params_file <<< "${entry}"
  config_path="${ckpt_dir}/${config_file}"
  params_path="${ckpt_dir}/${params_file}"
  out_dir="${OUT_BASE}/${run_name}"

  if [[ ! -f "${params_path}" ]]; then
    echo "Error: params not found: ${params_path}" >&2
    exit 1
  fi
  if [[ ! -f "${config_path}" ]]; then
    echo "Error: config not found: ${config_path}" >&2
    exit 1
  fi

  echo "============================================================"
  echo "Run:    ${run_name}"
  echo "Config: ${config_path}"
  echo "Params: ${params_path}"
  echo "Output: ${out_dir}"
  echo "============================================================"

  mmml ef-evaluate \
    --params "${params_path}" \
    --config "${config_path}" \
    --test-npz "${TEST_NPZ}" \
    --output-dir "${out_dir}" \
    --batch-size "${BATCH_SIZE}" \
    --save-output-npz \
    "${EXTRA_ARGS[@]}"

  echo
done

echo "All evaluations written under: ${OUT_BASE}"
echo "  metrics.json + plots + evaluation_output.npz per run subdirectory"
