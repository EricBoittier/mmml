#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=gpu08,gpu09,gpu10
# Job name / logs set by submit_all via --job-name and -o/-e overrides.

set -euo pipefail
# Slurm copies this script to /var/spool/slurm/job*/slurm_script — never derive
# ROOT from BASH_SOURCE under the batch allocation.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${MMML_ROOT:-}" && -d "${MMML_ROOT}" ]]; then
  ROOT="$(cd "${MMML_ROOT}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "$ROOT"
echo "ROOT=$ROOT (submit_dir=${SLURM_SUBMIT_DIR:-})"

TAG="${CAMPAIGN_TAG:?}"
BOX_DIR="${CAMPAIGN_BOX_DIR:?}"
BOX_A="${CAMPAIGN_BOX_A:?}"
ENSEMBLE="${CAMPAIGN_ENSEMBLE:?}"
PS="${CAMPAIGN_PS:?}"
DT_FS="${CAMPAIGN_DT_FS:?}"
X64="${CAMPAIGN_X64:?}"
SEED="${CAMPAIGN_SEED:?}"

echo "SLURM host=$(hostname) job=$SLURM_JOB_ID tag=$TAG $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true

# Wait up to 6h for certified denser box (model.psf + model.crd, or mini.*).
deadline=$((SECONDS + 21600))
ready() {
  [[ -f "${BOX_DIR}/box.json" && -f "${BOX_DIR}/model.psf" && -f "${BOX_DIR}/model.crd" ]] \
    || [[ -f "${BOX_DIR}/mini.psf" && -f "${BOX_DIR}/mini.crd" ]]
}
while ! ready; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: timed out waiting for certified box in $BOX_DIR"
    ls -la "$BOX_DIR" || true
    exit 3
  fi
  echo "$(date -Is) waiting for box.json + model.psf/crd in $BOX_DIR ..."
  sleep 60
done
echo "Box ready: $(ls -1 "${BOX_DIR}"/box.json "${BOX_DIR}"/model.* "${BOX_DIR}"/mini.* 2>/dev/null | tr '\n' ' ')"

bash "${ROOT}/scripts/slurm/dense_dt_campaign/run_one.sh" \
  "$TAG" "$BOX_DIR" "$BOX_A" "$ENSEMBLE" "$PS" "$DT_FS" "$X64" "$SEED"
