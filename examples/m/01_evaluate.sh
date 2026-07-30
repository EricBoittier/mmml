#!/usr/bin/env bash
# Evaluate the examples/m PhysNet checkpoint on NH3–CH3Cl dimers.
#
# Energies in the NPZ are absolute QM totals; the checkpoint predicts a
# mean-centered / interaction-like scale. Default eval uses --subtract-mean
# so energy MAE is comparable; forces/dipoles are absolute and usually the
# better quality signal for this checkpoint.
#
# Checkpoint: MMML_CKPT (default examples/m/model_ext.json from _env.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"
mmml_example_env_banner

OUT="${ARTIFACTS_DIR}/evaluate"
DIMER_NPZ="${ARTIFACTS_DIR}/dimer_only.npz"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "=== Prepare dimer-only NPZ (N=9) ==="
uv run python examples/m/00_prepare_eval_npz.py --data "${MMML_DATA}" -o "${DIMER_NPZ}"

echo "=== physnet-evaluate (${MMML_CKPT} × dimers, --subtract-mean) ==="
uv run mmml physnet-evaluate \
  --checkpoint "${MMML_CKPT}" \
  --data "${DIMER_NPZ}" \
  --natoms 9 \
  --batch-size "${BATCH_SIZE}" \
  --num-samples "${NUM_SAMPLES}" \
  --subtract-mean \
  --plots \
  -o "${OUT}"

test -f "${OUT}/metrics.json"
# Record how energies were referenced for the docs report.
uv run python - <<PY
import json
from pathlib import Path
p = Path("${OUT}/metrics.json")
m = json.loads(p.read_text())
m["energy_reference"] = "subtract_mean"
m["data_subset"] = "N=9 dimers"
m["source_npz"] = "${MMML_DATA}"
m["checkpoint"] = "${MMML_CKPT}"
p.write_text(json.dumps(m, indent=2))
print("Updated", p)
PY

echo "PASS: evaluate -> ${OUT}"
