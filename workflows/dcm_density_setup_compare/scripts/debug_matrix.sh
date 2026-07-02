#!/usr/bin/env bash
# Matrix-wide health report (pc-studix login node).
#
# Usage (from ~/mmml/workflows/dcm_density_setup_compare):
#   bash scripts/debug_matrix.sh
#   bash scripts/debug_matrix.sh --failed-only
#   bash scripts/debug_matrix.sh --grep 'post-overlap-rescue'
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"
# shellcheck source=debug_lib.sh
source "$WORKFLOW_ROOT/scripts/debug_lib.sh"
debug_bootstrap_cluster

ROOT="$(debug_artifact_root)"
FAILED_ONLY=false
EXTRA_GREP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --failed-only)
      FAILED_ONLY=true
      shift
      ;;
    --grep)
      EXTRA_GREP="${2:?--grep requires a pattern}"
      shift 2
      ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: artifact root missing: $ROOT" >&2
  exit 1
fi

echo "=== dcm_density_setup_compare matrix debug ==="
echo "host:          $(hostname)"
echo "artifact_root: $ROOT"
echo

echo "=== Your Slurm queue (mmml / setup_compare) ==="
debug_user_gpu_queue || true
echo

echo "=== Cells without done.txt (running or failed) ==="
missing=0
for d in "$ROOT"/*/; do
  [[ -d "$d" ]] || continue
  tag="$(basename "$d")"
  if [[ ! -s "$d/done.txt" ]]; then
    echo "  $tag"
    missing=$((missing + 1))
  fi
done
if [[ "$missing" -eq 0 ]]; then
  echo "  (none — all cells have done.txt)"
fi
echo "pending/failed count: $missing"
echo

if [[ -n "$EXTRA_GREP" ]]; then
  echo "=== Cells matching: $EXTRA_GREP ==="
  grep -rlE "$EXTRA_GREP" "$ROOT"/*/stdout.log 2>/dev/null \
    | sed "s|$ROOT/||; s|/stdout.log||" \
    | sort || echo "  (no matches)"
  echo
fi

echo "=== Hard errors across matrix ==="
grep -rlE 'pycharmm_mlpot: error:|post-overlap-rescue hybrid GRMS|Pre-dynamics GRMS [0-9]+\.[0-9]+ kcal/mol/Å >' \
  "$ROOT"/*/stdout.log 2>/dev/null \
  | sed "s|$ROOT/||; s|/stdout.log||" \
  | sort || echo "  (no matches)"
echo

echo "=== Heat / overlap failures ==="
grep -rlE 'heat segment.*overlap detected|post-overlap-rescue hybrid GRMS' \
  "$ROOT"/*/stdout.log 2>/dev/null \
  | sed "s|$ROOT/||; s|/stdout.log||" \
  | sort || echo "  (no matches)"
echo

echo "=== MPI BLOCK / hang signatures ==="
grep -rlE 'apply_bonded_mm_only_block|bonded-MM-mini: skipping' \
  "$ROOT"/*/stdout.log 2>/dev/null \
  | sed "s|$ROOT/||; s|/stdout.log||" \
  | sort || echo "  (no matches)"
echo

echo "=== Health report (tag | done | mini GRMS | abort) ==="
printf '%-45s %4s  %-14s  %s\n' "TAG" "DONE" "MINI_GRMS" "ABORT"
while IFS= read -r -d '' log; do
  tag="$(basename "$(dirname "$log")")"
  done="$(debug_cell_done "$tag")"
  grms="$(debug_extract_mini_grms "$log")"
  abort="$(debug_extract_abort "$log")"
  if $FAILED_ONLY && [[ "$done" == "OK" && -z "$abort" ]]; then
    continue
  fi
  printf '%-45s %4s  %-14s  %s\n' \
    "$tag" "$done" "${grms:---}" "${abort:-}"
done < <(find "$ROOT" -mindepth 2 -maxdepth 2 -name stdout.log -print0 2>/dev/null | sort -z)
