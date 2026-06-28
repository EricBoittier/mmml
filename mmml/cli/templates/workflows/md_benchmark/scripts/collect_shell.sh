#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?workflow dir}"
CSV="${2:?csv}"
MD="${3:?md}"
mkdir -p "$(dirname "$CSV")"
{
  echo "job,status"
  for d in "$ROOT"/results/*/; do
    j="$(basename "$d")"
    if [[ -f "$d/done.txt" ]]; then
      echo "$j,done"
    fi
  done
} > "$CSV"
echo "# Benchmark summary" > "$MD"
echo "See $CSV" >> "$MD"
