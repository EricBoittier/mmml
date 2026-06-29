#!/usr/bin/env bash
# Report whether a MMML CHARMM binary was built with CMake domdec=ON.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHARMM_EXE="${1:-${CHARMM_EXE:-$ROOT/setup/charmm/charmm}}"
CHARMM_HOME="${CHARMM_HOME:-$ROOT/setup/charmm}"

platform_tag() {
  local os arch
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "$os" in
    darwin) echo "darwin-${arch}" ;;
    linux) echo "linux-${arch}" ;;
    *) echo "${os}-${arch}" ;;
  esac
}

BUILD_DIRS=(
  "${CHARMM_BUILD_DIR:-}"
  "${HOME}/.cache/mmml-charmm-build/$(platform_tag)-exec"
  "${HOME}/.cache/mmml-charmm-build/$(platform_tag)"
  "$CHARMM_HOME/build/cmake"
)

echo "CHARMM_EXE: ${CHARMM_EXE}"
if [[ -x "$CHARMM_EXE" ]]; then
  echo "  executable: yes ($(readlink -f "$CHARMM_EXE" 2>/dev/null || echo "$CHARMM_EXE"))"
else
  echo "  executable: NO" >&2
fi

_found=0
for dir in "${BUILD_DIRS[@]}"; do
  [[ -n "$dir" && -f "$dir/CMakeCache.txt" ]] || continue
  _found=1
  echo "CMakeCache: $dir/CMakeCache.txt"
  grep -E '^domdec:|^DOMDEC:|domdec:BOOL' "$dir/CMakeCache.txt" 2>/dev/null || true
done
if [[ "$_found" == 0 ]]; then
  echo "No CMakeCache.txt found under native-exec / lib build dirs." >&2
  echo "Rebuild with: bash scripts/rebuild_charmm_native_exec.sh --clean" >&2
fi

if [[ -x "$CHARMM_EXE" ]] && command -v strings >/dev/null 2>&1; then
  echo "Binary strings (domdec-related, first 8):"
  strings "$CHARMM_EXE" | grep -i domdec | head -8 || echo "  (none)"
fi

cat <<EOF

Runtime check: extraneous "DOMDEC NDIR" after ENER => ENERGY did not accept DOMDec.
Site c47 with domdec compiled shows NDIR= and/or domdec_dr_common messages at np>1.

Docs: continued ENERGY with domdec ndir on the same command (not nbonds + bare energy)
  https://academiccharmm.org/documentation/version/c41b2/domdec
EOF
