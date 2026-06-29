#!/usr/bin/env bash
# Report whether a MMML CHARMM binary was built with CMake domdec=ON (requires colfft + FFTW/MKL).
#
# CMakeCache.txt stores the REQUESTED value (-Ddomdec=ON) even when CMakeLists.txt
# silently overrides it to OFF due to missing FFTW/MKL. Use `nm` on the binary to check
# the EFFECTIVE state — DOMDEC compiled in iff the binary exports `domdec_com`.
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

# CMakeCache check (informational — may show ON even when effective build is OFF).
echo ""
echo "=== CMakeCache (requested values — may differ from effective build) ==="
_found=0
for dir in "${BUILD_DIRS[@]}"; do
  [[ -n "$dir" && -f "$dir/CMakeCache.txt" ]] || continue
  _found=1
  echo "CMakeCache: $dir/CMakeCache.txt"
  grep -E '^(domdec|colfft):BOOL=' "$dir/CMakeCache.txt" 2>/dev/null || true
done
if [[ "$_found" == 0 ]]; then
  echo "No CMakeCache.txt found under native-exec / lib build dirs."
fi

# Effective DOMDEC check via nm — definitive; CMakeCache can lie when FFTW missing.
echo ""
echo "=== Effective DOMDEC in binary (KEY_DOMDEC + FFTW check) ==="
_domdec_compiled=0
if [[ -x "$CHARMM_EXE" ]] && command -v nm >/dev/null 2>&1; then
  # Use -E (ERE) so '|' is alternation — BRE '\|' is not portable across all grep versions.
  _nm_count="$(nm "$CHARMM_EXE" 2>/dev/null | grep -cE 'domdec_com|domdec_common' || true)"
  echo "  nm domdec_com symbol count: $_nm_count (>0 = domdec source compiled in)"
  if [[ "${_nm_count:-0}" -gt 0 ]]; then
    echo "  domdec symbols present in $CHARMM_EXE"
  else
    echo "  WARN: no domdec_com symbol in $CHARMM_EXE — DOMDEC was compiled OUT" >&2
  fi

  # Check whether KEY_DOMDEC==1 was set in eutil.F90 by looking for a string that
  # is ONLY compiled when #if KEY_DOMDEC==1 is true.
  _key_domdec=0
  if command -v strings >/dev/null 2>&1; then
    if strings "$CHARMM_EXE" 2>/dev/null | grep -qF 'DOMDec cannot be used with FASTer'; then
      echo "  PASS: KEY_DOMDEC==1 confirmed (strings: 'DOMDec cannot be used with FASTer' present)"
      _key_domdec=1
      _domdec_compiled=1
    else
      echo "  FAIL: KEY_DOMDEC==0 — 'DOMDec cannot be used with FASTer' string absent from binary" >&2
      echo "        COLFFT (FFTW/MKL) was not enabled at compile time → eutil.F90 compiled without DOMDEC" >&2
    fi
  fi

  # FFTW symbol count — confirms COLFFT was active (static FFTW linked into binary).
  if command -v nm >/dev/null 2>&1; then
    _fftw_count="$(nm "$CHARMM_EXE" 2>/dev/null | grep -cE 'fftw|FFTW' || true)"
    echo "  nm FFTW symbol count: ${_fftw_count:-0} (>0 = static FFTW/COLFFT linked in)"
    [[ "${_fftw_count:-0}" -gt 0 ]] && _domdec_compiled=1
  fi

  if [[ "$_key_domdec" == 0 ]]; then
    echo "" >&2
    echo "" >&2
    echo "  CMake silently disables DOMDEC when FFTW/MKL is not found (colfft=OFF)." >&2
    echo "  The CMakeCache above may still show domdec=ON (that is the requested value)." >&2
    echo "" >&2
    echo "  Fix — find FFTW on your cluster:" >&2
    echo "    module spider fftw                       # find available FFTW modules" >&2
    echo "    module load FFTW/3.3.10-GCC-12.2.0      # adjust to name above" >&2
    echo "    export FFTW_ROOT=\${EBROOTFFTW}" >&2
    echo "" >&2
    echo "  Or check the library build cache for an already-found FFTW path:" >&2
    _lib_cache="\${HOME}/.cache/mmml-charmm-build/$(platform_tag)/CMakeCache.txt"
    if [[ -f "$HOME/.cache/mmml-charmm-build/$(platform_tag)/CMakeCache.txt" ]]; then
      _inc="$(grep '^FFTW_INCLUDE_DIR:PATH=' "$HOME/.cache/mmml-charmm-build/$(platform_tag)/CMakeCache.txt" 2>/dev/null | cut -d= -f2- || true)"
      if [[ -n "$_inc" ]]; then
        echo "    Library build found FFTW include at: $_inc" >&2
        echo "    → export FFTW_ROOT=$(dirname "$_inc")" >&2
      else
        echo "    grep -i fftw_include $HOME/.cache/mmml-charmm-build/$(platform_tag)/CMakeCache.txt" >&2
      fi
    fi
    echo "" >&2
    echo "  Then rebuild: bash scripts/rebuild_charmm_native_exec.sh --clean" >&2
  fi
elif [[ -x "$CHARMM_EXE" ]]; then
  echo "  (nm not available — falling back to strings)" >&2
  if strings "$CHARMM_EXE" 2>/dev/null | grep -qF 'DOMDec cannot be used with FASTer'; then
    echo "  PASS (strings): KEY_DOMDEC==1 confirmed"
    _domdec_compiled=1
  else
    echo "  FAIL (strings): KEY_DOMDEC==0 — binary was compiled without COLFFT/DOMDEC"
  fi
fi

# FFTW path in library build cache (informational).
_lib_cache="$HOME/.cache/mmml-charmm-build/$(platform_tag)/CMakeCache.txt"
if [[ -f "$_lib_cache" ]]; then
  _fftw_inc="$(grep '^FFTW_INCLUDE_DIR:PATH=' "$_lib_cache" 2>/dev/null | cut -d= -f2- || true)"
  _fftw_lib="$(grep '^FFTW_LIBRARY:FILEPATH=' "$_lib_cache" 2>/dev/null | cut -d= -f2- || true)"
  if [[ -n "$_fftw_inc" || -n "$_fftw_lib" ]]; then
    echo ""
    echo "=== FFTW in library build cache (use this prefix for FFTW_ROOT) ==="
    [[ -n "$_fftw_inc" ]] && echo "  FFTW include: $_fftw_inc"
    [[ -n "$_fftw_lib" ]] && echo "  FFTW lib:     $_fftw_lib"
    if [[ -n "$_fftw_inc" ]]; then
      echo "  → export FFTW_ROOT=$(dirname "$_fftw_inc")"
    fi
  fi
fi

echo ""
if [[ "$_domdec_compiled" == 1 ]]; then
  echo "DOMDEC build: OK — KEY_DOMDEC==1 confirmed (or FFTW symbols present)."
  echo "Run the tier3 smoke:"
  echo "  CHARMM_EXE=$CHARMM_EXE bash scripts/run_domdec_dcm10_smoke.sh tier3"
else
  echo "DOMDEC build: KEY_DOMDEC==0 — COLFFT/FFTW was not linked; rebuild with FFTW (see above)." >&2
  exit 1
fi
