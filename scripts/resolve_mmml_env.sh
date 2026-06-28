#!/usr/bin/env bash
# Resolve MMML_PYTHON / MMML_BIN for workflow shells and mmml-charmm-mpirun.sh.
# Source this file, then call mmml_resolve_env [REPO_ROOT].
set -euo pipefail

mmml_resolve_python() {
  local repo_root="${1:?repo root required}"

  if [[ -n "${MMML_PYTHON:-}" && -x "${MMML_PYTHON}" ]]; then
    printf '%s\n' "${MMML_PYTHON}"
    return 0
  fi

  if [[ -x "${repo_root}/.venv/bin/python" ]]; then
    printf '%s\n' "${repo_root}/.venv/bin/python"
    return 0
  fi

  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    printf '%s\n' "${CONDA_PREFIX}/bin/python"
    return 0
  fi

  local py3
  py3="$(command -v python3 2>/dev/null || true)"
  if [[ -n "$py3" && -x "$py3" ]]; then
    printf '%s\n' "$py3"
    return 0
  fi

  return 1
}

mmml_resolve_bin() {
  local py="${1:?python required}"
  local repo_root="${2:?repo root required}"

  if [[ -n "${MMML_BIN:-}" && -x "${MMML_BIN}" ]]; then
    printf '%s\n' "${MMML_BIN}"
    return 0
  fi

  if [[ -x "${repo_root}/.venv/bin/mmml" ]]; then
    printf '%s\n' "${repo_root}/.venv/bin/mmml"
    return 0
  fi

  local env_mmml
  env_mmml="$(dirname "$py")/mmml"
  if [[ -x "$env_mmml" ]]; then
    printf '%s\n' "$env_mmml"
    return 0
  fi

  return 1
}

mmml_verify_imports() {
  local py="${1:?python required}"
  "$py" - <<'PY'
import importlib.util
import sys

missing = []
for mod in ("jax", "mmml"):
    if importlib.util.find_spec(mod) is None:
        missing.append(mod)
if missing:
    print(
        "resolve_mmml_env: missing Python packages: "
        + ", ".join(missing)
        + f" (interpreter={sys.executable})",
        file=sys.stderr,
    )
    print(
        "Activate your mmml env, run 'uv sync --extra gpu', or set "
        "MMML_PYTHON to a JAX-capable interpreter.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

mmml_resolve_uv() {
  if [[ -n "${MMML_UV:-}" && -x "${MMML_UV}" ]]; then
    printf '%s\n' "${MMML_UV}"
    return 0
  fi

  local candidate home
  home="${HOME:-}"
  for candidate in \
    "${home}/.local/bin/uv" \
    "${home}/.cargo/bin/uv"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  local uv_bin
  uv_bin="$(command -v uv 2>/dev/null || true)"
  if [[ -n "$uv_bin" && -x "$uv_bin" ]]; then
    printf '%s\n' "$uv_bin"
    return 0
  fi

  return 1
}

mmml_resolve_env() {
  local repo_root="${1:?repo root required}"
  local py uv_bin uv_dir

  if ! py="$(mmml_resolve_python "$repo_root")"; then
    echo "resolve_mmml_env: no Python interpreter found." >&2
    echo "Set MMML_PYTHON or create ${repo_root}/.venv (uv sync --extra gpu)." >&2
    return 1
  fi

  mmml_verify_imports "$py"

  export MMML_PYTHON="$py"
  if mmml_resolve_bin "$py" "$repo_root" >/dev/null 2>&1; then
    export MMML_BIN="$(mmml_resolve_bin "$py" "$repo_root")"
  else
    unset MMML_BIN
  fi

  if uv_bin="$(mmml_resolve_uv)"; then
    export MMML_UV="$uv_bin"
    uv_dir="$(dirname "$uv_bin")"
    case ":${PATH}:" in
      *":${uv_dir}:"*) ;;
      *) export PATH="${uv_dir}:${PATH}" ;;
    esac
  else
    unset MMML_UV
  fi
  return 0
}
