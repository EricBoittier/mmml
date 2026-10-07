#!/usr/bin/env bash
# Resolve KARML_PYTHON / KARML_BIN for workflow shells and karml-charmm-mpirun.sh.
# Source this file, then call karml_resolve_env [REPO_ROOT].
set -euo pipefail

karml_resolve_python() {
  local repo_root="${1:?repo root required}"

  if [[ -n "${KARML_PYTHON:-}" && -x "${KARML_PYTHON}" ]]; then
    printf '%s\n' "${KARML_PYTHON}"
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

karml_resolve_bin() {
  local py="${1:?python required}"
  local repo_root="${2:?repo root required}"

  if [[ -n "${KARML_BIN:-}" && -x "${KARML_BIN}" ]]; then
    printf '%s\n' "${KARML_BIN}"
    return 0
  fi

  if [[ -x "${repo_root}/.venv/bin/karml" ]]; then
    printf '%s\n' "${repo_root}/.venv/bin/karml"
    return 0
  fi

  local env_karml
  env_karml="$(dirname "$py")/karml"
  if [[ -x "$env_karml" ]]; then
    printf '%s\n' "$env_karml"
    return 0
  fi

  return 1
}

karml_verify_imports() {
  local py="${1:?python required}"
  "$py" - <<'PY'
import importlib.util
import sys

missing = []
for mod in ("jax", "karml"):
    if importlib.util.find_spec(mod) is None:
        missing.append(mod)
if missing:
    print(
        "resolve_karml_env: missing Python packages: "
        + ", ".join(missing)
        + f" (interpreter={sys.executable})",
        file=sys.stderr,
    )
    print(
        "Activate your karml env, run 'uv sync --extra gpu', or set "
        "KARML_PYTHON to a JAX-capable interpreter.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

karml_resolve_uv() {
  if [[ -n "${KARML_UV:-}" && -x "${KARML_UV}" ]]; then
    printf '%s\n' "${KARML_UV}"
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

karml_resolve_env() {
  local repo_root="${1:?repo root required}"
  local py uv_bin uv_dir

  if ! py="$(karml_resolve_python "$repo_root")"; then
    echo "resolve_karml_env: no Python interpreter found." >&2
    echo "Set KARML_PYTHON or create ${repo_root}/.venv (uv sync --extra gpu)." >&2
    return 1
  fi

  karml_verify_imports "$py"

  export KARML_PYTHON="$py"
  if karml_resolve_bin "$py" "$repo_root" >/dev/null 2>&1; then
    export KARML_BIN="$(karml_resolve_bin "$py" "$repo_root")"
  else
    unset KARML_BIN
  fi

  if uv_bin="$(karml_resolve_uv)"; then
    export KARML_UV="$uv_bin"
    uv_dir="$(dirname "$uv_bin")"
    case ":${PATH}:" in
      *":${uv_dir}:"*) ;;
      *) export PATH="${uv_dir}:${PATH}" ;;
    esac
  else
    unset KARML_UV
  fi
  return 0
}
