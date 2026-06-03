#!/usr/bin/env bash
# Stop local `git status` noise from uv.lock (file stays on disk; already in .gitignore).
#
# uv.lock is machine-specific after `uv sync --extra gpu`; it was committed historically
# so .gitignore alone did not help. Prefer this per clone, or untrack repo-wide (see below).
#
# Undo: git update-index --no-skip-worktree uv.lock

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository." >&2
  exit 1
fi
if ! git ls-files --error-unmatch uv.lock >/dev/null 2>&1; then
  echo "uv.lock is not tracked; nothing to do (gitignore is enough)." >&2
  exit 0
fi

git update-index --skip-worktree uv.lock
echo "ok: git will ignore local changes to tracked uv.lock on this clone."
echo "     Run: git update-index --no-skip-worktree uv.lock  to undo."
