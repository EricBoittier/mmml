#!/usr/bin/env bash
#
# slim_repo_history.sh — shrink the mmml git history and reclaim Git LFS quota.
#
# WHY: `git clone` is slow because large, regenerable binaries were committed
# over time. They live in *history*, so deleting them from the working tree does
# not shrink a clone — the history must be rewritten. The same rewrite drops the
# old Git LFS pointer commits so the LFS quota can be reclaimed.
#
# Uncompressed history by category (snapshot; run `--analyze` for current numbers):
#   checkpoints (*.ocdbt, epoch-*) ~1.8 GB   notebooks (*.ipynb)   ~0.7 GB
#   *.pov-state                    ~0.5 GB   MD traj (*.dcd/.traj) ~0.5 GB
#   EF params JSON                 ~0.5 GB   pdf/gif/png           ~0.2 GB
#   core dumps / logs / node_modules ~0.25 GB                      + misc data
#
# THIS SCRIPT IS DESTRUCTIVE. It rewrites every commit hash and REQUIRES a
# force-push that all collaborators must then re-clone. It NEVER force-pushes for
# you — it stops after the local rewrite and prints the exact next steps.
#
# Usage:
#   scripts/slim_repo_history.sh --analyze       # report only, no changes
#   scripts/slim_repo_history.sh --rewrite       # rewrite history locally (asks to confirm)
#
# Requirements: git-filter-repo  (pip install git-filter-repo)
#
set -euo pipefail

MODE="${1:---analyze}"

if ! command -v git-filter-repo >/dev/null 2>&1 && ! python3 -c "import git_filter_repo" >/dev/null 2>&1; then
  echo "ERROR: git-filter-repo is not installed. Install it with:" >&2
  echo "    pip install git-filter-repo        # or: uv tool install git-filter-repo" >&2
  exit 1
fi
FILTER_REPO=(git filter-repo)
command -v git-filter-repo >/dev/null 2>&1 || FILTER_REPO=(python3 -m git_filter_repo)

# Paths/globs to PURGE from all history. These are regenerable build artifacts,
# logs, render dumps, MD trajectories, core dumps, and training checkpoints.
# Review/adjust before running. (charmm.tar.xz is intentionally kept — it is the
# CHARMM source needed by setup/install.sh; host it externally if you want it gone.)
PURGE_GLOBS=(
  'mmml/gui/viewer/node_modules/**'
  '*.pov-state'
  '*.pov-state.gz'
  '*.dcd'
  '*.traj'
  '*.ocdbt'
  '*.out'
  '*.err'
  '*.gif'
  'core.*'
)
# AGGRESSIVE (uncomment after review — these may be referenced by notebooks/tests).
# Adding these took a test rewrite from 3.8 GB -> ~0.95 GB; pruning the items below
# the line further approaches a few hundred MB:
#   'mmml/models/EF/data/**'      # large model param JSONs
#   'tests/EF/**'                 # 18-23 MB test-fixture params + arrays
#   '**/epoch-*/**'               # orbax training checkpoints
#   '*.ipynb'                     # prefer `nbstripout` to keep notebooks, drop outputs
#   '*.pdf' '*.parquet' '*.npz' '*.h5' '*.hdf5' '*.xyz'
#   'docs/beamer_slides/**'
#   'hdf5-1.14.6/**'              # a full vendored HDF5 source tree got committed
#   'testdata/DYNA1' 'testdata/PRESS1'
# NOTE: keep setup/charmm.tar.xz (CHARMM source used by setup/install.sh) unless you
# move it to a release asset / external host and update the install instructions.

case "$MODE" in
  --analyze)
    echo "== git-filter-repo --analyze (writes report to .git/filter-repo/analysis) =="
    "${FILTER_REPO[@]}" --analyze
    echo
    echo "Top reports:"
    echo "  .git/filter-repo/analysis/path-all-sizes.txt"
    echo "  .git/filter-repo/analysis/blob-shas-and-paths.txt"
    ;;
  --rewrite)
    echo "This will REWRITE ALL HISTORY in the current repo (every commit hash changes)."
    echo "Globs to purge:"; printf '  %s\n' "${PURGE_GLOBS[@]}"
    read -r -p "Type 'rewrite' to continue: " ans
    [ "$ans" = "rewrite" ] || { echo "aborted"; exit 1; }

    backup="../mmml-backup-$(date +%Y%m%d-%H%M%S).git"
    echo "Creating mirror backup at: $backup"
    git clone --mirror . "$backup"

    args=()
    for g in "${PURGE_GLOBS[@]}"; do args+=(--path-glob "$g"); done
    "${FILTER_REPO[@]}" --invert-paths "${args[@]}"

    echo
    echo "Local history rewritten. Repacking…"
    git reflog expire --expire=now --all || true
    git gc --prune=now --aggressive || true
    du -sh .git || true

    cat <<'NEXT'

NEXT STEPS (review, then run manually):
  1. Inspect the rewritten tree and run the test suite.
  2. Re-add the remote (filter-repo removes it):
       git remote add origin <REPO_URL>
  3. Force-push the rewritten history (coordinate with collaborators first!):
       git push --force --all origin
       git push --force --tags origin
  4. Tell every collaborator to re-clone (old clones cannot fast-forward).
  5. Reclaim Git LFS storage: GitHub does not garbage-collect LFS objects
     automatically. After the rewrite, either delete + recreate the repo, or
     contact GitHub Support / use the LFS admin API to purge unreferenced objects.
     Verify current LFS objects with:  git lfs ls-files
NEXT
    ;;
  *)
    echo "Usage: $0 [--analyze|--rewrite]" >&2
    exit 2
    ;;
esac
