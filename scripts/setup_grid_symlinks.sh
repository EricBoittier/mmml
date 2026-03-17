#!/usr/bin/env bash
# Replace duplicate grids_esp*.npz with symlinks to preclassified_data (canonical source).
# Run from repo root. Saves ~800 MB LFS storage by removing duplicate copies.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PRECLASSIFIED="examples/other/co2/preclassified_data"
DCMNET_TRAIN="examples/other/co2/dcmnet_train"
DCMNET_PHYSNET="examples/other/co2/dcmnet_physnet_train"

for dir in "$DCMNET_TRAIN" "$DCMNET_PHYSNET"; do
  for f in grids_esp_train.npz grids_esp_valid.npz grids_esp_test.npz; do
    path="$dir/$f"
    if [ -e "$path" ] && [ ! -L "$path" ]; then
      echo "Replacing $path with symlink to ../preclassified_data/$f"
      git rm --cached "$path" 2>/dev/null || true
      rm -f "$path"
      ln -s "../preclassified_data/$f" "$path"
      git add "$path"
    fi
  done
done

echo "Done. Commit with: git commit -m 'Replace duplicate grids_esp with symlinks to preclassified_data'"
