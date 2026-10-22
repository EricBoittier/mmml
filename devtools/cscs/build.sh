#!/bin/bash
# Step 1 of build.sbatch, on the compute node host (not in a container): podman build -> .sqsh.
set -euo pipefail
NAME=mmml
C=/iopsstor/scratch/cscs/boittier/containers
DST=$C/$NAME.sqsh
cd /users/boittier/mmml/devtools/cscs

# Rootless podman needs a subuid/subgid range on *this* node (nid007417 had none, 2026-10-02).
grep -q "^$USER:" /etc/subuid && grep -q "^$USER:" /etc/subgid \
  || { echo "no subuid/subgid range for $USER on $(hostname): exclude this node" >&2; exit 2; }

mkdir -p "$C"
lfs getstripe -d "$C" | grep -q lcme_id || lfs setstripe -E 4M -c 1 -E 64M -c 4 -E -1 -c 16 -S 4M "$C"

podman build --no-cache -t "$NAME" -f Dockerfile .
rm -f "$DST".tmp.*
enroot import -o "$DST.tmp.$$" "podman://$NAME" || true   # exits 1 even on success
[ "$(stat -c %s "$DST.tmp.$$")" -gt 1073741824 ] || { echo "enroot import produced no usable image" >&2; exit 1; }
mv "$DST.tmp.$$" "$DST"
ls -la "$DST"
