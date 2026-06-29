#!/usr/bin/env bash
# Build MMML native CHARMM executable (as_library=OFF) for DOMDEC Tier 3 smoke.
#
# Uses a separate cmake build dir from libcharmm.so (default: .../linux-x86_64-exec).
# Same MPI + DOMDEC stack as scripts/rebuild_charmm_mlpot.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/rebuild_charmm_mlpot.sh" --native-exec "$@"
