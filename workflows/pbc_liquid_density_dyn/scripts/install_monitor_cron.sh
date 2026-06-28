#!/usr/bin/env bash
# Install hourly cron entry for monitor_health.sh --react
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONITOR="$WORKFLOW_ROOT/scripts/monitor_health.sh"
LOG_DIR="$WORKFLOW_ROOT/logs"
mkdir -p "$LOG_DIR"

CKPT="${MMML_CKPT:-/mmhome/boittier/home/mmml_tutorial/acodcm/ckpts/dcm1-c137fb42-1f65-4748-880b-8f8184a20f70}"
CRON_LINE="0 * * * * PATH=${HOME}/.local/bin:${HOME}/.cargo/bin:\$PATH MMML_CKPT=${CKPT} JAX_ENABLE_X64=1 ${MONITOR} --react >> ${LOG_DIR}/monitor.log 2>&1"

MARKER="# mmml-pbc-liquid-density-monitor"
TMP="$(mktemp)"
( crontab -l 2>/dev/null | grep -v "$MARKER" | grep -v "monitor_health.sh" || true ) > "$TMP"
echo "$CRON_LINE $MARKER" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"

echo "Installed hourly monitor cron:"
crontab -l | grep monitor_health || true
echo ""
echo "Logs: ${LOG_DIR}/monitor.log"
echo "Latest JSON: ${WORKFLOW_ROOT}/results/monitor_latest.json"
