#!/usr/bin/env bash
# Install every-15-minute cron for dense_dt_campaign monitor --react
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MONITOR="${ROOT}/scripts/slurm/dense_dt_campaign/monitor_and_progress.sh"
LOG_DIR="${ROOT}/artifacts/lj_scales/dense_dt_campaign"
mkdir -p "$LOG_DIR"
chmod +x "$MONITOR" "${ROOT}/scripts/slurm/dense_dt_campaign/"*.sh

CRON_LINE="*/15 * * * * PATH=${HOME}/.local/bin:${HOME}/.cargo/bin:/usr/bin:/bin:\$PATH bash ${MONITOR} --react >> ${LOG_DIR}/monitor.log 2>&1"
MARKER="# mmml-dense-dt-campaign-monitor"

TMP="$(mktemp)"
( crontab -l 2>/dev/null | grep -v "$MARKER" | grep -v "dense_dt_campaign/monitor_and_progress" || true ) > "$TMP"
echo "$CRON_LINE $MARKER" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"

echo "Installed 15-min dense_dt_campaign monitor cron:"
crontab -l | grep -F "$MARKER" || true
echo "Status: ${LOG_DIR}/STATUS.md"
echo "Log:    ${LOG_DIR}/monitor.log"
