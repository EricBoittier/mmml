#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python workflows/validation_campaign/scripts/campaign.py "${1:-prepare}" --environment scicore "${@:2}"

