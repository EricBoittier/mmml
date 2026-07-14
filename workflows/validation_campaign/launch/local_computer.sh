#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python workflows/validation_campaign/scripts/campaign.py run-local --environment local_computer "$@"

