#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROUND="${1:-1}"
if [ "$#" -gt 0 ]; then
  shift
fi
CONFIG="${CAIL_CONFIG:-configs/pipeline.linux.json}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m cail --config "$CONFIG" grpo --round "$ROUND" --execute "$@"
