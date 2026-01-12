#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 USERNAME [extra streamlit args...]" >&2
  exit 1
fi

USERNAME="$1"
shift

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

streamlit run labeller_app_ordered.py \
  --server.headless true \
  -- \
  --username "${USERNAME}" \
  --keys debate_keys.json \
  --automated-evals-dir automated_evaluations \
  "$@"
