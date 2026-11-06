#!/usr/bin/env bash
# Generic (provider- and embodiment-neutral) recording proxy with context
# compaction. Configure everything through environment variables.
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec python3 monitor.py \
  --profile generic \
  --provider "${RBNX_MONITOR_PROVIDER:-custom}" \
  --upstream "${RBNX_MONITOR_UPSTREAM:?set RBNX_MONITOR_UPSTREAM to the provider base URL}" \
  --port "${RBNX_MONITOR_PORT:-54090}" \
  --records "${RBNX_MONITOR_RECORDS:-records/generic}" \
  --compact-context \
  "$@"
