#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Gemini uses native transport; other model requests retain Chat passthrough.
rbnx_schema_args=()
case "${RBNX_REFINE_RTDL_SCHEMA:-0}" in
  1|true) rbnx_schema_args+=(--refine-rtdl-schema) ;;
  0|false) ;;
  *) echo 'RBNX_REFINE_RTDL_SCHEMA must be 0/1/false/true' >&2; exit 2 ;;
esac
exec python3 monitor.py \
  --profile "${RBNX_MONITOR_PROFILE:-generic}" \
  --provider "${RBNX_MONITOR_PROVIDER:-auto}" \
  --upstream "${RBNX_MONITOR_UPSTREAM:-https://api.gptsapi.net}" \
  --port "${RBNX_MONITOR_PORT:-54090}" \
  --gemini-thinking-level "${RBNX_GEMINI_THINKING_LEVEL:-low}" --claude-plan-depth "${RBNX_CLAUDE_PLAN_DEPTH:-1}" \
  --claude-schema-cap-limit "${RBNX_CLAUDE_SCHEMA_CAP_LIMIT:-1}" \
  --claude-schema-mode "${RBNX_CLAUDE_SCHEMA_MODE:-structured}" \
  --compact-context "${rbnx_schema_args[@]}" "$@"
