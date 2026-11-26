#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROBE_CKPT="${PROBE_CKPT:-$ROOT/token-probe-validation/qwen38-identity}"
export SAVE_DIR="${SAVE_DIR:-$ROOT/token-probe-validation/qwen38-outputs}"
export LAYER_IDS="${LAYER_IDS:-[15,31,47]}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV=0
exec bash "$ROOT/scripts/token_probe/launch_identity.sh" \
  --trust-remote-code --chunked-prefill-size -1 --max-prefill-tokens 8192 "$@"
