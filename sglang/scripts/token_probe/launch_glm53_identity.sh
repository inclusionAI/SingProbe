#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROBE_CKPT="${PROBE_CKPT:-$ROOT/token-probe-validation/glm53-identity}"
export SAVE_DIR="${SAVE_DIR:-$ROOT/token-probe-validation/glm53-outputs}"
export LAYER_IDS="${LAYER_IDS:-[14,29,43]}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
exec bash "$ROOT/scripts/token_probe/launch_identity.sh" \
  --trust-remote-code --chunked-prefill-size -1 --max-prefill-tokens 8192 "$@"
