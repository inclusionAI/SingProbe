#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROBE_CKPT="${PROBE_CKPT:-$ROOT/token-probe-validation/ling-identity}"
export SAVE_DIR="${SAVE_DIR:-$ROOT/token-probe-validation/ling-outputs}"
export LAYER_IDS="${LAYER_IDS:-[0,35,71]}"
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP="${SGLANG_JIT_DEEPGEMM_FAST_WARMUP:-1}"
# Avoid the image's Inductor subprocess-pool lifecycle failure on first decode.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
exec bash "$ROOT/scripts/token_probe/launch_identity.sh" "$@"
