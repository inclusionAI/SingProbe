#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the local base model directory}"
PROBE_CKPT="${PROBE_CKPT:-$ROOT/token-probe-validation/identity}"
SAVE_DIR="${SAVE_DIR:-$ROOT/token-probe-validation/outputs}"
PYTHON="${PYTHON:-$ROOT/.venv-token-probe/bin/python}"
export MODEL_PATH PROBE_CKPT SAVE_DIR
mkdir -p "$PROBE_CKPT" "$SAVE_DIR"
"$PYTHON" - <<'PY'
import json, os
from pathlib import Path
config=json.loads((Path(os.environ['MODEL_PATH'])/'config.json').read_text())
text=config.get('text_config', config)
layers=json.loads(os.environ.get('LAYER_IDS', '[0,14,39]'))
assert all(0 <= i < text['num_hidden_layers'] for i in layers)
probe={'model_type':'sing_probe_identity','hidden_size':text['hidden_size'],
       'base_model_layer_ids':layers,'num_labels':10}
(Path(os.environ['PROBE_CKPT'])/'config.json').write_text(json.dumps(probe,indent=2)+'\n')
print('Identity probe:',probe,flush=True)
PY
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_TOKEN_PROBE_SAVE_DIR="$SAVE_DIR"
export SGLANG_ENABLE_TOKEN_PROBE_PREFILL=1
exec "$PYTHON" -m sglang.launch_server \
  --model-path "$MODEL_PATH" --probe-ckpt "$PROBE_CKPT" \
  --tp-size "${TP_SIZE:-8}" --ep-size "${EP_SIZE:-${TP_SIZE:-8}}" --host 127.0.0.1 --port "${PORT:-31080}" \
  --context-length 2048 --max-total-tokens 4096 --max-running-requests 2 \
  --chunked-prefill-size 256 --mem-fraction-static 0.85 \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  "$@"
