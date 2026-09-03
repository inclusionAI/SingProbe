#!/usr/bin/env bash
#
# ling-3.0-tiny.sh — per-model training pipeline for Ling-3.0-tiny.
#
# Thin wrapper around scripts/run_train_pipeline.sh that pins --config to
# configs/all_models/ling-3.0-tiny.yaml so swapping models is a single
# command. The underlying pipeline script handles everything (SGLang
# startup/readiness/teardown, train.py, checkpoint ->
# safetensors conversion); the per-model knobs (model path, hidden_size,
# layer ids, SGLang tp/dp/gpus, output_dir) all live in the YAML.
#
# All other args are forwarded to the pipeline script unchanged:
#   --ddp <N>           torchrun --nproc_per_node=N (sglang backend only)
#   --seed <N>          RNG seed (default 42)
#   --log-level <lvl>   train.py log level (default INFO)
#   --resume <dir>      resume from a checkpoint dir
# SGLang-/train-side env vars (TP, DP, GPUS, TRAIN_GPUS, SGLANG_PORT,
# SGLANG_SGLOG, ...) are read by the wrapped script as usual.
#
# Usage (run from the repo root):
#   bash scripts/all/ling-3.0-tiny.sh
#   bash scripts/all/ling-3.0-tiny.sh --ddp 4
#   bash scripts/all/ling-3.0-tiny.sh --seed 7
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/all -> scripts -> repo root.
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="configs/all_models/ling-3.0-tiny.yaml"

# Re-state --chunked-prefill-size -1 (DISABLE chunked prefill): SGLang's
# token-probe save path only dumps ALL prompt tokens when a prompt is not
# split across forwards; auto chunked prefill tripped the dump-count mismatch
# ("dump has 1 tokens, sent N"). See scripts/run_train_pipeline.sh for the
# full story -- wrappers MUST re-state this flag.
export SGLANG_EXTRA_ARGS="--chunked-prefill-size -1"

export SGLANG_SGLOG="sglang_server_ling-3.0-tiny.log"

bash "scripts/run_train_pipeline.sh" --config "${CONFIG}" "$@"
