#!/usr/bin/env bash
#
# run_train_pipeline.sh — SGLang + SingProbe training pipeline orchestrator.
#
# Runs the complete flow end to end (no external launcher deps):
#   1. Start the SGLang hidden-state server in the background (GPUs $GPUS) and
#      wait until it is ready to serve (health endpoint + log signal).
#   2. Launch training (train.py, optionally via torchrun).
#   3. Convert the final SingProbe checkpoint (.pt) to safetensors + config.json
#      (skipped if training failed or no checkpoint exists).
#   On exit (any path), tear the SGLang server down so the GPUs release.
#
# GPU split: SGLang spans $GPUS (default: back 12, 4-15); the train phase pins
# CUDA_VISIBLE_DEVICES to the front GPUs (default 0-3). Keep the two sets
# disjoint. TRAIN_GPUS="cpu"/"-1" is a special value: the train phase empties
# CUDA_VISIBLE_DEVICES instead (the probe trains on CPU, restored to all-visible
# for the convert phase afterwards).
#
# Usage:
#   bash scripts/run_train_pipeline.sh --config configs/all_models/ling-3.0-flash.yaml
#   bash scripts/run_train_pipeline.sh --config configs/all_models/ling-3.0-tiny.yaml --ddp 4
#   SGLANG_SGLOG=sglang.log bash scripts/run_train_pipeline.sh --config <cfg>
#
# Options: --config <yaml> / --ddp <N> / --seed <N> / --log-level <lvl> /
#          --resume <checkpoint dir>.
# SGLang-side tunables (env):  TP DP PORT GPUS MODEL_PATH PROBE_CKPT SAVE_DIR
#   MEM_FRACTION HIDDEN_SIZE LAYER_IDS
# Train-side tunables (env):   TRAIN_GPUS (comma-list pinned for the train
#   phase; "cpu"/"-1" hides all GPUs from train.py so the probe trains on CPU)
#
# Single source of truth for the tapped layers: base_model.hidden_layers in the
# YAML config (the same list train.py uses to build the probe's concat input).
# The SGLang identity probe's layer_ids is read straight from there, so you only
# configure the layers once. Override with LAYER_IDS (env) for ad-hoc runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Environment prerequisites (PyTorch, the token-probe-enabled sglang build,
# fla for Ling's linear-attention layers, spaCy + en_core_web_sm) are NOT
# installed here -- set them up once as described in README.md.

# The BASE model identity (HuggingFace repo ID or local path) comes from
# base_model.name in the YAML; SGLang resolves it from the Hub (or disk) at
# launch -- swapping models is a one-file change.

# ============================== arg parsing ================================
CONFIG="configs/all_models/ling-3.0-flash.yaml"
RESUME=""
SEED=42
LOG_LEVEL="INFO"
DDP="4"           # empty => single-process; "N" => torchrun --nproc_per_node=N

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)    CONFIG="$2";    shift 2 ;;
        --resume)    RESUME="$2";    shift 2 ;;
        --seed)      SEED="$2";      shift 2 ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        --ddp)       DDP="$2";       shift 2 ;;
        --help)      sed -n '2,34p' "$0"; exit 0 ;;
        *) echo "Error: unknown option $1 (see --help)"; exit 1 ;;
    esac
done

# ============================ SGLang-side tunables =========================
SGLANG_PORT="${SGLANG_PORT:-6000}"
# Where to tee SGLang's stdout/stderr. Empty => discard. Set it to keep the log.
SGLANG_SGLOG="${SGLANG_SGLOG:-sglang_server.log}"
# How long to wait for SGLang to come up (seconds). Large model + probe build is slow.
SGLANG_READY_TIMEOUT="${SGLANG_READY_TIMEOUT:-1800}"

# Must match sglang_probe_ckpt in the YAML (the client validates against it).
PROBE_CKPT="${PROBE_CKPT:-path/to/probe_ckpt}"
SAVE_DIR="${SAVE_DIR:-/dev/shm/save_probe}"             # tmpfs: probe dumps never hit disk
MEM_FRACTION="${MEM_FRACTION:-0.85}"
# Extra args appended verbatim to every sglang.launch_server call below. Use it
# to pass model-specific overrides the YAML does not surface.
#
# DEFAULT (applies to every model): force
#   --chunked-prefill-size -1   (DISABLE chunked prefill entirely)
# SGLang's token-probe save path (`scheduler_output_processor_mixin.py`) only
# dumps ALL prompt tokens when the per-forward score row count equals
# sum(extend_lens) of THIS forward. Under auto chunked prefill a prompt larger
# than the chunk is split across forwards; each forward's `scores.shape[0]`
# then equals the CHUNK size while `extend_lens` (snapshotted at
# req.extend_input_len) reflects the chunked/remainder length, so the
# `all_token` gate trips to False and only ~0/1 rows per request are saved ->
# client-side "dump token count mismatch: dump has 1 tokens, sent N". This is
# NOT model-specific -- any model whose chunked_prefill_size splits a prompt
# hits it; when unset, SGLang auto-tunes chunked_prefill_size (2048/4096/8192 by
# GPU memory) so even large models trip it on long prompts.
#
# Training already truncates every prompt to <= max_seq_length (8192), so
# chunked prefill has NO upside here -- disabling it (--chunked-prefill-size -1)
# guarantees no prompt is ever split -> all_token=True -> all tokens dumped.
# The per-forward token count is then bounded by --max-prefill-tokens (sglang
# default 16384, left unchanged). An explicit SGLANG_EXTRA_ARGS from a wrapper
# still wins (see _append below); such wrappers must re-state
# --chunked-prefill-size -1 or they lose it and re-introduce the mismatch.
SGLANG_EXTRA_ARGS_DEFAULT="--chunked-prefill-size -1"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-${SGLANG_EXTRA_ARGS_DEFAULT}}"
# GPUS / TP / DP / MODEL_PATH / HIDDEN_SIZE used to be hardcoded here; they now
# come from the YAML config (base_model.inference.sglang_*, base_model.name /
# .hidden_size), read by the python probe below. Shell env (TP=, GPUS=,
# MODEL_PATH=, ...) still wins -- the YAML only supplies the default.

# Read the YAML config ONCE: validate it parses, pull framework + output dir +
# master port for DDP, ensure at least one train + one val dataset exists, and
# read base_model.hidden_layers (the single source of truth for the tapped
# layers). LAYER_IDS env var wins if set; otherwise the probe's hidden_layers is
# used for both the SGLang identity probe and the SingProbe concat input layers.
read -r FRAMEWORK OUTPUT_DIR MASTER_PORT CFG_HIDDEN_LAYERS \
      CFG_MODEL_PATH CFG_HIDDEN_SIZE CFG_TP CFG_DP CFG_GPUS \
      < <(python3 - "$CONFIG" <<'PYEOF' | tail -n 1
import os, sys
sys.path.insert(0, '.')
from config import load_config
cfg = load_config(sys.argv[1])
bm, inf = cfg.base_model, cfg.base_model.inference
layers = bm.hidden_layers
if not layers:
    print(f"ERROR: base_model.hidden_layers is empty in {sys.argv[1]}", file=sys.stderr)
    sys.exit(1)
train_ok = any(any(os.path.exists(x) for x in str(p).split(';') if x.strip()) for p in (cfg.data.train_safety_path, cfg.data.train_hallu_path))
val_ok   = any(any(os.path.exists(x) for x in str(p).split(';') if x.strip()) for p in (cfg.data.val_safety_path, cfg.data.val_hallu_path))
if not (train_ok and val_ok):
    print(f"ERROR: missing data (train_ok={train_ok}, val_ok={val_ok}) -- check data.*_path in {sys.argv[1]}", file=sys.stderr)
    sys.exit(1)
layers_str = "[" + ",".join(str(int(l)) for l in layers) + "]"
# Base-model identity (single source of truth from the YAML): name (a
# HuggingFace repo ID or local path) doubles as the SGLang --model-path;
# hidden_size sizes the identity probe; tp/dp/gpus launch the server. Shell
# env still overrides each at the bash level right after the read.
print(inf.framework, cfg.training.output_dir, cfg.distributed.master_port, layers_str,
      bm.name, bm.hidden_size or 0, inf.sglang_tp, inf.sglang_dp, inf.sglang_gpus)
PYEOF
)
if [ -z "$FRAMEWORK" ]; then
    echo "Error: config/data probe failed for '$CONFIG' (see message above)." >&2
    exit 1
fi

# Resolve the SGLang launch params + base-model identity from the YAML (the CFG_*
# fields read above). Shell env (TP=, DP=, GPUS=, MODEL_PATH=, HIDDEN_SIZE=) still
# wins -- the YAML only supplies the default when the env is unset.
MODEL_PATH="${MODEL_PATH:-$CFG_MODEL_PATH}"
HIDDEN_SIZE="${HIDDEN_SIZE:-$CFG_HIDDEN_SIZE}"
TP="${TP:-$CFG_TP}"
DP="${DP:-$CFG_DP}"
GPUS="${GPUS:-$CFG_GPUS}"

# No download step here: base_model.name is a HuggingFace repo ID (or a local
# path to a downloaded copy). SGLang resolves/downloads it from the Hub when
# the server starts; train.py's tokenizer load does the same.
LAYER_IDS="${LAYER_IDS:-${CFG_HIDDEN_LAYERS}}"

# Auto-parse the number of tapped layers from the resolved LAYER_IDS (JSON array
# form, e.g. "[1, 20, 30]"). num_layers must equal len(base_model_layer_ids);
# the SGLang identity probe advertises it so the probe dump width and the
# SingProbe model's concat-layer split agree. Computed from the FINAL LAYER_IDS
# (after the env override above), so an ad-hoc LAYER_IDS=... bash run stays
# consistent.
NUM_LAYERS=$(printf '%s\n' "$LAYER_IDS" | tr -d '[] \t' | awk -F',' '{ if ($0 == "") print 0; else print NF }')

# ============================ Train-side tunables ==========================
# GPUs pinned for the training phase (CUDA_VISIBLE_DEVICES). Keep disjoint from
# SGLANG GPUS; DDP spreads torchrun ranks across them. The special values "cpu"
# and "-1" empty CUDA_VISIBLE_DEVICES so train.py falls back to device=cpu
# (probe-only CPU training; the frozen Base Model still runs on the SGLang
# GPUs).
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"

# spaCy POS assertion-mask model location (only used when the YAML has
# training.use_hallu_assertion_mask: true). Points at the unzipped
# en_core_web_sm dir, so the model does NOT need a global `spacy download`.
# data/assertion_mask.py::get_spacy_nlp() reads SPACY_MODEL_PATH first; override
# with SPACY_MODEL_PATH=... bash scripts/run_train_pipeline.sh ... . Empty =>
# fall back to the globally-installed package name / SPACY_MODEL_NAME.
export SPACY_MODEL_PATH="${SPACY_MODEL_PATH:-path/to/en_core_web_sm}"

# ====================== SGLang readiness / kill helpers ====================

wait_for_sglang() {
    # Block until SGLang is serving on $SGLANG_PORT, or until
    # $SGLANG_READY_TIMEOUT elapses / the server process dies.
    local deadline=$(( $(date +%s) + SGLANG_READY_TIMEOUT ))
    local have_curl=1
    command -v curl >/dev/null 2>&1 || have_curl=0

    echo ">> Waiting for SGLang to become ready (port ${SGLANG_PORT}, timeout ${SGLANG_READY_TIMEOUT}s)..."
    while :; do
        # If the server process died while we waited, fail fast.
        if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
            echo "Error: SGLang server process exited before becoming ready." >&2
            return 1
        fi
        # Primary signal: 200 from any known health endpoint.
        if [ "$have_curl" -eq 1 ]; then
            for ep in /health /health_generate; do
                if curl -fsS "http://127.0.0.1:${SGLANG_PORT}${ep}" >/dev/null 2>&1; then
                    echo ">> SGLang is ready (${ep} responding)."
                    return 0
                fi
            done
        fi
        # Fallback: readiness line in the server log.
        if [ -n "${SGLANG_SGLOG:-}" ] && grep -q "The server is fired up" "${SGLANG_SGLOG}" 2>/dev/null; then
            echo ">> SGLang is ready (readiness line in log)."
            return 0
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "Error: timed out after ${SGLANG_READY_TIMEOUT}s waiting for SGLang to be ready." >&2
            return 1
        fi
        sleep 5
    done
}

teardown_sglang() {
    # Kill the entire SGLang process group (launch_server spawns TP*DP workers +
    # a scheduler subprocess). We started the parent as a process-group leader
    # (setsid), so kill the negated PGID.
    if [ -n "${SGLANG_PGID:-}" ] && kill -0 "$SGLANG_PGID" 2>/dev/null; then
        echo ">> Stopping SGLang server (process group ${SGLANG_PGID})..."
        kill -TERM -- "-${SGLANG_PGID}" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$SGLANG_PGID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SGLANG_PGID" 2>/dev/null; then
            echo ">> SGLang did not exit on TERM; sending KILL..."
            kill -KILL -- "-${SGLANG_PGID}" 2>/dev/null || true
        fi
        echo ">> SGLang stopped."
    fi
}

# Ensure teardown always runs, no matter how training exits.
SGLANG_PID=""
SGLANG_PGID=""
trap teardown_sglang EXIT

# ============================ 1. launch SGLang =============================
echo "=============================================================="
echo ">> [1/3] Starting SGLang server (background, GPUs: ${GPUS})"
echo "   log: ${SGLANG_SGLOG:-<discarded>}"
echo "   model: ${MODEL_PATH}  (TP=${TP} DP=${DP} mem_frac=${MEM_FRACTION})"
echo "   probe: hidden_size=${HIDDEN_SIZE} layers=${LAYER_IDS} -> ${PROBE_CKPT}"
echo "   dump:  ${SAVE_DIR}"
echo "=============================================================="

# Build probe checkpoint (idempotent).
# Probe config schema (this SGLang build's ProbeConfig reads these keys off
# config.json directly): guard_arch = head type ("identity" = pass-through tap),
# hidden_dim = per-layer hidden size of the Base Model, base_model_layer_ids =
# the tapped layers, num_layers = len(base_model_layer_ids). num_layers is
# auto-parsed above from the resolved LAYER_IDS so it stays consistent with the
# layers the Guard concatenates.
mkdir -p "$PROBE_CKPT" "$SAVE_DIR"
echo "{\"guard_arch\": \"identity\", \"hidden_dim\": ${HIDDEN_SIZE}, \"base_model_layer_ids\": ${LAYER_IDS}, \"num_layers\": ${NUM_LAYERS}}" \
    > "$PROBE_CKPT/config.json"

# Probe dump env.
export SGLANG_TOKEN_PROBE_SAVE_DIR="$SAVE_DIR"
export SGLANG_ENABLE_TOKEN_PROBE_PREFILL=1
export SGLANG_SAIL_CUDA_FLA=0
# Clean stale dumps so the training client starts with an empty directory.
find "$SAVE_DIR" -name '*.safetensors' -delete 2>/dev/null || true

# No YaRN RoPE extrapolation: the current Ling-3.0-flash build's native
# max_position_embeddings (131072) already covers the longest safety dataset
# samples (~34k), so SGLang serves them at model-native context with no
# --context-length / --json-model-override-args override. Earlier builds needed
# YaRN (original_max=20000, factor=16); removed as obsolete for this model.

# Start the server as its own process group (setsid) so we can kill the whole
# tree — workers + scheduler — with one negated PGID at teardown.
# Echo the FINAL launch_server argv (SGLANG_EXTRA_ARGS expanded) so the log
# shows exactly which args SGLang received.
echo ">> Final SGLang argv: python -m sglang.launch_server --model-path ${MODEL_PATH} --tp ${TP} --dp ${DP} ${SGLANG_EXTRA_ARGS}"
if [ -n "${SGLANG_SGLOG}" ]; then
    setsid bash -c "CUDA_VISIBLE_DEVICES=\"${GPUS}\" python -m sglang.launch_server \
        --model-path \"${MODEL_PATH}\" \
        --host 127.0.0.1 \
        --port ${SGLANG_PORT} \
        --trust-remote-code \
        --tp ${TP} \
        --dp ${DP} \
        --probe-ckpt \"${PROBE_CKPT}\" \
        --mem-fraction-static ${MEM_FRACTION} \
        --disable-radix-cache \
        --log-level-http warning \
        ${SGLANG_EXTRA_ARGS}" >"${SGLANG_SGLOG}" 2>&1 &
else
    setsid bash -c "CUDA_VISIBLE_DEVICES=\"${GPUS}\" python -m sglang.launch_server \
        --model-path \"${MODEL_PATH}\" \
        --host 127.0.0.1 \
        --port ${SGLANG_PORT} \
        --trust-remote-code \
        --tp ${TP} \
        --dp ${DP} \
        --probe-ckpt \"${PROBE_CKPT}\" \
        --mem-fraction-static ${MEM_FRACTION} \
        --disable-radix-cache \
        --log-level-http warning \
        ${SGLANG_EXTRA_ARGS}" >/dev/null 2>&1 &
fi
SGLANG_PID=$!
SGLANG_PGID=$!
echo ">> SGLang launched (pid=${SGLANG_PID}, pgid=${SGLANG_PGID})."

# ============================ wait for readiness ===========================
if ! wait_for_sglang; then
    echo "Error: SGLang failed to become ready. Check ${SGLANG_SGLOG:-the server output}." >&2
    exit 1
fi

# ============================ 2. run training ===============================
echo "=============================================================="
echo ">> [2/3] Starting training (train.py, GPUs: ${TRAIN_GPUS})"
echo "   config: ${CONFIG}  seed=${SEED}  log-level=${LOG_LEVEL}  ddp=${DDP:-none}"
[ -n "$RESUME" ] && echo "   resume: ${RESUME}"
echo "=============================================================="

# (Config was already probed above: FRAMEWORK/OUTPUT_DIR/MASTER_PORT/LAYER_IDS
# are resolved, and train + val data existence was checked there.)

# DDP is only supported on the sglang backend.
if [ -n "$DDP" ] && [ "$FRAMEWORK" != "sglang" ]; then
    echo "Error: --ddp is only supported on the sglang backend (config framework='${FRAMEWORK}')." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Build the train.py command. NOTE: do not `exec` — the EXIT trap must fire to
# tear down SGLang.
# TRAIN_GPUS="cpu" or "-1" -> hide all GPUs from train.py so the probe trains
# on CPU (torch.cuda.is_available() == False -> device=cpu). Anything else is
# passed through as the visible GPU id list.
if [ "$TRAIN_GPUS" = "cpu" ] || [ "$TRAIN_GPUS" = "-1" ]; then
    export CUDA_VISIBLE_DEVICES=""
    echo ">> TRAIN_GPUS=${TRAIN_GPUS} -> CUDA_VISIBLE_DEVICES emptied; probe trains on CPU"
else
    export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
fi

ARGS="--config ${CONFIG} --seed ${SEED} --log-level ${LOG_LEVEL}"
[ -n "$RESUME" ] && ARGS="${ARGS} --resume ${RESUME}"

if [ -n "$DDP" ]; then
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT="${MASTER_PORT}"
    CMD="torchrun --nproc_per_node=${DDP} train.py ${ARGS}"
    echo ">> Launching DDP: world_size=${DDP} (master_port=${MASTER_PORT})"
else
    CMD="python3 train.py ${ARGS}"
    echo ">> Launching single-process (no DDP)"
fi
echo ">> Command: ${CMD}"
echo ">> Output:  ${OUTPUT_DIR}"
# Persist torchrun/train combined stdout+stderr to a run log under OUTPUT_DIR.
# This is the critical forensic channel for a HARD rank death (a CUDA SIGABRT,
# an OOM-killer SIGKILL, or a faulthandler C-level stack) -- none of which the
# per-rank train_*.log files ever capture: MainProcessLogger gates logger.* to
# rank 0 (ranks 1..N-1 leave 0-byte files), and a hard death prints no Python
# traceback, so the dieing rank's only output -- faulthandler stack + [trace]
# flight-recorder line + any "CUDA error"/"out of memory" -- lands on STDERR,
# which previously lived only in the terminal scrollback and was easily lost.
# `2>&1 | tee -a` merges stderr into the saved stream and passes it through
# live, so the console still shows tqdm + the crash exactly as before. Under
# `set -o pipefail` a non-zero bash -c rc still propagates into TRAIN_RC.
TRAIN_RUN_LOG="${TRAIN_RUN_LOG:-${OUTPUT_DIR}/train_run.log}"
echo ">> Run log:  ${TRAIN_RUN_LOG}  (combined stdout+stderr; tee'd live)"
echo

TRAIN_RC=0
bash -c "$CMD" 2>&1 | tee -a "${TRAIN_RUN_LOG}" || TRAIN_RC=$?

# CPU-train mode only: the CUDA_VISIBLE_DEVICES="" exported above would leak
# into the checkpoint-convert phase below. Unset it so that phase sees all
# GPUs again.
if [ "$TRAIN_GPUS" = "cpu" ] || [ "$TRAIN_GPUS" = "-1" ]; then
    unset CUDA_VISIBLE_DEVICES
fi

# ============================ 3. convert checkpoint =========================
# Only convert when training succeeded (rc=0) and left a checkpoint behind.
#
# train.py writes checkpoints under $OUTPUT_DIR as:
#   - $OUTPUT_DIR/checkpoint-<step>        periodic snapshots (top level)
#   - $OUTPUT_DIR/best/checkpoint-<step>    best-val snapshots, one per eval
#   - $OUTPUT_DIR/final/checkpoint-<step>   final-model snapshot at end of run
# Each leaf holds guard_model.pt (+ optional optimizer.pt/training_state.json).
#
# The real checkpoint to convert is the highest-numbered checkpoint-<step> under
# best/ -- best keeps a snapshot at every eval boundary, so the max step there is
# the latest best-val model. A plain `-maxdepth 1` find on $OUTPUT_DIR only sees
# the top-level periodic checkpoints and would pick the wrong (smaller) step, so
# we search nested dirs with -maxdepth 2 instead.
#
# Selection (rank key = step, ascending; tie-break prefers best/ > final/ >
# top-level, so the latest best-val snapshot wins):
#   1. $OUTPUT_DIR/best/checkpoint-<step>      <- latest best-val (preferred)
#   2. $OUTPUT_DIR/final/checkpoint-<step>     <- explicit final-model dump
#   3. $OUTPUT_DIR/checkpoint-<step>           <- top-level periodic
#   4. $OUTPUT_DIR/best/guard_model.pt         <- legacy flat best/ layout
# convert_checkpoint_to_safetensors.py then emits model.safetensors + config.json
# (with base_model_layer_ids) and an injected standalone modeling file.
CONVERT_RC=0
FINAL_CKPT=""

# Pick the best checkpoint-<step> dir under $1. Searches depth 2 (covers $1/best,
# $1/final, and top level). Ranks by integer step desc, tie-break by location
# priority (best > final > top). Pairs each dir with a sortable key once; the
# ranked last line is the winner. Empty output if no checkpoint-* dir exists.
pick_latest_step_dir() {
    find "$1" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null \
        | awk -F/ '
            { dir=$0; n=split(dir, a, "/"); step=a[n]; sub(/^checkpoint-/, "", step);
              if (step !~ /^[0-9]+$/) next;
              base = (index(dir, "/best/") ? 1 : index(dir, "/final/") ? 2 : 3);
              printf "%012d %d %s\n", step+0, base, dir }' \
        | sort -n -k1,1r -k2,2n | head -n 1 | cut -d' ' -f3-
}

if [ "$TRAIN_RC" -eq 0 ] && [ -d "$OUTPUT_DIR" ]; then
    FINAL_CKPT="$(pick_latest_step_dir "$OUTPUT_DIR")"
    # Legacy fallback: flat best/ with guard_model.pt sitting directly in it.
    if [ -z "$FINAL_CKPT" ] && [ -f "${OUTPUT_DIR}/best/guard_model.pt" ]; then
        FINAL_CKPT="${OUTPUT_DIR}/best"
    fi
fi

echo "=============================================================="
echo ">> [3/3] Converting final checkpoint to safetensors"
if [ -n "$FINAL_CKPT" ] && [ -f "${FINAL_CKPT}/guard_model.pt" ]; then
    echo "   source: ${FINAL_CKPT}/guard_model.pt"
    echo "   output: ${FINAL_CKPT}/safetensors"
    if ! python3 "${SCRIPT_DIR}/convert_checkpoint_to_safetensors.py" \
            --checkpoint "${FINAL_CKPT}" \
            --output-dir "${FINAL_CKPT}/safetensors" \
            --copy-training-state; then
        CONVERT_RC=$?
        echo ">> Warning: checkpoint conversion failed (rc=${CONVERT_RC}); .pt checkpoint left intact." >&2
    fi
else
    echo ">> Skipped: no guard_model.pt found under $OUTPUT_DIR (training may not have saved a checkpoint)."
fi
echo "=============================================================="

echo "=============================================================="
echo ">> Training exited (rc=${TRAIN_RC}); convert rc=${CONVERT_RC}. Tearing down SGLang..."
echo "=============================================================="
# Propagate the first non-zero rc (train > convert); the SGLang server is torn
# down by the EXIT trap either way.
if [ "$TRAIN_RC" -ne 0 ]; then
    exit "$TRAIN_RC"
elif [ "$CONVERT_RC" -ne 0 ]; then
    exit "$CONVERT_RC"
fi
exit 0
