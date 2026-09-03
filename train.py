"""
End-to-End Training Script for SingProbe

Usage:
    python train.py --config configs/all_models/ling-3.0-flash.yaml
    python train.py --config configs/all_models/ling-3.0-flash.yaml --resume outputs/checkpoint-1000

Single-process only. The DeepSpeed TP / torchrun path has been retired; the base
model runs either via HuggingFace device_map sharding (single process) or an
external SGLang server. No torchrun/NCCL.
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, Optional
import json
import time
import math
import traceback
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, load_config, any_data_path_exists, same_data_path_set
from data.samplers import TaskRatioSampler, DistributedTaskRatioSampler

# Distributed training of the Guard: the "device_map" base-model backend runs
# single-process (the HF model already shards across all GPUs in one process),
# while the "sglang" backend can replicate the tiny Guard across N GPUs (one
# rank each) and all-reduce gradients. The DDP path is launched via torchrun;
# when launched as a plain single process, WORLD_SIZE defaults to 1 and every
# helper below degrades to a no-op, so the single-process path is unchanged.

def _dist_enabled() -> bool:
    """True iff a torch.distributed process group is initialized."""
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    """Rank-0 (or the sole process) when DDP is active; True otherwise."""
    return (not _dist_enabled()) or dist.get_rank() == 0


def _resolve_load_strategy(framework: str) -> str:
    """Map config base_model.inference.framework to BaseModelWrapper.load_strategy.

    - "device_map": HuggingFace pipeline sharding (safe for huge MoE; slow for big batches)
    - anything else (incl. "auto"): let the wrapper auto-pick (device_map when multi-GPU)
    """
    fw = (framework or "").lower()
    if fw == "device_map":
        return "device_map"
    return "auto"


def _build_lr_scheduler(
    optimizer: optim.Optimizer,
    config: Config,
    total_optimizer_steps: int,
    logger: logging.Logger,
) -> Optional[LambdaLR]:
    """Build the LR scheduler from config.training.lr_scheduler_type.

    - "cosine_with_warmup" (default): linear warmup from 0 -> base lr over
      warmup_ratio of total_optimizer_steps, then cosine anneal to 0 over the
      rest. Mirrors HuggingFace's `get_cosine_schedule_with_warmup` semantics.
    - "constant" (or unknown): None -> the optimizer keeps a flat base lr
      (legacy pre-scheduler behavior; warmup_ratio is then a dead knob).

    total_optimizer_steps is the per-rank optimizer-step count for the whole run
    (len(train_loader) // grad_accum * epochs). Returns None for the constant
    schedule so callers can treat scheduler.step() / scheduler state as optional.
    """
    sched_type = (config.training.lr_scheduler_type or "").lower()
    if sched_type != "cosine_with_warmup":
        logger.info(
            f"LR scheduler: '{config.training.lr_scheduler_type}' -> constant lr "
            f"(no schedule); warmup_ratio unused."
        )
        return None

    total = max(int(total_optimizer_steps), 1)
    warmup_ratio = float(config.training.warmup_ratio)
    warmup_steps = max(int(round(total * warmup_ratio)), 0)
    base_lr = optimizer.param_groups[0]['lr']

    logger.info(
        f"LR scheduler: cosine_with_warmup -- total_steps={total}, "
        f"warmup_steps={warmup_steps} ({warmup_ratio:.0%}), base_lr={base_lr:.2e} -> 0"
    )

    def lr_lambda(current_step: int) -> float:
        # linear warmup 0->1, then cosine 1->0 on the remaining fraction.
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ----------------------------------------------------------------------------
# Process-group / rank state.
#
# Resolved once in main() from the torchrun env vars (LOCAL_RANK/WORLD_SIZE/
# RANK). When train.py is launched as a plain single process those vars are
# absent and we default to world_size=1, rank=0, local_rank=0 -- at which point
# every DDP/sampler/all-reduce branch is a no-op and behavior matches today.
# ----------------------------------------------------------------------------
WORLD_SIZE: int = 1
RANK: int = 0
LOCAL_RANK: int = 0


def _resolve_distributed_env() -> None:
    """Read torchrun env vars into the module-level WORLD_SIZE/RANK/LOCAL_RANK."""
    global WORLD_SIZE, RANK, LOCAL_RANK
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    RANK = int(os.environ.get("RANK", "0"))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

    # Some multi-node cluster launchers inject RANK/NODE_RANK/MASTER_ADDR/
    # WORLD_SIZE into EVERY pod for role detection -- those vars survive into
    # a plain `python3 train.py` launch (no torchrun), so WORLD_SIZE>1 would
    # trigger dist.init_process_group(backend=nccl) below and hang ~601s
    # waiting for peers that never join (the head launches ONE process).
    # DDP here is NCCL+GPU only (no gloo/CPU path; CPU training is
    # single-process by design), so WORLD_SIZE>1 with no CUDA is never a real
    # DDP run, always leaked cluster env. Collapse to a lone process instead
    # of hanging. The real torchrun DDP path always has CUDA
    # (TRAIN_GPUS=real GPUs), so this guard never fires there.
    if WORLD_SIZE > 1 and not torch.cuda.is_available():
        _eprint(
            f"[dist] WORLD_SIZE={WORLD_SIZE} in env but CUDA unavailable -- "
            f"looks like leaked cluster/torchrun env on a single CPU process; "
            f"falling back to single-process (WORLD_SIZE=1, RANK=0, LOCAL_RANK=0)."
        )
        WORLD_SIZE = 1
        RANK = 0
        LOCAL_RANK = 0


def _all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Mean-reduce a scalar/0-d tensor across ranks; no-op when world_size==1."""
    if not _dist_enabled() or WORLD_SIZE <= 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= WORLD_SIZE
    return tensor


def _all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum-reduce a tensor across ranks; no-op when world_size==1."""
    if not _dist_enabled() or WORLD_SIZE <= 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _all_reduce_any(flag: torch.Tensor) -> torch.Tensor:
    """Return 1.0 if ANY rank has flag truthy (nonzero), else 0.0.

    Used by the NaN/Inf guards so every rank makes the IDENTICAL skip/step
    decision: if one rank sees a bad loss/grad, ALL ranks skip the same step
    (an all-reduce collective must be entered by every rank or the survivors
    deadlock). No-op when world_size==1.
    """
    if not _dist_enabled() or WORLD_SIZE <= 1:
        return flag
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    return (flag > 0).to(flag.dtype)


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the bare module behind a (possibly DDP-wrapped) Guard."""
    return model.module if isinstance(model, DDP) else model


def _base_compute_dtype(config: Config) -> torch.dtype:
    """Base-Model / autocast compute dtype, and the dtype Guard artifacts are
    saved in. Follows config.base_model.inference.dtype: "float16" -> fp16,
    anything else -> bf16. The Guard's own parameters are fp32 master weights
    (see save_checkpoint and the Guard init in main())."""
    return torch.float16 if config.base_model.inference.dtype == "float16" else torch.bfloat16


def _fnum(x):
    """Coerce a loss dict value (Tensor / float / None) to a finite float, or None.

    Used for logging dynamic task weights returned in the loss dict: they may be
    a 0-d tensor, a python float, or absent (None) depending on the criterion
    path. Returns None for missing/NaN/inf so callers can render a placeholder.
    """
    if x is None:
        return None
    try:
        v = float(x.item()) if hasattr(x, 'item') else float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float('inf'), float('-inf')):  # NaN / inf guard
        return None
    return v


class MainProcessLogger:
    """Logger wrapper that only outputs on main process in distributed mode"""
    def __init__(self, logger):
        self._logger = logger

    def _log(self, level, msg, *args, **kwargs):
        if is_main_process():
            self._logger.log(level, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)


from models.base_model import BaseModelWrapper
from models.sglang_client import SGLangHiddenStateClient
from models.guard import GuardMLP, GuardMLPConfig
from data.dataset import GuardrailDataset
from data.collator import GuardrailCollator
from trainers.loss import BalancedMultiTaskLoss


def setup_logging(output_dir: str, log_level: str = "INFO"):
    """Setup logging configuration (file + console handlers).

    Under DDP each rank writes its own per-rank log file (rank-suffixed name)
    so concurrent ranks don't collide on the same timestamped filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    rank_suffix = f"_rank{RANK}" if WORLD_SIZE > 1 else ""
    log_file = os.path.join(
        output_dir,
        f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}{rank_suffix}.log"
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Clear existing handlers
    root_logger.handlers.clear()

    # File handler (every rank keeps its own file; MainProcessLogger below
    # gates the *console* output to rank 0).
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(console_handler)

    return MainProcessLogger(logging.getLogger(__name__))


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    scheduler: Optional[optim.lr_scheduler.LambdaLR],
    epoch: int,
    step: int,
    output_dir: str,
    config: Config,
    logger: logging.Logger,
    save_training_state: bool = True
):
    """Save checkpoint (only on main process in distributed mode)

    Args:
        scheduler: Optional LR scheduler whose state is persisted alongside the
            optimizer/scaler when save_training_state=True, so a resumed run
            continues the cosine schedule from the right LR. None when no
            scheduler is used (constant LR); then nothing is saved for it.
        save_training_state: If True (default), also persist optimizer + scaler
            state so training can be resumed. Set False for inference-only
            checkpoints (e.g. best model) to avoid writing large optimizer state.
    """
    # Under DDP, all ranks rendezvous here on every checkpoint decision so rank 0
    # can write while the others wait -- without this, rank 0 (doing disk I/O) can
    # fall behind the replicas that skip and race into the next training step.
    if _dist_enabled() and WORLD_SIZE > 1:
        dist.barrier()

    # Only save on main process to avoid file conflicts
    if not is_main_process():
        if _dist_enabled() and WORLD_SIZE > 1:
            dist.barrier()  # second rendezvous: wait for rank 0 to finish writing
        return

    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the BARE Guard model (state_dict keys must be `module.`-free so a
    # checkpoint loads into a fresh non-DDP Guard with strict=True). The caller
    # passes the (possibly DDP-wrapped) module; unwrap here. In-memory params
    # are fp32 master weights; the artifact keeps the compute dtype (bf16/fp16)
    # so the downstream convert / eval / SGLang layout is unchanged. Rounding
    # at save costs at most half a bf16 ulp; between saves optimizer updates
    # accumulate in fp32 (this is what lets RMSNorm gains of magnitude ~1 move
    # at all -- see the Guard init in main()).
    save_dtype = _base_compute_dtype(config)
    state_dict = {
        k: (v.to(save_dtype) if v.is_floating_point() else v)
        for k, v in _unwrap(model).state_dict().items()
    }
    guard_path = os.path.join(checkpoint_dir, "guard_model.pt")
    torch.save(state_dict, guard_path)

    if save_training_state:
        # Save optimizer
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(optimizer.state_dict(), optimizer_path)

        # Save scaler (for mixed precision)
        if scaler is not None:
            scaler_path = os.path.join(checkpoint_dir, "scaler.pt")
            torch.save(scaler.state_dict(), scaler_path)

        # Save LR scheduler (cosine-with-warmup schedule state so resume
        # continues the warmup/anneal from the right LR instead of restarting).
        if scheduler is not None:
            scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
            torch.save(scheduler.state_dict(), scheduler_path)

    # Save training state (metadata; cheap, always write)
    state = {
        'epoch': epoch,
        'step': step,
        'config': config.to_dict(),
        'has_training_state': save_training_state,
    }
    state_path = os.path.join(checkpoint_dir, "training_state.json")
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    extra = " (+ optimizer/scaler)" if save_training_state else " (guard only)"
    logger.info(f"✓ Checkpoint saved to {checkpoint_dir}{extra}")

    if _dist_enabled() and WORLD_SIZE > 1:
        dist.barrier()  # second rendezvous: others may now proceed


def load_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    logger: logging.Logger
) -> Dict:
    """Load checkpoint"""
    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    # Load Guard model
    guard_path = os.path.join(checkpoint_dir, "guard_model.pt")
    if os.path.exists(guard_path):
        model.load_state_dict(torch.load(guard_path, map_location='cpu'))
        logger.info("✓ Guard model loaded")

    # Load optimizer
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
        logger.info("✓ Optimizer loaded")

    # Load scaler
    if scaler is not None:
        scaler_path = os.path.join(checkpoint_dir, "scaler.pt")
        if os.path.exists(scaler_path):
            scaler.load_state_dict(torch.load(scaler_path, map_location='cpu'))
            logger.info("✓ Scaler loaded")

    # Load training state
    state_path = os.path.join(checkpoint_dir, "training_state.json")
    with open(state_path, 'r') as f:
        state = json.load(f)

    logger.info(f"✓ Checkpoint loaded (epoch {state['epoch']}, step {state['step']})")
    return state


def _eprint(msg: str) -> None:
    """Write one line to stderr, flushed. torchrun labels stderr per rank, so
    this is the only channel guaranteed to reach the console from a NON-main
    rank -- MainProcessLogger (setup_logging) gates the `logger` output to
    rank 0, so a non-main rank's logger.info/warning/error is dropped entirely
    (both file and console). Crash/diag messages must therefore go to stderr to
    be seen from any rank."""
    print(msg, file=sys.stderr, flush=True)


def _log_cuda_env_diagnostics(logger: logging.Logger,
                              device: torch.device,
                              local_rank: int) -> None:
    """Log the read-only CUDA/cuBLAS runtime + library state ONCE at startup.

    Captures version + LD_LIBRARY_PATH + whether the loader's libcublasLt exposes
    cublasLtGetVersion -- the answers the 'Cannot load symbol cublasLtGetVersion'
    warning raises but never prints. READ-ONLY by design: this runs NO CUDA op.
    An earlier version also ran a bf16 matmul probe here; that was REMOVED
    because issuing ANY cuBLASLt GEMM this early (before the guard trains) can
    load a kernel path that leaves the per-process cuBLASLt handle in a state
    where the subsequent bf16 SDPA (F.scaled_dot_product_attention) then SIGABRTs
    with exactly that symbol message (reproducibly isolated: bfloat16 SDPA in a
    fresh process as the FIRST cuBLASLt op survives; the same SDPA after a prior
    cuBLASLt GEMM aborts). So this function must touch no torch CUDA tensors.
    """
    import ctypes
    import ctypes.util
    _tag = f"[diag r{RANK}]"
    logger.info(f"RUNTIME / CUDA ENVIRONMENT DIAGNOSTICS (rank={RANK})")
    _eprint(f"{_tag} ---- RUNTIME / CUDA ENVIRONMENT DIAGNOSTICS ----")
    _eprint(f"{_tag} python    : {sys.version.split()[0]} ({sys.executable})")
    _eprint(f"{_tag} torch     : {torch.__version__} (built for CUDA {torch.version.cuda})")
    _eprint(f"{_tag} cuda avail: {torch.cuda.is_available()} "
            f"devices={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    try:
        _eprint(f"{_tag} cudnn     : {torch.backends.cudnn.version()}")
    except Exception as e:
        _eprint(f"{_tag} cudnn     : <unavailable: {e}>")
    if torch.cuda.is_available():
        try:
            _eprint(f"{_tag} gpu       : {torch.cuda.get_device_name(local_rank)} "
                    f"({torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB)")
        except Exception as e:
            _eprint(f"{_tag} gpu       : <name query failed: {e}>")
    # Which cuBLASLt the loader finds + whether the symbol torch fails on resolves.
    # This is the env-shape question the cublasLtGetVersion warning raises but
    # never answers. Read-only dlopen -- no CUDA kernel issued.
    cublas_lt_path = ctypes.util.find_library("cublasLt")
    sym_ok = None
    for cand in (cublas_lt_path, "libcublasLt.so", "libcublasLt.so.13", "libcublasLt.so.12"):
        if not cand:
            continue
        try:
            lib = ctypes.CDLL(cand)
            sym_ok = hasattr(lib, "cublasLtGetVersion")
            cublas_lt_path = cand
            break
        except OSError:
            continue
    found = cublas_lt_path if cublas_lt_path else "<not found by loader>"
    _eprint(f"{_tag} cublasLt  : loader={found}  cublasLtGetVersion resolves={sym_ok}")
    ldp_hits = [p for p in (os.environ.get("LD_LIBRARY_PATH", "") or "").split(":")
                if p and ("cublas" in p or "cuda" in p.lower())]
    _eprint(f"{_tag} LD_LIBRARY_PATH (cuda/cublas hits): {ldp_hits or '<none>'}")
    logger.info(f"[diag] cublasLt loader={found} cublasLtGetVersion resolves={sym_ok} "
                f"(rank={RANK}); no CUDA op issued here by design.")


def _configure_sdpa_backend(logger: logging.Logger) -> None:
    """Force SDPA off the cuDNN fused-attention backend (and onto flash).

    On Hopper (H20) with torch 2.12.1+cu132, `F.scaled_dot_product_attention`'s
    cuDNN fused-attention bf16 kernel SIGABRTs (`Cannot load symbol
    cublasLtGetVersion`, exit -6, no Python traceback -- a C++ kernel abort) when
    it consults the cuBLASLt handle at runtime. The enable_gqa=True / GQA path
    deterministically selects that cuDNN kernel (stable crash); plain bf16 SDPA
    selects cuDNN-vs-flash nondeterministically per shape/seed (intermittent
    crash). The flash attention backend does NOT hit that path and survives.

    So we disable the cuDNN SDPA backend process-wide and ensure flash is on
    (with mem-efficient / math as fallbacks), routing every SDPA call off the
    crashing cuDNN kernel while keeping bf16 precision + the enable_gqa layout.
    Done once at startup (global state) before any guard forward. Opt out with
    SINGGUARD_SDP_CUDNN=1 on hosts where the cuDNN backend is known-good.
    """
    if os.environ.get("SINGGUARD_SDP_CUDNN", "0") == "1":
        logger.info("SDPA: leaving cuDNN backend enabled (SINGGUARD_SDP_CUDNN=1).")
        return
    bcuda = torch.backends.cuda
    # Order matters only cosmetically; the operative change is disabling cudnn.
    tried = []
    for setter, on in (("enable_cudnn_sdp", False),
                       ("enable_flash_sdp", True),
                       ("enable_mem_efficient_sdp", True),
                       ("enable_math_sdp", True)):
        fn = getattr(bcuda, setter, None)
        if fn is None:
            continue
        try:
            fn(on)
            tried.append(f"{setter}({'on' if on else 'off'})")
        except Exception as e:
            logger.warning(f"SDPA: {setter}({'on' if on else 'off'}) failed: {e}")
    if tried:
        logger.info("SDPA backend configured to avoid the H20 cuDNN-fused-attention "
                    f"SIGABRT ({'; '.join(tried)}); flash/mem-efficient preferred. "
                    f"Set SINGGUARD_SDP_CUDNN=1 to leave cuDNN on.")


def warmup_cuda(
    base_model: BaseModelWrapper,
    guard_model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
    logger: logging.Logger,
    amp_dtype: torch.dtype = torch.bfloat16
):
    """
    Warmup CUDA kernels before training to avoid slow first batch.

    This performs a few forward passes to:
    1. Compile and cache CUDA kernels
    2. Initialize cuDNN autotuning
    3. Pre-allocate GPU memory

    Args:
        base_model: The frozen base model
        guard_model: The trainable guard model
        criterion: Loss function
        device: Device to use
        config: Configuration
        logger: Logger
        amp_dtype: AMP dtype for mixed precision
    """
    # On the SGLang path the "base model" is an external server process: its
    # kernels are already warmed at server startup, and a forward() here would
    # fire a real /generate prefill (with random token ids) and block waiting on
    # a dump file that has no reason to arrive on schedule. The guard model is a
    # tiny MLP whose first-call kernel-compile cost is negligible and is paid on
    # the first real training step. So skip the whole warmup on this path.
    if config.base_model.inference.framework.lower() == "sglang":
        logger.info("Skipping CUDA warmup: SGLang path (server self-warms; guard MLP is trivial).")
        return

    logger.info("Warming up CUDA kernels (base model + guard model)...")

    guard_model.train()
    base_model.eval()

    batch_size = config.training.batch_size
    # Short sequence is enough to trigger cuDNN autotuning / kernel compilation /
    # hook first-execution without paying the cost of a full-length forward.
    # max_seq_length<=0 is the "no truncation" sentinel (dynamic per-batch length),
    # so cap warmup at 512 instead of producing a zero-length dummy.
    seq_len = min(config.data.max_seq_length, 512) if config.data.max_seq_length > 0 else 512
    hidden_dim = config.guard_model.hidden_dim
    hidden_layers = config.base_model.hidden_layers

    # Dummy inputs for a real base-model forward (so its CUDA kernels, cuDNN
    # autotuning, and forward hooks all warm up -- the slowest first-time path).
    # Token IDs are arbitrary within the vocab range.
    vocab_size = base_model.get_vocab_size()
    dummy_input_ids = torch.randint(
        0, max(vocab_size, 1), (batch_size, seq_len), dtype=torch.long, device=device
    )
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    # Warmup forward + backward.
    # Each iteration: a REAL base-model forward produces hidden states via hooks,
    # which are then fed through the guard model + loss + backward. This warms
    # both the heavy base-model kernels and the guard backward path, matching
    # the real training step as closely as possible.
    for i in range(3):  # 3 warmup iterations
        # Base model forward (frozen, no grad) -- triggers hooks + cuDNN autotuning
        # on the HF path; on the SGLang path this exercises the HTTP client +
        # dump read pipeline (real kernels live in the separate server process).
        # SGLangHiddenStateClient.forward auto-generates synthetic rids when
        # sample_ids is None; BaseModelWrapper also accepts (ids, mask).
        with torch.no_grad():
            hidden_states_dict = base_model(dummy_input_ids, dummy_attention_mask)
            concat_hidden = torch.cat(
                [hidden_states_dict[idx] for idx in hidden_layers], dim=-1
            )

        # Guard forward + loss under the same AMP context the trainer uses.
        # Mirror train_epoch: autocast when use_amp (FP16 or BF16); no autocast
        # for the no-AMP path. FP16 GradScaler only affects backward grad-scaling,
        # which is not needed for a simple warmup forward+backward.
        use_autocast = bool(config.training.use_amp)
        fwd_ctx = autocast('cuda', dtype=amp_dtype) if use_autocast else nullcontext()

        labels = torch.zeros(batch_size, seq_len, 10, dtype=torch.long, device=device)
        response_mask = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        response_mask[:, seq_len // 2:] = 1  # Half is response

        with fwd_ctx:
            logits = guard_model(concat_hidden)
            losses = criterion(logits, labels, response_mask)
            loss = losses['total_loss']

        # Backward pass (warms guard backward kernels)
        loss.backward()
        guard_model.zero_grad()

    # Drop any hidden-state cache the warmup forward populated
    try:
        base_model._hidden_states_cache.clear()
    except Exception:
        pass

    # Synchronize to ensure all operations are complete before timing real steps
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info("CUDA warmup completed")


def _prefetch_dispatch_send(client, batch):
    """Pull batch N+1 from the dataloader row and dispatch its SGLang send async.

    Returns (future, jobs, seq_len) -- the in-flight send whose result is the
    rid dict, plus the jobs/seq_len needed later to collect+assemble. If `batch`
    is None (no next batch), returns (None, None, None). Safe on a background
    thread: dispatch_send_async fans the B /generate calls over the client's
    shared send pool. This is the overlap primitive: by the time the caller
    finishes training the CURRENT batch and joins this future, SGLang has
    prefilled the NEXT batch.
    """
    if batch is None:
        return None, None, None
    inp = batch['input_ids'].detach().to('cpu', dtype=torch.long)
    attn = batch['attention_mask'].detach().to('cpu', dtype=torch.long)
    seq_len = inp.shape[1]
    jobs = client._build_jobs(attn, inp)
    future = client.dispatch_send_async(jobs)
    return future, jobs, seq_len


def train_epoch(
    base_model: BaseModelWrapper,
    guard_model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    epoch: int,
    config: Config,
    logger: logging.Logger,
    amp_dtype: torch.dtype = torch.bfloat16,
    scheduler: Optional[LambdaLR] = None,
    # Step-level cadence context. global_step is the running optimizer-step
    # count coming into this epoch (continues across resume/epochs); val_loader
    # enables mid-epoch validation every eval_steps. best_val_loss is threaded
    # in and out so the best-model decision persists across epochs.
    global_step: int = 0,
    best_val_loss: float = float('inf'),
    val_loader: Optional[DataLoader] = None,
    output_dir: Optional[str] = None,
    total_epochs: int = 0,
) -> Dict:
    """Train for one epoch

    Mid-epoch cadence: at each optimizer step (accumulation boundary), trigger
    validation + best-model save every `eval_steps` steps, and a resumable
    periodic checkpoint every `save_steps` steps (both optimizer-step
    denominated). Validation runs on the single process (base_model forward);
    best/periodic saves are main-gated inside save_checkpoint.

    Returns:
        Dict with averaged per-microbatch losses, 'optimizer_steps' (number of
        actual optimizer updates performed this epoch, accounting for gradient
        accumulation), and threaded-back 'global_step' / 'best_val_loss'.
    """
    guard_model.train()
    base_model.eval()  # Base model is frozen

    # Gradient accumulation: accumulate gradients over N micro-batches before each
    # optimizer step. Loss is divided by N so the accumulated gradient equals the
    # average over the N micro-batches (equivalent to a N*batch_size effective batch).
    grad_accum = max(1, int(getattr(config.training, 'gradient_accumulation_steps', 1)))

    # Accumulate losses as detached GPU tensors to avoid per-step GPU->CPU syncs.
    # A single .item() pass is done at the end of the epoch.
    total_loss = torch.zeros((), device=device)
    total_query_loss = torch.zeros((), device=device)
    total_safety_loss = torch.zeros((), device=device)
    total_hallu_loss = torch.zeros((), device=device)
    num_batches = 0
    num_skipped_batches = 0   # micro-batches dropped by the NaN/Inf guard
    num_skipped_steps = 0     # optimizer steps dropped by the grad NaN/Inf guard
    optimizer_steps = 0

    hidden_layers = config.base_model.hidden_layers

    # Progress bar (only on main process for distributed training)
    if is_main_process():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    else:
        pbar = dataloader  # No progress bar for non-main processes

    # Refresh the postfix at most every N steps to limit GPU->CPU syncs from .item().
    postfix_every = 20
    # Stash the most recent detached losses and (raw) dynamic task weights; only
    # convert to float every postfix_every steps. Dynamic weights are populated by
    # the loss criterion's forward only when dynamic_task_weight is enabled; we
    # stash None per-step and fall back to fixed display placeholders otherwise.
    last = {'loss': None, 'q': None, 's': None, 'h': None,
            'wq': None, 'ws': None, 'wh': None}
    # Whether the criterion emits dynamic task weights into `losses`. Logged once
    # so the postfix helper can decide whether to show weights or a disabled hint.
    dyn_w_enabled = bool(config.training.dynamic_task_weight)

    # ---- Per-step profiling ----
    # Accumulate wall-clock time per phase across the rolling window, then log a
    # breakdown every `profile_every` micro-batches (rank 0 only). Each phase is
    # bracketed by a CUDA sync so GPU work (guard fwd/bwd) is measured correctly,
    # not just the kernel-launch time. Set profile_every=0 to disable.
    profile_every = config.training.profile_every or 0
    # The per-phase CUDA syncs below exist ONLY to give p_gfwd/p_gbwd/p_opt an
    # accurate wall-clock reading when profiling is on. When profiling is off
    # they are pure stalls: they block the CPU at GPU completion, delaying the
    # next batch's H2D copy / SGLang send dispatch and destroying CPU/GPU
    # overlap for no benefit. Gate them behind profiling. Correctness is not
    # affected -- the NaN/Inf guards force their own sync via `.item()` on the
    # finite checks, and optimizer.step() is host-blocked by the param updates.
    profile_active = profile_every > 0
    p_total = p_base = p_concat = p_gfwd = p_gbwd = p_opt = 0.0
    # sub-phase of the base-model call (SGLang path only): send /generate (which
    # blocks on SGLang prefill+decode) vs read dump files vs assemble+H2D.
    p_send = p_read = p_asm = 0.0

    num_micro = len(dataloader)

    # ---- SGLang send-prefetch pipeline ----
    # The SGLang /generate send *is* the SGLang prefill+decode latency (it blocks
    # until the dump is triggered) and dominates the step (~70%). We overlap it:
    # while the GPU trains batch N (read+asm+guard fwd/bwd), a background thread
    # dispatches batch N+1's /generate so SGLang prefills N+1 in parallel. We
    # stay one batch ahead. Only the SGLang path uses this; device_map keeps the
    # synchronous base_model() call.
    use_sglang_prefetch = isinstance(base_model, SGLangHiddenStateClient)
    _pf_future = None      # in-flight dispatch_send for batch N+1
    # Flight recorder (SINGGUARD_TRACE=1): logs a START line per micro-batch to
    # stderr (per-rank) so the LAST printed line tells us exactly which batch the
    # rank stopped at -- invaluable when a CUDA/C++ abort or an SGLang dump-read
    # hang prints nothing (no Python traceback reaches the console). Default off
    # (zero log noise on a healthy long run).
    _trace = os.environ.get("SINGGUARD_TRACE", "") == "1"
    _pf_jobs = None        # jobs handed to that send
    _pf_seq_len = None     # seq_len of batch N+1 (for assemble later)
    _pf_batch = None       # batch N+1's raw dict, pulled one ahead
    _data_iter = iter(dataloader) if use_sglang_prefetch else None
    if use_sglang_prefetch:
        # Prime: pull batch 0 and dispatch its send before the loop body runs.
        # The first iteration joins this future at the top.
        _pf_batch = next(_data_iter, None)
        _pf_future, _pf_jobs, _pf_seq_len = _prefetch_dispatch_send(base_model, _pf_batch)

    # Single loop body, driven by `_pf_batch` on the SGLang path (so the next
    # batch's send can be dispatched before we train this one) and by the plain
    # enumerate(pbar) on the device_map path. `batch`/`batch_idx` are set below.
    if use_sglang_prefetch:
        batch_idx = -1
    else:
        _pbar_iter = iter(enumerate(pbar))
    while True:
        if use_sglang_prefetch:
            if _pf_batch is None:
                break
            batch_idx += 1
            batch = _pf_batch
            if is_main_process() and hasattr(pbar, 'update'):
                pbar.update(1)  # advance tqdm; we don't iterate pbar on this path
            # Join batch N's send -> rids; pull+dispatch batch N+1's send NOW so
            # SGLang inference of N+1 overlaps with the guard training below.
            with torch.no_grad():
                _t0 = time.perf_counter()
                rids = _pf_future.result()
                jobs_cur = _pf_jobs
                seq_len_cur = _pf_seq_len
                _pf_batch = next(_data_iter, None)
                _pf_future, _pf_jobs, _pf_seq_len = _prefetch_dispatch_send(base_model, _pf_batch)
                # The CPU-side now hands off to GPU; the next send runs concurrently.
            _pf_rids = rids
            _pf_jobs_cur = jobs_cur
            _pf_seq_cur = seq_len_cur
        else:
            try:
                batch_idx, batch = next(_pbar_iter)
            except StopIteration:
                break
        if _trace:
            _eprint(
                f"[trace r{RANK}] START micro-batch batch_idx={batch_idx} "
                f"global_step={global_step + optimizer_steps} "
                f"opt_steps={optimizer_steps} num_batches={num_batches}"
            )
        # Move to device (non_blocking overlaps H2D copy with the previous step's compute
        # since pin_memory is enabled in the DataLoader).
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        response_mask = batch['response_mask'].to(device, non_blocking=True)
        # Hallu-only POS content-word mask (None-absent safe; the loss gates it by
        # use_hallu_assertion_mask). Right-aligned like response_mask by the collator.
        hallu_assertion_mask = batch.get('hallu_assertion_mask')
        if hallu_assertion_mask is not None:
            hallu_assertion_mask = hallu_assertion_mask.to(device, non_blocking=True)
        # sample_ids -> SGLang rids (ignored by the HF BaseModelWrapper path).
        batch_sample_ids = batch.get('sample_id')

        # Debug: Check batch composition on first batch of first epoch
        if epoch == 0 and batch_idx == 0:
            batch_size = input_ids.shape[0]
            has_response = (response_mask.sum(dim=1) > 0).sum().item()
            query_valid = ((labels[:, :, 0] != -100) & (response_mask == 1)).sum().item()
            safety_valid = ((labels[:, :, 8] != -100) & (response_mask == 1)).sum().item()
            hallu_valid = ((labels[:, :, 9] != -100) & (response_mask == 1)).sum().item()
            dataset_types = batch.get('dataset_type', ['unknown'] * batch_size)

            logger.info(f"="*60)
            logger.info(f"DEBUG: Batch composition analysis")
            logger.info(f"="*60)
            logger.info(f"Batch size: {batch_size}")
            logger.info(
                f"Gradient accumulation steps: {grad_accum} "
                f"(per-rank batch {batch_size * grad_accum}, "
                f"global effective batch {batch_size * grad_accum * WORLD_SIZE} "
                f"= batch_size {batch_size} x grad_accum {grad_accum} x world_size {WORLD_SIZE})"
            )
            logger.info(f"Dataset types in batch: {dataset_types}")
            logger.info(f"Samples with response: {has_response}/{batch_size}")
            logger.info(f"Response mask sum per sample: {response_mask.sum(dim=1).tolist()}")
            logger.info(f"")
            logger.info(f"Valid token counts:")
            logger.info(f"  Query (response tokens): {query_valid}")
            logger.info(f"  Safety (response tokens): {safety_valid}")
            logger.info(f"  Hallucination (response tokens): {hallu_valid}")
            logger.info(f"")

            # Check label distribution
            safety_labels = labels[:, :, 8][labels[:, :, 8] != -100]
            hallu_labels = labels[:, :, 9][labels[:, :, 9] != -100]

            if len(safety_labels) > 0:
                logger.info(f"Safety label distribution: 0={(safety_labels==0).sum().item()}, 1={(safety_labels==1).sum().item()}")
            if len(hallu_labels) > 0:
                logger.info(f"Hallu label distribution: 0={(hallu_labels==0).sum().item()}, 1={(hallu_labels==1).sum().item()}")
            logger.info(f"="*60)

        _t_step = time.perf_counter()

        # Step 1: Base Model inference (frozen) -> concatenated hidden states.
        if use_sglang_prefetch:
            # Prefetch path: this batch's /generate send was dispatched last
            # iteration (and has been running concurrently with the previous
            # batch's guard training). `_pf_rids` are its results; we now read
            # the dump files + assemble. last_send_secs was already recorded by
            # the dispatcher -- note it measures dispatch-to-join wall time, so
            # under prefetch it INCLUDES the time it overlapped the previous
            # guard step (i.e. the *exposed* send cost is max(0, send - guard)).
            _t0 = time.perf_counter()
            with torch.no_grad():
                hidden_states_dict = base_model.collect_and_assemble(
                    _pf_rids, _pf_jobs_cur, _pf_seq_cur
                )
                _send = getattr(base_model, 'last_send_secs', 0.0)
                _read = getattr(base_model, 'last_read_secs', 0.0)
                _asm = getattr(base_model, 'last_assemble_secs', 0.0)
                _tc = time.perf_counter()
                concat_hidden = torch.cat(
                    [hidden_states_dict[layer_idx] for layer_idx in hidden_layers],
                    dim=-1,
                )
            p_base += (time.perf_counter() - _t0)
            p_send += _send
            p_read += _read
            p_asm += _asm
            p_concat += (time.perf_counter() - _tc)
        else:
            # Sync path: BaseModelWrapper(device_map) or synchronous SGLang fallback.
            _t0 = time.perf_counter()
            with torch.no_grad():
                hidden_states_dict = base_model(
                    input_ids, attention_mask, sample_ids=batch_sample_ids
                )
                _send = getattr(base_model, 'last_send_secs', 0.0)
                _read = getattr(base_model, 'last_read_secs', 0.0)
                _asm = getattr(base_model, 'last_assemble_secs', 0.0)
                _tc = time.perf_counter()
                concat_hidden = torch.cat(
                    [hidden_states_dict[layer_idx] for layer_idx in hidden_layers],
                    dim=-1,
                )
            p_base += (time.perf_counter() - _t0)
            p_send += _send
            p_read += _read
            p_asm += _asm
            p_concat += (time.perf_counter() - _tc)

        # Step 2: Guard Model forward + backward (accumulate gradients).
        # Scale loss by 1/grad_accum so summed gradients average over micro-batches.
        # AMP path is unified here: FP16 uses GradScaler + float16 autocast;
        # BF16 uses autocast with no scaler; no-AMP runs in raw dtype.
        # - When using FP16 scaler, autocast dtype is float16.
        # - When using AMP (BF16), autocast dtype is amp_dtype.
        # - Otherwise (no AMP), no autocast wrapper.
        if scaler is not None:
            autocast_dtype = torch.float16
            use_autocast = True
        elif config.training.use_amp:
            autocast_dtype = amp_dtype
            use_autocast = True
        else:
            use_autocast = False

        # DDP gradient accumulation: all-reduce only on the LAST micro-batch of an
        # accumulation window (the boundary). On non-boundary micro-batches wrap
        # forward+backward in model.no_sync() so the reduction is deferred and we
        # pay one all-reduce per optimizer step instead of one per micro-batch.
        # is_accum_boundary/is_last_microbatch are recomputed after the backward
        # (they only depend on batch_idx), so we can compute them once here.
        is_accum_boundary = (batch_idx + 1) % grad_accum == 0
        is_last_microbatch = batch_idx + 1 == num_micro
        should_step = is_accum_boundary or is_last_microbatch
        sync_ctx = (
            nullcontext()
            if should_step
            else guard_model.no_sync()
            if isinstance(guard_model, DDP)
            else nullcontext()
        )

        fwd_ctx = autocast('cuda', dtype=autocast_dtype) if use_autocast else nullcontext()
        _tgf = time.perf_counter()
        with sync_ctx, fwd_ctx:
            logits = guard_model(concat_hidden)
            losses = criterion(logits, labels, response_mask,
                               hallu_assertion_mask=hallu_assertion_mask)
            loss = losses['total_loss'] / grad_accum
        if profile_active and torch.cuda.is_available():
            torch.cuda.synchronize()
        p_gfwd += (time.perf_counter() - _tgf)

        # ===== NaN/Inf guard A: skip a non-finite loss BEFORE backward =====
        # A single non-finite loss (e.g. a long-sequence batch that triggers the
        # BCE numerical cliff) must never reach backward -- it would produce
        # inf grads that clip_grad_norm_ can't fix and that contaminate the
        # accumulated gradients for the whole grad-accum window. Drop the
        # micro-batch entirely: no backward, no loss accumulation, no step.
        # This decision is identical on every rank (all-reduce ANY), so DDP
        # collective ordering stays consistent (no rank backprops while another
        # skips -- both branches are step-free here, but we keep the all-reduce
        # for the should_step path below and to log an accurate global count).
        loss_finite = torch.tensor(
            float(torch.isfinite(loss).all().item()), device=device
        )
        loss_finite = _all_reduce_any(loss_finite)
        if loss_finite.item() == 0.0:
            num_skipped_batches += 1
            if is_main_process():
                logger.warning(
                    f"[step {global_step + optimizer_steps}] Non-finite loss "
                    f"(total={float(loss.detach().item() * grad_accum)}) "
                    f"-- skipping micro-batch {batch_idx} (no backward). "
                    f"Skipped so far this epoch: {num_skipped_batches}."
                )
            # No backward ran for this micro-batch, so the grads accumulated from
            # earlier valid micro-batches in this window are untouched; the next
            # should_step boundary will step on the mean over the survivors.
            continue

        _tgb = time.perf_counter()
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if profile_active and torch.cuda.is_available():
            torch.cuda.synchronize()
        p_gbwd += (time.perf_counter() - _tgb)

        # Accumulate the UN-averaged (per-microbatch) loss for reporting.
        # loss here was divided by grad_accum, so multiply back to get per-microbatch value.
        total_loss = total_loss + loss.detach() * grad_accum
        total_query_loss = total_query_loss + losses['query_loss'].detach()
        total_safety_loss = total_safety_loss + losses['safety_loss'].detach()
        total_hallu_loss = total_hallu_loss + losses['hallucination_loss'].detach()
        num_batches += 1

        # Step 3: Optimizer step -- only at accumulation boundaries, OR at the end of
        # the epoch if a partial accumulation remains (so no gradient is wasted).

        if should_step:
            _topt = time.perf_counter()
            # FP16 path needs unscale before clip; BF16/no-AMP grads are already
            # real. unscale_ is idempotent-safe on the FP16 path; the no-scaler
            # path skips it (grads are already in true scale).
            if scaler is not None:
                scaler.unscale_(optimizer)

            # ===== NaN/Inf guard B: skip the step on non-finite grads =====
            # Guard A catches a non-finite LOSS, but grads can still go inf/nan
            # from FP16 overflow/underflow even with a finite loss. Stepping with
            # them would poison the params irreversibly (clip_grad_norm_ on an inf
            # norm yields garbage). So: detect non-finite grads over ALL params,
            # all-reduce ANY so every rank agrees, and if bad -> zero_grad +
            # DROP the step (no increment, no eval/save cadence). Optimizer state
            # (momentum) is left untouched because we never stepped. This wastes
            # one grad-accum window of compute but keeps training alive.
            grad_bad = torch.tensor(0.0, device=device)
            for p in guard_model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_bad.fill_(1.0)
                    break
            grad_bad = _all_reduce_any(grad_bad)
            step_ok = grad_bad.item() == 0.0

            if not step_ok:
                num_skipped_steps += 1
                if is_main_process():
                    logger.warning(
                        f"[step {global_step + optimizer_steps}] Non-finite "
                        f"gradients -- skipping optimizer step (zeroing grads, "
                        f"optimizer state untouched). Skipped this epoch: "
                        f"{num_skipped_steps}."
                    )
                if scaler is not None:
                    scaler.update()   # acknowledge the skipped step (keeps scale sane)
                optimizer.zero_grad(set_to_none=True)
                if profile_active and torch.cuda.is_available():
                    torch.cuda.synchronize()
                p_opt += (time.perf_counter() - _topt)
            else:
                torch.nn.utils.clip_grad_norm_(
                    guard_model.parameters(),
                    config.training.max_grad_norm
                )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if profile_active and torch.cuda.is_available():
                    torch.cuda.synchronize()
                p_opt += (time.perf_counter() - _topt)
                optimizer_steps += 1

                # Advance the LR schedule on every REAL optimizer step (skipped
                # steps advance no progress, matching the cosine horizon). A None
                # scheduler means constant lr -- no step to take.
                if scheduler is not None:
                    scheduler.step()

            # ----- Step-level cadence (eval + best-save + periodic checkpoint) -----
            # Runs only on a REAL optimizer step (skipped steps advance no progress).
            if step_ok:
                # cur_step is the absolute optimizer-step count (continuous across
                # epochs/resume). Action order: eval-if-due -> best-save-if-improved
                # -> periodic-save-if-due. Eval runs first so the best decision uses
                # the fresh val result.
                cur_step = global_step + optimizer_steps

                eval_steps = config.training.eval_steps
                if eval_steps > 0 and val_loader is not None and cur_step % eval_steps == 0:
                    val_metrics = validate(
                        base_model, guard_model, val_loader, criterion,
                        device, config, logger
                    )
                    # validate() sets both models to eval; restore training mode.
                    guard_model.train()
                    base_model.eval()

                    if is_main_process():
                        logger.info(
                            f"[step {cur_step}] Val - "
                            f"Loss: {val_metrics['loss']:.4f} | "
                            f"Query: {val_metrics['query_loss']:.4f} | "
                            f"Safety: {val_metrics['safety_loss']:.4f} | "
                            f"Hallu: {val_metrics['hallucination_loss']:.4f}"
                        )

                    # Best-save decision is globally consistent (validate all-reduced
                    # the val loss), but save_checkpoint() carries a DDP barrier pair,
                    # so ALL ranks must call it -- otherwise rank 0 deadlocks on the
                    # entry barrier while the replicas skip (this was the bug behind
                    # train_test.sh not exiting). Rank 0 writes; the others pass
                    # through; logging stays rank-0-only.
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        save_checkpoint(
                            guard_model, optimizer, scaler, scheduler, epoch, cur_step,
                            os.path.join(output_dir, "best"), config, logger,
                            save_training_state=False,
                        )
                        if is_main_process():
                            logger.info(f"[step {cur_step}] ✓ Best model saved (val_loss: {best_val_loss:.4f})")

                save_steps = config.training.save_steps
                if save_steps > 0 and output_dir is not None and cur_step % save_steps == 0:
                    save_checkpoint(
                        guard_model, optimizer, scaler, scheduler, epoch, cur_step,
                        output_dir, config, logger,
                        save_training_state=True,  # full state for resume
                    )

        # Close out this micro-batch's total wall time.
        p_total += (time.perf_counter() - _t_step)

        # ---- rolling per-step profiling breakdown (rank 0 only) ----
        # Prints averaged seconds/micro-batch for each phase over the last
        # `profile_every` micro-batches, so you can see whether the bottleneck
        # is SGLang inference (base send), dump-file reads, assemble+H2D, or the
        # Guard forward/backward. Note p_send already includes SGLang's prefill
        # + 1-token decode latency (the /generate call blocks until the request
        # finishes and the dump is triggered).
        if profile_every and is_main_process() and (batch_idx + 1) % profile_every == 0:
            n = profile_every
            # `base` = whole base_model step on this path. On the SGLang path it
            # splits into send(/generate, which under PREFETCH measures the FULL
            # SGLang inference duration -- most of which runs concurrently with
            # the PREVIOUS batch's guard training), read (dump I/O), asm (pad+H2D).
            # The non_base serial cost is concat+gfwd+gbwd+opt -- the part that
            # send can be hidden behind.
            non_base = p_concat + p_gfwd + p_gbwd + p_opt
            logger.info(
                f"[prof] mb {batch_idx + 1 - n}-{batch_idx + 1} "
                f"avg ms/micro-batch | "
                f"total={p_total / n * 1e3:.1f} "
                f"base={p_base / n * 1e3:.1f}"
                f" [send={p_send / n * 1e3:.1f} read={p_read / n * 1e3:.1f} "
                f"asm={p_asm / n * 1e3:.1f}] "
                f"concat={p_concat / n * 1e3:.1f} "
                f"gfwd={p_gfwd / n * 1e3:.1f} "
                f"gbwd={p_gbwd / n * 1e3:.1f} "
                f"opt={p_opt / n * 1e3:.1f}"
            )
            # Under SGLang send-prefetch, total ideally approaches
            # max(send, non_base) -- i.e. send is hidden behind the guard work.
            # overlap_hiding = how much of send runs in parallel with guard work.
            if use_sglang_prefetch and p_total > 0:
                hidden = max(0.0, p_send - max(0.0, p_total - non_base))
                ideal = max(p_send, non_base)
                logger.info(
                    f"[prof]   prefetch: send={p_send / n * 1e3:.1f}ms "
                    f"hidden_behind_guard~{hidden / n * 1e3:.1f}ms | "
                    f"serial_after_overlap~{max(0.0, p_total - hidden) / n * 1e3:.1f}ms "
                    f"(ideal lower bound max(send,guard_work)={ideal / n * 1e3:.1f}ms)"
                )
            else:
                # Non-prefetch: which single disjoint phase dominates total.
                phases = {
                    'base-send': p_send, 'base-read': p_read, 'base-asm': p_asm,
                    'gfwd': p_gfwd, 'gbwd': p_gbwd, 'opt': p_opt,
                }
                top = max(phases, key=phases.get)
                if p_total > 0:
                    logger.info(
                        f"[prof]   -> bottleneck: {top} "
                        f"({phases[top] / p_total * 100:.0f}% of step time)"
                    )
            p_total = p_base = p_concat = p_gfwd = p_gbwd = p_opt = 0.0
            p_send = p_read = p_asm = 0.0

        # Update progress bar at a limited cadence to avoid per-step GPU syncs.
        if is_main_process():
            last['loss'] = loss.detach() * grad_accum  # report per-microbatch loss
            last['q'] = losses['query_loss'].detach()
            last['s'] = losses['safety_loss'].detach()
            last['h'] = losses['hallucination_loss'].detach()
            # Stash the raw dynamic weights from this micro-batch's loss forward,
            # if the criterion emitted them (only when dynamic_task_weight=True).
            last['wq'] = losses.get('query_weight')
            last['ws'] = losses.get('safety_weight')
            last['wh'] = losses.get('hallu_weight')
            if (batch_idx + 1) % postfix_every == 0 and last['loss'] is not None:
                # Single sync point per postfix_every steps
                postfix = {
                    'loss': f'{last["loss"].item():.4f}',
                    'q': f'{last["q"].item():.3f}',
                    's': f'{last["s"].item():.3f}',
                    'h': f'{last["h"].item():.3f}'
                }
                if dyn_w_enabled:
                    wq = _fnum(last['wq'])
                    ws = _fnum(last['ws'])
                    wh = _fnum(last['wh'])
                    postfix['w:q/s/h'] = (
                        f'{wq:.2f}/{ws:.2f}/{wh:.2f}'
                        if wq is not None else 'n/a'
                    )
                pbar.set_postfix(postfix)

    # Single GPU->CPU sync for the whole epoch's accumulated losses.
    # Under DDP, all-reduce the loss sums and the batch count so every rank
    # reports the same globally-averaged metric (the per-microbatch losses above
    # are already 1/grad_accum-scaled the same way on each rank, so a sum-reduce
    # over ranks then /world_size gives the global mean).
    # IMPORTANT: every rank MUST enter the all_reduce calls (even ranks whose
    # local shard is empty -> zero contributions), else the non-empty ranks'
    # all_reduce would hang.
    num_batches_t = torch.tensor(float(num_batches), device=device)
    _all_reduce_sum(total_loss)
    _all_reduce_sum(total_query_loss)
    _all_reduce_sum(total_safety_loss)
    _all_reduce_sum(total_hallu_loss)
    _all_reduce_sum(num_batches_t)
    num_batches_g = num_batches_t.item()
    if num_batches_g == 0:
        # Every rank's shard was empty (or no train data) -> global empty.
        return {
            'loss': 0.0,
            'query_loss': 0.0,
            'safety_loss': 0.0,
            'hallucination_loss': 0.0,
            'optimizer_steps': 0,
            'skipped_batches': num_skipped_batches,
            'skipped_steps': num_skipped_steps,
            'global_step': global_step,
            'best_val_loss': best_val_loss,
        }
    return {
        'loss': total_loss.item() / num_batches_g,
        'query_loss': total_query_loss.item() / num_batches_g,
        'safety_loss': total_safety_loss.item() / num_batches_g,
        'hallucination_loss': total_hallu_loss.item() / num_batches_g,
        'optimizer_steps': optimizer_steps,
        'skipped_batches': num_skipped_batches,
        'skipped_steps': num_skipped_steps,
        'global_step': global_step + optimizer_steps,
        'best_val_loss': best_val_loss,
    }


def validate(
    base_model: BaseModelWrapper,
    guard_model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
    logger: logging.Logger
) -> Dict:
    """Validation"""
    guard_model.eval()
    base_model.eval()

    # Accumulate on-GPU to avoid per-step GPU->CPU syncs.
    total_loss = torch.zeros((), device=device)
    total_query_loss = torch.zeros((), device=device)
    total_safety_loss = torch.zeros((), device=device)
    total_hallu_loss = torch.zeros((), device=device)
    num_batches = 0

    hidden_layers = config.base_model.hidden_layers

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device (non_blocking overlaps with compute thanks to pin_memory)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            response_mask = batch['response_mask'].to(device, non_blocking=True)
            hallu_assertion_mask = batch.get('hallu_assertion_mask')
            if hallu_assertion_mask is not None:
                hallu_assertion_mask = hallu_assertion_mask.to(device, non_blocking=True)
            batch_sample_ids = batch.get('sample_id')

            # Base Model inference
            hidden_states_dict = base_model(
                input_ids, attention_mask, sample_ids=batch_sample_ids
            )

            # Concatenate hidden states
            hidden_states_list = [
                hidden_states_dict[layer_idx]
                for layer_idx in hidden_layers
            ]
            concat_hidden = torch.cat(hidden_states_list, dim=-1)

            # Guard Model forward pass under the same AMP context train_epoch
            # uses. Guard params are fp32 master weights; without autocast the
            # forward would run in raw fp32 and val losses would be computed on
            # different numerics than training / SGLang serving (all bf16).
            fwd_ctx = (
                autocast('cuda', dtype=_base_compute_dtype(config))
                if config.training.use_amp else nullcontext()
            )
            with fwd_ctx:
                logits = guard_model(concat_hidden)
                losses = criterion(logits, labels, response_mask,
                                   hallu_assertion_mask=hallu_assertion_mask)

            # Accumulate losses (detached)
            total_loss = total_loss + losses['total_loss'].detach()
            total_query_loss = total_query_loss + losses['query_loss'].detach()
            total_safety_loss = total_safety_loss + losses['safety_loss'].detach()
            total_hallu_loss = total_hallu_loss + losses['hallucination_loss'].detach()
            num_batches += 1

    # Globally average the validation loss across ranks so every rank makes the
    # same best-model decision. IMPORTANT: every rank MUST enter the all_reduce
    # calls (even ranks whose local val shard is empty -> zero contributions),
    # else the non-empty ranks' all_reduce would hang. sum(loss)/sum(num_batches)
    # is robust to unequal per-rank batch counts from DistributedSampler sharding.
    num_batches_t = torch.tensor(float(num_batches), device=device)
    _all_reduce_sum(total_loss)
    _all_reduce_sum(total_query_loss)
    _all_reduce_sum(total_safety_loss)
    _all_reduce_sum(total_hallu_loss)
    _all_reduce_sum(num_batches_t)
    num_batches_g = num_batches_t.item()
    if num_batches_g == 0:
        # Every rank's val shard was empty (or no val data) -> global empty.
        return {
            'loss': 0.0,
            'query_loss': 0.0,
            'safety_loss': 0.0,
            'hallucination_loss': 0.0
        }
    return {
        'loss': total_loss.item() / num_batches_g,
        'query_loss': total_query_loss.item() / num_batches_g,
        'safety_loss': total_safety_loss.item() / num_batches_g,
        'hallucination_loss': total_hallu_loss.item() / num_batches_g
    }


def main():
    # Arm crash/hang diagnostics FIRST, before any CUDA/torch work. A rank that
    # ABORTs at the CUDA/C++ level (e.g. a bad cuBLASLt handle) prints only the
    # cryptic torch symbol line and exits with no Python traceback; faulthandler
    # dumps a Python trace on SIGSEGV/SIGFPE/SIGABRT so the dying rank shows
    # where. Opt-in SINGGUARD_HANG_TIMEOUT=<seconds> additionally dumps an
    # all-thread trace every N s to catch a rank that STALLS (e.g. blocked on an
    # SGLang dump file that never lands) and would otherwise print nothing.
    import faulthandler
    try:
        faulthandler.enable()
        _hang_to = os.environ.get("SINGGUARD_HANG_TIMEOUT", "")
        if _hang_to:
            faulthandler.dump_traceback_later(int(_hang_to), repeat=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="SingProbe End-to-End Training")
    parser.add_argument('--config', type=str, default='configs/all_models/ling-3.0-flash.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Resolve torchrun env vars (LOCAL_RANK/WORLD_SIZE/RANK) into module state.
    # When launched plain (no torchrun) these default to 1/0/0 -> single process.
    _resolve_distributed_env()

    # Initialize the NCCL process group before any CUDA work so each rank binds
    # to its own GPU. torchrun sets MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE.
    if WORLD_SIZE > 1 and not _dist_enabled():
        dist.init_process_group(backend=config.distributed.backend or "nccl")

    # Setup logging (rank-aware: per-rank log files, rank-0 console gating)
    logger = setup_logging(config.training.output_dir, args.log_level)
    logger.info("="*80)
    logger.info("SingProbe End-to-End Training")
    logger.info("="*80)
    if WORLD_SIZE > 1:
        logger.info(f"DDP: world_size={WORLD_SIZE}, rank={RANK}, local_rank={LOCAL_RANK}")

    # Set seed (same seed on every rank keeps dropout/init deterministic and
    # makes the all-reduced loss comparison apples-to-apples)
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Enable cuDNN benchmark for faster training
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark enabled for faster training")

    # Log config
    logger.info(f"\nConfiguration:\n{config}")

    # Setup device: under DDP each rank uses cuda:LOCAL_RANK; single-process
    # keeps cuda:0 as before.
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device(f"cuda:{LOCAL_RANK}")
    else:
        device = torch.device("cpu")
    logger.info(f"\nDevice: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(LOCAL_RANK)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(LOCAL_RANK).total_memory / 1e9:.2f} GB")

    # Configure the SDPA backend BEFORE any guard forward: on Hopper (H20) with
    # torch 2.12.1+cu132, `F.scaled_dot_product_attention`'s cuDNN fused-attention
    # bf16 kernel SIGABRTs (`Cannot load symbol cublasLtGetVersion`) when it
    # consults the cuBLASLt handle at runtime (the enable_gqa=True / GQA path
    # deterministically selects that kernel -> stable crash; plain bf16 SDPA
    # selects cuDNN-vs-flash nondeterministically -> intermittent crash). The
    # flash attention backend does NOT hit that path, so forcing SDPA off cuDNN
    # and onto flash (with mem-efficient / math as fallbacks) sidesteps the
    # abort while keeping bf16 precision and the enable_gqa=True GQA layout.
    # Opt out with SINGGUARD_SDP_CUDNN=1 (a known-good host that never aborts).
    _configure_sdpa_backend(logger)

    # Capture the CUDA/cuBLAS env + an isolated cuBLASLt probe before training so
    # the 'cublasLtGetVersion' symbol-line mystery is answered on THIS host (can
    # the loader find cublasLt? does the symbol resolve? does a bare bf16 matmul
    # reproduce/crash?). See _log_cuda_env_diagnostics for why this is an env, not
    # a code, issue.
    _log_cuda_env_diagnostics(logger, device, LOCAL_RANK)

    # Initialize Base Model
    logger.info("\n" + "="*80)
    logger.info("Initializing Base Model")
    logger.info("="*80)

    # Base-Model + autocast compute dtype. The Guard's parameters themselves
    # stay fp32 (master weights) -- see the Guard init below.
    guard_dtype = _base_compute_dtype(config)

    if config.base_model.inference.framework.lower() == "sglang":
        # External SGLang server extracts hidden states; no HF model on the
        # training GPUs. Each rank builds its own HTTP client (own connection
        # pool + p{pid}-prefixed rid namespace -> no cross-rank dump collisions).
        base_model = SGLangHiddenStateClient(
            model_name=config.base_model.name,
            hidden_layers=config.base_model.hidden_layers,
            url=config.base_model.inference.sglang_url,
            save_dir=config.base_model.inference.sglang_save_dir,
            probe_ckpt=config.base_model.inference.sglang_probe_ckpt,
            timeout=config.base_model.inference.sglang_timeout,
            max_concurrency=config.base_model.inference.sglang_max_concurrency,
            dtype=guard_dtype,
            device=device,
        )
    else:
        # device_map path: single-process, model sharded across all GPUs.
        # DDP is not supported here (the HF model already occupies every GPU).
        if WORLD_SIZE > 1:
            raise RuntimeError(
                "DDP is not supported on the device_map base-model backend "
                "(the HF model already shards across all GPUs in one process). "
                "Use the 'sglang' framework for DDP training of the Guard."
            )
        base_model = BaseModelWrapper(
            model_name=config.base_model.name,
            hidden_layers=config.base_model.hidden_layers,
            shard_gpus=config.base_model.inference.shard_gpus,
            dtype=guard_dtype,
            kernel_inject=config.base_model.inference.kernel_inject,
            load_strategy=_resolve_load_strategy(config.base_model.inference.framework),
        )

    # Initialize Guard Model
    logger.info("\n" + "="*80)
    logger.info("Initializing Guard Model")
    logger.info("="*80)

    # Auto-detect hidden_dim and num_layers from Base Model
    hidden_dim = base_model.get_hidden_dim()
    num_layers = len(config.base_model.hidden_layers)

    logger.info(f"Auto-detected from Base Model:")
    logger.info(f"  hidden_dim: {hidden_dim}")
    logger.info(f"  num_layers: {num_layers} (from base_model.hidden_layers: {config.base_model.hidden_layers})")

    # Update config with auto-detected values
    config.guard_model.set_hidden_dim(hidden_dim)
    config.guard_model.set_num_layers(num_layers)
    logger.info(f"  input_dim: {config.guard_model.input_dim} = {hidden_dim} × {num_layers}")

    arch = config.guard_model.arch
    if arch == "mlp":
        guard_config = GuardMLPConfig(
            input_dim=config.guard_model.input_dim,
            intermediate_dim=config.guard_model.intermediate_dim,
            num_classes=config.guard_model.num_classes,
            dropout=config.guard_model.dropout,
            activation=config.guard_model.activation,
            init_bias=config.guard_model.init_bias,
        )
        guard_model = GuardMLP(**guard_config.to_dict())
    elif arch == "attn":
        # "attn" token-probe Guard (GuardAttnProbe, models/sglang_attn.py):
        # the tapped+normed layer features are projected by separate
        # proj_q/proj_k/proj_v (one shared KV head = MQA), run through causal
        # multi-query attention along the token sequence, then an attention
        # output projection (o_proj), an additive query residual (h = q + o),
        # a post-residual RMSNorm, and a per-token linear classifier. Only
        # num_query_heads/head_dim/sliding_window matter. NO input
        # normalization (the SGLang dump path already rms-norms
        # (hidden+residual) per tapped layer).
        from models.sglang_attn import GuardAttnProbe
        guard_model = GuardAttnProbe(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=config.guard_model.num_classes,
            num_query_heads=config.guard_model.num_query_heads,
            head_dim=config.guard_model.head_dim,
            sliding_window=config.guard_model.sliding_window,
            init_bias=config.guard_model.init_bias,
        )
    else:
        raise ValueError(
            f"Unknown guard_model.arch='{arch}'. Supported: 'mlp', 'attn'."
        )

    # Guard parameters stay FLOAT32 (fp32 master weights). Forward/backward
    # compute still runs in guard_dtype (bf16/fp16) via the autocast wrappers
    # in train_epoch/validate, and every Guard arch aligns the bf16 SGLang
    # features to the param dtype itself via x.to(param_dtype) (a lossless
    # bf16->fp32 promotion), so nothing downstream changes.
    #
    # Why NOT bf16 params: storing params in the compute dtype pins any weight
    # of magnitude ~1 at its init value. bf16's ulp at 1.0 is 2**-7 (~0.008)
    # while an AdamW update is ~lr (1e-3), so every in-place update rounds
    # away: the attn probe's RMSNorm gains (nn.RMSNorm init = all-ones) never
    # train -- every saved checkpoint would carry norm weights exactly 1.
    # With fp32 master weights the same updates accumulate exactly; the artifact
    # is cast back to guard_dtype inside save_checkpoint.
    guard_model = guard_model.to(device)
    logger.info(f"Guard architecture: {arch}")
    logger.info(f"Guard Model dtype: params=torch.float32 (master weights), compute={guard_dtype} via autocast")
    logger.info(f"Guard Model Parameters: {guard_model.count_parameters():,}")
    logger.info(f"Guard classification-head bias init: {config.guard_model.init_bias} "
                f"(negative -> low positive-class predictions at start)")

    # Initialize Loss Function
    # Use BalancedMultiTaskLoss for class-imbalanced data
    # Features:
    # 1. Class weights: Auto-computed from batch statistics to handle imbalance
    # 2. Confidence weighting: Optional sequence-level softmax weighting
    criterion = BalancedMultiTaskLoss(
        num_query_classes=8,
        ignore_index=-100,
        query_weight=config.training.query_weight,
        safety_weight=config.training.safety_weight,
        hallucination_weight=config.training.hallucination_weight,
        # Class weights will be computed dynamically from batch statistics
        safety_class_weights=None,  # Auto-computed per batch
        hallu_class_weights=None,   # Auto-computed per batch
        # Confidence weighting (per-task; no global switch).
        use_safety_confidence_weight=config.training.use_safety_confidence_weight,
        use_hallu_confidence_weight=config.training.use_hallu_confidence_weight,
        confidence_temperature=config.training.confidence_temperature,
        confidence_beta_tau=config.training.confidence_beta_tau,
        confidence_aggregator=config.training.confidence_aggregator,
        confidence_min_weight=config.training.confidence_min_weight,
        confidence_weight_detach=config.training.confidence_weight_detach,
        # Per-batch pos_weight for Safety/Hallu BCE (per-task; no global switch).
        # Off by default safety: long sequences + confidence weighting -> NaN.
        use_safety_class_weight=config.training.use_safety_class_weight,
        use_hallu_class_weight=config.training.use_hallu_class_weight,
        # Dynamic task weighting (optional, for balancing task losses)
        dynamic_task_weight=config.training.dynamic_task_weight,
        ema_decay=config.training.ema_decay,
        # Query-head Safe-vs-risk partial mutual exclusion (dim 7 = Safe).
        safe_weight=config.training.safe_weight,
        mutex_weight=config.training.mutex_weight,
        # Query-label placement: False (default) -> Query labels broadcast over
        # Response tokens; True -> Query labels on the Query's last token only.
        use_single_token_as_query=config.training.use_single_token_as_query,
        use_hallu_assertion_mask=config.training.use_hallu_assertion_mask,
    )
    logger.info(f"Loss function config:")
    logger.info(
        f"  - pos_weight (class weighting): Safety={criterion.use_safety_class_weight} "
        f"| Hallu={criterion.use_hallu_class_weight}"
    )
    logger.info(
        f"  - Confidence weighting: Safety={criterion.use_safety_confidence_weight} "
        f"| Hallu={criterion.use_hallu_confidence_weight} "
        f"(aggregator={config.training.confidence_aggregator},"
        f" tau={config.training.confidence_beta_tau},"
        f" T={config.training.confidence_temperature},"
        f" detach={config.training.confidence_weight_detach})"
    )
    # Warn on the documented BCE NaN cliff: pos_weight AND confidence weighting
    # both ON for the SAME task (pos_weight~1e3 self-amplifies under confidence
    # weighting -> inf -> NaN). The per-task knobs exist precisely so the two
    # tasks don't have to share this risk; surface a bad combo at startup.
    for _task, _cw, _fw in (
        ("Safety", criterion.use_safety_class_weight, criterion.use_safety_confidence_weight),
        ("Hallu", criterion.use_hallu_class_weight, criterion.use_hallu_confidence_weight),
    ):
        if _cw and _fw:
            logger.warning(
                f"  - WARNING: {_task} has BOTH pos_weight and confidence weighting "
                f"ON -- this is the documented BCE NaN cliff. Consider turning "
                f"one of them off for {_task}."
            )
    logger.info(f"  - Dynamic task weighting: {config.training.dynamic_task_weight}")
    logger.info(f"  - Task weights: query={config.training.query_weight}, safety={config.training.safety_weight}, hallu={config.training.hallucination_weight}")
    logger.info(f"  - Query Safe-vs-risk: safe_weight={config.training.safe_weight}, mutex_weight={config.training.mutex_weight}")
    logger.info(
        f"  - Query label placement: use_single_token_as_query="
        f"{config.training.use_single_token_as_query} "
        f"({'Query last token' if config.training.use_single_token_as_query else 'all Response tokens'})"
    )
    logger.info(
        f"  - Hallu POS assertion mask: use_hallu_assertion_mask="
        f"{config.training.use_hallu_assertion_mask}"
        + ("" if not config.training.use_hallu_assertion_mask
            else " (content-words only; requires spacy + en_core_web_sm)")
    )

    # Initialize Optimizer
    optimizer = optim.AdamW(
        guard_model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # Initialize Mixed Precision
    # Note: BFloat16 doesn't need GradScaler (no gradient underflow issues)
    # Only use GradScaler for FP16
    if config.training.use_amp:
        if guard_dtype == torch.bfloat16:
            scaler = None  # BFloat16 doesn't need gradient scaling
            logger.info("Using BFloat16 mixed precision (no GradScaler needed)")
        else:
            scaler = GradScaler()
            logger.info("Using FP16 mixed precision with GradScaler")
    else:
        scaler = None

    # Load checkpoint if resuming.
    # IMPORTANT: resume loads the bare-module state_dict, so it MUST run before
    # the DDP wrap (a DDP-wrapped module exposes `module.`-prefixed keys).
    start_epoch = 0
    global_step = 0
    if args.resume:
        state = load_checkpoint(args.resume, guard_model, optimizer, scaler, logger)
        start_epoch = state['epoch']
        global_step = state['step']

    # Wrap the Guard in DDP for multi-GPU data-parallel training of the Guard
    # (sglang backend). The bare module is kept for save/load/eval/warmup; the
    # DDP module is used only for the train/val forward+backward so gradients
    # are all-reduced across ranks. world_size==1 -> no wrap (bare == ddp).
    if WORLD_SIZE > 1:
        # Both Guard arches (mlp / attn) use every parameter on every
        # micro-batch, so find_unused_parameters stays False.
        guard_model_ddp = DDP(
            guard_model,
            device_ids=[LOCAL_RANK],
            find_unused_parameters=False,
        )
        logger.info(f"Guard wrapped in DDP (world_size={WORLD_SIZE}, device=cuda:{LOCAL_RANK})")
    else:
        guard_model_ddp = guard_model

    # Load Tokenizer
    logger.info("\n" + "="*80)
    logger.info("Loading Tokenizer")
    logger.info("="*80)

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model.name,
        trust_remote_code=True,
        padding_side='left'
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # Load Dataset
    logger.info("\n" + "="*80)
    logger.info("Loading Dataset")
    logger.info("="*80)

    # Load two types of datasets: safety (unified) + response_hallu
    train_datasets = []
    val_datasets = []

    # Load unified Safety dataset.
    # The Safety file carries both Query risk labels (dims 0-7) and Response
    # safety labels (dim 8). When train and val share the same file
    # (train_safety_path == val_safety_path), val is carved out by RATIO first
    # (config.data.val_split_ratio), then train_sample_size / val_sample_size
    # downsample each side from its pool (guaranteed disjoint). When the paths
    # differ, each is loaded independently as before. The dataset is loaded once
    # and shared via two Subset views in the shared-file case (the converter
    # runs only once; tokenization is lazy in __getitem__).
    train_safety_path = config.data.train_safety_path
    val_safety_path = config.data.val_safety_path
    shared_safety_file = (
        any_data_path_exists(train_safety_path) and
        same_data_path_set(train_safety_path, val_safety_path)
    )

    if shared_safety_file:
        safety_full = GuardrailDataset(
            data_path=train_safety_path,
            dataset_type='safety',
            tokenizer=tokenizer,
            max_length=config.data.max_seq_length,
            use_single_token_as_query=config.training.use_single_token_as_query,
            use_hallu_assertion_mask=config.training.use_hallu_assertion_mask,
        )
        total_size = len(safety_full)

        # Step 1: ratio-based val holdout from the full file
        ratio = config.data.val_split_ratio or 0.0
        val_holdout = int(round(total_size * ratio))
        val_holdout = min(max(val_holdout, 0), total_size)
        if val_holdout > 0:
            all_idx = list(range(total_size))
            random.shuffle(all_idx)
            val_indices = all_idx[:val_holdout]
            val_set = set(val_indices)
            train_pool = [i for i in all_idx if i not in val_set]
        else:
            val_indices = []
            train_pool = list(range(total_size))

        # Step 2: downsample each side to its sample_size
        if config.data.val_sample_size and len(val_indices) > config.data.val_sample_size:
            val_indices = random.sample(val_indices, config.data.val_sample_size)
        if val_indices:
            val_datasets.append(Subset(safety_full, val_indices))
        logger.info(
            f"Loaded safety (split from single file {train_safety_path}, {total_size} total): "
            f"train_pool={len(train_pool)}, val_holdout={len(val_indices)} (ratio={ratio})"
        )

        if config.data.train_sample_size and len(train_pool) > config.data.train_sample_size:
            train_idx = random.sample(train_pool, config.data.train_sample_size)
            logger.info(f"Loaded train_safety: {len(train_idx)} samples (sampled from {len(train_pool)} non-val)")
        else:
            train_idx = train_pool
            logger.info(f"Loaded train_safety: {len(train_pool)} samples (all non-val)")
        train_datasets.append(Subset(safety_full, train_idx))
    else:
        # Separate train / val safety files (legacy): load + downsample each.
        if any_data_path_exists(train_safety_path):
            train_safety = GuardrailDataset(
                data_path=train_safety_path,
                dataset_type='safety',
                tokenizer=tokenizer,
                max_length=config.data.max_seq_length,
                use_single_token_as_query=config.training.use_single_token_as_query,
                use_hallu_assertion_mask=config.training.use_hallu_assertion_mask,
            )
            original_size = len(train_safety)
            if config.data.train_sample_size and original_size > config.data.train_sample_size:
                train_safety = Subset(train_safety, random.sample(range(original_size), config.data.train_sample_size))
                logger.info(f"Loaded train_safety: {config.data.train_sample_size} samples (sampled from {original_size})")
            else:
                logger.info(f"Loaded train_safety: {original_size} samples")
            train_datasets.append(train_safety)
        else:
            logger.warning(f"Safety data not found: {train_safety_path}")

        if any_data_path_exists(val_safety_path):
            val_safety = GuardrailDataset(
                data_path=val_safety_path,
                dataset_type='safety',
                tokenizer=tokenizer,
                max_length=config.data.max_seq_length,
                use_single_token_as_query=config.training.use_single_token_as_query,
                use_hallu_assertion_mask=config.training.use_hallu_assertion_mask,
            )
            original_size = len(val_safety)
            if config.data.val_sample_size and original_size > config.data.val_sample_size:
                val_safety = Subset(val_safety, random.sample(range(original_size), config.data.val_sample_size))
                logger.info(f"Loaded val_safety: {config.data.val_sample_size} samples (sampled from {original_size})")
            else:
                logger.info(f"Loaded val_safety: {original_size} samples")
            val_datasets.append(val_safety)

    # Load Response Hallucination dataset.
    # When train and val share the same file, load it ONCE (split_filter=None)
    # and carve val by val_split_ratio -- the SAME ratio-based holdout Safety
    # uses -- rather than by the per-row `split` field. (A shared train/val file
    # often carries no split='validation' rows, so a split-field carve produced
    # an empty val set and Hallu val loss was always 0.) `test` rows are still
    # HELD OUT for offline eval (never sampled into train/val); the ratio holdout
    # runs over the remaining usable pool. Each side is then downsampled to its
    # sample_size. When the paths differ, each file is loaded independently with
    # split_filter set so a file pointed at by train_hallu_path yields only its
    # train rows (and likewise for val); test rows are likewise excluded.
    train_hallu_path = config.data.train_hallu_path
    val_hallu_path = config.data.val_hallu_path
    shared_hallu_file = (
        any_data_path_exists(train_hallu_path) and
        same_data_path_set(train_hallu_path, val_hallu_path)
    )

    if shared_hallu_file:
        hallu_full = GuardrailDataset(
            data_path=train_hallu_path,
            dataset_type='response_hallu',
            tokenizer=tokenizer,
            max_length=config.data.max_seq_length,
            split_filter=None,  # keep all rows; partition below (test held out)
            # No-op for response_hallu (dims 0-7 stay -100); passed for parity.
            use_single_token_as_query=config.training.use_single_token_as_query,
            use_hallu_assertion_mask=config.training.use_hallu_assertion_mask,
        )
        total_size = len(hallu_full)
        all_idx = list(range(total_size))

        # Carve val from the file the SAME way Safety does -- by val_split_ratio
        # -- instead of relying on rows carrying split='validation'. A single
        # train-and-val hallu file (val_hallu_path == train_hallu_path) frequently
        # has NO split='validation' rows, so the old split-field carve produced an
        # EMPTY val set -> every val batch had dim 9 == -100 -> Hallu val loss was
        # always 0. `test` rows are still held out for offline eval (never sampled
        # into train/val); the ratio holdout runs over the remaining usable pool.
        test_idx = [i for i in all_idx
                    if hallu_full.data[i].get('metadata', {}).get('split') == 'test']
        test_set = set(test_idx)
        usable_idx = [i for i in all_idx if i not in test_set]

        ratio = config.data.val_split_ratio or 0.0
        val_holdout = int(round(len(usable_idx) * ratio))
        val_holdout = min(max(val_holdout, 0), len(usable_idx))
        if val_holdout > 0:
            random.shuffle(usable_idx)
            val_idx = usable_idx[:val_holdout]
            train_idx = usable_idx[val_holdout:]
        else:
            val_idx = []
            train_idx = list(usable_idx)

        # Downsample each side to its sample_size (test is never sampled in).
        if config.data.val_sample_size and len(val_idx) > config.data.val_sample_size:
            val_idx = random.sample(val_idx, config.data.val_sample_size)
        if val_idx:
            val_datasets.append(Subset(hallu_full, val_idx))
        if config.data.train_sample_size and len(train_idx) > config.data.train_sample_size:
            train_idx = random.sample(train_idx, config.data.train_sample_size)
        train_datasets.append(Subset(hallu_full, train_idx))

        logger.info(
            f"Loaded hallu (single file {train_hallu_path}, {total_size} total): "
            f"train={len(train_idx)}, val={len(val_idx)} (ratio={ratio}), "
            f"test={len(test_idx)} (HELD OUT for offline eval)"
        )
    else:
        # Separate train / val hallu files: each self-routes by its `split` field.
        if any_data_path_exists(train_hallu_path):
            train_hallu = GuardrailDataset(
                data_path=train_hallu_path,
                dataset_type='response_hallu',
                tokenizer=tokenizer,
                max_length=config.data.max_seq_length,
                split_filter='train',
                # No-op for response_hallu (dims 0-7 stay -100); passed for parity.
                use_single_token_as_query=config.training.use_single_token_as_query,
                use_hallu_assertion_mask=config.training.use_hallu_assertion_mask,
            )
            original_size = len(train_hallu)
            if config.data.train_sample_size and original_size > config.data.train_sample_size:
                indices = random.sample(range(original_size), config.data.train_sample_size)
                train_hallu = Subset(train_hallu, indices)
                logger.info(f"Loaded train_hallu: {config.data.train_sample_size} samples (sampled from {original_size})")
            else:
                logger.info(f"Loaded train_hallu: {original_size} samples")
            train_datasets.append(train_hallu)
        else:
            logger.warning(f"Train hallu data not found: {train_hallu_path}")

        if any_data_path_exists(val_hallu_path):
            val_hallu = GuardrailDataset(
                data_path=val_hallu_path,
                dataset_type='response_hallu',
                tokenizer=tokenizer,
                max_length=config.data.max_seq_length,
                split_filter='validation',
                # No-op for response_hallu (dims 0-7 stay -100); passed for parity.
                use_single_token_as_query=config.training.use_single_token_as_query,
                use_hallu_assertion_mask=config.training.use_hallu_assertion_mask,
            )
            original_size = len(val_hallu)
            if config.data.val_sample_size and original_size > config.data.val_sample_size:
                indices = random.sample(range(original_size), config.data.val_sample_size)
                val_hallu = Subset(val_hallu, indices)
                logger.info(f"Loaded val_hallu: {config.data.val_sample_size} samples (sampled from {original_size})")
            else:
                logger.info(f"Loaded val_hallu: {original_size} samples")
            val_datasets.append(val_hallu)

    # Check if we have any datasets
    if len(train_datasets) == 0:
        raise ValueError("No training datasets found! Please check data paths in config.")
    if len(val_datasets) == 0:
        raise ValueError("No validation datasets found! Please check data paths in config.")

    # Concatenate datasets
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    # Create DataLoader
    collator = GuardrailCollator()

    # Use TaskRatioSampler for balanced training across tasks
    # Get task ratios from config (safety + hallucination only)
    task_ratios = [
        config.training.task_ratios.get('safety', 0.5),
        config.training.task_ratios.get('hallucination', 0.5)
    ]

    # Only use TaskRatioSampler if we have all datasets (safety + hallu = 2)
    train_sampler = None
    if len(train_datasets) == 2:
        dataset_sizes = [len(ds) for ds in train_datasets]
        train_sampler = TaskRatioSampler(
            dataset_sizes=dataset_sizes,
            task_ratios=task_ratios,
            shuffle=True,
            seed=config.training.seed if hasattr(config.training, 'seed') else 42
        )

        # Under DDP, shard the task-ratio-balanced index stream across ranks.
        # world_size==1 -> wrapper is a no-op (returns base order unchanged).
        if WORLD_SIZE > 1:
            train_sampler = DistributedTaskRatioSampler(
                train_sampler, num_replicas=WORLD_SIZE, rank=RANK
            )

        # Log sampler statistics
        sampler_stats = train_sampler.get_stats()
        logger.info(f"\nTaskRatioSampler enabled:")
        logger.info(f"  Dataset sizes: {sampler_stats['dataset_sizes']}")
        logger.info(f"  Task ratios: {sampler_stats['task_ratios']}")
        logger.info(f"  Samples per task: {sampler_stats['samples_per_task']}")
        logger.info(f"  Dataset cycles (passes per epoch): {sampler_stats['dataset_cycles']}")
        logger.info(f"  Total samples per epoch: {sampler_stats['total_samples']}")
        if WORLD_SIZE > 1:
            logger.info(f"  Per-rank samples (after DDP shard): {sampler_stats['rank_total_samples']}"
                        f" (rank {RANK}/{WORLD_SIZE})")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,  # Use custom sampler
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=collator,
            persistent_workers=config.data.num_workers > 0,
            prefetch_factor=4 if config.data.num_workers > 0 else None,
        )
    else:
        # Fallback to standard DataLoader if not all datasets are present
        logger.warning(f"Only {len(train_datasets)} datasets loaded, TaskRatioSampler disabled. Using standard shuffle.")
        fallback_sampler = None
        if WORLD_SIZE > 1:
            fallback_sampler = DistributedSampler(
                train_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True,
                seed=config.training.seed if hasattr(config.training, 'seed') else 42,
            )
        # Reuse the `train_sampler` name so the per-epoch set_epoch() below
        # covers both the TaskRatio path and this fallback (DistributedSampler
        # also exposes set_epoch). None when neither DDP nor TaskRatio applies.
        train_sampler = fallback_sampler
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=(fallback_sampler is None),  # shuffle ignored when sampler set
            sampler=fallback_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=collator,
            persistent_workers=config.data.num_workers > 0,
            prefetch_factor=4 if config.data.num_workers > 0 else None,
        )

    val_sampler = None
    if WORLD_SIZE > 1:
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False,
            seed=config.training.seed if hasattr(config.training, 'seed') else 42,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=(val_sampler is None),  # shuffle ignored when sampler set
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collator,
        persistent_workers=config.data.num_workers > 0,
        prefetch_factor=4 if config.data.num_workers > 0 else None,
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Train batches per epoch: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    grad_accum = max(1, int(getattr(config.training, 'gradient_accumulation_steps', 1)))
    logger.info(
        f"Gradient accumulation: {grad_accum} micro-batches/step "
        f"=> per-rank effective batch = {config.training.batch_size * grad_accum}, "
        f"global effective batch = {config.training.batch_size * grad_accum * WORLD_SIZE} "
        f"(batch_size {config.training.batch_size} x grad_accum {grad_accum} x world_size {WORLD_SIZE}); "
        f"~{len(train_loader) // grad_accum if grad_accum else 0} optimizer steps/epoch (per rank)"
    )

    # Training Loop
    logger.info("\n" + "="*80)
    logger.info("Starting Training")
    logger.info("="*80)

    # Build the LR scheduler now that len(train_loader) is known. Optional build:
    # cosine-with-warmup over total per-rank optimizer steps. Reconstruct it AFTER
    # the resume load above has reloaded the optimizer's base lr; on resume we
    # then restore the saved schedule state so the cosine continues in place.
    steps_per_epoch = len(train_loader) // grad_accum if grad_accum else len(train_loader)
    total_optimizer_steps = max(1, steps_per_epoch * config.training.epochs)
    scheduler = _build_lr_scheduler(optimizer, config, total_optimizer_steps, logger)
    if scheduler is not None and args.resume:
        scheduler_path = os.path.join(args.resume, "scheduler.pt")
        if os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location='cpu'))
            logger.info(f"✓ LR scheduler resumed from {scheduler_path} (lr={optimizer.param_groups[0]['lr']:.2e})")
        else:
            logger.warning(
                f"Resuming but no scheduler state at {scheduler_path} -- scheduler "
                f"continues from its constructed warmup LR (no saved schedule)."
            )

    # Warmup CUDA kernels before training
    warmup_cuda(
        base_model, guard_model_ddp, criterion, device, config, logger, amp_dtype=guard_dtype
    )

    best_val_loss = float('inf')

    for epoch in range(start_epoch, config.training.epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch+1}/{config.training.epochs}")
        logger.info(f"{'='*80}")

        # Set epoch for sampler (ensures different shuffle order each epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        # Train
        try:
            train_metrics = train_epoch(
                base_model, guard_model_ddp, train_loader, optimizer, criterion,
                device, scaler, epoch, config, logger, amp_dtype=guard_dtype,
                scheduler=scheduler,
                global_step=global_step,
                best_val_loss=best_val_loss,
                val_loader=val_loader,
                output_dir=config.training.output_dir,
                total_epochs=config.training.epochs,
            )
        except Exception:
            # Flight-recorder crash capture. train_epoch has NO internal
            # try/except, and MainProcessLogger gates logger output to rank 0,
            # so an uncaught exception on a NON-main rank would otherwise reach
            # torchrun's bare default handler -- which, if the failure is a
            # CUDA/C++ abort, prints only the torch runtime's line (e.g. the
            # cryptic 'cublasLtGetVersion' symbol message) and no usable Python
            # traceback. Print the full traceback to stderr (torchrun labels it
            # per rank) + a rank-0 log line, flush, then exit fast (os._exit
            # skips interpreter teardown, matching the success path at the end
            # of main(); torchrun tears the other ranks down when this rank
            # exits). This is what surfaces WHERE the rank died.
            _eprint(
                f"[faultdiag] TRAIN ABORTED: rank={RANK} local_rank={LOCAL_RANK} "
                f"epoch={epoch + 1} global_step_in={global_step}; full traceback:"
            )
            traceback.print_exc()
            logger.error(
                f"Training aborted in epoch {epoch + 1} (rank={RANK}, "
                f"global_step={global_step}): see full traceback on stderr."
            )
            try:
                for _h in logging.getLogger().handlers:
                    _h.flush()
            except Exception:
                pass
            os._exit(1)

        logger.info(
            f"\nTrain Metrics - "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Query: {train_metrics['query_loss']:.4f} | "
            f"Safety: {train_metrics['safety_loss']:.4f} | "
            f"Hallu: {train_metrics['hallucination_loss']:.4f}"
        )
        sk_b = train_metrics.get('skipped_batches', 0)
        sk_s = train_metrics.get('skipped_steps', 0)
        if sk_b or sk_s:
            logger.warning(
                f"NaN/Inf guard this epoch: dropped {sk_b} micro-batch(es) "
                f"(pre-backward), {sk_s} optimizer step(s) (non-finite grads)."
            )

        # Thread the running optimizer-step count and best-loss back out. eval
        # and periodic checkpoints now happen mid-epoch inside train_epoch (at
        # each eval_steps / save_steps boundary), NOT at epoch end.
        global_step = train_metrics['global_step']
        best_val_loss = train_metrics['best_val_loss']

        # ---- Per-epoch forced validation + best-model save ----
        # Guarantees a validation pass (and a best-save decision) at the end of
        # every epoch, regardless of whether the optimizer-step count happened to
        # land on an eval_steps boundary mid-epoch. Mirrors the mid-epoch block:
        # validate() all-reduces the val loss (so the <best comparison is
        # globally consistent), and save_checkpoint()'s DDP barrier pair means
        # ALL ranks must reach the save call whenever the condition holds (rank 0
        # writes; the others pass through the barriers).
        epoch_val = validate(
            base_model, guard_model_ddp, val_loader, criterion, device, config, logger
        )
        # validate() set both models to eval; restore training mode for the next
        # epoch (no-op before the final block, which re-validates anyway).
        guard_model_ddp.train()
        base_model.eval()

        if is_main_process():
            logger.info(
                f"[epoch {epoch+1} end] Val - "
                f"Loss: {epoch_val['loss']:.4f} | "
                f"Query: {epoch_val['query_loss']:.4f} | "
                f"Safety: {epoch_val['safety_loss']:.4f} | "
                f"Hallu: {epoch_val['hallucination_loss']:.4f}"
            )

        if epoch_val['loss'] < best_val_loss:
            best_val_loss = epoch_val['loss']
            save_checkpoint(
                guard_model_ddp, optimizer, scaler, scheduler, epoch + 1, global_step,
                os.path.join(config.training.output_dir, "best"), config, logger,
                save_training_state=False,
            )
            if is_main_process():
                logger.info(
                    f"[epoch {epoch+1} end] ✓ Best model saved "
                    f"(val_loss: {best_val_loss:.4f})"
                )

    # Final safety-net validation + best-model save. If the total optimizer
    # step count never hit an eval_steps boundary, no best checkpoint would
    # otherwise exist -- so always run one final validation here.
    final_val = validate(
        base_model, guard_model_ddp, val_loader, criterion, device, config, logger
    )
    if is_main_process():
        logger.info(
            f"\nFinal Val - "
            f"Loss: {final_val['loss']:.4f} | "
            f"Query: {final_val['query_loss']:.4f} | "
            f"Safety: {final_val['safety_loss']:.4f} | "
            f"Hallu: {final_val['hallucination_loss']:.4f}"
        )

    # Best-model save decision: every rank computed the same final_val loss
    # (validate all-reduced it), so the <best comparison is identical across
    # ranks. But the save_checkpoint() call contains a DDP barrier pair -- so ALL
    # ranks must reach it (not just rank 0), else rank 0 deadlocks on the entry
    # barrier. Rank 0 writes; the others pass through. The idiom: every rank
    # calls save_checkpoint whenever the (globally-consistent) condition holds.
    if final_val['loss'] < best_val_loss:
        best_val_loss = final_val['loss']
        save_checkpoint(
            guard_model_ddp, optimizer, scaler, scheduler,
            config.training.epochs, global_step,
            os.path.join(config.training.output_dir, "best"),
            config, logger,
            save_training_state=False,
        )
        if is_main_process():
            logger.info(f"✓ Best model saved from final val (val_loss: {best_val_loss:.4f})")

    # Save final model
    if is_main_process():
        logger.info("\n" + "="*80)
        logger.info("Training Completed!")
        logger.info("="*80)

    # Called unconditionally (all ranks) -- save_checkpoint's internal barrier
    # pair requires every rank to reach it.
    save_checkpoint(
        guard_model_ddp, optimizer, scaler, scheduler,
        config.training.epochs, global_step,
        os.path.join(config.training.output_dir, "final"),
        config, logger
    )

    if is_main_process():
        logger.info(f"✓ Final model saved to {config.training.output_dir}/final")
        logger.info(f"✓ Best val loss: {best_val_loss:.4f}")

    # Cleanup resources. Frees hooks/caches/CUDA memory before exit, and tears down
    # the DDP process group (if any) so the NCCL backend exits cleanly under
    # torchrun.

    logger.info("Cleaning up resources...")

    # Step 1: Explicitly clear hooks first (before any other cleanup)
    try:
        base_model._clear_hooks()
    except Exception as e:
        logger.warning(f"Warning: Failed to clear hooks: {e}")

    # Step 2: Clear hidden states cache
    try:
        base_model._hidden_states_cache.clear()
    except Exception as e:
        logger.warning(f"Warning: Failed to clear cache: {e}")

    # Step 3: Delete model references (order matters: DDP/bare guard first, then base)
    try:
        del guard_model_ddp
    except Exception:
        pass
    try:
        del guard_model
    except Exception:
        pass

    try:
        del base_model
    except Exception:
        pass

    # Step 4: Force garbage collection
    import gc
    gc.collect()

    # Step 5: Clear CUDA cache and synchronize
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Warning: CUDA cleanup failed: {e}")

    # Step 6: Destroy the process group before hard-exit so torchrun's NCCL
    # backend tears down per-rank (without this, a rank that hard-exits mid
    # collective can hang the survivors). No-op under single-process (world_size==1).
    if _dist_enabled():
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception as e:
            logger.warning(f"Warning: process group teardown failed: {e}")

    logger.info("✓ Cleanup completed")

    # Hard-exit to skip interpreter teardown. With a trust_remote_code MoE
    # sharded across GPUs via device_map, the Python GC phase at process exit
    # frees CUDA tensors in a non-deterministic order while the CUDA runtime is
    # already partially torn down, raising a benign "Segmentation fault" AFTER
    # all real cleanup (above) has succeeded. os._exit(0) bypasses that teardown
    # cleanly. Every checkpoint / log line has already been flushed, so skipping
    # finalizers is safe here.
    os._exit(0)


if __name__ == '__main__':
    main()