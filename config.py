"""
Configuration module for SingProbe

Provides unified configuration management for:
- Model configuration (Base Model, Guard MLP)
- Training configuration (hyperparameters, optimizer)
- Data configuration (paths, processing)
- Distributed configuration (legacy; not wired into the training loop)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


def _coerce_bool(opt, default: bool) -> bool:
    """Coerce a YAML loss-switch value to bool, guarding the null-string trap.

    The per-task pos_weight / confidence-weight switches are plain bools. YAML's
    real null (``null``/``~``/empty) parses to Python None; a bare ``None``
    token does NOT (PyYAML leaves it as the string "None", which would be
    truthy). Treat both as "use the default" so an omitted or null key is
    forgiving, and coerce any other value via bool().
    """
    if opt is None or (isinstance(opt, str) and opt.strip().lower() in ("none", "null")):
        return bool(default)
    return bool(opt)


def split_data_paths(data_path: Optional[str]) -> List[str]:
    """Split a (possibly semicolon-separated) data-path string into individual
    file paths. Each ``data.*_path`` field may list multiple files to be loaded
    and concatenated, e.g. ``"a.jsonl;b.jsonl"``. Surrounding whitespace is
    stripped and empty entries are dropped, so ``"a;; b "`` -> ``["a", "b"]``.
    A plain single path (no separator) returns a one-element list -- fully
    backward compatible with pre-multi-path configs.
    """
    if not data_path:
        return []
    parts = [p.strip() for p in str(data_path).split(';')]
    return [p for p in parts if p]


def any_data_path_exists(data_path: Optional[str]) -> bool:
    """True if ANY of the (possibly multiple, ';'-separated) paths in
    ``data_path`` exists on disk. Multi-path-aware replacement for
    ``os.path.exists(path)`` on the ``data.*_path`` fields.
    """
    return any(os.path.exists(p) for p in split_data_paths(data_path))


def same_data_path_set(a: Optional[str], b: Optional[str]) -> bool:
    """True if two (possibly ';'-separated) path strings resolve to the SAME
    non-empty set of absolute files. Order-insensitive and dedup-aware. Used to
    detect the "train and val share the same file(s)" case so the loader runs
    those files once and carves val by ratio/split. Returns False when either
    side is empty -- two missing paths are NOT treated as a shared file.
    """
    set_a = {os.path.abspath(p) for p in split_data_paths(a)}
    set_b = {os.path.abspath(p) for p in split_data_paths(b)}
    return bool(set_a) and set_a == set_b


@dataclass
class InferenceConfig:
    """Inference configuration"""
    framework: str = "device_map"  # device_map / sglang / auto (deepspeed TP retired)
    dtype: str = "float16"  # float16 / bfloat16
    kernel_inject: bool = True

    # --- device_map sharding (framework="device_map") ---
    # Number of GPUs to pipeline-shard the frozen base model across (HuggingFace
    # device_map). This is pipeline sharding of layer slices, NOT tensor parallel.
    # Defaults to 0 = use all visible GPUs. Ignored by the sglang path.
    shard_gpus: int = 0

    # --- SGLang hidden-state client (framework="sglang") ---
    # When framework=="sglang", the Base Model is an external SGLang server that
    # dumps per-token hidden states (identity token probe) to `sglang_save_dir`.
    # The training process pulls them over HTTP and deletes each file after read.
    sglang_url: str = "http://127.0.0.1:6000"
    # Point this at a tmpfs (e.g. /dev/shm) so hidden-state dumps never hit
    # disk; the client reads + deletes each file per batch, so steady-state
    # usage is only the in-flight files. Must match SGLANG_TOKEN_PROBE_SAVE_DIR
    # on the server side (see scripts/run_train_pipeline.sh).
    sglang_save_dir: str = "/dev/shm/save_probe"
    # Probe checkpoint path (config.json with hidden_size + layer_ids) -- read to
    # derive hidden_dim for the Guard auto-config, and validated against the server.
    sglang_probe_ckpt: str = "path/to/probe_ckpt"
    sglang_timeout: float = 120.0   # seconds to wait for a single dump file
    sglang_max_concurrency: int = 8  # parallel in-flight /generate requests

    # --- SGLang server launch params (framework="sglang") ---
    # The pipeline scripts read these to launch `sglang.launch_server` with
    # --tp / --dp and the right CUDA_VISIBLE_DEVICES. Moving them out of the
    # shell scripts (where they were env-only) makes a base-model swap -- which
    # often changes parallelism needs -- a yaml-only change. Shell env overrides
    # (TP=, DP=, GPUS=) still win at runtime for ad-hoc runs.
    sglang_tp: int = 4
    sglang_dp: int = 3
    sglang_gpus: str = "4,5,6,7,8,9,10,11,12,13,14,15"


@dataclass
class CacheConfig:
    """Cache configuration"""
    backend: str = "redis"  # redis / shared_memory / file
    host: str = "localhost"
    port: int = 6379
    max_size: int = 100000
    compress: bool = True


@dataclass
class BaseModelConfig:
    """Base Model configuration"""
    # HuggingFace repo ID (auto-downloaded on first use) or a local path.
    name: str = "inclusionAI/Ling-3.0-flash"
    hidden_layers: List[int] = field(default_factory=lambda: [10, 22, 35])

    # Per-layer hidden size of the Base Model (Ling-3.0-flash -> 2560). Read by
    # the pipeline scripts to build the identity-probe config (so the dump width
    # matches) and by train.py's SGLang client to size the Guard input. Was a
    # shell-side env var (HIDDEN_SIZE); now yaml so a model swap is one file.
    hidden_size: Optional[int] = 2560

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'hidden_layers': self.hidden_layers,
            'hidden_size': self.hidden_size,
            'inference': {
                'framework': self.inference.framework,
                'dtype': self.inference.dtype,
                'kernel_inject': self.inference.kernel_inject,
                'shard_gpus': self.inference.shard_gpus,
                'sglang_url': self.inference.sglang_url,
                'sglang_save_dir': self.inference.sglang_save_dir,
                'sglang_probe_ckpt': self.inference.sglang_probe_ckpt,
                'sglang_timeout': self.inference.sglang_timeout,
                'sglang_max_concurrency': self.inference.sglang_max_concurrency,
                'sglang_tp': self.inference.sglang_tp,
                'sglang_dp': self.inference.sglang_dp,
                'sglang_gpus': self.inference.sglang_gpus,
            },
            'cache': {
                'backend': self.cache.backend,
                'host': self.cache.host,
                'port': self.cache.port,
                'max_size': self.cache.max_size,
                'compress': self.cache.compress,
            }
        }


@dataclass
class GuardModelConfig:
    """Guard model configuration.

    Supports two architectures via `arch`:
      - "mlp":  2-layer MLP (GuardMLP) over concatenated hidden states.
      - "attn": "attn" token-probe Guard (GuardAttnProbe, models/sglang_attn.py):
                the tapped+normed layer features are projected by separate
                proj_q/proj_k/proj_v (one shared KV head = MQA) and run through
                causal multi-query attention along the token sequence, then an
                attention output projection (o_proj), an additive query
                residual (h = q + o), a post-residual RMSNorm, and a per-token
                linear classifier. NO input normalization here (the SGLang
                dump path already rms-norms (hidden+residual) per tapped
                layer). Relevant knobs: num_query_heads, head_dim,
                sliding_window.
    """
    hidden_dim: Optional[int] = None  # Auto-detected from Base Model
    num_layers: Optional[int] = None  # Auto-inferred from base_model.hidden_layers
    intermediate_dim: int = 1024      # MLP hidden dim (arch "mlp" only)
    num_classes: int = 10  # 8 Query + 2 Response
    dropout: float = 0.1
    activation: str = "gelu"  # gelu / relu (GuardMLP only)

    # Final classification-head bias init constant. A negative value (default
    # -5.0) makes the Guard start from low positive-class predictions (sigmoid
    # of -5 ≈ 0.0067), so the model learns to "say no" by default and only
    # flips to positive once there's evidence. Applied to BOTH architectures'
    # final layer (GuardMLP.fc2 / GuardAttnProbe.classifier). Set 0.0 to
    # restore zero-init behavior for GuardMLP (the attn head keeps its
    # pytorch-default init when 0.0 is passed too).
    init_bias: float = 0.0

    # Architecture selection: "mlp" or "attn".
    arch: str = "mlp"

    # --- attn specific (singprobe_model.arch == "attn"; ignored by GuardMLP).
    # GuardAttnProbe (models/sglang_attn.py), the attn token probe. Multi-query
    # attention: num_query_heads query heads share ONE K/V head (head_dim wide
    # each). The classifier input is num_query_heads * head_dim. ---
    num_query_heads: int = 1        # number of MQA query heads
    head_dim: int = 64              # per-head width shared by Q/K/V
    # Causal attention window size in tokens (arch "attn"): 0 = full causal,
    # positive W = causal sliding window of the last W tokens.
    sliding_window: int = 0

    @property
    def input_dim(self) -> int:
        """Calculate input dimension"""
        if self.hidden_dim is None:
            raise ValueError("hidden_dim is not set. Call set_hidden_dim() first.")
        if self.num_layers is None:
            raise ValueError("num_layers is not set. Call set_num_layers() first.")
        return self.hidden_dim * self.num_layers

    def set_hidden_dim(self, hidden_dim: int):
        """Set hidden dimension from Base Model"""
        self.hidden_dim = hidden_dim
        return self

    def set_num_layers(self, num_layers: int):
        """Set number of layers from base_model.hidden_layers"""
        self.num_layers = num_layers
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'intermediate_dim': self.intermediate_dim,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'activation': self.activation,
            'init_bias': self.init_bias,
            'input_dim': self.input_dim if self.hidden_dim and self.num_layers else None,
            'arch': self.arch,
            'sliding_window': self.sliding_window,
            'num_query_heads': self.num_query_heads,
            'head_dim': self.head_dim,
        }


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    # LR schedule. "cosine_with_warmup" (default) linearly warms the LR from 0 to
    # learning_rate over warmup_ratio of the total optimizer steps, then cosine-
    # anneals to 0. "constant" disables the schedule (legacy behavior) so the
    # optimizer runs at a flat learning_rate -- warmup_ratio is then a dead knob.
    lr_scheduler_type: str = "cosine_with_warmup"
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # float16 / bfloat16

    # Checkpoint
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "outputs/"

    # Diagnostics: log a per-step phase-timing breakdown every N micro-batches
    # (SGLang send / dump read / assemble / guard fwd / bwd / opt). 0 disables.
    profile_every: int = 20

    # Task sampling
    task_ratios: Dict[str, float] = field(default_factory=lambda: {
        'safety': 0.5,
        'hallucination': 0.5
    })

    # Task loss weights (for multi-task weighting)
    query_weight: float = 1.0
    safety_weight: float = 1.0
    hallucination_weight: float = 1.0

    # Loss function configuration
    #
    # pos_weight (per-batch class-imbalance weighting, num_neg/num_pos) and
    # confidence weighting are configured INDEPENDENTLY per task -- there is no
    # global switch. Each of Safety (dim 8) and Hallu (dim 9) states its own
    # choice explicitly, so e.g. Hallu can use pos_weight while Safety stays
    # unweighted, without forcing the same choice on both. Omit a key (or write
    # null) to take the default.
    #
    # pos_weight: with long sequences the ratio explodes (pos_weight ~1e3) and,
    # under confidence weighting, self-amplifies the positive-class gradient into
    # BCE's numerical cliff (inf -> NaN at ~1k steps). Set True only when the
    # imbalance is controlled (e.g. global/EMA pos_weight); the BCE head
    # otherwise runs plain unweighted. Default True matches the pre-decoupling
    # from_yaml default so configs that omitted the old global knob are unchanged.
    use_safety_class_weight: bool = True
    use_hallu_class_weight: bool = True

    # Confidence weighting: sequence-level softmax weighting to focus on confident
    # tokens. The weighting sharpness is self-regulated by the model's own
    # prediction dispersion (see BalancedMultiTaskLoss._compute_confidence_weights):
    # the softmax inverse-temperature beta = clamp(std(p_+) / confidence_beta_tau,
    # 0, 1) over valid tokens. At init the negative bias (init_bias=-5) makes all
    # tokens' p_+ nearly identical -> std ~ 0 -> beta ~ 0 -> softmax(constant) is
    # EXACTLY uniform, so the cold-start period starts from uniform weighting (no
    # init-noise preference gets amplified). As training makes p_+ disperse, beta
    # grows and the weighting sharpens toward the confident (high-p_+) tokens --
    # no warmup schedule, no step counter, resume-safe by construction.
    # NOTE: enabling BOTH pos_weight and confidence weighting on the SAME task
    # is the documented NaN cliff above. Prefer at most one of the two per task;
    # the per-task knobs exist precisely so you don't have to couple them across
    # Safety and Hallu. Default False (off, the safe choice) matches the
    # pre-decoupling from_yaml default.
    use_safety_confidence_weight: bool = False
    use_hallu_confidence_weight: bool = False
    confidence_temperature: float = 1.0   # Temperature T in softmax(beta * p_+ / T)
    # Std threshold at which beta reaches 1 (full weighting). sigma in (0,1) so
    # std <= ~0.5; real unsafe-vs-safe dispersion easily clears 0.1, while the
    # cold-start all-low-sigma regime sits at ~1e-3. 0.1 is a sane default.
    confidence_beta_tau: float = 0.1
    # Confidence aggregator: "beta" (default, self-regulated beta above) or
    # "softmax" (original fixed softmax(sigma/T) + min_weight floor -- restores
    # the pre-beta behavior for ablation/regression; note it WILL amplify
    # cold-start init noise, which is exactly what "beta" fixes).
    confidence_aggregator: str = "beta"
    # Minimum per-token weight floor. Only used by the "softmax" aggregator; the
    # "beta" path is exactly uniform at cold start and needs no floor.
    confidence_min_weight: float = 0.1
    # When True, the confidence weights are detached from the autograd graph so
    # they act as pure per-token multipliers and only the per-token BCE term
    # carries gradient ("pure hard-example reweighting" semantics). When False
    # (default) the weights stay differentiable and the model also receives
    # gradient through the weighting itself. No-op when confidence weighting is
    # off (both use_*_confidence_weight are False).
    confidence_weight_detach: bool = False
    # Dynamic task weighting: adjust task weights based on loss magnitude (EMA)
    dynamic_task_weight: bool = False     # Enable dynamic task weighting
    ema_decay: float = 0.9                # EMA decay for loss tracking

    # Query-head structure: Safe (dim 7) is mutually exclusive with the seven
    # risk classes (dims 0-6). safe_weight scales the Safe-head BCE; mutex_weight
    # scales the soft Safe-vs-risk exclusion regularizer (set 0 to disable).
    safe_weight: float = 1.0
    mutex_weight: float = 0.1

    # Query-label placement for the Query classification task (dims 0-7, safety
    # dataset only). When False (default) the Query labels are broadcast onto
    # every Response token (legacy behavior). When True the Query labels are
    # placed on a SINGLE token -- the Query's last token (query_end_pos, i.e. the
    # token immediately before the Response starts) -- and every other position
    # (Response tokens and Query prefix tokens) is -100 for dims 0-7, so the
    # Query task is supervised from one token per sample. Response-safety (dim 8)
    # and hallucination (dim 9) are NOT affected -- they stay on the Response
    # tokens regardless of this flag.
    use_single_token_as_query: bool = False

    # POS assertion mask for the Hallu task (dim 9 ONLY). When True the dataset
    # computes a per-response-token boolean mask (True on content words -- spaCy
    # en_core_web_sm POS not in {PUNCT,ADP,DET,AUX,CCONJ,SCONJ,PRON,SPACE,PART})
    # and the loss ANDs it into hallu_mask, so function-word Response tokens
    # become unsupervised for the Hallu task. Safety (dim 8) / Query (dims 0-7)
    # are unaffected. Off by default: existing runs are byte-identical and require
    # no spaCy. Requires spacy + en_core_web_sm installed when True.
    use_hallu_assertion_mask: bool = False

    # Legacy parameters (kept for backward compatibility)
    temperature: float = 1.0
    min_weight: float = 0.1
    confidence_threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'lr_scheduler_type': self.lr_scheduler_type,
            'warmup_ratio': self.warmup_ratio,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_grad_norm': self.max_grad_norm,
            'use_amp': self.use_amp,
            'amp_dtype': self.amp_dtype,
            'save_steps': self.save_steps,
            'eval_steps': self.eval_steps,
            'output_dir': self.output_dir,
            'profile_every': self.profile_every,
            'task_ratios': self.task_ratios,
            'query_weight': self.query_weight,
            'safety_weight': self.safety_weight,
            'hallucination_weight': self.hallucination_weight,
            'use_safety_class_weight': self.use_safety_class_weight,
            'use_hallu_class_weight': self.use_hallu_class_weight,
            'use_safety_confidence_weight': self.use_safety_confidence_weight,
            'use_hallu_confidence_weight': self.use_hallu_confidence_weight,
            'confidence_temperature': self.confidence_temperature,
            'confidence_beta_tau': self.confidence_beta_tau,
            'confidence_aggregator': self.confidence_aggregator,
            'confidence_min_weight': self.confidence_min_weight,
            'confidence_weight_detach': self.confidence_weight_detach,
            'dynamic_task_weight': self.dynamic_task_weight,
            'ema_decay': self.ema_decay,
            'safe_weight': self.safe_weight,
            'mutex_weight': self.mutex_weight,
            'use_single_token_as_query': self.use_single_token_as_query,
            'use_hallu_assertion_mask': self.use_hallu_assertion_mask,
            'temperature': self.temperature,
            'min_weight': self.min_weight,
            'confidence_threshold': self.confidence_threshold,
        }


@dataclass
class DataConfig:
    """Data configuration"""
    # Data paths for each task
    # safety: single file carries both Query risk labels and Response safety
    # labels; val is split out of train_safety_path (val_safety_path retained
    # for config-validation parity but not used as a separate load source).
    train_safety_path: str = "data/safety.jsonl"
    train_hallu_path: str = "data/train_hallu.jsonl"

    val_safety_path: str = "data/safety.jsonl"
    val_hallu_path: str = "data/val_hallu.jsonl"

    # Sample size for quick test (None = use all data)
    train_sample_size: Optional[int] = None  # e.g., 100 for quick test
    val_sample_size: Optional[int] = None    # e.g., 20 for quick test

    # When train_safety_path == val_safety_path (safety val split from the same
    # file), this ratio selects the val holdout fraction of the safety file.
    # Only used while splitting; train_sample_size / val_sample_size then
    # downsample each side. Set 0 (or None) to disable the ratio split.
    val_split_ratio: float = 0.02

    max_seq_length: int = 2048
    num_workers: int = 4
    pin_memory: bool = True

    # Cache
    prefetch_batches: int = 4
    timeout: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'train_safety_path': self.train_safety_path,
            'train_hallu_path': self.train_hallu_path,
            'val_safety_path': self.val_safety_path,
            'val_hallu_path': self.val_hallu_path,
            'train_sample_size': self.train_sample_size,
            'val_sample_size': self.val_sample_size,
            'val_split_ratio': self.val_split_ratio,
            'max_seq_length': self.max_seq_length,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'prefetch_batches': self.prefetch_batches,
            'timeout': self.timeout,
        }


@dataclass
class ProducerConfig:
    """Producer configuration"""
    num_workers: int = 2
    queue_size: int = 1000


@dataclass
class ConsumerConfig:
    """Consumer configuration"""
    num_workers: int = 1


@dataclass
class ProducerConsumerConfig:
    """Producer-Consumer configuration"""
    mode: str = "separate"  # separate / integrated
    producer: ProducerConfig = field(default_factory=ProducerConfig)
    consumer: ConsumerConfig = field(default_factory=ConsumerConfig)


@dataclass
class DistributedConfig:
    """Distributed training configuration"""
    backend: str = "nccl"
    master_port: int = 29500
    producer_consumer: ProducerConsumerConfig = field(default_factory=ProducerConsumerConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'backend': self.backend,
            'master_port': self.master_port,
            'producer_consumer': {
                'mode': self.producer_consumer.mode,
                'producer': {
                    'num_workers': self.producer_consumer.producer.num_workers,
                    'queue_size': self.producer_consumer.producer.queue_size,
                },
                'consumer': {
                    'num_workers': self.producer_consumer.consumer.num_workers,
                }
            }
        }


class Config:
    """
    Main configuration class

    Usage:
        config = Config.from_yaml('configs/all_models/ling-3.0-flash.yaml')
        print(config.base_model.name)
        print(config.training.learning_rate)
    """

    def __init__(
        self,
        base_model: Optional[BaseModelConfig] = None,
        guard_model: Optional[GuardModelConfig] = None,
        training: Optional[TrainingConfig] = None,
        data: Optional[DataConfig] = None,
        distributed: Optional[DistributedConfig] = None,
    ):
        self.base_model = base_model or BaseModelConfig()
        self.guard_model = guard_model or GuardModelConfig()
        self.training = training or TrainingConfig()
        self.data = data or DataConfig()
        self.distributed = distributed or DistributedConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        # Parse base model config
        base_model_dict = config_dict.get('base_model', {})

        # Parse inference config
        inference_dict = base_model_dict.get('inference', {})
        inference = InferenceConfig(
            framework=inference_dict.get('framework', 'device_map'),
            dtype=inference_dict.get('dtype', 'float16'),
            kernel_inject=inference_dict.get('kernel_inject', True),
            shard_gpus=inference_dict.get('shard_gpus', 0),
            sglang_url=inference_dict.get('sglang_url', 'http://127.0.0.1:6000'),
            sglang_save_dir=inference_dict.get('sglang_save_dir', '/dev/shm/save_probe'),
            sglang_probe_ckpt=inference_dict.get('sglang_probe_ckpt', 'path/to/probe_ckpt'),
            sglang_timeout=inference_dict.get('sglang_timeout', 120.0),
            sglang_max_concurrency=inference_dict.get('sglang_max_concurrency', 8),
            sglang_tp=inference_dict.get('sglang_tp', 4),
            sglang_dp=inference_dict.get('sglang_dp', 3),
            sglang_gpus=inference_dict.get('sglang_gpus', '4,5,6,7,8,9,10,11,12,13,14,15'),
        )

        # Parse cache config
        cache_dict = base_model_dict.get('cache', {})
        cache = CacheConfig(
            backend=cache_dict.get('backend', 'redis'),
            host=cache_dict.get('host', 'localhost'),
            port=cache_dict.get('port', 6379),
            max_size=cache_dict.get('max_size', 100000),
            compress=cache_dict.get('compress', True)
        )

        base_model = BaseModelConfig(
            name=base_model_dict.get('name', 'inclusionAI/Ling-3.0-flash'),
            hidden_layers=base_model_dict.get('hidden_layers', [10, 22, 35]),
            hidden_size=base_model_dict.get('hidden_size', 2560),
            inference=inference,
            cache=cache
        )

        # Parse singprobe_model config
        guard_dict = config_dict.get('singprobe_model', {})
        guard_model = GuardModelConfig(
            hidden_dim=guard_dict.get('hidden_dim'),  # Auto-detected
            num_layers=guard_dict.get('num_layers'),  # Auto-inferred
            intermediate_dim=guard_dict.get('intermediate_dim', 1024),
            num_classes=guard_dict.get('num_classes', 10),
            dropout=guard_dict.get('dropout', 0.1),
            activation=guard_dict.get('activation', 'gelu'),
            init_bias=guard_dict.get('init_bias', -5.0),
            arch=guard_dict.get('arch', 'mlp'),
            num_query_heads=guard_dict.get('num_query_heads', 1),
            head_dim=guard_dict.get('head_dim', 64),
            sliding_window=guard_dict.get('sliding_window', 0),
        )

        # Parse training config
        training_dict = config_dict.get('training', {})
        training = TrainingConfig(
            epochs=training_dict.get('epochs', 10),
            batch_size=training_dict.get('batch_size', 32),
            learning_rate=training_dict.get('learning_rate', 1e-4),
            weight_decay=training_dict.get('weight_decay', 0.01),
            lr_scheduler_type=training_dict.get('lr_scheduler_type', 'cosine_with_warmup'),
            warmup_ratio=training_dict.get('warmup_ratio', 0.1),
            gradient_accumulation_steps=training_dict.get('gradient_accumulation_steps', 4),
            max_grad_norm=training_dict.get('max_grad_norm', 1.0),
            use_amp=training_dict.get('use_amp', True),
            amp_dtype=training_dict.get('amp_dtype', 'float16'),
            save_steps=training_dict.get('save_steps', 500),
            eval_steps=training_dict.get('eval_steps', 500),
            output_dir=training_dict.get('output_dir', 'outputs/'),
            profile_every=training_dict.get('profile_every', 20),
            task_ratios=training_dict.get('task_ratios', {'safety': 0.5, 'hallucination': 0.5}),
            query_weight=training_dict.get('query_weight', 1.0),
            safety_weight=training_dict.get('safety_weight', 1.0),
            hallucination_weight=training_dict.get('hallucination_weight', 1.0),
            use_safety_class_weight=_coerce_bool(training_dict.get('use_safety_class_weight'), True),
            use_hallu_class_weight=_coerce_bool(training_dict.get('use_hallu_class_weight'), True),
            use_safety_confidence_weight=_coerce_bool(training_dict.get('use_safety_confidence_weight'), False),
            use_hallu_confidence_weight=_coerce_bool(training_dict.get('use_hallu_confidence_weight'), False),
            confidence_temperature=training_dict.get('confidence_temperature', 1.0),
            confidence_beta_tau=training_dict.get('confidence_beta_tau', 0.1),
            confidence_aggregator=training_dict.get('confidence_aggregator', 'beta'),
            confidence_min_weight=training_dict.get('confidence_min_weight', 0.1),
            confidence_weight_detach=_coerce_bool(training_dict.get('confidence_weight_detach'), False),
            dynamic_task_weight=training_dict.get('dynamic_task_weight', False),
            ema_decay=training_dict.get('ema_decay', 0.9),
            safe_weight=training_dict.get('safe_weight', 1.0),
            mutex_weight=training_dict.get('mutex_weight', 0.1),
            use_single_token_as_query=training_dict.get('use_single_token_as_query', False),
            use_hallu_assertion_mask=_coerce_bool(training_dict.get('use_hallu_assertion_mask'), False),
            temperature=training_dict.get('temperature', 1.0),
            min_weight=training_dict.get('min_weight', 0.1),
            confidence_threshold=training_dict.get('confidence_threshold', 0.5),
        )

        # Parse data config
        data_dict = config_dict.get('data', {})
        data = DataConfig(
            train_safety_path=data_dict.get('train_safety_path', 'data/safety.jsonl'),
            train_hallu_path=data_dict.get('train_hallu_path', 'data/train_hallu.jsonl'),
            val_safety_path=data_dict.get('val_safety_path', 'data/safety.jsonl'),
            val_hallu_path=data_dict.get('val_hallu_path', 'data/val_hallu.jsonl'),
            train_sample_size=data_dict.get('train_sample_size'),
            val_sample_size=data_dict.get('val_sample_size'),
            val_split_ratio=data_dict.get('val_split_ratio', 0.02),
            max_seq_length=data_dict.get('max_seq_length', 2048),
            num_workers=data_dict.get('num_workers', 4),
            pin_memory=data_dict.get('pin_memory', True),
            prefetch_batches=data_dict.get('cache', {}).get('prefetch_batches', 4),
            timeout=data_dict.get('cache', {}).get('timeout', 30),
        )

        # Parse distributed config
        dist_dict = config_dict.get('producer_consumer', {})
        distributed = DistributedConfig(
            backend=config_dict.get('distributed', {}).get('backend', 'nccl'),
            master_port=config_dict.get('distributed', {}).get('master_port', 29500),
        )

        return cls(
            base_model=base_model,
            guard_model=guard_model,
            training=training,
            data=data,
            distributed=distributed,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'base_model': self.base_model.to_dict(),
            'singprobe_model': self.guard_model.to_dict(),
            'training': self.training.to_dict(),
            'data': self.data.to_dict(),
            'distributed': self.distributed.to_dict(),
        }

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def update(self, **kwargs):
        """Update configuration with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")

    def validate(self) -> bool:
        """Validate configuration"""
        errors = []

        # Validate singprobe model
        if self.guard_model.intermediate_dim <= 0:
            errors.append("singprobe_model.intermediate_dim must be positive")

        if self.guard_model.dropout < 0 or self.guard_model.dropout > 1:
            errors.append("singprobe_model.dropout must be in [0, 1]")

        if self.guard_model.arch not in ('mlp', 'attn'):
            errors.append(
                f"singprobe_model.arch must be 'mlp' or 'attn', "
                f"got '{self.guard_model.arch}'"
            )

        if self.guard_model.arch == 'attn':
            # "attn" token-probe (GuardAttnProbe, models/sglang_attn.py). Only
            # num_query_heads / head_dim / sliding_window matter.
            if self.guard_model.num_query_heads < 1:
                errors.append("singprobe_model.num_query_heads must be >= 1")
            if self.guard_model.head_dim < 1:
                errors.append("singprobe_model.head_dim must be >= 1")
            if self.guard_model.sliding_window < 0:
                errors.append("singprobe_model.sliding_window must be >= 0 (0 = full causal)")

        # Validate training
        if self.training.confidence_aggregator not in ("beta", "softmax"):
            errors.append(
                f"training.confidence_aggregator must be 'beta' or 'softmax', "
                f"got '{self.training.confidence_aggregator}'"
            )

        if self.training.learning_rate <= 0:
            errors.append("training.learning_rate must be positive")

        if self.training.batch_size <= 0:
            errors.append("training.batch_size must be positive")

        if self.training.gradient_accumulation_steps < 1:
            errors.append("training.gradient_accumulation_steps must be >= 1")

        if self.training.temperature <= 0:
            errors.append("training.temperature must be positive")

        # Validate data - at least one training dataset should exist.
        # The *_path fields may be ';'-separated multi-path lists, so existence
        # is checked per constituent path (any exists -> ok).
        train_data_exists = any([
            any_data_path_exists(self.data.train_safety_path),
            any_data_path_exists(self.data.train_hallu_path)
        ])
        if not train_data_exists:
            errors.append(
                f"No training data found. Checked:\n"
                f"  - {self.data.train_safety_path}\n"
                f"  - {self.data.train_hallu_path}"
            )

        # Validate validation data - at least one validation dataset should exist
        # (safety val is split from train_safety_path; val_safety_path kept for parity)
        val_data_exists = any([
            any_data_path_exists(self.data.val_safety_path),
            any_data_path_exists(self.data.val_hallu_path)
        ])
        if not val_data_exists:
            errors.append(
                f"No validation data found. Checked:\n"
                f"  - {self.data.val_safety_path}\n"
                f"  - {self.data.val_hallu_path}"
            )

        # Report errors
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False

        return True

    def __repr__(self) -> str:
        """String representation"""
        # Handle case where hidden_dim is not yet set
        if self.guard_model.hidden_dim and self.guard_model.num_layers:
            input_dim_str = str(self.guard_model.input_dim)
        else:
            input_dim_str = "auto-detect"

        return (
            f"Config(\n"
            f"  base_model={self.base_model.name},\n"
            f"  singprobe_model(input_dim={input_dim_str}, "
            f"intermediate_dim={self.guard_model.intermediate_dim}),\n"
            f"  training(lr={self.training.learning_rate}, "
            f"batch_size={self.training.batch_size}, epochs={self.training.epochs}),\n"
            f"  data(train={self.data.train_safety_path}),\n"
            f")"
        )


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use default

    Args:
        config_path: Path to YAML config file (optional)

    Returns:
        Config object
    """
    if config_path and os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        return Config.from_yaml(config_path)
    else:
        if config_path:
            print(f"Warning: Config file not found: {config_path}, using default")
        print("Using default configuration")
        return Config()


# Example usage
if __name__ == '__main__':
    # Load default config
    config = Config()
    print("Default Configuration:")
    print(config)

    # Validate
    print(f"\nValidation: {'✓ Passed' if config.validate() else '✗ Failed'}")

    # Save to YAML
    config.save_yaml('configs/default.yaml')
    print("\n✓ Configuration saved to configs/default.yaml")