from __future__ import annotations

import os
from typing import Any

from sglang.srt.arg_groups.overrides import model_config_of, resolving_view


def validate_token_probe(server_args: Any) -> None:
    cfg = resolving_view(server_args)
    if cfg.probe_ckpt is None:
        return

    architectures = model_config_of(server_args).hf_config.architectures or []
    # DeepseekV41Config rewrites V4.1 to this shared architecture.
    supported_architectures = {
        "DeepseekV4ForCausalLM",
        "BailingMoeV3ForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "Glm5NextForConditionalGeneration",
    }
    if not supported_architectures.intersection(architectures):
        raise ValueError(
            "--probe-ckpt is not wired for this architecture; supported "
            f"architectures are {sorted(supported_architectures)}; got "
            f"architectures={architectures!r}."
        )
    if cfg.pp_size != 1:
        raise ValueError("--probe-ckpt does not support pipeline parallelism")
    if cfg.attn_cp_size != 1 or cfg.dcp_size != 1:
        raise ValueError("--probe-ckpt does not support context parallelism")

    for field, option in (
        ("enable_encoder_swa_bounded_replay", "encoder SWA bounded replay"),
        ("enable_mixed_chunk", "mixed chunk prefill"),
        ("enable_layernorm_sp", "layernorm sequence parallelism"),
        ("enable_attn_tp_input_scattered", "attention TP input scattering"),
        (
            "enable_hierarchical_cache",
            "hierarchical cache (probe KV is not transferred)",
        ),
    ):
        if getattr(cfg, field, False):
            raise ValueError(f"--probe-ckpt does not support {option}")
    if cfg.disaggregation_mode != "null":
        raise ValueError("--probe-ckpt does not support disaggregated serving")

    if cfg.speculative_algorithm is None:
        return

    if {
        "Qwen4ExpForConditionalGeneration",
        "Glm5NextForConditionalGeneration",
    }.intersection(architectures):
        raise ValueError(
            "--probe-ckpt for Qwen4Exp and Glm5Next currently requires "
            "non-speculative decoding"
        )

    draft_path = cfg.speculative_draft_model_path
    same_checkpoint = draft_path is None
    if draft_path is not None:
        if os.path.exists(draft_path) or os.path.exists(cfg.model_path):
            same_checkpoint = os.path.realpath(draft_path) == os.path.realpath(
                cfg.model_path
            )
        else:
            same_checkpoint = draft_path.rstrip("/") == cfg.model_path.rstrip("/")

    if (
        cfg.speculative_algorithm != "EAGLE"
        or cfg.speculative_eagle_topk != 1
        or not same_checkpoint
    ):
        raise ValueError(
            "--probe-ckpt supports speculative decoding only with bundled "
            "MTP/NEXTN from the target checkpoint; got "
            f"algorithm={cfg.speculative_algorithm!r}, "
            f"topk={cfg.speculative_eagle_topk!r}, "
            f"draft_model={draft_path!r}."
        )
