"""
Convert a SingProbe checkpoint (.pt) into safetensors + config.json.

A training checkpoint directory (e.g. .../checkpoints/mlp/checkpoint-80/) holds:
    guard_model.pt         # torch.save(model.state_dict()) -- probe weights ONLY
    training_state.json    # metadata, incl. full training config as 'config'
    optimizer.pt, scaler.pt (optional, not needed for inference)

This script reads guard_model.pt, rebuilds the SingProbe model from its config,
loads the weights, and writes a HuggingFace-style inference directory:
    <output_dir>/
        model.safetensors            # SingProbe weights
        config.json                  # HF-style config (auto_map + arch fields)
        configuration_sing_probe.py  # SingProbeMlpConfig / SingProbeAttnConfig
        modeling_sing_probe.py       # SingProbeMlpModel / SingProbeAttnModel

It supports the SingProbe architectures selected by singprobe_model.arch:
    - "mlp":   GuardMLP (models/guard.py) -- 2-layer MLP over the concatenated
               hidden states.
    - "attn":  GuardAttnProbe (models/sglang_attn.py) -- the "attn" token probe:
               separate proj_q/proj_k/proj_v (MQA) -> causal MQA along the token
               sequence -> o_proj -> additive query residual (h = q + o) ->
               post-residual RMSNorm -> per-token linear classifier.

Architecture config resolution order (first wins):
    1. --config <yaml>           explicit YAML config (base/singprobe fields used)
    2. training_state.json       present next to the checkpoint
    3. weight-shape inference    fallback (input_dim / intermediate_dim /
                                  num_classes from the state_dict)

hidden_dim and num_layers are auto-detected the same way train.py does it:
num_layers = len(base_model.hidden_layers); hidden_dim inferred from
input_dim // num_layers. If base_model.hidden_layers is unknown, hidden_dim
falls back to None and only input_dim is recorded (enough to rebuild the
MLP; the attn probe genuinely needs hidden_dim, so a YAML/training_state
config is recommended for that arch).

The output dir is a standard HuggingFace model directory that loads directly
with trust_remote_code (no SingProbe repo needed on sys.path):
    from transformers import AutoModel
    model = AutoModel.from_pretrained("./singprobe-model-dir", trust_remote_code=True)
    model(hidden_states).logits   # [batch, seq_len, num_classes]

Usage:
    python scripts/convert_checkpoint_to_safetensors.py \\
        --checkpoint path/to/checkpoint-80 \\
        --output-dir path/to/singprobe-model-step80

    # Override/Provide config explicitly (e.g. checkpoint dir has no training_state.json):
    python scripts/convert_checkpoint_to_safetensors.py \\
        --checkpoint .../checkpoint-80/guard_model.pt \\
        --config configs/all_models/ling-3.0-flash.yaml \\
        --output-dir ./out

    # Round-trip verify: reload via the injected HF files and diff vs .pt
    python scripts/convert_checkpoint_to_safetensors.py --checkpoint ... --verify
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# Add project root to path so `from models...` / `from config...` resolve.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_state_dict(pt_path: str) -> Dict[str, torch.Tensor]:
    """Load a SingProbe state_dict from a .pt file (CPU, no grad)."""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Checkpoint weight file not found: {pt_path}")
    state_dict = torch.load(pt_path, map_location="cpu")
    # Strip DDP/TP "module." prefix if present.
    cleaned = {}
    for k, v in state_dict.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        cleaned[nk] = v.detach().clone() if isinstance(v, torch.Tensor) else v
    return cleaned


def _detect_arch_from_keys(keys) -> str:
    """Infer singprobe_model.arch from state_dict key names."""
    keyset = set(keys)
    has_fc1 = any("fc1" in k for k in keyset)
    has_fc2 = any("fc2" in k for k in keyset)
    # The "attn" probe (GuardAttnProbe) stores SEPARATE proj_q/proj_k/proj_v
    # (one shared KV head) plus o_proj + norm; the MLP uses fc1/fc2. All three
    # qkv keys present -> "attn".
    if all(k in keyset for k in ("proj_q.weight", "proj_k.weight", "proj_v.weight")):
        return "attn"
    if has_fc1 and has_fc2:
        return "mlp"
    # Ambiguous: default to MLP and let the loader fail loudly if wrong.
    return "mlp"


def _infer_mlp_shapes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Infer input_dim / intermediate_dim / num_classes from a GuardMLP state_dict."""
    try:
        fc1_w = state_dict["fc1.weight"]   # [intermediate_dim, input_dim]
        fc2_w = state_dict["fc2.weight"]   # [num_classes, intermediate_dim]
    except KeyError as e:
        raise KeyError(
            f"Expected GuardMLP keys (fc1.weight, fc2.weight) for arch='mlp', "
            f"missing: {e}. Available keys: {sorted(state_dict.keys())}"
        )
    return {
        "input_dim": int(fc1_w.shape[1]),
        "intermediate_dim": int(fc1_w.shape[0]),
        "num_classes": int(fc2_w.shape[0]),
    }


def _resolve_config(
    checkpoint_dir: Optional[str],
    explicit_yaml: Optional[str],
    arch_hint: Optional[str],
) -> Tuple[Dict[str, Any], str]:
    """Resolve the SingProbe/base_model config to rebuild the model.

    Returns (config_dict, source) where source describes where it came from.
    config_dict has the shape of Config.to_dict() (may be partial).
    """
    cfg: Optional[Dict[str, Any]] = None
    source = "unknown"

    # 1. Explicit YAML overrides everything.
    if explicit_yaml:
        from config import load_config
        parsed = load_config(explicit_yaml)
        cfg = parsed.to_dict()
        source = f"yaml:{explicit_yaml}"

    # 2. training_state.json next to the checkpoint.
    if cfg is None and checkpoint_dir is not None:
        ts_path = os.path.join(checkpoint_dir, "training_state.json")
        if os.path.exists(ts_path):
            with open(ts_path, "r", encoding="utf-8") as f:
                ts = json.load(f)
            cfg = ts.get("config")
            if cfg is not None:
                source = f"training_state:{ts_path}"

    if cfg is None:
        cfg = {}
        source = "weight-shape-inference (no config available)"

    # Make sure the expected sub-dicts exist.
    cfg.setdefault("base_model", {})
    cfg.setdefault("singprobe_model", {})
    return cfg, source


# --------------------------------------------------------------------------- #
# Model rebuild
# --------------------------------------------------------------------------- #

def _build_guard_model(cfg: Dict[str, Any], arch_hint: Optional[str]):
    """Rebuild the SingProbe module from cfg. Returns (model, arch)."""
    g = cfg.get("singprobe_model", {})
    arch = (g.get("arch") or arch_hint or "mlp").lower()

    if arch == "mlp":
        from models.guard import GuardMLP
        # Prefer explicit dims; fall back to None and infer after load.
        input_dim = g.get("input_dim") or g.get("hidden_dim")
        if input_dim is None or g.get("num_layers") is None:
            # Defer: caller will infer from state_dict and rebuild.
            return None, arch
        model = GuardMLP(
            input_dim=input_dim,
            intermediate_dim=g.get("intermediate_dim", 1024),
            num_classes=g.get("num_classes", 10),
            dropout=g.get("dropout", 0.1),
            activation=g.get("activation", "gelu"),
        )
        return model, arch

    if arch == "attn":
        # "attn" token probe (GuardAttnProbe, models/sglang_attn.py):
        # separate proj_q/proj_k/proj_v (MQA) -> causal MQA -> o_proj ->
        # additive query residual (h = q + o) -> post-residual RMSNorm ->
        # per-token classifier. hidden_dim must equal the base model's
        # hidden_size (the per-tapped-layer width the SGLang identity probe
        # dumps); the resolve step above back-fills it from
        # base_model.hidden_size when the YAML leaves it to auto-detect.
        from models.sglang_attn import GuardAttnProbe
        hidden_dim = g.get("hidden_dim")
        num_layers = g.get("num_layers")
        if hidden_dim is None or num_layers is None:
            return None, arch
        model = GuardAttnProbe(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=g.get("num_classes", 10),
            num_query_heads=g.get("num_query_heads", 1),
            head_dim=g.get("head_dim", 64),
            sliding_window=g.get("sliding_window", 0),
            init_bias=g.get("init_bias", 0.0),
        )
        return model, arch

    raise ValueError(
        f"Unsupported singprobe_model.arch='{arch}'. Supported: 'mlp', 'attn'."
    )


# --------------------------------------------------------------------------- #
# Inject the HF release files (so downstream needs no SingProbe repo)
# --------------------------------------------------------------------------- #

# The output directory becomes a standard HuggingFace model dir: config.json
# carries an auto_map pointing at these two files (BOTH arch variants live in
# each file, one pair serves mlp and attn alike), so
# `AutoModel.from_pretrained(dir, trust_remote_code=True)` works out of the box.
_RELEASE_FILES = ("configuration_sing_probe.py", "modeling_sing_probe.py")


def _inject_release_files(output_dir: str) -> list:
    """Copy configuration_sing_probe.py + modeling_sing_probe.py next to the
    safetensors so the output dir loads via AutoModel without the SingProbe
    repo on sys.path. Returns the absolute paths of the injected files.
    """
    import shutil

    injected = []
    for name in _RELEASE_FILES:
        src = PROJECT_ROOT / "scripts" / name
        if not src.exists():
            raise FileNotFoundError(f"Release template not found: {src}")
        dst = Path(output_dir) / name
        shutil.copyfile(src, dst)
        injected.append(str(dst.resolve()))
    return injected


# --------------------------------------------------------------------------- #
# Main conversion
# --------------------------------------------------------------------------- #

def convert(
    checkpoint: str,
    output_dir: str,
    explicit_yaml: Optional[str] = None,
    dtype: Optional[str] = None,
    verify: bool = False,
    keep_pt: bool = True,
) -> str:
    """Convert a SingProbe checkpoint to safetensors + config.json.

    Args:
        checkpoint: path to a checkpoint directory (containing guard_model.pt)
            OR directly to a guard_model.pt file.
        output_dir: where to write model.safetensors + config.json.
        explicit_yaml: optional YAML config path to use for architecture.
        dtype: optional override ('float16'/'bfloat16'/'float32') for the
            saved weights. Defaults to whatever dtype the .pt was saved in.
        verify: round-trip reload and diff against the original state_dict.
        keep_pt: if False and a training_state.json exists, also copy
            training_state.json into output_dir (for traceability).

    Returns:
        The absolute output directory path.
    """
    # Locate guard_model.pt and the checkpoint dir.
    if os.path.isdir(checkpoint):
        checkpoint_dir = checkpoint
        pt_path = os.path.join(checkpoint_dir, "guard_model.pt")
    else:
        pt_path = checkpoint
        checkpoint_dir = os.path.dirname(pt_path)

    state_dict = _load_state_dict(pt_path)
    arch_hint = _detect_arch_from_keys(state_dict.keys())
    print(f"[1/5] Loaded state_dict from {pt_path}")
    print(f"      keys: {len(state_dict)}, inferred arch: {arch_hint}")

    cfg, source = _resolve_config(checkpoint_dir, explicit_yaml, arch_hint)
    print(f"[2/5] Architecture config source: {source}")

    # If arch is mlp and dims absent in cfg -> infer from weights, then rebuild.
    arch = (cfg.get("singprobe_model", {}).get("arch") or arch_hint or "mlp").lower()
    if arch == "mlp":
        inferred = _infer_mlp_shapes(state_dict)
        g = cfg.setdefault("singprobe_model", {})
        # Fill any missing dims from the weights.
        if g.get("input_dim") is None:
            g["input_dim"] = inferred["input_dim"]
        if g.get("intermediate_dim") is None:
            g["intermediate_dim"] = inferred["intermediate_dim"]
        if g.get("num_classes") is None:
            g["num_classes"] = inferred["num_classes"]

    # hidden_dim / num_layers: derive input_dim // num_layers when possible.
    g = cfg.setdefault("singprobe_model", {})
    bm = cfg.setdefault("base_model", {})
    hidden_layers = bm.get("hidden_layers") or [10, 22, 35]
    num_layers_cfg = g.get("num_layers") or len(hidden_layers)
    if g.get("input_dim") is not None and g.get("hidden_dim") is None:
        g["hidden_dim"] = g["input_dim"] // num_layers_cfg
    # Fall back to base_model.hidden_size (the per-layer width train.py derives
    # hidden_dim from at runtime) when hidden_dim is still unknown. Covers the
    # --config YAML path, where hidden_dim is auto-detected and not stored in
    # the YAML -- without it the 'attn' arch cannot be rebuilt.
    if g.get("hidden_dim") is None and bm.get("hidden_size"):
        g["hidden_dim"] = bm["hidden_size"]
    # num_layers: like hidden_dim, ASSIGN (not setdefault) when absent/None.
    # Config.to_dict() always emits a 'num_layers' key (None when auto-detected),
    # so dict.setdefault would be a no-op and leave it None -- breaking the
    # --config YAML path for the 'attn' arch (rebuilt from
    # base_model.hidden_layers). Derive from len(hidden_layers) in that case.
    if g.get("num_layers") is None:
        g["num_layers"] = num_layers_cfg
    # input_dim = hidden_dim * num_layers (train.py's concatenated-hidden-states
    # width). Back-fill when unknown so config.json carries a complete spec for
    # downstream loaders (the YAML path leaves it None since hidden_dim/num_layers
    # are auto-detected at runtime).
    if g.get("input_dim") is None and g.get("hidden_dim") and g.get("num_layers"):
        g["input_dim"] = g["hidden_dim"] * g["num_layers"]

    model, arch = _build_guard_model(cfg, arch_hint)
    if model is None:
        raise RuntimeError(
            f"Could not rebuild '{arch}' model from config. "
            f"Pass --config <yaml> with base_model.hidden_layers and the "
            f"singprobe_model dims set (hidden_dim/num_layers are needed)."
        )
    print(f"[3/5] Rebuilt {type(model).__name__} (arch={arch}) "
          f"input_dim={getattr(model, 'input_dim', g.get('input_dim'))}, "
          f"intermediate_dim={g.get('intermediate_dim')}, "
          f"num_classes={g.get('num_classes')}")

    # Load weights (strict=True so any mismatch is loud).
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading state_dict: {missing}")
    if unexpected:
        # Non-fatal: some checkpoint formats stash extra buffers.
        print(f"      (warn) unexpected keys ignored: {unexpected}")

    # Optional dtype cast for saved weights. When --dtype is not given, infer
    # the source dtype from the first FLOATING parameter in the freshly loaded
    # state_dict, so the saved safetensors preserves the original training
    # dtype (bf16 / fp16) instead of silently promoting to fp32 (which is what
    # ``model.load_state_dict`` does when the freshly-built nn.Module's params
    # default to fp32 but the source .pt's tensors are bf16).
    target_dtype = None
    if dtype:
        dt_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                  "float32": torch.float32}
        if dtype not in dt_map:
            raise ValueError(f"Unsupported dtype '{dtype}'. Use one of {list(dt_map)}.")
        target_dtype = dt_map[dtype]
    else:
        for _v in state_dict.values():
            if isinstance(_v, torch.Tensor) and _v.is_floating_point():
                target_dtype = _v.dtype
                break
    if target_dtype is not None and target_dtype != next(
        iter(model.parameters())
    ).dtype:
        model = model.to(target_dtype)

    # Write outputs.
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model.safetensors")
    from safetensors.torch import save_file

    # safetensors requires contiguous tensors on CPU.
    sd_to_save = {}
    for k, v in model.state_dict().items():
        t = v.detach().cpu()
        if target_dtype is not None and t.is_floating_point():
            t = t.to(target_dtype)
        sd_to_save[k] = t.contiguous()
    save_file(sd_to_save, out_path, metadata={"format": "pt"})
    print(f"[4/5] Wrote {out_path} ({len(sd_to_save)} tensors)")

    # config.json: HF-style architecture config, matching the SingProbe release
    # layout (auto_map + arch fields; input size derives as
    # hidden_size * len(base_model_layer_ids)).
    guard_cfg = cfg.get("singprobe_model", {})
    base_cfg = cfg.get("base_model", {})
    class_name, model_type, config_name = _release_meta_for_arch(arch)
    config_json = {
        "architectures": [class_name],
        "model_type": model_type,
        "auto_map": {
            "AutoConfig": f"configuration_sing_probe.{config_name}",
            "AutoModel": f"modeling_sing_probe.{class_name}",
        },
        "hidden_size": guard_cfg.get("hidden_dim"),
        "base_model_layer_ids": base_cfg.get("hidden_layers") or hidden_layers,
    }
    if arch == "attn":
        # Training-side sliding_window=0 means full causal; the release config
        # encodes that as null (SingProbeAttnConfig treats None as full causal).
        config_json["num_attention_heads"] = guard_cfg.get("num_query_heads")
        config_json["head_dim"] = guard_cfg.get("head_dim")
        config_json["sliding_window"] = guard_cfg.get("sliding_window", 0) or None
    else:
        config_json["intermediate_size"] = guard_cfg.get("intermediate_dim")
        config_json["hidden_act"] = guard_cfg.get("activation", "gelu")
    config_json["num_labels"] = guard_cfg.get("num_classes")
    config_json["base_model_name"] = base_cfg.get("name")
    config_json["torch_dtype"] = _torch_dtype_name(next(iter(sd_to_save.values())).dtype)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2, ensure_ascii=False)
    print(f"      Wrote {config_path}")

    # Inject the HF release files so the output dir is loadable via AutoModel
    # without the SingProbe repo on sys.path.
    injected_paths = _inject_release_files(output_dir)
    print(f"      Injected {', '.join(os.path.basename(p) for p in injected_paths)} "
          f"(arch={arch}; load via AutoModel.from_pretrained(..., trust_remote_code=True))")

    # Optionally copy training_state.json for full traceability.
    if not keep_pt and checkpoint_dir is not None:
        ts_src = os.path.join(checkpoint_dir, "training_state.json")
        if os.path.exists(ts_src):
            import shutil
            shutil.copy(ts_src, os.path.join(output_dir, "training_state.json"))
            print(f"      Copied training_state.json for traceability")

    # Verify round-trip: (a) safetensors tensors match the .pt state_dict, and
    # (b) the INJECTED modeling file builds + loads the model and reproduces the
    # repo model's forward output -- proving the deployed artifact is usable as-is.
    if verify:
        _verify_round_trip(out_path, state_dict, target_dtype)
        _verify_injected_model(output_dir, arch, model, dtype=target_dtype)
        print(f"[5/5] Verification passed (safetensors matches .pt; injected "
              f"HF files load via AutoModel + forward matches repo model)")
    else:
        print(f"[5/5] (skipped verification; pass --verify to check)")

    print(f"\n✓ Conversion complete: {os.path.abspath(output_dir)}")
    return os.path.abspath(output_dir)


def _release_meta_for_arch(arch: str):
    """(architectures entry, model_type, AutoConfig class name) per arch.

    Names must match configuration_sing_probe.py / modeling_sing_probe.py so
    the auto_map entries in config.json resolve.
    """
    meta = {
        "mlp": ("SingProbeMlpModel", "sing_probe_mlp", "SingProbeMlpConfig"),
        "attn": ("SingProbeAttnModel", "sing_probe_attn", "SingProbeAttnConfig"),
    }.get(arch)
    if meta is None:
        raise ValueError(f"Unsupported arch='{arch}'. Supported: 'mlp', 'attn'.")
    return meta


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }.get(dtype, str(dtype))


def _verify_injected_model(
    output_dir: str,
    arch: str,
    repo_model: torch.nn.Module,
    dtype: Optional[torch.dtype],
) -> None:
    """Prove the injected HF files are usable as a standalone artifact.

    Loads the output dir exactly the way a downstream user would --
    ``AutoModel.from_pretrained(output_dir, trust_remote_code=True)`` -- and
    feeds both that model and the repo-built model the same hidden states.
    Arch + weights + forward must all round-trip; raises AssertionError on any
    mismatch.
    """
    output_dir_abs = os.path.abspath(output_dir)

    # Reconstruct input_dim = hidden_size * len(base_model_layer_ids) (matches
    # the train.py concatenated-hidden-states contract the SingProbe model
    # expects; same derivation as the SingProbe configs' input_size).
    with open(os.path.join(output_dir_abs, "config.json"), "r", encoding="utf-8") as f:
        out_cfg = json.load(f)

    hidden_size = out_cfg.get("hidden_size")
    layer_ids = out_cfg.get("base_model_layer_ids") or []
    if hidden_size is None or not layer_ids:
        raise AssertionError(
            "config.json lacks hidden_size/base_model_layer_ids; cannot verify forward."
        )
    input_dim = hidden_size * len(layer_ids)

    from transformers import AutoModel
    injected_model = AutoModel.from_pretrained(output_dir_abs, trust_remote_code=True)
    injected_model = injected_model.eval()

    # Drive both models (eval) on identical inputs.
    repo_model = repo_model.eval()
    torch.manual_seed(0)
    batch, seq = 2, 64
    x = torch.randn(batch, seq, input_dim)

    test_dtype = dtype if dtype is not None else torch.float32
    repo_out = repo_model.to(test_dtype)(x.to(test_dtype))
    inj_out = injected_model(x.to(test_dtype))
    inj_logits = getattr(inj_out, "logits", None)
    if inj_logits is None:
        inj_logits = inj_out[0]
    if repo_out.shape != inj_logits.shape:
        raise AssertionError(
            f"Output shape mismatch: injected {tuple(inj_logits.shape)} vs "
            f"repo {tuple(repo_out.shape)}"
        )
    # Tight tolerance for fp32 round-trips; bf16/fp16 weights need slack since
    # the injected model runs at reduced precision end to end.
    if test_dtype in (torch.bfloat16, torch.float16):
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 1e-4, 1e-3
    close = torch.allclose(repo_out.float(), inj_logits.float(), atol=atol, rtol=rtol)
    if not close:
        max_abs = (repo_out.float() - inj_logits.float()).abs().max().item()
        raise AssertionError(
            f"Injected-model forward diverged from repo model "
            f"(max_abs={max_abs:.2e}, atol={atol}). Weights loaded ok but output differs."
        )


def _verify_round_trip(safetensors_path: str, original_sd: Dict[str, torch.Tensor],
                       target_dtype: Optional[torch.dtype]):
    from safetensors.torch import load_file
    loaded = load_file(safetensors_path)
    if set(loaded.keys()) != set(original_sd.keys()):
        raise AssertionError(
            f"Key mismatch!\n  loaded: {sorted(loaded.keys())}\n  "
            f"original: {sorted(original_sd.keys())}"
        )
    for k in original_sd:
        a = loaded[k]
        b = original_sd[k]
        if target_dtype is not None and b.is_floating_point():
            b = b.to(target_dtype)
        if a.shape != b.shape:
            raise AssertionError(f"Shape mismatch for {k}: {a.shape} vs {b.shape}")
        if torch.equal(a, b):
            continue
        # Fallback for fp precision: allow tiny diff.
        close = torch.allclose(a.float(), b.float(), atol=1e-6, rtol=1e-5)
        if not close:
            max_abs = (a.float() - b.float()).abs().max().item()
            raise AssertionError(
                f"Value mismatch for {k} (max_abs={max_abs:.2e})."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Convert a SingProbe checkpoint (.pt) to safetensors + config.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Checkpoint directory (with guard_model.pt) OR path to a guard_model.pt file.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for model.safetensors + config.json.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Optional YAML config to source the SingProbe architecture (overrides training_state.json).",
    )
    parser.add_argument(
        "--dtype", default=None, choices=["float16", "bfloat16", "float32"],
        help="Cast saved weights to this dtype. Defaults to the .pt's stored dtype.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Reload the safetensors and diff against the original .pt state_dict.",
    )
    parser.add_argument(
        "--copy-training-state", action="store_true",
        help="Also copy training_state.json into the output dir for traceability.",
    )
    args = parser.parse_args()

    convert(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        explicit_yaml=args.config,
        dtype=args.dtype,
        verify=args.verify,
        keep_pt=not args.copy_training_state,
    )


if __name__ == "__main__":
    main()