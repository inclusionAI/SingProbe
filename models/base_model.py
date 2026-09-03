"""
Base Model Wrapper (frozen, inference-only)

This module implements:
1. Load a large model (e.g. 100B MoE), sharded across GPUs via HuggingFace
   device_map (pipeline-parallel) when shard_gpus > 1, else on a single GPU.
2. Extract hidden states from specified layers using forward hooks.
3. The model is frozen (requires_grad=False); only the Guard trains on its
   hidden states. Runs in a single process (no torchrun / NCCL).

Hook-based extraction advantages:
- Memory efficient: only stores needed layers (~1.5GB vs ~30GB for all layers)
- Gets post-norm features: captures Transformer block output (after LayerNorm)
- Minimal overhead: no extra forward pass needed
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


# Layer naming patterns for different model architectures
MODEL_LAYER_PATTERNS = {
    # Qwen series (Qwen2, Qwen2.5, Qwen3)
    "Qwen2ForCausalLM": "model.layers.{}",
    "Qwen2MoeForCausalLM": "model.layers.{}",
    "Qwen3ForCausalLM": "model.layers.{}",  # Qwen3 architecture

    # Ling/Bailing series (MoE models)
    "BailingMoeV2_5ForCausalLM": "model.layers.{}",  # Ling-Flash-2.6 (104B MoE)
    "BailingForCausalLM": "model.layers.{}",

    # LLaMA series
    "LlamaForCausalLM": "model.layers.{}",

    # Baichuan
    "BaichuanForCausalLM": "model.model.layers.{}",

    # ChatGLM
    "ChatGLMModel": "transformer.encoder.layers.{}",
    "ChatGLMForConditionalGeneration": "transformer.encoder.layers.{}",

    # DeepSeek
    "DeepseekForCausalLM": "model.layers.{}",

    # Mistral
    "MistralForCausalLM": "model.layers.{}",

    # Yi
    "YiForCausalLM": "model.layers.{}",

    # InternLM
    "InternLMForCausalLM": "model.layers.{}",

    # Generic fallback patterns (tried in order)
    "_fallback": [
        "model.layers.{}",
        "model.model.layers.{}",
        "transformer.encoder.layers.{}",
        "model.transformer.layers.{}",
        "layers.{}",
    ],
}


class BaseModelWrapper(nn.Module):
    """
    Base Model Wrapper (frozen, inference-only)

    Features:
    - Load a large model sharded across GPUs via HuggingFace device_map
      (pipeline-parallel) when shard_gpus > 1, else on a single GPU.
    - Extract hidden states from specified layers using forward hooks.
    - Freeze model parameters (inference only).
    - Single-process: no torchrun / NCCL.

    Hook-based extraction:
    - Memory efficient: only stores needed layers
    - Gets post-norm features (Transformer block output)
    - Automatic model architecture detection

    Usage:
        model = BaseModelWrapper(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            hidden_layers=[10, 22, 35],
            shard_gpus=8
        )

        with torch.no_grad():
            hidden_states = model(input_ids, attention_mask)
            # hidden_states: {10: tensor, 22: tensor, 35: tensor}
    """

    def __init__(
        self,
        model_name: str,
        hidden_layers: List[int],
        shard_gpus: int = 0,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
        kernel_inject: bool = True,
        load_strategy: str = "auto",
        **kwargs
    ):
        """
        Initialize Base Model.

        Args:
            model_name: HuggingFace model name or path
            hidden_layers: List of layer indices to extract hidden states (e.g., [10, 22, 35])
            shard_gpus: Number of GPUs to pipeline-shard the model across (device_map).
                0 = use all visible GPUs. Not tensor-parallel (the DeepSpeed TP path
                has been retired); this only controls HuggingFace device_map sharding.
            dtype: Model dtype (float16 or bfloat16)
            device: Device to use (default: cuda if available)
            kernel_inject: unused (kept for call-site compatibility)
            load_strategy: How to spread the model across GPUs:
                - "auto":       pick "device_map" when shard_gpus>1, else single-GPU
                - "device_map": HuggingFace pipeline-style sharding (each GPU holds a slice
                                of layers). Works for any HF model incl. trust_remote_code
                                MoE; never loads the whole model on one GPU -> no OOM on
                                large MoE. Forward is sequential across GPUs (slower than
                                true TP for big batches).
            **kwargs: Additional arguments for model loading
        """
        super().__init__()

        self.model_name = model_name
        self.hidden_layers = hidden_layers
        # Resolve "use all visible GPUs": shard_gpus<=0 -> device_count().
        if shard_gpus and shard_gpus > 0:
            self.shard_gpus = shard_gpus
        else:
            self.shard_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # Backward-compat alias for any code reading the old name.
        self.tp_degree = self.shard_gpus
        self.dtype = dtype
        self.kernel_inject = kernel_inject

        # Resolve load strategy
        if load_strategy == "auto":
            load_strategy = "device_map" if self.shard_gpus > 1 else "single"
        if load_strategy not in ("device_map", "single"):
            raise ValueError(
                f"Unknown or unsupported load_strategy='{load_strategy}'. "
                f"The DeepSpeed TP path has been retired; use 'device_map' or 'single'."
            )
        self.load_strategy = load_strategy

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Hook-related attributes
        self._hooks = []
        self._hidden_states_cache = {}
        self._layer_modules = {}
        # For device_map sharding: the device of the first layer (where inputs go)
        self._input_device = None
        # hf_device_map populated by from_pretrained(device_map=...)
        self._hf_device_map = None

        # Load model and tokenizer
        print(f"Loading model {model_name} (strategy={load_strategy}, shard_gpus={self.shard_gpus})...")
        self._load_model(**kwargs)

        # Register hooks for hidden state extraction
        self._register_hooks()

        print(f"Base Model initialized successfully!")
        print(f"  - Model: {model_name}")
        print(f"  - Strategy: {load_strategy} (shard_gpus={self.shard_gpus})")
        print(f"  - Hidden layers: {hidden_layers}")
        print(f"  - Device: {self.device}")
        print(f"  - Extraction mode: forward hooks (memory efficient)")

    def _load_model(self, **kwargs):
        """Load model using the configured strategy."""

        if self.load_strategy == "device_map":
            self._load_with_device_map(**kwargs)
        else:
            self._load_single_gpu(**kwargs)

    def _load_single_gpu(self, **kwargs):
        """Load model on single GPU"""
        print(f"Loading model on single GPU...")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            **kwargs
        )

        # Freeze all parameters (inference only)
        for param in model.parameters():
            param.requires_grad = False

        self.model = model.to(self.device)
        self.model.eval()

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {num_params / 1e9:.2f}B parameters")

    def _load_with_device_map(self, **kwargs):
        """Load model sharded across GPUs via HuggingFace device_map (pipeline parallel).

        Each GPU holds a contiguous slice of layers; `from_pretrained(device_map=...)`
        loads each weight directly onto the GPU that owns it, so the whole model is
        NEVER materialized on a single GPU. This works for any HF model including
        trust_remote_code MoE (e.g. Ling-2.6-flash 104B).

        Trade-off: forward is sequential across GPUs (pipeline-parallel), slower than
        true tensor parallel for big batches. For a model that fits on one GPU use
        the "single" strategy instead.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("device_map strategy requires CUDA GPUs.")

        num_gpus = torch.cuda.device_count()
        if self.shard_gpus > num_gpus:
            raise ValueError(
                f"shard_gpus={self.shard_gpus} > available GPUs ({num_gpus})"
            )

        # Budget per GPU: leave headroom for activations / KV / guard model.
        # Reserve a generous chunk so training-time activations + the Guard MLP fit
        # alongside the weight slice on each GPU.
        total_mem_per_gpu = torch.cuda.get_device_properties(0).total_memory
        # Reserve ~12GB headroom on each 80GB card (tune if needed)
        # NOTE: do NOT lower this aggressively. HuggingFace's device_map="auto" can
        # pack an uneven shard onto the LAST card (the residual after greedy
        # splitting) that pushes it past nominal capacity -- observed GPU3 hitting
        # 82GB at reserved=2GB, which then surfaced ECC errors. Keep headroom to
        # stay safely under 80GB/card. Disk offload was tested and is NOT the
        # forward-time bottleneck (forward stayed ~4s with offload eliminated).
        reserved = 12 * 1024 ** 3
        max_per_gpu = max(int(total_mem_per_gpu - reserved), 1)
        max_memory = {i: max_per_gpu for i in range(self.shard_gpus)}
        # NOTE: do not add a "cpu" key -- mixing int and str keys breaks transformers'
        # internal sort of the device_map. If the model doesn't fit on the GPUs,
        # from_pretrained will raise a memory error (which is what we want).

        print(f"[device_map] Sharding across {self.shard_gpus} GPU(s), "
              f"~{max_per_gpu / 1e9:.1f}GB budget each (reserved {reserved/1e9:.0f}GB)")

        # NOTE: do NOT force attn_implementation="flash_attention_2" here. This
        # trust_remote_code model's MLA forward (BailingMoeV3MultiLatentAttention)
        # only check `_attn_implementation == "flash_attention_2"` to pad the
        # value tensor and to keep a *2D* attention_mask -- but it then hard-codes
        # `attention_interface = eager_attention_forward` (never actually calls
        # flash). So setting flash_attention_2 makes the mask 2D while the compute
        # still runs eager, causing `IndexError: too many indices for tensor of
        # dimension 2`. We instead leave the default (eager) so the backbone
        # prepares a *4D* causal mask, and patch the MLA attention to compute via
        # torch.nn.functional.scaled_dot_product_attention (which never
        # materializes the [B,32,seq,seq] attention matrix -> no OOM, and accepts
        # a 4D mask natively). See _patch_mla_sdpa() below.
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            **kwargs
        )

        # Freeze all parameters (inference only)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        self.model = model
        self._hf_device_map = getattr(model, "hf_device_map", None)

        # MTP (multi-token prediction) heads are only needed for token generation.
        # For hidden-state extraction they are pure overhead, and under device_map
        # sharding they actively break: the MTP forward recomputes input embeddings
        # from `word_embeddings` (on the embedding GPU) and concatenates them with
        # the running `hidden_states` (on the last decoder layer's GPU), raising a
        # device-mismatch error inside torch.cat. Strip them so neither happens.
        self._strip_mtp_layers()

        # Replace the MLA softmax-attention's eager compute (materializes a huge
        # [B,32,seq,seq] attention matrix -> OOM) with SDPA. See method docs.
        self._patch_mla_sdpa()

        # The input device is where the first embedding/layer lives -- inputs must be
        # moved there before the forward pass.
        self._input_device = self._infer_input_device()

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded (device_map): {num_params / 1e9:.2f}B parameters")
        if self._hf_device_map:
            per_gpu = {}
            for k, v in self._hf_device_map.items():
                per_gpu[v] = per_gpu.get(v, 0) + 1
            # Sort by stringified device -- torch.device objects can't be compared
            # directly (sorted() raises "'<' not supported between str and int"
            # when the set mixes 'cpu' with indexed devices like cuda:0).
            sorted_devs = sorted(per_gpu.keys(), key=lambda d: str(d))
            print(f"  device_map: {len(self._hf_device_map)} module groups across "
                  f"GPUs {sorted_devs} ({per_gpu})")
        print(f"  input device: {self._input_device}")

    def _strip_mtp_layers(self):
        """Remove MTP (multi-token prediction) heads from the backbone.

        MTP heads are auxiliary next-token-prediction layers appended past the main
        decoder stack (e.g. Ling-3.0-flash's `num_nextn_predict_layers`). They are
        only relevant when generating tokens, never when extracting intermediate
        hidden states. Under device_map sharding they additionally trigger a
        cross-device torch.cat (recomputed embeddings on the embedding GPU vs. the
        running hidden state on the last decoder GPU) that aborts the forward pass.

        This drops the MTP layers from the backbone's `.layers` and zeros its
        `num_nextn_predict_layers` so the decoder-layer loop skips them entirely.
        Idempotent and silent for models without MTP (no matching attribute / zero).
        """
        backbone = self._get_backbone()
        n_mtp = getattr(backbone, "num_nextn_predict_layers", 0) or 0
        if n_mtp <= 0:
            return
        try:
            # Truncate the MTP tail off the decoder layer list.
            backbone.layers = nn.ModuleList(backbone.layers[:-n_mtp])
            backbone.num_nextn_predict_layers = 0
            print(f"[device_map] Stripped {n_mtp} MTP layer(s) from backbone "
                  f"(not needed for hidden-state extraction).")
        except Exception as e:
            # Non-fatal: if for some reason we can't strip, just warn. The forward
            # may then fail with the device-mismatch error, which is the original
            # symptom -- better to surface it than to crash at load time.
            print(f"[device_map] WARNING: could not strip MTP layers: {e}")

    def _patch_mla_sdpa(self):
        """Patch the MLA softmax-attention's eager compute to use SDPA.

        The trust_remote_code `BailingMoeV3MultiLatentAttention.forward` hard-codes
        `attention_interface = eager_attention_forward`, which materializes a
        [B, num_heads=32, seq, seq] attention matrix and runs softmax in fp32 --
        at seq=2048 that is ~1GB/batch per such layer and OOMs the device_map
        GPUs (each nearly full of weights). Forcing
        `attn_implementation="flash_attention_2"` does NOT help: the model file
        only uses that flag to pad V and keep a *2D* mask, but still calls eager,
        which then fails with `IndexError: too many indices for tensor of dim 2`.

        Cleanest fix that leaves the model file untouched: replace the
        `eager_attention_forward` function ON THE MODEL MODULE with one that uses
        `F.scaled_dot_product_attention` -- same signature and return contract
        (returns attn_output as [B, seq, H, dim], attn_weights=None). All Q/K/V
        projection logic in the MLA forward stays exactly as shipped; only the
        attention math changes. SDPA never materializes the attention matrix,
        accepts a 4D additive mask (the default/eager path prepares one), and
        does fp32 softmax internally under autocast. Q/K last_dim (qk_head_dim=192)
        differing from V last_dim (v_head_dim=128) is legal for SDPA.

        Idempotent and silent for models that don't define `eager_attention_forward`
        in their modeling module.
        """
        import torch.nn.functional as F
        import types

        # The modeling module is loaded by transformers under transformers_modules.
        # The MLA layer's `attention` attribute was built from that module's class,
        # so reach the module via the class.
        backbone = self._get_backbone()
        modeling_mod = None
        try:
            for layer in backbone.layers:
                attn = getattr(layer, "attention", None)
                if attn is not None and type(attn).__name__ == "BailingMoeV3MultiLatentAttention":
                    modeling_mod = type(attn).__module__
                    break
        except Exception:
            modeling_mod = None

        if modeling_mod is None:
            return  # not a Bailing MoE V3 model -> nothing to patch

        import sys
        mod = sys.modules.get(modeling_mod)
        if mod is None or not hasattr(mod, "eager_attention_forward"):
            return
        if getattr(mod.eager_attention_forward, "_ds_sdpa_patched", False):
            return

        def sdpa_attention_forward(module, query, key, value, attention_mask,
                                   scaling, dropout=0.0, **kwargs):
            # `query`/`key`/`value` here are [B, H, seq, dim] (already projected
            # and RoPE-applied by the MLA forward). repeat_kv expands GQA keys/
            # values to the full head count, exactly like eager does.
            repeat_kv = getattr(mod, "repeat_kv2", None)
            key_states = repeat_kv(key, module.num_key_value_groups) if repeat_kv else key
            value_states = repeat_kv(value, module.num_key_value_groups) if repeat_kv else value

            # attention_mask is None or a 4D additive [B,1,seq,seq] causal mask
            # (the default/eager path prepares it). SDPA broadcasts over the head
            # dim. When None, use SDPA's built-in causal masking.
            attn_mask = attention_mask
            is_causal = attn_mask is None

            with torch.backends.cuda.sdp_kernel(enable_flash=True,
                                                 enable_math=True,
                                                 enable_mem_efficient=True):
                attn_output = F.scaled_dot_product_attention(
                    query,
                    key_states,
                    value_states,
                    attn_mask=attn_mask,
                    dropout_p=dropout,
                    is_causal=is_causal,
                )
            # SDPA returns [B, H, seq, dim]; eager returns [B, seq, H, dim].
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, None

        sdpa_attention_forward._ds_sdpa_patched = True
        mod.eager_attention_forward = sdpa_attention_forward
        print("[device_map] Patched eager_attention_forward -> SDPA on "
              f"{modeling_mod} (avoids [B,32,seq,seq] OOM and flash 2D-mask "
              "IndexError).")

    def _infer_input_device(self) -> torch.device:
        """Infer the device where inputs should be placed (the first layer's device).

        For device_map-sharded models this is the device of the embedding / first
        decoder layer. Falls back to self.device if not determinable.
        """
        # hf_device_map maps module-name-prefixes -> torch.device
        if self._hf_device_map:
            # Sort by prefix specificity / take the first assigned device.
            # The map is an OrderedDict; first entry is typically the root/embedding.
            first_device = next(iter(self._hf_device_map.values()))
            return torch.device(first_device)
        # Fallback: probe the embedding parameter's device
        try:
            for name, param in self.model.named_parameters():
                if param is not None and 'embed' in name.lower():
                    return param.device
        except Exception:
            pass
        return self.device

    def _get_model_architecture(self) -> str:
        """Get model architecture name for layer pattern matching"""
        # Get the model class name
        model_class = self.model.__class__.__name__

        # Handle DeepSpeed wrapped models
        if hasattr(self.model, 'module'):
            model_class = self.model.module.__class__.__name__

        return model_class

    def _find_layer_module(self, layer_idx: int) -> nn.Module:
        """
        Find the Transformer layer module by index.

        Args:
            layer_idx: Layer index (0-based)

        Returns:
            nn.Module: The Transformer layer module

        Raises:
            ValueError: If layer not found
        """
        model_class = self._get_model_architecture()

        # Try model-specific pattern first
        if model_class in MODEL_LAYER_PATTERNS:
            pattern = MODEL_LAYER_PATTERNS[model_class].format(layer_idx)
            for name, module in self.model.named_modules():
                if name == pattern:
                    return module

        # Try fallback patterns
        for pattern in MODEL_LAYER_PATTERNS["_fallback"]:
            full_pattern = pattern.format(layer_idx)
            for name, module in self.model.named_modules():
                if name == full_pattern:
                    return module

        # Last resort: find by layer index in all modules
        layer_name_indicators = ['layer', 'block', 'h']
        for name, module in self.model.named_modules():
            if any(ind in name.lower() for ind in layer_name_indicators):
                # Try to parse layer index from name
                import re
                match = re.search(r'\.(\d+)(?:\.|$)', name)
                if match and int(match.group(1)) == layer_idx:
                    return module

        # Collect available layers for error message
        available_layers = []
        for name, module in self.model.named_modules():
            match = re.search(r'\.(\d+)(?:\.|$)', name)
            if match:
                idx = int(match.group(1))
                if idx not in available_layers:
                    available_layers.append(idx)
        available_layers.sort()

        raise ValueError(
            f"Layer {layer_idx} not found. "
            f"Model class: {model_class}. "
            f"Available layer indices: {available_layers[:20]}{'...' if len(available_layers) > 20 else ''}. "
            f"Please add the layer pattern to MODEL_LAYER_PATTERNS."
        )

    def _register_hooks(self):
        """
        Register forward hooks to capture hidden states from specified layers.

        Hook-based extraction advantages:
        1. Memory efficient: only stores needed layers
        2. Gets post-norm features (Transformer block output)
        3. No extra computation overhead
        """
        print(f"Registering forward hooks for layers: {self.hidden_layers}")

        for layer_idx in self.hidden_layers:
            try:
                layer_module = self._find_layer_module(layer_idx)
                self._layer_modules[layer_idx] = layer_module

                # Create hook function that captures the layer output
                def make_hook(idx):
                    def hook(module, input, output):
                        # Output is typically a tuple (hidden_states, ...) or tensor
                        if isinstance(output, tuple):
                            hidden = output[0]
                        else:
                            hidden = output
                        # Store in cache (detached to avoid retaining computation graph)
                        self._hidden_states_cache[idx] = hidden.detach()
                    return hook

                # Register the hook
                handle = layer_module.register_forward_hook(make_hook(layer_idx))
                self._hooks.append(handle)

                print(f"  - Layer {layer_idx}: hook registered")

            except ValueError as e:
                print(f"  - Layer {layer_idx}: FAILED - {e}")
                raise

    def _clear_hooks(self):
        """Remove all registered hooks safely"""
        try:
            for handle in self._hooks:
                try:
                    handle.remove()
                except Exception:
                    pass  # Ignore errors when removing individual hooks
            self._hooks.clear()
        except Exception:
            pass  # Ignore errors during hook cleanup

        try:
            self._layer_modules.clear()
        except Exception:
            pass

        try:
            self._hidden_states_cache.clear()
        except Exception:
            pass

    def _get_backbone(self) -> nn.Module:
        """
        Get the inner backbone model (e.g. Qwen3Model) that runs the transformer
        layers WITHOUT the final lm_head projection.

        We only need hidden states from intermediate layers; the lm_head projects
        hidden states to logits over the full vocabulary ([batch, seq, vocab_size]),
        which is a large, unnecessary computation here. Calling the backbone directly
        skips that projection and saves a substantial fraction of every forward pass.

        Handles:
        - Plain HuggingFace causal LM (has `.model` attribute, e.g. Qwen3ForCausalLM)
        - DeepSpeed-wrapped models (`.module` then `.model`)
        - Models whose top-level module IS already the backbone (fallback to self.model)

        Returns:
            nn.Module: the backbone to call in forward()
        """
        target = self.model
        # Unwrap DeepSpeed inference wrapper
        if hasattr(target, 'module'):
            target = target.module
        # Most ForCausalLM models expose the backbone as `.model`
        if hasattr(target, 'model') and hasattr(target.model, 'forward'):
            return target.model
        return target

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[int, torch.Tensor]:
        """
        Forward pass to extract hidden states from specified layers.

        Runs the backbone model only (skipping the lm_head logits projection) since
        only intermediate hidden states are needed. Forward hooks capture the
        specified layer outputs.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments for the model

        Returns:
            Dict[int, torch.Tensor]: Hidden states for specified layers
                {layer_idx: [batch_size, seq_len, hidden_dim]}
        """
        # Clear previous cache
        self._hidden_states_cache.clear()

        # Resolve backbone once (cheap, attribute lookups only)
        if not hasattr(self, '_backbone') or self._backbone is None:
            self._backbone = self._get_backbone()

        # Move inputs to the device that owns the first layer.
        # - device_map sharding: that's self._input_device (one specific GPU).
        # - single: self.device.
        if self.load_strategy == "device_map":
            target_dev = self._input_device if self._input_device is not None else self.device
        else:
            target_dev = self.device

        input_ids = input_ids.to(target_dev)
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_dev)

        # Forward pass - hooks will automatically capture hidden states.
        # Call the backbone directly to skip the lm_head projection.
        with torch.no_grad():
            _ = self._backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **kwargs
            )

        # Collect captured hidden states.
        # With pipeline-sharded (device_map) models, different layers live on
        # different GPUs, so their captured hidden states are on different devices.
        # Move them all to self.device so the caller can torch.cat() them freely
        # (no device mismatch) and the Guard model (on self.device) can consume them.
        hidden_states = {}
        for layer_idx in self.hidden_layers:
            if layer_idx in self._hidden_states_cache:
                hs = self._hidden_states_cache[layer_idx]
                if hs.device != self.device:
                    hs = hs.to(self.device)
                hidden_states[layer_idx] = hs
            else:
                raise RuntimeError(
                    f"Hidden state for layer {layer_idx} was not captured. "
                    f"Hook may have failed to register."
                )

        return hidden_states

    def get_hidden_dim(self) -> int:
        """Get the hidden dimension of the model"""
        # Try to get from config
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'hidden_size'):
                return self.model.config.hidden_size
            # Handle DeepSpeed wrapped models
            if hasattr(self.model, 'module') and hasattr(self.model.module.config, 'hidden_size'):
                return self.model.module.config.hidden_size

        # Fallback: infer from first hidden state
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            hidden_states = self.forward(dummy_input)
            first_layer = list(hidden_states.values())[0]
            return first_layer.shape[-1]

    def get_num_layers(self) -> int:
        """Get the total number of layers in the model"""
        # Try to get from config
        if hasattr(self.model, 'config'):
            config = self.model.config
        elif hasattr(self.model, 'module'):
            config = self.model.module.config
        else:
            raise ValueError("Cannot find model config")

        # Common attribute names for number of layers
        for attr in ['num_hidden_layers', 'num_layers', 'n_layer', 'num_layers']:
            if hasattr(config, attr):
                return getattr(config, attr)

        # Fallback: count layer modules
        layer_indices = []
        for name, module in self.model.named_modules():
            import re
            match = re.search(r'\.(\d+)(?:\.|$)', name)
            if match:
                idx = int(match.group(1))
                if idx not in layer_indices:
                    layer_indices.append(idx)

        if layer_indices:
            return max(layer_indices) + 1

        raise ValueError("Cannot determine number of layers from model")

    def get_vocab_size(self) -> int:
        """Get the tokenizer vocabulary size from the model config.

        Used for building dummy token-ID inputs during CUDA warmup.
        Returns 0 if the config exposes no vocab size (caller should guard).
        """
        for target in (self,):
            config = getattr(target, 'config', None)
            if config is None and hasattr(self.model, 'module'):
                config = getattr(self.model.module, 'config', None)
            if config is None:
                config = getattr(self.model, 'config', None)
            if config is not None:
                for attr in ['vocab_size', 'n_vocab']:
                    if hasattr(config, attr):
                        return int(getattr(config, attr))
        return 0
        """
        Explicitly cleanup resources (hooks, caches).

        Call this before destroying the process group in distributed training
        to avoid segmentation faults.
        """
        self._clear_hooks()
        # Note: CUDA cache clearing is done in train.py after model deletion

    def __del__(self):
        """Cleanup hooks on deletion"""
        try:
            self._clear_hooks()
        except Exception:
            # Ignore all errors during garbage collection
            # This is critical to avoid segfaults during process shutdown
            pass

    def __repr__(self) -> str:
        return (
            f"BaseModelWrapper(\n"
            f"  model_name='{self.model_name}',\n"
            f"  hidden_layers={self.hidden_layers},\n"
            f"  shard_gpus={self.shard_gpus},\n"
            f"  dtype={self.dtype},\n"
            f"  device={self.device}\n"
            f")"
        )


def test_base_model():
    """Test Base Model loading and inference"""
    print("\n" + "="*80)
    print("Testing Base Model Wrapper")
    print("="*80)

    # Test configuration (use small model for testing)
    model_name = "gpt2"  # Small model for testing
    hidden_layers = [0, 1, 2]  # Test first 3 layers
    shard_gpus = 1  # Single GPU for testing

    try:
        # Initialize model
        model = BaseModelWrapper(
            model_name=model_name,
            hidden_layers=hidden_layers,
            shard_gpus=shard_gpus,
            dtype=torch.float32
        )

        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        print(f"\nTest forward pass:")
        print(f"  Input shape: {input_ids.shape}")

        hidden_states = model(input_ids)

        print(f"  Output layers: {list(hidden_states.keys())}")
        for layer_idx, hidden in hidden_states.items():
            print(f"    Layer {layer_idx}: {hidden.shape}")

        print(f"\nHidden dimension: {model.get_hidden_dim()}")
        print(f"Number of layers: {model.get_num_layers()}")

        print("\n✅ Base Model test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Base Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_base_model()