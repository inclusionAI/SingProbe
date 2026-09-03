from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import TokenClassifierOutput

from .configuration_sing_probe import SingProbeAttnConfig, SingProbeMlpConfig


class SingProbePreTrainedModel(PreTrainedModel):
    base_model_prefix = ""
    main_input_name = "hidden_states"

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    @staticmethod
    def _validate_input(config: Any, hidden_states: torch.Tensor) -> None:
        if hidden_states.shape[-1] != config.input_size:
            raise ValueError(
                f"input feature size {hidden_states.shape[-1]} does not match "
                f"hidden_size * number of tapped layers ({config.input_size})"
            )


class SingProbeMlpModel(SingProbePreTrainedModel):
    config_class = SingProbeMlpConfig

    def __init__(self, config: SingProbeMlpConfig) -> None:
        super().__init__(config)
        self.fc1 = nn.Linear(config.input_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.num_labels)
        self.act_fn = ACT2FN[config.hidden_act]
        self.post_init()

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_dict: bool | None = None,
        **_: Any,
    ) -> TokenClassifierOutput | tuple[torch.Tensor]:
        self._validate_input(self.config, hidden_states)
        hidden_states = hidden_states.to(self.fc1.weight.dtype)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        logits = self.fc2(hidden_states)
        if return_dict is False:
            return (logits,)
        return TokenClassifierOutput(logits=logits)


class SingProbeAttnModel(SingProbePreTrainedModel):
    config_class = SingProbeAttnConfig

    def __init__(self, config: SingProbeAttnConfig) -> None:
        super().__init__(config)
        if config.num_attention_heads < 1 or config.head_dim < 1:
            raise ValueError("num_attention_heads and head_dim must be positive")
        if config.sliding_window is not None and config.sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.projection_size = self.num_attention_heads * self.head_dim
        self.proj_q = nn.Linear(config.input_size, self.projection_size, bias=False)
        self.proj_k = nn.Linear(config.input_size, self.head_dim, bias=False)
        self.proj_v = nn.Linear(config.input_size, self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.projection_size, self.projection_size, bias=False)
        self.norm = nn.RMSNorm(self.projection_size, eps=1e-6)
        self.classifier = nn.Linear(self.projection_size, config.num_labels)
        self.post_init()

    def _sliding_window_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = query.shape[2]
        window = self.config.sliding_window
        assert window is not None
        outputs = []
        for start in range(0, seq_len, window):
            end = min(start + window, seq_len)
            key_start = max(0, start - window + 1)
            query_positions = torch.arange(start, end, device=query.device)
            key_positions = torch.arange(key_start, end, device=query.device)
            relative_positions = query_positions[:, None] - key_positions
            attention_mask = (relative_positions >= 0) & (relative_positions < window)
            attention_mask = attention_mask[None, None]
            query_block = query[:, :, start:end]
            key_block = key[:, :, key_start:end]
            value_block = value[:, :, key_start:end]
            attention_output = F.scaled_dot_product_attention(
                query_block,
                key_block,
                value_block,
                attn_mask=attention_mask,
                enable_gqa=self.num_attention_heads > 1,
            )
            outputs.append(attention_output)
        return torch.cat(outputs, dim=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_dict: bool | None = None,
        **_: Any,
    ) -> TokenClassifierOutput | tuple[torch.Tensor]:
        self._validate_input(self.config, hidden_states)
        hidden_states = hidden_states.to(self.classifier.weight.dtype)
        batch_size, seq_len, _ = hidden_states.shape
        query_features = self.proj_q(hidden_states)
        query_shape = (batch_size, seq_len, self.num_attention_heads, self.head_dim)
        kv_shape = (batch_size, seq_len, 1, self.head_dim)
        query = query_features.view(query_shape).transpose(1, 2)
        key = self.proj_k(hidden_states).view(kv_shape).transpose(1, 2)
        value = self.proj_v(hidden_states).view(kv_shape).transpose(1, 2)
        if self.config.sliding_window and self.config.sliding_window < seq_len:
            context = self._sliding_window_attention(query, key, value)
        else:
            context = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                is_causal=True,
                enable_gqa=self.num_attention_heads > 1,
            )
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.projection_size)
        hidden_states = self.o_proj(context) + query_features
        logits = self.classifier(self.norm(hidden_states))
        if return_dict is False:
            return (logits,)
        return TokenClassifierOutput(logits=logits)
