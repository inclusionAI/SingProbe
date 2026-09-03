# Models Module

Implements the SingProbe training stack:

1. **Base Model** (frozen, inference-only) — extracts hidden states from the
   configured layers, either in-process (`base_model.py`, HuggingFace
   `device_map` sharding) or via an external SGLang server
   (`sglang_client.py`, HTTP identity-token-probe dumps).
2. **SingProbe model** (trainable) — a lightweight per-token classifier on top of
   the concatenated tapped hidden states `[batch, seq, hidden_dim × num_layers]`
   → `[batch, seq, num_classes]` (default 10).
3. **Guardrail Model** (`guardrail_model.py`) — Base + SingProbe composed end to end.

## SingProbe architectures (`singprobe_model.arch`)

Two interchangeable SingProbe architectures, selected in the YAML config:

| arch  | module                | class           | description |
|-------|-----------------------|-----------------|-------------|
| `mlp` | `models/guard.py`     | `GuardMLP`      | 2-layer MLP over the concatenated hidden states. |
| `attn`| `models/sglang_attn.py` | `GuardAttnProbe` | Causal multi-query attention token probe over the layer axis: separate `proj_q/proj_k/proj_v` (one shared K/V head = MQA) → causal MQA along the token sequence → `o_proj` (attention W_O) → additive query residual (`h = q + o`) → post-residual RMSNorm → per-token linear classifier. |

Both accept the **same training-framework input contract**: `train.py` always
feeds the SingProbe model the concatenated hidden states
`[batch, seq, hidden_dim × num_layers]`, and both output per-token logits
`[batch, seq, num_classes]` so `trainers/loss.py` works without modification.

Note: on the SGLang backend the tapped features are already per-layer
`rmsnorm(hidden_states + residual)` (dumped by the identity probe), so the
SingProbe model itself applies NO input normalization.

## Smoke tests

Each SingProbe module is runnable as a self-test (no test framework needed):

```bash
python models/guard.py        # GuardMLP forward/backward smoke test
python models/sglang_attn.py  # GuardAttnProbe forward/backward + causality + sliding window
```
