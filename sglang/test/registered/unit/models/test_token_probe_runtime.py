"""Real TokenProbe runtime with an Identity checkpoint and synthetic GPU states."""

import json
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.models.token_probe import token_probe as runtime


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("layout", ["streams", "hc_count_flat", "hc_mult_flat"])
def test_identity_runtime_prefill_decode_and_replay(
    tmp_path, monkeypatch, overlap, layout
):
    config = {
        "model_type": "sing_probe_identity",
        "hidden_size": 128,
        "base_model_layer_ids": [0, 2],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = NS(
        probe_ckpt=str(tmp_path),
        enable_attn_tp_input_scattered=False,
        speculative_algorithm=None,
    )
    monkeypatch.setattr(runtime, "get_server_args", lambda: args)
    monkeypatch.setattr(runtime, "resolved_view", lambda value: value)
    monkeypatch.setattr(runtime, "get_parallel", lambda: NS(attn_tp_rank=0))
    monkeypatch.setenv("SGLANG_ENABLE_TOKEN_PROBE_PREFILL", "1")
    monkeypatch.setenv("SGLANG_ENABLE_TOKEN_PROBE_OVERLAP", str(int(overlap)))
    config_key = "hc_count" if layout == "hc_count_flat" else "hc_mult"
    probe = runtime.TokenProbe(
        NS(hidden_size=128, num_hidden_layers=3, **{config_key: 4})
    )
    probe.prepare_device(torch.device("cuda"))
    probe.init_overlap(torch.device("cuda"))
    states = [
        torch.randn(3, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    mode = NS(is_decode=lambda: True, is_extend=lambda: False)
    batch = NS(
        forward_mode=mode,
        batch_size=3,
        out_cache_loc=None,
        req_pool_indices=None,
        positions=None,
        seq_lens_cpu=None,
    )

    def run():
        inputs = states if layout == "streams" else [x.flatten(1) for x in states]
        assert probe.begin_forward(batch, inputs[0])
        assert probe._features_buf.shape == (3, 256)
        for layer, x in zip([0, 2], inputs):
            probe.capture(layer_id=layer, hidden_states=x, residual=None)
        return probe.finish()

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    states[1].add_(0.5)
    graph.replay()
    result = run()
    torch.testing.assert_close(captured, result)
    reference = []
    for x in states:
        v = x.float().sum(1)
        reference.append(
            (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6))
            .to(x.dtype)
            .float()
        )
    torch.testing.assert_close(
        result, torch.cat(reference, dim=-1), atol=0.016, rtol=0.016
    )
    # Prefill uses the same selected layer inputs and clears earlier scores.
    mode.is_decode = lambda: False
    mode.is_extend = lambda: True
    mode.is_mixed = lambda: False
    mode.is_target_verify = lambda: False
    torch.testing.assert_close(run(), result)
    probe.skip_forward()
    assert probe.last_scores is None
    assert probe._features_buf is None


@pytest.mark.parametrize(
    "override",
    [
        {"pp_size": 2},
        {"attn_cp_size": 2},
        {"dcp_size": 2},
        {"enable_mixed_chunk": True},
        {"enable_layernorm_sp": True},
        {"enable_encoder_swa_bounded_replay": True},
        {"disaggregation_mode": "decode"},
        {"speculative_algorithm": "EAGLE3"},
        {"speculative_algorithm": "EAGLE", "speculative_eagle_topk": 2},
        {
            "speculative_algorithm": "EAGLE",
            "speculative_draft_model_path": "/external/draft",
        },
    ],
)
def test_unsupported_execution_modes_fail_before_loading(monkeypatch, override):
    from sglang.srt.arg_groups import token_probe_hook as hook

    cfg = NS(
        probe_ckpt="/probe",
        pp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        disaggregation_mode="null",
        speculative_algorithm=None,
        speculative_eagle_topk=1,
        speculative_draft_model_path=None,
        model_path="/target",
    )
    for key, value in override.items():
        setattr(cfg, key, value)
    monkeypatch.setattr(hook, "resolving_view", lambda _: cfg)
    monkeypatch.setattr(
        hook,
        "model_config_of",
        lambda _: NS(hf_config=NS(architectures=["DeepseekV4ForCausalLM"])),
    )
    with pytest.raises(ValueError):
        hook.validate_token_probe(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["mlp", "attn"])
def test_checkpoint_heads_load_and_match_reference(tmp_path, kind):
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from sglang.srt.models.token_probe.config import ProbeConfig
    from sglang.srt.models.token_probe.heads import (
        SingProbeAttnModel,
        SingProbeMlpModel,
    )
    from sglang.srt.models.token_probe.loader import load_probe_head

    cfg = {
        "model_type": f"sing_probe_{kind}",
        "hidden_size": 64,
        "base_model_layer_ids": [0, 2],
        "labels": ["a", "b", "c"],
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "head_dim": 32,
    }
    cls = SingProbeMlpModel if kind == "mlp" else SingProbeAttnModel
    head = cls.from_config(ProbeConfig.from_dict(cfg), dtype=torch.bfloat16)
    torch.manual_seed(17)
    weights = {name: torch.randn_like(p) * 0.05 for name, p in head.named_parameters()}
    if kind == "attn":
        q, k, v = weights.pop("proj_qkv.weight").split([64, 32, 32])
        weights.update(
            {
                "proj_q.weight": q.contiguous(),
                "proj_k.weight": k.contiguous(),
                "proj_v.weight": v.contiguous(),
            }
        )
        weights["norm.weight"].fill_(1)
    save_file(weights, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    loaded = load_probe_head(str(tmp_path), dtype=torch.bfloat16, device="cuda").cuda()
    features = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    if kind == "mlp":
        hidden = F.linear(features, loaded.fc1.weight, loaded.fc1.bias)
        expected = F.linear(
            F.gelu(hidden.float()), loaded.fc2.weight.float(), loaded.fc2.bias.float()
        ).sigmoid()
        actual = loaded.forward_features(features)["probe_score"]
    else:
        qkv = loaded.project_fused(features)
        torch.testing.assert_close(qkv, F.linear(features, loaded.proj_qkv.weight))
        query = qkv[:, :64].contiguous()
        attention = torch.randn_like(query)
        projected = F.linear(attention, loaded.o_proj.weight)
        residual = projected.float() + query.float()
        normalized = residual * torch.rsqrt(
            residual.square().mean(-1, keepdim=True) + 1e-6
        )
        expected = F.linear(
            normalized * loaded.norm.weight.float(),
            loaded.classifier.weight.float(),
            loaded.classifier.bias.float(),
        ).sigmoid()
        actual = loaded.classify(attention, query)
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.003)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_attention_runtime_preserves_prefix_with_virtual_slots(tmp_path, monkeypatch):
    import torch.nn.functional as F
    from safetensors.torch import save_file

    cfg = {
        "model_type": "sing_probe_attn",
        "hidden_size": 64,
        "base_model_layer_ids": [0],
        "labels": ["a", "b"],
        "num_attention_heads": 2,
        "head_dim": 32,
    }
    torch.manual_seed(29)
    shapes = {
        "proj_q.weight": (64, 64),
        "proj_k.weight": (32, 64),
        "proj_v.weight": (32, 64),
        "o_proj.weight": (64, 64),
        "norm.weight": (64,),
        "classifier.weight": (2, 64),
        "classifier.bias": (2,),
    }
    weights = {
        name: (torch.randn(shape) * 0.05).to(torch.bfloat16)
        for name, shape in shapes.items()
    }
    weights["norm.weight"].fill_(1)
    save_file(weights, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    args = NS(
        probe_ckpt=str(tmp_path),
        enable_attn_tp_input_scattered=False,
        speculative_algorithm=None,
    )
    monkeypatch.setattr(runtime, "get_server_args", lambda: args)
    monkeypatch.setattr(runtime, "resolved_view", lambda x: x)
    monkeypatch.setattr(runtime, "get_parallel", lambda: NS(attn_tp_rank=0))
    monkeypatch.setenv("SGLANG_ENABLE_TOKEN_PROBE_PREFILL", "1")
    monkeypatch.setenv("SGLANG_ENABLE_TOKEN_PROBE_OVERLAP", "0")
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        probe = runtime.TokenProbe(NS(hidden_size=64, hc_mult=4, num_hidden_layers=1))
    finally:
        torch.set_default_dtype(dtype)
    probe.prepare_device(torch.device("cuda"))
    table = torch.zeros(1, 8, device="cuda", dtype=torch.int32)
    table[0, :4] = torch.tensor([5, 2, 8, 7], device="cuda")
    probe.init_kv_cache(16, table)
    states = torch.randn(4, 4, 64, device="cuda", dtype=torch.bfloat16)
    feature = states.float().sum(1)
    feature = (
        feature * torch.rsqrt(feature.square().mean(-1, keepdim=True) + 1e-6)
    ).to(states.dtype)
    qkv = probe.probe_model.project_fused(feature)
    q, k, v = qkv.split([64, 32, 32], dim=-1)
    attn = F.scaled_dot_product_attention(
        q.view(1, 4, 2, 32).transpose(1, 2),
        k.view(1, 1, 4, 32).expand(1, 2, 4, 32),
        v.view(1, 1, 4, 32).expand(1, 2, 4, 32),
        is_causal=True,
    )
    expected = probe.probe_model.classify(attn.transpose(1, 2).reshape(4, 64), q)
    for decode, lo, hi in [(False, 0, 3), (True, 3, 4)]:
        mode = NS(
            is_decode=lambda: decode,
            is_extend=lambda: not decode,
            is_mixed=lambda: False,
            is_target_verify=lambda: False,
        )
        batch = NS(
            forward_mode=mode,
            batch_size=1,
            out_cache_loc=torch.ones(hi - lo, device="cuda", dtype=torch.int32),
            out_cache_loc_virtual=table[0, lo:hi],
            req_pool_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
            positions=torch.arange(lo, hi, device="cuda"),
            seq_lens_cpu=torch.tensor([hi]),
            extend_seq_lens=torch.tensor([hi - lo], device="cuda", dtype=torch.int32),
            extend_seq_lens_cpu=[hi - lo],
        )
        assert probe.begin_forward(batch, states[lo:hi])
        probe.capture(layer_id=0, hidden_states=states[lo:hi], residual=None)
        actual = probe.finish()
        torch.testing.assert_close(actual, expected[lo:hi], atol=0.008, rtol=0.008)
    torch.testing.assert_close(probe._kv_pool[table[0, :4].long()], qkv[:, 64:])
    assert torch.count_nonzero(probe._kv_pool[1]) == 0
