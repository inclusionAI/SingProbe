"""Isolated transport and CUDA numerical tests; no model weights required.

Run: pytest -q test/registered/unit/models/test_token_probe_port.py
CUDA tests compare production kernels with independent PyTorch references.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"


def load_file(name, relative):
    import sys

    spec = importlib.util.spec_from_file_location(name, SRT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def methods(relative, names, namespace=None):
    """Exercise real methods without importing unrelated serving backends."""
    tree = ast.parse((SRT / relative).read_text())
    selected = [
        n
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(selected) == len(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    scope = {"torch": torch, **(namespace or {})}
    exec(compile(ast.fix_missing_locations(module), str(SRT / relative), "exec"), scope)
    return NS(**{n: scope[n] for n in names})


def request(middle=0):
    return NS(
        token_probe_probs=None,
        inflight_middle_chunks=middle,
        finished=lambda: False,
        is_retracted=False,
    )


def collector(save=False):
    envs = NS(SGLANG_TOKEN_PROBE_SAVE_DIR=NS(get=lambda: "/unused" if save else None))
    m = methods(
        "managers/scheduler_components/batch_result_processor.py",
        [
            "_collect_token_probe_scores_for_request",
            "_record_token_probe_prefill_scores",
            "_record_token_probe_decode_scores",
        ],
        {"envs": envs, "_get_speculative_output_stride": lambda r: r.stride},
    )
    owner = NS(enable_overlap=True, enable_overlap_mlx=False)
    owner._collect_token_probe_scores_for_request = lambda req, row: (
        m._collect_token_probe_scores_for_request(owner, req, row)
    )
    return m, owner


def test_chunked_prefill_only_emits_final_prompt_row():
    m, owner = collector()
    reqs = [request(1), request()]
    result = NS(
        token_probe_scores=torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]]),
        extend_input_len_per_req=[2, 3],
    )
    m._record_token_probe_prefill_scores(owner, NS(reqs=reqs), result, [10, 20])
    assert reqs[0].token_probe_probs is None
    assert reqs[1].token_probe_probs == [[0.5]]


def test_save_keeps_all_prefill_rows_including_middle_chunks():
    m, owner = collector(save=True)
    reqs = [request(1), request()]
    scores = torch.arange(5.0).view(-1, 1)
    result = NS(token_probe_scores=scores, extend_input_len_per_req=[2, 3])
    m._record_token_probe_prefill_scores(owner, NS(reqs=reqs), result, [10, 20])
    torch.testing.assert_close(torch.stack(reqs[0].token_probe_probs), scores[:2])
    torch.testing.assert_close(torch.stack(reqs[1].token_probe_probs), scores[2:])


def test_speculative_rows_follow_padded_stride_and_accepted_lengths():
    m, owner = collector()
    reqs = [request(), request(), request()]
    scores = torch.arange(12.0).view(-1, 1)
    result = NS(token_probe_scores=scores, token_probe_finish_func=None, stride=4)
    batch = NS(reqs=reqs, spec_algorithm=NS(is_none=lambda: False))
    m._record_token_probe_decode_scores(owner, batch, result, [[1], [2, 3, 4], []])
    assert reqs[0].token_probe_probs == [[0.0]]
    assert reqs[1].token_probe_probs == [[4.0], [5.0], [6.0]]
    assert reqs[2].token_probe_probs is None


def test_probe_uses_virtual_cache_ids():
    fn = methods(
        "models/token_probe/token_probe.py", ["_cache_locations"]
    )._cache_locations
    virtual, physical = torch.tensor([100]), torch.tensor([3])
    assert fn(NS(out_cache_loc_virtual=virtual, out_cache_loc=physical)) is virtual
    assert fn(NS(out_cache_loc=physical)) is physical


@pytest.mark.parametrize("verify,rows", [(True, 6), (False, 3)])
def test_padded_graph_output_slicing(verify, rows):
    fn = methods(
        "model_executor/forward_batch_info.py",
        ["post_forward_mlp_sync_batch"],
        {"SpecInputType": NS(EAGLE_DRAFT_EXTEND=1)},
    ).post_forward_mlp_sync_batch
    mode = NS(
        is_decode=lambda: not verify,
        is_idle=lambda: False,
        is_target_verify=lambda: verify,
        is_extend=lambda: False,
    )
    fb = NS(
        _original_forward_mode=None,
        _original_batch_size=None,
        batch_size=3,
        _original_num_tokens=None,
        forward_mode=mode,
        spec_info=NS(num_tokens_per_req=2) if verify else None,
    )
    output = NS(
        next_token_logits=torch.zeros(8, 2),
        hidden_states=None,
        last_hidden_states=None,
        token_probe_scores=torch.arange(8.0).view(-1, 1),
    )
    fn(fb, output)
    assert output.token_probe_scores.shape[0] == rows


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
@pytest.mark.parametrize("hc", [1, 4])
def test_tap_and_graph_replay_match_rms_normalized_residual(hc):
    k = load_file("probe_test_kernels", "models/token_probe/probe_kernels.py")
    torch.manual_seed(7)
    x = torch.randn(
        (3, hc, 128) if hc > 1 else (3, 128), device="cuda", dtype=torch.bfloat16
    )
    residual = None if hc > 1 else torch.randn_like(x)
    out = torch.zeros(3, 256, device="cuda", dtype=x.dtype)

    def run():
        k.tap_into(out, 1, x, residual)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for scale in (1.0, 2.0):
        x.mul_(scale)
        graph.replay()
        value = x.float().sum(1) if hc > 1 else x.float() + residual.float()
        expected = (
            value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        ).to(x.dtype)
        torch.testing.assert_close(out[:, 128:], expected, atol=0.016, rtol=0.016)
        assert torch.count_nonzero(out[:, :128]) == 0


@cuda
@pytest.mark.parametrize("window", [None, 4])
@pytest.mark.parametrize("splits", [1, 4])
def test_paged_attention_with_noncontiguous_slots(window, splits):
    k = load_file("probe_test_attention", "models/token_probe/paged_attention.py")
    torch.manual_seed(19)
    heads, dim, width = 2, 32, 12
    table = torch.randperm(2 * width, device="cuda").view(2, width).to(torch.int32)
    kv = torch.randn(2 * width, 2 * dim, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([4, 5, 6, 8, 9], device="cuda", dtype=torch.int64)
    q = torch.randn(5, heads * dim, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(
        q=q,
        kv_pool=kv,
        req_to_token=table,
        req_indices=torch.tensor([1, 0], device="cuda", dtype=torch.int32),
        positions=positions,
        qo_indptr=torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32),
        max_qo_len=3,
        num_heads=heads,
        head_dim=dim,
        window=window,
        force_splits=splits,
    )
    expected = []
    for row, (req, pos) in enumerate(zip([1, 1, 1, 0, 0], [4, 5, 6, 8, 9])):
        start = max(0, pos + 1 - window) if window else 0
        keys, values = kv[table[req, start : pos + 1].long()].float().chunk(2, -1)
        logits = q[row].float().view(heads, dim) @ keys.T / dim**0.5
        expected.append((logits.softmax(-1) @ values).flatten())
    eager = k.probe_paged_attention(**kwargs)
    torch.testing.assert_close(
        eager.float(), torch.stack(expected), atol=0.025, rtol=0.025
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = k.probe_paged_attention(**kwargs)
    graph.replay()
    torch.testing.assert_close(replay, eager)


def test_prefill_representative_rows_use_request_index():
    m, owner = collector()
    reqs = [request(), request()]
    scores = torch.tensor([[1.0], [2.0]])
    result = NS(token_probe_scores=scores, extend_input_len_per_req=[2, 3])
    m._record_token_probe_prefill_scores(owner, NS(reqs=reqs), result, [10, 20])
    assert reqs[0].token_probe_probs == [[1.0]]
    assert reqs[1].token_probe_probs == [[2.0]]
