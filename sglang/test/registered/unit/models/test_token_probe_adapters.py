"""Run the production adapter loops with small deterministic decoder layers."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

SRT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def method(file, cls, name, scope):
    tree = ast.parse((SRT / "models" / file).read_text())
    body = next(
        n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
    )
    fn = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    scope = {"torch": torch, **scope}
    exec(compile(ast.fix_missing_locations(module), file, "exec"), scope)
    return scope[name]


class Probe:
    def __init__(self):
        self.enabled = True
        self.tap_slots = {0: 0, 1: 1}
        self.rows = []
        self.last_scores = None

    def begin_forward(self, batch, states):
        self.rows.clear()
        self.last_scores = None
        return True

    def capture(self, *, layer_id, hidden_states, residual):
        assert residual is None
        self.rows.append((layer_id, hidden_states.clone()))

    def finish(self):
        self.last_scores = torch.cat([row for _, row in self.rows], dim=-1)


class Aux(list):
    def capture(self, value, *, owned=False):
        self.append(value.clone())


def common():
    return {
        "AuxHiddenStateList": Aux,
        "get_global_expert_distribution_recorder": lambda: NS(
            with_current_layer=lambda _: nullcontext()
        ),
    }


def test_qwen_taps_completed_streams_before_ple():
    probe = Probe()
    embedded = torch.arange(8.0).reshape(2, 4)
    # A producer may defer its communication until the next layer enters.
    class Deferred:
        def __init__(self, value):
            self.value = value

    def complete(value, batch):
        return value.value if isinstance(value, Deferred) else value

    class Layer:
        ple = None
        attn_hyper_connection = NS(hc_norm=None)

        def hc_norms(self, normalized_input, next_norm):
            return nullcontext(NS(output=None))

        def __call__(self, *, hidden_states, **kwargs):
            # Emulate a PLE update before attention. The tap must precede it.
            value = hidden_states if hidden_states.shape[-1] == 8 else hidden_states.repeat(1, 2)
            return Deferred(value + 100)

    batch = NS(forward_mode=NS(is_extend=lambda: True, is_idle=lambda: False))
    owner = NS(
        pp_group=NS(is_first_rank=True, is_last_rank=True),
        embed_tokens=lambda _: embedded,
        has_ple=False, start_layer=0, end_layer=2, layers=[Layer(), Layer()],
        hyper_connection_mixer=NS(hc_norm=None), token_probe=probe,
    )
    fn = method("qwen4_exp.py", "Qwen4ExpModel", "forward", {
        **common(),
        "residual_batch": NS(
            start=lambda _: None, complete_output=complete,
            final_norm=lambda states, batch, norm: complete(states, batch),
        ),
        "_shards_rows_over_cp": lambda _: False,
        "_commit_ple_batch": lambda **kwargs: None,
        "_MixStreams": lambda *args: None,
    })
    result, streams = fn(owner, torch.tensor([1, 2]), torch.arange(2), batch)
    assert isinstance(streams, Deferred)
    torch.testing.assert_close(result, embedded.repeat(1, 2) + 200)
    torch.testing.assert_close(probe.rows[0][1], embedded)
    torch.testing.assert_close(probe.rows[1][1], embedded.repeat(1, 2) + 100)
    assert probe.last_scores.shape == (2, 12)


@pytest.mark.parametrize("mhc", [False, True])
def test_glm_capture_keeps_aux_states_and_bypasses_tbo(mhc):
    probe = Probe()
    embedded = torch.arange(8.0).reshape(2, 4)

    class Layer:
        def __call__(self, pos, hidden, batch, *args, capture_output=None, **kwargs):
            if capture_output:
                capture_output(hidden, owned=True)
            if mhc and hidden.shape[-1] == 4:
                hidden = hidden.repeat(1, 2)
            return hidden + 10, None

    owner = NS(
        pp_group=NS(is_first_rank=True, is_last_rank=True),
        embed_tokens=lambda _: embedded, start_layer=0, end_layer=2,
        first_k_dense_replace=1, dflash_capture=False, layers=[Layer(), Layer()],
        layers_to_capture=[1], enable_a2a_moe=False, token_probe=probe,
        _prepare_aux_hidden_state=lambda value: value, norm=None,
    )
    batch = NS(can_run_tbo=True, forward_mode=NS(is_idle=lambda: False))
    fn = method("glm5_next.py", "Glm5NextModel", "forward", {
        **common(), "nullcontext": nullcontext,
        "BumpAllocator": lambda **kwargs: None,
        "check_cuda_graph_backend": lambda *args: False,
        "Phase": NS(PREFILL=None), "Backend": NS(TC_PIECEWISE=None),
        "residual_batch": NS(
            start=lambda _: None,
            complete_output=lambda x, batch: x,
            final_norm=lambda x, batch, norm: x,
        ),
        # Any TBO call would miss the second selected layer.
        "model_forward_stages": lambda **kwargs: pytest.fail("probe entered TBO"),
    })
    result, aux = fn(owner, torch.tensor([1, 2]), torch.arange(2), batch)
    expected = embedded.repeat(1, 2) if mhc else embedded
    torch.testing.assert_close(result, expected + 20)
    torch.testing.assert_close(probe.rows[1][1], expected + 10)
    torch.testing.assert_close(aux[0], expected + 10)
    assert len(aux) == 1
    assert [i for i, _ in probe.rows] == [0, 1]


@pytest.mark.parametrize("architecture", [
    "Qwen4ExpForConditionalGeneration", "Glm5NextForConditionalGeneration",
])
def test_new_architectures_validate(monkeypatch, architecture):
    from sglang.srt.arg_groups import token_probe_hook as hook
    cfg = NS(
        probe_ckpt="/probe", pp_size=1, attn_cp_size=1, dcp_size=1,
        disaggregation_mode="null", speculative_algorithm=None,
    )
    monkeypatch.setattr(hook, "resolving_view", lambda _: cfg)
    monkeypatch.setattr(
        hook, "model_config_of", lambda _: NS(hf_config=NS(architectures=[architecture]))
    )
    hook.validate_token_probe(cfg)
    cfg.speculative_algorithm = "EAGLE"
    with pytest.raises(ValueError, match="non-speculative"):
        hook.validate_token_probe(cfg)
