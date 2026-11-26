import pytest
from sglang.srt.models.token_probe.config import (
    SING_PROBE_ATTN_MODEL_TYPE,
    SING_PROBE_IDENTITY_MODEL_TYPE,
    SING_PROBE_MLP_MODEL_TYPE,
    ProbeConfig,
)


def test_mlp_probe_config_resolves_canonical_shape():
    config = ProbeConfig.from_dict(
        {
            "model_type": SING_PROBE_MLP_MODEL_TYPE,
            "hidden_size": 7168,
            "base_model_layer_ids": [2, 39, 76],
            "labels": ["safe", "jailbreak"],
            "intermediate_size": 256,
        }
    )

    assert config.input_size == 3 * 7168
    assert config.num_labels == 2
    assert config.base_model_layer_ids == (2, 39, 76)


def test_attention_probe_config_defaults_and_synthetic_labels():
    config = ProbeConfig.from_dict(
        {
            "model_type": SING_PROBE_ATTN_MODEL_TYPE,
            "hidden_size": 7168,
            "base_model_layer_ids": [20],
            "num_labels": 3,
        }
    )

    assert config.labels == ("label_0", "label_1", "label_2")
    assert config.num_attention_heads == 4
    assert config.head_dim == 64


def test_identity_probe_does_not_require_labels():
    config = ProbeConfig.from_dict(
        {
            "model_type": SING_PROBE_IDENTITY_MODEL_TYPE,
            "base_model_layer_ids": [0],
        }
    )

    assert config.labels == ()
    assert config.hidden_size is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"model_type": SING_PROBE_MLP_MODEL_TYPE, "num_labels": 0},
        {"model_type": SING_PROBE_MLP_MODEL_TYPE},
    ],
)
def test_invalid_probe_config_fails(payload):
    with pytest.raises(ValueError):
        ProbeConfig.from_dict(payload)
