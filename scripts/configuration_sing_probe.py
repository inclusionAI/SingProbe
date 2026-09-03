from transformers import PretrainedConfig


class SingProbeMlpConfig(PretrainedConfig):
    model_type = "sing_probe_mlp"

    def __init__(
        self,
        hidden_size: int = 2560,
        base_model_layer_ids: list[int] | None = None,
        intermediate_size: int = 1024,
        num_labels: int = 10,
        hidden_act: str = "gelu",
        base_model_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.base_model_layer_ids = base_model_layer_ids or []
        self.intermediate_size = int(intermediate_size)
        self.num_labels = int(num_labels)
        self.hidden_act = hidden_act
        self.base_model_name = base_model_name

    @property
    def input_size(self) -> int:
        return self.hidden_size * len(self.base_model_layer_ids)


class SingProbeAttnConfig(PretrainedConfig):
    model_type = "sing_probe_attn"

    def __init__(
        self,
        hidden_size: int = 2560,
        base_model_layer_ids: list[int] | None = None,
        num_attention_heads: int = 4,
        head_dim: int = 64,
        sliding_window: int | None = None,
        num_labels: int = 10,
        base_model_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.base_model_layer_ids = base_model_layer_ids or []
        self.num_attention_heads = int(num_attention_heads)
        self.head_dim = int(head_dim)
        self.sliding_window = None if sliding_window is None else int(sliding_window)
        self.num_labels = int(num_labels)
        self.base_model_name = base_model_name

    @property
    def input_size(self) -> int:
        return self.hidden_size * len(self.base_model_layer_ids)
