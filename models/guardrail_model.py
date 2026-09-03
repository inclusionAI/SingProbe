"""
Complete Guardrail Model (Base Model + Guard Model)

This module implements the complete guardrail model:
1. Base Model extracts hidden states from specified layers
2. Concatenate hidden states from multiple layers
3. Guard Model performs classification
4. Output: token-level classification logits
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .base_model import BaseModelWrapper
from .guard import GuardMLP


class GuardrailModel(nn.Module):
    """
    Complete Guardrail Model for Multi-task Classification

    Architecture:
        Input: [batch, seq_len]
        -> Base Model (frozen, TP distributed)
        -> Extract hidden states from layers [10, 22, 35]
        -> Concatenate: [batch, seq_len, hidden_dim * num_layers]
        -> Guard MLP
        -> Output: [batch, seq_len, 10]  (8 Query + 2 Response tasks)

    Multi-task outputs:
        - Dimensions 0-7: Query risk categories (multi-label)
        - Dimension 8: Response safety (binary)
        - Dimension 9: Response hallucination (binary)
    """

    def __init__(
        self,
        base_model_name: str,
        hidden_layers: List[int] = [10, 22, 35],
        intermediate_dim: int = 1024,
        num_classes: int = 10,  # 8 Query + 2 Response
        shard_gpus: int = 0,
        dtype: torch.dtype = torch.float16,
        dropout: float = 0.1,
        activation: str = 'gelu',
        device: Optional[torch.device] = None,
        **base_model_kwargs
    ):
        """
        Initialize Guardrail Model

        Args:
            base_model_name: HuggingFace model name or path
            hidden_layers: Layers to extract hidden states from (default: [10, 22, 35])
            intermediate_dim: Intermediate dimension for Guard MLP
            num_classes: Number of output classes (default: 10 for 8+2 tasks)
            shard_gpus: Number of GPUs to pipeline-shard the base model across
                (HuggingFace device_map). 0 = all visible GPUs. NOT tensor
                parallel (the DeepSpeed TP path has been retired).
            dtype: Model dtype
            dropout: Dropout probability
            activation: Activation function for Guard MLP
            device: Device to use
            **base_model_kwargs: Additional arguments for Base Model
        """
        super().__init__()

        self.base_model_name = base_model_name
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.num_hidden_layers = len(hidden_layers)

        # Initialize Base Model
        print("="*80)
        print("Initializing Base Model...")
        print("="*80)
        self.base_model = BaseModelWrapper(
            model_name=base_model_name,
            hidden_layers=hidden_layers,
            shard_gpus=shard_gpus,
            dtype=dtype,
            device=device,
            **base_model_kwargs
        )

        # Get hidden dimension
        self.hidden_dim = self.base_model.get_hidden_dim()
        print(f"\nBase Model hidden dimension: {self.hidden_dim}")
        print(f"Extracting from {len(hidden_layers)} layers: {hidden_layers}")

        # Calculate input dimension for Guard
        self.guard_input_dim = self.hidden_dim * self.num_hidden_layers
        print(f"Guard input dimension: {self.guard_input_dim}")

        # Initialize Guard Model
        print("\n" + "="*80)
        print("Initializing Guard Model...")
        print("="*80)
        self.guard = GuardMLP(
            input_dim=self.guard_input_dim,
            intermediate_dim=intermediate_dim,
            num_classes=num_classes,
            dropout=dropout,
            activation=activation
        )

        print(f"\nGuard Model initialized:")
        print(f"  Input dim: {self.guard_input_dim}")
        print(f"  Intermediate dim: {intermediate_dim}")
        print(f"  Output classes: {num_classes}")
        print(f"  Trainable params: {self.guard.count_parameters():,}")

        # Move Guard to device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.guard = self.guard.to(device)

        print(f"\n✅ Guardrail Model initialized successfully!")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_hidden_states: Whether to return hidden states
            **kwargs: Additional arguments for Base Model

        Returns:
            Dict containing:
                - logits: Classification logits [batch_size, seq_len, num_classes]
                - hidden_states: (optional) Dict of hidden states {layer_idx: tensor}
        """
        # Step 1: Extract hidden states from Base Model
        hidden_states_dict = self.base_model(input_ids, attention_mask, **kwargs)

        # hidden_states_dict: {layer_idx: [batch, seq_len, hidden_dim]}

        # Step 2: Concatenate hidden states from specified layers
        # Order by layer index to ensure consistency
        hidden_states_list = [hidden_states_dict[idx] for idx in sorted(self.hidden_layers)]

        # Concatenate along last dimension
        # [batch, seq_len, hidden_dim * num_layers]
        concat_hidden = torch.cat(hidden_states_list, dim=-1)

        # Step 3: Guard Model classification
        logits = self.guard(concat_hidden)
        # [batch, seq_len, num_classes]

        # Prepare output
        output = {'logits': logits}

        if return_hidden_states:
            output['hidden_states'] = hidden_states_dict

        return output

    def get_logits_split(
        self,
        logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split logits into Query and Response tasks

        Args:
            logits: Full logits [batch, seq_len, 10]

        Returns:
            query_logits: Query risk logits [batch, seq_len, 8]
            safety_logits: Response safety logits [batch, seq_len, 1]
            hallu_logits: Response hallucination logits [batch, seq_len, 1]
        """
        # Split logits
        query_logits = logits[:, :, :8]      # Dimensions 0-7
        safety_logits = logits[:, :, 8:9]    # Dimension 8
        hallu_logits = logits[:, :, 9:10]    # Dimension 9

        return query_logits, safety_logits, hallu_logits

    def freeze_base_model(self):
        """Freeze Base Model parameters (already frozen in initialization)"""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        """Unfreeze Base Model parameters (not recommended for Guard training)"""
        print("Warning: Unfreezing Base Model is not recommended for Guard training!")
        for param in self.base_model.parameters():
            param.requires_grad = True

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters"""
        base_params = sum(p.numel() for p in self.base_model.parameters())
        guard_params = sum(p.numel() for p in self.guard.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'base_model': base_params,
            'guard_model': guard_params,
            'total': base_params + guard_params,
            'trainable': trainable_params
        }

    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            'base_model_name': self.base_model_name,
            'hidden_layers': self.hidden_layers,
            'hidden_dim': self.hidden_dim,
            'guard_input_dim': self.guard_input_dim,
            'intermediate_dim': self.guard.intermediate_dim,
            'num_classes': self.num_classes,
            'num_hidden_layers': self.num_hidden_layers,
        }

    def __repr__(self) -> str:
        config = self.get_config()
        params = self.count_parameters()

        return (
            f"GuardrailModel(\n"
            f"  Base Model: {config['base_model_name']}\n"
            f"    Layers: {config['hidden_layers']}\n"
            f"    Hidden dim: {config['hidden_dim']}\n"
            f"    Params: {params['base_model']:,}\n"
            f"  Guard Model:\n"
            f"    Input dim: {config['guard_input_dim']}\n"
            f"    Intermediate dim: {config['intermediate_dim']}\n"
            f"    Output classes: {config['num_classes']}\n"
            f"    Params: {params['guard_model']:,}\n"
            f"  Total params: {params['total']:,}\n"
            f"  Trainable params: {params['trainable']:,}\n"
            f")"
        )


def create_guardrail_model(
    base_model_name: str = "Qwen/Qwen2.5-72B-Instruct",
    hidden_layers: List[int] = [10, 22, 35],
    shard_gpus: int = 0,
    **kwargs
) -> GuardrailModel:
    """
    Factory function to create Guardrail Model

    Args:
        base_model_name: Base model name
        hidden_layers: Layers to extract
        shard_gpus: GPUs to pipeline-shard the base model across (device_map).
            0 = all visible GPUs. Not tensor parallel.
        **kwargs: Additional arguments

    Returns:
        GuardrailModel instance
    """
    return GuardrailModel(
        base_model_name=base_model_name,
        hidden_layers=hidden_layers,
        shard_gpus=shard_gpus,
        **kwargs
    )


def test_guardrail_model():
    """Test Guardrail Model"""
    print("\n" + "="*80)
    print("Testing Guardrail Model")
    print("="*80)

    # Test configuration (use small model)
    model_name = "gpt2"  # Small model for testing
    hidden_layers = [0, 1, 2]  # Test first 3 layers
    batch_size = 2
    seq_len = 10

    try:
        # Create model
        model = GuardrailModel(
            base_model_name=model_name,
            hidden_layers=hidden_layers,
            intermediate_dim=256,  # Smaller for testing
            num_classes=10,
            shard_gpus=1,  # Single GPU for testing
            dtype=torch.float32
        )

        print(f"\n{model}")

        # Test forward pass
        print("\n" + "="*80)
        print("Testing forward pass...")
        print("="*80)

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        print(f"Input shape: {input_ids.shape}")

        output = model(input_ids, return_hidden_states=True)

        print(f"Output logits shape: {output['logits'].shape}")
        print(f"Hidden states keys: {list(output['hidden_states'].keys())}")

        for layer_idx, hidden in output['hidden_states'].items():
            print(f"  Layer {layer_idx}: {hidden.shape}")

        # Test logits split
        query_logits, safety_logits, hallu_logits = model.get_logits_split(output['logits'])

        print(f"\nLogits split:")
        print(f"  Query logits: {query_logits.shape}")
        print(f"  Safety logits: {safety_logits.shape}")
        print(f"  Hallu logits: {hallu_logits.shape}")

        # Count parameters
        params = model.count_parameters()
        print(f"\nParameters:")
        for key, value in params.items():
            print(f"  {key}: {value:,}")

        print("\n✅ Guardrail Model test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Guardrail Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_guardrail_model()