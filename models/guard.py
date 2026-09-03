"""
Guard Model - Two-layer MLP Classifier

This module implements the lightweight Guard classifier:
- Input: Concatenated hidden states from multiple layers
- Structure: Linear -> Activation -> Dropout -> Linear
- Output: Classification logits
"""

import torch
import torch.nn as nn
from typing import Optional


class GuardMLP(nn.Module):
    """
    Two-layer MLP Classifier for Guardrail Detection

    Architecture:
        Input [batch, seq_len, hidden_dim * num_layers]
        -> Linear1 [batch, seq_len, intermediate_dim]
        -> Activation (GELU or ReLU)
        -> Dropout
        -> Linear2 [batch, seq_len, num_classes]
        -> Output [batch, seq_len, num_classes]

    Args:
        input_dim: Input dimension (hidden_dim * num_layers)
        intermediate_dim: Hidden dimension of intermediate layer
        num_classes: Number of output classes (default: 2 for binary classification)
        dropout: Dropout probability (default: 0.1)
        activation: Activation function ('gelu' or 'relu', default: 'gelu')
        init_bias: Constant init value for the final classification layer's bias
            (fc2.bias). When None, bias is zero-initialized (legacy behavior).
            A negative value makes the Guard start from low positive-class
            predictions, so it learns to "say no" until shown evidence.
    """

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int = 1024,
        num_classes: int = 2,
        dropout: float = 0.1,
        activation: str = 'gelu',
        init_bias: Optional[float] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.init_bias = init_bias

        # First layer: input_dim -> intermediate_dim
        self.fc1 = nn.Linear(input_dim, intermediate_dim)

        # Activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'gelu' or 'relu'.")

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Second layer: intermediate_dim -> num_classes
        self.fc2 = nn.Linear(intermediate_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization

        The final classification head's bias (fc2.bias) is initialized to
        `self.init_bias` when set (a negative value biases early predictions
        toward the negative class); otherwise it stays zero-initialized.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.init_bias is not None:
            nn.init.constant_(self.fc2.bias, self.init_bias)
        else:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
               where input_dim = hidden_dim * num_layers

        Returns:
            logits: Classification logits [batch_size, seq_len, num_classes]
        """
        # First layer
        x = self.fc1(x)  # [batch, seq_len, intermediate_dim]

        # Activation
        x = self.activation(x)

        # Dropout
        x = self.dropout(x)

        # Second layer
        logits = self.fc2(x)  # [batch, seq_len, num_classes]

        return logits

    def __repr__(self) -> str:
        return (
            f"GuardMLP(\n"
            f"  input_dim={self.input_dim},\n"
            f"  intermediate_dim={self.intermediate_dim},\n"
            f"  num_classes={self.num_classes},\n"
            f"  dropout={self.dropout_prob}\n"
            f")"
        )

    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GuardMLPConfig:
    """Configuration for GuardMLP"""

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int = 1024,
        num_classes: int = 2,
        dropout: float = 0.1,
        activation: str = 'gelu',
        init_bias: Optional[float] = None
    ):
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.activation = activation
        self.init_bias = init_bias

    def to_dict(self) -> dict:
        return {
            'input_dim': self.input_dim,
            'intermediate_dim': self.intermediate_dim,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'activation': self.activation,
            'init_bias': self.init_bias,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GuardMLPConfig':
        return cls(**config_dict)


def test_guard_model():
    """Test Guard MLP"""
    print("\n" + "="*80)
    print("Testing Guard MLP")
    print("="*80)

    # Test configuration
    batch_size = 4
    seq_len = 128
    hidden_dim = 3584  # Example: Qwen2.5-72B hidden dimension
    num_layers = 3  # Extract from 3 layers
    input_dim = hidden_dim * num_layers

    # Create model
    config = GuardMLPConfig(
        input_dim=input_dim,
        intermediate_dim=1024,
        num_classes=2,
        dropout=0.1,
        activation='gelu'
    )

    model = GuardMLP(**config.to_dict())

    print(f"\nModel config:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Intermediate dim: {config.intermediate_dim}")
    print(f"  Num classes: {config.num_classes}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Activation: {config.activation}")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)

    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")

    logits = model(x)

    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: [{batch_size}, {seq_len}, {config.num_classes}]")

    # Count parameters
    total_params = model.count_parameters()
    print(f"\nTotal trainable parameters: {total_params:,}")

    # Verify output
    assert logits.shape == (batch_size, seq_len, config.num_classes), \
        f"Output shape mismatch: {logits.shape}"

    print("\n✅ Guard MLP test passed!")
    return True


if __name__ == '__main__':
    test_guard_model()