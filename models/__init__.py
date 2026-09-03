"""
Model initialization for SingProbe

This module provides:
1. Base Model (frozen, inference-only; HuggingFace device_map sharding)
2. Guard Model (2-layer MLP or attn token probe)
3. Complete Guardrail Model (Base + Guard)
"""

# Import Guard first (no dependencies)
from .guard import GuardMLP, GuardMLPConfig
from .sglang_attn import GuardAttnProbe, GuardAttnProbeNet

# Lazy imports for models with heavy dependencies (transformers, etc.)
def get_base_model():
    """Lazy import of BaseModelWrapper."""
    from .base_model import BaseModelWrapper
    return BaseModelWrapper

def get_guardrail_model():
    """Lazy import of GuardrailModel."""
    from .guardrail_model import GuardrailModel
    return GuardrailModel

# For backward compatibility
BaseModelWrapper = None
GuardrailModel = None

__all__ = [
    'GuardMLP',
    'GuardMLPConfig',
    'GuardAttnProbe',
    'GuardAttnProbeNet',
    'BaseModelWrapper',
    'GuardrailModel',
    'get_base_model',
    'get_guardrail_model'
]
