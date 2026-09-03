"""
Data module for SingProbe

This module handles dataset processing for two types of datasets:
1. Safety: Query risk classification (7 classes, multi-label) + Query Safe,
   plus Response safety (binary), all broadcast onto Response tokens
2. Response_Hallu: Token-level hallucination detection
"""

from .dataset import GuardrailDataset
from .collator import GuardrailCollator

__all__ = ['GuardrailDataset', 'GuardrailCollator']