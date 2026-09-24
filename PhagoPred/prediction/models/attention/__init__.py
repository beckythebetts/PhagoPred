"""
Attention mechanisms for temporal sequences.
"""
from .mechanisms import (
    AttentionVector,
    FCAttention,
    MultiHeadSelfAttention,
    MeanPooling,
    LastPooling,
    get_attention_mechanism
)

__all__ = [
    'AttentionVector',
    'FCAttention',
    'MultiHeadSelfAttention',
    'MeanPooling',
    'LastPooling',
    'get_attention_mechanism'
]
