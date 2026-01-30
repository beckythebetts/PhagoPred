"""
Attention mechanisms for temporal sequence models.
All mechanisms handle variable-length sequences via padding masks.
"""
import torch
import torch.nn as nn


class AttentionVector(nn.Module):
    """
    Simplest attention: learnable vector that computes attention weights.
    Most parameter-efficient option.

    Parameters: embed_dim (just one vector)
    """
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__()
        self.attention_vector = nn.Parameter(torch.randn(embed_dim))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> tuple:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            padding_mask: (batch, seq_len) - True for valid positions

        Returns:
            context: (batch, embed_dim)
            weights: (batch, seq_len)
        """
        # Compute attention scores
        attn_weights = torch.einsum("btc,c->bt", x, self.attention_vector)

        # Mask padding
        if padding_mask is not None:
            attn_weights = attn_weights.masked_fill(~padding_mask, -1e9)

        # Softmax and weighted sum
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)

        return context, attn_weights


class FCAttention(nn.Module):
    """
    FC-based attention: small MLP computes attention scores.
    More expressive than attention vector, fewer params than self-attention.

    Parameters: ~embed_dim * hidden_dims
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dims: list[int] = [64, 32],
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()

        layers = []
        in_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.attention_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> tuple:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            padding_mask: (batch, seq_len) - True for valid positions

        Returns:
            context: (batch, embed_dim)
            weights: (batch, seq_len)
        """
        # Compute attention scores
        attn_weights = self.attention_net(x).squeeze(-1)  # (batch, seq_len)

        # Mask padding
        if padding_mask is not None:
            attn_weights = attn_weights.masked_fill(~padding_mask, -1e9)

        # Softmax and weighted sum
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)

        return context, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention (standard Transformer-style).
    Most parameters, most expressive, but may overfit with limited data.

    Parameters: ~4 * embed_dim^2 (Q, K, V, output projections)
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 2,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> tuple:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            padding_mask: (batch, seq_len) - True for valid positions

        Returns:
            context: (batch, embed_dim)
            weights: (batch, seq_len)
        """
        # MultiheadAttention expects key_padding_mask as True for INVALID positions
        key_padding_mask = ~padding_mask if padding_mask is not None else None

        # Self-attention
        attn_output, attn_weights = self.attention(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )

        # Pool over sequence (mean pooling)
        context = attn_output.mean(dim=1)

        # Average attention weights over heads
        attn_weights = attn_weights.mean(dim=1) if len(attn_weights.shape) == 3 else attn_weights

        return context, attn_weights


class MeanPooling(nn.Module):
    """
    No attention - just mean pooling over valid timesteps.
    Baseline with zero parameters.
    """
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> tuple:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            padding_mask: (batch, seq_len) - True for valid positions

        Returns:
            context: (batch, embed_dim)
            weights: (batch, seq_len) - uniform over valid positions
        """
        if padding_mask is not None:
            # Masked mean
            valid_mask = padding_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            context = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

            # Uniform weights over valid positions
            weights = valid_mask.squeeze(-1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            # Simple mean
            context = x.mean(dim=1)
            weights = torch.ones(x.shape[0], x.shape[1], device=x.device) / x.shape[1]

        return context, weights


class LastPooling(nn.Module):
    """
    No attention - use the last valid timestep.
    Baseline with zero parameters, good for LSTMs.
    """
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> tuple:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            padding_mask: (batch, seq_len) - True for valid positions

        Returns:
            context: (batch, embed_dim)
            weights: (batch, seq_len) - 1.0 at last valid position
        """
        batch_size, seq_len, embed_dim = x.shape

        if padding_mask is not None:
            # Find last valid index per sequence
            lengths = padding_mask.sum(dim=1)  # (batch,)
            last_indices = (lengths - 1).clamp(min=0)
        else:
            last_indices = torch.full((batch_size,), seq_len - 1, device=x.device)

        # Gather last timesteps
        last_indices = last_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, embed_dim)
        context = torch.gather(x, 1, last_indices).squeeze(1)

        # One-hot weights at last position
        weights = torch.zeros(batch_size, seq_len, device=x.device)
        weights.scatter_(1, last_indices[:, 0, 0].unsqueeze(1), 1.0)

        return context, weights


# Registry for easy access
ATTENTION_MECHANISMS = {
    'vector': AttentionVector,
    'fc': FCAttention,
    'multihead': MultiHeadSelfAttention,
    'mean_pool': MeanPooling,
    'last_pool': LastPooling,
}


def get_attention_mechanism(name: str, **kwargs) -> nn.Module:
    """
    Factory function to create attention mechanisms.

    Args:
        name: one of 'vector', 'fc', 'multihead', 'mean_pool', 'last_pool'
        **kwargs: passed to the attention mechanism constructor

    Returns:
        attention_module: nn.Module

    Example:
        >>> attn = get_attention_mechanism('fc', embed_dim=64, hidden_dims=[32, 16])
    """
    if name not in ATTENTION_MECHANISMS:
        raise ValueError(f"Unknown attention mechanism: {name}. Choose from {list(ATTENTION_MECHANISMS.keys())}")

    return ATTENTION_MECHANISMS[name](**kwargs)
