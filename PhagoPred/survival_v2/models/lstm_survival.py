"""
LSTM-based survival model (Dynamic DeepHit style).
Supports modular attention mechanisms.
"""
import torch
import torch.nn as nn
from .base import SurvivalModel, build_fc_layers
from ..attention.mechanisms import get_attention_mechanism


class LSTMSurvival(SurvivalModel):
    """
    LSTM + Attention + FC for survival prediction.
    Optionally includes next-timestep feature prediction.
    """

    def __init__(
        self,
        num_features: int,
        num_bins: int,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.0,
        predictor_layers: list[int] = [32],
        fc_layers: list[int] = [64, 64],
        attention_type: str = 'multihead',
        attention_config: dict = None,
        use_predictor: bool = False,
        use_mask_embedding: bool = False,
        **kwargs
    ):
        """
        Args:
            num_features: number of input features per timestep
            num_bins: number of discrete time bins for survival prediction
            lstm_hidden_size: LSTM hidden dimension
            lstm_num_layers: number of LSTM layers
            lstm_dropout: dropout between LSTM layers
            predictor_layers: hidden layers for feature predictor
            fc_layers: hidden layers for survival prediction head
            attention_type: 'vector', 'fc', 'multihead', 'mean_pool', 'last_pool'
            attention_config: dict of kwargs for attention mechanism
            use_predictor: whether to predict next timestep features
            use_mask_embedding: whether to concat feature mask to input
        """
        super().__init__(num_bins)

        self.num_features = num_features
        self.use_predictor = use_predictor
        self.use_mask_embedding = use_mask_embedding

        # Input size
        input_size = num_features
        if use_mask_embedding:
            input_size *= 2

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            batch_first=True
        )

        # Optional: predict next timestep features
        if use_predictor:
            self.predictor = build_fc_layers(
                input_size=lstm_hidden_size,
                output_size=num_features,
                layer_sizes=predictor_layers
            )

        # Attention mechanism
        if attention_config is None:
            attention_config = {}
        attention_config['embed_dim'] = lstm_hidden_size
        self.attention = get_attention_mechanism(attention_type, **attention_config)

        # Survival prediction head
        self.fc = build_fc_layers(
            input_size=lstm_hidden_size * 2,
            output_size=num_bins,
            layer_sizes=fc_layers
        )

        # Optional mask embedding
        if use_mask_embedding:
            self.mask_embedding = nn.Parameter(torch.zeros(num_features))

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"LSTMSurvival initialized with {num_params:,} parameters")
        print(f"  Attention type: {attention_type}")

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        return_attention: bool = False,
        mask: torch.Tensor = None
    ):
        """
        Args:
            x: (batch, seq_len, num_features)
            lengths: (batch,) - valid sequence lengths
            return_attention: whether to return attention weights
            mask: (batch, seq_len, num_features) - feature validity mask

        Returns:
            outputs: (batch, num_bins) - survival logits
            y_pred: (batch, seq_len, num_features) - predicted features (if use_predictor)
            attn_weights: (batch, seq_len) - attention weights (if return_attention)
        """
        batch_size, seq_len, num_features = x.shape

        # Optional: concat mask as additional features
        if self.use_mask_embedding and mask is not None:
            x = torch.cat([x, mask.float()], dim=-1)

        # Padding mask for attention
        padding_mask = torch.arange(seq_len, device=lengths.device).unsqueeze(0)
        padding_mask = padding_mask < lengths.unsqueeze(1)

        # LSTM encoding with packed sequences
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, (h_n, c_n) = self.lstm(x_packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )

        # Optional: predict next timestep features
        y_pred = None
        if self.use_predictor:
            y_pred = self.predictor(lstm_out)

        # Attention pooling
        context, attn_weights = self.attention(lstm_out, padding_mask)

        # Survival prediction
        outputs = self.fc(torch.concat((context, h_n[-1]), dim=-1))
        # Return based on flags
        if return_attention:
            if self.use_predictor:
                return outputs, y_pred, attn_weights
            else:
                return outputs, attn_weights
        else:
            if self.use_predictor:
                return outputs, y_pred
            else:
                return outputs
