import torch

from PhagoPred.survival_analysis.models import losses

def build_fc_layers(input_size,
                    output_size,
                    layer_sizes: list[int] = [],
                    dropout: float=0.0,
                    activation=torch.nn.ReLU,
                    batch_norm: bool=False,
                    final_layer_activation: bool=False):
    modules = []
    dim_size = input_size
    for layer_size in layer_sizes:
        modules.append(torch.nn.Linear(dim_size, layer_size))
        if batch_norm:
            modules.append(torch.nn.BatchNorm1d(layer_size))
        modules.append(activation())
        if dropout > 0:
            modules.append(torch.nn.Dropout(dropout))
        dim_size = layer_size
    modules.append(torch.nn.Linear(dim_size, output_size))
    if final_layer_activation is True:
        modules.append(activation())
    return torch.nn.Sequential(*modules)


class DynamicDeepHit(torch.nn.Module):
    """A dynamic deep hit survival analysis model (https://ieeexplore.ieee.org/document/8681104).
    Only one 'cause' (no competing risks). Fixed time between each time step. LSTM + attention as shared subnetwork.
    The covariates of the final observation are NOT used in the attention mechanism."""

    def __init__(self, 
                 num_features: int, 
                 output_size: int,
                 lstm_hidden_size: int = 64,
                 lstm_dropout: float = 0.0,
                 predictor_layers: list[int] = [32],
                 attention_layers: list[int] = [64, 64],
                 fc_layers: list[int] = [64, 64],
                 mask: bool = True,
                 **kwargs
    ):
        super().__init__()
        input_size = num_features
        if mask:
            input_size *= 2
        self.lstm = torch.nn.LSTM(input_size=input_size, 
                                  hidden_size=lstm_hidden_size, 
                                  num_layers=2,
                                  dropout=lstm_dropout,
                                  batch_first=True,
                                #   bidirectional=True,
                                  )

        self.predictor = build_fc_layers(input_size=lstm_hidden_size, output_size=num_features, layer_sizes=predictor_layers)
        # self.attention = build_fc_layers(input_size=lstm_hidden_size, output_size=1, layer_sizes=attention_layers)
        self.attention = torch.nn.MultiheadAttention(embed_dim=lstm_hidden_size, num_heads=2, dropout=lstm_dropout, batch_first=True)

        self.fc = build_fc_layers(input_size=lstm_hidden_size, output_size=output_size, layer_sizes=fc_layers)
        
        self.mask_embedding = torch.nn.parameter.Parameter(torch.zeros(num_features)).unsqueeze(0).unsqueeze(0)
        self.pool_query = torch.nn.Parameter(torch.randn(1, 1, lstm_hidden_size))
        
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model loaded. Trainable parameters: {num_trainable_params}")



    def forward(self, x, lengths, return_attention: bool = False, mask=None):
        """
        Args:
            x: [batch_size, sequence length, num_features]
            mask: [batch_size, sequence length, num_features] boolean mask indicating valid time steps
        """
        batch_size, seq_len, num_features = x.size()
        # print(batch_size, seq_len, num_features)
        
        if mask is not None:

            # mask_embedding_exp = self.mask_embedding.expand(batch_size, seq_len, num_features).to(device=x.device)
            # x = torch.where(mask.bool(), x, mask_embedding_exp)
            pass
        
        padding_mask = torch.arange(seq_len, device=lengths.device).unsqueeze(0) 
        # padding_mask = padding_mask >= (seq_len - lengths).unsqueeze(1) 
        padding_mask = padding_mask < lengths.unsqueeze(1) 
            
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out (batch_size, seq_len, lstm_hidden_size)
        lstm_out = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)[0]
        y = self.predictor(lstm_out)  # y (batch_size, seq_length, num_features)
        # y = y*padding_mask
        # lstm_hidden = lstm_hidden.permute(1, 0, 2)  # (batch_size, seq_length, lstm_hidden_size)
        
        # ===Attention mechanism===
        # attn_weights = self.attention(lstm_out).squeeze(-1) # (batch, seq_len)
        # attn_weights = attn_weights.masked_fill(padding_mask, -1e9)
        # attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)  # (batch, seq_len)
        # context_vector = torch.sum(attn_weights.unsqueeze(-1) * lstm_out, dim=1)  # (batch, lstm_hidden_size)
        
        # ===Multihead attention===
        attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=~padding_mask)  # attn_output: (batch, seq_len, lstm_hidden_size)
        attn_out = attn_output.mean(1)  # (batch, lstm_hidden_size)
        # attn_out = lstm_out * attn_output  # (batch, seq_len, lstm_hidden_size)
        # attn_out = attn_out.sum(dim=1) / lengths.unsqueeze(-1)
        attn_weights = attn_weights.mean(dim=1).squeeze(1) # (batch, seq_len)
        
        output = self.fc(attn_out)  # (batch, output_size)
        # output = torch.nn.functional.softmax(output, dim=-1)  # (batch, output_size)
        # output = torch.nn.functional.sigmoid(output) #output sigmoid o foutputting CIF
        if return_attention:
            return output, y, attn_weights
        else:
            return output, y


# === Model outputs pmf ===
def estimated_pmf(outputs: torch.Tensor) -> torch.Tensor:
    """Probability mass function from Dynamic DeepHit outputs.
    Args:
        outputs: (batch_size, num_time_bins) - predicted probabilities for each time bin
        
    Returns: 
        PMF: (torch.Tensor) - Probabality of event occuring at time indexed by dim=1
    """
    pmf = torch.nn.functional.softmax(outputs, dim=1)
    return pmf

# def estimated_cif(
#     outputs: torch.Tensor,
#     # t: torch.Tensor
#     ) -> torch.Tensor:
#     """Cumulative incidence function from Dynamic DeepHit outputs.
#     Args:
#         outputs: (batch_size, num_time_bins) - predicted probabilities for each time bin
#         t: (batch_size,) - true event/censoring times (as indices of time bins)
        
#     Returns: 
#         CIF: (torch.Tensor) - Probabality of event occuring on or before time indexed by dim=1
#     """
#     pmf = estimated_pmf(outputs)
#     cif = torch.cumsum(pmf, dim=1)
#     return cif

# === Model outputs hazards ===
# def estimated_cif(
#     outputs: torch.Tensor,
#     # t: torch.Tensor
#     ) -> torch.Tensor:
#     """Cumulative incidence function from Dynamic DeepHit outputs.
#     Args:
#         outputs: (batch_size, num_time_bins) - predicted probabilities for each time bin
#         t: (batch_size,) - true event/censoring times (as indices of time bins)
        
#     Returns: 
#         CIF: (torch.Tensor) - Probabality of event occuring on or before time indexed by dim=1
#     """
    
#     hazards = torch.nn.functional.sigmoid(outputs)
#     sf = torch.cumprod(1 - hazards, dim=1)
#     cif = 1 - sf       
    
#     return cif

# def estimated_pmf(outputs: torch.Tensor) -> torch.Tensor:
#     """Probability mass function from Dynamic DeepHit outputs.
#     Args:
#         outputs: (batch_size, num_time_bins) - predicted probabilities for each time bin
#     Returns: 
#         PMF: (torch.Tensor) - Probabality of event occuring at time indexed by dim=1
#     """
#     hazards = torch.nn.functional.sigmoid(outputs)
#     sf = torch.cumprod(1 - hazards, dim=1)
#     pmf = torch.zeros_like(hazards)
#     pmf[:, 0] = hazards[:, 0]
#     pmf[:, 1:] = hazards[:, 1:] * sf[:, :-1]
#     pmf[:, -1] += 1.0 - pmf.sum(dim=1)  # Adjust last bin to ensure sums to 1
#     return pmf

def compute_loss(
    outputs: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    x: torch.Tensor = None,
    y: torch.Tensor = None,
    pmf: torch.Tensor = None,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    

    t = t.long()
    # frame_mask = mask.any(dim=-1)
    
    # t_last = (torch.arange(frame_mask.size(1), device=frame_mask.device)
    #           .unsqueeze(0)
    #           .expand(frame_mask.size(0), -1) * frame_mask).max(dim=1).values

    # --- Check intermediate tensors for NaNs ---
    # nan_mask = torch.isnan(features)  # True wherever NaN
    # if nan_mask.any():
    #     print("NaNs found in features!")
    #     nan_batches, nan_rows, nan_cols = nan_mask.nonzero(as_tuple=True)
    #     print("Batches with NaNs:", nan_batches.unique())
    #     print("Frames with NaNs:", nan_rows.unique())
    #     print("Features with NaNs:", nan_cols.unique())
    # cif = estimated_cif(outputs, t, t_last)
    
    # cif = estimated_cif(outputs)
    # cif = outputs
    t = t.long()
    
    # if cif is None:
    #     # hazards = torch.nn.functional.sigmoid(outputs)
    #     # # hazards = 1 - torch.exp((-torch.exp(outputs)))
    #     # sf = torch.cumprod(1 - hazards, dim=1)
    #     # cif = 1 - sf
    #     cif = estimated_cif(outputs)
    
    if pmf is None:
        pmf = estimated_pmf(outputs)
    # assert (pmf >= 0.0).all(), f"PMF has negative values: {pmf}"
    pmf = torch.clamp(pmf, min=0.0, max=1.0)
    # pmf = torch.nn.functional.softmax(outputs, dim=1)
    negative_log_likelihood, censored_loss, uncesnored_loss = losses.negative_log_likelihood(pmf, t, e)
    # negative_log_likelihood, censored_loss, uncesnored_loss = losses.soft_NLL(outputs, cif, t, e)

    ranking_loss = losses.ranking_loss(pmf, t, e)   

    if x is None or y is None:
        prediction_loss = torch.tensor(0.0, device=t.device)
    else:
        # mask = None
        # mask = torch.isnan(x)
        # features = x
        
        
        if mask is None:
            mask = torch.ones_like(x, device=x.device)
        # else:
        #     features = x[:, :, :x.size(-1)//2]
        #     mask = x[:, :, x.size(-1)//2:] 
            
        prediction_loss = losses.prediction_loss(y, x, mask)

    # loss = negative_log_likelihood + ranking_loss + prediction_loss
    # loss = negative_log_likelihood + prediction_loss + ranking_loss
    loss = negative_log_likelihood
    # loss = negative_log_likelihood
    if torch.isnan(loss) or torch.isinf(loss):
        print("NaN or inf loss encountered:")
        print(f"  Negative Log Likelihood: {negative_log_likelihood.item()}")
        print(f"  Ranking Loss: {ranking_loss.item()}")
        print(f"  NLL censored: {censored_loss.item()}")
        print(f"  NLL uncensored: {uncesnored_loss.item()}")
        print(f"  Prediction Loss: {prediction_loss.item()}")
        print(f"  Inputs: {x}")
        print(f"  Outputs: {outputs}")
        print(f"  PMF: {pmf}")
        print(f"  t: {t}")
        print(f"  e: {e}")
        raise ValueError("NaN or inf loss encountered in compute_loss")
    
    return loss, negative_log_likelihood, ranking_loss, prediction_loss, censored_loss, uncesnored_loss
