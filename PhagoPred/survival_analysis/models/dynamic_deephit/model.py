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
                 input_size: int, 
                 output_size: int,
                 lstm_hidden_size: int = 64,
                 lstm_dropout: float = 0.0,
                 predictor_layers: list[int] = [32],
                 attention_layers: list[int] = [64, 64],
                 fc_layers: list[int] = [64, 64],
                 **kwargs
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, 
                                  hidden_size=lstm_hidden_size, 
                                  dropout=lstm_dropout,
                                  batch_first=True
                                  )

        self.predictor = build_fc_layers(input_size=lstm_hidden_size, output_size=input_size, layer_sizes=predictor_layers)
        self.attention = build_fc_layers(input_size=lstm_hidden_size, output_size=1, layer_sizes=attention_layers)

        self.fc = build_fc_layers(input_size=lstm_hidden_size, output_size=output_size, layer_sizes=fc_layers)
        
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model loaded. Trainable parameters: {num_trainable_params}")



    def forward(self, x, return_attention: bool = False):
        """
        Args:
            x: [batch_size, sequence length, num_features]
        """
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out (batch_size, seq_len, lstm_hidden_size)
        y = self.predictor(lstm_out)  # y (batch_size, seq_length, num_features//2)
        # lstm_hidden = lstm_hidden.permute(1, 0, 2)  # (batch_size, seq_length, lstm_hidden_size)
        attn_weights = self.attention(lstm_out).squeeze(-1) # (batch, seq_len)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)  # (batch, seq_len)
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * lstm_out, dim=1)  # (batch, lstm_hidden_size)
        output = self.fc(context_vector)  # (batch, output_size)
        output = torch.nn.functional.softmax(output, dim=-1)  # (batch, output_size)
        if return_attention:
            return output, y, attn_weights
        else:
            return output, y


# def estimated_cif(outputs: torch.Tensor,
#         t: torch.Tensor,
#         t_last: torch.Tensor,
#         ) -> torch.Tensor:
#     """Cumulative incidence function from Dynamic DeepHit outputs.
#     Args:
#         outputs: (batch_size, num_time_bins) - predicted probabilities for each time bin
#         t: (batch_size,) - true event/censoring times (as indices of time bins)
#         t_last: (batch_size,) - last observed times (as indices of time bins)
#     """
#     batch_size, num_bins = outputs.size()
#     time_index = torch.arange(num_bins, device=outputs.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, num_time_bins)
    
#     numerator_mask = (time_index > t_last.unsqueeze(1)) & (time_index <= t.unsqueeze(1))  # (batch_size, num_time_bins)
#     numerator = torch.cumsum(outputs * numerator_mask.float(), dim=1)  # (batch_size, time_bins)

#     denominator_mask = time_index < t_last.unsqueeze(1)  # (batch_size, num_time_bins)
#     denominator = 1 - torch.sum(outputs * denominator_mask.float(), dim=1)  # (batch_size,)

#     cif = numerator / (denominator.unsqueeze(1) + 1e-8)  # (batch_size, num_time_bins)
#     return cif

def estimated_cif(
    outputs: torch.Tensor,
    # t: torch.Tensor
    ) -> torch.Tensor:
    """Cumulative incidence function from Dynamic DeepHit outputs.
    Args:
        outputs: (batch_size, num_time_bins) - predicted probabilities for each time bin
        t: (batch_size,) - true event/censoring times (as indices of time bins)
        
    Returns: 
        CIF: (torch.Tensor) - Probabality of event occuring on or before time indexed by dim=1
    """
    
    cif = torch.cumsum(outputs, dim=1)
    return cif
        
    


def compute_loss(
    outputs: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    x: torch.Tensor = None,
    y: torch.Tensor = None,
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
    cif = estimated_cif(outputs)

    negative_log_likelihood, censored_loss, uncesnored_loss = losses.negative_log_likelihood(outputs, cif, t, e)
    # negative_log_likelihood, censored_loss, uncesnored_loss = losses.soft_NLL(outputs, cif, t, e)

    ranking_loss = losses.ranking_loss(cif, t, e)   

    if x is None or y is None:
        prediction_loss = torch.tensor(0.0, device=outputs.device)
    else:
        mask = None
        
        if mask is None:
            features = x
            mask = torch.ones_like(features, device=features.device)
        else:
            features = x[:, :, :x.size(-1)//2]
            mask = x[:, :, x.size(-1)//2:] 
            
        prediction_loss = losses.prediction_loss(y, features, mask)

    # loss = negative_log_likelihood + ranking_loss + prediction_loss
    loss = negative_log_likelihood + prediction_loss + ranking_loss
    if torch.isnan(loss) or torch.isinf(loss):
        print("NaN or inf loss encountered in final loss!")
        print("NaN or inf loss encountered:")
        print(f"  Negative Log Likelihood: {negative_log_likelihood.item()}")
        print(f"  Ranking Loss: {ranking_loss.item()}")
        print(f"  NLL censored: {censored_loss.item()}")
        print(f"  NLL uncensored: {uncesnored_loss.item()}")
        print(f"  Prediction Loss: {prediction_loss.item()}")
        print(f"  Outputs: {outputs}")
        print(f"  CIF: {cif}")
        print(f"  t: {t}")
        print(f"  e: {e}")
        raise ValueError("NaN or inf loss encountered in compute_loss")
    
    return loss, negative_log_likelihood, ranking_loss, prediction_loss, censored_loss, uncesnored_loss
