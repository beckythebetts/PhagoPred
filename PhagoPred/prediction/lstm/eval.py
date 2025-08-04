# eval.py

import torch
from model import LSTMAttention
import yaml

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def evaluate():
    config = load_config()

    model = LSTMAttention(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout'],
        attn_hidden_size=config['model']['attn_hidden_size'],
        bidirectional=config['model']['bidirectional']
    )
    model.load_state_dict(torch.load(config['paths']['model_save_path']))
    model.eval()

    # Dummy eval input
    x = torch.randn(50, 1, config['model']['input_size'])  # (seq_len, batch, features)
    with torch.no_grad():
        logits, attn_weights = model(x)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).int()
        print(f"Predicted: {pred.item()}, Probability: {prob.item():.3f}")

if __name__ == "__main__":
    evaluate()
