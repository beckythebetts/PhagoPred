# train.py

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import yaml
import os
from model import LSTMAttention

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def train():
    config = load_config()

    # Dummy data for illustration
    x_train = torch.randn(100, 50, config['model']['input_size'])  # (samples, seq_len, features)
    y_train = torch.randint(0, 2, (100, 1)).float()

    x_train = x_train.permute(1, 0, 2)  # (seq_len, batch, features)
    dataset = TensorDataset(x_train.permute(1, 0, 2), y_train)
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'])

    model = LSTMAttention(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout'],
        attn_hidden_size=config['model']['attn_hidden_size'],
        bidirectional=config['model']['bidirectional']
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    model.train()
    for epoch in range(config['training']['num_epochs']):
        for x_batch, y_batch in loader:
            x_batch = x_batch.permute(1, 0, 2)  # (seq_len, batch, features)
            logits, _ = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    os.makedirs(os.path.dirname(config['paths']['model_save_path']), exist_ok=True)
    torch.save(model.state_dict(), config['paths']['model_save_path'])

if __name__ == "__main__":
    train()
