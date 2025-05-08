
import torch
import torch.nn as nn
import pandas as pd
import os

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_autoencoder(df, input_size):
    X = torch.tensor(df.values, dtype=torch.float32)
    model = AutoEncoder(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for epoch in range(30):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, X)
        loss.backward()
        optimizer.step()
    return model

def detect_anomalies(model, df, threshold=0.1):
    X = torch.tensor(df.values, dtype=torch.float32)
    recon = model(X)
    mse = ((X - recon)**2).mean(dim=1)
    return (mse > threshold).nonzero(as_tuple=True)[0]

data_dir = "data_per_node"
for fname in sorted(os.listdir(data_dir)):
    if fname.endswith(".csv"):
        print(f"\nProcessing {fname}")
        df = pd.read_csv(os.path.join(data_dir, fname))
        label_col = "label" if "label" in df.columns else "is_sybil"
        X = df.drop(columns=[label_col, "node_id"])
        model = train_autoencoder(X, input_size=X.shape[1])
        anomalies = detect_anomalies(model, X)
        print(f"Anomalies in {fname}: {list(anomalies.numpy())}")
