import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import itertools
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# === FEATURES ===
FEATURE_COLUMNS = [
    "sent_total", "received_total", "protocol_diversity",
    "message_burstiness", "mqtt_ratio", "discovery_ratio",
    "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
    "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity",
    "latency_spike_ratio", "energy_variance", "peer_overlap_ratio"
]

# === CNN Model Builder ===
def build_model(conv_params, activation_fn, dropout_rate):
    layers = []
    for in_c, out_c, k, p in conv_params:
        layers.append(nn.Conv1d(in_c, out_c, kernel_size=k, padding=p))
        layers.append(activation_fn())
    layers.append(nn.AdaptiveAvgPool1d(1))

    conv = nn.Sequential(*layers)
    final_out_channels = conv_params[-1][1]

    fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(final_out_channels, 16), activation_fn(),
        nn.Dropout(dropout_rate),
        nn.Linear(16, 8), activation_fn(),
        nn.Linear(8, 2)
    )

    class DynamicCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = conv
            self.fc = fc

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv(x)
            return self.fc(x)

    return DynamicCNN()

# === Trainer Class ===
class CNNTrainer:
    def __init__(self, path, model, lr, loss_fn, epochs=200):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.4, patience=10)
        self.X_train, self.X_val, self.y_train, self.y_val = self.load_data(path)

    def load_data(self, path):
        df = pd.read_csv(path)
        X = df[FEATURE_COLUMNS].values
        y = df["is_sybil"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    def train(self):
        self.model.train()
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = self.loss_fn(output, self.y_train)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.X_val)
            preds = torch.argmax(output, dim=1)
            acc = accuracy_score(self.y_val.numpy(), preds.numpy())
        return acc

# === Grid Search Configuration ===
grid = {
    "loss_fn": [nn.CrossEntropyLoss()],
    "activation": [nn.ReLU, nn.GELU],
    "dropout_rate": [0.1, 0.3],
    "num_conv_layers": [2, 3],
    "kernel_size": [3, 5],
    "lr": [0.001, 0.005]
}

# Build conv_params templates
def build_conv_blocks(kernel_size, num_layers):
    channels = [(1, 32), (32, 64), (64, 128)]
    return [(in_c, out_c, kernel_size, kernel_size // 2) for in_c, out_c in channels[:num_layers]]

# Run Search
keys, values = zip(*grid.items())
combinations = list(itertools.product(*values))
results = []

for i, combo in enumerate(combinations):
    config = dict(zip(keys, combo))
    conv_blocks = build_conv_blocks(config["kernel_size"], config["num_conv_layers"])
    model = build_model(conv_blocks, config["activation"], config["dropout_rate"])
    trainer = CNNTrainer(
        path="new3_dataset.csv",
        model=model,
        lr=config["lr"],
        loss_fn=config["loss_fn"],
        epochs=200
    )
    trainer.train()
    acc = trainer.evaluate()
    
    if acc >= 0.75:
        print(f"\n✅ Config {i+1}/{len(combinations)} Passed - Accuracy: {acc:.4f}")
        print(f"→ lr={config['lr']}, kernel={config['kernel_size']}, layers={config['num_conv_layers']}, "
              f"act={config['activation'].__name__}, drop={config['dropout_rate']}")
        print("-" * 70)
        results.append((config, acc))

    del model, trainer
    gc.collect()

# Best Result
if results:
    best_config, best_acc = max(results, key=lambda x: x[1])
    print("\n🏆 Best Config:")
    for k, v in best_config.items():
        print(f"{k}: {v.__name__ if callable(v) else type(v).__name__}")
    print(f" Best Accuracy: {best_acc:.4f}")
else:
    print("❌ No configuration passed the 75% accuracy threshold.")
