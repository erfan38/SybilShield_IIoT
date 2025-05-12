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


# ========== Dynamic CNN Model Builder ==========
def build_model(conv_params, activation_fn, use_batchnorm, dropout_rate):
    layers = []
    for in_c, out_c, k, p in conv_params:
        layers.append(nn.Conv1d(in_c, out_c, kernel_size=k, padding=p))
        layers.append(activation_fn())
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_c))
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


# ========== CNN Trainer Class ==========
class CNNTrainer:
    def __init__(self, path, model, optimizer_name, lr, loss_fn, epochs):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.epochs = epochs
        self.optimizer = self._get_optimizer(optimizer_name)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.4, patience=5)
        self.X_train, self.X_val, self.y_train, self.y_val = self.load_data(path)

    def _get_optimizer(self, name):
        if name == "Adam":
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif name == "AdamW":
            return optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        elif name == "SGD":
            return optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

    def load_data(self, path):
        df = pd.read_csv(path)
        X = df[[
            "sent_total", "received_total", "protocol_diversity",
            "message_burstiness", "mqtt_ratio", "discovery_ratio",
            "protocol_entropy", "unique_peers", "dominant_protocol_ratio"
        ]].values
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


# ========== Grid Search Config ==========
grid = {
    "loss_fn": [nn.CrossEntropyLoss(), nn.BCELoss()],
    "num_conv_layers": [2, 3, 4],
    "use_batchnorm": [True, False],
    "activation": [nn.ReLU, nn.LeakyReLU, nn.GELU],
    "dropout_rate": [0.1, 0.3, 0.5],
    "conv_params": [
        # (in_channels, out_channels, kernel_size, padding)
        [(1, 16, 3, 1), (16, 32, 3, 1)],
        [(1, 32, 5, 2), (32, 64, 5, 2)],
        [(1, 16, 3, 1), (16, 32, 3, 1), (32, 64, 3, 1)],
    ],
    "epochs": [50, 100],
    "optimizer": ["Adam", "AdamW"],
    "lr": [0.001, 0.01, 0.015]
}

keys, values = zip(*grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
results = []

# ========== Run Grid Search ==========
for i, config in enumerate(combinations):
    print(f"\n🔧 Config {i+1}/{len(combinations)}:")
    conv_layers = config["conv_params"][:config["num_conv_layers"]]
    model = build_model(conv_layers, config["activation"], config["use_batchnorm"], config["dropout_rate"])
    trainer = CNNTrainer(
        path="balanced_sybil_dataset.csv",
        model=model,
        optimizer_name=config["optimizer"],
        lr=config["lr"],
        loss_fn=config["loss_fn"],
        epochs=config["epochs"]
    )
    trainer.train()
    acc = trainer.evaluate()
    print(f"✅ Validation Accuracy: {acc:.4f}")
    print(f"\nConfig {i+1}:")
    for k, v in config.items():
        if isinstance(v, nn.Module):
            print(f"{k}: {type(v).__name__}")
        elif callable(v):
            print(f"{k}: {v.__name__}")
        else:
            print(f"{k}: {v}")
    print(f"Validation Accuracy: {acc:.4f}")
    print(80 * "-")
    results.append((config, acc))
    del model, trainer
    gc.collect()

# ========== Print All Configs ==========
# print("\n📋 All Results:")
# for i, (config, acc) in enumerate(results):
#     print(f"\nConfig {i+1}:")
#     for k, v in config.items():
#         if isinstance(v, nn.Module):
#             print(f"{k}: {type(v).__name__}")
#         elif callable(v):
#             print(f"{k}: {v.__name__}")
#         else:
#             print(f"{k}: {v}")
#     print(f"Validation Accuracy: {acc:.4f}")

# ========== Best Config ==========
best_config, best_acc = max(results, key=lambda x: x[1])
print("\n🏆 Best Config:")
for k, v in best_config.items():
    if isinstance(v, nn.Module):
        print(f"{k}: {type(v).__name__}")
    elif callable(v):
        print(f"{k}: {v.__name__}")
    else:
        print(f"{k}: {v}")
print(f"🏁 Best Validation Accuracy: {best_acc:.4f}")
