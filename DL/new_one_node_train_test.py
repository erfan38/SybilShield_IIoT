import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ------------------ Model ----------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch_size, 1, features]
        x = self.conv(x)
        return self.fc(x)

# ------------------ Trainer --------------------------------------------------
class CNNTrainer:
    def __init__(self, csv_path, epochs=200, lr=1.5e-2, batch_size=64,
                 test_size=0.15, val_size=0.15, seed=42):
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.model  = ConvNet().to(self.device)
        self.loss_fn  = nn.CrossEntropyLoss()
        self.opt   = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=4e-4)
        self.sched = ReduceLROnPlateau(self.opt, mode='min', factor=0.4,
                                       patience=6, verbose=True)
        # ---- data
        (self.train_loader,
         self.val_loader,
         self.test_loader) = self._prepare_data(csv_path, batch_size,
                                                test_size, val_size, seed)
        self.epochs = epochs

    # ---------------- data pipeline ----------------
    @staticmethod
    def _prepare_data(path, batch, test_frac, val_frac, seed):
        df = pd.read_csv(path)
        X = df[[  # <-- Updated 
    "sent_total", "received_total",
    "protocol_diversity",
    "message_burstiness", "mqtt_ratio", "discovery_ratio",
    "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
    "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity"
        ]].values
        y = df["is_sybil"].astype(int).values

        # Split
        X_train_val, X_test, y_train_val, y_test = \
            train_test_split(X, y, test_size=test_frac, stratify=y, random_state=seed)

        relative_val = val_frac / (1 - test_frac)
        X_train, X_val, y_train, y_val = \
            train_test_split(X_train_val, y_train_val, test_size=relative_val,
                             stratify=y_train_val, random_state=seed)

        # Scale
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

        def to_loader(x, y):
            ds = TensorDataset(torch.tensor(x, dtype=torch.float32),
                               torch.tensor(y, dtype=torch.long))
            return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

        return to_loader(X_train, y_train), to_loader(X_val, y_val), to_loader(X_test, y_test)

    # ---------------- training loop ----------------
    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss, correct, count = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss   = self.loss_fn(logits, y)

            if train:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            count += y.size(0)

        return total_loss / count, correct / count

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss,   val_acc   = self._run_epoch(self.val_loader,   train=False)
            self.sched.step(val_loss)

            print(f"Epoch {epoch:03d}/{self.epochs} | "
                  f"Train L {train_loss:.4f}  A {train_acc:.4f} | "
                  f"Val   L {val_loss:.4f}  A {val_acc:.4f} | "
                  f"LR {self.opt.param_groups[0]['lr']:.6f}")

        return self

    def test(self):
        loss, acc = self._run_epoch(self.test_loader, train=False)
        print(f"\nTest accuracy: {acc:.4f}")
        return acc

# ------------------ Run ------------------------------------------------------
if __name__ == "__main__":
    trainer = CNNTrainer(csv_path="new3_dataset.csv", epochs=150, lr=0.02)
    trainer.fit()
    trainer.test()
