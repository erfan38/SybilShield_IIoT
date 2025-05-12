import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# 1. Define a simple CNN for 1D feature input
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
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



# 2. Trainer class
class CNNTrainer:
    def __init__(self, path, lr=0.001, epochs=200):
        self.path = path
        self.lr = lr
        self.epochs = epochs
        self.model = ConvNet()
        self.loss_fn = nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=4e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.4, patience=6, verbose=True)
        self.train_losses = []
        self.X_tensor, self.y_tensor = self.load_data()

    def load_data(self):
        df = pd.read_csv(self.path)
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
        return X_tensor, y_tensor

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.X_tensor)
            loss = self.loss_fn(outputs, self.y_tensor)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)  # Adjust LR if no improvement
            self.train_losses.append(loss.item())
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_tensor)
            preds = torch.argmax(outputs, dim=1)
            acc = accuracy_score(self.y_tensor.numpy(), preds.numpy())
            print(f"Evaluation Accuracy: {acc:.4f}")
        return acc

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.show()


# 3. Usage
if __name__ == "__main__":
    trainer = CNNTrainer(path="balanced_sybil_dataset.csv", lr=0.015, epochs=200)
    trainer.train()
    trainer.evaluate()
    trainer.plot_losses()
