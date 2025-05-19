# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.data import TensorDataset, DataLoader

# # Define your ConvNet model
# class ConvNet(nn.Module):
#     def __init__(self, input_dim=13, num_classes=3):
#         super(ConvNet, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):
#         return self.fc(x)

# class CNNTrainer:
#     def __init__(self, csv_path, epochs=200, lr=1.5e-2, batch_size=64,
#                  test_size=0.15, val_size=0.15, seed=42):
#         torch.manual_seed(seed)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = ConvNet().to(self.device)
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=4e-4)
#         self.sched = ReduceLROnPlateau(self.opt, mode='min', factor=0.4,
#                                        patience=6, verbose=True)
#         # Prepare data
#         (self.train_loader,
#          self.val_loader,
#          self.test_loader) = self._prepare_data(csv_path, batch_size,
#                                                 test_size, val_size, seed)
#         self.epochs = epochs

#     @staticmethod
#     def _prepare_data(path, batch, test_frac, val_frac, seed):
#             # Load dataset
#             df = pd.read_csv(path)

#             # Select features
#             X = df[
#                 [
#                     "sent_total", "received_total",
#                     "protocol_diversity", "message_burstiness", "mqtt_ratio", "discovery_ratio",
#                     "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
#                     "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity"
#                 ]
#             ].values

#             # Extract raw labels
#             y_raw = df["severity"].values.reshape(-1, 1)  # shape (n_samples, 1)

#             # Encode labels using OneHotEncoder
#             encoder = OneHotEncoder(
#                 categories=[["honest", "low", "high"]],
#                 sparse_output=False,
#                 handle_unknown='error'
#             )
#             y = encoder.fit_transform(y_raw)

#             # Split data into train+val and test sets
#             X_train_val, X_test, y_train_val, y_test, y_raw_train_val, y_raw_test = train_test_split(
#                 X, y, y_raw,
#                 test_size=test_frac,
#                 stratify=y_raw,
#                 random_state=seed
#             )

#             # Calculate validation fraction relative to the train+val set
#             relative_val = val_frac / (1 - test_frac)

#             # Split train+val into train and validation sets
#             X_train, X_val, y_train, y_val, y_raw_train, y_raw_val = train_test_split(
#                 X_train_val, y_train_val, y_raw_train_val,
#                 test_size=relative_val,
#                 stratify=y_raw_train_val,
#                 random_state=seed
#             )

#             # Scale features using StandardScaler
#             scaler = StandardScaler().fit(X_train)
#             X_train = scaler.transform(X_train)
#             X_val   = scaler.transform(X_val)
#             X_test  = scaler.transform(X_test)



#             def to_loader(x, y):
#                 ds = TensorDataset(torch.tensor(x, dtype=torch.float32),
#                                 torch.tensor(y, dtype=torch.long))
#                 return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

#             return to_loader(X_train, y_train), to_loader(X_val, y_val), to_loader(X_test, y_test)

#     def _run_epoch(self, loader, train=True):
#         self.model.train() if train else self.model.eval()
#         total_loss, correct, count = 0.0, 0, 0
#         for x, y in loader:
#             x, y = x.to(self.device), y.to(self.device).float()
#             logits = self.model(x)
#             loss = self.loss_fn(logits, y)

#             if train:
#                 self.opt.zero_grad()
#                 loss.backward()
#                 self.opt.step()

#             total_loss += loss.item() * y.size(0)
#             preds = logits.argmax(1)
#             y_labels = y.argmax(dim=1)
#             correct += (preds == y_labels).sum().item()

#             count += y.size(0)

#         return total_loss / count, correct / count

#     def fit(self):
#         for epoch in range(1, self.epochs + 1):
#             train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
#             val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
#             self.sched.step(val_loss)

#             print(f"Epoch {epoch:03d}/{self.epochs} | "
#                   f"Train L {train_loss:.4f}  A {train_acc:.4f} | "
#                   f"Val   L {val_loss:.4f}  A {val_acc:.4f} | "
#                   f"LR {self.opt.param_groups[0]['lr']:.6f}")

#         return self

#     def test(self):
#         loss, acc = self._run_epoch(self.test_loader, train=False)
#         print(f"\nTest accuracy: {acc:.4f}")
#         return acc

# # ------------------ Run ------------------------------------------------------
# if __name__ == "__main__":
#     trainer = CNNTrainer(csv_path="dataset_25-05-17-18-17-48.csv", epochs=250, lr=0.02)
#     trainer.fit()
#     trainer.test()



"""
cnn_trainer.py
Usage:
    python cnn_trainer.py --csv dataset_25-05-17-18-17-48.csv --epochs 250 --lr 0.02
"""
import os, time, argparse, logging, csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from nn_models import get_model
from data_loader import DataLoaderWrapper


# ----------------------  Model  ---------------------- #
# class ConvNet(nn.Module):
#     def __init__(self, input_dim=13, num_classes=3):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):
#         return self.fc(x)





# ----------------------  Trainer  -------------------- #
class CNNTrainer:
    def __init__(self, csv_path, epochs=200, lr=1.5e-2, batch_size=64,
                 test_size=0.15, val_size=0.15, seed=42):
        ts = time.strftime("%y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join("runs", f"run_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.run_dir, "training.log"))
            ],
        )
        self.log = logging.getLogger(__name__)
        self.log.info(f"Run directory: {self.run_dir}")

        torch.manual_seed(seed)
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model    = get_model()
        self.loss_fn  = nn.CrossEntropyLoss()
        self.opt      = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=4e-4)
        self.sched    = ReduceLROnPlateau(self.opt, mode='min', factor=0.4, patience=6, verbose=True)

        (self.train_loader,
         self.val_loader,
         self.test_loader) = self._prepare_data(csv_path, batch_size,
                                                test_size, val_size)
        self.epochs = epochs

        self.lr_hist = []
        self.train_losses, self.val_losses = [], []
        self.train_accuracies, self.val_accuracies = [], []

    @staticmethod
    def _prepare_data(path, batch, test_frac, val_frac):

        loader = DataLoaderWrapper(path, test_frac, val_frac)
    #     self.X_train   = loader.client_data["train"][self.client_id]["x"]
    #     self.y_train = loader.client_data["train"][self.client_id]["y"]
    #     self.x_train_node_id = loader.client_data["train"][self.client_id]["node_id"]

    #     self.X_test   = loader.client_data["test"][self.client_id]["x"]
    #     self.y_test = loader.client_data["test"][self.client_id]["y"]
    #     self.x_test_node_id = loader.client_data["test"][self.client_id]["node_id"]

    #     self.X_val   = loader.client_data["val"][self.client_id]["x"]
    #     self.y_val = loader.client_data["val"][self.client_id]["y"]
    #     self.x_val_node_id = loader.client_data["val"][self.client_id]["node_id"]
    #     df = pd.read_csv(path)

    #     X = df[[
    #        "sent_total", "received_total", "protocol_diversity",
    # "message_burstiness", "mqtt_ratio", "discovery_ratio",
    # "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
    # "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity",
    # "latency_spike_ratio", "energy_variance", "peer_overlap_ratio"
    #     ]].values

    #     label_map = {"honest": 0, "low": 1, "high": 2}
    #     y = df["severity"].map(label_map).astype(int).values

    #     X_train_val, X_test, y_train_val, y_test = train_test_split(
    #         X, y, test_size=test_frac, stratify=y, random_state=seed
    #     )
    #     rel_val = val_frac / (1 - test_frac)
    #     X_train, X_val, y_train, y_val = train_test_split(
    #         X_train_val, y_train_val, test_size=rel_val,
    #         stratify=y_train_val, random_state=seed
    #     )

        scaler = StandardScaler().fit(loader.X_train)
        X_train, X_val, X_test = map(scaler.transform, [loader.X_train, loader.X_val, loader.X_test])

        def to_loader(x, y):
            ds = TensorDataset(torch.tensor(x, dtype=torch.float32),
                               torch.tensor(y, dtype=torch.long))
            return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

        return to_loader(X_train, loader.y_train), to_loader(X_val, loader.y_val), to_loader(X_test, loader.y_test)

    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss, correct, count = 0.0, 0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)

            if train:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            count   += y.size(0)

        return total_loss / count, correct / count

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc     = self._run_epoch(self.val_loader, train=False)
            self.sched.step(val_loss)
            lr_now = self.opt.param_groups[0]['lr']

            self.lr_hist.append(lr_now)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            self.log.info(f"Epoch {epoch:03d}/{self.epochs} | "
                          f"Train L {train_loss:.4f} A {train_acc:.4f} | "
                          f"Val   L {val_loss:.4f} A {val_acc:.4f} | "
                          f"LR {lr_now:.6f}")

        self._save_lr_artifacts()
        self._save_learning_curves()
        return self

    def test(self):
        loss, acc = self._run_epoch(self.test_loader, train=False)
        self.log.info(f"\nTest accuracy: {acc:.4f}")
        return acc

    def _save_lr_artifacts(self):
        csv_path = os.path.join(self.run_dir, "lr_history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "learning_rate"])
            writer.writerows([(i + 1, lr) for i, lr in enumerate(self.lr_hist)])

        plt.figure()
        plt.plot(range(1, len(self.lr_hist) + 1), self.lr_hist, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title("LR Schedule")
        plt.tight_layout()
        fig_path = os.path.join(self.run_dir, "lr_curve.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        self.log.info(f"Saved LR diagram -> {fig_path}")

    def _save_learning_curves(self):
        epochs = range(1, self.epochs + 1)

        # Loss curve
        plt.figure()
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "loss_curve.png"), dpi=300)
        plt.close()

        # Accuracy curve
        plt.figure()
        plt.plot(epochs, self.train_accuracies, label="Train Accuracy")
        plt.plot(epochs, self.val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "accuracy_curve.png"), dpi=300)
        plt.close()

        self.log.info("Saved loss and accuracy plots.")

# ----------------------  main  ---------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset_25-05-17-21-29-19.csv")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()

    trainer = CNNTrainer(csv_path=args.csv, epochs=args.epochs, lr=args.lr)
    trainer.fit()
    trainer.test()
