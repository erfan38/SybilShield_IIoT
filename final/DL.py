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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_DIR = "models/DL"
# ----------------------  Trainer  -------------------- #
class CNNTrainer:
    def __init__(self, csv_path, epochs=50, lr=1.5e-2, batch_size=64,
                 test_size=0.15, val_size=0.15, seed=42):
        ts = time.strftime("%y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join("logs/DL", f"run_{ts}")
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

    # def test(self):
    #     loss, acc = self._run_epoch(self.test_loader, train=False)
    #     self.log.info(f"\nTest accuracy: {acc:.4f}")
    #     return acc
    def test(self):
        self.model.eval()
        start_time = time.time()

        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                preds = logits.argmax(1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        end_time = time.time()
        elapsed_time = round(end_time - start_time, 4)

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        accuracy  = round(accuracy_score(y_true, y_pred), 4)
        precision = round(precision_score(y_true, y_pred, average='weighted'), 4)
        recall    = round(recall_score(y_true, y_pred, average='weighted'), 4)
        f1        = round(f1_score(y_true, y_pred, average='weighted'), 4)

        # Save model and calculate storage
        model_path = os.path.join(MODEL_DIR, "final_model.pt")
        torch.save(self.model.state_dict(), model_path)
        overhead_bytes = os.path.getsize(model_path)

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "time_seconds": elapsed_time,
            "overhead_bytes": overhead_bytes
        }

        df = pd.DataFrame(results.items(), columns=["Metric", "Value"])
        output_csv = os.path.join(self.run_dir, "test_metrics.csv")
        df.to_csv(output_csv, index=False)

        self.log.info("\nTest Metrics:\n" + df.to_markdown(index=False))
        self.log.info(f"Metrics saved to {output_csv}")
        return results

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
    parser.add_argument("--csv", default="data/dataset_25-05-19-13-30-43.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005) # was 0.02
    args = parser.parse_args()

    trainer = CNNTrainer(csv_path=args.csv, epochs=args.epochs, lr=args.lr)
    trainer.fit()
    trainer.test()
