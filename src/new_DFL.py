import os
import time
import json
import torch
import hashlib
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
from nn_models import get_model
from data_loader import DataLoaderWrapper
import matplotlib.pyplot as plt

# ---- Configuration ----
ROUNDS = 5
EPOCHS = 100
PEERS = [f"client_{i}" for i in range(5)]
FEATURE_COLUMNS = [
    "sent_total", "received_total", "protocol_diversity",
    "message_burstiness", "mqtt_ratio", "discovery_ratio",
    "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
    "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity",
    "latency_spike_ratio", "energy_variance", "peer_overlap_ratio"
]

DATA_DIR = "nodes_final_data"
MODEL_DIR = "models/DFL"
LOG_DIR = "logs/DFL"
REPORT_DIR = "reports/DFL"
lock = multiprocessing.Lock()
CSV_PATH = "data/dataset_25-05-19-13-30-43.csv"

# ---- Ensure Directories ----
for d in [MODEL_DIR, LOG_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# # ---- Define model ----
# def get_model(input_dim=16, num_classes=3):
#     return nn.Sequential(
#         nn.Linear(input_dim, 64),
#         nn.ReLU(),
#         nn.Linear(64, num_classes)
#     )

# ---- Utilities ----
def get_model_hash(model):
    flat_params = torch.cat([p.flatten() for p in model.parameters()]).detach().numpy()
    byte_data = bytes(np.array(flat_params, dtype=np.float32))
    return hashlib.sha256(byte_data).hexdigest()

# ---- Node Class ----
class DecentralizedNode:
    def __init__(self, node_id):
        self.client_id = node_id
        self.model = get_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005) #lr=1e-4
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=EPOCHS)
        self._load_and_split_data()
        self.train_loss_hist = []
        self.val_loss_hist = []
        self.train_acc_hist = []
        self.val_acc_hist = []


    def _load_and_split_data(self):

        loader = DataLoaderWrapper(CSV_PATH, number_of_clients=5)
        self.X_train   = loader.client_data["train"][self.client_id]["x"]
        self.y_train = loader.client_data["train"][self.client_id]["y"]
        self.x_train_node_id = loader.client_data["train"][self.client_id]["node_id"]

        self.X_test   = loader.client_data["test"][self.client_id]["x"]
        self.y_test = loader.client_data["test"][self.client_id]["y"]
        self.x_test_node_id = loader.client_data["test"][self.client_id]["node_id"]

        self.X_val   = loader.client_data["val"][self.client_id]["x"]
        self.y_val = loader.client_data["val"][self.client_id]["y"]
        self.x_val_node_id = loader.client_data["val"][self.client_id]["node_id"]

        # weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        self.loss_fn = nn.CrossEntropyLoss()#weight=torch.tensor(weights, dtype=torch.float32))


    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(self.X_train)
        loss = self.loss_fn(outputs, self.y_train)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        acc = (outputs.argmax(1) == self.y_train).float().mean().item()
        return loss.item(), acc

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.loss_fn(outputs, y)
            acc = (outputs.argmax(1) == y).float().mean().item()
            return loss.item(), acc

    def save_model(self):
        with lock:
            torch.save(self.model.state_dict(), f"{MODEL_DIR}/{self.client_id}.pt")

    def load_peer_models(self):
        peer_weights = []
        with lock:
            for peer in PEERS:
                if peer == self.client_id:
                    continue
                path = f"{MODEL_DIR}/{peer}.pt"
                if os.path.exists(path):
                    peer_weights.append(torch.load(path))
        return peer_weights

    def average_with_peers(self, peer_weights):
        if not peer_weights:
            return
        my_state = self.model.state_dict()
        new_state = {k: sum([peer[k] for peer in peer_weights] + [my_state[k]]) / (len(peer_weights) + 1)
                     for k in my_state}
        self.model.load_state_dict(new_state)

    def log_epoch(self, round, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(f"{LOG_DIR}/{self.client_id}_log.txt", "a") as f:
            f.write(f"Round {round+1}, Epoch {epoch+1}, "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

    def generate_report(self):
        report = {
            "node_id": self.client_id,
            "model_hash": get_model_hash(self.model),
            "timestamp": int(time.time()),
            "suspicious_nodes": []  # renamed from "reports"
        }

        # Empirical thresholds for anomaly
        baseline = {"peer_overlap": 0.3, "discovery_ratio": 0.2, "proto_entropy": 0.75}

        all_nodes = [f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        for peer_id in all_nodes:
            if peer_id == self.client_id:
                continue

            try:
                df = pd.read_csv(os.path.join(DATA_DIR, f"{peer_id}.csv"))
                peer_overlap = df["peer_overlap_ratio"].mean()
                discovery_ratio = df["discovery_ratio"].mean()
                proto_entropy = df["protocol_entropy"].mean()

                # Compute anomaly score
                anomaly_score = (
                    max(0, (peer_overlap - baseline["peer_overlap"]) * 2.0) +
                    max(0, (discovery_ratio - baseline["discovery_ratio"]) * 2.0) +
                    max(0, (baseline["proto_entropy"] - proto_entropy) * 2.0)
                )

                # Assign severity
                if anomaly_score > 3:
                    severity_label = "high"
                elif anomaly_score > 1.5:
                    severity_label = "low"
                else:
                    severity_label = "honest"

                if severity_label != "honest":
                    report["suspicious_nodes"].append({
                        "node_id": peer_id,
                        "severity": severity_label
                    })

            except Exception as e:
                print(f"[{self.client_id}] Skipped {peer_id}: {e}")

        # Save only detected (non-honest) nodes
        out_file = os.path.join(REPORT_DIR, f"{self.client_id}_report.json")
        with open(out_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[✓] {self.client_id} detected {len(report['suspicious_nodes'])} suspicious nodes.")

    def evaluate_and_log_final_metrics(self, start_time):
        import hashlib

        def get_model_hash(model):
            buffer = []
            for param in model.parameters():
                buffer.append(param.detach().cpu().numpy().tobytes())
            model_bytes = b"".join(buffer)
            return hashlib.sha256(model_bytes).hexdigest()

        end_time = time.time()
        model_path = f"{MODEL_DIR}/{self.client_id}.pt"
        overhead = os.path.getsize(model_path)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test)
            y_pred = outputs.argmax(1).cpu().numpy()
            y_true = self.y_test.cpu().numpy()
            test_loss = self.loss_fn(outputs, self.y_test).item()
            test_acc = (y_pred == y_true).mean()

        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # ===== Summary Metrics =====
        metrics = {
            "node_id": self.client_id,
            "test_loss": round(test_loss, 4),
            "accuracy": round(test_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "computation_time": round(end_time - start_time, 4),
            "overhead_bytes": overhead
        }

        # ===== Per-node Predictions =====
        class_map = {0: "honest", 1: "low", 2: "high"}
        node_reports = []
        for node_id, pred in zip(self.x_test_node_id, y_pred):
            node_reports.append({
                "node_id": node_id,
                "severity": class_map[pred]
            })

        # ===== Final Report Structure =====
        prediction_report = {
            "client_id": self.client_id,
            "model_hash": get_model_hash(self.model),
            "timestamp": time.strftime("%y-%m-%d-%H-%M-%S"),
            "report": node_reports
        }

        # ===== Write Reports =====
        with open(os.path.join(REPORT_DIR, f"{self.client_id}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        with open(os.path.join(REPORT_DIR, f"{self.client_id}_predictions.json"), "w") as f:
            json.dump(prediction_report, f, indent=2)

        print(f"\n Final Report for {self.client_id}:")
        print(json.dumps(metrics, indent=2))


    def _save_learning_curves(self):
        ########### each client has only one figure combining loss and acc:
        max_epochs_to_plot = 100
        epochs_range = range(1, max_epochs_to_plot + 1)

        fig, ax1 = plt.subplots(figsize=(8, 4))

        # Accuracy (Left Y-axis)
        ax1.plot(epochs_range, self.train_acc_hist[:max_epochs_to_plot], label="Train Accuracy", color='tab:blue')
        ax1.plot(epochs_range, self.val_acc_hist[:max_epochs_to_plot], label="Val Accuracy", color='tab:cyan', linestyle='--')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True)

        # Loss (Right Y-axis)
        ax2 = ax1.twinx()
        ax2.plot(epochs_range, self.train_loss_hist[:max_epochs_to_plot], label="Train Loss", color='tab:red')
        ax2.plot(epochs_range, self.val_loss_hist[:max_epochs_to_plot], label="Val Loss", color='tab:orange', linestyle='--')
        ax2.set_ylabel("Loss", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize='small')

        fig.suptitle(f"{self.client_id} - Accuracy & Loss (First Round)")
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

        filename = os.path.join(LOG_DIR, f"{self.client_id}_acc_loss_combined_first_round.png")
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"[✓] Saved combined accuracy/loss curve for {self.client_id}")

        ################### each client has two figures:
        # # Only use the first 100 epochs
        # max_epochs_to_plot = 100

        # # Loss Curve
        # plt.figure()
        # plt.plot(range(1, max_epochs_to_plot + 1), self.train_loss_hist[:max_epochs_to_plot], label="Train Loss")
        # plt.plot(range(1, max_epochs_to_plot + 1), self.val_loss_hist[:max_epochs_to_plot], label="Validation Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title("")#f"{self.client_id} - Loss Curve (First Round Only)")
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(LOG_DIR, f"{self.client_id}_loss_curve_first_round.png"), dpi=300)
        # plt.close()

        # # Accuracy Curve
        # plt.figure()
        # plt.plot(range(1, max_epochs_to_plot + 1), self.train_acc_hist[:max_epochs_to_plot], label="Train Accuracy")
        # plt.plot(range(1, max_epochs_to_plot + 1), self.val_acc_hist[:max_epochs_to_plot], label="Validation Accuracy")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.title("")#f"{self.client_id} - Accuracy Curve (First Round Only)")
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(LOG_DIR, f"{self.client_id}_accuracy_curve_first_round.png"), dpi=300)
        # plt.close()
##################### first one:
        # epochs = range(1, len(self.train_loss_hist) + 1)

        # # Loss Curve
        # plt.figure()
        # plt.plot(epochs, self.train_loss_hist, label="Train Loss")
        # plt.plot(epochs, self.val_loss_hist, label="Validation Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title(f"{self.client_id} - Loss Curve")
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(LOG_DIR, f"{self.client_id}_loss_curve.png"), dpi=300)
        # plt.close()

        # # Accuracy Curve
        # plt.figure()
        # plt.plot(epochs, self.train_acc_hist, label="Train Accuracy")
        # plt.plot(epochs, self.val_acc_hist, label="Validation Accuracy")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.title(f"{self.client_id} - Accuracy Curve")
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(LOG_DIR, f"{self.client_id}_accuracy_curve.png"), dpi=300)
        # plt.close()

        print(f"[✓] Saved learning curves for {self.client_id}")

    def run(self):
        print(f" Starting node {self.client_id}")
        start = time.time()
        for round_ in range(ROUNDS):
            print(f" {self.client_id} - Round {round_+1}")
            for epoch in range(EPOCHS):
                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc = self.evaluate(self.X_val, self.y_val)
                self.log_epoch(round_, epoch, train_loss, train_acc, val_loss, val_acc)
                self.train_loss_hist.append(train_loss)
                self.train_acc_hist.append(train_acc)
                self.val_loss_hist.append(val_loss)
                self.val_acc_hist.append(val_acc)

            self.save_model()
            time.sleep(5)
            peer_weights = self.load_peer_models()
            self.average_with_peers(peer_weights)
        self.evaluate(self.X_test, self.y_test)
        #self.generate_report()
        self._save_learning_curves()
        self.evaluate_and_log_final_metrics(start)
        

# ---- Multiprocessing Entry Point ----
def launch_node(node_id):
    node = DecentralizedNode(node_id)
    node.run()

if __name__ == "__main__":
    processes = []
    for client_id in PEERS:
        p = multiprocessing.Process(target=launch_node, args=(client_id,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("✅ DFL Simulation Completed.")

