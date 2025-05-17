# ####################### Improved version :D
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# import multiprocessing
# import numpy as np
# import time
# from nn_models import get_model

# # === CONFIGURATION ===
# ROUNDS = 5
# EPOCHS = 200
# PEERS = [f"client_{i}" for i in range(5)]
# FEATURE_COLUMNS = [
#     "sent_total", "received_total", "protocol_diversity",
#     "message_burstiness", "mqtt_ratio", "discovery_ratio",
#     "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
#     "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity",
#     "latency_spike_ratio", "energy_variance", "peer_overlap_ratio"
# ]

# lock = multiprocessing.Lock()

# def ensure_dir(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# class DecentralizedNode:
#     def __init__(self, node_id):
#         self.node_id = node_id
#         self.model = get_model()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
#         self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=EPOCHS)
#         self.model_dir = "models"
#         self.log_dir = "logs"
#         ensure_dir(self.model_dir)
#         ensure_dir(self.log_dir)
#         self._load_and_split_data()

#     def _load_and_split_data(self):
#         path = f"nodes3_data/{self.node_id}.csv"
#         df = pd.read_csv(path)

#         X = df[FEATURE_COLUMNS].values
#         y = df["is_sybil"].values

#         X_train, X_temp, y_train, y_temp = train_test_split(
#             X, y, test_size=0.4, stratify=y, random_state=42)
#         X_val, X_test, y_val, y_test = train_test_split(
#             X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

#         scaler = StandardScaler()
#         self.X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
#         self.y_train = torch.tensor(y_train, dtype=torch.long)

#         self.X_val = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
#         self.y_val = torch.tensor(y_val, dtype=torch.long)

#         self.X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
#         self.y_test = torch.tensor(y_test, dtype=torch.long)

#         weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
#         class_weights = torch.tensor(weights, dtype=torch.float32)
#         self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

#     def train_epoch(self):
#         self.model.train()
#         self.optimizer.zero_grad()
#         outputs = self.model(self.X_train.unsqueeze(1))
#         loss = self.loss_fn(outputs, self.y_train)
#         loss.backward()
#         self.optimizer.step()
#         self.scheduler.step(loss) 
#         train_acc = (outputs.argmax(1) == self.y_train).float().mean().item()
#         return loss.item(), train_acc

#     def evaluate(self, X, y):
#         self.model.eval()
#         with torch.no_grad():
#             outputs = self.model(X.unsqueeze(1))
#             loss = self.loss_fn(outputs, y)
#             acc = (outputs.argmax(1) == y).float().mean().item()
#             return loss.item(), acc

#     def save_model(self):
#         with lock:
#             torch.save(self.model.state_dict(), f"{self.model_dir}/{self.node_id}.pt")

#     def load_peer_models(self):
#         peer_weights = []
#         with lock:
#             for peer in PEERS:
#                 if peer == self.node_id:
#                     continue
#                 model_path = f"{self.model_dir}/{peer}.pt"
#                 if os.path.exists(model_path):
#                     state_dict = torch.load(model_path)
#                     peer_weights.append(state_dict)
#         return peer_weights

#     def average_with_peers(self, peer_weights):
#         if not peer_weights:
#             return
#         new_state = {}
#         my_state = self.model.state_dict()
#         for key in my_state:
#             avg = sum([peer[key] for peer in peer_weights] + [my_state[key]]) / (len(peer_weights) + 1)
#             new_state[key] = avg
#         self.model.load_state_dict(new_state)

#     def log_epoch(self, round, epoch, train_loss, train_acc, val_loss, val_acc):
#         with open(f"{self.log_dir}/{self.node_id}_log.txt", "a") as log_file:
#             log_file.write(
#                 f"Round {round+1}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n"
#             )

#     def run(self):
#         print(f"🚀 Starting node {self.node_id}")
#         for round in range(ROUNDS):
#             print(f"[⏱] {self.node_id} - Round {round + 1}")
#             for epoch in range(EPOCHS):
#                 train_loss, train_acc = self.train_epoch()
#                 val_loss, val_acc = self.evaluate(self.X_val, self.y_val)
#                 self.log_epoch(round, epoch, train_loss, train_acc, val_loss, val_acc)

#             self.save_model()
#             time.sleep(5)
#             peer_weights = self.load_peer_models()
#             self.average_with_peers(peer_weights)

#         test_loss, test_acc = self.evaluate(self.X_test, self.y_test)
#         print(f"[🏁] {self.node_id} Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2%}")

# if __name__ == "__main__":
#     processes = []
#     for node_id in PEERS:
#         p = multiprocessing.Process(target=DecentralizedNode(node_id).run)
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

#     print("[✅] DFL Simulation Completed.")
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import multiprocessing
import numpy as np
import time
import hashlib
import json
from nn_models import get_model
from sklearn.metrics import precision_score, recall_score, f1_score

ROUNDS = 5
EPOCHS = 200 ## i changed it
PEERS = [f"client_{i}" for i in range(5)]
FEATURE_COLUMNS = [
    "sent_total", "received_total", "protocol_diversity",
    "message_burstiness", "mqtt_ratio", "discovery_ratio",
    "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
    "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity",
    "latency_spike_ratio", "energy_variance", "peer_overlap_ratio"
]

REPORT_DIR = "reports"
lock = multiprocessing.Lock()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_model_hash(model):
    flat_params = torch.cat([p.flatten() for p in model.parameters()]).detach().numpy()
    byte_data = bytes(np.array(flat_params, dtype=np.float32))
    return hashlib.sha256(byte_data).hexdigest()

class DecentralizedNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.model = get_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=EPOCHS)
        self.model_dir = "models"
        self.log_dir = "logs"
        ensure_dir(self.model_dir)
        ensure_dir(self.log_dir)
        ensure_dir(REPORT_DIR)
        self._load_and_split_data()

    def _load_and_split_data(self):
        path = f"nodes3_data/{self.node_id}.csv"
        df = pd.read_csv(path)

        X = df[FEATURE_COLUMNS].values
        y = df["is_sybil"].values

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        scaler = StandardScaler()
        self.X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_val = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.long)
        self.X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

        weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(self.X_train.unsqueeze(1))
        loss = self.loss_fn(outputs, self.y_train)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        train_acc = (outputs.argmax(1) == self.y_train).float().mean().item()
        return loss.item(), train_acc

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X.unsqueeze(1))
            loss = self.loss_fn(outputs, y)
            acc = (outputs.argmax(1) == y).float().mean().item()
            return loss.item(), acc

    def save_model(self):
        with lock:
            torch.save(self.model.state_dict(), f"{self.model_dir}/{self.node_id}.pt")

    def load_peer_models(self):
        peer_weights = []
        with lock:
             for peer in PEERS:
                if peer == self.node_id:
                    continue
                model_path = f"{self.model_dir}/{peer}.pt"
                if os.path.exists(model_path):
                    state_dict = torch.load(model_path)
                    peer_weights.append(state_dict)
        return peer_weights

    def average_with_peers(self, peer_weights):
        if not peer_weights:
            return
        new_state = {}
        my_state = self.model.state_dict()
        for key in my_state:
            avg = sum([peer[key] for peer in peer_weights] + [my_state[key]]) / (len(peer_weights) + 1)
            new_state[key] = avg
        self.model.load_state_dict(new_state)

    def log_epoch(self, round, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(f"{self.log_dir}/{self.node_id}_log.txt", "a") as log_file:
            log_file.write(
                f"Round {round+1}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n"
            )

    def generate_report(self):
        report = {
            "node_id": self.node_id,
            "model_hash": get_model_hash(self.model),
            "timestamp": int(time.time()),
            "reports": []
        }

        # === Baselines (tunable) ===
        baseline = {
            "peer_overlap": 0.3,
            "discovery_ratio": 0.2,
            "proto_entropy": 0.75
        }

        #for peer_id in PEERS:
        all_nodes = [f[:-4] for f in os.listdir("nodes3_data") if f.endswith(".csv")]
        for peer_id in all_nodes:
            if peer_id == self.node_id:
                continue  # Skip self
            # if peer_id == self.node_id:
                # continue  # Skip self

            try:
                df = pd.read_csv(f"nodes3_data/{peer_id}.csv")
                peer_overlap = df["peer_overlap_ratio"].mean()
                discovery_ratio = df["discovery_ratio"].mean()
                proto_entropy = df["protocol_entropy"].mean()

                # Composite anomaly score
                anomaly_score = (
                    max(0, (peer_overlap - baseline["peer_overlap"]) * 2.0) +
                    max(0, (discovery_ratio - baseline["discovery_ratio"]) * 2.0) +
                    max(0, (baseline["proto_entropy"] - proto_entropy) * 2.0)
                )

                # Map to severity
                if anomaly_score > 3:
                    severity = 8
                elif anomaly_score > 2:
                    severity = 5
                elif anomaly_score > 1:
                    severity = 3
                else:
                    severity = 0

                if severity > 0:
                    report["reports"].append({
                        "suspect_node": peer_id,
                        "severity": severity,
                        "anomaly_score": round(anomaly_score, 3),
                        "features": {
                            "peer_overlap": round(peer_overlap, 3),
                            "discovery_ratio": round(discovery_ratio, 3),
                            "protocol_entropy": round(proto_entropy, 3)
                        }
                    })

            except Exception as e:
                print(f" Skipping {peer_id}: {e}")

        # Save final report
        with open(f"{REPORT_DIR}/{self.node_id}_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f" {self.node_id} report saved with {len(report['reports'])} suspects.")


    # def generate_report(self):
    #     df = pd.read_csv(f"nodes3_data/{self.node_id}.csv")

    #     # Compute averages of key behavioral features
    #     peer_overlap = df["peer_overlap_ratio"].mean()
    #     discovery_ratio = df["discovery_ratio"].mean()
    #     proto_entropy = df["protocol_entropy"].mean()

    #     # Heuristic to assign severity
    #     if peer_overlap > 0.75 and discovery_ratio > 0.4 and proto_entropy < 0.6:
    #         severity = 8
    #     elif peer_overlap > 0.6 and proto_entropy < 0.5:
    #         severity = 5
    #     else:
    #         severity = 0

    #     model_hash = get_model_hash(self.model)

    #     # Generate single-node report
    #     report = {
    #         "node_id": self.node_id,
    #         "model_hash": model_hash,
    #         "timestamp": int(time.time()),
    #         "reports": []
    #     }

    #     if severity > 0:
    #         report["reports"].append({
    #             "suspect_node": self.node_id,
    #             "severity": severity
    #         })

    #     with open(f"{REPORT_DIR}/{self.node_id}_report.json", "w") as f:
    #         json.dump(report, f, indent=2)

    #     print(f" Saved report for {self.node_id} with severity {severity}")


    def evaluate_and_log_final_metrics(self, start_time):
        end_time = time.time()
        computation_time = end_time - start_time
        model_path = f"{self.model_dir}/{self.node_id}.pt"
        overhead = os.path.getsize(model_path)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test.unsqueeze(1))
            y_pred = outputs.argmax(1).cpu().numpy()
            y_true = self.y_test.cpu().numpy()
            test_loss = self.loss_fn(outputs, self.y_test).item()
            test_acc = (y_pred == y_true).mean()

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Save to JSON
        metrics = {
            "node_id": self.node_id,
            "test_loss": round(test_loss, 4),
            "accuracy": round(test_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "computation_time": round(computation_time, 4),
            "overhead_bytes": overhead
        }

        metrics_file = os.path.join(REPORT_DIR, f"{self.node_id}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Console Output
        print(f"\n Final Report for {self.node_id}:")
        print(f" Test Loss: {test_loss:.4f}")
        print(f" Accuracy: {test_acc:.4f}")
        print(f" Precision: {precision:.4f}")
        print(f" Recall: {recall:.4f}")
        print(f" F1 Score: {f1:.4f}")
        print(f" Computation Time: {computation_time:.4f} seconds")
        print(f" Overhead (Model Size): {overhead} bytes")
        print(f" Saved metrics to {metrics_file}")

    def run(self):
        print(f" Starting node {self.node_id}")
        start_time = time.time()

        for round_ in range(ROUNDS):
            print(f" {self.node_id} - Round {round_ + 1}")
            for epoch in range(EPOCHS):
                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc = self.evaluate(self.X_val, self.y_val)
                self.log_epoch(round_, epoch, train_loss, train_acc, val_loss, val_acc)

            self.save_model()
            time.sleep(5)
            peer_weights = self.load_peer_models()
            self.average_with_peers(peer_weights)

        test_loss, test_acc = self.evaluate(self.X_test, self.y_test)
        print(f" {self.node_id} Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2%}")

        self.generate_report()
        self.evaluate_and_log_final_metrics(start_time)

if __name__ == "__main__":
    processes = []
    for node_id in PEERS:
        p = multiprocessing.Process(target=DecentralizedNode(node_id).run)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(" DFL Simulation Completed.")
