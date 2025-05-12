# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# import time
# import multiprocessing
# from sklearn.preprocessing import StandardScaler
# from nn_models import get_model

# # === CONFIGURATION ===
# ROUNDS = 5
# EPOCHS = 200
# PEERS = [f"client_{i}" for i in range(5)]  # You can increase number of nodes
# FEATURE_COLUMNS = [
#     "sent_total", "received_total", "protocol_diversity",
#     "message_burstiness", "mqtt_ratio", "discovery_ratio",
#     "protocol_entropy", "unique_peers", "dominant_protocol_ratio"
# ]

# def ensure_dir(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# class DecentralizedNode:
#     def __init__(self, node_id):
#         self.node_id = node_id
#         self.model = get_model()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.model_dir = "models"
#         ensure_dir(self.model_dir)
#         self._load_data()

#     def _load_data(self):
#         path = f"nodes_data/{self.node_id}.csv"
#         df = pd.read_csv(path)
#         X = df[FEATURE_COLUMNS].values
#         y = df["is_sybil"].values
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
#         self.X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
#         self.y_tensor = torch.tensor(y, dtype=torch.long)
        

#     def train(self, epochs):
#         self.model.train()
#         for _ in range(epochs):
#             self.optimizer.zero_grad()
#             outputs = self.model(self.X_tensor.unsqueeze(1))
#             loss = self.loss_fn(outputs, self.y_tensor)
#             loss.backward()
#             self.optimizer.step()

#     def evaluate(self):
#         self.model.eval()
#         with torch.no_grad():
#             outputs = self.model(self.X_tensor.unsqueeze(1))
#             acc = (outputs.argmax(1) == self.y_tensor).float().mean().item()
#             return acc

#     def save_model(self):
#         torch.save(self.model.state_dict(), f"{self.model_dir}/{self.node_id}.pt")

#     def load_peer_models(self):
#         peer_weights = []
#         for peer in PEERS:
#             if peer == self.node_id:
#                 continue
#             model_path = f"{self.model_dir}/{peer}.pt"
#             if os.path.exists(model_path):
#                 state_dict = torch.load(model_path)
#                 peer_weights.append(state_dict)
#         return peer_weights

#     def average_with_peers(self, peer_weights):
#         if not peer_weights:
#             return
#         new_state = {}
#         my_state = self.model.state_dict()
#         for key in my_state:
#             avg = sum([peer[key] for peer in peer_weights]) / len(peer_weights)
#             new_state[key] = (avg + my_state[key]) / 2  # Optionally adjust weighting
#         self.model.load_state_dict(new_state)

#     def run(self):
#         print(f"🚀 Starting node {self.node_id}")
#         for round in range(ROUNDS):
#             print(f"[⏱] {self.node_id} - Round {round + 1}")
#             self.train(EPOCHS)
#             self.save_model()
#             time.sleep(3)  # Give peers time to save
#             peer_weights = self.load_peer_models()
#             self.average_with_peers(peer_weights)
#             acc = self.evaluate()
#             print(f"[📊] {self.node_id} Accuracy after round {round + 1}: {acc:.2%}")

# if __name__ == "__main__":
#     processes = []
#     for node_id in PEERS:
#         p = multiprocessing.Process(target=DecentralizedNode(node_id).run)
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

#     print("[✅] DFL Simulation Completed.")






################# ACCuracy 78##############################
############################################################
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import multiprocessing
# from nn_models import get_model

# # === CONFIGURATION ===
# ROUNDS = 5
# EPOCHS = 200
# PEERS = [f"client_{i}" for i in range(5)]
# FEATURE_COLUMNS = [
#     "sent_total", "received_total", "protocol_diversity",
#     "message_burstiness", "mqtt_ratio", "discovery_ratio",
#     "protocol_entropy", "unique_peers", "dominant_protocol_ratio"
# ]

# def ensure_dir(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# class DecentralizedNode:
#     def __init__(self, node_id):
#         self.node_id = node_id
#         self.model = get_model()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.model_dir = "models"
#         self.log_dir = "logs"
#         ensure_dir(self.model_dir)
#         ensure_dir(self.log_dir)
#         self._load_and_split_data()

#     def _load_and_split_data(self):
#         path = f"nodes_data/{self.node_id}.csv"
#         df = pd.read_csv(path)

#         X = df[FEATURE_COLUMNS].values
#         y = df["is_sybil"].values

#         X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
#         X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

#         scaler = StandardScaler()
#         self.X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
#         self.y_train = torch.tensor(y_train, dtype=torch.long)

#         self.X_val = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
#         self.y_val = torch.tensor(y_val, dtype=torch.long)

#         self.X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
#         self.y_test = torch.tensor(y_test, dtype=torch.long)

#     def train_epoch(self):
#         self.model.train()
#         self.optimizer.zero_grad()
#         outputs = self.model(self.X_train.unsqueeze(1))
#         loss = self.loss_fn(outputs, self.y_train)
#         loss.backward()
#         self.optimizer.step()

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
#         torch.save(self.model.state_dict(), f"{self.model_dir}/{self.node_id}.pt")

#     def load_peer_models(self):
#         peer_weights = []
#         for peer in PEERS:
#             if peer == self.node_id:
#                 continue
#             model_path = f"{self.model_dir}/{peer}.pt"
#             if os.path.exists(model_path):
#                 state_dict = torch.load(model_path)
#                 peer_weights.append(state_dict)
#         return peer_weights

#     def average_with_peers(self, peer_weights):
#         if not peer_weights:
#             return
#         new_state = {}
#         my_state = self.model.state_dict()
#         for key in my_state:
#             avg = sum([peer[key] for peer in peer_weights]) / len(peer_weights)
#             new_state[key] = (avg + my_state[key]) / 2
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


####################### Improved version :D
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
from nn_models import get_model

# === CONFIGURATION ===
ROUNDS = 5
EPOCHS = 200
PEERS = [f"client_{i}" for i in range(5)]
FEATURE_COLUMNS = [
    "sent_total", "received_total", "protocol_diversity",
    "message_burstiness", "mqtt_ratio", "discovery_ratio",
    "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
    "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity",
    "latency_spike_ratio", "energy_variance", "peer_overlap_ratio"
]
### features for dataset4:
# FEATURE_COLUMNS = [
#     "sent_total", "received_total", "protocol_diversity",
#     "message_burstiness", "mqtt_ratio", "discovery_ratio",
#     "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
#     "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity",
#     "latency_spike_ratio", "energy_variance", "peer_overlap_ratio",
#     "timing_entropy", "latency_cv", "findnode_ratio", "enr_message_ratio", "degree_centrality"
# ]
lock = multiprocessing.Lock()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

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
        self._load_and_split_data()

    def _load_and_split_data(self):
        path = f"nodes3_data/{self.node_id}.csv"
        df = pd.read_csv(path)

        X = df[FEATURE_COLUMNS].values
        y = df["is_sybil"].values

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

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
        self.scheduler.step(loss) 
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
                f"Round {round+1}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n"
            )

    def run(self):
        print(f"🚀 Starting node {self.node_id}")
        for round in range(ROUNDS):
            print(f"[⏱] {self.node_id} - Round {round + 1}")
            for epoch in range(EPOCHS):
                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc = self.evaluate(self.X_val, self.y_val)
                self.log_epoch(round, epoch, train_loss, train_acc, val_loss, val_acc)

            self.save_model()
            time.sleep(5)
            peer_weights = self.load_peer_models()
            self.average_with_peers(peer_weights)

        test_loss, test_acc = self.evaluate(self.X_test, self.y_test)
        print(f"[🏁] {self.node_id} Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2%}")

if __name__ == "__main__":
    processes = []
    for node_id in PEERS:
        p = multiprocessing.Process(target=DecentralizedNode(node_id).run)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("[✅] DFL Simulation Completed.")
