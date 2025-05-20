import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

FEATURE_COLUMNS = [
    "sent_total", "received_total", "protocol_diversity",
    "message_burstiness", "mqtt_ratio", "discovery_ratio",
    "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
    "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity",
    "latency_spike_ratio", "energy_variance", "peer_overlap_ratio"
]

class DataLoaderWrapper:
    def __init__(self, csv_path, test_size=0.2, val_size=0.25, seed=42, number_of_clients=None):
        self.csv_path = csv_path
        self.feature_columns = FEATURE_COLUMNS
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.number_of_clients = number_of_clients
        self._load_and_split_data()

    def _split_among_clients(self, X, y, node_ids):
        client_data = {}
        X_parts = np.array_split(X, self.number_of_clients)
        y_parts = np.array_split(y, self.number_of_clients)
        id_parts = np.array_split(node_ids, self.number_of_clients)

        for i in range(self.number_of_clients):
            client_key = f"client_{i}"
            client_data[client_key] = {
                "x": torch.tensor(X_parts[i], dtype=torch.float32),
                "y": torch.tensor(y_parts[i], dtype=torch.long),
                "node_id": id_parts[i]
            }
        return client_data

    def _load_and_split_data(self):
        df = pd.read_csv(self.csv_path)
        X = df[self.feature_columns].values
        y = df["severity"].map({"honest": 0, "low": 1, "high": 2}).values
        node_ids = df["node_id"].values

        # Split: Train+Val vs Test
        X_train_val, X_test, y_train_val, y_test, id_train_val, id_test = train_test_split(
            X, y, node_ids,
            test_size=self.test_size, stratify=y, random_state=self.seed
        )

        # Split: Train vs Val
        rel_val_size = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
            X_train_val, y_train_val, id_train_val,
            test_size=rel_val_size, stratify=y_train_val, random_state=self.seed
        )

        # Normalize features
        scaler = StandardScaler()
        self.X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
        self.X_val = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
        self.X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.y_val = torch.tensor(y_val, dtype=torch.long)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

        self.x_train_node_id = id_train
        self.x_val_node_id = id_val
        self.x_test_node_id = id_test


        # Optional: Split train/val/test among clients
        if self.number_of_clients:
            self.client_data = {
                "train": self._split_among_clients(self.X_train.numpy(), self.y_train.numpy(), self.x_train_node_id),
                "val": self._split_among_clients(self.X_val.numpy(), self.y_val.numpy(), self.x_val_node_id),
                "test": self._split_among_clients(self.X_test.numpy(), self.y_test.numpy(), self.x_test_node_id),
            }
        else:
            self.client_data = None
