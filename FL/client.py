import multiprocessing
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from sklearn.preprocessing import StandardScaler
from nn_models import SimpleNN, ConvNN, get_model


# class SimpleNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             # nn.Linear(9, 128), nn.ReLU(),
#             # nn.Dropout(0.3),
#             # nn.Linear(128, 64), nn.ReLU(),
#             # nn.Dropout(0.2),
#             # nn.Linear(64, 32), nn.ReLU(),
#             # nn.Linear(32, 2)
#             nn.Linear(9, 256), nn.ReLU(),
#             nn.Linear(256, 128), nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, 64), nn.ReLU(),
#             nn.Linear(64, 2)


#         )

    # def forward(self, x):
    #     return self.net(x)
# def get_model():
#     return ConvNN()

def run_node(node_id):
    print(f"🚀 Launching client {node_id}")
    path = f"nodes_data/{node_id}.csv"
    if not os.path.exists(path):
        print(f"[!] Missing data for {node_id}")
        return

    df = pd.read_csv(path)
    X = df[[
        "sent_total", "received_total", "protocol_diversity",
        "message_burstiness", "mqtt_ratio", "discovery_ratio",
        "protocol_entropy", "unique_peers", "dominant_protocol_ratio"]].values
    y = df["is_sybil"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    mu = 0.001  # FedProx regularization parameter

    def get_parameters():
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(params):
        state_dict = model.state_dict()
        for k, v in zip(state_dict.keys(), params):
            state_dict[k] = torch.tensor(v)
        model.load_state_dict(state_dict)

    def train(initial_params):
        model.train()
        for _ in range(50): # i changed it to 50
            optimizer.zero_grad()
            outputs = model(X_tensor)
            ce_loss = loss_fn(outputs, y_tensor)
            prox_loss = 0.0
            for param, init in zip(model.parameters(), initial_params):
                prox_loss += ((param - init) ** 2).sum()
            total_loss = ce_loss + mu * prox_loss
            total_loss.backward()
            optimizer.step()

    class FLClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return get_parameters()

        def fit(self, parameters, config):
            set_parameters(parameters)
            initial_params = [param.clone().detach() for param in model.parameters()]
            train(initial_params)
            return get_parameters(), len(X_tensor), {}

        def evaluate(self, parameters, config):
            set_parameters(parameters)
            acc = (model(X_tensor).argmax(1) == y_tensor).float().mean().item()
            print(f"[✓] {node_id} accuracy: {acc:.2%}")
            return 1 - acc, len(X_tensor), {}

    fl.client.start_client(server_address="127.0.0.1:9090", client=FLClient().to_client())

if __name__ == "__main__":
    node_ids = [f"client_{i}" for i in range(5)]
    processes = []

    for node_id in node_ids:
        p = multiprocessing.Process(target=run_node, args=(node_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("[✓] All clients completed.")
