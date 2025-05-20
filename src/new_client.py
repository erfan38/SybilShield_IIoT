##################### Using Data_loader.py

# Modified client.py to use DataLoaderWrapper instead of reading CSV directly
import multiprocessing
import os, json
import pandas as pd
import numpy as np
import torch, time
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from sklearn.metrics import precision_score, recall_score, f1_score
from nn_models import get_model
from data_loader import DataLoaderWrapper

REPORT_DIR = "reports/FL"
MODEL_DIR = "models/FL"
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH = "data/dataset_25-05-19-11-34-50.csv"
NUM_CLIENTS = 5
FEATURE_COLUMNS = [
    "sent_total", "received_total", "protocol_diversity",
    "message_burstiness", "mqtt_ratio", "discovery_ratio",
    "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
    "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity"
]

def evaluate_and_log_final_metrics(model, X_test, y_test, loss_fn, node_id, start_time, model_path):
    end_time = time.time()
    computation_time = end_time - start_time
    overhead = os.path.getsize(model_path)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        y_pred = outputs.argmax(1).cpu().numpy()
        y_true = y_test.cpu().numpy()
        test_loss = loss_fn(outputs, y_test).item()
        test_acc = (y_pred == y_true).mean()

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {
        "node_id": node_id,
        "test_loss": round(test_loss, 4),
        "accuracy": round(test_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "computation_time": round(computation_time, 4),
        "overhead_bytes": overhead
    }

    metrics_file = os.path.join(REPORT_DIR, f"{node_id}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n Final Report for {node_id}:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

def run_node(node_id):
    print(f" Launching client {node_id}")
    start_time = time.time()

    loader = DataLoaderWrapper(csv_path=CSV_PATH, number_of_clients=NUM_CLIENTS)

    X_train = loader.client_data["train"][node_id]["x"]
    y_train = loader.client_data["train"][node_id]["y"]
    X_test  = loader.client_data["test"][node_id]["x"]
    y_test  = loader.client_data["test"][node_id]["y"]

    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    mu = 0.001

    def get_parameters():
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(params):
        state_dict = model.state_dict()
        for k, v in zip(state_dict.keys(), params):
            state_dict[k] = torch.tensor(v)
        model.load_state_dict(state_dict)

    def train(initial_params):
        model.train()
        for _ in range(100):
            optimizer.zero_grad()
            outputs = model(X_train)
            ce_loss = loss_fn(outputs, y_train)
            prox_loss = sum(((param - init) ** 2).sum() for param, init in zip(model.parameters(), initial_params))
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

            model_path = os.path.join(MODEL_DIR, f"{node_id}.pt")
            torch.save(model.state_dict(), model_path)

            evaluate_and_log_final_metrics(
                model=model,
                X_test=X_test,
                y_test=y_test,
                loss_fn=loss_fn,
                node_id=node_id,
                start_time=start_time,
                model_path=model_path
            )

            return get_parameters(), len(X_train), {}

        def evaluate(self, parameters, config):
            set_parameters(parameters)
            acc = (model(X_test).argmax(1) == y_test).float().mean().item()
            print(f"[\u2713] {node_id} accuracy: {acc:.2%}")
            return 1 - acc, len(X_test), {}

    fl.client.start_client(server_address="127.0.0.1:9090", client=FLClient().to_client())

if __name__ == "__main__":
    node_ids = [f"client_{i}" for i in range(NUM_CLIENTS)]
    processes = []

    for node_id in node_ids:
        p = multiprocessing.Process(target=run_node, args=(node_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(" All clients completed.")
