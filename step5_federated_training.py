import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class FederatedContractSimulator:
    def __init__(self):
        self.model_updates = {}

    def submit_model_update(self, node_id, accuracy):
        self.model_updates[node_id] = accuracy
        print(f"[+] {node_id} submitted update with accuracy {accuracy:.3f}")

    def aggregate(self):
        if not self.model_updates:
            return None
        avg = sum(self.model_updates.values()) / len(self.model_updates)
        print(f"[*] Aggregated model accuracy: {avg:.3f}")
        return avg

class ReputationContractSimulator:
    def __init__(self, reputations):
        self.reputation_scores = reputations

    def flag_suspicious_nodes(self, threshold=-3.5):
        flagged = [nid for nid, r in self.reputation_scores.items() if r['reputation'] <= threshold]
        print(f"[!] Flagged nodes for banning: {flagged}")
        return set(flagged)

with open("node_reputations.json") as f:
    reputations = json.load(f)

DATA_DIR = "data_per_node"
global_accuracies = []

for round_num in range(1, 6):
    print(f"\n===== Federated Round {round_num} =====")
    rep_contract = ReputationContractSimulator(reputations)
    fl_contract = FederatedContractSimulator()
    flagged_nodes = rep_contract.flag_suspicious_nodes()

    local_models = []
    feature_columns = None

    for file in os.listdir(DATA_DIR):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(path)
        node_id = df.iloc[0]["node_id"]

        if node_id in flagged_nodes:
            continue

        y = df["is_sybil"].astype(int)
        X = df.drop(columns=["node_id", "is_sybil"])

        print(f"[DEBUG] {file}: rows={len(df)}, classes={y.unique().tolist()}")

        if len(np.unique(y)) < 2:
            print(f"[-] Skipping {node_id}: only one class present in data")
            continue

        print(f"[✓] Using {node_id} — class diversity OK")

        if feature_columns is None:
            feature_columns = X.columns.tolist()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            if len(df) <= 4:
                model = LogisticRegression(max_iter=1000)
                model.fit(X_scaled, y)
                acc = model.score(X_scaled, y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)

            local_models.append(model.coef_[0])
            fl_contract.submit_model_update(node_id, acc)
        except Exception as e:
            print(f"[!] Error training model for {node_id}: {e}")
            continue

    if local_models:
        global_weights = np.mean(local_models, axis=0)
        print("[*] Aggregated global model weights via FedAvg")

        y_true_all, y_pred_all = [], []
        for file in os.listdir(DATA_DIR):
            if not file.endswith(".csv"):
                continue

            path = os.path.join(DATA_DIR, file)
            df = pd.read_csv(path)
            node_id = df.iloc[0]["node_id"]
            if node_id in flagged_nodes:
                continue

            y = df["is_sybil"].astype(int)
            X = df[feature_columns]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logits = np.dot(X_scaled, global_weights)
            preds = (logits > 0).astype(int)

            y_true_all.extend(y.tolist())
            y_pred_all.extend(preds.tolist())

        global_acc = accuracy_score(y_true_all, y_pred_all)
        global_accuracies.append(global_acc)
        print(f"[✓] Global model accuracy across honest nodes: {global_acc:.3f}")
    else:
        print("[!] No models aggregated this round")
        global_accuracies.append(0.0)

plt.plot(range(1, 6), global_accuracies, marker='o')
plt.title("Global Model Accuracy over FL Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.savefig("fl_accuracy_trend.png")
print("[✓] Accuracy trend saved as fl_accuracy_trend.png")