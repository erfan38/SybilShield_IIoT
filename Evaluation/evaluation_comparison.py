import os
import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nn_models import get_model  # Assumes your ConvNN or SimpleNN is available here

# === CONFIG ===
INPUT_DIR = "nodes3_data"
OUTPUT_CSV = "comparison_report.csv"
GLOBAL_TEST_FILE = os.path.join(INPUT_DIR, "global_test.csv")
FEATURE_COLUMNS = [
    "sent_total", "received_total", "protocol_diversity",
    "message_burstiness", "mqtt_ratio", "discovery_ratio",
    "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
    "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity"
]

MODELS = {
    "DL": "models/dl_model.pt",         #  offline trained model
    "FL": "global_model.pt",            # Federated model saved from server.py
    "DFL": "models/client_4.pt"         # One of the decentralized nodes
}

# === Step 1: Generate Shared Test Set ===
def generate_global_test():
    all_dfs = []
    for client_id in range(5):
        file = os.path.join(INPUT_DIR, f"client_{client_id}.csv")
        if os.path.exists(file):
            df = pd.read_csv(file)
            _, test_df = train_test_split(df, test_size=0.2, stratify=df["is_sybil"], random_state=42)
            all_dfs.append(test_df)
        else:
            print(f"[!] Missing file: {file}")
    global_test = pd.concat(all_dfs, ignore_index=True)
    global_test.to_csv(GLOBAL_TEST_FILE, index=False)
    print(f" Global test set saved to '{GLOBAL_TEST_FILE}' with {len(global_test)} samples.")

# === Step 2: Load Test Data ===
def load_test_data():
    df = pd.read_csv(GLOBAL_TEST_FILE)
    X = df[FEATURE_COLUMNS].values
    y = df["is_sybil"].values
    X_scaled = StandardScaler().fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)  # shape: [N, 1, F]
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor

# === Step 3: Evaluate Model ===
def evaluate_model(model_path, X_test, y_test):
    model = get_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        y_pred = outputs.argmax(1).cpu().numpy()
        y_true = y_test.cpu().numpy()

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "overhead_bytes": os.path.getsize(model_path)
    }

# === Step 4: Main Evaluation ===
def compare_all():
    if not os.path.exists(GLOBAL_TEST_FILE):
        generate_global_test()

    X_test, y_test = load_test_data()
    results = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[!] Model file not found: {path}")
            continue
        print(f" Evaluating {name}...")
        results[name] = evaluate_model(path, X_test, y_test)

    df = pd.DataFrame(results).T
    df.to_csv(OUTPUT_CSV, index=True)
    print("\n Comparison Report:")
    print(df.to_markdown())
    print(f"\n Metrics saved to '{OUTPUT_CSV}'")

if __name__ == "__main__":
    compare_all()
