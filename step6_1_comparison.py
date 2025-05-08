
import os
import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# === Step 1: Autoencoder Anomaly Detection Export ===
data_dir = "data_per_node"
output_file = "autoencoder_anomalies.json"
anomaly_results = {}

for fname in sorted(os.listdir(data_dir)):
    if fname.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_dir, fname))
        label_col = "label" if "label" in df.columns else "is_sybil"
        X = df.drop(columns=[label_col, "node_id"])
        anomalies = [i for i, row in X.iterrows() if (row > 0.5).any()]  # simulated reconstruction error
        anomaly_results[fname] = {
            "is_sybil": bool(df[label_col].iloc[0]),
            "autoencoder_flagged": bool(anomalies),
            "row_indices": anomalies
        }

with open(output_file, "w") as f:
    json.dump(anomaly_results, f, indent=2)
print(f"[+] Saved autoencoder anomaly report to {output_file}")

# === Step 2: Comparison with Federated Learning Predictions ===
class TorchNodeModel(torch.nn.Module):
    def __init__(self, input_size):
        super(TorchNodeModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 16)
        self.fc2 = torch.nn.Linear(16, 8)
        self.fc3 = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

global_model_path = "global_model_round5.pth"
global_model = None
input_size = None

true_labels = []
auto_preds = []
fed_preds = []

for fname in sorted(os.listdir(data_dir)):
    if fname.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_dir, fname))
        label_col = "label" if "label" in df.columns else "is_sybil"
        y_true = int(df[label_col].iloc[0])
        X = df.drop(columns=[label_col, "node_id"])
        x_tensor = torch.tensor(X.values, dtype=torch.float32)

        if global_model is None:
            input_size = X.shape[1]
            global_model = TorchNodeModel(input_size)
            global_model.load_state_dict(torch.load(global_model_path))

        with torch.no_grad():
            pred = (global_model(x_tensor) > 0.5).float().item()

        true_labels.append(y_true)
        fed_preds.append(int(pred))
        auto_preds.append(int(anomaly_results[fname]["autoencoder_flagged"]))

# === Export classification reports ===
with open("classification_report.txt", "w") as f:
    f.write("=== Autoencoder ===\n")
    f.write(classification_report(true_labels, auto_preds))
    f.write("\n=== Federated Learning ===\n")
    f.write(classification_report(true_labels, fed_preds))

# === Plot confusion matrices ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

cm_auto = confusion_matrix(true_labels, auto_preds)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_auto, display_labels=["Honest", "Sybil"])
disp1.plot(ax=axs[0], values_format='d')
axs[0].set_title("Autoencoder")

cm_fed = confusion_matrix(true_labels, fed_preds)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_fed, display_labels=["Honest", "Sybil"])
disp2.plot(ax=axs[1], values_format='d')
axs[1].set_title("Federated Learning")

plt.tight_layout()
plt.savefig("step6_autoencoder_vs_federated_comparison.png")
print("Saved confusion matrix plot to step6_autoencoder_vs_federated_comparison.png")
