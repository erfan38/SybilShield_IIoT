import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

class TorchNodeModel(nn.Module):
    def __init__(self, input_size):
        super(TorchNodeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    label_col = "label" if "label" in df.columns else "is_sybil"
    X = torch.tensor(df.drop(columns=[label_col, "node_id"]).values, dtype=torch.float32)
    y = torch.tensor(df[label_col].astype(float).values, dtype=torch.float32).unsqueeze(1)
    return X, y, df[label_col].astype(int).values, df

def predict_and_evaluate(model, X, y_true):
    model.eval()
    with torch.no_grad():
        y_pred = (model(X) > 0.5).int().numpy()
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    return report, cm, y_pred

# Automatically pick a Sybil node
data_dir = "data_per_node"
sybil_file = None

for fname in os.listdir(data_dir):
    if not fname.endswith(".csv"):
        continue
    df_check = pd.read_csv(os.path.join(data_dir, fname))
    label_col = "label" if "label" in df_check.columns else "is_sybil"
    if df_check[label_col].astype(int).sum() > 0:  # has at least one Sybil label
        sybil_file = fname
        break

if sybil_file is None:
    print("❌ No Sybil-labeled node found.")
    exit()

print(f"[✓] Found Sybil-labeled node: {sybil_file}")
file_path = os.path.join(data_dir, sybil_file)
X, y, y_true, df_meta = load_and_prepare_data(file_path)
input_size = X.shape[1]

# Load local model
local_model = TorchNodeModel(input_size)
local_model.load_state_dict(torch.load("global_model.pth"))
local_report, local_cm, _ = predict_and_evaluate(local_model, X, y_true)

# Load global model
global_model = TorchNodeModel(input_size)
global_model.load_state_dict(torch.load("global_model_round5.pth"))
global_report, global_cm, _ = predict_and_evaluate(global_model, X, y_true)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

disp1 = ConfusionMatrixDisplay(confusion_matrix=local_cm, display_labels=["Honest", "Sybil"])
disp1.plot(ax=axes[0], values_format="d", cmap="Blues", colorbar=False)
axes[0].set_title(f"Local Model on {sybil_file}")

disp2 = ConfusionMatrixDisplay(confusion_matrix=global_cm, display_labels=["Honest", "Sybil"])
disp2.plot(ax=axes[1], values_format="d", cmap="Blues", colorbar=False)
axes[1].set_title(f"Global Model on {sybil_file}")

plt.tight_layout()
plt.savefig("sybil_node_comparison.png")
print("[✓] Saved confusion matrix: sybil_node_comparison.png")
