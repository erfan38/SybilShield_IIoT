
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

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

def evaluate_model(model, df):
    label_col = "label" if "label" in df.columns else "is_sybil"
    X = torch.tensor(df.drop(columns=[label_col, "node_id"]).values, dtype=torch.float32)
    y = torch.tensor(df[label_col].astype(float).values, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        pred = model(X)
        pred_binary = (pred > 0.5).float()
        accuracy = (pred_binary == y).float().mean().item()
    return accuracy

def train_node_model(df, input_size):
    label_col = "label" if "label" in df.columns else "is_sybil"
    X = torch.tensor(df.drop(columns=[label_col, "node_id"]).values, dtype=torch.float32)
    y = torch.tensor(df[label_col].astype(float).values, dtype=torch.float32).unsqueeze(1)
    model = TorchNodeModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    for epoch in range(10):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    return model.state_dict()

def average_models(model_dicts):
    avg_model = {}
    for key in model_dicts[0].keys():
        avg_model[key] = sum(model[key] for model in model_dicts) / len(model_dicts)
    return avg_model

def load_node_data(data_dir):
    data = []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, file))
            data.append((file, df))
    return data

def run_federated_rounds(data_dir="data_per_node", rounds=5):
    node_data = load_node_data(data_dir)
    label_col = "label" if "label" in node_data[0][1].columns else "is_sybil"
    input_size = node_data[0][1].drop(columns=[label_col, "node_id"]).shape[1]

    global_model = TorchNodeModel(input_size)
    accuracy_log = []

    for rnd in range(rounds):
        local_models = []
        round_accuracies = []

        for fname, df in node_data:
            model_weights = train_node_model(df, input_size)
            local_models.append(model_weights)

            global_model.load_state_dict(model_weights)
            acc = evaluate_model(global_model, df)
            round_accuracies.append(acc)

        avg_model = average_models(local_models)
        global_model.load_state_dict(avg_model)
        torch.save(global_model.state_dict(), f"global_model_round{rnd+1}.pth")

        avg_acc = sum(round_accuracies) / len(round_accuracies)
        accuracy_log.append(avg_acc)
        print(f"[Round {rnd+1}] Avg Accuracy: {avg_acc:.4f}")

    plt.plot(range(1, rounds + 1), accuracy_log, marker='o')
    plt.xlabel("Round")
    plt.ylabel("Average Accuracy")
    plt.title("Federated Learning Accuracy over Rounds")
    plt.grid(True)
    plt.savefig("fl_accuracy_trend.png")
    print("Saved accuracy trend plot as fl_accuracy_trend.png")
    # === Save node reputations based on global model ===
    reputations = {}
    for fname, df in node_data:
        acc = evaluate_model(global_model, df)
        reputations[fname] = {
            "accuracy": round(acc, 4),
            "reputation": round(acc * 10 - 5, 3)  # scale to [-5, +5] range
        }

    import json
    with open("node_reputations.json", "w") as f:
        json.dump(reputations, f, indent=2)
    print("[✓] Exported node_reputations.json")

if __name__ == "__main__":
    run_federated_rounds()
