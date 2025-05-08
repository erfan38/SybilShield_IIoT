import torch
import torch.nn as nn
import pandas as pd
import os

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

def train_node_model(path):
    df = pd.read_csv(path)
    label_col = "label" if "label" in df.columns else "is_sybil"
    X = torch.tensor(df.drop(columns=[label_col, "node_id"]).values, dtype=torch.float32)
    y = torch.tensor(df[label_col].astype(float).values, dtype=torch.float32).unsqueeze(1)

    model = TorchNodeModel(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    return model.state_dict()

# Pick a valid node file to test
test_file = "data_per_node/Node000.csv"
model_weights = train_node_model(test_file)
torch.save(model_weights, "global_model.pth")
print(f"[✓] Trained and saved model for {test_file}")
