import os
import pandas as pd
from sklearn.utils import resample

INPUT_CSV = "new3_dataset.csv"
OUTPUT_DIR = "nodes3_data"
CLIENT_COUNT = 5  # Adjust number of FL clients

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load dataset and shuffle rows
df = pd.read_csv(INPUT_CSV)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Step 2: Group by node_id (each real node)
grouped = list(df.groupby("node_id"))

# Step 3: Distribute nodes across CLIENT_COUNT clients
clients = [[] for _ in range(CLIENT_COUNT)]
for i, (node_id, group_df) in enumerate(grouped):
    clients[i % CLIENT_COUNT].append(group_df)

# Step 4: Combine node groups into per-client dataframes
for i, client_groups in enumerate(clients):
    client_df = pd.concat(client_groups, ignore_index=True)

    # Step 5: Balance Sybil and Honest samples per client
    sybil_df = client_df[client_df['is_sybil'] == True]
    honest_df = client_df[client_df['is_sybil'] == False]
    min_len = min(len(sybil_df), len(honest_df))

    if min_len < 20:
        print(f" Skipping client_{i}: not enough data to balance")
        continue

    balanced_df = pd.concat([
        resample(sybil_df, replace=False, n_samples=min_len, random_state=1),
        resample(honest_df, replace=False, n_samples=min_len, random_state=1)
    ]).sample(frac=1.0, random_state=1).reset_index(drop=True)

    output_path = os.path.join(OUTPUT_DIR, f"client_{i}.csv")
    balanced_df.to_csv(output_path, index=False)
    print(f"[✓] Saved balanced client_{i}.csv with {len(balanced_df)} samples")

print(" All clients processed and saved.")
for i in range(5):
    df = pd.read_csv(f"nodes3_data/client_{i}.csv")
    print(f"client_{i}: {df['is_sybil'].value_counts().to_dict()}")

