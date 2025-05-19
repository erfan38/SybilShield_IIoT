# import os
# import pandas as pd
# from sklearn.utils import resample

# INPUT_CSV = "dataset_25-05-17-18-17-48.csv"
# OUTPUT_DIR = "nodes_finaal_data"
# CLIENT_COUNT = 5  # Adjust number of FL clients

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Step 1: Load dataset and shuffle rows
# df = pd.read_csv(INPUT_CSV)
# df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# # Step 2: Group by node_id (each real node)
# grouped = list(df.groupby("node_id"))

# # Step 3: Distribute nodes across CLIENT_COUNT clients
# clients = [[] for _ in range(CLIENT_COUNT)]
# for i, (node_id, group_df) in enumerate(grouped):
#     clients[i % CLIENT_COUNT].append(group_df)

# # Step 4: Combine node groups into per-client dataframes
# for i, client_groups in enumerate(clients):
#     client_df = pd.concat(client_groups, ignore_index=True)

#     # Step 5: Balance Sybil and Honest samples per client
#     sybil_df = client_df[client_df['is_sybil'] == True]
#     honest_df = client_df[client_df['is_sybil'] == False]
#     min_len = min(len(sybil_df), len(honest_df))

#     if min_len < 20:
#         print(f" Skipping client_{i}: not enough data to balance")
#         continue

#     balanced_df = pd.concat([
#         resample(sybil_df, replace=False, n_samples=min_len, random_state=1),
#         resample(honest_df, replace=False, n_samples=min_len, random_state=1)
#     ]).sample(frac=1.0, random_state=1).reset_index(drop=True)

#     output_path = os.path.join(OUTPUT_DIR, f"client_{i}.csv")
#     balanced_df.to_csv(output_path, index=False)
#     print(f"[✓] Saved balanced client_{i}.csv with {len(balanced_df)} samples")

# print(" All clients processed and saved.")
# for i in range(5):
#     df = pd.read_csv(f"nodes_final_data/client_{i}.csv")
#     print(f"client_{i}: {df['is_sybil'].value_counts().to_dict()}")





import os
import pandas as pd
from sklearn.utils import resample

INPUT_CSV = "dataset_25-05-17-18-17-48.csv"
OUTPUT_DIR = "nodes_final_data"  # corrected typo from "finaal" to "final"
CLIENT_COUNT = 5  # Number of federated clients

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load and shuffle
df = pd.read_csv(INPUT_CSV)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Step 2: Group by node_id
grouped = list(df.groupby("node_id"))

# Step 3: Distribute nodes across CLIENT_COUNT clients
clients = [[] for _ in range(CLIENT_COUNT)]
for i, (node_id, group_df) in enumerate(grouped):
    clients[i % CLIENT_COUNT].append(group_df)

# Step 4: Prepare client datasets
for i, client_groups in enumerate(clients):
    client_df = pd.concat(client_groups, ignore_index=True)

    # Step 5: Balance across severity classes
    class_counts = client_df["severity"].value_counts().to_dict()
    if len(class_counts) < 3:
        print(f"[!] Skipping client_{i}: missing classes in {class_counts}")
        continue

    min_class_size = min(class_counts.values())
    if min_class_size < 10:
        print(f"[!] Skipping client_{i}: not enough samples to balance ({min_class_size})")
        continue

    balanced_df = pd.concat([
        resample(client_df[client_df["severity"] == "honest"], n_samples=min_class_size, random_state=1, replace=False),
        resample(client_df[client_df["severity"] == "low"],    n_samples=min_class_size, random_state=1, replace=False),
        resample(client_df[client_df["severity"] == "high"],   n_samples=min_class_size, random_state=1, replace=False)
    ]).sample(frac=1.0, random_state=1).reset_index(drop=True)

    output_path = os.path.join(OUTPUT_DIR, f"client_{i}.csv")
    balanced_df.to_csv(output_path, index=False)
    print(f"[✓] Saved client_{i}.csv with {len(balanced_df)} balanced samples")

# Step 6: Show class distribution summary
print("\n[Summary of saved clients]")
for i in range(CLIENT_COUNT):
    path = os.path.join(OUTPUT_DIR, f"client_{i}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"client_{i}: {df['severity'].value_counts().to_dict()}")
