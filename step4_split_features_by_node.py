# step4_split_by_node.py
# Split node_features.csv into one local dataset per node for FL

import pandas as pd
import os

# Load full feature dataset
df = pd.read_csv("node_features.csv")

# Output directory
os.makedirs("data_per_node", exist_ok=True)

# Save one CSV file per node
for _, row in df.iterrows():
    node_id = row["node_id"]
    out_path = os.path.join("data_per_node", f"{node_id}.csv")
    row.to_frame().T.to_csv(out_path, index=False)

print(f"[✓] Saved {len(df)} local node files in data_per_node/")
