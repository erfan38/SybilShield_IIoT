# step4_split_features_by_node.py
# Consolidates per-node multi-sample CSVs into a flat structure in data_per_node/
# do not use this I merged it with step2
import os
import pandas as pd

input_dir = "features_per_node"
output_dir = "data_per_node"
os.makedirs(output_dir, exist_ok=True)

for fname in sorted(os.listdir(input_dir)):
    if fname.endswith(".csv"):
        df = pd.read_csv(os.path.join(input_dir, fname))
        node_id = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{node_id}.csv")

        if os.path.exists(out_path):
            df.to_csv(out_path, mode='a', header=False, index=False)
        else:
            df.to_csv(out_path, index=False)

print(f"[✓] Consolidated data written to {output_dir}/")
