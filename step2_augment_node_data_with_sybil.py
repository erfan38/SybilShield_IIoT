import os
import pandas as pd

DATA_DIR = "data_per_node"
SYBIL_PER_HONEST = 5
processed_nodes = set()

for file in sorted(os.listdir(DATA_DIR)):
    if not file.endswith(".csv") or file.startswith("Sybil"):
        continue

    node_id = file.split("_")[0]  # Extract base node ID from Node000_sampleX.csv
    if node_id in processed_nodes:
        continue  # Already processed this node

    # Find all its sample files
    sample_files = [f for f in os.listdir(DATA_DIR) if f.startswith(node_id) and f.endswith(".csv") and "Sybil" not in f]
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in sample_files]
    df = pd.concat(dfs, ignore_index=True)

    df["is_sybil"] = False
    honest_path = os.path.join(DATA_DIR, f"{node_id}.csv")
    df.to_csv(honest_path, index=False)

    for i in range(SYBIL_PER_HONEST):
        sybil_id = f"Sybil{str(i).zfill(3)}"
        sybil_df = df.copy()
        sybil_df["node_id"] = sybil_id
        sybil_df["is_sybil"] = True
        sybil_path = os.path.join(DATA_DIR, f"{sybil_id}.csv")
        sybil_df.to_csv(sybil_path, index=False)

    processed_nodes.add(node_id)
    print(f"[✓] Balanced {node_id} — {len(df)} honest samples + {SYBIL_PER_HONEST} sybils")
