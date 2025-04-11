import os
import pandas as pd

DATA_DIR = "data_per_node"
SYBIL_PER_HONEST = 5

for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv") or file.startswith("Sybil"):
        continue

    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path)
    honest_node_id = df.iloc[0]["node_id"]
    honest_df = df.copy()

    # Identify honest label
    honest_df["is_sybil"] = False

    # Save honest data
    honest_path = os.path.join(DATA_DIR, f"{honest_node_id}.csv")
    honest_df.to_csv(honest_path, index=False)

    # Create sybil clones
    for i in range(SYBIL_PER_HONEST):
        sybil_id = f"Sybil{str(i).zfill(3)}"
        sybil_df = df.copy()
        sybil_df["node_id"] = sybil_id
        sybil_df["is_sybil"] = True
        sybil_path = os.path.join(DATA_DIR, f"{sybil_id}.csv")
        sybil_df.to_csv(sybil_path, index=False)

    print(f"[✓] Balanced {honest_node_id} — 1 honest + {SYBIL_PER_HONEST} sybil")