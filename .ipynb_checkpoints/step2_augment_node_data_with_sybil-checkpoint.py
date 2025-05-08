import os
import pandas as pd
import numpy as np
import random

FEATURES_DIR = "features_per_node"
OUTPUT_DIR = "data_per_node"
SYBIL_PER_HONEST = 5
PERSONALITIES = ["spammer", "evasive", "flooder"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in sorted(os.listdir(FEATURES_DIR)):
    if not fname.endswith(".csv") or fname.startswith("Sybil"):
        continue

    node_id = fname.replace(".csv", "")
    df = pd.read_csv(os.path.join(FEATURES_DIR, fname))

    df["is_sybil"] = False
    honest_path = os.path.join(OUTPUT_DIR, f"{node_id}.csv")
    df.to_csv(honest_path, index=False)
    print(f"[✓] Saved honest node: {node_id}.csv")

    # Create unique Sybil clones
    for i in range(SYBIL_PER_HONEST):
        sybil_id = f"{node_id}_Sybil{str(i).zfill(2)}"
        sybil_df = df.copy()
        sybil_df["node_id"] = sybil_id
        sybil_df["is_sybil"] = True

        role = random.choice(PERSONALITIES)

        if role == "spammer":
            sybil_df["sent_total"] *= np.random.randint(2, 4)
            sybil_df["protocol_diversity"] = 1
            sybil_df["message_burstiness"] *= 1.2

        elif role == "evasive":
            sybil_df["sent_total"] *= 0.5
            sybil_df["received_total"] *= 1.2
            sybil_df["protocol_diversity"] = np.random.randint(2, 4)
            sybil_df["message_burstiness"] *= np.random.uniform(2.0, 3.5)

        elif role == "flooder":
            sybil_df["sent_to_sybil"] += np.random.randint(5, 10)
            sybil_df["protocol_diversity"] = 1
            sybil_df["message_burstiness"] *= 3
            sybil_df["sent_total"] *= np.random.randint(3, 6)

        sybil_path = os.path.join(OUTPUT_DIR, f"{sybil_id}.csv")
        sybil_df.to_csv(sybil_path, index=False)
        print(f"[✓] Created Sybil clone: {sybil_id} as {role}")
