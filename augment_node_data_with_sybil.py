import os
import pandas as pd
import random

DATA_DIR = "data_per_node"
TARGET_PER_CLASS = 25

# Collect all Sybil samples
sybil_samples = []
for fname in os.listdir(DATA_DIR):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    if df.iloc[0]["node_id"].startswith("Sybil"):
        sybil_samples.append(df[df["is_sybil"] == 1])

all_sybil_df = pd.concat(sybil_samples, ignore_index=True)

for fname in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)
    node_id = df.iloc[0]["node_id"]

    if node_id.startswith("Sybil"):
        continue

    honest_df = df[df["is_sybil"] == 0]
    if len(honest_df) < 2 or len(all_sybil_df) < 2:
        print(f"[!] Skipping {node_id} — not enough honest or sybil samples")
        continue

    n = min(len(honest_df), TARGET_PER_CLASS, len(all_sybil_df))
    honest_sample = honest_df.sample(n=n, random_state=42)
    sybil_sample = all_sybil_df.sample(n=n, random_state=42)

    combined = pd.concat([honest_sample, sybil_sample], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42)
    combined.to_csv(path, index=False)

    print(f"[✓] Balanced {node_id} — {n} honest + {n} sybil")
