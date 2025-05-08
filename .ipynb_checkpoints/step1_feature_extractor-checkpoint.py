import os, json
import pandas as pd
from collections import defaultdict
import numpy as np

# Load and sort logs
with open("network_logs.jsonl", "r") as f:
    logs = [json.loads(line) for line in f]

logs.sort(key=lambda x: x["timestamp"])

WINDOW_SIZE = 5
node_messages = defaultdict(list)

# Organize messages per node
for log in logs:
    node_messages[log["from"]].append(("sent", log))
    node_messages[log["to"]].append(("received", log))

rows = []

# Generate features per node in time windows
for node_id, messages in node_messages.items():
    messages.sort(key=lambda x: x[1]["timestamp"])
    for i in range(0, len(messages) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = messages[i:i + WINDOW_SIZE]
        sent_total = sent_to_sybil = sent_to_honest = 0
        received_total = received_from_sybil = received_from_honest = 0
        protocols = set()
        timestamps = []

        for direction, msg in window:
            proto = msg["protocol"]
            timestamps.append(msg["timestamp"])
            protocols.add(proto)
            if direction == "sent":
                sent_total += 1
                if msg["is_sybil"]:
                    sent_to_sybil += 1
                else:
                    sent_to_honest += 1
            else:
                received_total += 1
                if msg["is_sybil"]:
                    received_from_sybil += 1
                else:
                    received_from_honest += 1

        burstiness = 0.0
        if len(timestamps) > 1:
            gaps = np.diff(sorted(timestamps))
            if len(gaps) > 1:
                burstiness = np.std(gaps)

        row = {
            "node_id": node_id,
            "sent_total": sent_total,
            "sent_to_sybil": sent_to_sybil,
            "sent_to_honest": sent_to_honest,
            "received_total": received_total,
            "received_from_sybil": received_from_sybil,
            "received_from_honest": received_from_honest,
            "protocol_diversity": len(protocols),
            "message_burstiness": round(burstiness, 5),
            "is_sybil": node_id.startswith("Sybil")
        }
        rows.append(row)

# Output to features_per_node/
os.makedirs("features_per_node", exist_ok=True)
df_all = pd.DataFrame(rows)

for node_id, node_df in df_all.groupby("node_id"):
    node_df.to_csv(f"features_per_node/{node_id}.csv", index=False)

print(f"[✓] Extracted {len(df_all)} samples across {len(df_all['node_id'].unique())} nodes into features_per_node/")