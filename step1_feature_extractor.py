# Modified step1_feature_extractor.py
# Creates multiple time-windowed samples per node for diversity

import json
import pandas as pd
from collections import defaultdict
import numpy as np

# Load logs
with open("network_logs.jsonl", "r") as f:
    logs = [json.loads(line) for line in f]

# Sort logs by time
logs.sort(key=lambda x: x["timestamp"])

# Parameters
WINDOW_SIZE = 20  # messages per window

# Organize messages per node
node_messages = defaultdict(list)
for log in logs:
    node_messages[log["from"]].append(("sent", log))
    node_messages[log["to"]].append(("received", log))

rows = []

for node_id, messages in node_messages.items():
    # Sort node's messages by timestamp
    messages.sort(key=lambda x: x[1]["timestamp"])

    for i in range(0, len(messages) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = messages[i:i+WINDOW_SIZE]
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

        # Label: true if node is sybil
        is_sybil = node_id.startswith("Sybil")

        # Burstiness
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
            "is_sybil": is_sybil
        }
        rows.append(row)

# Output to CSV
out_df = pd.DataFrame(rows)
out_df.to_csv("node_features.csv", index=False)
print(f"[\u2713] Extracted {len(out_df)} samples to node_features.csv")
