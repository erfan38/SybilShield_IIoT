import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FEATURE_COLUMNS = [
    "sent_total", "received_total", "protocol_diversity",
    "message_burstiness", "mqtt_ratio", "discovery_ratio",
    "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
    "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity",
    "latency_spike_ratio", "energy_variance", "peer_overlap_ratio"
]

input_dir = "nodes3_data"
output_dir = "nodes3_data_pca"
os.makedirs(output_dir, exist_ok=True)

# Fit PCA on full data
all_dfs = []
for fname in os.listdir(input_dir):
    if fname.endswith(".csv"):
        df = pd.read_csv(os.path.join(input_dir, fname))
        all_dfs.append(df)

full_df = pd.concat(all_dfs, ignore_index=True)
scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(full_df[FEATURE_COLUMNS])

pca = PCA(n_components=0.95)
pca.fit(X_scaled_full)

# Apply PCA to each client file
for fname in os.listdir(input_dir):
    if fname.endswith(".csv"):
        df = pd.read_csv(os.path.join(input_dir, fname))
        X = df[FEATURE_COLUMNS]
        y = df["is_sybil"]

        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

        pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        result_df = pd.DataFrame(X_pca, columns=pca_columns)
        result_df["is_sybil"] = y

        result_df.to_csv(os.path.join(output_dir, fname), index=False)

print("✅ PCA transformation completed and saved to 'nodes3_data_pca/'")
