import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# === Load Dataset ===
df = pd.read_csv("new3_dataset.csv")  # Update path if needed

# === Feature Columns ===
FEATURE_COLUMNS = [
    # 'sent_total', 'received_total'
    # , 'protocol_diversity', 'message_burstiness',
    # 'dominant_protocol_ratio', 'mqtt_ratio', 'discovery_ratio', 'protocol_entropy',
    # 'unique_peers', 'avg_latency', 'avg_msg_size', 'avg_energy', 'avg_enr_similarity_realistic'
    "sent_total", "received_total", "protocol_diversity",
    "message_burstiness", "mqtt_ratio", "discovery_ratio",
    "protocol_entropy", "unique_peers", "dominant_protocol_ratio",
    "avg_latency", "avg_msg_size", "avg_energy", "avg_enr_similarity",
    "latency_spike_ratio", "energy_variance", "peer_overlap_ratio"
]

X = df[FEATURE_COLUMNS]
y = df["is_sybil"]

# === Normalize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
eigenvalues = pca.explained_variance_

print("PCA Eigenvalues:")
for i, val in enumerate(eigenvalues, 1):
    print(f"PC{i}: {val:.4f}")

# === Plot PCA ===
plt.figure(figsize=(7, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.6, s=5)
plt.title("PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.show()

# === t-SNE === (on a sample of 3500 for speed)
df_sample = df.sample(n=3500, random_state=42)
X_sample = df_sample[FEATURE_COLUMNS]
y_sample = df_sample["is_sybil"]

X_sample_scaled = scaler.fit_transform(X_sample)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto', n_iter=300)
X_tsne = tsne.fit_transform(X_sample_scaled)

# === Plot t-SNE ===
plt.figure(figsize=(7, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap="coolwarm", alpha=0.6, s=5)
plt.title("t-SNE Projection (Sampled 3500)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.tight_layout()
plt.show()
