# step2_visualize_network.py
# Visualize network graph based on reputation and protocol types
# + Export subgraphs over time by protocol category

import json
import networkx as nx
import matplotlib.pyplot as plt
import os
from collections import defaultdict

with open("network_logs.jsonl") as f:
    logs = [json.loads(l) for l in f]

with open("node_reputations.json") as f:
    reputations = json.load(f)

# Create base graph
G = nx.DiGraph()
for node_id, rep_data in reputations.items():
    G.add_node(node_id,
               reputation=rep_data["reputation"],
               color="red" if "Sybil" in node_id else "green")

# Add all edges to the base graph
for log in logs:
    proto = log["protocol"]
    G.add_edge(log["from"], log["to"], protocol=proto, timestamp=log["timestamp"])

# Draw full network
pos = nx.spring_layout(G, seed=42)
colors = [G.nodes[n]['color'] for n in G.nodes()]
sizes = [max(10, G.nodes[n]['reputation'] * 100) for n in G.nodes()]

# Edge coloring
edge_colors = []
for u, v, d in G.edges(data=True):
    proto = d["protocol"]
    if "MQTT" in proto:
        edge_colors.append("blue")
    elif "devp2p" in proto:
        edge_colors.append("purple")
    elif "discovery" in proto:
        edge_colors.append("orange")
    elif "ENR" in proto or "discv5" in proto:
        edge_colors.append("brown")
    else:
        edge_colors.append("gray")

nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.7)
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=6)
plt.title("PoA Network Communication Graph with Protocol Coloring")
plt.axis('off')
plt.tight_layout()
plt.savefig("network_graph.png")
plt.clf()
print("[✓] Full network visualization saved as network_graph.png")

# Export protocol-specific subgraphs over time windows
output_dir = "protocol_subgraphs"
os.makedirs(output_dir, exist_ok=True)

protocol_categories = ["MQTT", "devp2p", "discovery", "ENR", "discv5"]
time_intervals = [0, 1, 2, 3, 4]  # 5 intervals
min_time = min(log["timestamp"] for log in logs)
max_time = max(log["timestamp"] for log in logs)
delta = (max_time - min_time) / len(time_intervals)

for proto in protocol_categories:
    for i in range(len(time_intervals)):
        t_start = min_time + i * delta
        t_end = t_start + delta

        subG = nx.DiGraph()
        for node_id in G.nodes:
            subG.add_node(node_id, **G.nodes[node_id])

        for u, v, d in G.edges(data=True):
            if proto in d["protocol"] and t_start <= d["timestamp"] < t_end:
                subG.add_edge(u, v, **d)

        if subG.number_of_edges() == 0:
            continue

        nx.draw(subG, pos, node_color=colors, node_size=sizes, edge_color="gray", alpha=0.6, with_labels=False)
        plt.title(f"{proto} Subgraph Round {i+1}")
        fname = f"{output_dir}/{proto}_round{i+1}.png"
        plt.savefig(fname)
        plt.clf()
        print(f"[✓] Saved {fname}")
