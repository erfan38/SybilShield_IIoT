import os
import time
import json
import random
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
PROTOCOLS = [
    "MQTT:PUBLISH", "MQTT:SUBSCRIBE", "MQTT:UNSUBSCRIBE",
    "libp2p:GRAFT", "libp2p:PRUNE", "libp2p:IHAVE", "libp2p:IWANT",
    "devp2p:GetBlockHeaders", "devp2p:Transactions", "devp2p:NewBlock",
    "discovery:Ping", "discovery:Pong", "discovery:FindNode", "discovery:Neighbors",
    "ENR:Request", "ENR:Response"
]
LOG_FILE = "new4_logs.jsonl"
FINAL_CSV = "new4_dataset.csv"
WINDOW_SIZE = 5

# --- ENR Simulation ---
def simulate_enr(is_sybil):
    ip = f"10.0.0.{random.randint(1, 255)}" if is_sybil else f"192.168.1.{random.randint(1, 255)}"
    port = random.choice([30303, 30304, 30305]) if is_sybil else 30303
    pub_key = ''.join(random.choices(list('abc0123' if is_sybil else 'abcdef0123456789'), k=random.randint(45, 65)))
    protocols = random.sample(["eth", "snap", "les", "custom"], k=random.randint(1, 3))
    return {"ip": ip, "port": port, "public_key": pub_key, "protocols": protocols}

def enr_to_vector(enr):
    ip_parts = [int(p) for p in enr["ip"].split(".")]
    proto_flags = [1 if p in enr["protocols"] else 0 for p in ["eth", "snap", "les", "custom"]]
    return np.array(ip_parts + [enr["port"], len(enr["public_key"])] + proto_flags)

# --- Node Class ---
class Node:
    def __init__(self, node_id, is_sybil=False):
        self.node_id = node_id
        self.is_sybil = is_sybil or random.random() < 0.05
        self.behavior_type = random.choice(["spammer", "evasive", "flooder"] if self.is_sybil else ["normal"])

    def generate_metrics(self):
        latency = random.uniform(5, 50) if self.is_sybil else random.uniform(100, 300)
        energy = random.uniform(30, 70) if self.is_sybil else random.uniform(60, 100)
        enr_similarity = random.uniform(0.9, 1.0) if self.is_sybil else random.uniform(0.0, 0.3)
        return latency, energy, enr_similarity

    def send_message(self, network, forced_target=None):
        target = forced_target or network.random_node(exclude=self.node_id)
        if not target:
            return

        protocol = random.choice(["MQTT:PUBLISH"] * 5 + ["discovery:Ping", "libp2p:IWANT"]) if self.behavior_type == "spammer" else \
                   random.choice(["ENR:Request", "libp2p:PRUNE", "discovery:FindNode"]) if self.behavior_type == "evasive" else \
                   random.choice(PROTOCOLS[:6] if self.is_sybil else PROTOCOLS)

        latency, energy, enr_similarity = self.generate_metrics()
        msg = {
            "timestamp": time.time(),
            "from": self.node_id,
            "to": target.node_id,
            "protocol": protocol,
            "latency_ms": latency,
            "message_size_bytes": random.randint(128, 2048),
            "energy_remaining": energy,
            "enr_similarity": enr_similarity,
            "is_sybil": self.is_sybil
        }
        network.logs.append(msg)

# --- Network Class ---
class Network:
    def __init__(self, honest_count=150, sybil_clones=2):
        self.nodes = {}
        self.logs = []
        self.sybil_ids = set()
        self.enrs = {}

        for i in range(honest_count):
            node_id = f"Node{i:03}"
            self.nodes[node_id] = Node(node_id, is_sybil=False)
            self.enrs[node_id] = simulate_enr(False)
            for j in range(sybil_clones):
                sybil_id = f"{node_id}_Sybil{j:02}"
                self.nodes[sybil_id] = Node(sybil_id, is_sybil=True)
                self.sybil_ids.add(sybil_id)
                self.enrs[sybil_id] = simulate_enr(True)

    def random_node(self, exclude=None):
        return random.choice([n for n in self.nodes.values() if n.node_id != exclude])

    def simulate_round(self, messages_per_node=10):
        for node in self.nodes.values():
            for _ in range(messages_per_node):
                node.send_message(self)

    def export_logs(self):
        with open(LOG_FILE, "w") as f:
            for m in self.logs:
                f.write(json.dumps(m) + "\n")

# --- Main Execution ---
if __name__ == "__main__":
    net = Network(honest_count=50, sybil_clones=2)
    for _ in range(15):
        net.simulate_round(messages_per_node=10)
        time.sleep(0.05)
    net.export_logs()

    with open(LOG_FILE, "r") as f:
        logs = [json.loads(line) for line in f]
    logs.sort(key=lambda x: x["timestamp"])

    node_messages = defaultdict(list)
    for log in logs:
        node_messages[log["from"]].append(("sent", log))
        node_messages[log["to"]].append(("received", log))

    G = nx.DiGraph()
    for log in logs:
        G.add_edge(log["from"], log["to"])

    rows = []
    for node_id, messages in node_messages.items():
        messages.sort(key=lambda x: x[1]["timestamp"])
        for i in range(0, len(messages) - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = messages[i:i + WINDOW_SIZE]
            sent_total = received_total = 0
            protocols, peers, timestamps = [], set(), []
            latencies, msg_sizes, energies = [], [], []
            findnode_count = enr_count = 0

            for direction, msg in window:
                proto = msg["protocol"]
                peer = msg["to"] if direction == "sent" else msg["from"]
                timestamps.append(msg["timestamp"])
                protocols.append(proto)
                peers.add(peer)
                latencies.append(msg["latency_ms"])
                msg_sizes.append(msg["message_size_bytes"])
                energies.append(msg["energy_remaining"])
                if direction == "sent":
                    sent_total += 1
                else:
                    received_total += 1
                if "FindNode" in proto or "Neighbors" in proto:
                    findnode_count += 1
                if "ENR:" in proto:
                    enr_count += 1

            proto_counts = Counter(protocols)
            dominant_ratio = max(proto_counts.values()) / len(protocols) if protocols else 0.0
            mqtt_ratio = sum(1 for p in protocols if p.startswith("MQTT")) / len(protocols)
            discovery_ratio = sum(1 for p in protocols if p.startswith("discovery")) / len(protocols)
            proto_entropy = entropy(list(proto_counts.values()), base=2) if len(proto_counts) > 1 else 0.0
            burstiness = np.std(np.diff(sorted(timestamps))) if len(timestamps) > 2 else 0.0
            timing_entropy = entropy(np.histogram(np.diff(sorted(timestamps)), bins=3)[0] + 1, base=2) if len(timestamps) > 3 else 0.0
            latency_cv = np.std(latencies) / np.mean(latencies) if np.mean(latencies) > 0 else 0.0

            enr_vec = enr_to_vector(net.enrs[node_id])
            peer_vecs = [enr_to_vector(net.enrs[p]) for p in random.sample([n for n in net.enrs if n != node_id], 5)]
            avg_enr_similarity = np.mean([cosine_similarity([enr_vec], [v])[0][0] for v in peer_vecs])
            degree_centrality = len(list(G.successors(node_id))) + len(list(G.predecessors(node_id)))

            row = {
                "node_id": node_id,
                "sent_total": sent_total,
                "received_total": received_total,
                "protocol_diversity": len(set(protocols)),
                "message_burstiness": round(burstiness, 5),
                "dominant_protocol_ratio": round(dominant_ratio, 4),
                "mqtt_ratio": round(mqtt_ratio, 4),
                "discovery_ratio": round(discovery_ratio, 4),
                "protocol_entropy": round(proto_entropy, 4),
                "timing_entropy": round(timing_entropy, 4),
                "latency_cv": round(latency_cv, 4),
                "unique_peers": len(peers),
                "avg_latency": round(np.mean(latencies), 2),
                "avg_msg_size": round(np.mean(msg_sizes), 2),
                "avg_energy": round(np.mean(energies), 2),
                "avg_enr_similarity": round(avg_enr_similarity, 4),
                "findnode_ratio": round(findnode_count / len(window), 4),
                "enr_message_ratio": round(enr_count / len(window), 4),
                "latency_spike_ratio": round(sum(l > 250 for l in latencies) / len(latencies), 4),
                "energy_variance": round(np.var(energies), 4),
                "peer_overlap_ratio": round(len(peers) / WINDOW_SIZE, 4),
                "degree_centrality": degree_centrality,
                "is_sybil": node_id in net.sybil_ids
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(FINAL_CSV, index=False)
    print(f"[\u2713] Final enhanced dataset saved to {FINAL_CSV}")
