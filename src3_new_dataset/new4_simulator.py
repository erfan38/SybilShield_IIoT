# Updated `new_simulator.py` — 5-class severity label generator
import os
import time
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

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


def simulate_enr(severity):
    ip = f"10.0.0.{random.randint(1, 255)}" if severity > 0 else f"192.168.1.{random.randint(1, 255)}"
    port = random.choice([30303, 30304, 30305]) if severity > 0 else 30303
    pub_key = ''.join(random.choices(list('abc0123' if severity > 0 else 'abcdef0123456789'), k=random.randint(45, 65)))
    protocols = random.sample(["eth", "snap", "les", "custom"], k=random.randint(1, 3))
    return {"ip": ip, "port": port, "public_key": pub_key, "protocols": protocols}


def enr_to_vector(enr):
    ip_parts = [int(p) for p in enr["ip"].split(".")]
    proto_flags = [1 if p in enr["protocols"] else 0 for p in ["eth", "snap", "les", "custom"]]
    return np.array(ip_parts + [enr["port"], len(enr["public_key"])] + proto_flags)


class Node:
    def __init__(self, node_id, severity=0):
        self.node_id = node_id
        self.severity = severity
        self.behavior_type = random.choice(
            ["normal"] if severity == 0 else
            ["spammer", "evasive", "flooder"]
        )

    def generate_metrics(self):
        if self.severity == 0:
            latency = random.uniform(100, 300)
            energy = random.uniform(60, 100)
            enr_similarity = random.uniform(0.0, 0.3)
        else:
            base = self.severity / 10
            latency = random.uniform(5, 50) * (1 + 0.1 * (10 - self.severity))
            energy = random.uniform(30, 70)
            enr_similarity = random.uniform(0.7 + 0.03 * base, 1.0)
        return latency, energy, enr_similarity

    def send_message(self, network):
        target = network.random_node(exclude=self.node_id)
        if not target:
            return

        protocol = random.choice(
            ["MQTT:PUBLISH"] * 5 + ["discovery:Ping", "libp2p:IWANT"]
        ) if self.behavior_type == "spammer" else \
        random.choice(["ENR:Request", "libp2p:PRUNE", "discovery:FindNode"]) if self.behavior_type == "evasive" else \
        random.choice(PROTOCOLS[:6]) if self.behavior_type == "flooder" else \
        random.choice(PROTOCOLS)

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
            "severity": self.severity
        }
        network.logs.append(msg)


class Network:
    def __init__(self, honest_count=50, low=30, moderate=30, high=30, sybil=30):
        self.nodes = {}
        self.logs = []
        self.enrs = {}

        for i in range(honest_count):
            node_id = f"Node{i:03}"
            self.nodes[node_id] = Node(node_id, severity=0)
            self.enrs[node_id] = simulate_enr(0)

        def add_severity_group(start_idx, count, sev_level):
            for i in range(count):
                node_id = f"Node{start_idx + i:03}_S{sev_level}"
                self.nodes[node_id] = Node(node_id, severity=sev_level)
                self.enrs[node_id] = simulate_enr(sev_level)

        base_idx = honest_count
        for sev, count in zip([3, 5, 8, 10], [low, moderate, high, sybil]):
            add_severity_group(base_idx, count, sev)
            base_idx += count

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


if __name__ == "__main__":
    net = Network()
    for _ in range(15):
        net.simulate_round(messages_per_node=10)
        time.sleep(0.01)
    net.export_logs()

    with open(LOG_FILE, "r") as f:
        logs = [json.loads(line) for line in f]
    logs.sort(key=lambda x: x["timestamp"])

    node_messages = defaultdict(list)
    for log in logs:
        node_messages[log["from"]].append(("sent", log))
        node_messages[log["to"]].append(("received", log))

    rows = []
    for node_id, messages in node_messages.items():
        messages.sort(key=lambda x: x[1]["timestamp"])
        for i in range(0, len(messages) - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = messages[i:i + WINDOW_SIZE]
            sent_total = received_total = 0
            protocols = []
            peers = set()
            timestamps = []
            latencies, msg_sizes, energies = [], [], []

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

            proto_counts = Counter(protocols)
            dominant_ratio = max(proto_counts.values()) / len(protocols) if protocols else 0.0
            mqtt_ratio = sum(1 for p in protocols if p.startswith("MQTT")) / len(protocols)
            discovery_ratio = sum(1 for p in protocols if p.startswith("discovery")) / len(protocols)
            proto_entropy = entropy(list(proto_counts.values()), base=2) if len(proto_counts) > 1 else 0.0

            burstiness = np.std(np.diff(sorted(timestamps))) if len(timestamps) > 2 else 0.0
            enr_vec = enr_to_vector(net.enrs[node_id])
            peer_vecs = [enr_to_vector(net.enrs[p]) for p in random.sample([n for n in net.enrs if n != node_id], 5)]
            avg_enr_similarity = np.mean([cosine_similarity([enr_vec], [v])[0][0] for v in peer_vecs])

            latency_spike_ratio = sum(l > 250 for l in latencies) / len(latencies)
            energy_var = np.var(energies)
            peer_overlap_ratio = len(peers) / WINDOW_SIZE

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
                "unique_peers": len(peers),
                "avg_latency": round(np.mean(latencies), 2),
                "avg_msg_size": round(np.mean(msg_sizes), 2),
                "avg_energy": round(np.mean(energies), 2),
                "avg_enr_similarity": round(avg_enr_similarity, 4),
                "latency_spike_ratio": round(latency_spike_ratio, 4),
                "energy_variance": round(energy_var, 4),
                "peer_overlap_ratio": round(peer_overlap_ratio, 4),
                "severity": window[0][1]["severity"]
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(FINAL_CSV, index=False)
    print(f"[✓] Dataset with 5-class severity labels saved to {FINAL_CSV}")