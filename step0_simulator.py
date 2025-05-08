import os
import random
import json
import time
import pandas as pd
import numpy as np

PROTOCOLS = [
    "MQTT:PUBLISH", "MQTT:SUBSCRIBE", "MQTT:UNSUBSCRIBE",
    "libp2p:GRAFT", "libp2p:PRUNE", "libp2p:IHAVE", "libp2p:IWANT",
    "devp2p:GetBlockHeaders", "devp2p:Transactions", "devp2p:NewBlock",
    "discovery:Ping", "discovery:Pong", "discovery:FindNode", "discovery:Neighbors",
    "ENR:Request", "ENR:Response"
]

class Node:
    def __init__(self, node_id, is_sybil=False):
        self.node_id = node_id
        self.is_sybil = is_sybil

    def send_message(self, network, forced_target=None):
        target = forced_target or network.random_node(exclude=self.node_id)
        if not target:
            return
        if self.is_sybil:
            behavior_type = int(self.node_id[-1]) % 3
            if behavior_type == 0:
                protocol = random.choice(["MQTT:PUBLISH", "MQTT:SUBSCRIBE", "MQTT:PUBLISH"])
            elif behavior_type == 1:
                protocol = random.choice(["discovery:Ping", "discovery:FindNode", "discovery:Neighbors"])
            else:
                protocol = random.choice(["ENR:Request", "ENR:Response", "libp2p:PRUNE"])
        else:
            protocol = random.choice(PROTOCOLS)
        msg = {
            "from": self.node_id,
            "to": target.node_id,
            "protocol": protocol,
            "timestamp": time.time(),
            "is_sybil": self.is_sybil
        }
        network.logs.append(msg)

class Network:
    def __init__(self, honest_count=50, sybil_count=10):
        self.nodes = {}
        self.logs = []
        for i in range(honest_count):
            self.nodes[f"Node{i:03}"] = Node(f"Node{i:03}", is_sybil=False)
        for i in range(sybil_count):
            self.nodes[f"Sybil{i:03}"] = Node(f"Sybil{i:03}", is_sybil=True)

    def random_node(self, exclude=None, honest_only=False, sybil_only=False):
        choices = [n for n in self.nodes.values() if n.node_id != exclude]
        if honest_only:
            choices = [n for n in choices if not n.is_sybil]
        if sybil_only:
            choices = [n for n in choices if n.is_sybil]
        return random.choice(choices) if choices else None

    def simulate_round(self, count=100):
        all_nodes = list(self.nodes.values())
        for _ in range(count):
            sender = random.choice(all_nodes)
            receiver = self.random_node(exclude=sender.node_id, honest_only=not sender.is_sybil)
            sender.send_message(self, forced_target=receiver)

    def inject_sybil_to_honest(self, messages_per_node=3):
        honest = [n for n in self.nodes.values() if not n.is_sybil]
        sybils = [n for n in self.nodes.values() if n.is_sybil]
        for h in honest:
            for s in random.sample(sybils, k=min(messages_per_node, len(sybils))):
                s.send_message(self, forced_target=h)

    def export_logs(self, out_path="network_logs.jsonl"):
        with open(out_path, "w") as f:
            for m in self.logs:
                f.write(json.dumps(m) + "\n")

    def export_multi_samples(self, out_dir="data_per_node", samples_per_node=5):
        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame(self.logs)

        for node_id in df["from"].unique():
            node_df = df[df["from"] == node_id]
            recv_df = df[df["to"] == node_id]
            for i in range(samples_per_node):
                sent_sample = node_df.sample(frac=1.0, replace=True)
                total_sent = len(sent_sample)
                sent_sybil = len(sent_sample[sent_sample["to"].str.startswith("Sybil")])
                sent_honest = total_sent - sent_sybil

                recv_sample = recv_df.sample(frac=1.0, replace=True)
                total_recv = len(recv_sample)
                recv_sybil = len(recv_sample[recv_sample["from"].str.startswith("Sybil")])
                recv_honest = total_recv - recv_sybil

                features = {
                    "node_id": node_id,
                    "sent_total": total_sent,
                    "sent_to_sybil": sent_sybil,
                    "sent_to_honest": sent_honest,
                    "received_total": total_recv,
                    "received_from_sybil": recv_sybil,
                    "received_from_honest": recv_honest,
                    "protocol_diversity": len(set(sent_sample["protocol"])),
                    "message_burstiness": round(np.std([total_sent, total_recv]), 4),
                    "is_sybil": node_id.startswith("Sybil")
                }

                df_out = pd.DataFrame([features])
                df_out.to_csv(os.path.join(out_dir, f"{node_id}_sample{i}.csv"), index=False)

# Run simulation
if __name__ == "__main__":
    sim = Network()
    for r in range(5):
        print(f"[+] Round {r+1}")
        sim.simulate_round(100)
        sim.inject_sybil_to_honest()
    sim.export_logs("network_logs.jsonl")
    sim.export_multi_samples("data_per_node", samples_per_node=5)
    print("[✓] Done. Multi-sample CSVs exported.")
