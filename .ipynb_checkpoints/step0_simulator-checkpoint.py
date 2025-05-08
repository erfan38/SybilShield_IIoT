import os
import json
import random
import time

# ---------------------- Configuration ----------------------
PROTOCOLS = [
    "MQTT:PUBLISH", "MQTT:SUBSCRIBE", "MQTT:UNSUBSCRIBE",
    "libp2p:GRAFT", "libp2p:PRUNE", "libp2p:IHAVE", "libp2p:IWANT",
    "devp2p:GetBlockHeaders", "devp2p:Transactions", "devp2p:NewBlock",
    "discovery:Ping", "discovery:Pong", "discovery:FindNode", "discovery:Neighbors",
    "ENR:Request", "ENR:Response"
]

LOG_FILE = "network_logs.jsonl"

# ---------------------- Node Class ----------------------
class Node:
    def __init__(self, node_id, is_sybil=False):
        self.node_id = node_id
        self.is_sybil = is_sybil
        self.behavior_type = random.choice(["spammer", "evasive", "flooder"]) if is_sybil else "normal"

    def send_message(self, network, forced_target=None):
        target = forced_target or network.random_node(exclude=self.node_id)
        if not target:
            return

        if self.is_sybil:
            if self.behavior_type == "spammer":
                protocol = random.choice(["MQTT:PUBLISH", "MQTT:PUBLISH", "MQTT:PUBLISH"])
            elif self.behavior_type == "evasive":
                protocol = random.choice(["ENR:Request", "libp2p:PRUNE"])
            else:  # flooder
                protocol = random.choice(["discovery:Ping", "discovery:FindNode"])
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

# ---------------------- Network Class ----------------------
class Network:
    def __init__(self, honest_count=50, sybil_count=10):
        self.nodes = {}
        self.logs = []
        for i in range(honest_count):
            self.nodes[f"Node{i:03}"] = Node(f"Node{i:03}", is_sybil=False)
        for i in range(sybil_count):
            self.nodes[f"Sybil{i:03}"] = Node(f"Sybil{i:03}", is_sybil=True)

    def random_node(self, exclude=None, honest_only=False, sybil_only=False):
        candidates = [
            n for n in self.nodes.values()
            if n.node_id != exclude and
            (not honest_only or not n.is_sybil) and
            (not sybil_only or n.is_sybil)
        ]
        return random.choice(candidates) if candidates else None

    def simulate_round(self, messages_per_node=10):
        for node in self.nodes.values():
            for _ in range(messages_per_node):
                node.send_message(self)

    def inject_sybil_to_honest(self, count=3):
        honest_nodes = [n for n in self.nodes.values() if not n.is_sybil]
        sybil_nodes = [n for n in self.nodes.values() if n.is_sybil]
        for h in honest_nodes:
            selected_sybils = random.sample(sybil_nodes, k=min(count, len(sybil_nodes)))
            for s in selected_sybils:
                s.send_message(self, forced_target=h)

    def export_logs(self):
        with open(LOG_FILE, "w") as f:
            for m in self.logs:
                f.write(json.dumps(m) + "\n")
        print(f"[✓] Exported {len(self.logs)} messages to {LOG_FILE}")

# ---------------------- Run Simulation ----------------------
if __name__ == "__main__":
    sim = Network()
    for round_num in range(5):
        print(f"[+] Simulating Round {round_num + 1}")
        sim.simulate_round(messages_per_node=10)
        sim.inject_sybil_to_honest(count=2)
        time.sleep(0.05)
    sim.export_logs()
