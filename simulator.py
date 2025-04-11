# # simulator.py
# # Simulates PoA Ethereum IIoT network with Sybil and honest nodes, ensuring FL works

# # Logs raw message-level data to allow per-node sample aggregation later

# import random
# import time
# import json
# from hashlib import sha256

# PROTOCOLS = [
#     "MQTT:PUBLISH", "MQTT:SUBSCRIBE", "MQTT:UNSUBSCRIBE",
#     "libp2p:GRAFT", "libp2p:PRUNE", "libp2p:IHAVE", "libp2p:IWANT",
#     "devp2p:GetBlockHeaders", "devp2p:Transactions", "devp2p:NewBlock",
#     "discovery:Ping", "discovery:Pong", "discovery:FindNode", "discovery:Neighbors",
#     "ENR:Request", "ENR:Response"
# ]

# class Node:
#     def __init__(self, node_id, is_sybil=False):
#         self.node_id = node_id
#         self.is_sybil = is_sybil

#     # def send_message(self, network, forced_target=None):
#     #     target = forced_target or network.random_node(exclude=self.node_id)
#     #     protocol = random.choice(PROTOCOLS)

#     #     msg = {
#     #         "from": self.node_id,
#     #         "to": target.node_id,
#     #         "protocol": protocol,
#     #         "timestamp": time.time(),
#     #         "is_sybil": self.is_sybil
#     #     }
#     #     network.log_message(msg)

#     #### make various behavior type for sybil:
    
# def send_message(self, network, forced_target=None):
#     target = forced_target or network.random_node(exclude=self.node_id)

#     # Different Sybil behaviors
#     if self.is_sybil:
#         behavior_type = int(self.node_id[-1]) % 3  # simple hash to diversify Sybils

#         if behavior_type == 0:
#             # Spam MQTT protocols only
#             protocol = random.choice(["MQTT:PUBLISH", "MQTT:SUBSCRIBE", "MQTT:PUBLISH", "MQTT:PUBLISH"])
#         elif behavior_type == 1:
#             # Spam discovery protocols with delay
#             time.sleep(0.01)
#             protocol = random.choice(["discovery:Ping", "discovery:FindNode", "discovery:Neighbors"])
#         else:
#             # Use uncommon protocols to hide
#             protocol = random.choice(["ENR:Request", "ENR:Response", "libp2p:PRUNE"])

#     else:
#         # Honest nodes use all protocols more evenly
#         protocol = random.choice(PROTOCOLS)

#     msg = {
#         "from": self.node_id,
#         "to": target.node_id,
#         "protocol": protocol,
#         "timestamp": time.time(),
#         "is_sybil": self.is_sybil
#     }
#     network.log_message(msg)


# class Network:
#     def __init__(self, honest_count=50, sybil_count=10):
#         self.nodes = {}
#         self.logs = []
#         for i in range(honest_count):
#             node = Node(f"Node{i:03}", is_sybil=False)
#             self.nodes[node.node_id] = node
#         for i in range(sybil_count):
#             node = Node(f"Sybil{i:03}", is_sybil=True)
#             self.nodes[node.node_id] = node

#     def random_node(self, exclude=None, honest_only=False, sybil_only=False):
#         choices = [n for n in self.nodes.values() if n.node_id != exclude]
#         if honest_only:
#             choices = [n for n in choices if not n.is_sybil]
#         if sybil_only:
#             choices = [n for n in choices if n.is_sybil]
#         return random.choice(choices) if choices else None

#     def simulate_round(self, count=100):
#         all_nodes = list(self.nodes.values())
#         for _ in range(count):
#             sender = random.choice(all_nodes)
#             if not sender.is_sybil and random.random() < 0.5:
#                 receiver = self.random_node(exclude=sender.node_id, honest_only=True)
#             else:
#                 receiver = self.random_node(exclude=sender.node_id)
#             if receiver:
#                 sender.send_message(self, forced_target=receiver)

#     def inject_sybil_to_honest(self, messages_per_node=3):
#         honest_nodes = [n for n in self.nodes.values() if not n.is_sybil]
#         sybil_nodes = [n for n in self.nodes.values() if n.is_sybil]
#         for honest in honest_nodes:
#         # Use random number of injections per node
#             count = random.randint(1, messages_per_node + 2)
#             selected_sybil = random.sample(sybil_nodes, k=min(count, len(sybil_nodes)))

#         for sybil in selected_sybil:
#             sybil.send_message(self, forced_target=honest)
#         # for honest in honest_nodes:
#         #     for _ in range(messages_per_node):
#         #         sybil = random.choice(sybil_nodes)
#         #         sybil.send_message(self, forced_target=honest)

#     def log_message(self, msg):
#         self.logs.append(msg)

#     def export_logs(self, filename="network_logs.jsonl"):
#         with open(filename, "w") as f:
#             for log in self.logs:
#                 f.write(json.dumps(log) + "\n")
#         print(f"[\u2713] Exported {len(self.logs)} logs to {filename}")

# if __name__ == "__main__":
#     sim = Network(honest_count=50, sybil_count=10)
#     for r in range(5):
#         print(f"[+] Simulating Round {r+1}")
#         sim.simulate_round(100)
#         sim.inject_sybil_to_honest(messages_per_node=2)
#         time.sleep(0.1)
#     sim.export_logs()

####################3 Final version ################
# simulator.py
# Simulates PoA Ethereum IIoT network with Sybil and honest nodes, ensuring FL works
# Logs raw message-level data to allow per-node sample aggregation later

import random
import time
import json
from hashlib import sha256

# Communication protocols used in IIoT and blockchain networks
PROTOCOLS = [
    "MQTT:PUBLISH", "MQTT:SUBSCRIBE", "MQTT:UNSUBSCRIBE",
    "libp2p:GRAFT", "libp2p:PRUNE", "libp2p:IHAVE", "libp2p:IWANT",
    "devp2p:GetBlockHeaders", "devp2p:Transactions", "devp2p:NewBlock",
    "discovery:Ping", "discovery:Pong", "discovery:FindNode", "discovery:Neighbors",
    "ENR:Request", "ENR:Response"
]

# ---------------------- Node Definition ----------------------
class Node:
    def __init__(self, node_id, is_sybil=False):
        self.node_id = node_id
        self.is_sybil = is_sybil

    def send_message(self, network, forced_target=None):
        target = forced_target or network.random_node(exclude=self.node_id)

        if self.is_sybil:
            behavior_type = int(self.node_id[-1]) % 3

            if behavior_type == 0:
                # Spam MQTT protocols only
                protocol = random.choice(["MQTT:PUBLISH", "MQTT:SUBSCRIBE", "MQTT:PUBLISH", "MQTT:PUBLISH"])
            elif behavior_type == 1:
                # Spam discovery protocols with delay
                time.sleep(0.01)
                protocol = random.choice(["discovery:Ping", "discovery:FindNode", "discovery:Neighbors"])
            else:
                # Use uncommon protocols to hide
                protocol = random.choice(["ENR:Request", "ENR:Response", "libp2p:PRUNE"])
        else:
            # Honest nodes use all protocols more evenly
            protocol = random.choice(PROTOCOLS)

        msg = {
            "from": self.node_id,
            "to": target.node_id,
            "protocol": protocol,
            "timestamp": time.time(),
            "is_sybil": self.is_sybil
        }
        network.log_message(msg)

# ---------------------- Network Simulator ----------------------
class Network:
    def __init__(self, honest_count=50, sybil_count=10):
        self.nodes = {}
        self.logs = []

        for i in range(honest_count):
            node = Node(f"Node{i:03}", is_sybil=False)
            self.nodes[node.node_id] = node

        for i in range(sybil_count):
            node = Node(f"Sybil{i:03}", is_sybil=True)
            self.nodes[node.node_id] = node

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
            if not sender.is_sybil and random.random() < 0.5:
                receiver = self.random_node(exclude=sender.node_id, honest_only=True)
            else:
                receiver = self.random_node(exclude=sender.node_id)
            if receiver:
                sender.send_message(self, forced_target=receiver)

    def inject_sybil_to_honest(self, messages_per_node=3):
        honest_nodes = [n for n in self.nodes.values() if not n.is_sybil]
        sybil_nodes = [n for n in self.nodes.values() if n.is_sybil]

        for honest in honest_nodes:
            count = random.randint(1, messages_per_node + 2)
            selected_sybil = random.sample(sybil_nodes, k=min(count, len(sybil_nodes)))

            for sybil in selected_sybil:
                sybil.send_message(self, forced_target=honest)

    def log_message(self, msg):
        self.logs.append(msg)

    def export_logs(self, filename="network_logs.jsonl"):
        with open(filename, "w") as f:
            for log in self.logs:
                f.write(json.dumps(log) + "\n")
        print(f"[✓] Exported {len(self.logs)} logs to {filename}")

# ---------------------- Run the Simulation ----------------------
if __name__ == "__main__":
    sim = Network(honest_count=50, sybil_count=10)
    for r in range(5):
        print(f"[+] Simulating Round {r+1}")
        sim.simulate_round(100)
        sim.inject_sybil_to_honest(messages_per_node=2)
        time.sleep(0.1)
    sim.export_logs()
