# step3_contract_simulation.py
# Simulate smart contract logic in-memory

import json

class FederatedContractSimulator:
    def __init__(self):
        self.model_updates = {}

    def submit_model_update(self, node_id, accuracy):
        self.model_updates[node_id] = accuracy
        print(f"[+] {node_id} submitted update with accuracy {accuracy:.3f}")

    def aggregate(self):
        if not self.model_updates:
            return None
        avg = sum(self.model_updates.values()) / len(self.model_updates)
        print(f"[*] Aggregated model accuracy: {avg:.3f}")
        return avg

class ReputationContractSimulator:
    def __init__(self, reputations):
        self.reputation_scores = reputations

    def flag_suspicious_nodes(self, threshold=-3.0):
        flagged = [nid for nid, r in self.reputation_scores.items() if r['reputation'] <= threshold]
        print(f"[!] Flagged nodes for banning: {flagged}")
        return flagged

# Load reputation scores
with open("node_reputations.json") as f:
    reputations = json.load(f)

# Simulate reputation smart contract
rep_contract = ReputationContractSimulator(reputations)
flagged_nodes = rep_contract.flag_suspicious_nodes(threshold=-3.5)

# Simulate federated learning contract
fl_contract = FederatedContractSimulator()
for node_id in reputations:
    acc = max(0.5, 1.0 + reputations[node_id]['reputation'] / 10.0)  # Fake accuracy
    fl_contract.submit_model_update(node_id, acc)
fl_contract.aggregate()
