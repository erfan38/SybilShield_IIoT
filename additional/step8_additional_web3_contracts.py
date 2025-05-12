from web3 import Web3
import json

# Connect to Ethereum node
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
assert w3.is_connected()

# Load compiled ABI
with open("FederatedLearningABI.json") as f:
    fl_abi = json.load(f)
with open("ReputationABI.json") as f:
    rep_abi = json.load(f)

with open("deployed_contracts.json") as f:
    addresses = json.load(f)

fl_contract = w3.eth.contract(address=addresses["FederatedLearning"], abi=fl_abi)
rep_contract = w3.eth.contract(address=addresses["ReputationManagement"], abi=rep_abi)

account = w3.eth.accounts[0]
w3.eth.default_account = account

# Upload dummy weights
with open("global_model.pth", "rb") as f:
    weights = f.read()

fl_contract.functions.submitModelUpdate(weights).transact({"from": account, "gas": 2000000})

# Update reputation scores
with open("node_reputations.json") as f:
    reputations = json.load(f)

for node_id, data in reputations.items():
    score = max(0, 100 - int(data["reputation"] * 10))
    try:
        rep_contract.functions.updateReputation(account, score).transact({"from": account})
        print(f"[✓] Updated reputation for {node_id} to {score}")
    except Exception as e:
        print(f"[!] Failed to update {node_id}: {e}")