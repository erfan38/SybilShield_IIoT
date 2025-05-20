import json
import os
from web3 import Web3
from eth_account import Account
from eth_account.messages import encode_defunct
from dotenv import load_dotenv


load_dotenv()

# === Load Environment Variables ===
RPC_URL = os.getenv("RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")

# === Connect to Ethereum ===
w3 = Web3(Web3.HTTPProvider(RPC_URL))
account = Account.from_key(PRIVATE_KEY)
w3.eth.default_account = account.address

# === Load ABI ===
with open("artifacts/contracts/ReputationManager.sol/ReputationManager.json") as f:
    abi = json.load(f)["abi"]
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)

# === Severity Mapping ===
SEVERITY_MAP = {
    "low": 3,
    "high": 8,
    # "honest": 0  # optionally skip
}

# === Sign a report (ECDSA) ===
def sign_report(suspect_node, severity):
    message_hash = w3.solidityKeccak(["address", "uint8"], [suspect_node, severity])
    message = encode_defunct(message_hash)
    signed = Account.sign_message(message, PRIVATE_KEY)
    return signed.signature

# === Submit Reports ===
REPORT_DIR = "reports"

for file in os.listdir(REPORT_DIR):
    if file.endswith("_report.json"):
        with open(os.path.join(REPORT_DIR, file)) as f:
            data = json.load(f)

        node = account.address
        model_hash = data["model_hash"]
        reports = data.get("report", [])  # from raw input

        print(f"\n📤 Submitting model hash for {node}")
        tx1 = contract.functions.recordModelHash(node, "0x" + model_hash).build_transaction({
            "from": node,
            "nonce": w3.eth.get_transaction_count(node),
            "gas": 300000,
            "gasPrice": w3.to_wei("20", "gwei")
        })
        signed_tx1 = w3.eth.account.sign_transaction(tx1, PRIVATE_KEY)
        tx_hash1 = w3.eth.send_raw_transaction(signed_tx1.rawTransaction)
        w3.eth.wait_for_transaction_receipt(tx_hash1)
        print(" Model hash submitted.")

        for entry in reports:
            label = entry["severity"].lower()
            if label not in SEVERITY_MAP:
                continue  # skip 'honest' or unknown labels
            severity = SEVERITY_MAP[label]
            target = entry["node_id"]  # ensure this is a valid Ethereum address
            signature = sign_report(target, severity)

            print(f"📡 Reporting {target} with severity {severity}")
            tx2 = contract.functions.reportSybil(target, severity, signature).build_transaction({
                "from": node,
                "nonce": w3.eth.get_transaction_count(node),
                "gas": 300000,
                "gasPrice": w3.to_wei("20", "gwei")
            })
            signed_tx2 = w3.eth.account.sign_transaction(tx2, PRIVATE_KEY)
            tx_hash2 = w3.eth.send_raw_transaction(signed_tx2.rawTransaction)
            w3.eth.wait_for_transaction_receipt(tx_hash2)
            print(f"Reported {target}.")

print("\n All reports submitted successfully.")
