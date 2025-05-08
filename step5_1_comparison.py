import json
import json
from sklearn.metrics import classification_report

with open("node_reputations.json") as f:
    reputations = json.load(f)
y_true = []
y_pred = []
print("\n[Node Reputations]")
print("-" * 40)
for node, info in sorted(reputations.items()):
    rep = info["reputation"]
    acc = info["accuracy"]
    tag = "Sybil?" if "Sybil" in node or rep < 0 else "✅ Honest"
    print(f"{node:25}  Acc: {acc:.3f}  Rep: {rep:.2f}  {tag}")

for node_id, info in reputations.items():
    true_label = 1 if "Sybil" in node_id else 0
    predicted_label = 1 if info["reputation"] < 0 else 0  # threshold

    y_true.append(true_label)
    y_pred.append(predicted_label)

print("\n[Classification Report based on reputation threshold 0]")
print(classification_report(y_true, y_pred, target_names=["Honest", "Sybil"]))
