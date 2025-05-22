########## curves for validation and training
# import os
# import re
# import matplotlib.pyplot as plt

# # Configuration
# log_dir = "logs/DFL"
# clients = [f"client_{i}" for i in range(5)]
# epochs = 100
# pattern = r"Train Loss: ([\d\.]+), Train Acc: ([\d\.]+), Val Loss: ([\d\.]+), Val Acc: ([\d\.]+)"

# def extract_all_curves(filepath, num_entries=100):
#     with open(filepath, "r") as f:
#         lines = f.readlines()
#     recent_lines = lines[-num_entries:]
    
#     train_loss, train_acc = [], []
#     val_loss, val_acc = [], []
    
#     for line in recent_lines:
#         match = re.search(pattern, line)
#         if match:
#             train_loss.append(float(match.group(1)))
#             train_acc.append(float(match.group(2)))
#             val_loss.append(float(match.group(3)))
#             val_acc.append(float(match.group(4)))
#     return train_loss, train_acc, val_loss, val_acc

# # Generate one figure per client
# for client in clients:
#     path = os.path.join(log_dir, f"{client}_log.txt")
#     if not os.path.exists(path):
#         print(f"⚠️ Log file not found for {client}")
#         continue

#     train_loss, train_acc, val_loss, val_acc = extract_all_curves(path, epochs)

#     fig, ax1 = plt.subplots(figsize=(8, 4))
    
#     epochs_range = range(1, len(train_loss)+1)

#     # Accuracy on left y-axis
#     ax1.plot(epochs_range, train_acc, color='tab:blue', linestyle='-', label='Train Accuracy')
#     ax1.plot(epochs_range, val_acc, color='tab:cyan', linestyle='--', label='Val Accuracy')
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Accuracy", color='tab:blue')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')

#     # Loss on right y-axis
#     ax2 = ax1.twinx()
#     ax2.plot(epochs_range, train_loss, color='tab:red', linestyle='-', label='Train Loss')
#     ax2.plot(epochs_range, val_loss, color='tab:orange', linestyle='--', label='Val Loss')
#     ax2.set_ylabel("Loss", color='tab:red')
#     ax2.tick_params(axis='y', labelcolor='tab:red')

#     # Combine legends
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize='small')

#     fig.suptitle(f"{client.replace('_', ' ').capitalize()} - Accuracy & Loss (First Round)")
#     fig.tight_layout()
#     fig.subplots_adjust(top=0.88)

#     filename = os.path.join(log_dir, f"{client}_train_val_accuracy_loss_first_round.png")
#     plt.savefig(filename, dpi=300)
#     plt.close()

#     print(f"✅ Saved: {filename}")

########## curves for training

import os
import re
import matplotlib.pyplot as plt

# Configuration
log_dir = "logs/DFL"
clients = [f"client_{i}" for i in range(5)]
epochs = 100
pattern = r"Train Loss: ([\d\.]+), Train Acc: ([\d\.]+), Val Loss: ([\d\.]+), Val Acc: ([\d\.]+)"

def extract_train_curves(filepath, num_entries=100):
    with open(filepath, "r") as f:
        lines = f.readlines()
    recent_lines = lines[-num_entries:]
    
    train_loss, train_acc = [], []
    
    for line in recent_lines:
        match = re.search(pattern, line)
        if match:
            train_loss.append(float(match.group(1)))
            train_acc.append(float(match.group(2)))
    return train_loss, train_acc

# Generate one figure per client
for client in clients:
    path = os.path.join(log_dir, f"{client}_log.txt")
    if not os.path.exists(path):
        print(f"⚠️ Log file not found for {client}")
        continue

    train_loss, train_acc = extract_train_curves(path, epochs)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    epochs_range = range(1, len(train_loss)+1)

    # Accuracy on left y-axis
    ax1.plot(epochs_range, train_acc, color='tab:blue', linestyle='-', label='Train Accuracy')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Loss on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs_range, train_loss, color='tab:red', linestyle='-', label='Train Loss')
    ax2.set_ylabel("Loss", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize='small')

    fig.suptitle(f"{client.replace('_', ' ').capitalize()} - Training Accuracy & Loss (First Round)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    filename = os.path.join(log_dir, f"{client}_train_accuracy_loss_first_round.png")
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"✅ Saved: {filename}")
