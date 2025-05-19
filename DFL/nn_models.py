# import torch.nn as nn



# class SimpleNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             # nn.Linear(9, 128), nn.ReLU(),
#             # nn.Dropout(0.3),
#             # nn.Linear(128, 64), nn.ReLU(),
#             # nn.Dropout(0.2),
#             # nn.Linear(64, 32), nn.ReLU(),
#             # nn.Linear(32, 2)
#             nn.Linear(9, 256), nn.ReLU(),
#             nn.Linear(256, 128), nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, 64), nn.ReLU(),
#             nn.Linear(64, 2)

#         )

#     def forward(self, x):
#         return self.net(x)
    
# import torch
# import torch.nn as nn

# import torch
# import torch.nn as nn

# class ConvNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
#             nn.BatchNorm1d(16),

#             nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
#             nn.BatchNorm1d(32),
#             nn.Dropout(0.1),

#             nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
#             nn.BatchNorm1d(64),
#             nn.AdaptiveAvgPool1d(1)
#         )

#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64, 32), nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(32, 2)
#         )

#     def forward(self, x):
#         #x = x.unsqueeze(1)  # Shape: [batch_size, 1, 9]
#         x = self.conv(x)
#         return self.fc(x)

#     def count_parameters(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)



# def get_model():
#     model = ConvNN()
#     print(f"Model with {model.count_parameters()} parameters created!")
#     return ConvNN()

import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # (B, 1, 13) → (B, 32, 13)
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # (B, 32, 13) → (B, 64, 13)
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # (B, 64, 13) → (B, 128, 13)
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)  # → (B, 128, 1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),            # (B, 128)
            nn.Linear(128, 64),      # Fully connected
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)         # Output for 3 classes: honest, high, low
        )

    def forward(self, x):
        # Input x: (batch_size, 13) → reshape for Conv1d
        x = x.unsqueeze(1)  # (batch_size, 1, 13)
        x = self.conv(x)
        return self.fc(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def get_model():
    model = ConvNet()
    print(f"Model with {model.count_parameters()} parameters created!")
    return model
