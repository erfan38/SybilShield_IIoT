import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),  # previous: 1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),  # kernel was 3, padding was 1
            nn.GELU(),
            nn.BatchNorm1d(64),
            #nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),  # kernel was 3, padding was 1
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)  # → (B, 128, 1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),            # (B, 128)
            nn.Linear(128, 64),      # Fully connected
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)         # Output for 3 classes: honest, high, low
        )
        self.softmax = nn.Softmax(dim=1) 
    def forward(self, x,  return_logits=False):
        # Input x: (batch_size, 13) → reshape for Conv1d
        x = x.unsqueeze(1)  # (batch_size, 1, 13)
        x = self.conv(x)
        #return self.fc(x)
        logits = self.fc(x)
        if return_logits:
            return logits  # Needed for training with CrossEntropyLoss
        return self.softmax(logits) 
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def get_model():
    model = ConvNet()
    print(f"Model with {model.count_parameters()} parameters created!")
    return model
