import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, grid_shape, feature_size, output_size, 
                 conv_channels=32, hidden_size=256):
        super().__init__()
        
        # CNN Branch (processes 2D grid)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, conv_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_channels * grid_shape[0] * grid_shape[1], hidden_size)
        )
        
        # Dense Branch (processes DT-20 features)
        self.dense = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Combined Head
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 64, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, grid, features):
        grid = grid.unsqueeze(1)  # Add channel dimension [batch_size, 1, 20, 10]
        grid_out = self.cnn(grid)
        feature_out = self.dense(features)
        combined = torch.cat([grid_out, feature_out], dim=1)
        return self.head(combined)