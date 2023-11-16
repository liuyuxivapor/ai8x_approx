import torch
import torch.nn as nn

class mlp(nn.Module):
    def __init__(self, len):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1 * len, 4 * len),
            nn.Linear(4 * len, 4 * len),
            nn.Linear(4 * len, 2 * len)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
