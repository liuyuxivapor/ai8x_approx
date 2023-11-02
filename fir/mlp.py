import torch
import torch.nn as nn

class fir_mlp(nn.Module):
    def __init__(self, len):
        super(fir_mlp, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(len, 4 * len),
            # nn.ReLU(),
            nn.Linear(4 * len, 4 * len),
            # nn.ReLU(),
            nn.Linear(4 * len, len)
        )
        
    def forward(self, x):
        return self.layers(x)
        